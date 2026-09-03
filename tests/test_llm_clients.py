"""LLM adapter tests: config-driven provider switch, replayed JSON + tool-use, budget guard."""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from counterstrat.config import DEV_GEMINI_MODEL, AppConfig, resolve_latest_sonnet
from counterstrat.llm import (
    AnthropicClient,
    ChatTurn,
    GeminiClient,
    LLMBudgetError,
    ToolSpec,
    make_client,
)
from counterstrat.llm.base import call_with_retries

FIXTURES = Path(__file__).parent / "fixtures" / "llm"

TOOL = ToolSpec(
    name="get_tendency",
    description="Look up a team tendency",
    input_schema={
        "type": "object",
        "properties": {"team": {"type": "string"}, "metric": {"type": "string"}},
        "required": ["team"],
    },
)


class SiteCall(BaseModel):
    site: str
    confidence: float


class ReplayTransport:
    """Replays recorded provider responses in order and records the requests sent."""

    def __init__(self, *names: str):
        self.responses = [
            json.loads((FIXTURES / f"{n}.json").read_text(encoding="utf-8")) for n in names
        ]
        self.requests: list[dict[str, Any]] = []

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(req)
        if not self.responses:
            raise AssertionError("replay transport exhausted")
        return self.responses.pop(0)


class Boom(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def never_called(_req: dict[str, Any]) -> dict[str, Any]:
    raise AssertionError("transport must not be called")


@pytest.fixture
def no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "LLM_PROVIDER",
        "ANTHROPIC_MODEL",
        "GEMINI_MODEL",
        "MAX_INPUT_TOKENS",
        "DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_make_client_provider_switch(monkeypatch, no_env_keys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = AppConfig.load(tmp_path / "settings.json")
    assert cfg.provider == "anthropic"
    assert make_client(cfg).__class__.__name__ == "AnthropicClient"

    cfg2 = cfg.model_copy(update={"provider": "gemini", "gemini_api_key": "k"})
    assert make_client(cfg2).__class__.__name__ == "GeminiClient"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_GENERATIVE_AI_API_KEY", "envkey")
    cfg3 = AppConfig.load(tmp_path / "settings.json")
    assert cfg3.provider == "gemini"
    assert cfg3.gemini_api_key == "envkey"
    client = make_client(cfg3)
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.5-pro"


def test_settings_json_beats_env(monkeypatch, no_env_keys, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("GEMINI_MODEL", "from-env")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "provider": "gemini",
                "gemini_model": "from-settings",
                "gemini_api_key": "sk",
                "max_input_tokens": 12345,
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig.load(settings)
    assert cfg.provider == "gemini"
    assert cfg.gemini_model == "from-settings"
    assert cfg.max_input_tokens == 12345
    client = make_client(cfg)
    assert isinstance(client, GeminiClient) and client.max_input_tokens == 12345


def test_missing_key_raises():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make_client(AppConfig(provider="gemini", gemini_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_client(AppConfig(provider="anthropic", anthropic_api_key=None))
    unknown = AppConfig(anthropic_api_key="k").model_copy(update={"provider": "grok"})
    with pytest.raises(ValueError, match="Unknown provider: grok"):
        make_client(unknown)


def test_anthropic_complete_replay():
    from counterstrat.llm.anthropic_client import THINKING_HEADROOM

    transport = ReplayTransport("anthropic_complete")
    client = AnthropicClient(api_key="k", model="m", transport=transport)
    res = client.complete(system="card", user="q", max_tokens=256)
    assert res.text.startswith("They default")
    assert (res.input_tokens, res.output_tokens, res.cache_read_tokens) == (1520, 42, 1408)
    assert res.provider == "anthropic" and res.model == "claude-sonnet-5"
    # Map Card system block carries the explicit cache breakpoint.
    system = transport.requests[0]["system"]
    assert system == [{"type": "text", "text": "card", "cache_control": {"type": "ephemeral"}}]
    # Thinking models spend reasoning tokens from the same wire ceiling: the
    # caller's cap bounds VISIBLE text, the headroom absorbs the thoughts
    # (a 64-token cap returned empty truncated text on claude-sonnet-5).
    assert transport.requests[0]["max_tokens"] == 256 + THINKING_HEADROOM
    # A small effective cap stays on the fast non-streaming path.
    assert not transport.requests[0].get("stream")


def test_anthropic_complete_json_replay():
    transport = ReplayTransport("anthropic_complete_json")
    client = AnthropicClient(api_key="k", model="m", transport=transport)
    out, usage = client.complete_json(system="s", user="u", schema=SiteCall)
    assert isinstance(out, SiteCall)
    assert out.site == "A" and out.confidence == 0.82
    assert usage.provider == "anthropic" and usage.cache_read_tokens == 1408
    req = transport.requests[0]
    assert req["kind"] == "parse" and req["output_format"] is SiteCall


def test_gemini_complete_json_replay():
    transport = ReplayTransport("gemini_complete_json")
    client = GeminiClient(api_key="k", model="m", transport=transport)
    out, usage = client.complete_json(system="s", user="u", schema=SiteCall)
    assert isinstance(out, SiteCall)
    assert out.site == "A" and out.confidence == 0.82
    assert usage.provider == "gemini" and usage.model == "gemini-2.5-pro"
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens) == (1533, 39, 1408)
    config = transport.requests[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] is SiteCall
    assert config["system_instruction"] == "s"


def test_gemini_complete_replay():
    from counterstrat.llm.gemini_client import THINKING_HEADROOM

    transport = ReplayTransport("gemini_complete")
    client = GeminiClient(api_key="k", model="m", transport=transport)
    res = client.complete(system="card", user="q", max_tokens=128)
    assert res.text.startswith("They default")
    assert res.truncated is False
    # Thinking models spend reasoning tokens from the same ceiling: the caller's
    # max_tokens buys visible text, the headroom absorbs the thoughts.
    assert transport.requests[0]["config"]["max_output_tokens"] == 128 + THINKING_HEADROOM
    assert "response_schema" not in transport.requests[0]["config"]


def test_default_max_tokens_is_uncapped():
    """No caller cap = no ceiling on Gemini, model-max-bounded default on Anthropic."""
    from counterstrat.llm.anthropic_client import DEFAULT_MAX_OUTPUT

    g_transport = ReplayTransport("gemini_complete")
    gemini = GeminiClient(api_key="k", model="m", transport=g_transport)
    gemini.complete(system="s", user="u")
    assert "max_output_tokens" not in g_transport.requests[0]["config"]

    a_transport = ReplayTransport("anthropic_complete")
    anthropic = AnthropicClient(api_key="k", model="m", transport=a_transport)
    anthropic.complete(system="s", user="u")
    assert a_transport.requests[0]["max_tokens"] == DEFAULT_MAX_OUTPUT
    # 32k output over the SDK's ~10-minute non-streaming guard: must stream
    # (observed: every uncapped claude-sonnet-5 call raised
    # 'Streaming is required for operations that may take longer than 10 minutes').
    assert a_transport.requests[0]["stream"] is True


def test_anthropic_streaming_threshold():
    """Effective caps at/over STREAM_THRESHOLD stream; smaller ones do not."""
    from counterstrat.llm.anthropic_client import STREAM_THRESHOLD, THINKING_HEADROOM

    transport = ReplayTransport("anthropic_complete", "anthropic_complete")
    client = AnthropicClient(api_key="k", model="m", transport=transport)
    client.complete(system="s", user="u", max_tokens=STREAM_THRESHOLD - THINKING_HEADROOM)
    assert transport.requests[0]["stream"] is True
    client.complete(system="s", user="u", max_tokens=64)
    assert not transport.requests[1].get("stream")


def test_anthropic_complete_json_streams_when_uncapped():
    transport = ReplayTransport("anthropic_complete_json")
    client = AnthropicClient(api_key="k", model="m", transport=transport)
    out, _ = client.complete_json(system="s", user="u", schema=SiteCall)
    assert out.site == "A"
    req = transport.requests[0]
    assert req["kind"] == "parse" and req["stream"] is True


def test_gemini_truncation_is_flagged():
    """finish_reason MAX_TOKENS surfaces as LLMResult.truncated."""

    def transport(req):
        return {
            "candidates": [
                {
                    "finish_reason": "MAX_TOKENS",
                    "content": {"parts": [{"text": "cut off mid-sent"}]},
                }
            ],
            "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 4096},
            "model_version": "m",
        }

    client = GeminiClient(api_key="k", model="m", transport=transport)
    res = client.complete(system="s", user="u")
    assert res.truncated is True
    assert res.text == "cut off mid-sent"


def test_anthropic_truncation_is_flagged():
    def transport(req):
        return {
            "content": [{"type": "text", "text": "cut"}],
            "usage": {"input_tokens": 5, "output_tokens": 4096},
            "model": "m",
            "stop_reason": "max_tokens",
        }

    client = AnthropicClient(api_key="k", model="m", transport=transport)
    res = client.complete(system="s", user="u")
    assert res.truncated is True


def test_anthropic_chat_tool_roundtrip():
    transport = ReplayTransport("anthropic_chat_tool_use", "anthropic_chat_final")
    client = AnthropicClient(api_key="k", model="m", transport=transport)
    turn, _ = client.chat(system="s", turns=[ChatTurn(role="user", text="q")], tools=[TOOL])
    assert turn.role == "assistant"
    assert turn.tool_calls and turn.tool_calls[0].name == "get_tendency"
    assert turn.tool_calls[0].arguments == {"team": "NAVI", "metric": "site_split"}
    assert transport.requests[0]["tools"] == [
        {
            "name": "get_tendency",
            "description": "Look up a team tendency",
            "input_schema": TOOL.input_schema,
        }
    ]

    turns = [
        ChatTurn(role="user", text="q"),
        turn,
        ChatTurn(role="tool", tool_call_id=turn.tool_calls[0].id, text='{"rows": []}'),
    ]
    final, usage = client.chat(system="s", turns=turns, tools=[TOOL])
    assert final.text and not final.tool_calls
    assert usage.provider == "anthropic"

    messages = transport.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["type"] for b in messages[1]["content"]] == ["text", "tool_use"]
    assert messages[1]["content"][1]["id"] == "toolu_01Fixture"
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01Fixture", "content": '{"rows": []}'}
    ]


def test_gemini_chat_tool_roundtrip():
    transport = ReplayTransport("gemini_chat_tool_use", "gemini_chat_final")
    client = GeminiClient(api_key="k", model="m", transport=transport)
    turn, _ = client.chat(system="s", turns=[ChatTurn(role="user", text="q")], tools=[TOOL])
    assert turn.tool_calls and turn.tool_calls[0].name == "get_tendency"
    assert turn.tool_calls[0].arguments == {"team": "NAVI", "metric": "site_split"}
    # This fixture's function_call carries no id, so we synthesize name:index.
    assert turn.tool_calls[0].id == "get_tendency:0"
    assert transport.requests[0]["config"]["tools"] == [
        {
            "function_declarations": [
                {
                    "name": "get_tendency",
                    "description": "Look up a team tendency",
                    "parameters_json_schema": TOOL.input_schema,
                }
            ]
        }
    ]

    turns = [
        ChatTurn(role="user", text="q"),
        turn,
        ChatTurn(role="tool", tool_call_id=turn.tool_calls[0].id, text='{"rows": []}'),
    ]
    final, usage = client.chat(system="s", turns=turns, tools=[TOOL])
    assert final.text and not final.tool_calls
    assert usage.provider == "gemini"

    contents = transport.requests[1]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][1] == {
        "function_call": {"name": "get_tendency", "args": {"team": "NAVI", "metric": "site_split"}}
    }
    assert contents[2]["parts"] == [
        {"function_response": {"name": "get_tendency", "response": {"rows": []}}}
    ]


def test_gemini_chat_tool_roundtrip_with_provider_call_id():
    """A populated FunctionCall.id must not displace the function name on the way back."""
    transport = ReplayTransport("gemini_chat_tool_use_id", "gemini_chat_final")
    client = GeminiClient(api_key="k", model="m", transport=transport)
    turn, _ = client.chat(system="s", turns=[ChatTurn(role="user", text="q")], tools=[TOOL])
    assert turn.tool_calls and turn.tool_calls[0].name == "get_tendency"
    assert turn.tool_calls[0].id == "fc_9xyz"

    turns = [
        ChatTurn(role="user", text="q"),
        turn,
        ChatTurn(role="tool", tool_call_id=turn.tool_calls[0].id, text='{"rows": []}'),
    ]
    client.chat(system="s", turns=turns, tools=[TOOL])

    contents = transport.requests[1]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"][1] == {
        "function_call": {"name": "get_tendency", "args": {"team": "NAVI"}, "id": "fc_9xyz"}
    }
    assert contents[2]["parts"] == [
        {"function_response": {"name": "get_tendency", "id": "fc_9xyz", "response": {"rows": []}}}
    ]


def test_gemini_chat_preserves_thought_signature():
    """Gemini 3 rejects a tool loop whose function_call part lost its thought_signature."""
    signature = "AAFhYmP_"  # base64, exactly as model_dump(mode="json") renders the bytes
    tool_use = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {"name": "get_tendency", "args": {"team": "NAVI"}},
                            "thought_signature": signature,
                        }
                    ],
                }
            }
        ],
        "usage_metadata": {"prompt_token_count": 12, "candidates_token_count": 4},
    }
    responses: list[dict[str, Any]] = [
        tool_use,
        json.loads((FIXTURES / "gemini_chat_final.json").read_text(encoding="utf-8")),
    ]
    requests: list[dict[str, Any]] = []

    def transport(req: dict[str, Any]) -> dict[str, Any]:
        requests.append(req)
        return responses.pop(0)

    client = GeminiClient(api_key="k", model="m", transport=transport)
    turn, _ = client.chat(system="s", turns=[ChatTurn(role="user", text="q")], tools=[TOOL])
    assert turn.tool_calls[0].signature == signature

    client.chat(
        system="s",
        turns=[
            ChatTurn(role="user", text="q"),
            turn,
            ChatTurn(role="tool", tool_call_id=turn.tool_calls[0].id, text='{"rows": []}'),
        ],
        tools=[TOOL],
    )
    model_part = requests[1]["contents"][1]["parts"][0]
    assert model_part["function_call"]["name"] == "get_tendency"
    assert model_part["thought_signature"] == signature


def test_gemini_non_json_tool_result_is_wrapped():
    transport = ReplayTransport("gemini_chat_final")
    client = GeminiClient(api_key="k", model="m", transport=transport)
    client.chat(
        system="s",
        turns=[ChatTurn(role="tool", tool_call_id="get_tendency:0", text="not json")],
        tools=[],
    )
    part = transport.requests[0]["contents"][0]["parts"][0]
    assert part["function_response"]["response"] == {"result": "not json"}


def test_budget_guard():
    anthropic_client = AnthropicClient(
        api_key="k", model="m", max_input_tokens=1000, transport=never_called
    )
    with pytest.raises(LLMBudgetError, match="max_input_tokens=1000"):
        anthropic_client.complete(system="x" * 3_000_000, user="u")
    with pytest.raises(LLMBudgetError):
        anthropic_client.complete_json(system="x" * 3_000_000, user="u", schema=SiteCall)
    with pytest.raises(LLMBudgetError):
        anthropic_client.chat(
            system="s", turns=[ChatTurn(role="user", text="x" * 3_000_000)], tools=[]
        )

    gemini_client = GeminiClient(
        api_key="k", model="m", max_input_tokens=1000, transport=never_called
    )
    with pytest.raises(LLMBudgetError):
        gemini_client.complete(system="x" * 3_000_000, user="u")

    # Just under the ceiling goes through (3.5 chars per token).
    ok = ReplayTransport("anthropic_complete")
    assert AnthropicClient(api_key="k", max_input_tokens=1000, transport=ok).complete(
        system="x" * 3_000, user="u"
    )


def test_retry_backoff_on_429_then_5xx():
    delays: list[float] = []
    statuses = [429, 503]

    def flaky() -> str:
        if statuses:
            raise Boom(statuses.pop(0))
        return "ok"

    assert call_with_retries(flaky, tries=3, sleep=delays.append) == "ok"
    assert delays == [0.5, 1.0]


def test_retry_gives_up_and_does_not_retry_4xx():
    calls: list[int] = []

    def always_429() -> str:
        calls.append(1)
        raise Boom(429)

    with pytest.raises(Boom):
        call_with_retries(always_429, tries=3, sleep=lambda _: None)
    assert len(calls) == 3

    def bad_request() -> str:
        calls.append(1)
        raise Boom(400)

    with pytest.raises(Boom):
        call_with_retries(bad_request, tries=3, sleep=lambda _: None)
    assert len(calls) == 4


def test_adapter_retries_transport(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    replay = ReplayTransport("anthropic_complete")
    attempts: list[int] = []

    def flaky(req: dict[str, Any]) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            raise Boom(529)
        return replay(req)

    res = AnthropicClient(api_key="k", model="m", transport=flaky).complete(system="s", user="u")
    assert len(attempts) == 2 and res.output_tokens == 42


@pytest.mark.live
def test_anthropic_live_smoke():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    client = AnthropicClient(api_key=key, model=resolve_latest_sonnet(key))
    out, usage = client.complete_json(
        system="Answer with the bombsite letter only.",
        user='Which site is "A site" on de_anubis? Return {"site": "...", "confidence": 0.0-1.0}.',
        schema=SiteCall,
        max_tokens=256,
    )
    assert out.site
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    print(f"anthropic live usage: {usage.model_dump()}")
    # A tiny visible cap must still return text on a thinking model (headroom),
    # and an uncapped call must not raise the SDK's streaming ValueError.
    small = client.complete(system="Reply with only the letter B.", user="?", max_tokens=8)
    assert small.text.strip()
    big = client.complete(system="Reply with only the letter B.", user="?")
    assert big.text.strip()


@pytest.mark.live
def test_gemini_live_smoke():
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    )
    if not key:
        pytest.skip("no Gemini API key set")
    client = GeminiClient(api_key=key, model=DEV_GEMINI_MODEL)
    out, usage = client.complete_json(
        system="Answer with the bombsite letter only.",
        user='Which site is "A site" on de_anubis? Return {"site": "...", "confidence": 0.0-1.0}.',
        schema=SiteCall,
        max_tokens=256,
    )
    assert out.site
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    print(f"gemini live usage: {usage.model_dump()}")


@pytest.mark.live
def test_gemini_live_tool_call():
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    )
    if not key:
        pytest.skip("no Gemini API key set")
    client = GeminiClient(api_key=key, model=DEV_GEMINI_MODEL)
    turn, _ = client.chat(
        system="Use get_tendency before answering.",
        turns=[ChatTurn(role="user", text="What does NAVI do on the T side of Anubis?")],
        tools=[TOOL],
        max_tokens=512,
    )
    assert turn.tool_calls or turn.text
