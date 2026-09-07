"""Google Gemini adapter for the LLMClient protocol (spec Phase 2a, Task 18)."""

import json
from typing import Any

from pydantic import BaseModel

from counterstrat.llm.base import (
    ChatTurn,
    LLMResult,
    ToolCall,
    ToolSpec,
    Transport,
    call_with_retries,
)

PROVIDER = "gemini"

# Gemini thinking models count reasoning tokens against max_output_tokens, so a
# hard task can think the whole budget away and emit a few hundred visible
# characters (observed: 3929 thought tokens + 163 text tokens under a 4096 cap;
# a 71-round corpus pushed thinking past 12k tokens). Callers size max_tokens
# for VISIBLE text; the wire ceiling adds this headroom for the thoughts.
THINKING_HEADROOM = 24576


def _response_payload(text: str | None) -> dict[str, Any]:
    """Coerces a tool-result string into the dict shape a function_response needs."""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"result": text}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _synth_call_id(name: str, index: int) -> str:
    """The id we mint when a Gemini function_call has no `id` of its own."""
    return f"{name}:{index}"


def _calls_by_id(turns: list[ChatTurn]) -> dict[str, tuple[str, str | None]]:
    """tool_call_id -> (function name, provider call id or None) from the calling turns.

    A tool turn carries only `tool_call_id`, so the function name a
    function_response must name comes from the assistant turn that requested the
    call. Ids matching `_synth_call_id` are ours, not the provider's, and are not
    echoed back; a populated `FunctionCall.id` is, since Gemini matches responses
    to calls by id whenever the model sends one.
    """
    lookup: dict[str, tuple[str, str | None]] = {}
    for turn in turns:
        for index, call in enumerate(turn.tool_calls):
            provider_id = None if call.id == _synth_call_id(call.name, index) else call.id
            lookup[call.id] = (call.name, provider_id)
    return lookup


def _contents(turns: list[ChatTurn]) -> list[dict[str, Any]]:
    """Normalized turns -> Gemini contents (function responses ride on user turns)."""
    lookup = _calls_by_id(turns)
    contents: list[dict[str, Any]] = []
    prev_tool = False
    for turn in turns:
        if turn.role == "tool":
            call_id = turn.tool_call_id or ""
            # No calling turn in `turns` (a bare tool turn): fall back to our own id form.
            name, provider_id = lookup.get(call_id, (call_id.rsplit(":", 1)[0], None))
            response: dict[str, Any] = {"name": name}
            if provider_id:
                response["id"] = provider_id
            response["response"] = _response_payload(turn.text)
            part = {"function_response": response}
            if prev_tool:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            prev_tool = True
            continue
        prev_tool = False
        if turn.role == "user":
            contents.append({"role": "user", "parts": [{"text": turn.text or ""}]})
            continue
        parts: list[dict[str, Any]] = []
        if turn.text:
            parts.append({"text": turn.text})
        for index, call in enumerate(turn.tool_calls):
            function_call: dict[str, Any] = {"name": call.name, "args": call.arguments}
            if call.id != _synth_call_id(call.name, index):
                function_call["id"] = call.id
            part: dict[str, Any] = {"function_call": function_call}
            # Gemini 3 validates that each function_call part returns with the
            # thought_signature it arrived with; base64 round-trips as a plain string.
            if call.signature:
                part["thought_signature"] = call.signature
            parts.append(part)
        contents.append({"role": "model", "parts": parts})
    return contents


def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters_json_schema": t.input_schema,
                }
                for t in tools
            ]
        }
    ]


def _parts(resp: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = resp.get("candidates") or []
    if not candidates:
        return []
    return (candidates[0].get("content") or {}).get("parts") or []


# Bounded waiting, mirroring anthropic_client: a dead route to the API must
# surface as an error card within minutes, never an endless spinner. The unit
# is MILLISECONDS (google-genai HttpOptions.timeout) and it caps the WHOLE
# non-streaming request, so it stays generous enough for big First Reads.
REQUEST_TIMEOUT_MS = 300_000


class GeminiClient:
    """LLMClient over `google-genai`; `transport` swaps the SDK out for replay in tests.

    Gemini caches long prompt prefixes implicitly for 2.5+ models, so the cached
    Map Card needs no explicit breakpoint -- it just has to lead the system
    instruction (mirrors the Anthropic adapter's cached system block).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        transport: Transport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._sdk_transport
        self._sdk: Any = None

    def _ensure_sdk(self) -> Any:
        from google import genai
        from google.genai import types

        if self._sdk is None:
            self._sdk = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        return self._sdk

    def _sdk_transport(self, req: dict[str, Any]) -> dict[str, Any]:
        self._ensure_sdk()
        resp = self._sdk.models.generate_content(**req)
        return resp.model_dump(mode="json", exclude_none=True)

    def _send(self, req: dict[str, Any]) -> dict[str, Any]:
        return call_with_retries(lambda: self._transport(req))

    def _config(
        self,
        system: str,
        max_tokens: int | None,
        *,
        tools: list[ToolSpec] | None = None,
        schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {"system_instruction": system}
        if max_tokens is not None:
            # An explicit cap bounds VISIBLE text; the headroom absorbs thoughts.
            # None sends no ceiling at all: the model thinks and writes up to its
            # own maximum, which is what analysis-grade calls want.
            config["max_output_tokens"] = max_tokens + THINKING_HEADROOM
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema
        if tools:
            config["tools"] = _tools(tools)
        return config

    def _result(self, resp: dict[str, Any]) -> LLMResult:
        text = "\n".join(
            p.get("text") or "" for p in _parts(resp) if p.get("text") and not p.get("thought")
        )
        usage = resp.get("usage_metadata") or {}
        candidates = resp.get("candidates") or [{}]
        finish = str(candidates[0].get("finish_reason") or "").upper()
        return LLMResult(
            text=text,
            input_tokens=usage.get("prompt_token_count") or 0,
            output_tokens=usage.get("candidates_token_count") or 0,
            cache_read_tokens=usage.get("cached_content_token_count") or 0,
            model=resp.get("model_version") or self.model,
            provider=PROVIDER,
            truncated=finish == "MAX_TOKENS",
        )

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        resp = self._send(
            {
                "model": self.model,
                "contents": user,
                "config": self._config(system, max_tokens),
            }
        )
        return self._result(resp)

    def complete_json[T: BaseModel](
        self, *, system: str, user: str, schema: type[T], max_tokens: int | None = None
    ) -> tuple[T, LLMResult]:
        resp = self._send(
            {
                "model": self.model,
                "contents": user,
                "config": self._config(system, max_tokens, schema=schema),
            }
        )
        result = self._result(resp)
        return schema.model_validate_json(result.text), result

    def chat(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        tools: list[ToolSpec],
        max_tokens: int | None = None,
    ) -> tuple[ChatTurn, LLMResult]:
        resp = self._send(
            {
                "model": self.model,
                "contents": _contents(turns),
                "config": self._config(system, max_tokens, tools=tools),
            }
        )
        result = self._result(resp)
        calls: list[ToolCall] = []
        for part in _parts(resp):
            fc = part.get("function_call")
            if not fc:
                continue
            name = fc.get("name") or ""
            calls.append(
                ToolCall(
                    id=fc.get("id") or _synth_call_id(name, len(calls)),
                    name=name,
                    arguments=fc.get("args") or {},
                    signature=part.get("thought_signature"),
                )
            )
        return ChatTurn(role="assistant", text=result.text or None, tool_calls=calls), result
