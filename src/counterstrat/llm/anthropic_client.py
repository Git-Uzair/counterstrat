"""Anthropic adapter for the LLMClient protocol (spec Phase 2a, Task 18)."""

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

PROVIDER = "anthropic"

# The Messages API REQUIRES max_tokens, so "no self-imposed cap" maps to a
# generous default well under the model maximum (claude-sonnet-5 supports 128k
# output tokens, platform.claude.com sonnet-5 overview, checked 2026-09-03).
DEFAULT_MAX_OUTPUT = 32_000

# Thinking models spend reasoning tokens from the same wire ceiling: a caller's
# small max_tokens (e.g. 64 for an MCQ answer) came back as EMPTY truncated
# text on claude-sonnet-5 (2026-09-04 representation lab). Callers size
# max_tokens for VISIBLE text; the wire ceiling adds this headroom for the
# thoughts, mirroring gemini_client.THINKING_HEADROOM.
THINKING_HEADROOM = 8_192

# The SDK refuses non-streaming requests whose max_tokens implies a >10 minute
# worst case ("Streaming is required for operations that may take longer than
# 10 minutes", anthropic 1.3.0, observed live 2026-09-04 at 32k on
# claude-sonnet-5). At or over this effective cap the transport streams and
# assembles the final message.
STREAM_THRESHOLD = 16_000

# Bounded waiting: the SDK default is a 600s timeout with 2 internal retries,
# and call_with_retries wraps more attempts around that - an unreachable
# api.anthropic.com (blocked route, AV proxy) spun a First Read for tens of
# minutes with zero feedback (field bug, 2026-09-07). Connect fails fast; the
# read window only trips when the stream goes DEAD (httpx read timeout is
# per-chunk, so healthy long generations keep flowing).
CONNECT_TIMEOUT_S = 15.0
REQUEST_TIMEOUT_S = 180.0
SDK_MAX_RETRIES = 1


def _wire_cap(max_tokens: int | None) -> int:
    """Visible-text cap -> wire ceiling (headroom for thinking tokens)."""
    if max_tokens is None:
        return DEFAULT_MAX_OUTPUT
    return max_tokens + THINKING_HEADROOM


def _system_blocks(system: str) -> list[dict[str, Any]]:
    """System prompt as one cached text block.

    The Map Card system prompt is far over Sonnet's 1,024-token cache minimum, so
    it always gets an explicit ephemeral breakpoint (5-min TTL).
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _messages(turns: list[ChatTurn]) -> list[dict[str, Any]]:
    """Normalized turns -> Anthropic messages (tool results ride on user turns)."""
    msgs: list[dict[str, Any]] = []
    prev_tool = False
    for turn in turns:
        if turn.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": turn.tool_call_id,
                "content": turn.text or "",
            }
            if prev_tool:
                msgs[-1]["content"].append(block)
            else:
                msgs.append({"role": "user", "content": [block]})
            prev_tool = True
            continue
        prev_tool = False
        if turn.role == "user":
            msgs.append({"role": "user", "content": [{"type": "text", "text": turn.text or ""}]})
            continue
        content: list[dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        for call in turn.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        msgs.append({"role": "assistant", "content": content})
    return msgs


def _tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


class AnthropicClient:
    """LLMClient over `anthropic`; `transport` swaps the SDK out for replay in tests."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        transport: Transport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self._transport = transport or self._sdk_transport
        self._sdk: Any = None

    def _ensure_sdk(self) -> Any:
        import anthropic
        import httpx

        if self._sdk is None:
            self._sdk = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
                max_retries=SDK_MAX_RETRIES,
            )
        return self._sdk

    def _sdk_transport(self, req: dict[str, Any]) -> dict[str, Any]:
        self._ensure_sdk()
        req = dict(req)
        kind = req.pop("kind")
        stream = req.pop("stream", False)
        messages = self._sdk.messages
        if stream:
            # messages.stream accepts output_format too (anthropic 1.3.0), so
            # one path serves both create- and parse-shaped requests.
            with messages.stream(**req) as s:
                final = s.get_final_message()
            return final.model_dump(mode="json", warnings=False)
        if kind == "parse":
            # ParsedMessage carries schema-typed blocks the Message union does not
            # know; silence the serializer warnings, the JSON text is what we read.
            return messages.parse(**req).model_dump(mode="json", warnings=False)
        return messages.create(**req).model_dump(mode="json")

    def _send(self, req: dict[str, Any]) -> dict[str, Any]:
        return call_with_retries(lambda: self._transport(req))

    def _result(self, resp: dict[str, Any]) -> LLMResult:
        blocks = resp.get("content") or []
        text = "\n".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        usage = resp.get("usage") or {}
        return LLMResult(
            text=text,
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            cache_read_tokens=usage.get("cache_read_input_tokens") or 0,
            model=resp.get("model") or self.model,
            provider=PROVIDER,
            truncated=(resp.get("stop_reason") == "max_tokens"),
        )

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
        cap = _wire_cap(max_tokens)
        resp = self._send(
            {
                "kind": "create",
                "model": self.model,
                "max_tokens": cap,
                "stream": cap >= STREAM_THRESHOLD,
                "system": _system_blocks(system),
                "messages": [{"role": "user", "content": user}],
            }
        )
        return self._result(resp)

    def complete_json[T: BaseModel](
        self, *, system: str, user: str, schema: type[T], max_tokens: int | None = None
    ) -> tuple[T, LLMResult]:
        cap = _wire_cap(max_tokens)
        resp = self._send(
            {
                "kind": "parse",
                "model": self.model,
                "max_tokens": cap,
                "stream": cap >= STREAM_THRESHOLD,
                "system": _system_blocks(system),
                "messages": [{"role": "user", "content": user}],
                "output_format": schema,
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
        cap = _wire_cap(max_tokens)
        req: dict[str, Any] = {
            "kind": "create",
            "model": self.model,
            "max_tokens": cap,
            "stream": cap >= STREAM_THRESHOLD,
            "system": _system_blocks(system),
            "messages": _messages(turns),
        }
        if tools:
            req["tools"] = _tools(tools)
        resp = self._send(req)
        result = self._result(resp)
        calls = [
            ToolCall(id=b.get("id") or "", name=b.get("name") or "", arguments=b.get("input") or {})
            for b in resp.get("content") or []
            if b.get("type") == "tool_use"
        ]
        return ChatTurn(role="assistant", text=result.text or None, tool_calls=calls), result
