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
    check_budget,
    turn_texts,
)

PROVIDER = "anthropic"


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
        max_input_tokens: int = 190_000,
        transport: Transport | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.max_input_tokens = max_input_tokens
        self._transport = transport or self._sdk_transport
        self._sdk: Any = None

    def _sdk_transport(self, req: dict[str, Any]) -> dict[str, Any]:
        import anthropic

        if self._sdk is None:
            self._sdk = anthropic.Anthropic(api_key=self.api_key)
        req = dict(req)
        kind = req.pop("kind")
        messages = self._sdk.messages
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

    def complete(self, *, system: str, user: str, max_tokens: int = 4096) -> LLMResult:
        check_budget(self.max_input_tokens, system, user)
        resp = self._send(
            {
                "kind": "create",
                "model": self.model,
                "max_tokens": max_tokens,
                "system": _system_blocks(system),
                "messages": [{"role": "user", "content": user}],
            }
        )
        return self._result(resp)

    def complete_json[T: BaseModel](
        self, *, system: str, user: str, schema: type[T], max_tokens: int = 4096
    ) -> tuple[T, LLMResult]:
        check_budget(self.max_input_tokens, system, user)
        resp = self._send(
            {
                "kind": "parse",
                "model": self.model,
                "max_tokens": max_tokens,
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
        max_tokens: int = 4096,
    ) -> tuple[ChatTurn, LLMResult]:
        check_budget(self.max_input_tokens, system, *turn_texts(turns))
        req: dict[str, Any] = {
            "kind": "create",
            "model": self.model,
            "max_tokens": max_tokens,
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
