"""LLM client protocol and types (spec Phase 2a, Task 11 stub, Task 18 implementation)."""

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = "mock"
    provider: str = "mock"


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict


class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, max_tokens: int = 4096) -> LLMResult: ...

    def complete_json[T: BaseModel](
        self, *, system: str, user: str, schema: type[T], max_tokens: int = 4096
    ) -> tuple[T, LLMResult]: ...

    def chat(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
    ) -> tuple[ChatTurn, LLMResult]: ...
