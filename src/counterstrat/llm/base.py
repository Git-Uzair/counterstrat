"""LLM client protocol and types (spec Phase 2a, Task 11 stub, Task 18 implementation)."""

import time
from collections.abc import Callable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from counterstrat.config import AppConfig

RETRY_TRIES = 3
RETRY_BASE_DELAY_S = 0.5

# A transport takes one provider-shaped request dict and returns the provider's
# response as a dict (SDK objects go through ``model_dump``). Tests inject
# replay transports over the JSON fixtures in ``tests/fixtures/llm/``.
Transport = Callable[[dict], dict]


class LLMResult(BaseModel):
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str = "mock"
    provider: str = "mock"
    # The provider stopped at its output token ceiling: the text is cut short.
    truncated: bool = False


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict
    # Provider-opaque reasoning signature (Gemini's per-part `thought_signature`).
    # Gemini 3 models reject a tool loop whose function_call parts come back without
    # it, so adapters carry it across turns untouched.
    signature: str | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class LLMClient(Protocol):
    # max_tokens bounds VISIBLE output text; None means "the model's own maximum"
    # so analysis-grade calls are never cut short by a self-imposed ceiling.
    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> LLMResult: ...

    def complete_json[T: BaseModel](
        self, *, system: str, user: str, schema: type[T], max_tokens: int | None = None
    ) -> tuple[T, LLMResult]: ...

    def chat(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        tools: list[ToolSpec],
        max_tokens: int | None = None,
    ) -> tuple[ChatTurn, LLMResult]: ...


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    tries: int = RETRY_TRIES,
    base_delay: float = RETRY_BASE_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Calls fn, retrying 429/5xx responses with exponential backoff."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            status = _status_of(exc)
            retryable = status == 429 or (status is not None and 500 <= status < 600)
            if not retryable or attempt == tries - 1:
                raise
            sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def make_client(cfg: AppConfig) -> LLMClient:
    """Builds the provider adapter selected by config (keys come from config/env)."""
    if cfg.provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        from counterstrat.llm.anthropic_client import AnthropicClient

        return AnthropicClient(api_key=cfg.anthropic_api_key, model=cfg.anthropic_model)
    if cfg.provider == "gemini":
        if not cfg.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        from counterstrat.llm.gemini_client import GeminiClient

        return GeminiClient(api_key=cfg.gemini_api_key, model=cfg.gemini_model)
    raise ValueError(f"Unknown provider: {cfg.provider}")
