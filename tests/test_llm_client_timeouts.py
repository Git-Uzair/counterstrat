"""Provider SDKs are constructed with bounded timeouts and retries: a dead
route to the API surfaces as an error within minutes, never an endless
spinner (field bug, 2026-09-07: Anthropic First Read spun forever)."""

from counterstrat.llm.anthropic_client import (
    CONNECT_TIMEOUT_S,
    REQUEST_TIMEOUT_S,
    SDK_MAX_RETRIES,
    AnthropicClient,
)
from counterstrat.llm.gemini_client import REQUEST_TIMEOUT_MS, GeminiClient


def test_anthropic_sdk_is_time_bounded():
    sdk = AnthropicClient(api_key="k")._ensure_sdk()
    assert sdk.max_retries == SDK_MAX_RETRIES
    assert sdk.timeout.connect == CONNECT_TIMEOUT_S
    assert sdk.timeout.read == REQUEST_TIMEOUT_S


def test_gemini_sdk_is_time_bounded():
    sdk = GeminiClient(api_key="k")._ensure_sdk()
    timeout = sdk._api_client._http_options.timeout
    assert timeout == REQUEST_TIMEOUT_MS
