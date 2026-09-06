"""Application configuration (spec Phase 2a, Task 18).

Precedence: ``data/settings.json`` (written by the settings endpoint) > env vars
(via dotenv) > the defaults declared on :class:`AppConfig`.
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

DEFAULT_SETTINGS_PATH = Path("data") / "settings.json"

# --- Development policy (user directive; never used at app runtime) ---
DEV_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_SONNET_MODEL = "claude-sonnet-5"


class AppConfig(BaseModel):
    provider: Literal["anthropic", "gemini"] = "anthropic"
    # First-run seeds only -- the runtime source of truth is the settings UI
    # (user-provided key + user-selected model, Tasks 21/23).
    anthropic_model: str = DEFAULT_SONNET_MODEL
    gemini_model: str = "gemini-2.5-pro"
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    data_root: Path = Path("data")
    cs2_install_path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        """Builds a config from settings file, then environment, then defaults."""
        load_dotenv()
        settings_path = path or DEFAULT_SETTINGS_PATH
        data: dict[str, Any] = {}
        if settings_path.exists():
            try:
                loaded = json.loads(settings_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError):
                data = {}

        # First hit wins; GOOGLE_GENERATIVE_AI_API_KEY is what dev machines carry.
        gemini_key = (
            data.get("gemini_api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        cs2_path = data.get("cs2_install_path") or os.getenv("CS2_INSTALL_PATH")

        return cls.model_validate(
            {
                "provider": data.get("provider") or os.getenv("LLM_PROVIDER") or "anthropic",
                "anthropic_model": (
                    data.get("anthropic_model")
                    or os.getenv("ANTHROPIC_MODEL")
                    or DEFAULT_SONNET_MODEL
                ),
                "gemini_model": (
                    data.get("gemini_model") or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
                ),
                "anthropic_api_key": (
                    data.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
                ),
                "gemini_api_key": gemini_key,
                "data_root": Path(data.get("data_root") or os.getenv("DATA_ROOT") or "data"),
                "cs2_install_path": Path(cs2_path) if cs2_path else None,
            }
        )


def resolve_latest_sonnet(api_key: str, fallback: str = DEFAULT_SONNET_MODEL) -> str:
    """Newest sonnet-family model id from ``GET /v1/models`` (created_at desc).

    Used by live-marked tests and dev harnesses (quiz/benchmark) only; falls back
    to the pinned id when the listing is unavailable.
    """
    import anthropic

    try:
        page = anthropic.Anthropic(api_key=api_key).models.list(limit=100)
        sonnets = [m for m in page.data if "sonnet" in m.id]
        sonnets.sort(key=lambda m: m.created_at, reverse=True)
        return sonnets[0].id if sonnets else fallback
    except anthropic.AnthropicError:
        return fallback
