"""Setup readiness: everything ingest needs, prepared up front and reported.

The app refuses demo ingestion until setup is complete (owner decree,
2026-09-06: a new user must never ingest, hit lazy one-time work mid-flow,
and think the app broke). Three blockers - the CS2 install folder, the map
decompiler, an LLM API key - plus one background bootstrap thread that does
the slow work (decompiler download, shipped-pool map-asset warmup) at
startup/save time instead of on first touch. ``GET /api/readiness`` reports
state and the drop zone stays gated until ``ready`` is true.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from counterstrat.config import AppConfig
from counterstrat.web.ingest import _ensure_vrf_cli, _find_vpk_path, _find_vrf_cli

logger = logging.getLogger(__name__)


def cs2_path_valid(install: Path | None) -> bool:
    """True when the folder looks like a CS2 install (has game/csgo)."""
    return bool(install) and (Path(install) / "game" / "csgo").is_dir()


def _api_key_ok(cfg: AppConfig) -> bool:
    key = cfg.anthropic_api_key if cfg.provider == "anthropic" else cfg.gemini_api_key
    return bool(key)


class BootstrapState(BaseModel):
    """The background bootstrap's observable state (one per process)."""

    status: Literal["waiting", "downloading", "warming", "ready", "failed"] = "waiting"
    detail: str = ""


_STATE = BootstrapState()
_STATE_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None


def _set_state(status: str, detail: str = "") -> None:
    with _STATE_LOCK:
        _STATE.status = status  # type: ignore[assignment]
        _STATE.detail = detail


def _warm_maps(cfg: AppConfig) -> list[str]:
    """Shipped-pool maps whose VPK is reachable but vents cache is missing."""
    from counterstrat.mapcard.cards import SHIPPED_CARDS_DIR

    pool = sorted(p.stem for p in SHIPPED_CARDS_DIR.glob("*.yaml"))
    todo: list[str] = []
    for map_name in pool:
        vents = (
            cfg.data_root
            / "tmp_assets"
            / map_name
            / "maps"
            / map_name
            / "entities"
            / "default_ents.vents"
        )
        if vents.exists():
            continue
        if _find_vpk_path(map_name, cfg) is not None:
            todo.append(map_name)
    return todo


def _run_bootstrap(cfg: AppConfig) -> None:
    from counterstrat.mapcard.vrf import extract_map_assets

    try:
        _set_state("downloading", "fetching the map decompiler (one-time, ~50 MB)")
        cli = _ensure_vrf_cli()
        if cli is None:
            _set_state(
                "failed",
                "decompiler download failed (offline or blocked by antivirus); "
                "it retries automatically, or place the CLI under tools/vrf manually",
            )
            return
        todo = _warm_maps(cfg)
        for i, map_name in enumerate(todo, 1):
            _set_state("warming", f"extracting map data {i}/{len(todo)}: {map_name}")
            vpk = _find_vpk_path(map_name, cfg)
            if vpk is None:
                continue
            try:
                extract_map_assets(vpk, cli, cfg.data_root / "tmp_assets" / map_name)
            except Exception as exc:  # noqa: BLE001 - warmup is best-effort
                logger.warning("Map asset warmup failed for %s: %s", map_name, exc)
        _set_state("ready", "")
    except Exception as exc:
        logger.exception("Setup bootstrap failed")
        _set_state("failed", str(exc))


def start_bootstrap(cfg: AppConfig) -> None:
    """Idempotently run the bootstrap in the background; retries after failure.

    No-op under COUNTERSTRAT_NO_VRF_DOWNLOAD (tests, air-gapped operators):
    readiness then reports plain filesystem truth without spawning work.
    """
    global _THREAD
    if os.environ.get("COUNTERSTRAT_NO_VRF_DOWNLOAD"):
        return
    with _STATE_LOCK:
        if _THREAD is not None and _THREAD.is_alive():
            return
        if _STATE.status == "ready" and not _warm_maps(cfg) and _find_vrf_cli() is not None:
            return
        _THREAD = threading.Thread(target=_run_bootstrap, args=(cfg,), daemon=True)
        _THREAD.start()


def ingest_blockers(cfg: AppConfig) -> list[str]:
    """Human-readable list of what still blocks demo ingestion; [] = ready."""
    blockers: list[str] = []
    if not cs2_path_valid(cfg.cs2_install_path):
        blockers.append("Set your CS2 install folder in Settings")
    if _find_vrf_cli() is None:
        blockers.append("The map decompiler is not ready yet")
    if not _api_key_ok(cfg):
        blockers.append(f"Add your {cfg.provider} API key in Settings")
    return blockers


def snapshot(cfg: AppConfig) -> dict:
    """The readiness payload the UI polls."""
    with _STATE_LOCK:
        bootstrap = _STATE.model_copy()
    cs2_ok = cs2_path_valid(cfg.cs2_install_path)
    cli_ok = _find_vrf_cli() is not None
    key_ok = _api_key_ok(cfg)
    return {
        "ready": cs2_ok and cli_ok and key_ok,
        "items": [
            {
                "id": "cs2_path",
                "label": "CS2 install folder",
                "ok": cs2_ok,
                "detail": "" if cs2_ok else "Set it in Settings (folder containing game\\csgo)",
            },
            {
                "id": "decompiler",
                "label": "Map decompiler",
                "ok": cli_ok,
                "status": bootstrap.status,
                "detail": "" if cli_ok else (bootstrap.detail or "waiting to download"),
            },
            {
                "id": "api_key",
                "label": "AI API key",
                "ok": key_ok,
                "detail": "" if key_ok else f"Add your {cfg.provider} key in Settings",
            },
        ],
        "bootstrap": bootstrap.model_dump(),
    }
