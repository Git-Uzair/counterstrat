"""Radar image, calibration, and coordinate-layer endpoints (spec item N2)."""

import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from counterstrat.config import AppConfig
from counterstrat.corpus import load_manifest
from counterstrat.radar.extract import RadarAssets, extract_radar_assets, load_cached_assets
from counterstrat.radar.layers import LayerFilters, build_layers, lake_frames
from counterstrat.teams import load_or_build_clusters
from counterstrat.web.ingest import _find_vrf_cli
from counterstrat.web.routes import ConfigDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/radar")

# map_name lands in filesystem paths, so keep it to the CS2 naming convention.
_MAP_NAME = re.compile(r"^[a-z0-9_]{1,64}$")

Level = Literal["default", "lower"]
LayerLevel = Literal["default", "lower", "all"]


class RadarInfo(BaseModel):
    map_name: str
    pos_x: float
    pos_y: float
    scale: float
    image_px: int
    lower_altitude_max: float | None
    levels: list[str]
    image_url: str
    lower_image_url: str | None


def _check_map_name(map_name: str) -> str:
    if not _MAP_NAME.match(map_name):
        raise HTTPException(status_code=400, detail=f"Invalid map name '{map_name}'")
    return map_name


def _resolve_assets(map_name: str, cfg: AppConfig) -> RadarAssets:
    """Cache-first radar bundle, extracting from the CS2 install on a miss."""
    _check_map_name(map_name)
    cached = load_cached_assets(cfg.data_root, map_name)
    if cached is not None:
        return cached

    if cfg.cs2_install_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No cached radar for this map and no CS2 install path is configured. "
                "Set cs2_install_path in Settings to enable radar overlays "
                "(see docs/NEEDS-FROM-YOU.md item N2)."
            ),
        )
    vrf_cli = _find_vrf_cli()
    if vrf_cli is None:
        raise HTTPException(
            status_code=503, detail="Source2Viewer-CLI is not vendored under tools/vrf/"
        )
    try:
        return extract_radar_assets(cfg.cs2_install_path, vrf_cli, map_name, cfg.data_root)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"No radar art for map '{map_name}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Radar extraction failed for %s", map_name)
        raise HTTPException(status_code=500, detail=f"Radar extraction failed: {exc}") from exc


def _parse_rounds(rounds: str | None) -> list[int] | None:
    if not rounds:
        return None
    try:
        return [int(part) for part in rounds.split(",") if part.strip()]
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"'rounds' must be comma-separated integers, got {rounds!r}"
        ) from exc


def _parse_players(players: str | None) -> list[int] | None:
    if not players:
        return None
    try:
        return [int(part) for part in players.split(",") if part.strip()]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"'players' must be comma-separated steamids, got {players!r}",
        ) from exc


@router.get("/{map_name}/info", response_model=RadarInfo)
def get_radar_info(map_name: str, cfg: ConfigDep) -> RadarInfo:
    """Calibration plus the image URLs the client should load."""
    assets = _resolve_assets(map_name, cfg)
    cal = assets.calibration
    return RadarInfo(
        map_name=cal.map_name,
        pos_x=cal.pos_x,
        pos_y=cal.pos_y,
        scale=cal.scale,
        image_px=cal.image_px,
        lower_altitude_max=cal.lower_altitude_max,
        levels=assets.levels,
        image_url=f"/api/radar/{map_name}/image?level=default",
        lower_image_url=(
            f"/api/radar/{map_name}/image?level=lower" if assets.lower_image else None
        ),
    )


@router.get("/{map_name}/image")
def get_radar_image(map_name: str, cfg: ConfigDep, level: Level = "default") -> FileResponse:
    """The VRF-decompiled overhead PNG, served unmodified."""
    assets = _resolve_assets(map_name, cfg)
    path: Path | None = assets.lower_image if level == "lower" else assets.image
    if path is None or not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Map '{map_name}' has no '{level}' radar image"
        )
    return FileResponse(path, media_type="image/png")


@router.get("/{team_key}/{map_name}/layers")
def get_radar_layers(
    team_key: str,
    map_name: str,
    cfg: ConfigDep,
    side: Literal["T", "CT"] | None = None,
    level: LayerLevel = "all",
    rounds: str | None = None,
    players: str | None = None,
    matches: str | None = None,
    trail_rounds: Annotated[int | None, Query(ge=1, le=30)] = 4,
    stride: Annotated[int, Query(ge=1, le=64)] = 8,
    grid: Annotated[int, Query(ge=8, le=256)] = 128,
) -> dict[str, Any]:
    """Normalized coordinate layers for one (team, map) selection.

    ``matches`` (csv of match ids) scopes the view to specific demos; the team
    key may be a cluster id or any lineup key - stand-in lineups merge.
    """
    # Resolve the corpus BEFORE the radar assets: a map with no ingested match
    # must answer 404, not the 503 that an un-extractable radar would raise.
    _check_map_name(map_name)
    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    match_ids = sorted(mid for mid, rec in manifest.items() if rec.map_name == map_name)
    if matches:
        wanted = {part.strip() for part in matches.split(",") if part.strip()}
        match_ids = [mid for mid in match_ids if mid in wanted]
    if not match_ids:
        raise HTTPException(status_code=404, detail=f"No ingested matches on map '{map_name}'")

    cluster = load_or_build_clusters(cfg.data_root).get(team_key)
    team_keys: list[str] = sorted(cluster.all_keys()) if cluster else [team_key]

    cal = _resolve_assets(map_name, cfg).calibration
    filters = LayerFilters(
        side=side,
        level=level,
        round_nums=_parse_rounds(rounds),
        trail_rounds=trail_rounds,
        stride=stride,
        grid=grid,
        players=_parse_players(players),
    )
    frames = lake_frames(cfg.data_root / "lake", match_ids)
    return build_layers(frames, cal, team_keys, filters)
