"""Catalog, upload, job, and settings API endpoints."""

import bz2
import gzip
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import polars as pl
import yaml
import zstandard
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from pydantic import BaseModel

from counterstrat.aliases import alias_fingerprint, load_aliases, load_renamer, save_aliases
from counterstrat.config import AppConfig
from counterstrat.corpus import load_manifest
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mining.tendencies import TeamBook
from counterstrat.roundscript.models import RoundScript
from counterstrat.teams import load_or_build_clusters, resolve_team_id
from counterstrat.web.ingest import JobState, _rekey, load_job_state, run_ingest, save_job_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_models_cache: dict[str, tuple[float, list[str]]] = {}
CACHE_TTL = 300.0  # 5 minutes

DEFAULT_MODELS = {
    "anthropic": [
        "claude-sonnet-5",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ],
}


def get_cfg(request: Request) -> AppConfig:
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        cfg = AppConfig.load()
        request.app.state.cfg = cfg
    return cfg


ConfigDep = Annotated[AppConfig, Depends(get_cfg)]


class SettingsResponse(BaseModel):
    provider: str
    model: str
    keys_present: dict[str, bool]


class SettingsUpdateRequest(BaseModel):
    provider: Literal["anthropic", "gemini"]
    model: str
    api_key: str | None = None


# --- 1. Demos Upload & Jobs ---


@router.post("/demos")
def upload_demo(
    background_tasks: BackgroundTasks,
    demo: Annotated[UploadFile, File()],
    cfg: ConfigDep,
) -> dict[str, str]:
    filename = demo.filename or "uploaded.dem"
    lower = filename.lower()
    if lower.endswith(".dem.zst"):
        stem = filename[:-8]
        fmt = "zst"
    elif lower.endswith(".zst"):
        stem = filename[:-4]
        fmt = "zst"
    elif lower.endswith(".dem.gz"):
        stem = filename[:-7]
        fmt = "gz"
    elif lower.endswith(".gz"):
        stem = filename[:-3]
        fmt = "gz"
    elif lower.endswith(".dem.bz2"):
        stem = filename[:-8]
        fmt = "bz2"
    elif lower.endswith(".bz2"):
        stem = filename[:-4]
        fmt = "bz2"
    elif lower.endswith(".dem"):
        stem = filename[:-4]
        fmt = "dem"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format: expected .dem, .dem.zst, .dem.gz, or .dem.bz2",
        )

    if not stem:
        stem = "demo"
    stem = stem.removesuffix(".dem")

    upload_dir = cfg.data_root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / f"{stem}.dem"

    try:
        if fmt == "zst":
            dctx = zstandard.ZstdDecompressor()
            with open(dest_path, "wb") as out_f:
                dctx.copy_stream(demo.file, out_f)
        elif fmt == "gz":
            with (
                gzip.GzipFile(fileobj=demo.file, mode="rb") as gz_f,
                open(dest_path, "wb") as out_f,
            ):
                shutil.copyfileobj(gz_f, out_f)
        elif fmt == "bz2":
            with bz2.BZ2File(demo.file, mode="rb") as bz2_f, open(dest_path, "wb") as out_f:
                shutil.copyfileobj(bz2_f, out_f)
        else:
            with open(dest_path, "wb") as out_f:
                shutil.copyfileobj(demo.file, out_f)
    except Exception as exc:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=400, detail=f"Failed to decompress demo upload: {exc}"
        ) from exc

    # Magic byte validation (first 8 bytes must start with PBDEMS2)
    with open(dest_path, "rb") as f:
        magic = f.read(8)

    if not magic.startswith(b"PBDEMS2"):
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(
            status_code=400, detail="Invalid demo file: missing PBDEMS2 header magic"
        )

    job_id = uuid.uuid4().hex[:12]
    job_state = JobState(job_id=job_id, stage="queued")
    save_job_state(cfg.data_root, job_state)

    background_tasks.add_task(run_ingest, job_id, dest_path, cfg)
    return {"job_id": job_id, "filename": filename}


@router.get("/jobs/{job_id}", response_model=JobState)
def get_job_status(job_id: str, cfg: ConfigDep) -> JobState:
    state = load_job_state(cfg.data_root, job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return state


# --- 2. Catalog (Demos, Teams, TeamBooks, Reports) ---


@router.get("/demos")
def list_demos(cfg: ConfigDep) -> list[dict[str, Any]]:
    manifest_path = cfg.data_root / "corpus.jsonl"
    manifest = load_manifest(manifest_path)
    lake_dir = cfg.data_root / "lake"

    out = []
    for match_id, rec in manifest.items():
        rec_dict = rec.model_dump()
        team_keys: list[str] = []
        rosters_p = lake_dir / match_id / "rosters.parquet"
        if rosters_p.exists():
            try:
                df = pl.read_parquet(rosters_p)
                team_keys = sorted(
                    [
                        str(k)
                        for k in df["team_key"].unique().to_list()
                        if k and str(k) not in ("T", "CT", "None")
                    ]
                )
            except Exception:  # noqa: BLE001, S110
                pass
        rec_dict["team_keys"] = team_keys
        out.append(rec_dict)
    return out


@router.delete("/demos/{match_id}")
def remove_demo(match_id: str, request: Request, cfg: ConfigDep) -> dict[str, Any]:
    """Delete one demo and everything derived from it, then rebuild the rest."""
    from counterstrat.web.maintenance import delete_demo

    try:
        rec = delete_demo(cfg, match_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Demo '{match_id}' not found") from exc
    # In-memory chat sessions may reference the deleted match; drop them all -
    # live ones rebuild from their transcripts on next access.
    sessions = getattr(request.app.state, "chat_sessions", None)
    if isinstance(sessions, dict):
        sessions.clear()
    return {"deleted": match_id, "map_name": rec.map_name}


@router.get("/teams")
def list_teams(cfg: ConfigDep) -> list[dict[str, Any]]:
    """One entry per team CLUSTER: stand-in lineups merge, demos accumulate."""
    clusters = load_or_build_clusters(cfg.data_root)
    unique = {c.team_id: c for c in clusters.values()}

    result = []
    for team_id in sorted(unique):
        c = unique[team_id]
        map_stats: dict[str, dict[str, Any]] = {}
        for match_id, tm in sorted(c.matches.items()):
            st = map_stats.setdefault(tm.map_name, {"demos": 0, "rounds": 0, "matches": []})
            st["demos"] += 1
            st["rounds"] += tm.rounds
            st["matches"].append({"match_id": match_id, "rounds": tm.rounds})
        result.append(
            {
                "team_key": team_id,
                "names": [c.name] if c.name and c.name != team_id else [],
                "maps": sorted(map_stats),
                "demos": len(c.matches),
                "rounds": sum(tm.rounds for tm in c.matches.values()),
                "map_stats": map_stats,
            }
        )
    return result


@router.get("/teams/{team_key}/{map_name}/teambook")
def get_teambook(team_key: str, map_name: str, cfg: ConfigDep) -> dict[str, Any]:
    team_key = resolve_team_id(cfg.data_root, team_key)
    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )
    try:
        return json.loads(tb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read TeamBook: {exc}") from exc


@router.get("/teams/{team_key}/{map_name}/brief")
def get_scout_brief(team_key: str, map_name: str, cfg: ConfigDep) -> dict[str, Any]:
    """The deterministic post-ingest First Look brief (plan Task 8)."""
    team_key = resolve_team_id(cfg.data_root, team_key)
    brief_path = cfg.data_root / "teambooks" / team_key / map_name / "scout_brief.json"
    if not brief_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Scout brief for {team_key} on {map_name} not found"
        )
    try:
        raw = brief_path.read_text(encoding="utf-8")
        # Serve in the user's callout vocabulary (stored artifacts stay canonical).
        return json.loads(load_renamer(cfg.data_root, map_name).rename_text(raw))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read scout brief: {exc}") from exc


def _load_team_bundle(cfg: AppConfig, team_key: str, map_name: str):
    """Card, teambook, re-keyed scripts and lexicon for one (team, map); raises 404s.

    ``team_key`` must already be the canonical cluster id. Scripts are re-keyed
    to it so miners see one team identity across stand-in lineups.
    """
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        raise HTTPException(status_code=404, detail=f"Map card for {map_name} not found")
    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )
    card = MapCard(**yaml.safe_load(card_path.read_text(encoding="utf-8")))
    teambook = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
    overlay_path = get_default_overlay_path(map_name)
    lex = build_lexicon(
        map_name, list(card.zones.keys()), overlay_path if overlay_path.exists() else None
    )
    cluster = load_or_build_clusters(cfg.data_root).get(team_key)
    keys = cluster.all_keys() if cluster else {team_key}
    scripts: list[RoundScript] = []
    for mid in teambook.generated_from:
        scripts_dir = cfg.data_root / "scripts" / mid
        if scripts_dir.exists():
            for sp in sorted(scripts_dir.glob("round_*.json")):
                script = RoundScript.model_validate_json(sp.read_text(encoding="utf-8"))
                scripts.append(_rekey(script, keys, team_key))
    return card, teambook, scripts, lex


def _game_labels(cfg: AppConfig, team_key: str, teambook: TeamBook, scripts) -> dict[str, str]:
    """Human game labels: 'Game 1 (vs team_x)' keyed by match id."""
    clusters = load_or_build_clusters(cfg.data_root)
    labels: dict[str, str] = {}
    for i, mid in enumerate(teambook.generated_from, start=1):
        opponent = None
        for s in scripts:
            if s.match_id != mid:
                continue
            opp_key = s.ct_team_key if s.t_team_key == team_key else s.t_team_key
            if opp_key:
                oc = clusters.get(opp_key)
                if oc is not None and oc.name and oc.name != oc.team_id:
                    opponent = oc.name
                else:
                    opponent = str(opp_key)[:8]
            break
        labels[mid] = f"Game {i} (vs {opponent})" if opponent else f"Game {i}"
    return labels


_MOCK_INSIGHTS = (
    "## 1. Offline mock read\nThey favor `BombsiteA` executes on full buys - "
    "stack utility there (mock insight for UI tests).\n"
)


@router.get("/teams/{team_key}/{map_name}/insights")
def get_insights(
    team_key: str,
    map_name: str,
    cfg: ConfigDep,
    generate: str | None = None,
    mock: str | None = None,
) -> dict[str, Any]:
    """LLM First Read: cached when fresh, generated over the full corpus on demand.

    Without ``generate=1`` this only serves a fresh cache (404 otherwise), so
    the UI can poll cheaply and let the analyst trigger the paid call.
    """
    team_key = resolve_team_id(cfg.data_root, team_key)
    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )
    teambook = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
    alias_fp = alias_fingerprint(load_aliases(cfg.data_root, map_name))

    cache_path = tb_path.parent / "insights.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            fresh = (
                cached.get("generated_from") == list(teambook.generated_from)
                and cached.get("alias_fp", alias_fingerprint({})) == alias_fp
            )
            if fresh and generate != "1":
                return cached
        except Exception as exc:  # noqa: BLE001 - a torn cache regenerates below
            logger.warning("Unreadable insights cache %s: %s", cache_path, exc)
    if generate != "1":
        raise HTTPException(
            status_code=404,
            detail="No AI First Read generated yet for this data; call with generate=1",
        )

    key = cfg.anthropic_api_key if cfg.provider == "anthropic" else cfg.gemini_api_key
    if not key and mock == "1":
        from counterstrat.llm.insights import default_game_labels

        labels = default_game_labels(list(teambook.generated_from))
        payload: dict[str, Any] = {
            "team_key": team_key,
            "map_name": map_name,
            "text": _MOCK_INSIGHTS,
            "warnings": [],
            "generated_from": list(teambook.generated_from),
            "games": [{"label": labels[mid], "match_id": mid} for mid in teambook.generated_from],
            "model": "mock",
            "alias_fp": alias_fp,
        }
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    if not key:
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for provider '{cfg.provider}'. Set it in Settings.",
        )

    card, teambook, scripts, lex = _load_team_bundle(cfg, team_key, map_name)
    try:
        from counterstrat.llm.base import make_client
        from counterstrat.llm.insights import generate_insights
        from counterstrat.mining.econ_policy import build_econ_policy
        from counterstrat.mining.gaps import build_gap_report
        from counterstrat.mining.utility_book import build_utility_book

        client = make_client(cfg)
        insights = generate_insights(
            client,
            card,
            teambook=teambook,
            utility_book=build_utility_book(scripts, team_key),
            gap_report=build_gap_report(scripts, team_key),
            econ_policy=build_econ_policy(scripts, team_key),
            scripts=scripts,
            lexicon=lex,
            game_labels=_game_labels(cfg, team_key, teambook, scripts),
            renamer=load_renamer(cfg.data_root, map_name),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Insights generation failed: {exc}") from exc

    payload = {
        "team_key": team_key,
        "map_name": map_name,
        "text": insights.text,
        "warnings": insights.warnings,
        "generated_from": insights.generated_from,
        "games": insights.games,
        "model": insights.usage.model,
        "alias_fp": alias_fp,
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# --- Callouts: user-defined zone names (feature B) ---


def _vpk_map_names() -> set[str]:
    """Maps shipped as VPKs under maps/<name>/<name>.vpk."""
    from counterstrat.web import ingest as ingest_mod

    maps_dir = ingest_mod.REPO_ROOT / "maps"
    if not maps_dir.exists():
        return set()
    return {d.name for d in maps_dir.iterdir() if d.is_dir() and (d / f"{d.name}.vpk").exists()}


@router.get("/maps")
def list_maps(cfg: ConfigDep) -> list[str]:
    """Every editable map: compiled cards plus VPKs that can supply zone names."""
    root = cfg.data_root / "mapcards"
    card_maps = {p.parent.name for p in root.glob("*/card.yaml")} if root.exists() else set()
    return sorted(card_maps | _vpk_map_names())


def _map_places(cfg: AppConfig, map_name: str) -> list:
    """env_cs_place volumes (name + world origin) from the map VPK, cached."""
    from counterstrat.mapcard.vents import parse_places
    from counterstrat.mapcard.vrf import extract_map_assets
    from counterstrat.web.ingest import _find_vpk_path, _find_vrf_cli

    out_dir = cfg.data_root / "tmp_assets" / map_name
    vents = out_dir / "maps" / map_name / "entities" / "default_ents.vents"
    if not vents.exists():
        vpk = _find_vpk_path(map_name, cfg)
        vrf_cli = _find_vrf_cli()
        if vpk is None or vrf_cli is None:
            return []
        try:
            extract_map_assets(vpk, vrf_cli, out_dir)
        except Exception as exc:  # noqa: BLE001 - positions are a nicety, not a requirement
            logger.warning("VPK asset extraction failed for %s: %s", map_name, exc)
            return []
    try:
        return parse_places(vents)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable vents for %s: %s", map_name, exc)
        return []


def _radar_calibration(cfg: AppConfig, map_name: str):
    """Cached radar calibration; extracted from the CS2 install when possible."""
    from counterstrat.radar.extract import extract_radar_assets, load_cached_assets
    from counterstrat.web.ingest import _find_vrf_cli

    assets = load_cached_assets(cfg.data_root, map_name)
    if assets is not None:
        return assets.calibration
    install = getattr(cfg, "cs2_install_path", None)
    vrf_cli = _find_vrf_cli()
    if not install or vrf_cli is None:
        return None
    try:
        return extract_radar_assets(Path(install), vrf_cli, map_name, cfg.data_root).calibration
    except Exception as exc:  # noqa: BLE001 - the editor degrades to a table
        logger.warning("Radar extraction failed for %s: %s", map_name, exc)
        return None


def _callout_zone_names(cfg: AppConfig, map_name: str, places: list) -> list[str] | None:
    """Card zones when compiled (the mined vocabulary), else VPK place names."""
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if card_path.exists():
        card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        return sorted((card_data.get("zones") or {}).keys())
    if places:
        return sorted({p.place_name for p in places})
    return None


def _tick_positions(cfg: AppConfig, map_name: str, zones: list[str], cal) -> dict[str, tuple]:
    """Fallback anchors from lake ticks for zones the VPK volumes did not name."""
    from counterstrat.radar.coords import game_to_norm, is_lower_level

    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    for match_id, rec in sorted(manifest.items()):
        if rec.map_name != map_name:
            continue
        ticks_path = cfg.data_root / "lake" / match_id / "ticks.parquet"
        if not ticks_path.exists():
            continue
        try:
            df = pl.read_parquet(ticks_path, columns=["X", "Y", "Z", "last_place_name", "is_alive"])
            centroids = (
                df.filter(pl.col("is_alive") & pl.col("last_place_name").is_in(zones))
                .group_by("last_place_name")
                .agg(pl.col("X").mean(), pl.col("Y").mean(), pl.col("Z").mean())
            )
            out: dict[str, tuple] = {}
            for row in centroids.iter_rows(named=True):
                u, v = game_to_norm(cal, row["X"], row["Y"])
                if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                    level = "lower" if is_lower_level(cal, row["Z"]) else "default"
                    out[row["last_place_name"]] = (round(u, 4), round(v, 4), level)
            if out:
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("Zone positions unavailable from %s: %s", ticks_path, exc)
    return {}


def _zone_anchors(cfg: AppConfig, map_name: str, zones: list[str], places: list) -> dict:
    """(u, v, level) per zone: lake-tick centroids first, VPK volumes as fallback.

    Tick centroids mark where players actually occupy a zone. Volume origins are
    entity pivots - good enough to label a never-ingested map, but visibly off
    on played maps, so they only fill the gaps.
    """
    from counterstrat.radar.coords import game_to_norm, is_lower_level

    cal = _radar_calibration(cfg, map_name)
    if cal is None:
        return {}
    anchors: dict[str, tuple] = dict(_tick_positions(cfg, map_name, zones, cal))
    by_name: dict[str, list] = {}
    for p in places:
        by_name.setdefault(p.place_name, []).append(p.origin)
    for zone in zones:
        if zone in anchors:
            continue
        origins = by_name.get(zone)
        if not origins:
            continue
        x = sum(o[0] for o in origins) / len(origins)
        y = sum(o[1] for o in origins) / len(origins)
        z = sum(o[2] for o in origins) / len(origins)
        u, v = game_to_norm(cal, x, y)
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            level = "lower" if is_lower_level(cal, z) else "default"
            anchors[zone] = (round(u, 4), round(v, 4), level)
    return anchors


@router.get("/maps/{map_name}/callouts")
def get_callouts(map_name: str, cfg: ConfigDep) -> dict[str, Any]:
    """Every zone with its game name, the user's alias, a label anchor, and level."""
    places = _map_places(cfg, map_name)
    zones = _callout_zone_names(cfg, map_name, places)
    if zones is None:
        raise HTTPException(status_code=404, detail=f"No map card or VPK zone data for {map_name}")
    aliases = load_aliases(cfg.data_root, map_name)
    anchors = _zone_anchors(cfg, map_name, zones, places)
    levels = sorted({a[2] for a in anchors.values()}) or ["default"]
    return {
        "map_name": map_name,
        "levels": ["default", "lower"] if "lower" in levels else ["default"],
        "zones": [
            {
                "name": z,
                "alias": aliases.get(z),
                "u": anchors.get(z, (None, None, None))[0],
                "v": anchors.get(z, (None, None, None))[1],
                "level": anchors.get(z, (None, None, None))[2],
            }
            for z in zones
        ],
    }


class AliasUpdateRequest(BaseModel):
    aliases: dict[str, str]


@router.put("/maps/{map_name}/aliases")
def put_aliases(map_name: str, req: AliasUpdateRequest, cfg: ConfigDep) -> dict[str, Any]:
    """Persist the user's callouts; empty values remove an alias."""
    zones = _callout_zone_names(cfg, map_name, _map_places(cfg, map_name))
    if zones is None:
        raise HTTPException(status_code=404, detail=f"No map card or VPK zone data for {map_name}")
    try:
        saved = save_aliases(cfg.data_root, map_name, req.aliases, set(zones))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"map_name": map_name, "aliases": saved}


@router.get("/reports/{team_key}/{map_name}")
def get_report(team_key: str, map_name: str, cfg: ConfigDep, mock: str | None = None) -> Response:
    team_key = resolve_team_id(cfg.data_root, team_key)
    dossier_path = cfg.data_root / "teambooks" / team_key / map_name / "dossier.md"
    if dossier_path.exists():
        return Response(
            content=dossier_path.read_text(encoding="utf-8"), media_type="text/markdown"
        )

    key = cfg.anthropic_api_key if cfg.provider == "anthropic" else cfg.gemini_api_key
    if not key:
        if mock == "1":
            mock_content = f"# Anti-Strat Dossier: {team_key} on {map_name}\n\n## 1. Executive Summary\nOffline UI test dossier."
            return Response(content=mock_content, media_type="text/markdown")
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for provider '{cfg.provider}'. Please set it in Settings.",
        )

    # Bundle loading resolves the cluster and re-keys scripts, so the dossier
    # mines the full merged corpus across stand-in lineups.
    card, teambook, scripts, lex = _load_team_bundle(cfg, team_key, map_name)
    try:
        from counterstrat.llm.base import make_client
        from counterstrat.llm.dossier import generate as generate_dossier

        client = make_client(cfg)
        dossier = generate_dossier(
            client, card, teambook, scripts, lex, renamer=load_renamer(cfg.data_root, map_name)
        )
        dossier_path.parent.mkdir(parents=True, exist_ok=True)
        dossier_path.write_text(dossier.text, encoding="utf-8")
        return Response(content=dossier.text, media_type="text/markdown")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate dossier: {exc}") from exc


# --- 3. Settings & Models ---


@router.get("/settings", response_model=SettingsResponse)
def get_settings(cfg: ConfigDep) -> SettingsResponse:
    model_name = cfg.anthropic_model if cfg.provider == "anthropic" else cfg.gemini_model
    return SettingsResponse(
        provider=cfg.provider,
        model=model_name,
        keys_present={
            "anthropic": bool(cfg.anthropic_api_key),
            "gemini": bool(cfg.gemini_api_key),
        },
    )


@router.post("/settings", response_model=SettingsResponse)
def update_settings(
    req: SettingsUpdateRequest,
    cfg: ConfigDep,
) -> SettingsResponse:
    settings_path = cfg.data_root / "settings.json"
    data: dict[str, Any] = {}
    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:  # noqa: BLE001
            data = {}

    data["provider"] = req.provider
    if req.provider == "anthropic":
        data["anthropic_model"] = req.model
        if req.api_key:
            data["anthropic_api_key"] = req.api_key
    elif req.provider == "gemini":
        data["gemini_model"] = req.model
        if req.api_key:
            data["gemini_api_key"] = req.api_key

    data["data_root"] = str(cfg.data_root)

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Update in-memory configuration
    cfg.provider = req.provider
    if req.provider == "anthropic":
        cfg.anthropic_model = req.model
        if req.api_key:
            cfg.anthropic_api_key = req.api_key
    elif req.provider == "gemini":
        cfg.gemini_model = req.model
        if req.api_key:
            cfg.gemini_api_key = req.api_key

    return SettingsResponse(
        provider=cfg.provider,
        model=cfg.anthropic_model if cfg.provider == "anthropic" else cfg.gemini_model,
        keys_present={
            "anthropic": bool(cfg.anthropic_api_key),
            "gemini": bool(cfg.gemini_api_key),
        },
    )


@router.get("/models")
def list_models(cfg: ConfigDep, provider: str | None = None) -> list[str]:
    selected_provider = provider or cfg.provider
    now = time.time()
    if selected_provider in _models_cache:
        cached_time, models = _models_cache[selected_provider]
        if now - cached_time < CACHE_TTL:
            return models

    key = cfg.anthropic_api_key if selected_provider == "anthropic" else cfg.gemini_api_key
    if not key:
        return DEFAULT_MODELS.get(selected_provider, [])

    try:
        if selected_provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=key)
            page = client.models.list(limit=100)
            models = [m.id for m in page.data]
        elif selected_provider == "gemini":
            from google import genai

            client = genai.Client(api_key=key)
            models = [m.name.removeprefix("models/") for m in client.models.list()]
        else:
            models = DEFAULT_MODELS.get(selected_provider, [])

        _models_cache[selected_provider] = (now, models)
        return models
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream provider failure: {exc}") from exc
