"""Background ingest pipeline chaining register -> extract -> card -> serialize -> mine."""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal

import polars as pl
import yaml
from pydantic import BaseModel

from counterstrat.config import AppConfig
from counterstrat.corpus import load_manifest, register_demo
from counterstrat.lake.extract import extract_lake
from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.transitions import zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.vrf import extract_map_assets
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.mining.brief import build_scout_brief
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.tendencies import build_teambook
from counterstrat.mining.utility_book import build_utility_book
from counterstrat.roundscript.models import RoundScript
from counterstrat.roundscript.serialize import serialize_match
from counterstrat.teams import build_team_clusters, write_team_clusters

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


class JobState(BaseModel):
    job_id: str
    stage: Literal["queued", "extracting", "mapcard", "serializing", "mining", "done", "error"]
    match_id: str | None = None
    map_name: str | None = None
    detail: str = ""


def save_job_state(data_root: Path, state: JobState) -> None:
    """Atomically save job state to <data_root>/jobs/<job_id>.json."""
    jobs_dir = data_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    target = jobs_dir / f"{state.job_id}.json"
    temp_target = jobs_dir / f"{state.job_id}.json.tmp_{os.getpid()}_{uuid.uuid4().hex}"
    temp_target.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp_target, target)


def load_job_state(data_root: Path, job_id: str) -> JobState | None:
    """Load job state from <data_root>/jobs/<job_id>.json or return None."""
    job_path = data_root / "jobs" / f"{job_id}.json"
    if not job_path.exists():
        return None
    try:
        return JobState.model_validate_json(job_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _find_vpk_path(map_name: str, cfg: AppConfig) -> Path | None:
    candidates: list[Path | None] = [
        REPO_ROOT / "maps" / map_name / f"{map_name}.vpk",
        REPO_ROOT / f"maps/{map_name}/{map_name}.vpk",
        Path("maps") / map_name / f"{map_name}.vpk",
    ]
    if cfg.cs2_install_path:
        candidates.append(cfg.cs2_install_path / "game" / "csgo" / "maps" / f"{map_name}.vpk")
    for cand in candidates:
        if cand and cand.exists():
            return cand
    return None


def _find_vrf_cli() -> Path | None:
    candidates = [
        REPO_ROOT / "tools" / "vrf" / "Source2Viewer-CLI.exe",
        REPO_ROOT / "tools" / "vrf" / "Source2Viewer-CLI",
        Path("tools") / "vrf" / "Source2Viewer-CLI.exe",
        Path("tools") / "vrf" / "Source2Viewer-CLI",
    ]
    which_path = shutil.which("Source2Viewer-CLI")
    if which_path:
        candidates.append(Path(which_path))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _rekey(script: RoundScript, keys: set[str], canonical: str) -> RoundScript:
    """The script with any cluster lineup key replaced by the canonical team id."""
    update: dict[str, str] = {}
    if script.t_team_key in keys and script.t_team_key != canonical:
        update["t_team_key"] = canonical
    if script.ct_team_key in keys and script.ct_team_key != canonical:
        update["ct_team_key"] = canonical
    return script.model_copy(update=update) if update else script


def _scripts_for_keys(
    data_root: Path,
    map_name: str,
    keys: set[str],
    canonical: str,
    current: list[RoundScript],
) -> list[RoundScript]:
    """Current + stored scripts on this map where either side's key is in ``keys``.

    Matching side keys are rewritten to ``canonical`` so miners and tools see one
    team identity across stand-in lineups (team clustering). Without this union
    each ingest would rebuild the teambook from one demo and overwrite the
    accumulated profile (plan Task 0).
    """
    current_ids = {s.match_id for s in current}
    out = [
        _rekey(s, keys, canonical)
        for s in current
        if keys & {s.t_team_key, s.ct_team_key} or canonical in (s.t_team_key, s.ct_team_key)
    ]
    manifest = load_manifest(data_root / "corpus.jsonl")
    for match_id, rec in sorted(manifest.items()):
        if rec.map_name != map_name or match_id in current_ids:
            continue
        for script_path in sorted((data_root / "scripts" / match_id).glob("round_*.json")):
            try:
                s = RoundScript.model_validate_json(script_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - one bad script must not kill mining
                logger.warning("Skipping unreadable round script %s: %s", script_path, exc)
                continue
            if keys & {s.t_team_key, s.ct_team_key}:
                out.append(_rekey(s, keys, canonical))
    return out


def _scripts_for_team(
    data_root: Path, map_name: str, team_key: str, current: list[RoundScript]
) -> list[RoundScript]:
    """Single-key convenience wrapper over :func:`_scripts_for_keys`."""
    return _scripts_for_keys(data_root, map_name, {team_key}, team_key, current)


def run_ingest(job_id: str, demo_path: Path, cfg: AppConfig) -> None:
    """Execute the full offline ingest pipeline for a demo file."""
    state = load_job_state(cfg.data_root, job_id) or JobState(job_id=job_id, stage="queued")
    try:
        # 1. Register and Extract
        state.stage = "extracting"
        save_job_state(cfg.data_root, state)

        rec = register_demo(demo_path, cfg.data_root / "corpus.jsonl")
        state.match_id = rec.match_id
        state.map_name = rec.map_name
        save_job_state(cfg.data_root, state)

        lake = extract_lake(rec, cfg.data_root / "lake")

        # 2. Map Card & Lexicon
        state.stage = "mapcard"
        save_job_state(cfg.data_root, state)

        card_path = cfg.data_root / "mapcards" / rec.map_name / "card.yaml"
        card: MapCard | None = None
        if card_path.exists():
            try:
                card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
                if isinstance(card_data, dict):
                    card = MapCard(**card_data)
            except Exception:  # noqa: BLE001
                card = None

        if card is None:
            vpk_path = _find_vpk_path(rec.map_name, cfg)
            vrf_cli = _find_vrf_cli()
            if vpk_path and vrf_cli:
                try:
                    assets_dir = cfg.data_root / "tmp_assets" / rec.map_name
                    assets = extract_map_assets(vpk_path, vrf_cli, assets_dir)
                    places = unique_places(parse_places(assets.vents))
                    overlay_path = get_default_overlay_path(rec.map_name)
                    lexicon_card = build_lexicon(
                        rec.map_name, places, overlay_path if overlay_path.exists() else None
                    )
                    ticks_df = pl.read_parquet(lake.ticks)
                    rounds_df = pl.read_parquet(lake.rounds)
                    graph = zone_graph(ticks_df)
                    card = compile_card(
                        lexicon=lexicon_card,
                        graph=graph,
                        ticks=ticks_df,
                        rounds=rounds_df,
                        map_name=rec.map_name,
                        patch_version=rec.patch_version,
                    )
                    card_path.parent.mkdir(parents=True, exist_ok=True)
                    card_path.write_text(card.to_yaml(), encoding="utf-8")
                    state.detail = ""
                except Exception as exc:
                    logger.exception("Mapcard compilation failed: %s", exc)  # noqa: TRY401
                    state.detail = f"card missing: supply maps/{rec.map_name}/{rec.map_name}.vpk"
            else:
                state.detail = f"card missing: supply maps/{rec.map_name}/{rec.map_name}.vpk"
        else:
            state.detail = ""

        ticks = pl.read_parquet(lake.ticks)
        mapper = ZoneMapper.fit(ticks)

        overlay_path = get_default_overlay_path(rec.map_name)
        if card is not None:
            places = list(card.zones.keys())
            lex = build_lexicon(
                rec.map_name, places, overlay_path if overlay_path.exists() else None
            )
        else:
            tick_places = [
                str(p)
                for p in ticks["last_place_name"].drop_nulls().unique().to_list()
                if str(p).strip()
            ]
            overlay_zones: list[str] = []
            if overlay_path.exists():
                try:
                    overlay_data = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
                    overlay_zones = list((overlay_data.get("zones") or {}).keys())
                except Exception:  # noqa: BLE001, S110
                    pass
            all_places = sorted(set(tick_places) | set(overlay_zones))
            lex = build_lexicon(
                rec.map_name, all_places, overlay_path if overlay_path.exists() else None
            )

        # 3. Serializing
        state.stage = "serializing"
        save_job_state(cfg.data_root, state)

        card_checksum = card.checksum if card is not None else "none"
        scripts = serialize_match(lake, mapper, lex, card_checksum)

        for s in scripts:
            s_path = cfg.data_root / "scripts" / rec.match_id / f"round_{s.round_num}.json"
            s_path.parent.mkdir(parents=True, exist_ok=True)
            s_path.write_text(s.to_json(), encoding="utf-8")

        # 4. Mining
        state.stage = "mining"
        save_job_state(cfg.data_root, state)

        team_keys = {
            k
            for s in scripts
            for k in (s.t_team_key, s.ct_team_key)
            if k and str(k) not in ("T", "CT", "None", "")
        }
        # Cluster lineups into team identities (stand-in lineups merge) now that
        # this match's rosters are in the lake.
        clusters = build_team_clusters(cfg.data_root)
        write_team_clusters(cfg.data_root, clusters)
        if not team_keys:
            team_keys = {s.t_team_key for s in scripts} | {s.ct_team_key for s in scripts}

        mined: set[str] = set()
        for tk in sorted(team_keys):
            cluster = clusters.get(tk)
            team_id = cluster.team_id if cluster else tk
            if team_id in mined:
                continue  # both lineups of one team can appear in team_keys
            mined.add(team_id)
            keys = cluster.all_keys() if cluster else {tk}
            team_scripts = _scripts_for_keys(cfg.data_root, rec.map_name, keys, team_id, scripts)
            tb = build_teambook(team_scripts, team_id)
            tb_dir = cfg.data_root / "teambooks" / team_id / rec.map_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            (tb_dir / "teambook.json").write_text(tb.model_dump_json(indent=2), encoding="utf-8")

            # Instant scout brief from the same accumulated scripts (plan Task 8).
            try:
                brief = build_scout_brief(
                    team_scripts,
                    team_id,
                    teambook=tb,
                    utility_book=build_utility_book(team_scripts, team_id),
                    gap_report=build_gap_report(team_scripts, team_id),
                    econ_policy=build_econ_policy(team_scripts, team_id),
                )
                (tb_dir / "scout_brief.json").write_text(
                    brief.model_dump_json(indent=2), encoding="utf-8"
                )
            except Exception as exc:  # noqa: BLE001 - a brief failure must not fail ingest
                logger.warning("Scout brief generation failed for %s: %s", tk, exc)

        # 5. Done
        state.stage = "done"
        save_job_state(cfg.data_root, state)

    except Exception as exc:
        logger.exception("Ingest job %s failed", job_id)
        state.stage = "error"
        state.detail = str(exc)
        save_job_state(cfg.data_root, state)
