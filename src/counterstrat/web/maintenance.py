"""Corpus maintenance: demo deletion, zone rebuilds, derived-artifact rebuild."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from counterstrat.config import AppConfig
from counterstrat.corpus import DemoRecord, load_manifest
from counterstrat.customzones import load_custom_zones, rezone_ticks
from counterstrat.lake.extract import LakePaths
from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.transitions import zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places
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

_LAKE_TABLES = [
    "rounds",
    "kills",
    "damages",
    "shots",
    "grenades",
    "smokes",
    "infernos",
    "bomb",
    "item_purchase",
    "ticks",
    "rosters",
    "player_blind",
]


def _lake_paths(data_root: Path, match_id: str) -> LakePaths:
    """LakePaths for an already-extracted match directory."""
    root = data_root / "lake" / match_id
    return LakePaths(
        root=str(root), **{name: str(root / f"{name}.parquet") for name in _LAKE_TABLES}
    )


def _effective_tick_places(ticks_df: pl.DataFrame) -> set[str]:
    """Zone names the lake speaks: effective names UNION the preserved game
    names (place_default) - a custom zone that swallows a parent's every tick
    must not erase the parent from the map vocabulary."""
    out: set[str] = set()
    for col in ("last_place_name", "place_default"):
        if col in ticks_df.columns:
            out |= {str(p) for p in ticks_df[col].drop_nulls().unique().to_list() if str(p).strip()}
    return out


def _lexicon_or_fallback(map_name: str, places: list[str]):
    """Lexicon with the bundled overlay when it still fits, plain otherwise.

    An overlay can reference zones a tick-derived place list does not carry
    (chat.py uses the same fallback); the vocabulary must never fail closed.
    """
    overlay = get_default_overlay_path(map_name)
    if overlay.exists():
        try:
            return build_lexicon(map_name, places, overlay)
        except ValueError as exc:
            logger.warning("Overlay skipped for %s: %s", map_name, exc)
    return build_lexicon(map_name, places, None)


def compile_map_card(
    map_name: str,
    places: list[str],
    ticks_df: pl.DataFrame,
    rounds_df: pl.DataFrame,
    patch_version: str,
) -> MapCard | None:
    """Compile a card whose zone set is the VPK places UNION the effective
    tick vocabulary - custom zones ride into the card (and so into prompts,
    lint, and the editor) automatically."""
    all_places = sorted(set(places) | _effective_tick_places(ticks_df))
    if not all_places:
        return None
    lex = _lexicon_or_fallback(map_name, all_places)
    return compile_card(
        lexicon=lex,
        graph=zone_graph(ticks_df),
        ticks=ticks_df,
        rounds=rounds_df,
        map_name=map_name,
        patch_version=patch_version,
    )


def _cached_vpk_places(cfg: AppConfig, map_name: str) -> list[str]:
    """VPK place names from the tmp_assets vents cache; [] when not extracted.

    No VRF run here - the cache exists for any map that compiled a card.
    """
    vents = (
        cfg.data_root
        / "tmp_assets"
        / map_name
        / "maps"
        / map_name
        / "entities"
        / "default_ents.vents"
    )
    if not vents.exists():
        return []
    try:
        return unique_places(parse_places(vents))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable vents for %s: %s", map_name, exc)
        return []


def rebuild_map_zones(cfg: AppConfig, map_name: str) -> dict[str, Any]:
    """Bake the map's custom zones into the lake and rebuild all derivatives.

    Order matters: ticks first (the vocabulary), then the card
    (zones/topology), then scripts (movements/beats/utility via a mapper
    trained on re-zoned ticks), then miners, then stale LLM caches.
    """
    zones = load_custom_zones(cfg.data_root, map_name)
    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    matches = [mid for mid, rec in sorted(manifest.items()) if rec.map_name == map_name]

    # 1. Re-zone every match's ticks, atomically.
    for mid in matches:
        ticks_path = cfg.data_root / "lake" / mid / "ticks.parquet"
        if not ticks_path.exists():
            continue
        df = rezone_ticks(pl.read_parquet(ticks_path), zones)
        tmp = ticks_path.with_suffix(f".tmp_{os.getpid()}")
        df.write_parquet(tmp)
        os.replace(tmp, ticks_path)

    # 2. Recompile the card from the first match (mirrors ingest).
    card: MapCard | None = None
    if matches:
        first_lake = _lake_paths(cfg.data_root, matches[0])
        if Path(first_lake.ticks).exists():
            ticks_df = pl.read_parquet(first_lake.ticks)
            rounds_df = (
                pl.read_parquet(first_lake.rounds)
                if Path(first_lake.rounds).exists()
                else pl.DataFrame()
            )
            card = compile_map_card(
                map_name,
                _cached_vpk_places(cfg, map_name),
                ticks_df,
                rounds_df,
                manifest[matches[0]].patch_version,
            )
            if card is not None:
                card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
                card_path.parent.mkdir(parents=True, exist_ok=True)
                card_path.write_text(card.to_yaml(), encoding="utf-8")

    # 3. Sightlines before scripts so stint gaze sees the matrix. Calibration
    # evidence first, re-derived under the JUST-SAVED custom vocabulary (a
    # rect carved from `Middle` gets its own rows); user lake kills remain
    # the fallback for uncalibrated maps.
    sightlines: list[dict] = []
    try:
        from counterstrat.mapcard.anchors import shipped_sightlines
        from counterstrat.mapcard.visibility import refresh_card_sightlines

        sightlines = shipped_sightlines(cfg.data_root, map_name)
        if not sightlines:
            sightlines = refresh_card_sightlines(cfg.data_root, map_name)
    except Exception as exc:  # noqa: BLE001 - sightlines must never fail a rebuild
        logger.warning("Sightline refresh failed for %s: %s", map_name, exc)

    # 4. Re-serialize scripts per match with a mapper trained on the new
    # vocabulary.
    for mid in matches:
        lake = _lake_paths(cfg.data_root, mid)
        if not Path(lake.ticks).exists():
            continue
        ticks_df = pl.read_parquet(lake.ticks)
        try:
            mapper = ZoneMapper.fit(ticks_df)
        except ValueError:
            continue  # no valid tick rows - keep old scripts
        if card is not None:
            places = list(card.zones.keys())
        else:
            places = sorted(_effective_tick_places(ticks_df)) or ["Default"]
        lex = _lexicon_or_fallback(map_name, places)
        scripts = serialize_match(
            lake, mapper, lex, card.checksum if card else "none", sightlines=sightlines or None
        )
        if scripts:
            scripts_dir = cfg.data_root / "scripts" / mid
            shutil.rmtree(scripts_dir, ignore_errors=True)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            for s in scripts:
                (scripts_dir / f"round_{s.round_num}.json").write_text(
                    s.to_json(), encoding="utf-8"
                )

    # 5. Re-mine every teambook (recluster + prune, existing machinery).
    stats = rebuild_artifacts(cfg)

    # 6. Stale LLM caches for THIS map speak the old vocabulary - delete.
    tb_root = cfg.data_root / "teambooks"
    if tb_root.exists():
        for team_dir in tb_root.iterdir():
            if not team_dir.is_dir():
                continue
            for stale in ("dossier.md", "insights.json"):
                p = team_dir / map_name / stale
                if p.exists():
                    p.unlink()

    return {"matches": len(matches), "zones": len(zones), **stats}


def run_zone_rebuild(job_id: str, map_name: str, cfg: AppConfig) -> None:
    """Background-job wrapper mirroring run_ingest's state machine."""
    from counterstrat.web.ingest import JobState, load_job_state, save_job_state

    state = load_job_state(cfg.data_root, job_id) or JobState(job_id=job_id, stage="queued")
    state.map_name = map_name
    try:
        state.stage = "serializing"
        save_job_state(cfg.data_root, state)
        result = rebuild_map_zones(cfg, map_name)
        state.stage = "mining"
        save_job_state(cfg.data_root, state)
        state.stage = "done"
        state.detail = f"rebuilt {result['matches']} match(es), {result['zones']} custom zone(s)"
        try:
            from counterstrat.mapcard.health import check_map_health

            report = check_map_health(cfg, map_name)
            warns = int(report.get("warnings", 0))
            if warns:
                state.detail += (
                    f"; map health: {warns} warning(s) - see /api/maps/{map_name}/health"
                )
                for f in report["findings"]:
                    if f["level"] == "warn":
                        logger.warning("Map health %s: %s - %s", map_name, f["code"], f["msg"])
        except Exception as exc:  # noqa: BLE001 - health must never fail a rebuild
            logger.warning("Map health check failed for %s: %s", map_name, exc)
        save_job_state(cfg.data_root, state)
    except Exception as exc:
        logger.exception("Zone rebuild job %s failed", job_id)
        state.stage = "error"
        state.detail = str(exc)
        try:
            save_job_state(cfg.data_root, state)
        except Exception:
            logger.exception("Could not persist error state for job %s", job_id)


def mine_team_artifacts(
    cfg: AppConfig, team_id: str, map_name: str, team_scripts: list[RoundScript]
) -> None:
    """Write the teambook + scout brief for one (team, map) from its scripts."""
    tb = build_teambook(team_scripts, team_id)
    tb_dir = cfg.data_root / "teambooks" / team_id / map_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    (tb_dir / "teambook.json").write_text(tb.model_dump_json(indent=2), encoding="utf-8")
    try:
        from counterstrat.mapcard.topologies import load_shipped_topology

        brief = build_scout_brief(
            team_scripts,
            team_id,
            teambook=tb,
            utility_book=build_utility_book(team_scripts, team_id),
            gap_report=build_gap_report(
                team_scripts, team_id, topology=load_shipped_topology(map_name)
            ),
            econ_policy=build_econ_policy(team_scripts, team_id),
        )
        (tb_dir / "scout_brief.json").write_text(brief.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - a brief failure must not fail mining
        logger.warning("Scout brief generation failed for %s/%s: %s", team_id, map_name, exc)


def rebuild_artifacts(cfg: AppConfig) -> dict[str, Any]:
    """Recluster teams and re-mine every (team, map) artifact; prune dead dirs.

    Used after corpus mutations (demo deletion). Insight caches inside pruned
    directories die with them; surviving stale caches 404 via generated_from.
    """
    from counterstrat.web.ingest import _scripts_for_keys

    clusters = build_team_clusters(cfg.data_root)
    write_team_clusters(cfg.data_root, clusters)
    unique = {c.team_id: c for c in clusters.values()}

    live: set[tuple[str, str]] = set()
    for team_id in sorted(unique):
        cluster = unique[team_id]
        for map_name in sorted({tm.map_name for tm in cluster.matches.values()}):
            team_scripts = _scripts_for_keys(
                cfg.data_root, map_name, cluster.all_keys(), team_id, []
            )
            if not team_scripts:
                continue
            mine_team_artifacts(cfg, team_id, map_name, team_scripts)
            live.add((team_id, map_name))

    pruned = 0
    tb_root = cfg.data_root / "teambooks"
    if tb_root.exists():
        for tb_file in list(tb_root.glob("*/*/teambook.json")):
            key = (tb_file.parent.parent.name, tb_file.parent.name)
            if key not in live:
                shutil.rmtree(tb_file.parent, ignore_errors=True)
                pruned += 1
        for team_dir in list(tb_root.iterdir()):
            if team_dir.is_dir() and not any(team_dir.iterdir()):
                team_dir.rmdir()

    return {"teams": len(unique), "rebuilt": len(live), "pruned": pruned}


def delete_demos(cfg: AppConfig, match_ids: list[str]) -> list[DemoRecord]:
    """Remove N demos and everything derived from them; rebuild the rest ONCE.

    All ids are validated first: an unknown id rejects the whole batch
    (KeyError naming the missing ids) with nothing deleted. The .dem files
    themselves are only deleted when they live inside the app's data root
    (i.e. they were uploaded); externally registered files are left alone.
    """
    manifest_path = cfg.data_root / "corpus.jsonl"
    manifest = load_manifest(manifest_path)
    ordered = list(dict.fromkeys(match_ids))
    missing = [mid for mid in ordered if mid not in manifest]
    if missing:
        raise KeyError(", ".join(missing))

    recs = []
    for match_id in ordered:
        rec = manifest.pop(match_id)
        recs.append(rec)
        for derived in (cfg.data_root / "lake" / match_id, cfg.data_root / "scripts" / match_id):
            shutil.rmtree(derived, ignore_errors=True)
        try:
            demo_path = Path(rec.path).resolve()
            if demo_path.is_file() and demo_path.is_relative_to(cfg.data_root.resolve()):
                demo_path.unlink()
        except OSError as exc:
            logger.warning("Could not delete demo file %s: %s", rec.path, exc)

    lines = [manifest[mid].model_dump_json() for mid in manifest]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    _purge_scopes_touching(cfg, set(ordered))
    rebuild_artifacts(cfg)
    return recs


def _purge_scopes_touching(cfg: AppConfig, deleted: set[str]) -> None:
    """Invalidate every analysis scope that used a deleted match: its chat
    transcript and its scoped First Read cache are removed together."""
    chats_dir = cfg.data_root / "chats"
    if chats_dir.exists():
        for transcript in chats_dir.glob("*.jsonl"):
            try:
                first = transcript.read_text(encoding="utf-8").splitlines()[0]
                meta = json.loads(first)
                ids = set(
                    meta.get("match_ids") or ([meta["match_id"]] if meta.get("match_id") else [])
                )
            except Exception:  # noqa: BLE001 - unreadable meta: purge conservatively
                ids = deleted
            if ids & deleted:
                transcript.unlink(missing_ok=True)

    tb_root = cfg.data_root / "teambooks"
    if tb_root.exists():
        for cache in tb_root.glob("*/*/insights/*.json"):
            try:
                ids = set(json.loads(cache.read_text(encoding="utf-8")).get("match_ids") or [])
            except Exception:  # noqa: BLE001
                ids = deleted
            if ids & deleted:
                cache.unlink(missing_ok=True)


def delete_demo(cfg: AppConfig, match_id: str) -> DemoRecord:
    """Remove one demo and everything derived from it; rebuild the rest."""
    return delete_demos(cfg, [match_id])[0]
