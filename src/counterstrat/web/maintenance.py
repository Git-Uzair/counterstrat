"""Corpus maintenance: demo deletion and full derived-artifact rebuild."""

import logging
import shutil
from pathlib import Path
from typing import Any

from counterstrat.config import AppConfig
from counterstrat.corpus import DemoRecord, load_manifest
from counterstrat.mining.brief import build_scout_brief
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.tendencies import build_teambook
from counterstrat.mining.utility_book import build_utility_book
from counterstrat.roundscript.models import RoundScript
from counterstrat.teams import build_team_clusters, write_team_clusters

logger = logging.getLogger(__name__)


def mine_team_artifacts(
    cfg: AppConfig, team_id: str, map_name: str, team_scripts: list[RoundScript]
) -> None:
    """Write the teambook + scout brief for one (team, map) from its scripts."""
    tb = build_teambook(team_scripts, team_id)
    tb_dir = cfg.data_root / "teambooks" / team_id / map_name
    tb_dir.mkdir(parents=True, exist_ok=True)
    (tb_dir / "teambook.json").write_text(tb.model_dump_json(indent=2), encoding="utf-8")
    try:
        brief = build_scout_brief(
            team_scripts,
            team_id,
            teambook=tb,
            utility_book=build_utility_book(team_scripts, team_id),
            gap_report=build_gap_report(team_scripts, team_id),
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


def delete_demo(cfg: AppConfig, match_id: str) -> DemoRecord:
    """Remove one demo and everything derived from it; rebuild the rest.

    The .dem file itself is only deleted when it lives inside the app's data
    root (i.e. it was uploaded); externally registered files are left alone.
    """
    manifest_path = cfg.data_root / "corpus.jsonl"
    manifest = load_manifest(manifest_path)
    if match_id not in manifest:
        raise KeyError(match_id)
    rec = manifest.pop(match_id)

    lines = [manifest[mid].model_dump_json() for mid in manifest]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for derived in (cfg.data_root / "lake" / match_id, cfg.data_root / "scripts" / match_id):
        shutil.rmtree(derived, ignore_errors=True)

    try:
        demo_path = Path(rec.path).resolve()
        if demo_path.is_file() and demo_path.is_relative_to(cfg.data_root.resolve()):
            demo_path.unlink()
    except OSError as exc:
        logger.warning("Could not delete demo file %s: %s", rec.path, exc)

    rebuild_artifacts(cfg)
    return rec
