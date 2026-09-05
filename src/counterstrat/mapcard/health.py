"""Map data health checks: make silent gaps loud.

de_inferno shipped for weeks with empty sites/rotates/timings and
teleport-polluted graph edges; nothing surfaced it (2026-09-04 plan,
Execution record). Every gap class found there has a mechanical signature,
so this module checks them all deterministically - no LLM, no network.

Surfaces:
- rebuild jobs append the warning count to their status detail,
- GET /api/maps/{map_name}/health returns the full report,
- CLI: uv run python -m counterstrat.mapcard.health [map_name]
"""

import json
import math
from pathlib import Path
from typing import Any

from counterstrat.config import AppConfig
from counterstrat.mapcard.compile import MapCard

# An edge faster than this while its zones' anchors sit further apart than
# IMPLAUSIBLE_UV is teleport-class residue, not movement.
IMPLAUSIBLE_S = 2.0
IMPLAUSIBLE_UV = 0.25
# Two anchors closer than this are ambiguous grounding targets for the LLM.
OVERLAP_UV = 0.02


def _f(level: str, code: str, msg: str) -> dict[str, str]:
    return {"level": level, "code": code, "msg": msg}


def check_card_health(card: MapCard, anchors: dict[str, tuple]) -> list[dict[str, str]]:
    """Deterministic findings over one compiled card + its radar anchors."""
    out: list[dict[str, str]] = []
    zones = sorted(card.zones)
    sites = (card.objectives or {}).get("sites") or []

    if len(sites) < 2:
        named = [z for z in zones if z.lower().startswith("bombsite")]
        hint = (
            "the card predates site-name inference: re-ingest a demo or rebuild to recompile"
            if len(named) >= 2
            else "tag zones 'site' in an overlay or name them Bombsite*"
        )
        out.append(
            _f(
                "warn",
                "no_sites",
                f"{len(sites)} bombsite zone(s) identified: rotates, site timings and "
                f"objectives stay empty ({hint})",
            )
        )
    elif not card.rotates:
        out.append(
            _f(
                "warn",
                "no_rotates",
                "two sites but no rotate routes: the mined graph does not connect them "
                "(too few demos, or zones never walked between)",
            )
        )
    for side in ("CT", "T"):
        if not (card.timings or {}).get(side):
            out.append(
                _f(
                    "warn",
                    "no_timings",
                    f"no earliest-reach timings for {side}: LLM "
                    "answers about that side's timing windows will be guesses",
                )
            )

    missing_anchor = [z for z in zones if z not in anchors]
    if missing_anchor:
        out.append(
            _f(
                "warn",
                "missing_anchor",
                f"{len(missing_anchor)} zone(s) without a radar anchor "
                f"({', '.join(missing_anchor[:6])}...): invisible to coordinate "
                "grounding and bearings",
            )
        )

    edged = set(card.topology or {})
    for nbrs in (card.topology or {}).values():
        edged |= set(nbrs)
    isolated = [z for z in zones if z not in edged]
    if isolated:
        out.append(
            _f(
                "info",
                "isolated_zones",
                f"{len(isolated)} zone(s) with no graph edges ({', '.join(isolated[:8])}): "
                "fine for rarely-visited spots, but they cannot appear in routes",
            )
        )

    overlaps = []
    zs = [z for z in zones if z in anchors]
    for i, a in enumerate(zs):
        for b in zs[i + 1 :]:
            if math.dist(anchors[a][:2], anchors[b][:2]) < OVERLAP_UV:
                overlaps.append(f"{a}~{b}")
    if overlaps:
        out.append(
            _f(
                "info",
                "anchor_overlap",
                f"anchor pairs nearly coincide ({', '.join(overlaps[:5])}): ambiguous "
                "targets for coordinate grounding",
            )
        )

    implausible = []
    for u, nbrs in (card.topology or {}).items():
        for v, sec in nbrs.items():
            # spawn zones are large areas: centroid distance systematically
            # overstates the actual edge-to-edge path, so they are exempt
            if "spawn" in u.lower() or "spawn" in v.lower():
                continue
            if (
                u in anchors
                and v in anchors
                and float(sec) < IMPLAUSIBLE_S
                and math.dist(anchors[u][:2], anchors[v][:2]) > IMPLAUSIBLE_UV
            ):
                implausible.append(f"{u}->{v} {float(sec):.1f}s")
    if implausible:
        out.append(
            _f(
                "warn",
                "implausible_edge",
                f"edges cross large distances in under {IMPLAUSIBLE_S:.0f}s "
                f"({', '.join(implausible[:5])}): teleport-class artifacts - "
                "re-ingest or rebuild with the fixed graph miner",
            )
        )

    no_quadrant = [z for z in zones if not (card.zones[z] or {}).get("quadrant")]
    if no_quadrant:
        out.append(
            _f(
                "info",
                "no_quadrant",
                f"{len(no_quadrant)} zone(s) without quadrant/elevation "
                f"({', '.join(no_quadrant[:6])}): too few ticks observed there",
            )
        )

    if not card.sightlines:
        out.append(
            _f(
                "info",
                "no_sightlines",
                "no sightline data yet: built from clean kills at ingest/rebuild, "
                "grows with the corpus",
            )
        )
    return out


def check_scripts_health(cfg: AppConfig, map_name: str) -> list[dict[str, str]]:
    """Round scripts on disk: pre-v2 artifacts without movement tracks."""
    from counterstrat.corpus import load_manifest

    manifest_path = cfg.data_root / "corpus.jsonl"
    if not manifest_path.exists():
        return []
    manifest = load_manifest(manifest_path)
    total = 0
    without = 0
    for mid, rec in sorted(manifest.items()):
        if rec.map_name != map_name:
            continue
        for p in sorted((cfg.data_root / "scripts" / mid).glob("round_*.json")):
            total += 1
            if not json.loads(p.read_text(encoding="utf-8")).get("tracks"):
                without += 1
    if without:
        return [
            _f(
                "warn",
                "scripts_missing_tracks",
                f"{without}/{total} round scripts have no movement tracks "
                "(serialized before v2): run a rebuild for this map",
            )
        ]
    return []


def check_map_health(cfg: AppConfig, map_name: str) -> dict[str, Any]:
    """Full report for one map; card-less maps report that as the finding."""
    import yaml

    from counterstrat.web.routes import map_zone_anchors

    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    card = None
    if card_path.exists():
        try:
            data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                card = MapCard(**data)
        except Exception:  # noqa: BLE001 - empty/torn card counts as no card
            card = None
    if card is None:
        return {
            "map": map_name,
            "findings": [_f("warn", "no_card", "no compiled card: ingest a demo first")],
        }
    anchors = map_zone_anchors(cfg, map_name)
    findings = check_card_health(card, anchors) + check_scripts_health(cfg, map_name)
    return {
        "map": map_name,
        "zones": len(card.zones),
        "warnings": sum(1 for f in findings if f["level"] == "warn"),
        "findings": findings,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map data health report")
    parser.add_argument("map_name", nargs="?", default=None, help="default: every compiled map")
    args = parser.parse_args()
    cfg = AppConfig.load()
    maps = (
        [args.map_name]
        if args.map_name
        else sorted(
            p.name for p in (cfg.data_root / "mapcards").iterdir() if (p / "card.yaml").exists()
        )
    )
    for name in maps:
        report = check_map_health(cfg, name)
        print(f"== {name}: {report.get('warnings', 0)} warning(s)")
        for f in report["findings"]:
            print(f"  [{f['level']}] {f['code']}: {f['msg']}")
    if not maps:
        print(f"no compiled maps under {Path(cfg.data_root) / 'mapcards'}")
