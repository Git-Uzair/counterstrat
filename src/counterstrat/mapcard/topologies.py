"""Shipped zone topologies: move-time graphs snapshotted from compiled cards.

Site-hold complexes (``counterstrat.mining.gaps``) need the card's
zone-to-zone move times, but compiled cards live under the personal data root
(gitignored, and compiling one needs a CS2 install). The snapshot ships with
the app as ``anchors/topologies/<map>.json`` so a fresh clone mines correct
hold complexes out of the box; a live compiled card always wins over the
snapshot.

Refresh the snapshots after recalibrating cards:

    uv run python -c "from pathlib import Path; \\
        from counterstrat.mapcard.topologies import export_shipped_topologies; \\
        print(export_shipped_topologies(Path('data')))"
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

from counterstrat.mapcard.anchors import SHIPPED_ANCHORS_DIR

logger = logging.getLogger(__name__)

# Own subdirectory: shipped_anchor_maps() inventories anchors/*.json by stem,
# so topology files must not share that namespace.
SHIPPED_TOPOLOGIES_DIR = SHIPPED_ANCHORS_DIR / "topologies"

Topology = dict[str, dict[str, float]]


def shipped_topology_path(map_name: str, root: Path | None = None) -> Path:
    return (root or SHIPPED_TOPOLOGIES_DIR) / f"{map_name}.json"


def shipped_topology_maps(root: Path | None = None) -> set[str]:
    """Every map with a shipped topology - usable straight from a clone."""
    base = root or SHIPPED_TOPOLOGIES_DIR
    if not base.exists():
        return set()
    return {p.stem for p in base.glob("*.json")}


def load_shipped_topology(map_name: str, root: Path | None = None) -> Topology:
    """The snapshotted move-time graph for a map; {} when absent or torn."""
    path = shipped_topology_path(map_name, root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        topo = raw.get("topology") or {}
        return {
            str(u): {str(v): float(w) for v, w in (nbrs or {}).items()} for u, nbrs in topo.items()
        }
    except Exception as exc:  # noqa: BLE001 - a torn file must not kill mining
        logger.warning("Unreadable shipped topology for %s: %s", map_name, exc)
        return {}


def save_shipped_topology(
    map_name: str,
    topology: Topology,
    *,
    card_checksum: str = "",
    game_version: str = "",
    root: Path | None = None,
) -> Path:
    path = shipped_topology_path(map_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map_name": map_name,
        "computed_at": datetime.now(UTC).isoformat(),
        "card_checksum": card_checksum,
        "game_version": game_version,
        "topology": {u: dict(sorted(nbrs.items())) for u, nbrs in sorted(topology.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def export_shipped_topologies(data_root: Path, root: Path | None = None) -> list[Path]:
    """Snapshot every compiled card's topology into the shipped set."""
    written: list[Path] = []
    for card_path in sorted(data_root.glob("mapcards/*/card.yaml")):
        try:
            card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - skip torn cards, keep exporting
            logger.warning("Skipping unreadable card %s: %s", card_path, exc)
            continue
        map_name = str(card.get("map") or card_path.parent.name)
        topology = card.get("topology") or {}
        if not topology:
            continue
        written.append(
            save_shipped_topology(
                map_name,
                topology,
                card_checksum=str(card.get("checksum") or ""),
                game_version=str(card.get("game_version") or ""),
                root=root,
            )
        )
    return written
