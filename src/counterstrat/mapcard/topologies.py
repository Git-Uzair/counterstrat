"""Shipped zone topologies: engine-vocabulary move-time graphs, never customs.

Site-hold complexes (``counterstrat.mining.gaps``) need zone-to-zone move
times, but compiled cards live under the personal data root (gitignored) and
speak the operator's custom-callout vocabulary. The export ships only the
callouts that come with the map, as ``anchors/topologies/<map>.json``, drawing
per map as the edge-level UNION of every engine-vocabulary source, because a
missing edge shrinks a hold complex and produces false "site vacant" reads,
while a stale measured edge merely widens one. Where sources disagree on an
edge, the fresher wins: compiled card < corpus lake < calibration demos.

- Calibration movement caches (``mapcard.calibrate --moves-only``): curated
  operator demos, engine vocabulary natively, provenance recorded.
- The corpus lake, rebuilt with custom zones undone (``place_default`` keeps
  the game's own name under every re-zoned tick).
- The compiled card, with the operator's custom zones renamed to their engine
  parents - the modal engine place inside each rect, evidenced by the
  calibration occupancy caches - or dropped when unresolvable.

At runtime the shipped engine skeleton merges UNDER the live compiled card
(:func:`merge_topologies`): a clone mines correct complexes before its first
zone rebuild, and once a user bakes their own callouts, their card's
custom-zone edges join the graph and their vocabulary counts toward holds.

Refresh the snapshots after ingesting new demos or recalibrating:

    uv run python -c "from pathlib import Path; \\
        from counterstrat.mapcard.topologies import export_shipped_topologies; \\
        print(export_shipped_topologies(Path('data')))"
"""

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from counterstrat.customzones import RECT_Z_BAND, CustomZone, load_custom_zones
from counterstrat.mapcard.anchors import SHIPPED_ANCHORS_DIR
from counterstrat.mapcard.transitions import zone_graph

logger = logging.getLogger(__name__)

# Own subdirectory: shipped_anchor_maps() inventories anchors/*.json by stem,
# so topology files must not share that namespace.
SHIPPED_TOPOLOGIES_DIR = SHIPPED_ANCHORS_DIR / "topologies"

# Same edge rule as the card compiler (mapcard/compile.py step 3).
MIN_EDGE_N = 5

_TICK_COLUMNS = (
    "match_id",
    "steamid",
    "round_num",
    "tick",
    "clock_s",
    "is_alive",
    "last_place_name",
    "place_default",
)

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


def merge_topologies(base: Topology, live: Topology) -> Topology:
    """Edge-level union; the live graph wins where both know an edge.

    The shipped engine skeleton keeps complexes intact on sparse corpora while
    a rebuilt card contributes the user's custom zones and fresher times.
    """
    merged: Topology = {u: dict(nbrs) for u, nbrs in base.items()}
    for u, nbrs in live.items():
        merged.setdefault(u, {}).update(nbrs)
    return merged


def save_shipped_topology(
    map_name: str,
    topology: Topology,
    *,
    game_version: str = "",
    generated_from: list[str] | None = None,
    root: Path | None = None,
) -> Path:
    path = shipped_topology_path(map_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map_name": map_name,
        "computed_at": datetime.now(UTC).isoformat(),
        "game_version": game_version,
        "generated_from": sorted(generated_from or []),
        "topology": {u: dict(sorted(nbrs.items())) for u, nbrs in sorted(topology.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _engine_ticks(df: pl.DataFrame) -> pl.DataFrame:
    """Ticks in the game's own vocabulary: custom-zone renames undone."""
    if "place_default" in df.columns:
        df = df.with_columns(pl.col("place_default").alias("last_place_name"))
    return df.drop("place_default", strict=False)


def topology_from_ticks(ticks: pl.DataFrame) -> Topology:
    """The card compiler's edge rule (n >= MIN_EDGE_N, median transit, 1dp)."""
    topo: Topology = {}
    for (src, dst), edge in sorted(zone_graph(ticks).edges.items()):
        if edge.n >= MIN_EDGE_N:
            topo.setdefault(src, {})[dst] = round(float(edge.median_transit_s), 1)
    return topo


def _lake_topology(
    data_root: Path, mids: list[str], manifest: dict
) -> tuple[Topology, list[str], str]:
    """Engine topology rebuilt from the lake (custom zones undone)."""
    frames: list[pl.DataFrame] = []
    versions: list[str] = []
    used: list[str] = []
    for mid in sorted(mids):
        path = data_root / "lake" / mid / "ticks.parquet"
        if not path.exists():
            continue
        have = pl.read_parquet_schema(path)
        df = pl.read_parquet(path, columns=[c for c in _TICK_COLUMNS if c in have])
        if "match_id" not in df.columns:
            df = df.with_columns(pl.lit(mid).alias("match_id"))
        frames.append(_engine_ticks(df))
        used.append(mid)
        versions.append(manifest[mid].patch_version)
    if not frames:
        return {}, [], ""
    topology = topology_from_ticks(pl.concat(frames, how="vertical_relaxed"))
    return topology, used, max(versions) if versions else ""


def _calibration_index(data_root: Path) -> dict:
    try:
        path = data_root / "calibration" / ".cache" / "index.json"
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no calibration demos on this machine
        return {}


def _calibration_topology(data_root: Path, map_name: str) -> tuple[Topology, list[str], str]:
    """Engine topology measured from the curated calibration demos' movement."""
    cache = data_root / "calibration" / ".cache"
    frames: list[pl.DataFrame] = []
    files: list[str] = []
    patches: list[str] = []
    for sha, meta in sorted(_calibration_index(data_root).items()):
        path = cache / f"{sha}.moves.parquet"
        if meta.get("map") != map_name or not path.exists():
            continue
        try:
            df = pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unreadable movement cache %s: %s", path, exc)
            continue
        frames.append(df.with_columns(pl.lit(sha).alias("match_id")))
        files.append(str(meta.get("file") or sha))
        if meta.get("patch"):
            patches.append(str(meta["patch"]))
    if not frames:
        return {}, [], ""
    topology = topology_from_ticks(pl.concat(frames, how="vertical_relaxed"))
    return topology, files, max(patches) if patches else ""


def _calibration_ticks(data_root: Path, map_name: str) -> pl.DataFrame:
    """Pooled engine-vocabulary occupancy ticks from the calibration cache."""
    cache = data_root / "calibration" / ".cache"
    frames: list[pl.DataFrame] = []
    for sha, meta in sorted(_calibration_index(data_root).items()):
        path = cache / f"{sha}.parquet"
        if meta.get("map") != map_name or not path.exists():
            continue
        try:
            frames.append(pl.read_parquet(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unreadable calibration cache %s: %s", path, exc)
    return pl.concat(frames) if frames else pl.DataFrame()


def _engine_renames(zones: list[CustomZone], cal_ticks: pl.DataFrame) -> dict[str, str]:
    """Custom zone -> the engine place its rect was carved from (modal evidence)."""
    if cal_ticks.is_empty() or not {"X", "Y", "Z", "last_place_name"} <= set(cal_ticks.columns):
        return {}
    out: dict[str, str] = {}
    for z in zones:
        inside = cal_ticks.filter(
            pl.col("X").is_between(z.x - z.half_x, z.x + z.half_x)
            & pl.col("Y").is_between(z.y - z.half_y, z.y + z.half_y)
            & ((pl.col("Z") - z.z).abs() <= RECT_Z_BAND)
        )
        if inside.is_empty():
            continue
        modal = (
            inside.group_by("last_place_name")
            .len()
            .sort(["len", "last_place_name"], descending=[True, False])
        )
        out[z.name] = str(modal["last_place_name"][0])
    return out


def _rename_or_drop(topology: Topology, renames: dict[str, str], drop: set[str]) -> Topology:
    """Undo custom-zone renames at graph level; unresolvable nodes leave the graph."""
    out: Topology = {}
    for u, nbrs in topology.items():
        if u in drop:
            continue
        ru = renames.get(u, u)
        for v, w in nbrs.items():
            if v in drop:
                continue
            rv = renames.get(v, v)
            if ru == rv:
                continue
            cur = out.setdefault(ru, {}).get(rv)
            out[ru][rv] = w if cur is None else min(cur, w)
    return out


def _card_topology(data_root: Path, map_name: str) -> tuple[Topology, list[str], str]:
    """Engine topology from the compiled card, operator customs renamed away."""
    card_path = data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        return {}, [], ""
    try:
        card = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - skip torn cards
        logger.warning("Skipping unreadable card %s: %s", card_path, exc)
        return {}, [], ""
    topology: Topology = {
        str(u): {str(v): float(w) for v, w in (nbrs or {}).items()}
        for u, nbrs in (card.get("topology") or {}).items()
    }
    custom = load_custom_zones(data_root, map_name)
    if custom:
        renames = _engine_renames(custom, _calibration_ticks(data_root, map_name))
        drop = {z.name for z in custom} - set(renames)
        if drop:
            logger.warning(
                "%s: dropping custom zones with no engine evidence: %s",
                map_name,
                sorted(drop),
            )
        topology = _rename_or_drop(topology, renames, drop)
    provenance = f"card:{card.get('checksum') or ''}"
    return topology, [provenance], str(card.get("game_version") or "")


def export_shipped_topologies(data_root: Path, root: Path | None = None) -> list[Path]:
    """Snapshot every map's ENGINE-vocabulary topology.

    Edge-level union of card, lake, and calibration measurements (in that
    order, so the fresher source wins where they disagree). Missing edges
    cause false "site vacant" reads; measured-but-stale ones only widen a
    complex - so every measured engine edge ships.
    """
    from counterstrat.corpus import load_manifest

    manifest = load_manifest(data_root / "corpus.jsonl")
    by_map: dict[str, list[str]] = defaultdict(list)
    for mid, rec in manifest.items():
        by_map[rec.map_name].append(mid)
    card_maps = {p.parent.name for p in data_root.glob("mapcards/*/card.yaml")}
    cal_maps = {str(m.get("map")) for m in _calibration_index(data_root).values() if m.get("map")}

    written: list[Path] = []
    for map_name in sorted(cal_maps | set(by_map) | card_maps):
        cal_topo, cal_from, cal_ver = _calibration_topology(data_root, map_name)
        lake_topo, lake_from, lake_ver = (
            _lake_topology(data_root, by_map[map_name], manifest)
            if by_map.get(map_name)
            else ({}, [], "")
        )
        card_topo, card_from, card_ver = _card_topology(data_root, map_name)
        topology = merge_topologies(merge_topologies(card_topo, lake_topo), cal_topo)
        if not topology:
            continue
        generated_from = cal_from + lake_from + (card_from if card_topo else [])
        versions = [v for v in (cal_ver, lake_ver, card_ver) if v]
        written.append(
            save_shipped_topology(
                map_name,
                topology,
                game_version=max(versions) if versions else "",
                generated_from=generated_from,
                root=root,
            )
        )
    return written
