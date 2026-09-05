"""Shipped callout anchors: calibrated once from curated demos, never user data.

The anchor math (median of occupancy mass, snapped to a real standing
position, on the zone's dominant level) previously lived in the web layer and
recomputed from whatever the user had ingested - so labels moved when demos
came and went. It now runs once, offline, over operator-provided calibration
demos (``counterstrat.mapcard.calibrate``) and the results ship with the app
as ``anchors/<map>.json``. User uploads never move a label again.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from counterstrat.radar.coords import game_to_norm, is_lower_level, level_expr

logger = logging.getLogger(__name__)

SHIPPED_ANCHORS_DIR = Path(__file__).resolve().parent / "anchors"

Anchor = tuple[float, float, str]  # (u, v, level)
AnchorPoint = tuple[float, float, str, float]  # (u, v, level, world z)


def compute_tick_anchors(df: pl.DataFrame, cal) -> dict[str, AnchorPoint]:
    """(u, v, level, ground z) per place name from pooled occupancy ticks.

    A zone straddling a two-level map labels the lower radar only when clearly
    below (>= 60% of its ticks), and its anchor comes from that level's ticks
    only. Per-axis medians center on the occupancy mass; snapping to the
    closest real tick keeps the label on walkable ground even for ring- and
    L-shaped zones whose geometric center nobody ever stands on.
    """
    needed = {"X", "Y", "Z", "last_place_name", "is_alive"}
    if df.is_empty() or not needed <= set(df.columns):
        return {}
    occupied = df.filter(
        pl.col("is_alive")
        & pl.col("last_place_name").is_not_null()
        & (pl.col("last_place_name") != "")
    ).drop_nulls(["X", "Y", "Z"])
    if occupied.is_empty():
        return {}

    occupied = occupied.with_columns(level_expr(cal, "Z").alias("_lvl"))
    dominant = occupied.group_by("last_place_name").agg(
        pl.when((pl.col("_lvl") == "lower").mean() >= 0.6)
        .then(pl.lit("lower"))
        .otherwise(pl.lit("default"))
        .alias("_dom")
    )
    occupied = occupied.join(dominant, on="last_place_name").filter(
        pl.col("_lvl") == pl.col("_dom")
    )
    medians = occupied.group_by("last_place_name").agg(
        pl.col("X").median().alias("_mx"), pl.col("Y").median().alias("_my")
    )
    centroids = (
        occupied.join(medians, on="last_place_name")
        .with_columns(
            ((pl.col("X") - pl.col("_mx")) ** 2 + (pl.col("Y") - pl.col("_my")) ** 2).alias("_d2")
        )
        .group_by("last_place_name")
        .agg(pl.col("X", "Y", "Z").sort_by(["_d2", "X", "Y", "Z"]).first())
    )

    out: dict[str, AnchorPoint] = {}
    for row in centroids.iter_rows(named=True):
        u, v = game_to_norm(cal, row["X"], row["Y"])
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            level = "lower" if is_lower_level(cal, row["Z"]) else "default"
            out[str(row["last_place_name"])] = (
                round(u, 4),
                round(v, 4),
                level,
                round(float(row["Z"]), 1),
            )
    return out


def shipped_anchor_path(map_name: str, root: Path | None = None) -> Path:
    return (root or SHIPPED_ANCHORS_DIR) / f"{map_name}.json"


def shipped_anchor_maps(root: Path | None = None) -> set[str]:
    """Every map with shipped calibration - usable straight from a clone."""
    base = root or SHIPPED_ANCHORS_DIR
    if not base.exists():
        return set()
    return {p.stem for p in base.glob("*.json")}


def _read_anchor_file(map_name: str, root: Path | None) -> dict:
    path = shipped_anchor_path(map_name, root)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("anchors") or {}
    except Exception as exc:  # noqa: BLE001 - a torn file must not kill the editor
        logger.warning("Unreadable shipped anchors for %s: %s", map_name, exc)
        return {}


def load_shipped_anchors(map_name: str, root: Path | None = None) -> dict[str, Anchor]:
    """The calibrated (u, v, level) anchors shipped for a map; {} when absent."""
    return {
        str(name): (float(a["u"]), float(a["v"]), str(a.get("level") or "default"))
        for name, a in _read_anchor_file(map_name, root).items()
    }


def load_shipped_anchor_points(map_name: str, root: Path | None = None) -> dict[str, AnchorPoint]:
    """Anchors including ground Z (NaN-free; entries without z are skipped).

    Used to ground new custom-zone placements when no lake data exists.
    """
    out: dict[str, AnchorPoint] = {}
    for name, a in _read_anchor_file(map_name, root).items():
        if a.get("z") is None:
            continue
        out[str(name)] = (
            float(a["u"]),
            float(a["v"]),
            str(a.get("level") or "default"),
            float(a["z"]),
        )
    return out


def save_shipped_anchors(
    map_name: str,
    anchors: dict[str, AnchorPoint] | dict[str, Anchor],
    *,
    generated_from: list[str],
    root: Path | None = None,
) -> Path:
    path = shipped_anchor_path(map_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _entry(a) -> dict:
        entry = {"u": a[0], "v": a[1], "level": a[2]}
        if len(a) > 3:
            entry["z"] = a[3]
        return entry

    payload = {
        "map_name": map_name,
        "computed_at": datetime.now(UTC).isoformat(),
        "demos": len(generated_from),
        "generated_from": sorted(generated_from),
        "anchors": {name: _entry(a) for name, a in sorted(anchors.items())},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
