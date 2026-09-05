"""User-defined callout zones baked into the lake vocabulary.

A custom zone is a named axis-aligned world-space rectangle: half-extents
around a grounded center, with a fixed +-RECT_Z_BAND vertical band so
stacked levels stay independent. :func:`rezone_ticks` rewrites
``last_place_name`` for ticks inside a region while preserving the game's
own name in ``place_default`` - idempotent, reversible, and applied at the
data layer so every consumer (scripts, miners, card, anchors, SQL, prompts)
inherits the user's vocabulary without translation.
"""

import json
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from counterstrat.aliases import _ALIAS_RE

# Rects may be tiny: hide spots and one-way angles are held positions, which
# the 4 Hz ticks capture densely no matter how small the footprint is.
MIN_RECT_HALF = 5.0
MAX_RECT_HALF = 600.0
RECT_Z_BAND = 200.0  # vertical half-extent: covers ramps, excludes nuke's other level
_LEVELS = {"default", "lower"}


class CustomZone(BaseModel):
    name: str
    x: float
    y: float
    z: float
    half_x: float  # X half-extent
    half_y: float  # Y half-extent
    level: str = "default"


def _zones_overlap(a: "CustomZone", b: "CustomZone") -> bool:
    """True when two zones claim the same ground (XY footprint AND Z band)."""
    if abs(a.z - b.z) > 2 * RECT_Z_BAND:
        return False  # stacked levels (nuke) never conflict
    return abs(a.x - b.x) < a.half_x + b.half_x and abs(a.y - b.y) < a.half_y + b.half_y


def _zones_path(data_root: Path, map_name: str) -> Path:
    return data_root / "mapcards" / map_name / "zones.json"


def load_custom_zones(data_root: Path, map_name: str) -> list[CustomZone]:
    path = _zones_path(data_root, map_name)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [CustomZone(**z) for z in raw] if isinstance(raw, list) else []
    except Exception:  # noqa: BLE001 - a torn file must not kill the editor
        return []


def save_custom_zones(
    data_root: Path, map_name: str, zones: list[CustomZone], *, reserved: set[str]
) -> list[CustomZone]:
    """Validate and persist; raises ValueError with a user-facing message.

    ``reserved`` is the namespace custom names may not enter: game/card zone
    names plus alias values (one vocabulary, no double meanings).
    """
    seen: set[str] = set()
    for z in zones:
        if not _ALIAS_RE.match(z.name):
            raise ValueError(f"Zone name {z.name!r} contains invalid characters")
        if z.name in reserved:
            raise ValueError(f"Zone name {z.name!r} is reserved (existing zone or callout)")
        if z.name in seen:
            raise ValueError(f"Duplicate zone name {z.name!r}")
        seen.add(z.name)
        for half in (z.half_x, z.half_y):
            if not (MIN_RECT_HALF <= half <= MAX_RECT_HALF):
                raise ValueError(
                    f"rect half-extents must be {MIN_RECT_HALF:.0f}-{MAX_RECT_HALF:.0f} units"
                )
        if z.level not in _LEVELS:
            raise ValueError(f"Unknown level {z.level!r}")
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            if _zones_overlap(a, b):
                raise ValueError(f"Zones {a.name!r} and {b.name!r} overlap - shrink or move one")

    ordered = sorted(zones, key=lambda z: z.name)
    path = _zones_path(data_root, map_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([z.model_dump() for z in ordered], indent=2), encoding="utf-8")
    return ordered


def rezone_ticks(df: pl.DataFrame, zones: list[CustomZone]) -> pl.DataFrame:
    """Effective vocabulary: custom name inside a sphere, game name outside."""
    if "last_place_name" not in df.columns:
        return df
    if "place_default" not in df.columns:
        df = df.with_columns(pl.col("last_place_name").alias("place_default"))
    if not {"X", "Y", "Z"} <= set(df.columns) or df.is_empty():
        return df.with_columns(pl.col("place_default").alias("last_place_name"))
    expr = pl.col("place_default")
    # Descending name order wraps ascending names outermost: deterministic
    # precedence even though overlaps are rejected at save time.
    for z in sorted(zones, key=lambda z: z.name, reverse=True):
        inside = (
            pl.col("X").is_between(z.x - z.half_x, z.x + z.half_x)
            & pl.col("Y").is_between(z.y - z.half_y, z.y + z.half_y)
            & ((pl.col("Z") - z.z).abs() <= RECT_Z_BAND)
        )
        expr = pl.when(inside).then(pl.lit(z.name)).otherwise(expr)
    return df.with_columns(expr.alias("last_place_name"))
