"""User-defined callout zones baked into the lake vocabulary.

A custom zone is a named world-space region: a sphere (Z scaled x2,
mirroring ZoneMapper verticality weighting) or an axis-aligned rectangle
(half-extents around the center, with a fixed +-RECT_Z_BAND vertical band
so stacked levels stay independent). :func:`rezone_ticks` rewrites
``last_place_name`` for ticks inside a region while preserving the game's
own name in ``place_default`` - idempotent, reversible, and applied at the
data layer so every consumer (scripts, miners, card, anchors, SQL, prompts)
inherits the user's vocabulary without translation.
"""

import json
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel

from counterstrat.aliases import _ALIAS_RE

Z_SCALE = 2.0  # verticality weight, mirrors ZoneMapper.z_scale
MIN_RADIUS = 64.0  # spheres: below this, transit ticks slip through
MAX_RADIUS = 600.0
# Rects may be tiny: hide spots and one-way angles are held positions, which
# the 4 Hz ticks capture densely no matter how small the footprint is.
MIN_RECT_HALF = 5.0
RECT_Z_BAND = 200.0  # rect vertical half-extent: covers ramps, excludes nuke's other level
_LEVELS = {"default", "lower"}


class CustomZone(BaseModel):
    name: str
    x: float
    y: float
    z: float
    level: str = "default"
    shape: Literal["sphere", "rect"] = "sphere"
    radius: float = 150.0  # sphere only
    half_x: float | None = None  # rect only: X half-extent
    half_y: float | None = None  # rect only: Y half-extent

    def z_half(self) -> float:
        """Vertical half-extent: the Z distance at which a tick still belongs."""
        return RECT_Z_BAND if self.shape == "rect" else self.radius / Z_SCALE


def _zones_overlap(a: "CustomZone", b: "CustomZone") -> bool:
    """True when two zones claim the same ground (XY footprint AND Z band)."""
    if abs(a.z - b.z) > a.z_half() + b.z_half():
        return False  # stacked levels (nuke) never conflict
    if a.shape == "sphere" and b.shape == "sphere":
        d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z - b.z) * Z_SCALE) ** 2
        return d2 < (a.radius + b.radius) ** 2
    if a.shape == "rect" and b.shape == "rect":
        return abs(a.x - b.x) < (a.half_x or 0) + (b.half_x or 0) and abs(a.y - b.y) < (
            a.half_y or 0
        ) + (b.half_y or 0)
    rect, sphere = (a, b) if a.shape == "rect" else (b, a)
    # Closest point of the rect footprint to the sphere center, in 2D.
    cx = min(max(sphere.x, rect.x - (rect.half_x or 0)), rect.x + (rect.half_x or 0))
    cy = min(max(sphere.y, rect.y - (rect.half_y or 0)), rect.y + (rect.half_y or 0))
    return (sphere.x - cx) ** 2 + (sphere.y - cy) ** 2 < sphere.radius**2


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
        if z.shape == "rect":
            if z.half_x is None or z.half_y is None:
                raise ValueError(f"Rect zone {z.name!r} needs half_x and half_y")
            for half in (z.half_x, z.half_y):
                if not (MIN_RECT_HALF <= half <= MAX_RADIUS):
                    raise ValueError(
                        f"rect half-extents must be {MIN_RECT_HALF:.0f}-{MAX_RADIUS:.0f} units"
                    )
        elif not (MIN_RADIUS <= z.radius <= MAX_RADIUS):
            raise ValueError(f"radius must be {MIN_RADIUS:.0f}-{MAX_RADIUS:.0f} units")
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
        if z.shape == "rect":
            inside = (
                pl.col("X").is_between(z.x - (z.half_x or 0), z.x + (z.half_x or 0))
                & pl.col("Y").is_between(z.y - (z.half_y or 0), z.y + (z.half_y or 0))
                & ((pl.col("Z") - z.z).abs() <= RECT_Z_BAND)
            )
        else:
            d2 = (
                (pl.col("X") - z.x) ** 2
                + (pl.col("Y") - z.y) ** 2
                + ((pl.col("Z") - z.z) * Z_SCALE) ** 2
            )
            inside = d2 <= z.radius**2
        expr = pl.when(inside).then(pl.lit(z.name)).otherwise(expr)
    return df.with_columns(expr.alias("last_place_name"))
