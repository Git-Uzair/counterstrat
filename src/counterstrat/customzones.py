"""User-defined callout zones: spheres baked into the lake vocabulary.

A custom zone is a named world-space sphere (Z scaled x2, mirroring
ZoneMapper verticality weighting). :func:`rezone_ticks` rewrites
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

Z_SCALE = 2.0  # verticality weight, mirrors ZoneMapper.z_scale
MIN_RADIUS = 64.0
MAX_RADIUS = 600.0
_LEVELS = {"default", "lower"}


class CustomZone(BaseModel):
    name: str
    x: float
    y: float
    z: float
    level: str = "default"
    radius: float = 150.0


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
        if not (MIN_RADIUS <= z.radius <= MAX_RADIUS):
            raise ValueError(f"radius must be {MIN_RADIUS:.0f}-{MAX_RADIUS:.0f} units")
        if z.level not in _LEVELS:
            raise ValueError(f"Unknown level {z.level!r}")
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z - b.z) * Z_SCALE) ** 2
            if d2 < (a.radius + b.radius) ** 2:
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
        d2 = (
            (pl.col("X") - z.x) ** 2
            + (pl.col("Y") - z.y) ** 2
            + ((pl.col("Z") - z.z) * Z_SCALE) ** 2
        )
        expr = pl.when(d2 <= z.radius**2).then(pl.lit(z.name)).otherwise(expr)
    return df.with_columns(expr.alias("last_place_name"))
