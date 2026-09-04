"""Empirical zone visibility from kill evidence (space-vision research §4.1, Route A).

Every clean kill proves a sightline: the attacker saw the victim unless the
shot went through smoke or a wall. Aggregating kill pairs across every match
on a map yields an observed zone-to-zone visibility matrix - zero geometry,
zero new dependencies, confidence grows with the corpus. Survivorship caveat:
pairs nobody ever fought across stay unknown, which is acceptable for
counter-stratting (fights that never happen matter less than ones that do).
"""

import logging
from pathlib import Path

import polars as pl
import yaml

from counterstrat.constants import range_band

logger = logging.getLogger(__name__)

MIN_PAIR_EVENTS = 2  # a single kill across a pair is an anecdote, not a sightline
MAX_SIGHTLINES = 60  # card token budget

_REQUIRED = {"attacker_place", "victim_place", "distance"}


def build_sightlines(kills: pl.DataFrame, valid_zones: set[str] | None = None) -> list[dict]:
    """Zone-pair sightlines from clean kills: n, median distance (meters), band.

    Pairs are unordered (visibility is symmetric); ``from``/``to`` are the
    alphabetical order of the pair. Sorted by n desc, capped.
    """
    if kills.is_empty() or not _REQUIRED.issubset(kills.columns):
        return []

    df = kills.filter(
        pl.col("attacker_place").is_not_null()
        & (pl.col("attacker_place").cast(pl.String).str.strip_chars() != "")
        & pl.col("victim_place").is_not_null()
        & (pl.col("victim_place").cast(pl.String).str.strip_chars() != "")
        & pl.col("distance").is_not_null()
        & (pl.col("attacker_place") != pl.col("victim_place"))
    )
    if "thrusmoke" in df.columns:
        df = df.filter(~pl.col("thrusmoke").fill_null(False))
    if "penetrated" in df.columns:
        df = df.filter(pl.col("penetrated").fill_null(0).cast(pl.Int64) == 0)
    if valid_zones is not None:
        df = df.filter(
            pl.col("attacker_place").is_in(valid_zones) & pl.col("victim_place").is_in(valid_zones)
        )
    if df.is_empty():
        return []

    paired = df.with_columns(
        pl.min_horizontal("attacker_place", "victim_place").alias("_a"),
        pl.max_horizontal("attacker_place", "victim_place").alias("_b"),
    )
    agg = (
        paired.group_by(["_a", "_b"])
        .agg(pl.len().alias("n"), pl.col("distance").median().alias("median_dist"))
        .filter(pl.col("n") >= MIN_PAIR_EVENTS)
        .sort(["n", "_a", "_b"], descending=[True, False, False])
        .head(MAX_SIGHTLINES)
    )
    return [
        {
            "from": row["_a"],
            "to": row["_b"],
            "n": int(row["n"]),
            "median_dist": round(float(row["median_dist"]), 1),
            "range": range_band(float(row["median_dist"])),
        }
        for row in agg.iter_rows(named=True)
    ]


def refresh_card_sightlines(data_root: Path, map_name: str) -> int:
    """Recompute the map card's sightlines from every lake kill on this map.

    Called after ingest and zone rebuilds so the matrix grows with the corpus.
    Rewrites card.yaml in place; the compile-time checksum is provenance and
    stays untouched. Returns the number of sightline pairs written (0 when no
    card or no evidence).
    """
    from counterstrat.corpus import load_manifest

    card_path = data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        return 0
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        return 0

    manifest = load_manifest(data_root / "corpus.jsonl")
    frames: list[pl.DataFrame] = []
    for mid, rec in sorted(manifest.items()):
        if rec.map_name != map_name:
            continue
        kills_path = data_root / "lake" / mid / "kills.parquet"
        if not kills_path.exists():
            continue
        try:
            df = pl.read_parquet(kills_path)
        except Exception as exc:  # noqa: BLE001 - one bad table must not kill the refresh
            logger.warning("Unreadable kills table %s: %s", kills_path, exc)
            continue
        if _REQUIRED.issubset(df.columns):
            cols = list(_REQUIRED) + [c for c in ("thrusmoke", "penetrated") if c in df.columns]
            frames.append(df.select(sorted(cols)))

    kills = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()
    zones = set(card.get("zones") or {}) or None
    sightlines = build_sightlines(kills, valid_zones=zones)
    card["sightlines"] = sightlines

    from counterstrat.mapcard.compile import MapCard

    card_path.write_text(MapCard(**card).to_yaml(), encoding="utf-8")
    return len(sightlines)
