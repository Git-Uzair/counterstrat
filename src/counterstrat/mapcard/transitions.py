"""Zone-level adjacency + transit times observed from lake tick transitions.

Fallback (and calibrator) for the nav-mesh zone graph: instead of walkable-area
geometry, take the adjacency players actually demonstrate by walking. An edge
``(A, B)`` exists when some player was sampled in zone A and then, on the very
next sample, in zone B. Transit time is measured dwell-boundary to
dwell-boundary (entry into A -> entry into B), capped at ``MAX_TRANSIT_S``.
"""

import polars as pl
from pydantic import BaseModel

MAX_TRANSIT_S = 30.0

_KEY = ["steamid", "round_num"]


class EdgeStat(BaseModel):
    n: int
    median_transit_s: float


class ZoneGraph(BaseModel):
    edges: dict[tuple[str, str], EdgeStat]


def _key_cols(ticks: pl.DataFrame) -> list[str]:
    """Grouping key; includes match_id when present so multi-match frames
    cannot interleave the same (steamid, round_num) across matches."""
    return (["match_id"] if "match_id" in ticks.columns else []) + _KEY


def zone_graph(ticks: pl.DataFrame) -> ZoneGraph:
    key = _key_cols(ticks)
    # clock_s < 0 is freeze time; those rows belong to the NEXT round but can
    # arrive under the previous round_num, so a round-end position followed by
    # the respawn teleport would otherwise read as a zone transition.
    df = ticks.filter(
        pl.col("is_alive")
        & pl.col("last_place_name").is_not_null()
        & (pl.col("last_place_name") != "")
        & (pl.col("clock_s") >= 0.0)
    ).sort([*key, "tick"])
    if df.height < 2:
        return ZoneGraph(edges={})

    df = df.with_columns(
        pl.col("last_place_name").shift(1).over(key).alias("_prev_zone"),
        (pl.col("tick") - pl.col("tick").shift(1).over(key)).alias("_gap"),
    )

    # The sampling interval is the smallest gap between two kept samples of the
    # same player-round; anything wider means samples were dropped in between,
    # so the observed zone change may hide intermediate zones (no skip edges).
    step = df.filter(pl.col("_gap") > 0)["_gap"].min()
    if step is None:
        return ZoneGraph(edges={})

    df = df.with_columns(
        (pl.col("_prev_zone").is_null() | (pl.col("_prev_zone") != pl.col("last_place_name")))
        .cum_sum()
        .over(key)
        .alias("_run")
    )
    df = df.with_columns(pl.col("clock_s").first().over([*key, "_run"]).alias("_run_start_s"))
    # On a boundary row the previous row is the last sample of the previous run,
    # so its run start is when the player entered the zone they are leaving.
    df = df.with_columns(
        (pl.col("clock_s") - pl.col("_run_start_s").shift(1).over(key))
        .clip(upper_bound=MAX_TRANSIT_S)
        .alias("_transit_s")
    )

    transitions = df.filter(
        pl.col("_prev_zone").is_not_null()
        & (pl.col("_prev_zone") != pl.col("last_place_name"))
        & (pl.col("_gap") <= step)
        # A clock that went backwards is a round boundary (respawn teleport),
        # not movement: transit must be strictly positive.
        & (pl.col("_transit_s") > 0.0)
    )

    agg = transitions.group_by(["_prev_zone", "last_place_name"]).agg(
        pl.len().alias("n"), pl.col("_transit_s").median().alias("median_transit_s")
    )
    return ZoneGraph(
        edges={
            (row["_prev_zone"], row["last_place_name"]): EdgeStat(
                n=row["n"], median_transit_s=row["median_transit_s"]
            )
            for row in agg.iter_rows(named=True)
        }
    )
