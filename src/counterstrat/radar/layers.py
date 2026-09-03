"""Lake tables -> normalized radar overlay layers (spec item N2).

Every layer is scoped to one ``team_key`` through ``rosters``: the team's rounds
give the side it played, and its ``steamids`` give the players to keep. Getting
that scope wrong silently plots the opponent, so it is built once in
:func:`build_team_scope` and reused by every layer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from counterstrat.radar.coords import (
    ROUND_DP,
    level_expr,
    norm_x_expr,
    norm_y_expr,
    side_expr,
)
from counterstrat.radar.extract import RadarCalibration

logger = logging.getLogger(__name__)

LAKE_TABLES = ("rosters", "ticks", "kills", "grenades", "bomb")

# Only the *Projectile entities are thrown nades; the plain C*Grenade rows track
# the inventory-held entity for the whole round (~2600 rows each vs ~250).
PROJECTILE_KIND = {
    "CSmokeGrenadeProjectile": "smoke",
    "CFlashbangProjectile": "flash",
    "CHEGrenadeProjectile": "he",
    "CMolotovProjectile": "molotov",  # incendiaries spawn this too
    "CDecoyProjectile": "decoy",
}

BOMB_EVENTS = ("plant", "defuse")


class LayerFilters(BaseModel):
    side: Literal["T", "CT"] | None = None
    level: Literal["default", "lower", "all"] = "all"
    round_nums: list[int] | None = None
    trail_rounds: int | None = 4
    stride: int = 8
    grid: int = 128


@dataclass(slots=True)
class TeamScope:
    team_key: str
    rounds: pl.DataFrame
    members: pl.DataFrame


def _empty_scope(team_key: str) -> TeamScope:
    return TeamScope(
        team_key=team_key,
        rounds=pl.DataFrame(
            schema={"match_id": pl.String, "round_num": pl.Int64, "side": pl.String}
        ),
        members=pl.DataFrame(
            schema={"match_id": pl.String, "round_num": pl.Int64, "steamid": pl.Int64}
        ),
    )


def lake_frames(lake_root: Path, match_ids: list[str]) -> dict[str, pl.LazyFrame | None]:
    """Lazy scans of the radar-relevant tables for ``match_ids``; None when absent."""
    out: dict[str, pl.LazyFrame | None] = {}
    for table in LAKE_TABLES:
        paths = [p for m in match_ids if (p := Path(lake_root) / m / f"{table}.parquet").exists()]
        out[table] = pl.scan_parquet(paths) if paths else None
    return out


def _keys(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Normalizes the join keys every table disagrees on."""
    return frame.with_columns(pl.col("round_num").cast(pl.Int64))


def _apply_level(frame: Any, f: LayerFilters) -> Any:
    return frame if f.level == "all" else frame.filter(pl.col("level") == f.level)


def _in_bounds() -> pl.Expr:
    return (pl.col("u") >= 0) & (pl.col("u") < 1) & (pl.col("v") >= 0) & (pl.col("v") < 1)


def _r(value: Any, dp: int = ROUND_DP) -> float | None:
    """Rounds a wire coordinate to ``dp`` decimals. Must stay in Python, not Polars.

    The lake's X/Y are Float32, so ``pl.col("u").round(4)`` snaps to the nearest
    *Float32* and iter_rows widens it back to a 17-digit Float64
    (0.4092999994754791). ``float(value)`` widens first, so the round lands on a
    clean 4-decimal double. ``+ 0.0`` folds -0.0 (from coordinates a hair below
    the image origin) to 0.0.
    """
    return None if value is None else round(float(value), dp) + 0.0


def _sid(value: Any) -> str | None:
    """steamids cross the wire as strings: they exceed JS's 2**53 safe range."""
    return None if value is None else str(int(value))


def build_team_scope(rosters: pl.LazyFrame | None, team_key: str, f: LayerFilters) -> TeamScope:
    """Resolves which (match, round, side) and which players belong to ``team_key``."""
    if rosters is None:
        return _empty_scope(team_key)

    rounds = (
        _keys(rosters.select("match_id", "round_num", "side", "team_key", "steamids"))
        .filter(pl.col("team_key") == team_key)
        .with_columns(side_expr("side"))
        .collect()
    )
    if f.side is not None:
        rounds = rounds.filter(pl.col("side") == f.side)
    if f.round_nums:
        rounds = rounds.filter(pl.col("round_num").is_in(f.round_nums))
    rounds = rounds.sort("match_id", "round_num")

    if rounds.is_empty():
        return _empty_scope(team_key)

    members = (
        rounds.explode("steamids", empty_as_null=True)
        .rename({"steamids": "steamid"})
        .select("match_id", "round_num", pl.col("steamid").cast(pl.Int64))
        .drop_nulls()
        .unique()
    )
    return TeamScope(
        team_key=team_key,
        rounds=rounds.select("match_id", "round_num", "side"),
        members=members,
    )


def _team_ticks(ticks: pl.LazyFrame, cal: RadarCalibration, scope: TeamScope) -> pl.LazyFrame:
    return (
        _keys(ticks)
        .with_columns(pl.col("steamid").cast(pl.Int64))
        .filter(pl.col("is_alive"))
        .join(scope.members.lazy(), on=["match_id", "round_num", "steamid"], how="semi")
        .with_columns(
            norm_x_expr(cal, "X").alias("u"),
            norm_y_expr(cal, "Y").alias("v"),
            level_expr(cal, "Z").alias("level"),
        )
    )


def heatmap_layer(
    ticks: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> dict[str, Any]:
    """Occupancy histogram over a ``f.grid`` x ``f.grid`` lattice of the radar image."""
    empty: dict[str, Any] = {"grid": f.grid, "max": 0, "samples": 0, "cells": []}
    if ticks is None or scope.members.is_empty():
        return empty

    df = (
        _apply_level(_team_ticks(ticks, cal, scope), f)
        .filter(_in_bounds())
        .with_columns(
            (pl.col("u") * f.grid).floor().cast(pl.Int32).alias("gx"),
            (pl.col("v") * f.grid).floor().cast(pl.Int32).alias("gy"),
        )
        .group_by("gx", "gy")
        .agg(pl.len().alias("n"))
        .sort("gy", "gx")
        .collect()
    )
    cells = [[int(gx), int(gy), int(n)] for gx, gy, n in df.iter_rows()]
    return {
        "grid": f.grid,
        "max": max((c[2] for c in cells), default=0),
        "samples": sum(c[2] for c in cells),
        "cells": cells,
    }


def trails_layer(
    ticks: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """One decimated polyline per (round, player), capped at ``f.trail_rounds`` rounds."""
    if ticks is None or scope.members.is_empty():
        return []

    keep = scope.rounds.select("match_id", "round_num").unique().sort("match_id", "round_num")
    if f.trail_rounds is not None:
        keep = keep.head(f.trail_rounds)
    members = scope.members.join(keep, on=["match_id", "round_num"], how="semi")
    if members.is_empty():
        return []

    stride = max(1, f.stride)
    df = (
        _apply_level(_team_ticks(ticks, cal, TeamScope(scope.team_key, scope.rounds, members)), f)
        .with_columns(side_expr("team_name").alias("side"))
        .sort("match_id", "round_num", "steamid", "tick")
        .with_columns(pl.int_range(pl.len()).over("match_id", "round_num", "steamid").alias("i"))
        .filter(pl.col("i") % stride == 0)
        .group_by("match_id", "round_num", "steamid", maintain_order=True)
        .agg(
            pl.col("name").first(),
            pl.col("side").first(),
            pl.col("u"),
            pl.col("v"),
            pl.col("clock_s"),
        )
        .sort("match_id", "round_num", "steamid")
        .collect()
    )
    return [
        {
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "steamid": _sid(r["steamid"]),
            "name": r["name"],
            "side": r["side"],
            # Rounded here, not in the aggregation above: see _r().
            "points": [
                [_r(u), _r(v), _r(c, 1)]
                for u, v, c in zip(r["u"], r["v"], r["clock_s"], strict=True)
            ],
        }
        for r in df.iter_rows(named=True)
    ]


def utility_layer(
    grenades: pl.LazyFrame | None,
    cal: RadarCalibration,
    scope: TeamScope,
    f: LayerFilters,
) -> list[dict[str, Any]]:
    """The team's own thrown nades: throw origin + landing, per projectile entity."""
    if grenades is None or scope.members.is_empty():
        return []

    df = (
        _keys(
            grenades.select(
                "match_id",
                "round_num",
                "thrower_steamid",
                "thrower",
                "grenade_type",
                "entity_id",
                "X",
                "Y",
                "Z",
                "tick",
            )
        )
        .filter(pl.col("grenade_type").is_in(list(PROJECTILE_KIND)))
        .with_columns(pl.col("thrower_steamid").cast(pl.Int64).alias("steamid"))
        .join(scope.members.lazy(), on=["match_id", "round_num", "steamid"], how="semi")
        .group_by("match_id", "round_num", "entity_id")
        .agg(
            pl.col("grenade_type").first(),
            pl.col("thrower").first(),
            pl.col("steamid").first(),
            pl.col("tick").min().alias("throw_tick"),
            pl.col("X").sort_by("tick").first().alias("from_x"),
            pl.col("Y").sort_by("tick").first().alias("from_y"),
            pl.col("X").sort_by("tick").last().alias("land_x"),
            pl.col("Y").sort_by("tick").last().alias("land_y"),
            pl.col("Z").sort_by("tick").last().alias("land_z"),
        )
        .with_columns(
            norm_x_expr(cal, "land_x").alias("u"),
            norm_y_expr(cal, "land_y").alias("v"),
            norm_x_expr(cal, "from_x").alias("from_u"),
            norm_y_expr(cal, "from_y").alias("from_v"),
            level_expr(cal, "land_z").alias("level"),
        )
        .sort("match_id", "round_num", "throw_tick")
        .collect()
    )
    df = _apply_level(df, f)
    return [
        {
            "kind": PROJECTILE_KIND[r["grenade_type"]],
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "thrower": r["thrower"],
            "steamid": _sid(r["steamid"]),
            "u": _r(r["u"]),
            "v": _r(r["v"]),
            "from_u": _r(r["from_u"]),
            "from_v": _r(r["from_v"]),
            "level": r["level"],
        }
        for r in df.iter_rows(named=True)
    ]


def duels_layer(
    kills: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """Every kill in the team's rounds, tagged ``by_team`` for the team's own kills."""
    if kills is None or scope.rounds.is_empty():
        return []

    df = (
        _keys(
            kills.select(
                "match_id",
                "round_num",
                "tick",
                "weapon",
                "headshot",
                "attacker_X",
                "attacker_Y",
                "attacker_Z",
                "attacker_name",
                "attacker_side",
                "victim_X",
                "victim_Y",
                "victim_Z",
                "victim_name",
                "victim_side",
            )
        )
        .with_columns(side_expr("attacker_side"), side_expr("victim_side"))
        .join(scope.rounds.lazy(), on=["match_id", "round_num"], how="inner")
        .with_columns(
            norm_x_expr(cal, "victim_X").alias("u"),
            norm_y_expr(cal, "victim_Y").alias("v"),
            norm_x_expr(cal, "attacker_X").alias("au"),
            norm_y_expr(cal, "attacker_Y").alias("av"),
            level_expr(cal, "victim_Z").alias("level"),
            (pl.col("attacker_side") == pl.col("side")).fill_null(False).alias("by_team"),
        )
        .sort("match_id", "round_num", "tick")
        .collect()
    )
    df = _apply_level(df, f)
    out: list[dict[str, Any]] = []
    for r in df.iter_rows(named=True):
        attacker = None
        if r["au"] is not None and r["av"] is not None:
            attacker = {
                "name": r["attacker_name"],
                "side": r["attacker_side"],
                "u": _r(r["au"]),
                "v": _r(r["av"]),
            }
        out.append(
            {
                "match_id": r["match_id"],
                "round_num": int(r["round_num"]),
                "tick": int(r["tick"]),
                "weapon": r["weapon"],
                "headshot": bool(r["headshot"]),
                "by_team": bool(r["by_team"]),
                "level": r["level"],
                "victim": {
                    "name": r["victim_name"],
                    "side": r["victim_side"],
                    "u": _r(r["u"]),
                    "v": _r(r["v"]),
                },
                "attacker": attacker,
            }
        )
    return out


def bomb_layer(
    bomb: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """Plant and defuse positions inside the team's rounds."""
    if bomb is None or scope.rounds.is_empty():
        return []

    df = (
        _keys(
            bomb.select(
                "match_id",
                "round_num",
                "tick",
                "event",
                "X",
                "Y",
                "Z",
                "steamid",
                "name",
                "bombsite",
            )
        )
        .filter(pl.col("event").is_in(list(BOMB_EVENTS)))
        .join(scope.rounds.lazy(), on=["match_id", "round_num"], how="inner")
        .with_columns(
            norm_x_expr(cal, "X").alias("u"),
            norm_y_expr(cal, "Y").alias("v"),
            level_expr(cal, "Z").alias("level"),
        )
        .sort("match_id", "round_num", "tick")
        .collect()
    )
    df = _apply_level(df, f)
    return [
        {
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "event": r["event"],
            "bombsite": r["bombsite"],
            "name": r["name"],
            "steamid": _sid(r["steamid"]),
            "tick": int(r["tick"]),
            "u": _r(r["u"]),
            "v": _r(r["v"]),
            "level": r["level"],
        }
        for r in df.iter_rows(named=True)
    ]


def build_layers(
    frames: dict[str, pl.LazyFrame | None],
    cal: RadarCalibration,
    team_key: str,
    f: LayerFilters,
) -> dict[str, Any]:
    """Assembles the full radar payload for one (team, map) selection."""
    scope = build_team_scope(frames.get("rosters"), team_key, f)
    return {
        "map_name": cal.map_name,
        "team_key": team_key,
        "image_px": cal.image_px,
        "lower_altitude_max": cal.lower_altitude_max,
        "filters": f.model_dump(),
        "rounds": [
            {
                "match_id": r["match_id"],
                "round_num": int(r["round_num"]),
                "side": r["side"],
            }
            for r in scope.rounds.iter_rows(named=True)
        ],
        "layers": {
            "heatmap": heatmap_layer(frames.get("ticks"), cal, scope, f),
            "trails": trails_layer(frames.get("ticks"), cal, scope, f),
            "utility": utility_layer(frames.get("grenades"), cal, scope, f),
            "duels": duels_layer(frames.get("kills"), cal, scope, f),
            "bombs": bomb_layer(frames.get("bomb"), cal, scope, f),
        },
    }
