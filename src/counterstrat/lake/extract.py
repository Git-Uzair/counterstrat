"""Extraction lake: converts registered demo into Level-0 parquet tables."""

import hashlib
from pathlib import Path

import polars as pl
from awpy import Demo
from demoparser2 import DemoParser
from pydantic import BaseModel

from counterstrat.corpus import DemoRecord

TICK_PROPS = [
    "X",
    "Y",
    "Z",
    "last_place_name",
    "health",
    "armor_value",
    "has_helmet",
    "has_defuser",
    "is_alive",
    "active_weapon_name",
    "balance",
    "cash_spent_this_round",
    "current_equip_value",
    "round_start_equip_value",
    "is_walking",
    "flash_duration",
    "is_scoped",
    "is_defusing",
    "in_bomb_zone",
    "team_name",
    "team_clan_name",
    "is_bomb_planted",
    "total_rounds_played",
    "ct_losing_streak",
    "t_losing_streak",
    "velocity",
    "pitch",
    "yaw",
]

AWPY_TABLES = [
    "rounds",
    "kills",
    "damages",
    "shots",
    "grenades",
    "smokes",
    "infernos",
    "bomb",
]


class LakePaths(BaseModel):
    root: str
    rounds: str
    kills: str
    damages: str
    shots: str
    grenades: str
    smokes: str
    infernos: str
    bomb: str
    item_purchase: str
    ticks: str
    rosters: str
    player_blind: str = ""


def _sample_ticks(parser: DemoParser, rounds: pl.DataFrame, step: int = 4) -> pl.DataFrame:
    if rounds.is_empty():
        return pl.DataFrame()
    start_min = int(rounds["start"].min())
    end_max = int(rounds["end"].max())
    tick_list = list(range(start_min, end_max, step))
    ticks_df = parser.parse_ticks(TICK_PROPS, ticks=tick_list)
    ticks_pl = pl.from_pandas(ticks_df)
    rounds_sub = rounds.select(["round_num", "start", "freeze_end", "end"]).sort("start")
    ticks_joined = (
        ticks_pl.sort("tick")
        .join_asof(rounds_sub, left_on="tick", right_on="start", strategy="backward")
        .filter(pl.col("tick") <= pl.col("end"))
        .with_columns(
            ((pl.col("tick") - pl.col("freeze_end")) / 64.0).alias("clock_s"),
            pl.when(pl.col("last_place_name") == "")
            .then(None)
            .otherwise(pl.col("last_place_name"))
            .alias("last_place_name"),
        )
        .drop(["start", "freeze_end", "end"])
    )
    return ticks_joined


def _rosters(parser: DemoParser, rounds: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "round_num": pl.Int64,
        "side": pl.String,
        "team_key": pl.String,
        "steamids": pl.List(pl.Int64),
        "clan_name": pl.String,
    }
    if rounds.is_empty():
        return pl.DataFrame(schema=schema)

    freeze_end_ticks = rounds["freeze_end"].to_list()
    df = parser.parse_ticks(["team_name", "team_clan_name", "is_alive"], ticks=freeze_end_ticks)
    tick_to_round = dict(
        zip(rounds["freeze_end"].to_list(), rounds["round_num"].to_list(), strict=True)
    )

    df = df[df["is_alive"] & df["team_name"].isin(["CT", "TERRORIST"])]

    rows = []
    for (tick, team_name), group in df.groupby(["tick", "team_name"]):
        round_num = tick_to_round.get(tick)
        steamids = sorted(group["steamid"].unique().tolist())
        key_str = ",".join(sorted(str(s) for s in steamids))
        team_key = hashlib.sha1(key_str.encode()).hexdigest()[:12]
        clans = [str(c) for c in group["team_clan_name"].dropna().tolist() if str(c).strip()]
        clan_name = clans[0] if clans else ""
        rows.append(
            {
                "round_num": round_num,
                "side": str(team_name),
                "team_key": team_key,
                "steamids": steamids,
                "clan_name": clan_name,
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort(["round_num", "side"])


def _attach_team_keys(rounds: pl.DataFrame, rosters: pl.DataFrame) -> pl.DataFrame:
    if rounds.is_empty() or rosters.is_empty():
        return rounds.with_columns(
            pl.lit(None).cast(pl.String).alias("ct_team_key"),
            pl.lit(None).cast(pl.String).alias("t_team_key"),
        )
    ct_keys = rosters.filter(pl.col("side") == "CT").select(
        pl.col("round_num"), pl.col("team_key").alias("ct_team_key")
    )
    t_keys = rosters.filter(pl.col("side") == "TERRORIST").select(
        pl.col("round_num"), pl.col("team_key").alias("t_team_key")
    )
    return rounds.join(ct_keys, on="round_num", how="left").join(t_keys, on="round_num", how="left")


def extract_lake(record: DemoRecord, out_root: Path) -> LakePaths:
    out = Path(out_root) / record.match_id
    out.mkdir(parents=True, exist_ok=True)
    dem = Demo(record.path)
    dem.parse()
    tables: dict[str, pl.DataFrame] = {}
    for name in AWPY_TABLES:
        try:
            val = getattr(dem, name)
            if isinstance(val, pl.DataFrame):
                tables[name] = val
            elif val is not None:
                tables[name] = pl.DataFrame(val)
            else:
                tables[name] = pl.DataFrame()
        except Exception:  # noqa: BLE001
            tables[name] = pl.DataFrame()

    rounds = tables.get("rounds", pl.DataFrame())
    parser = DemoParser(record.path)
    try:
        purchases_df = parser.parse_event("item_purchase", other=["total_rounds_played"])
        purchases = pl.from_pandas(purchases_df)
    except Exception:  # noqa: BLE001
        purchases = pl.DataFrame()

    try:
        blind_df = parser.parse_event("player_blind", other=["total_rounds_played"])
        player_blind = pl.from_pandas(blind_df)
    except Exception:  # noqa: BLE001
        player_blind = pl.DataFrame()

    ticks = _sample_ticks(parser, rounds)
    rosters = _rosters(parser, rounds)
    rounds = _attach_team_keys(rounds, rosters)

    tables["item_purchase"] = purchases
    tables["player_blind"] = player_blind
    tables["ticks"] = ticks
    tables["rosters"] = rosters
    tables["rounds"] = rounds

    paths: dict[str, str] = {}
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        if df.height > 0:
            df = df.with_columns(pl.lit(record.match_id).alias("match_id"))
        else:
            df = df.with_columns(pl.Series("match_id", [], dtype=pl.String))
        df.write_parquet(p)
        paths[name] = str(p)

    return LakePaths(root=str(out), **paths)
