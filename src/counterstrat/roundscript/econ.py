"""Buy classification and economy summarization for CS2 rounds."""

from typing import Any

import polars as pl
from pydantic import BaseModel

from counterstrat.constants import BUY_BINS


class EconSummary(BaseModel):
    buy_type: str
    spend: int
    equip: int
    awps: int
    loss_streak: int


def classify_buy(team_equip_value: float) -> str:
    """Classify buy type according to team equipment value bins."""
    if team_equip_value < BUY_BINS["full_eco"]:
        return "full_eco"
    elif team_equip_value < BUY_BINS["semi_eco"]:
        return "semi_eco"
    elif team_equip_value < BUY_BINS["semi_buy"]:
        return "semi_buy"
    else:
        return "full_buy"


def _has_awp(row: dict[str, Any]) -> bool:
    active = str(row.get("active_weapon_name") or "").lower()
    if "awp" in active:
        return True
    inv = row.get("inventory")
    if inv is not None:
        if isinstance(inv, (list, tuple)):
            return any("awp" in str(item).lower() for item in inv)
        return "awp" in str(inv).lower()
    return False


def round_economy(ticks: pl.DataFrame, round_num: int) -> dict[str, EconSummary]:
    """Summarize the round economy for T and CT sides at freeze-end."""
    round_ticks = ticks.filter(pl.col("round_num") == round_num)
    if round_ticks.is_empty():
        return {
            "T": EconSummary(buy_type=classify_buy(0), spend=0, equip=0, awps=0, loss_streak=0),
            "CT": EconSummary(buy_type=classify_buy(0), spend=0, equip=0, awps=0, loss_streak=0),
        }

    post_freeze = round_ticks.filter(pl.col("clock_s") >= 0)
    sample_source = post_freeze if not post_freeze.is_empty() else round_ticks

    if "steamid" in sample_source.columns:
        freeze_sample = sample_source.sort("clock_s").unique(subset=["steamid"], keep="first")
    else:
        min_clock = sample_source["clock_s"].min()
        freeze_sample = sample_source.filter(pl.col("clock_s") == min_clock)

    out: dict[str, EconSummary] = {}
    for side in ("T", "CT"):
        if "team_name" in freeze_sample.columns:
            if side == "T":
                side_filter = (pl.col("team_name") == "TERRORIST") | (pl.col("team_name") == "T")
            else:
                side_filter = pl.col("team_name") == "CT"
            team_rows = freeze_sample.filter(side_filter)
        else:
            team_rows = pl.DataFrame(schema=freeze_sample.schema)

        if team_rows.is_empty():
            out[side] = EconSummary(
                buy_type=classify_buy(0), spend=0, equip=0, awps=0, loss_streak=0
            )
            continue

        equip = (
            int(team_rows["current_equip_value"].drop_nulls().sum())
            if "current_equip_value" in team_rows.columns
            else 0
        )
        spend = (
            int(team_rows["cash_spent_this_round"].drop_nulls().sum())
            if "cash_spent_this_round" in team_rows.columns
            else 0
        )
        buy_type = classify_buy(equip)

        awps = sum(1 for row in team_rows.iter_rows(named=True) if _has_awp(row))

        loss_streak_col = "t_losing_streak" if side == "T" else "ct_losing_streak"
        if loss_streak_col in team_rows.columns:
            streak_series = team_rows[loss_streak_col].drop_nulls()
            loss_streak = int(streak_series.max()) if not streak_series.is_empty() else 0
        else:
            loss_streak = 0

        out[side] = EconSummary(
            buy_type=buy_type,
            spend=spend,
            equip=equip,
            awps=awps,
            loss_streak=loss_streak,
        )

    return out
