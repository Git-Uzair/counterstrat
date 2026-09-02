import polars as pl
import pytest

from counterstrat.roundscript.econ import EconSummary, classify_buy, round_economy


def test_classify_buy_bins():
    assert classify_buy(3_000) == "full_eco"
    assert classify_buy(5_000) == "semi_eco"  # boundary goes up
    assert classify_buy(19_999) == "semi_buy"
    assert classify_buy(26_000) == "full_buy"


def test_round_economy_synthetic():
    ticks = pl.DataFrame(
        {
            "round_num": [1, 1, 1],
            "clock_s": [-1.0, 0.0, 0.5],
            "steamid": [101, 101, 101],
            "team_name": ["TERRORIST", "TERRORIST", "TERRORIST"],
            "current_equip_value": [1000, 2000, 2500],
            "cash_spent_this_round": [500, 1000, 1000],
            "active_weapon_name": ["knife", "AWP", "knife"],
            "t_losing_streak": [2, 2, 2],
            "ct_losing_streak": [0, 0, 0],
        }
    )
    econ = round_economy(ticks, round_num=1)
    assert isinstance(econ["T"], EconSummary)
    assert isinstance(econ["CT"], EconSummary)
    # Freeze-end sample is clock_s == 0.0
    assert econ["T"].equip == 2000
    assert econ["T"].spend == 1000
    assert econ["T"].buy_type == "full_eco"
    assert econ["T"].awps == 1
    assert econ["T"].loss_streak == 2
    # CT is empty
    assert econ["CT"].equip == 0
    assert econ["CT"].spend == 0
    assert econ["CT"].buy_type == "full_eco"
    assert econ["CT"].awps == 0
    assert econ["CT"].loss_streak == 0


@pytest.mark.demo
def test_round_economy_pistol(anubis_lake):
    econ = round_economy(pl.read_parquet(anubis_lake.ticks), round_num=1)
    assert econ["T"].buy_type in {"full_eco", "semi_eco"}
    assert econ["CT"].equip > 0
    assert "T" in econ and "CT" in econ
    assert isinstance(econ["T"].loss_streak, int)
    assert isinstance(econ["CT"].spend, int)


@pytest.mark.demo
def test_round_economy_awp_round(anubis_lake):
    econ = round_economy(pl.read_parquet(anubis_lake.ticks), round_num=12)
    assert econ["T"].awps == 1
    assert econ["T"].buy_type == "full_buy"
    assert econ["CT"].buy_type == "full_buy"
