import polars as pl
import pytest

from counterstrat.corpus import register_demo
from counterstrat.lake.extract import extract_lake


@pytest.mark.demo
def test_extract_lake(demo_path, tmp_path):
    rec = register_demo(demo_path, tmp_path / "corpus.jsonl")
    paths = extract_lake(rec, tmp_path / "lake")
    rounds = pl.read_parquet(paths.rounds)
    assert rounds.height == 30  # verified count
    assert {"t_team_key", "ct_team_key"} <= set(rounds.columns)
    ticks = pl.read_parquet(paths.ticks)
    assert ticks.height > 400_000  # ~2M/4 minus freeze gaps
    assert {"last_place_name", "clock_s", "round_num"} <= set(ticks.columns)
    # every gameplay tick maps into a round and clock is sane
    assert ticks["clock_s"].max() < 130
    rosters = pl.read_parquet(paths.rosters)
    assert rosters["team_key"].n_unique() == 2  # two teams, PUG-verified
