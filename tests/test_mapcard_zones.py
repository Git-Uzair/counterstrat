from pathlib import Path

import polars as pl
import pytest

from counterstrat.mapcard.zones import ZoneMapper


def test_zone_mapper_synthetic():
    df = pl.DataFrame(
        {
            "X": [0.0] * 50 + [1000.0] * 50,
            "Y": [0.0] * 100,
            "Z": [0.0] * 100,
            "last_place_name": ["A"] * 50 + ["B"] * 50,
        }
    )
    zm = ZoneMapper.fit(df)
    assert zm.zone(10, 0, 0) == "A" and zm.zone(990, 0, 0) == "B"


@pytest.mark.demo
def test_zone_mapper_holdout(anubis_lake):
    ticks = pl.read_parquet(anubis_lake.ticks).filter(
        (pl.col("is_alive"))
        & (pl.col("last_place_name").is_not_null())
        & (pl.col("last_place_name") != "")
    )
    train, test = ticks.head(ticks.height - 20_000), ticks.tail(20_000)
    zm = ZoneMapper.fit(train)
    acc = (zm.zones(test) == test["last_place_name"]).mean()
    assert acc >= 0.97


def test_zone_mapper_save_load(tmp_path: Path):
    df = pl.DataFrame(
        {
            "X": [0.0] * 50 + [1000.0] * 50,
            "Y": [0.0] * 100,
            "Z": [0.0] * 100,
            "last_place_name": ["A"] * 50 + ["B"] * 50,
        }
    )
    zm = ZoneMapper.fit(df, z_scale=3.0)
    save_path = tmp_path / "zone_mapper.pkl"
    zm.save(save_path)
    assert save_path.exists()

    loaded = ZoneMapper.load(save_path)
    assert loaded.z_scale == 3.0
    assert loaded.zone(10, 0, 0) == "A"
    assert loaded.zone(990, 0, 0) == "B"
    assert sorted(loaded.classes_) == ["A", "B"]
