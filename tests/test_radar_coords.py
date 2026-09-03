import polars as pl
import pytest

from counterstrat.radar.coords import (
    game_to_norm,
    game_to_pixel,
    is_lower_level,
    level_expr,
    norm_x_expr,
    norm_y_expr,
    normalize_side,
    pixel_to_game,
    side_expr,
)
from counterstrat.radar.extract import RadarCalibration

# Deliberately round numbers: 2048 world units across 1024 px at 2 u/px, so
# world (0, 0) sits dead centre and the corners are exactly (0,0) and (1,1).
CAL = RadarCalibration(map_name="de_test", pos_x=-1024.0, pos_y=1024.0, scale=2.0, image_px=1024)
NUKE = RadarCalibration(
    map_name="de_nuke", pos_x=-3453.0, pos_y=2887.0, scale=7.0, lower_altitude_max=-495.0
)
ANUBIS = RadarCalibration(map_name="de_anubis", pos_x=-2796.0, pos_y=3328.0, scale=5.22)


def test_game_to_pixel_origin_and_corners() -> None:
    assert game_to_pixel(CAL, -1024.0, 1024.0) == (0.0, 0.0)
    assert game_to_pixel(CAL, 0.0, 0.0) == (512.0, 512.0)
    assert game_to_pixel(CAL, 1024.0, -1024.0) == (1024.0, 1024.0)


def test_game_to_norm_is_pixel_over_image_px() -> None:
    assert game_to_norm(CAL, 0.0, 0.0) == (0.5, 0.5)
    assert game_to_norm(CAL, -1024.0, 1024.0) == (0.0, 0.0)
    assert game_to_norm(CAL, 512.0, -512.0) == (0.75, 0.75)


def test_game_to_norm_matches_real_anubis_bomb_plant() -> None:
    # Verified against data/lake/07d1db1b3671d660/bomb.parquet round 1 plant.
    u, v = game_to_norm(ANUBIS, 1283.711304, 1873.848755)
    assert u == pytest.approx(0.7632362, abs=1e-6)
    assert v == pytest.approx(0.2720440, abs=1e-6)


def test_pixel_to_game_round_trips() -> None:
    for x, y in [(0.0, 0.0), (-800.5, 733.25), (1024.0, -1024.0)]:
        u, v = game_to_pixel(CAL, x, y)
        rx, ry = pixel_to_game(CAL, u, v)
        assert rx == pytest.approx(x)
        assert ry == pytest.approx(y)


def test_norm_exprs_match_scalar_form() -> None:
    df = pl.DataFrame({"X": [0.0, 512.0, -1024.0], "Y": [0.0, -512.0, 1024.0]})
    got = df.select(norm_x_expr(CAL).alias("u"), norm_y_expr(CAL).alias("v"))
    assert got["u"].to_list() == [0.5, 0.75, 0.0]
    assert got["v"].to_list() == [0.5, 0.75, 0.0]


def test_norm_exprs_accept_alternate_columns() -> None:
    df = pl.DataFrame({"victim_X": [0.0], "victim_Y": [0.0]})
    got = df.select(
        norm_x_expr(CAL, "victim_X").alias("u"), norm_y_expr(CAL, "victim_Y").alias("v")
    )
    assert got.row(0) == (0.5, 0.5)


def test_level_classification_single_and_multi_level() -> None:
    assert is_lower_level(CAL, -9999.0) is False  # no lower_altitude_max
    assert is_lower_level(NUKE, -495.0) is True  # boundary is inclusive
    assert is_lower_level(NUKE, -494.9) is False
    df = pl.DataFrame({"Z": [-600.0, 0.0]})
    assert df.select(level_expr(NUKE))["Z"].to_list() == ["lower", "default"]
    assert df.select(level_expr(CAL))["Z"].to_list() == ["default", "default"]


def test_normalize_side_handles_every_lake_spelling() -> None:
    assert normalize_side("TERRORIST") == "T"
    assert normalize_side("t") == "T"
    assert normalize_side("T") == "T"
    assert normalize_side("ct") == "CT"
    assert normalize_side("CT") == "CT"
    assert normalize_side(None) is None
    assert normalize_side("") is None
    assert normalize_side("Spectator") is None


def test_side_expr_matches_scalar_form() -> None:
    df = pl.DataFrame({"team_name": ["CT", "TERRORIST", None, "weird"]})
    got = df.select(side_expr("team_name"))["team_name"].to_list()
    assert got == ["CT", "T", None, None]
