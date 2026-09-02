from collections.abc import Sequence

import polars as pl
import pytest

from counterstrat.roundscript.models import KillEvent, PlantEvent
from counterstrat.roundscript.movement import movement_sentences


def _synthetic_player_walk(zones: Sequence[str | None], walking_from: int = 999) -> pl.DataFrame:
    # 4 Hz ticks (dt = 0.25s), tick step 16
    n = len(zones)
    return pl.DataFrame(
        {
            "steamid": [12345] * n,
            "name": ["p1"] * n,
            "team_name": ["TERRORIST"] * n,
            "round_num": [1] * n,
            "tick": [i * 16 for i in range(n)],
            "clock_s": [i * 0.25 for i in range(n)],
            "last_place_name": zones,
            "is_alive": [True] * n,
            "is_walking": [i >= walking_from for i in range(n)],
            "is_defusing": [False] * n,
            "in_bomb_zone": [False] * n,
        }
    )


def test_movement_dwell_merge_and_marks():
    ticks = _synthetic_player_walk(
        zones=["TSpawn"] * 8 + ["Mid"] * 2 + ["TSpawn"] * 2 + ["Mid"] * 40,
        walking_from=10,
    )
    kills = [
        KillEvent(
            t=10.0,
            killer="p1",
            victim="e1",
            killer_side="T",
            zone="Mid",
            weapon="ak47",
            headshot=True,
            traded_within_4s=False,
        )
    ]
    lines = movement_sentences(ticks, kills, round_num=1)
    assert len(lines) == 1
    s = lines[0].sentence
    assert s.startswith("p1(T): TSpawn > ")
    assert "TSpawn > Mid > TSpawn" not in s  # flicker merged
    assert "k(e1)" in s


def test_movement_marks_death_plant_defuse():
    n = 20
    ticks = pl.DataFrame(
        {
            "steamid": [12345] * n,
            "name": ["donk"] * n,
            "team_name": ["TERRORIST"] * n,
            "round_num": [1] * n,
            "tick": [i * 16 for i in range(n)],
            "clock_s": [i * 0.25 for i in range(n)],
            "last_place_name": ["TSpawn"] * 10 + ["BombsiteB"] * 10,
            "is_alive": [True] * (n - 2) + [False, False],
            "is_walking": [False] * n,
            "is_defusing": [False] * 12 + [True] * 2 + [False] * 6,
            "in_bomb_zone": [False] * 10 + [True] * 10,
        }
    )
    kills = [
        KillEvent(
            t=3.0,
            killer="donk",
            victim="b1",
            killer_side="T",
            zone="BombsiteB",
            weapon="ak47",
            headshot=True,
            traded_within_4s=False,
        )
    ]
    plant = PlantEvent(t=3.2, site="B", planter="donk", alive_t=1, alive_ct=1)
    lines = movement_sentences(ticks, kills, round_num=1, plants=plant)
    assert len(lines) == 1
    line = lines[0]
    assert line.player == "donk"
    assert line.side == "T"
    assert "k(b1)" in line.sentence
    assert "p" in line.sentence
    assert "x" in line.sentence
    assert "d" in line.sentence
    # death mark should appear at the end of the last visit
    assert line.sentence.endswith("d")


def test_movement_dwell_display_over_20s():
    # 80 ticks at 0.25s = 20.0s -> should show (20s)
    ticks = _synthetic_player_walk(zones=["Tunnel"] * 80)
    lines = movement_sentences(ticks, [], round_num=1)
    assert len(lines) == 1
    assert "Tunnel(20s)" in lines[0].sentence

    # 100 ticks at 0.25s = 25.0s -> should show (25s)
    ticks_100 = _synthetic_player_walk(zones=["Tunnel"] * 100)
    lines_100 = movement_sentences(ticks_100, [], round_num=1)
    assert "Tunnel(25s)" in lines_100[0].sentence


def test_movement_role_hint():
    # 2 teammates: p1 stays away from p2 across multiple beats
    n = 160  # 40 seconds (beats at 0, 15, 30 -> 3 beats)
    ticks = pl.DataFrame(
        {
            "steamid": [1] * n + [2] * n,
            "name": ["lurker"] * n + ["anchor"] * n,
            "team_name": ["TERRORIST"] * (2 * n),
            "round_num": [1] * (2 * n),
            "tick": [i * 16 for i in range(n)] * 2,
            "clock_s": [i * 0.25 for i in range(n)] * 2,
            "last_place_name": ["Ruins"] * n + ["ASite"] * n,
            "is_alive": [True] * (2 * n),
            "is_walking": [False] * (2 * n),
            "is_defusing": [False] * (2 * n),
            "in_bomb_zone": [False] * (2 * n),
        }
    )
    lines = movement_sentences(ticks, [], round_num=1)
    assert len(lines) == 2
    lurk_line = next(line for line in lines if line.player == "lurker")
    pack_line = next(line for line in lines if line.player == "anchor")
    # Both are alone in their zone with >= 3 beats -> lurk
    assert lurk_line.role_hint == "lurk"
    assert pack_line.role_hint == "lurk"


def test_movement_flicker_at_ends():
    # 2 ticks of Mid (< 2.0s) at start, followed by 40 ticks of TSpawn (10.0s)
    ticks_start = _synthetic_player_walk(zones=["Mid"] * 2 + ["TSpawn"] * 40)
    lines = movement_sentences(ticks_start, [], round_num=1)
    assert len(lines) == 1
    # Mid should merge into TSpawn at start
    assert lines[0].sentence == "p1(T): TSpawn"

    # 40 ticks of TSpawn followed by 2 ticks of Mid at end
    ticks_end = _synthetic_player_walk(zones=["TSpawn"] * 40 + ["Mid"] * 2)
    lines_end = movement_sentences(ticks_end, [], round_num=1)
    assert len(lines_end) == 1
    # Mid should merge into preceding TSpawn
    assert lines_end[0].sentence == "p1(T): TSpawn"


def test_movement_empty_and_unknown():
    assert movement_sentences(pl.DataFrame(), [], round_num=1) == []

    ticks_unknown = _synthetic_player_walk(zones=[None] * 20)
    lines = movement_sentences(ticks_unknown, [], round_num=1)
    assert len(lines) == 1
    assert "Unknown" in lines[0].sentence


@pytest.mark.demo
def test_movement_anubis_round1(anubis_lake):
    ticks = pl.read_parquet(anubis_lake.ticks)
    kills_df = pl.read_parquet(anubis_lake.kills).filter(pl.col("round_num") == 1)
    # Test with kills_df as DataFrame
    lines_with_df = movement_sentences(ticks, kills_df, round_num=1)
    assert len(lines_with_df) == 10
    assert any("k(" in l.sentence for l in lines_with_df)

    # Test with empty kills
    lines = movement_sentences(ticks, [], round_num=1)
    assert len(lines) == 10
    assert all(l.sentence for l in lines)
