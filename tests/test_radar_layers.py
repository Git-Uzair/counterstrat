from pathlib import Path

import pytest
from conftest import build_radar_lake, radar_test_cal

from counterstrat.radar.layers import (
    LayerFilters,
    build_layers,
    build_team_scope,
    lake_frames,
)


@pytest.fixture
def frames(tmp_path: Path):
    build_radar_lake(tmp_path / "lake")
    return lake_frames(tmp_path / "lake", ["m1"])


@pytest.fixture
def cal():
    return radar_test_cal()


def test_lake_frames_reports_missing_tables_as_none(tmp_path: Path) -> None:
    got = lake_frames(tmp_path / "empty", ["m1"])
    assert set(got) == {"rosters", "ticks", "kills", "grenades", "bomb"}
    assert all(v is None for v in got.values())


def test_team_scope_tracks_side_swap(frames) -> None:
    scope = build_team_scope(frames["rosters"], "teamA", LayerFilters())
    rows = sorted(scope.rounds.iter_rows(named=True), key=lambda r: r["round_num"])
    assert [(r["round_num"], r["side"]) for r in rows] == [(1, "CT"), (2, "T")]
    assert sorted(scope.members["steamid"].unique().to_list()) == [1, 2]


def test_team_scope_side_filter_keeps_one_round(frames) -> None:
    scope = build_team_scope(frames["rosters"], "teamA", LayerFilters(side="CT"))
    assert scope.rounds.height == 1
    assert scope.rounds.row(0, named=True)["round_num"] == 1


def test_spread_indices_covers_both_halves() -> None:
    """Trail sampling must span the match, not take the first N rounds."""
    from counterstrat.radar.layers import _spread_indices

    idx = _spread_indices(24, 4)
    assert len(idx) == 4
    assert idx[0] == 0 and idx[-1] == 23  # first and last round included
    assert any(i >= 12 for i in idx), "second half must be represented"
    assert _spread_indices(2, 4) == [0, 1]  # fewer rounds than requested
    assert _spread_indices(5, 1) == [0]
    assert _spread_indices(0, 4) == []


def test_trails_spread_across_rounds(frames, cal) -> None:
    """With trail_rounds=1 both fixture rounds compete; the spread picks round 1,
    and trail_rounds=None returns every round's trails."""
    one = build_layers(frames, cal, "teamA", LayerFilters(trail_rounds=1))["layers"]["trails"]
    assert {t["round_num"] for t in one} == {1}
    both = build_layers(frames, cal, "teamA", LayerFilters(trail_rounds=None))["layers"]["trails"]
    assert {t["round_num"] for t in both} == {1, 2}


def test_team_scope_accepts_cluster_key_set(frames) -> None:
    """A cluster's key set unions rounds across lineup keys (team identity merge)."""
    merged = build_team_scope(frames["rosters"], ["teamA", "teamB"], LayerFilters())
    solo = build_team_scope(frames["rosters"], "teamA", LayerFilters())
    assert merged.rounds.height == 2 * solo.rounds.height
    assert merged.team_key == "teamA"  # canonical = first sorted key


def test_players_filter_scopes_member_layers(frames, cal) -> None:
    """Task 9: a players filter isolates one player's trails/heatmap/utility."""
    payload = build_layers(frames, cal, "teamA", LayerFilters(players=[1]))
    trails = payload["layers"]["trails"]
    assert trails and all(t["steamid"] == "1" for t in trails)

    # Player 2's samples vanish from the heatmap; duels (round-scoped) remain.
    solo = payload["layers"]["heatmap"]["samples"]
    full = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["heatmap"]["samples"]
    assert solo == full // 2
    assert payload["layers"]["duels"], "duels stay round-scoped, not player-scoped"

    # The teamA smoke was thrown by player 1: still present. Filtering to
    # player 2 drops it.
    assert any(u["steamid"] == "1" for u in payload["layers"]["utility"])
    p2 = build_layers(frames, cal, "teamA", LayerFilters(players=[2]))
    assert all(u["steamid"] == "2" for u in p2["layers"]["utility"])


def test_heatmap_counts_cells_and_excludes_dead_samples(frames, cal) -> None:
    payload = build_layers(frames, cal, "teamA", LayerFilters(grid=8))
    hm = payload["layers"]["heatmap"]
    cells = {(gx, gy): n for gx, gy, n in hm["cells"]}
    assert hm["grid"] == 8
    assert hm["samples"] == 16  # 2 rounds x 2 players x 4 alive samples
    assert len(cells) == 7
    assert cells[(4, 4)] == 4  # both players start dead centre, both rounds
    assert cells[(7, 4)] == 2  # NOT 3: the is_alive=False sample is excluded
    assert hm["max"] == 4


def test_heatmap_scoped_to_team_not_opponent(frames, cal) -> None:
    a = build_layers(frames, cal, "teamA", LayerFilters(grid=8))["layers"]["heatmap"]
    b = build_layers(frames, cal, "teamB", LayerFilters(grid=8))["layers"]["heatmap"]
    a_cells = {(gx, gy) for gx, gy, _ in a["cells"]}
    b_cells = {(gx, gy) for gx, gy, _ in b["cells"]}
    assert (7, 4) in a_cells and (7, 4) not in b_cells  # teamA walks +X
    assert (1, 4) in b_cells and (1, 4) not in a_cells  # teamB walks -X


def test_heatmap_side_filter_halves_samples(frames, cal) -> None:
    hm = build_layers(frames, cal, "teamA", LayerFilters(grid=8, side="CT"))
    assert hm["layers"]["heatmap"]["samples"] == 8
    assert hm["layers"]["heatmap"]["max"] == 2


def test_trails_one_per_player_round_with_normalized_points(frames, cal) -> None:
    trails = build_layers(frames, cal, "teamA", LayerFilters(stride=1, trail_rounds=None))[
        "layers"
    ]["trails"]
    assert len(trails) == 4  # 2 rounds x 2 players
    assert all(len(t["points"]) == 4 for t in trails)
    first = next(t for t in trails if t["round_num"] == 1 and t["steamid"] == "1")
    assert first["name"] == "pA1"
    assert first["side"] == "CT"
    assert first["points"][0] == [0.5, 0.5, 0.0]
    assert first["points"][3] == [0.875, 0.5, 3.0]
    second = next(t for t in trails if t["round_num"] == 2 and t["steamid"] == "1")
    assert second["side"] == "T"  # side swapped at half


def test_trails_stride_decimates_and_trail_rounds_caps(frames, cal) -> None:
    strided = build_layers(frames, cal, "teamA", LayerFilters(stride=2, trail_rounds=None))[
        "layers"
    ]["trails"]
    assert all(len(t["points"]) == 2 for t in strided)
    capped = build_layers(frames, cal, "teamA", LayerFilters(stride=1, trail_rounds=1))["layers"][
        "trails"
    ]
    assert len(capped) == 2
    assert {t["round_num"] for t in capped} == {1}


def test_trails_coordinates_are_exact_4_decimal_floats(frames) -> None:
    """A non-dyadic calibration (every real map) must still ship 4-decimal floats.

    radar_test_cal() hides Float32 rounding error because its coordinates are all
    dyadic fractions; de_anubis' scale=5.22 is not.
    """
    from counterstrat.radar.extract import RadarCalibration

    anubis = RadarCalibration(
        map_name="de_anubis", pos_x=-2796.0, pos_y=3328.0, scale=5.22, image_px=1024
    )
    trails = build_layers(frames, anubis, "teamA", LayerFilters(stride=1, trail_rounds=None))[
        "layers"
    ]["trails"]
    assert trails
    coords = [c for t in trails for pt in t["points"] for c in pt[:2]]
    assert len(coords) == 32  # 2 rounds x 2 players x 4 samples x (u, v)
    assert [c for c in coords if round(c, 4) != c] == []
    clocks = [pt[2] for t in trails for pt in t["points"]]
    assert [c for c in clocks if round(c, 1) != c] == []


def test_trails_never_ship_negative_zero(frames) -> None:
    """Coordinates a hair below the image origin round to -0.0; the wire wants 0.0."""
    from counterstrat.radar.extract import RadarCalibration

    just_off = RadarCalibration(
        map_name="de_test", pos_x=0.01, pos_y=-0.01, scale=5.22, image_px=1024
    )
    trails = build_layers(frames, just_off, "teamA", LayerFilters(stride=1, trail_rounds=None))[
        "layers"
    ]["trails"]
    coords = [c for t in trails for pt in t["points"] for c in pt[:2]]
    assert 0.0 in coords  # the near-origin samples are present...
    assert not any(str(c) == "-0.0" for c in coords)  # ...as +0.0, not -0.0


def test_utility_keeps_only_team_projectiles(frames, cal) -> None:
    util = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["utility"]
    assert [u["kind"] for u in util] == ["smoke", "molotov"]
    smoke = util[0]
    assert (smoke["from_u"], smoke["from_v"]) == (0.25, 0.25)  # throw origin
    assert (smoke["u"], smoke["v"]) == (0.75, 0.75)  # landing
    assert smoke["thrower"] == "pA1"
    assert smoke["steamid"] == "1"
    assert util[1]["u"] == 0.375
    # teamB's flashbang and the held (non-projectile) CFlashbang are both gone.
    assert all(u["kind"] != "flash" for u in util)


def test_duels_tag_by_team_and_tolerate_null_attacker(frames, cal) -> None:
    duels = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["duels"]
    assert len(duels) == 3
    assert [d["by_team"] for d in duels] == [True, False, False]
    assert duels[0]["victim"]["u"] == 0.75
    assert duels[0]["attacker"]["u"] == 0.5
    assert duels[0]["weapon"] == "ak47" and duels[0]["headshot"] is True
    assert duels[2]["attacker"] is None  # null attacker_side/coords
    assert duels[2]["victim"]["v"] == 0.125


def test_bombs_keep_plant_and_defuse_only(frames, cal) -> None:
    bombs = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["bombs"]
    assert [b["event"] for b in bombs] == ["defuse", "plant"]
    assert bombs[0]["u"] == 0.25
    assert bombs[1]["u"] == 0.5 and bombs[1]["bombsite"] == "BombsiteA"
    assert bombs[1]["steamid"] == "1"


def test_level_filter_uses_lower_altitude_max(tmp_path: Path) -> None:
    from counterstrat.radar.extract import RadarCalibration

    build_radar_lake(tmp_path / "lake")
    frames_ = lake_frames(tmp_path / "lake", ["m1"])
    multi = RadarCalibration(
        map_name="de_test",
        pos_x=-1024.0,
        pos_y=1024.0,
        scale=2.0,
        image_px=1024,
        lower_altitude_max=100.0,  # every synthetic Z=0 is "lower"
    )
    lower = build_layers(frames_, multi, "teamA", LayerFilters(grid=8, level="lower"))
    assert lower["layers"]["heatmap"]["samples"] == 16
    assert lower["layers"]["bombs"][0]["level"] == "lower"
    default = build_layers(frames_, multi, "teamA", LayerFilters(grid=8, level="default"))
    assert default["layers"]["heatmap"]["samples"] == 0
    assert default["layers"]["bombs"] == []


def test_build_layers_on_empty_lake_returns_empty_payload(tmp_path: Path, cal) -> None:
    payload = build_layers(lake_frames(tmp_path / "nope", ["m1"]), cal, "teamA", LayerFilters())
    assert payload["rounds"] == []
    assert payload["layers"]["heatmap"]["cells"] == []
    assert payload["layers"]["trails"] == []
    assert payload["layers"]["utility"] == []
    assert payload["layers"]["duels"] == []
    assert payload["layers"]["bombs"] == []


def test_unknown_team_key_yields_empty_layers(frames, cal) -> None:
    payload = build_layers(frames, cal, "nobody", LayerFilters())
    assert payload["rounds"] == []
    assert payload["layers"]["heatmap"]["samples"] == 0
    assert payload["layers"]["trails"] == []
