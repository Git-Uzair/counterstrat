import json
from pathlib import Path

import polars as pl
import pytest

from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.models import UtilEvent
from counterstrat.roundscript.utility import (
    cluster_lineups,
    subdivision_report,
    utility_events,
    utility_events_with_xyz,
)


def _two_synthetic_lineups(
    n_each: int,
    separation: float,
    nade: str = "smoke",
    to_zone: str = "Window",
) -> tuple[list[UtilEvent], pl.DataFrame]:
    """Two tight (origin, landing) lineups `separation` units apart, landing in one place."""
    events: list[UtilEvent] = []
    rows: list[dict[str, float]] = []
    for lineup in (0, 1):
        shift = lineup * separation
        for i in range(n_each):
            jitter = i * 5.0  # <= 20 units of spread: same cluster at eps=150
            events.append(
                UtilEvent(
                    t=float(10 * lineup + i),
                    thrower=f"p{lineup}",
                    side="T",
                    nade=nade,
                    from_zone="TSpawn" if lineup == 0 else "Middle",
                    to_zone=to_zone,
                )
            )
            rows.append(
                {
                    "origin_x": shift + jitter,
                    "origin_y": shift + jitter,
                    "origin_z": 0.0,
                    "landing_x": 2000.0 + shift + jitter,
                    "landing_y": 2000.0 + shift + jitter,
                    "landing_z": 0.0,
                }
            )
    return events, pl.DataFrame(rows)


def _mini_lake(tmp_path: Path):
    """One flash: 104 moving trajectory rows (1.61s flight), then a 5s
    stationary tail of repeated rows - the demoparser artifact that made
    every flash/HE look seconds late."""
    from counterstrat.lake.extract import LakePaths

    moving = [
        {
            "entity_id": 7,
            "grenade_type": "CFlashbangProjectile",
            "thrower": "donk",
            "thrower_steamid": 111,
            "tick": 1640 + i,
            "X": 100.0 + 10.0 * i,
            "Y": 100.0 + 10.0 * i,
            "Z": 64.0,
            "round_num": 5,
        }
        for i in range(104)  # last moving tick = 1743 -> pop at 11.61s
    ]
    final = moving[-1]
    tail = [
        {**final, "tick": 1743 + j}
        for j in range(1, 322)  # ~5s of stationary post-pop rows
    ]
    grenades = pl.DataFrame(moving + tail)
    rounds = pl.DataFrame({"round_num": [5], "freeze_end": [1000], "start": [0], "end": [9000]})
    rosters = pl.DataFrame(
        {
            "round_num": [5, 5],
            "side": ["TERRORIST", "CT"],
            "team_key": ["abc", "xyz"],
            "steamids": [[111, 444], [222, 333]],
            "clan_name": ["x", "y"],
        }
    )
    # Two enemies (2.5s + 1.8s) and one teammate (0.4s) blinded at the pop:
    # the split convention is SUM per side (MAX would hide multi-blinds).
    blinds = pl.DataFrame(
        {
            "attacker_steamid": [111, 111, 111],
            "user_steamid": [222, 333, 444],
            "user_name": ["victim1", "victim2", "mate1"],
            "blind_duration": [2.5, 1.8, 0.4],
            "tick": [1744, 1744, 1745],  # the real pop moment
        }
    )
    grenades.write_parquet(tmp_path / "grenades.parquet")
    rounds.write_parquet(tmp_path / "rounds.parquet")
    rosters.write_parquet(tmp_path / "rosters.parquet")
    blinds.write_parquet(tmp_path / "player_blind.parquet")
    names = [
        "rounds",
        "kills",
        "damages",
        "shots",
        "grenades",
        "smokes",
        "infernos",
        "bomb",
        "item_purchase",
        "ticks",
        "rosters",
        "player_blind",
    ]
    return LakePaths(root=str(tmp_path), **{n: str(tmp_path / f"{n}.parquet") for n in names})


def _mini_mapper() -> ZoneMapper:
    return ZoneMapper.fit(
        pl.DataFrame(
            {
                "X": [100.0, 150.0, 1100.0, 1150.0],
                "Y": [100.0, 150.0, 1100.0, 1150.0],
                "Z": [64.0] * 4,
                "last_place_name": ["TSpawn", "TSpawn", "Window", "Window"],
            }
        )
    )


def test_projectile_pop_is_last_moving_row_not_trajectory_tail(tmp_path):
    """The flash's time is its POP (last moving row), never the stationary
    tail - and the blind join must land on that same moment."""
    lake = _mini_lake(tmp_path)
    evs = utility_events(lake, _mini_mapper(), 5)
    assert len(evs) == 1
    e = evs[0]
    assert e.nade == "flash" and e.thrower == "donk" and e.side == "T"
    # Pop at tick 1743 -> (1743 - 1000) / 64 = 11.609s; the tail would say 16.6s.
    assert abs(e.t - (1743 - 1000) / 64.0) < 1e-6
    # The victim blinded at the pop is attributed to this flash.
    assert e.blinded and e.blinded[0][0] == "victim1"


def test_cluster_lineups_two_smokes():
    evs, xyz = _two_synthetic_lineups(n_each=5, separation=800.0)
    names = cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    labelled = {e.lineup_id for e in evs if e.lineup_id}
    assert len(labelled) == 2  # two distinct lineup ids
    assert all("-" in x for x in labelled)
    # same landing place -> ordinals disambiguate, size tie broken by centroid order
    assert labelled == {"Window-S1", "Window-S2"}
    assert set(names.values()) == labelled
    assert [e.lineup_id for e in evs[:5]] == ["Window-S1"] * 5
    assert [e.lineup_id for e in evs[5:]] == ["Window-S2"] * 5


def test_cluster_lineups_noise_unlabelled():
    evs, xyz = _two_synthetic_lineups(n_each=1, separation=800.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    assert all(e.lineup_id is None for e in evs)


def test_cluster_lineups_separates_nade_types():
    smokes, smoke_xyz = _two_synthetic_lineups(n_each=3, separation=0.0)
    flashes, flash_xyz = _two_synthetic_lineups(n_each=3, separation=0.0, nade="flash")
    evs = smokes + flashes
    names = cluster_lineups(evs, pl.concat([smoke_xyz, flash_xyz]), eps=150.0, min_samples=3)
    # identical coordinates, different nade -> two clusters, distinct initials
    assert set(names.values()) == {"Window-S1", "Window-F1"}


def test_cluster_lineups_empty_and_mismatched():
    assert cluster_lineups([], pl.DataFrame()) == {}
    evs, xyz = _two_synthetic_lineups(n_each=3, separation=800.0)
    with pytest.raises(ValueError, match="rows"):
        cluster_lineups(evs, xyz.head(2))


def test_cluster_lineups_resets_previous_ids():
    evs, xyz = _two_synthetic_lineups(n_each=1, separation=800.0)
    for e in evs:
        e.lineup_id = "stale"
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    assert all(e.lineup_id is None for e in evs)


def test_subdivision_report_flags_collapsed_place(tmp_path):
    evs, xyz = _two_synthetic_lineups(n_each=5, separation=800.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    mapper = ZoneMapper.fit(
        pl.DataFrame(
            {
                "X": [0.0] * 50 + [1000.0] * 50,
                "Y": [0.0] * 100,
                "Z": [0.0] * 100,
                "last_place_name": ["Window"] * 50 + ["Middle"] * 50,
            }
        )
    )
    out = tmp_path / "subdivision_report.json"
    report = subdivision_report(evs, mapper, out_path=out)
    assert report == {"Window": 2}
    assert json.loads(out.read_text()) == {"Window": 2}


def test_subdivision_report_single_cluster_place_not_flagged():
    evs, xyz = _two_synthetic_lineups(n_each=3, separation=0.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    mapper = ZoneMapper.fit(
        pl.DataFrame(
            {
                "X": [0.0] * 50,
                "Y": [0.0] * 50,
                "Z": [0.0] * 50,
                "last_place_name": ["Window"] * 50,
            }
        )
    )
    assert subdivision_report(evs, mapper) == {}


def test_bloom_rows_drop_orphan_thrower(tmp_path):
    """A smoke/molly with an unattributable thrower (all thrower_* null - seen
    on a real FACEIT dust2 demo, round 20) must be dropped, not crash ingest
    with a KNN "Input X contains NaN"."""
    lake = _mini_lake(tmp_path)
    bloom_schema = {
        "entity_id": [11, 12],
        "start_tick": [1100, 1200],
        "end_tick": [2100, 2200],
        "thrower_X": [105.0, None],
        "thrower_Y": [105.0, None],
        "thrower_Z": [64.0, None],
        "thrower_health": [100, None],
        "thrower_place": ["TSpawn", None],
        "thrower_name": ["donk", None],
        "thrower_steamid": ["111", None],
        "thrower_side": ["TERRORIST", None],
        "X": [1120.0, 1130.0],
        "Y": [1120.0, 1130.0],
        "Z": [64.0, 64.0],
        "round_num": [5, 5],
    }
    pl.DataFrame(bloom_schema).write_parquet(tmp_path / "infernos.parquet")

    evs, xyz = utility_events_with_xyz(lake, _mini_mapper(), 5)

    mollies = [e for e in evs if e.nade == "molly"]
    assert len(mollies) == 1  # the orphan row is gone, the attributed one kept
    assert mollies[0].thrower == "donk"
    assert mollies[0].to_zone in {"TSpawn", "Window"}  # mapped, not NaN-poisoned
    assert xyz.height == len(evs)  # frames stay aligned for clustering


# --- Effect fields (2026-09-05 advanced-analytics plan Task 1) ---


def _add_he(tmp_path: Path) -> None:
    """Append an HE trajectory (entity 8, det tick 1660) to the mini lake."""
    he = pl.DataFrame(
        [
            {
                "entity_id": 8,
                "grenade_type": "CHEGrenadeProjectile",
                "thrower": "donk",
                "thrower_steamid": 111,
                "tick": 1650 + i,
                "X": 200.0 + 90.0 * i,
                "Y": 200.0 + 90.0 * i,
                "Z": 64.0,
                "round_num": 5,
            }
            for i in range(11)  # last moving tick = 1660 -> the pop
        ]
    )
    old = pl.read_parquet(tmp_path / "grenades.parquet")
    pl.concat([old, he]).write_parquet(tmp_path / "grenades.parquet")


def test_flash_blind_split_sums_per_side(tmp_path):
    lake = _mini_lake(tmp_path)
    evs = utility_events(lake, _mini_mapper(), 5)
    flash = next(e for e in evs if e.nade == "flash")
    assert flash.enemy_blind_s == pytest.approx(4.3)  # 2.5 + 1.8, SUM not MAX
    assert flash.team_blind_s == pytest.approx(0.4)
    assert flash.damage is None and flash.kills_through is None  # wrong nade type


def test_blind_fields_none_without_blind_table(tmp_path):
    lake = _mini_lake(tmp_path)
    (tmp_path / "player_blind.parquet").unlink()
    evs = utility_events(lake, _mini_mapper(), 5)
    flash = next(e for e in evs if e.nade == "flash")
    assert flash.blinded == []
    assert flash.enemy_blind_s is None and flash.team_blind_s is None


def test_he_damage_attributed_within_pop_window(tmp_path):
    lake = _mini_lake(tmp_path)
    _add_he(tmp_path)
    pl.DataFrame(
        {
            "round_num": [5, 5, 5],
            "tick": [1700, 1700, 3000],  # in-window HE, rifle noise, late HE
            "weapon": ["hegrenade", "ak47", "hegrenade"],
            "attacker_steamid": [111, 111, 111],
            "dmg_health": [34, 27, 50],
        }
    ).write_parquet(tmp_path / "damages.parquet")
    evs = utility_events(lake, _mini_mapper(), 5)
    he = next(e for e in evs if e.nade == "he")
    assert he.damage == 34
    assert he.kills_through is None


def test_effect_fields_none_when_tables_missing(tmp_path):
    lake = _mini_lake(tmp_path)
    _add_he(tmp_path)  # no damages.parquet, no kills.parquet
    evs = utility_events(lake, _mini_mapper(), 5)
    he = next(e for e in evs if e.nade == "he")
    assert he.damage is None


def test_molly_damage_over_bloom_window(tmp_path):
    lake = _mini_lake(tmp_path)
    pl.DataFrame(
        {
            "entity_id": [12],
            "start_tick": [1200],
            "end_tick": [2200],
            "thrower_X": [105.0],
            "thrower_Y": [105.0],
            "thrower_Z": [64.0],
            "thrower_place": ["TSpawn"],
            "thrower_name": ["donk"],
            "thrower_steamid": ["111"],
            "thrower_side": ["TERRORIST"],
            "X": [1120.0],
            "Y": [1120.0],
            "Z": [64.0],
            "round_num": [5],
        }
    ).write_parquet(tmp_path / "infernos.parquet")
    pl.DataFrame(
        {
            "round_num": [5, 5, 5],
            "tick": [1300, 1400, 5000],  # two burn ticks in-window, one after
            "weapon": ["inferno", "molotov", "inferno"],
            "attacker_steamid": [111, 111, 111],
            "dmg_health": [12, 8, 40],
        }
    ).write_parquet(tmp_path / "damages.parquet")
    evs = utility_events(lake, _mini_mapper(), 5)
    molly = next(e for e in evs if e.nade == "molly")
    assert molly.damage == 20


def test_smoke_kills_through_crossing_segment(tmp_path):
    lake = _mini_lake(tmp_path)
    pl.DataFrame(
        {
            "entity_id": [21],
            "start_tick": [1100],
            "end_tick": [2100],
            "thrower_X": [105.0],
            "thrower_Y": [105.0],
            "thrower_Z": [64.0],
            "thrower_place": ["TSpawn"],
            "thrower_name": ["donk"],
            "thrower_steamid": ["111"],
            "thrower_side": ["TERRORIST"],
            "X": [1120.0],
            "Y": [1120.0],
            "Z": [64.0],
            "round_num": [5],
        }
    ).write_parquet(tmp_path / "smokes.parquet")
    pl.DataFrame(
        {
            "round_num": [5, 5, 5, 5],
            "tick": [1500, 1500, 5000, 1500],
            "thrusmoke": [True, False, True, True],
            "attacker_X": [100.0, 100.0, 100.0, 100.0],
            "attacker_Y": [100.0, 100.0, 100.0, 100.0],
            "victim_X": [2000.0, 2000.0, 2000.0, 100.0],
            "victim_Y": [2000.0, 2000.0, 2000.0, 5000.0],
            "attacker_name": ["donk"] * 4,
            "victim_name": ["v"] * 4,
        }
    ).write_parquet(tmp_path / "kills.parquet")
    evs = utility_events(lake, _mini_mapper(), 5)
    smoke = next(e for e in evs if e.nade == "smoke")
    # Only the in-window thrusmoke kill whose line crosses the bloom counts:
    # not the no-flag kill, not the late kill, not the far-away segment.
    assert smoke.kills_through == 1
    assert smoke.damage is None


def test_smoke_kills_through_none_without_kills_table(tmp_path):
    lake = _mini_lake(tmp_path)
    pl.DataFrame(
        {
            "entity_id": [21],
            "start_tick": [1100],
            "end_tick": [2100],
            "thrower_X": [105.0],
            "thrower_Y": [105.0],
            "thrower_Z": [64.0],
            "thrower_place": ["TSpawn"],
            "thrower_name": ["donk"],
            "thrower_steamid": ["111"],
            "thrower_side": ["TERRORIST"],
            "X": [1120.0],
            "Y": [1120.0],
            "Z": [64.0],
            "round_num": [5],
        }
    ).write_parquet(tmp_path / "smokes.parquet")
    evs = utility_events(lake, _mini_mapper(), 5)
    smoke = next(e for e in evs if e.nade == "smoke")
    assert smoke.kills_through is None


@pytest.fixture(scope="module")
def anubis_utility(anubis_lake, anubis_assets):
    ticks = pl.read_parquet(anubis_lake.ticks)
    mapper = ZoneMapper.fit(ticks)
    lex = build_lexicon(
        "de_anubis",
        unique_places(parse_places(anubis_assets.vents)),
        get_default_overlay_path("de_anubis"),
    )
    rounds = pl.read_parquet(anubis_lake.rounds)["round_num"].to_list()
    events: list[UtilEvent] = []
    frames: list[pl.DataFrame] = []
    for rn in rounds:
        evs, xyz = utility_events_with_xyz(anubis_lake, mapper, int(rn))
        assert [e.t for e in evs] == sorted(e.t for e in evs)
        events.extend(evs)
        frames.append(xyz)
    return {"events": events, "raw_xyz": pl.concat(frames), "mapper": mapper, "lex": lex}


@pytest.mark.demo
def test_utility_events_speak_lexicon(anubis_utility):
    events = anubis_utility["events"]
    lex = anubis_utility["lex"]
    # 173 smokes + 131 infernos verified in the lake, plus flashes/HEs
    assert len(events) >= 304
    assert {e.nade for e in events} <= {"smoke", "molly", "flash", "he"}
    assert all(lex.is_valid_zone(e.to_zone) for e in events)
    assert all(lex.is_valid_zone(e.from_zone) for e in events)
    assert {e.side for e in events} == {"T", "CT"}
    assert all(e.thrower for e in events)
    # blind durations land on flashes only
    flashes = [e for e in events if e.nade == "flash"]
    assert any(e.blinded for e in flashes)
    assert all(not e.blinded for e in events if e.nade != "flash")
    assert all(d > 0.0 for e in flashes for _, d in e.blinded)


@pytest.mark.demo
def test_utility_events_single_round_matches_helper(anubis_lake, anubis_utility):
    mapper = anubis_utility["mapper"]
    events, xyz = utility_events_with_xyz(anubis_lake, mapper, 1)
    assert [e.model_dump() for e in utility_events(anubis_lake, mapper, 1)] == [
        e.model_dump() for e in events
    ]
    assert xyz.height == len(events)
    assert any(e.nade == "smoke" for e in events)


@pytest.mark.demo
def test_cluster_lineups_and_subdivision_on_demo(anubis_utility, tmp_path):
    events = anubis_utility["events"]
    names = cluster_lineups(events, anubis_utility["raw_xyz"], min_samples=3)
    assert len(names) >= 1
    assert all(n.count("-") == 1 for n in names.values())
    assert any(e.lineup_id is not None for e in events)

    # Never write into the real data/ tree: tests must not touch user data.
    out_json = tmp_path / "subdivision_report.json"
    report = subdivision_report(events, anubis_utility["mapper"], out_path=out_json)
    assert out_json.exists()
    assert json.loads(out_json.read_text()) == report
    assert all(v >= 2 for v in report.values())
