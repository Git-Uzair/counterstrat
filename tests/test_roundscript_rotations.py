"""Rotation detection + death contexts (2026-09-05 advanced-analytics plan Task 1).

A RotationEvent is heuristic correlation: a player who had held a zone >= 4s
when a trigger fired and left within 8s. Latency comes from the 16 Hz tick
positions (first sustained displacement from the hold), and the nearest
earlier trigger claims the response. Death contexts (victim_moving /
victim_preaim_off_deg / victim_weapon) are folded into KillEvents at
serialize time from the same tick frame.
"""

import polars as pl
import pytest

from counterstrat.mapcard.lexicon import build_lexicon
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.models import (
    KillEvent,
    PlantEvent,
    RotationEvent,
    RoundScript,
    UtilEvent,
)
from counterstrat.roundscript.movement import detect_rotations, zone_stints
from counterstrat.roundscript.serialize import serialize_round

# Layout: "anchor" (CT) holds SiteB at (0,0); at move_start he sprints east and
# crosses into Mid (x >= 800) half a second later. "midguy" (CT) holds Mid all
# round; "raider" (T) idles in TSpawn (optionally entering Connector).


def _ticks(
    move_start: float | None = 11.5,
    seconds: float = 30.0,
    raider_moves_at: float | None = None,
) -> pl.DataFrame:
    rows = []
    for i in range(int(seconds * 16)):
        clock = i / 16.0
        x = (
            100.0 + (clock - move_start) * 1600.0
            if move_start is not None and clock >= move_start
            else 0.0
        )
        base = {
            "round_num": 1,
            "clock_s": clock,
            "tick": 1000 + i * 4,
            "is_alive": True,
            "Z": 0.0,
            "yaw": 0.0,
        }
        rows.append(
            base
            | {
                "name": "anchor",
                "team_name": "CT",
                "last_place_name": "SiteB" if x < 800.0 else "Mid",
                "X": x,
                "Y": 0.0,
            }
        )
        rows.append(
            base
            | {
                "name": "midguy",
                "team_name": "CT",
                "last_place_name": "Mid",
                "X": 2000.0,
                "Y": 2000.0,
            }
        )
        raider_zone = (
            "Connector" if raider_moves_at is not None and clock >= raider_moves_at else "TSpawn"
        )
        rows.append(
            base
            | {
                "name": "raider",
                "team_name": "TERRORIST",
                "last_place_name": raider_zone,
                "X": 5000.0,
                "Y": 5000.0,
            }
        )
    return pl.DataFrame(rows)


def _flash(t: float, side: str = "T") -> UtilEvent:
    return UtilEvent(
        t=t, thrower="raider", side=side, nade="flash", from_zone="TSpawn", to_zone="SiteA"
    )


def _detect(ticks: pl.DataFrame, **kw) -> list[RotationEvent]:
    tracks, sides = zone_stints(ticks, round_num=1)
    return detect_rotations(tracks, sides, ticks, round_num=1, **kw)


def test_utility_near_rotation_latency_from_ticks():
    # Enemy flash pops two zones away at t=10; the anchor breaks hold at 11.5.
    evs = _detect(_ticks(), utility=[_flash(10.0)])
    assert len(evs) == 1
    ev = evs[0]
    assert ev.trigger == "utility_near"
    assert ev.player == "anchor" and ev.side == "CT"
    assert ev.from_zone == "SiteB" and ev.to_zone == "Mid"
    assert ev.t_trigger == pytest.approx(10.0)
    assert ev.latency_s == pytest.approx(1.5, abs=0.11)


def test_own_side_utility_never_triggers():
    assert _detect(_ticks(), utility=[_flash(10.0, side="CT")]) == []


def test_short_hold_is_transit_not_rotation():
    # Held only 3.5s when the flash pops: below the 4s hold precondition.
    assert _detect(_ticks(), utility=[_flash(3.5)]) == []


def test_break_outside_8s_window_is_not_a_response():
    # Flash at 5.0, zone flip at 14.0: 9s later, outside the response window.
    assert _detect(_ticks(move_start=13.5), utility=[_flash(5.0)]) == []


def test_no_next_stint_means_no_rotation():
    # Flash at 25.0; the anchor sits in Mid until the data ends - no move.
    assert _detect(_ticks(), utility=[_flash(25.0)]) == []


def test_nearest_trigger_wins_one_event_per_break():
    kill = KillEvent(
        t=9.0,
        killer="raider",
        victim="ghost",
        killer_side="T",
        zone="SiteA",
        weapon="ak47",
        headshot=False,
        traded_within_4s=False,
    )
    evs = _detect(_ticks(), kills=[kill], utility=[_flash(10.0)])
    assert len(evs) == 1  # one break, one event - the nearest trigger claims it
    assert evs[0].trigger == "utility_near" and evs[0].t_trigger == pytest.approx(10.0)


def test_first_blood_plant_and_shots_triggers():
    kill = KillEvent(
        t=10.0,
        killer="raider",
        victim="ghost",
        killer_side="T",
        zone="SiteA",
        weapon="ak47",
        headshot=False,
        traded_within_4s=False,
    )
    evs = _detect(_ticks(), kills=[kill])
    assert [e.trigger for e in evs] == ["first_blood"]
    assert evs[0].latency_s == pytest.approx(1.5, abs=0.11)

    plant = PlantEvent(t=10.0, site="SiteA", planter="raider", alive_t=4, alive_ct=5)
    evs = _detect(_ticks(), plant=plant)
    assert [e.trigger for e in evs] == ["plant"]

    evs = _detect(_ticks(), first_shot_t=10.0)
    assert [e.trigger for e in evs] == ["shots"]


def test_visible_contact_needs_the_sightline_matrix():
    sightlines = [
        {"from": "SiteB", "to": "Connector", "n": 3, "median_dist": 20.0, "range": "medium"}
    ]
    evs = _detect(_ticks(raider_moves_at=8.0), sightlines=sightlines)
    assert [e.trigger for e in evs] == ["visible_contact"]
    ev = evs[0]
    assert ev.t_trigger == pytest.approx(8.0)
    assert ev.latency_s == pytest.approx(3.5, abs=0.11)

    # No matrix -> the trigger cannot exist.
    assert _detect(_ticks(raider_moves_at=8.0)) == []


def test_empty_inputs_no_crash():
    assert _detect(_ticks()) == []
    assert detect_rotations({}, {}, pl.DataFrame(), round_num=1) == []


# --- Death contexts through serialize_round ---


def _death_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(ticks, rounds, kills): vic stands still 40 deg off the killer; runner sprints."""
    rows = []
    for i in range(220):  # ticks 1000..1876, clock 0..13.7
        tick = 1000 + i * 4
        clock = (tick - 1000) / 64.0
        base = {"round_num": 1, "clock_s": clock, "tick": tick, "is_alive": True, "Z": 0.0}
        rows.append(
            base
            | {
                "name": "kil",
                "team_name": "CT",
                "last_place_name": "Mid",
                "X": 0.0,
                "Y": 0.0,
                "yaw": 0.0,
                "active_weapon_name": "M4A4",
            }
        )
        rows.append(
            base
            | {
                "name": "vic",
                "team_name": "TERRORIST",
                "last_place_name": "SiteB",
                "X": 1000.0,
                "Y": 0.0,
                "yaw": 140.0,  # bearing to killer is 180 -> 40 deg off
                "active_weapon_name": "AK-47",
            }
        )
        rows.append(
            base
            | {
                "name": "runner",
                "team_name": "TERRORIST",
                "last_place_name": "TSpawn",
                "X": 5000.0 + i * 40.0,  # 640 u/s - sprinting
                "Y": 5000.0,
                "yaw": 0.0,
                "active_weapon_name": "Glock-18",
            }
        )
    ticks = pl.DataFrame(rows)
    rounds = pl.DataFrame(
        {
            "round_num": [1],
            "start": [0],
            "freeze_end": [1000],
            "end": [1900],
            "winner": ["CT"],
            "reason": ["t_killed"],
        }
    )
    kills = pl.DataFrame(
        {
            "round_num": [1, 1],
            "tick": [1800, 1808],
            "attacker_name": ["kil", "kil"],
            "attacker_side": ["CT", "CT"],
            "attacker_place": ["Mid", "Mid"],
            "attacker_X": [0.0, 0.0],
            "attacker_Y": [0.0, 0.0],
            "attacker_Z": [0.0, 0.0],
            "victim_name": ["vic", "runner"],
            "victim_X": [1000.0, 5000.0 + 200 * 40.0],
            "victim_Y": [0.0, 5000.0],
            "victim_Z": [0.0, 0.0],
            "weapon": ["m4a1", "m4a1"],
            "headshot": [False, False],
        }
    )
    return ticks, rounds, kills


def _fake_lake(tmp_path):
    from counterstrat.lake.extract import LakePaths

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


def test_death_contexts_on_kills(tmp_path):
    ticks, rounds, kills = _death_frames()
    lex = build_lexicon("de_anubis", ["Mid", "SiteB", "TSpawn"])
    script = serialize_round(
        lake=_fake_lake(tmp_path),
        mapper=ZoneMapper.fit(ticks),
        lex=lex,
        card_checksum="chk",
        round_num=1,
        ticks_df=ticks,
        rounds_df=rounds,
        kills_df=kills,
        bomb_df=pl.DataFrame(),
        utils=[],
        shots_df=pl.DataFrame(),
    )
    k_vic = next(k for k in script.kills if k.victim == "vic")
    assert k_vic.victim_moving is False
    assert k_vic.victim_preaim_off_deg == pytest.approx(40.0, abs=1.0)
    assert k_vic.victim_weapon == "AK-47"
    k_run = next(k for k in script.kills if k.victim == "runner")
    assert k_run.victim_moving is True
    assert k_run.victim_weapon == "Glock-18"


def test_shots_trigger_flows_through_serialize(tmp_path):
    ticks = _ticks()
    rounds = pl.DataFrame(
        {
            "round_num": [1],
            "start": [0],
            "freeze_end": [1000],
            "end": [3000],
            "winner": ["CT"],
            "reason": ["t_killed"],
        }
    )
    shots = pl.DataFrame({"round_num": [1], "tick": [1640]})  # (1640-1000)/64 = 10.0s
    lex = build_lexicon("de_anubis", ["SiteB", "Mid", "TSpawn", "Connector"])
    script = serialize_round(
        lake=_fake_lake(tmp_path),
        mapper=ZoneMapper.fit(ticks),
        lex=lex,
        card_checksum="chk",
        round_num=1,
        ticks_df=ticks,
        rounds_df=rounds,
        kills_df=pl.DataFrame(),
        bomb_df=pl.DataFrame(),
        utils=[],
        shots_df=shots,
    )
    assert [e.trigger for e in script.rotations] == ["shots"]
    assert script.rotations[0].player == "anchor"
    assert script.rotations[0].latency_s == pytest.approx(1.5, abs=0.11)


# --- Schema compatibility ---


def test_pre_v3_script_json_loads_with_empty_new_fields():
    old = {
        "match_id": "m1",
        "map_name": "de_anubis",
        "card_checksum": "chk",
        "round_num": 1,
        "score_t": 0,
        "score_ct": 0,
        "t_team_key": "abc",
        "ct_team_key": "xyz",
        "economy": {},
        "beats": [],
        "kills": [
            {
                "t": 10.0,
                "killer": "a",
                "victim": "b",
                "killer_side": "T",
                "zone": "Mid",
                "weapon": "ak47",
                "headshot": False,
                "traded_within_4s": False,
            }
        ],
        "utility": [
            {
                "t": 5.0,
                "thrower": "a",
                "side": "T",
                "nade": "flash",
                "from_zone": "TSpawn",
                "to_zone": "Mid",
            }
        ],
        "plant": None,
        "first_contact": None,
        "winner": "T",
        "reason": "ct_killed",
        "clock_used_s": 60.0,
    }
    s = RoundScript.model_validate(old)
    assert s.rotations == []
    k = s.kills[0]
    assert k.victim_moving is None and k.victim_preaim_off_deg is None and k.victim_weapon is None
    u = s.utility[0]
    assert u.enemy_blind_s is None and u.team_blind_s is None
    assert u.damage is None and u.kills_through is None
    # And the enriched schema round-trips through its own JSON.
    reloaded = RoundScript.model_validate_json(s.to_json())
    assert reloaded == s
