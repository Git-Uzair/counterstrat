"""Death profile miner tests (2026-09-05 advanced-analytics plan Task 2)."""

from typing import Any

import pytest

from counterstrat.mining.deaths import build_death_profiles
from counterstrat.roundscript.models import KillEvent, RoundScript

TEAM = "abc"
OPP = "xyz"


def _kill(victim: str, **overrides) -> KillEvent:
    base: dict[str, Any] = {
        "t": 20.0,
        "killer": "e1",
        "victim": victim,
        "killer_side": "T",
        "zone": "Middle",
        "weapon": "ak47",
        "headshot": False,
        "traded_within_4s": False,
    }
    base.update(overrides)
    return KillEvent(**base)


def _script(round_num: int, **overrides) -> RoundScript:
    base: dict[str, Any] = {
        "match_id": "m1",
        "map_name": "de_anubis",
        "card_checksum": "chk",
        "round_num": round_num,
        "score_t": 0,
        "score_ct": 0,
        "t_team_key": OPP,
        "ct_team_key": TEAM,
        "economy": {},
        "beats": [],
        "kills": [],
        "utility": [],
        "plant": None,
        "first_contact": None,
        "winner": "T",
        "reason": "x",
        "clock_used_s": 60.0,
        "sides": {"v1": "CT", "v2": "CT", "e1": "T"},
    }
    base.update(overrides)
    return RoundScript(**base)


def test_death_profiles_rates_medians_weapons_and_bands():
    scripts = [
        _script(
            1,
            kills=[
                _kill(
                    "v1",
                    victim_moving=True,
                    victim_preaim_off_deg=40.0,
                    victim_weapon="AK-47",
                    distance=10.0,  # close
                ),
                _kill(
                    "v2",
                    victim_moving=False,
                    victim_preaim_off_deg=10.0,
                    victim_weapon="AK-47",
                    distance=40.0,  # long
                ),
                # Our own kill of an enemy is not one of our deaths.
                _kill("e1", killer_side="CT", victim_moving=True),
            ],
        ),
        _script(
            2,
            # Pre-upgrade kill: nothing measured - counted, never guessed.
            kills=[_kill("v1")],
        ),
    ]
    profiles = build_death_profiles(scripts, TEAM)
    assert profiles.team_key == TEAM and profiles.map_name == "de_anubis"
    by_player = {p.player: p for p in profiles.players}
    assert set(by_player) == {"v1", "v2"}

    v1 = by_player["v1"]
    assert v1.n == 2
    assert v1.moving_n == 1 and v1.moving_rate == 1.0
    assert v1.preaim_n == 1 and v1.median_preaim_off_deg == pytest.approx(40.0)
    assert v1.weapons == {"AK-47": 1}
    assert [(b.band, b.n) for b in v1.by_range] == [("close", 1)]
    assert v1.evidence == ["m1:1", "m1:2"]

    v2 = by_player["v2"]
    assert v2.n == 1 and v2.moving_rate == 0.0
    assert [(b.band, b.n) for b in v2.by_range] == [("long", 1)]


def test_side_fallback_without_sides_map():
    # Pre-v2 script (no sides): a kill BY the enemy side counts as our death.
    s = _script(1, sides={}, kills=[_kill("v1", victim_moving=True)])
    profiles = build_death_profiles([s], TEAM)
    assert [p.player for p in profiles.players] == ["v1"]


def test_empty_inputs_empty_report():
    assert build_death_profiles([], TEAM).players == []
    assert build_death_profiles([_script(1)], "nobody").players == []
