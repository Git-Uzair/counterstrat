"""Utility ROI miner tests (2026-09-05 advanced-analytics plan Task 2)."""

from typing import Any

import pytest

from counterstrat.mining.utility_roi import build_utility_roi
from counterstrat.roundscript.models import RoundScript, UtilEvent

TEAM = "abc"
OPP = "xyz"


def _script(round_num: int, team_side: str = "T", **overrides) -> RoundScript:
    base: dict[str, Any] = {
        "match_id": "m1",
        "map_name": "de_anubis",
        "card_checksum": "chk",
        "round_num": round_num,
        "score_t": 0,
        "score_ct": 0,
        "t_team_key": TEAM if team_side == "T" else OPP,
        "ct_team_key": TEAM if team_side == "CT" else OPP,
        "economy": {},
        "beats": [],
        "kills": [],
        "utility": [],
        "plant": None,
        "first_contact": None,
        "winner": "T",
        "reason": "x",
        "clock_used_s": 60.0,
    }
    base.update(overrides)
    return RoundScript(**base)


def _flash(enemy_s: float | None, team_s: float | None, lineup: str | None = "Mid-F1") -> UtilEvent:
    return UtilEvent(
        t=10.0,
        thrower="p1",
        side="T",
        nade="flash",
        from_zone="TSpawn",
        to_zone="Middle",
        lineup_id=lineup,
        enemy_blind_s=enemy_s,
        team_blind_s=team_s,
    )


def test_roi_aggregates_per_pattern_with_evidence():
    scripts = [
        _script(
            1,
            utility=[
                _flash(2.0, 0.0),
                UtilEvent(
                    t=15.0,
                    thrower="p1",
                    side="T",
                    nade="he",
                    from_zone="TSpawn",
                    to_zone="BombsiteA",
                    damage=34,
                ),
                UtilEvent(
                    t=20.0,
                    thrower="p2",
                    side="T",
                    nade="smoke",
                    from_zone="TSpawn",
                    to_zone="Middle",
                    lineup_id="Mid-S1",
                    kills_through=1,
                ),
            ],
        ),
        _script(
            2,
            utility=[
                _flash(4.0, 1.0),
                UtilEvent(
                    t=21.0,
                    thrower="p2",
                    side="T",
                    nade="smoke",
                    from_zone="TSpawn",
                    to_zone="Middle",
                    lineup_id="Mid-S1",
                    kills_through=None,  # old-lake event: not measured
                ),
            ],
        ),
    ]
    roi = build_utility_roi(scripts, TEAM)
    by_pattern = {(r.nade, r.pattern): r for r in roi.rows}

    fl = by_pattern[("flash", "Mid-F1")]
    assert fl.n == 2 and fl.side == "T"
    assert fl.avg_enemy_blind_s == pytest.approx(3.0)
    assert fl.avg_team_blind_s == pytest.approx(0.5)
    assert fl.cost == 200
    assert fl.cost_per_enemy_blind_s is None  # n=2 < 5: numbers only, no verdict
    assert fl.evidence == ["m1:1", "m1:2"]

    he = by_pattern[("he", "TSpawn>BombsiteA")]  # no lineup: zone pattern key
    assert he.n == 1 and he.avg_damage == pytest.approx(34.0) and he.cost == 300

    sm = by_pattern[("smoke", "Mid-S1")]
    assert sm.n == 2 and sm.kills_through == 1  # None events aggregate as unmeasured


def test_dollar_verdicts_gated_by_n():
    scripts = [_script(i, utility=[_flash(2.0, 0.0)]) for i in range(1, 6)]
    roi = build_utility_roi(scripts, TEAM)
    fl = roi.rows[0]
    assert fl.n == 5
    assert fl.cost_per_enemy_blind_s == pytest.approx(100.0)  # 200 / 2.0s


def test_ct_molly_costs_incendiary_price():
    molly = UtilEvent(
        t=9.0,
        thrower="p9",
        side="CT",
        nade="molly",
        from_zone="CTSpawn",
        to_zone="Middle",
        damage=30,
    )
    roi = build_utility_roi([_script(1, team_side="CT", utility=[molly])], TEAM)
    assert roi.rows[0].cost == 600
    assert roi.rows[0].avg_damage == pytest.approx(30.0)


def test_opponent_and_unmeasured_events_stay_out():
    # Enemy utility never enters our book; a flash with no measured split
    # yields a row with n but None averages.
    scripts = [
        _script(
            1,
            utility=[
                UtilEvent(
                    t=5.0,
                    thrower="e1",
                    side="CT",
                    nade="flash",
                    from_zone="CTSpawn",
                    to_zone="Middle",
                    enemy_blind_s=9.9,
                ),
                _flash(None, None),
            ],
        )
    ]
    roi = build_utility_roi(scripts, TEAM)
    assert len(roi.rows) == 1
    row = roi.rows[0]
    assert row.n == 1 and row.avg_enemy_blind_s is None and row.avg_team_blind_s is None


def test_empty_inputs_empty_report():
    assert build_utility_roi([], TEAM).rows == []
    assert build_utility_roi([_script(1)], "nobody").rows == []
