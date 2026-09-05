"""Rotation report miner tests (2026-09-05 advanced-analytics plan Task 2)."""

from typing import Any

from counterstrat.mining.rotations import build_rotation_report
from counterstrat.roundscript.models import KillEvent, RotationEvent, RoundScript, UtilEvent

TEAM = "abc"
OPP = "xyz"


def _kill(t: float, zone: str, killer_side: str = "T") -> KillEvent:
    return KillEvent(
        t=t,
        killer="e1",
        victim="v1",
        killer_side=killer_side,
        zone=zone,
        weapon="ak47",
        headshot=False,
        traded_within_4s=False,
    )


def _script(round_num: int, **overrides) -> RoundScript:
    base: dict[str, Any] = {
        "match_id": "m1",
        "map_name": "de_anubis",
        "card_checksum": "chk",
        "round_num": round_num,
        "score_t": 0,
        "score_ct": 0,
        "t_team_key": OPP,
        "ct_team_key": TEAM,  # the observed team defends by default
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


def _rot(trigger: str, t_trigger: float, latency: float, player: str = "b4") -> RotationEvent:
    return RotationEvent(
        t_trigger=t_trigger,
        trigger=trigger,
        player=player,
        side="CT",
        from_zone="BombsiteB",
        to_zone="Middle",
        latency_s=latency,
    )


def test_rotation_report_rates_n_and_evidence():
    flash = UtilEvent(
        t=20.0, thrower="e1", side="T", nade="flash", from_zone="TSpawn", to_zone="BombsiteA"
    )
    scripts = [
        # Real trigger: utility on A followed by a kill on A within 10s.
        _script(
            1,
            utility=[flash],
            kills=[_kill(25.0, "BombsiteA")],
            rotations=[_rot("utility_near", 20.0, 2.0)],
        ),
        # Fake: utility on A, no follow-up contact there within 10s.
        _script(
            2,
            utility=[flash],
            kills=[_kill(45.0, "Middle")],
            rotations=[_rot("utility_near", 20.0, 3.0)],
        ),
        # First blood in Middle with no second contact there: a fake trigger.
        _script(
            3,
            kills=[_kill(15.0, "Middle")],
            rotations=[_rot("first_blood", 15.0, 1.0)],
        ),
    ]
    report = build_rotation_report(scripts, TEAM)
    assert report.team_key == TEAM and report.map_name == "de_anubis"
    by_key = {(r.player, r.trigger): r for r in report.rows}

    util = by_key[("b4", "utility_near")]
    assert util.n == 2
    assert util.median_latency_s == 2.5
    assert util.fakes_n == 1 and util.fake_rate == 0.5
    assert util.evidence == ["m1:1", "m1:2"]

    fb = by_key[("b4", "first_blood")]
    assert fb.n == 1 and fb.median_latency_s == 1.0
    assert fb.fake_rate == 1.0
    assert fb.evidence == ["m1:3"]


def test_other_side_rotations_are_not_ours():
    # The team played CT; a T-side rotation belongs to the opponent.
    enemy_rot = RotationEvent(
        t_trigger=10.0,
        trigger="plant",
        player="e1",
        side="T",
        from_zone="TSpawn",
        to_zone="Middle",
        latency_s=1.0,
    )
    report = build_rotation_report([_script(1, rotations=[enemy_rot])], TEAM)
    assert report.rows == []


def test_unlocatable_triggers_have_no_fake_rate():
    # 'shots' has no zone in the script: fake-rate must be None, not a guess.
    report = build_rotation_report([_script(1, rotations=[_rot("shots", 12.0, 2.0)])], TEAM)
    assert len(report.rows) == 1
    assert report.rows[0].fake_rate is None and report.rows[0].fakes_n == 0


def test_empty_inputs_empty_report():
    report = build_rotation_report([], TEAM)
    assert report.rows == [] and report.team_key == TEAM
    report = build_rotation_report([_script(1)], "nobody")
    assert report.rows == []
