"""Retake report miner tests (2026-09-05 advanced-analytics plan Task 2)."""

from typing import Any

from counterstrat.mining.retakes import build_retake_report
from counterstrat.roundscript.models import PlantEvent, RoundScript, ZoneStint

TEAM = "abc"
OPP = "xyz"


def _script(round_num: int, team_side: str, **overrides) -> RoundScript:
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
        "plant": PlantEvent(t=45.0, site="BombsiteB", planter="e1", alive_t=3, alive_ct=2),
        "first_contact": None,
        "winner": "T",
        "reason": "x",
        "clock_used_s": 70.0,
    }
    base.update(overrides)
    return RoundScript(**base)


def test_retake_rows_per_site_and_man_diff():
    scripts = [
        # CT retake at 2v3, won: c1 comes from Middle, c2 through Heaven.
        _script(
            1,
            "CT",
            winner="CT",
            tracks={
                "c1": [
                    ZoneStint(t0=0, t1=46, zone="Middle"),
                    ZoneStint(t0=46, t1=60, zone="BombsiteB"),
                ],
                "c2": [
                    ZoneStint(t0=0, t1=50, zone="CTSpawn"),
                    ZoneStint(t0=50, t1=56, zone="Heaven"),
                    ZoneStint(t0=56, t1=70, zone="BombsiteB"),
                ],
                "e1": [ZoneStint(t0=0, t1=70, zone="BombsiteB")],
            },
            sides={"c1": "CT", "c2": "CT", "e1": "T"},
        ),
        # Same spot next round, lost; only c1 re-enters, again via Middle.
        _script(
            2,
            "CT",
            winner="T",
            tracks={
                "c1": [
                    ZoneStint(t0=0, t1=50, zone="Middle"),
                    ZoneStint(t0=50, t1=62, zone="BombsiteB"),
                ],
            },
            sides={"c1": "CT"},
        ),
    ]
    report = build_retake_report(scripts, TEAM)
    assert report.team_key == TEAM and report.map_name == "de_anubis"
    assert len(report.rows) == 1
    row = report.rows[0]
    assert (row.side, row.site, row.man_diff) == ("CT", "BombsiteB", "-1")
    assert row.n == 2 and row.win_rate == 0.5
    assert row.evidence == ["m1:1", "m1:2"]

    by_approach = {tuple(a.approaches): a for a in report.approaches}
    both = by_approach[("Heaven", "Middle")]
    assert both.side == "CT" and both.site == "BombsiteB"
    assert both.n == 1 and both.win_rate == 1.0 and both.evidence == ["m1:1"]
    solo = by_approach[("Middle",)]
    assert solo.n == 1 and solo.win_rate == 0.0


def test_post_plant_hold_rows_for_t_side():
    s = _script(
        3,
        "T",
        winner="T",
        plant=PlantEvent(t=40.0, site="BombsiteA", planter="p1", alive_t=4, alive_ct=2),
        tracks={
            "p1": [ZoneStint(t0=0, t1=70, zone="BombsiteA")],
            "e9": [
                ZoneStint(t0=0, t1=48, zone="Connector"),
                ZoneStint(t0=48, t1=70, zone="BombsiteA"),
            ],
        },
        sides={"p1": "T", "e9": "CT"},
    )
    report = build_retake_report([s], TEAM)
    assert len(report.rows) == 1
    row = report.rows[0]
    assert (row.side, row.site, row.man_diff) == ("T", "BombsiteA", "+2")
    assert row.n == 1 and row.win_rate == 1.0

    # The enemy retake vector we held against is itself a read.
    assert len(report.approaches) == 1
    ap = report.approaches[0]
    assert ap.side == "T" and ap.approaches == ["Connector"] and ap.win_rate == 1.0


def test_rounds_without_plants_or_tracks_still_count_rates():
    s = _script(4, "CT", winner="CT", tracks={}, sides={})
    report = build_retake_report([s, _script(5, "CT", plant=None)], TEAM)
    assert len(report.rows) == 1  # the plantless round contributes nothing
    assert report.rows[0].n == 1 and report.rows[0].win_rate == 1.0
    assert report.approaches == []  # no tracks -> no vectors, never a crash


def test_empty_inputs_empty_report():
    report = build_retake_report([], TEAM)
    assert report.rows == [] and report.approaches == []
