"""Gap/vacancy miner tests (plan Task 3)."""

import pytest
from conftest import SYNTHETIC_TEAM

from counterstrat.mining.gaps import build_gap_report
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import (
    Beat,
    Formation,
    PlantEvent,
    RoundScript,
    UtilEvent,
    ZoneStint,
)
from counterstrat.roundscript.serialize import serialize_match

TEAM = "abc"
OPP = "xyz"


def _script(
    round_num: int,
    *,
    beats: list[Beat],
    winner: str,
    utility: list[UtilEvent] | None = None,
    plant_site: str | None = "BombsiteB",
    plant_t: float = 45.0,
    team_side: str = "CT",
    tracks: dict[str, list[ZoneStint]] | None = None,
    sides: dict[str, str] | None = None,
) -> RoundScript:
    """A round for TEAM (CT by default) with configurable beats/winner/utility."""
    return RoundScript(
        match_id="gm1",
        map_name="de_anubis",
        card_checksum="chk123",
        round_num=round_num,
        score_t=0,
        score_ct=0,
        t_team_key=TEAM if team_side == "T" else OPP,
        ct_team_key=TEAM if team_side == "CT" else OPP,
        economy={
            "CT": EconSummary(buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0),
            "T": EconSummary(buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0),
        },
        beats=beats,
        kills=[],
        utility=utility or [],
        plant=(
            PlantEvent(t=plant_t, site=plant_site, planter="e1", alive_t=3, alive_ct=2)
            if plant_site
            else None
        ),
        first_contact=None,
        winner=winner,
        reason="bomb_defused" if winner == "CT" else "t_killed",
        clock_used_s=60.0,
        movements=[],
        tracks=tracks or {},
        sides=sides or {},
    )


def _beat(label: str, t: float, ct_zones: list[tuple[int, str]]) -> Beat:
    return Beat(
        label=label,
        t=t,
        t_form=Formation(zones=[(5, "TSpawn")]),
        ct_form=Formation(zones=ct_zones),
    )


@pytest.fixture
def after_loss_scripts() -> list[RoundScript]:
    """CT rounds 1-7: B held while winning (r1-4), vacated after each loss (r5-7)."""
    holding = [_beat("B+15", 15.0, [(2, "BombsiteB"), (3, "BombsiteA")])]
    vacated = [_beat("B+15", 15.0, [(3, "BombsiteA"), (2, "CTSpawn")])]
    scripts = []
    for rn in (1, 2, 3, 4):
        scripts.append(_script(rn, beats=holding, winner="CT"))
    # r4 lost -> r5 after_loss; r5 lost -> r6 after_loss; r6 lost -> r7 after_loss
    scripts[-1] = _script(4, beats=holding, winner="T")
    for rn in (5, 6):
        scripts.append(_script(rn, beats=vacated, winner="T"))
    scripts.append(_script(7, beats=vacated, winner="CT"))
    return scripts


def test_gap_report_base_coverage(synthetic_scripts):
    rep = build_gap_report(synthetic_scripts, SYNTHETIC_TEAM)
    assert rep.findings, "expected base coverage findings"
    assert rep.key_zones  # derived from plant sites
    base = [f for f in rep.findings if f.trigger == "base"]
    assert base
    for f in base:
        assert f.n >= 3
        assert 0.0 <= f.vacancy_rate <= 1.0
        assert f.window.startswith(("B+", "post-"))
        assert f.lift == 0.0
    # Synthetic CT formation holds BombsiteA and never BombsiteB.
    ct_b = next(f for f in base if f.side == "CT" and f.zone == "BombsiteB" and f.window == "B+15")
    assert ct_b.vacancy_rate == 1.0
    ct_a = next(f for f in base if f.side == "CT" and f.zone == "BombsiteA" and f.window == "B+15")
    assert ct_a.vacancy_rate == 0.0


def test_gap_trigger_after_loss(after_loss_scripts):
    rep = build_gap_report(after_loss_scripts, TEAM)
    trig = [f for f in rep.findings if f.trigger == "after_loss" and f.zone == "BombsiteB"]
    assert len(trig) == 1
    f = trig[0]
    assert f.side == "CT" and f.window == "B+15"
    assert f.n == 3
    assert abs(f.vacancy_rate - 1.0) < 1e-9
    assert abs(f.baseline_rate - 3 / 7) < 1e-9
    assert f.lift >= 0.15
    assert f.evidence == ["gm1:5", "gm1:6", "gm1:7"]
    # after_win rounds hold the site: no finding may claim otherwise.
    assert not any(f.trigger == "after_win" and f.zone == "BombsiteB" for f in rep.findings)


def test_gap_trigger_after_util_dump():
    """3 dump rounds vacate B at B+30; 2 quiet rounds hold it."""

    def nades(k: int) -> list[UtilEvent]:
        return [
            UtilEvent(
                t=5.0 + i,
                thrower="p1",
                side="CT",
                nade="smoke",
                from_zone="CTSpawn",
                to_zone="Middle",
            )
            for i in range(k)
        ]

    held = [_beat("B+30", 30.0, [(2, "BombsiteB"), (3, "BombsiteA")])]
    empty = [_beat("B+30", 30.0, [(5, "BombsiteA")])]
    scripts = [
        _script(1, beats=empty, winner="CT", utility=nades(3)),
        _script(2, beats=empty, winner="CT", utility=nades(3)),
        _script(3, beats=empty, winner="CT", utility=nades(4)),
        _script(4, beats=held, winner="CT", utility=[]),
        _script(5, beats=held, winner="CT", utility=[]),
    ]
    rep = build_gap_report(scripts, TEAM)
    f = next(
        f
        for f in rep.findings
        if f.trigger == "after_util_dump" and f.zone == "BombsiteB" and f.window == "B+30"
    )
    assert f.n == 3
    assert abs(f.vacancy_rate - 1.0) < 1e-9
    assert abs(f.baseline_rate - 0.6) < 1e-9
    assert abs(f.lift - 0.4) < 1e-9


def test_post_plant_gaps_only_consider_the_planted_site():
    """Vacating the NON-planted site post-plant is normal retake behavior, not a gap.

    Bomb is planted on B every round; CTs vacate A post-plant (they rotate to
    retake) - that must NOT be reported. CTs also absent from the planted B in
    all rounds - that IS a read (they never contest the retake).
    """
    beats = [
        _beat("B+15", 15.0, [(2, "BombsiteB"), (2, "BombsiteA"), (1, "Middle")]),
        _beat("PL+50", 50.0, [(5, "Middle")]),  # nobody on either site post-plant
    ]
    scripts = [_script(rn, beats=beats, winner="T", plant_site="BombsiteB") for rn in (1, 2, 3, 4)]
    rep = build_gap_report(scripts, TEAM)

    post_pl = [f for f in rep.findings if f.window == "post-PL"]
    assert post_pl, "expected a post-plant finding for the planted site"
    assert all(f.zone == "BombsiteB" for f in post_pl), post_pl
    assert not any(f.zone == "BombsiteA" and f.window == "post-PL" for f in rep.findings), (
        "non-planted site vacancy post-plant is a tautology"
    )


def test_gap_vacancy_uses_site_complex():
    """Holding a site means being anywhere in its travel-time complex, not
    standing on the plant zone. Vacancy = the whole complex is empty.

    Regression: a CT watching the A entrance from Main was mined as 'CT vacate
    BombsiteA', which the First Read escalated to 'completely conceding A'.
    """
    # Connector is 3.8s from B (inside the 5s complex); Canal is 3.8+3.2=7.0s (out).
    topo = {"BombsiteB": {"Connector": 3.8, "Alley": 4.9}, "Connector": {"Canal": 3.2}}
    onsite = [_beat("B+15", 15.0, [(2, "BombsiteB"), (3, "BombsiteA")])]
    offsite = [_beat("B+15", 15.0, [(3, "BombsiteA"), (2, "Connector")])]
    empty = [_beat("B+15", 15.0, [(3, "BombsiteA"), (2, "Canal")])]
    scripts = [
        _script(1, beats=onsite, winner="CT"),
        _script(2, beats=offsite, winner="CT"),
        _script(3, beats=offsite, winner="CT"),
        _script(4, beats=empty, winner="CT"),
    ]

    rep = build_gap_report(scripts, TEAM, topology=topo)
    assert rep.site_complexes["BombsiteB"] == ["Alley", "BombsiteB", "Connector"]
    f = next(
        f
        for f in rep.findings
        if f.trigger == "base" and f.zone == "BombsiteB" and f.window == "B+15"
    )
    assert abs(f.vacancy_rate - 1 / 4) < 1e-9  # only the round with nobody in the complex
    assert f.evidence == ["gm1:4"]
    assert f.top_holds == {"Connector": 2, "BombsiteB": 1}

    # Symmetric lookup: the topology may record the edge on Connector's side only.
    rep_sym = build_gap_report(scripts, TEAM, topology={"Connector": {"BombsiteB": 3.8}})
    f_sym = next(
        f
        for f in rep_sym.findings
        if f.trigger == "base" and f.zone == "BombsiteB" and f.window == "B+15"
    )
    assert abs(f_sym.vacancy_rate - 1 / 4) < 1e-9

    # No topology -> the complex degrades to the zone itself (literal vacancy).
    rep_lit = build_gap_report(scripts, TEAM)
    f_lit = next(
        f
        for f in rep_lit.findings
        if f.trigger == "base" and f.zone == "BombsiteB" and f.window == "B+15"
    )
    assert abs(f_lit.vacancy_rate - 3 / 4) < 1e-9
    assert f_lit.top_holds == {"BombsiteB": 1}
    assert rep_lit.site_complexes["BombsiteB"] == ["BombsiteB"]


def test_gap_report_respects_explicit_key_zones(after_loss_scripts):
    rep = build_gap_report(after_loss_scripts, TEAM, key_zones=["BombsiteA"])
    assert rep.key_zones == ["BombsiteA"]
    assert all(f.zone == "BombsiteA" for f in rep.findings)


def test_gap_report_empty_for_unknown_team(synthetic_scripts):
    rep = build_gap_report(synthetic_scripts, "nobody")
    assert rep.findings == []


# --- Setup-intact semantics (2026-09-06): vacancy is only a read while the
# --- defensive setup exists - pre-plant, enough players alive, defending side.


def test_no_round_start_window():
    """B+00 vacancy is a tautology (everyone is at spawn), never a window."""
    beats = [
        _beat("B+00", 0.0, [(5, "CTSpawn")]),
        _beat("B+15", 15.0, [(2, "BombsiteB"), (3, "BombsiteA")]),
    ]
    scripts = [_script(rn, beats=beats, winner="CT") for rn in (1, 2, 3, 4)]
    rep = build_gap_report(scripts, TEAM)
    assert not any(f.window == "B+00" for f in rep.findings)
    assert any(f.window == "B+15" for f in rep.findings)


def test_beats_after_the_plant_are_not_setup_windows():
    """Once the bomb is down, leaving a site is retake rotation, not a gap.

    Regression: 14 of the 33 'vacant BombsiteA at B+45' rounds behind the
    'they abandon A (45%)' First Read had the bomb already planted.
    """
    beats = [
        _beat("B+15", 15.0, [(2, "BombsiteB"), (3, "BombsiteA")]),
        _beat("B+30", 30.0, [(5, "BombsiteB")]),  # collapsed to the planted site
    ]
    scripts = [
        _script(rn, beats=beats, winner="T", plant_site="BombsiteB", plant_t=20.0)
        for rn in (1, 2, 3, 4)
    ]
    rep = build_gap_report(scripts, TEAM)
    assert not any(f.window == "B+30" for f in rep.findings), (
        "post-plant beats must not feed setup windows"
    )


def test_man_down_beats_are_not_setup_observations():
    """A 2-man CT side has no setup to read; vacancy there is not a gap."""
    intact = [_beat("B+15", 15.0, [(4, "BombsiteA"), (1, "BombsiteB")])]
    broken = [_beat("B+15", 15.0, [(2, "Middle")])]  # 3 dead, site empty
    scripts = [
        _script(1, beats=intact, winner="CT"),
        _script(2, beats=intact, winner="CT"),
        _script(3, beats=intact, winner="CT"),
        _script(4, beats=broken, winner="T"),
        _script(5, beats=broken, winner="T"),
    ]
    rep = build_gap_report(scripts, TEAM, key_zones=["BombsiteA"])
    f = next(
        f
        for f in rep.findings
        if f.trigger == "base" and f.zone == "BombsiteA" and f.window == "B+15"
    )
    assert f.n == 3, "man-down rounds must not count as setup observations"
    assert f.vacancy_rate == 0.0


def test_pre_plant_site_rows_are_defense_only():
    """T-side pre-plant 'site vacancy' is structural (Ts hold sites only when
    executing); only the post-plant hold of the planted site is a T read."""
    t_beats = [
        Beat(
            label="B+15",
            t=15.0,
            t_form=Formation(zones=[(3, "Middle"), (2, "TSpawn")]),
            ct_form=Formation(zones=[(5, "CTSpawn")]),
        ),
        Beat(
            label="PL+40",
            t=40.0,
            t_form=Formation(zones=[(4, "BombsiteB"), (1, "Middle")]),
            ct_form=Formation(zones=[(3, "CTSpawn")]),
        ),
    ]
    scripts = [
        _script(rn, beats=t_beats, winner="T", team_side="T", plant_t=35.0) for rn in (1, 2, 3, 4)
    ]
    rep = build_gap_report(scripts, TEAM)
    assert not any(f.side == "T" and f.window.startswith("B+") for f in rep.findings), (
        "attacking-side pre-plant vacancy is a tautology"
    )
    post_pl = [f for f in rep.findings if f.side == "T" and f.window == "post-PL"]
    assert post_pl and post_pl[0].zone == "BombsiteB"
    assert post_pl[0].vacancy_rate == 0.0


def test_stint_cover_counts_when_the_snapshot_misses():
    """A CT whose movement track covers the beat instant from inside the
    complex counts as cover even when the one-tick formation missed him."""
    topo = {"BombsiteA": {"Walkway": 2.1, "Main": 3.1}}
    beats = [_beat("B+15", 15.0, [(5, "Canal")])]  # snapshot: nobody near A
    covered = _script(
        1,
        beats=beats,
        winner="CT",
        plant_site="BombsiteA",
        tracks={"c1": [ZoneStint(t0=10, t1=20, zone="Main")]},
        sides={"c1": "CT"},
    )
    bare = [_script(rn, beats=beats, winner="CT") for rn in (2, 3, 4)]
    rep = build_gap_report(scripts=[covered, *bare], team_key=TEAM, topology=topo)
    f = next(
        f
        for f in rep.findings
        if f.trigger == "base" and f.zone == "BombsiteA" and f.window == "B+15"
    )
    assert abs(f.vacancy_rate - 3 / 4) < 1e-9, "the tracked round must count as covered"
    assert "gm1:1" not in f.evidence
    assert f.top_holds.get("Main") == 1


def test_vacant_round_annotations_watchers_and_pressure():
    """Vacant-but-watched and vacant-under-pressure are distinct from open."""
    topo = {"BombsiteA": {"Walkway": 2.1, "Main": 3.1}}
    held = [_beat("B+15", 15.0, [(4, "BombsiteA"), (1, "Middle")])]
    vacant_beat = [
        Beat(
            label="B+15",
            t=15.0,
            t_form=Formation(zones=[(4, "TSpawn"), (1, "Walkway")]),  # T entering the complex
            ct_form=Formation(zones=[(4, "Middle"), (1, "CTSpawn")]),
        )
    ]
    watcher = _script(
        4,
        beats=vacant_beat,
        winner="T",
        tracks={"c1": [ZoneStint(t0=5, t1=25, zone="Middle", watched="Walkway", locked=True)]},
        sides={"c1": "CT"},
    )
    scripts = [_script(rn, beats=held, winner="CT") for rn in (1, 2, 3)] + [watcher]
    rep = build_gap_report(scripts, TEAM, key_zones=["BombsiteA"], topology=topo)
    f = next(
        f
        for f in rep.findings
        if f.trigger == "base" and f.zone == "BombsiteA" and f.window == "B+15"
    )
    assert abs(f.vacancy_rate - 1 / 4) < 1e-9
    assert f.watched_n == 1, "the Middle watcher had eyes on the complex"
    assert f.top_watch_zones == {"Middle": 1}
    assert f.pressured_n == 1, "a T inside the complex means contested, not open"


@pytest.mark.demo
def test_gap_report_demo_smoke(anubis_gap_bundle):
    scripts = serialize_match(**anubis_gap_bundle)
    team_key = scripts[0].ct_team_key
    rep = build_gap_report(scripts, team_key)
    assert any(f.trigger == "base" for f in rep.findings)
    assert len(rep.findings) <= 30


@pytest.fixture(scope="module")
def anubis_gap_bundle(anubis_lake, anubis_assets):
    import polars as pl

    from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
    from counterstrat.mapcard.vents import parse_places, unique_places
    from counterstrat.mapcard.zones import ZoneMapper

    ticks = pl.read_parquet(anubis_lake.ticks)
    mapper = ZoneMapper.fit(ticks)
    places = unique_places(parse_places(anubis_assets.vents))
    overlay = get_default_overlay_path("de_anubis")
    lex = build_lexicon("de_anubis", places, overlay)
    return {
        "lake": anubis_lake,
        "mapper": mapper,
        "lex": lex,
        "card_checksum": lex.checksum,
    }
