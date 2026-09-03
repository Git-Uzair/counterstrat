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
) -> RoundScript:
    """A CT-side round for TEAM with configurable beats/winner/utility."""
    return RoundScript(
        match_id="gm1",
        map_name="de_anubis",
        card_checksum="chk123",
        round_num=round_num,
        score_t=0,
        score_ct=0,
        t_team_key=OPP,
        ct_team_key=TEAM,
        economy={
            "CT": EconSummary(buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0),
            "T": EconSummary(buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0),
        },
        beats=beats,
        kills=[],
        utility=utility or [],
        plant=(
            PlantEvent(t=45.0, site=plant_site, planter="e1", alive_t=3, alive_ct=2)
            if plant_site
            else None
        ),
        first_contact=None,
        winner=winner,
        reason="bomb_defused" if winner == "CT" else "t_killed",
        clock_used_s=60.0,
        movements=[],
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


def test_gap_report_respects_explicit_key_zones(after_loss_scripts):
    rep = build_gap_report(after_loss_scripts, TEAM, key_zones=["BombsiteA"])
    assert rep.key_zones == ["BombsiteA"]
    assert all(f.zone == "BombsiteA" for f in rep.findings)


def test_gap_report_empty_for_unknown_team(synthetic_scripts):
    rep = build_gap_report(synthetic_scripts, "nobody")
    assert rep.findings == []


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
