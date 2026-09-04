"""Engagement-range profile miner tests (space-vision research item 2)."""

import pytest

from counterstrat.mining.range_profile import build_range_profile
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import Beat, Formation, KillEvent, RoundScript
from counterstrat.roundscript.serialize import serialize_match

TEAM = "abc"
OPP = "xyz"


def _kill(killer: str, side: str, dist: float | None, **kw) -> KillEvent:
    return KillEvent(
        t=kw.get("t", 20.0),
        killer=killer,
        victim="e1",
        killer_side=side,
        zone="Middle",
        weapon=kw.get("weapon", "ak47"),
        headshot=False,
        traded_within_4s=False,
        distance=dist,
        thrusmoke=kw.get("thrusmoke", False),
        penetrated=kw.get("penetrated", False),
    )


def _round(rn: int, team_side: str, buy: str, kills: list[KillEvent]) -> RoundScript:
    opp_side = "CT" if team_side == "T" else "T"
    return RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="chk",
        round_num=rn,
        score_t=0,
        score_ct=0,
        t_team_key=TEAM if team_side == "T" else OPP,
        ct_team_key=TEAM if team_side == "CT" else OPP,
        economy={
            team_side: EconSummary(buy_type=buy, spend=1, equip=1, awps=0, loss_streak=0),
            opp_side: EconSummary(buy_type="full_buy", spend=1, equip=1, awps=0, loss_streak=0),
        },
        beats=[
            Beat(
                label="B+15",
                t=15.0,
                t_form=Formation(zones=[(5, "Middle")]),
                ct_form=Formation(zones=[(5, "CTSpawn")]),
            )
        ],
        kills=kills,
        utility=[],
        plant=None,
        first_contact=kills[0] if kills else None,
        winner=team_side,
        reason="ct_killed",
        clock_used_s=60.0,
    )


def test_kill_event_defaults_keep_old_scripts_loading():
    """Pre-upgrade scripts carry no range fields; the model must default them."""
    k = KillEvent(
        t=1.0,
        killer="a",
        victim="b",
        killer_side="T",
        zone="Middle",
        weapon="ak47",
        headshot=False,
        traded_within_4s=False,
    )
    assert k.distance is None
    assert k.thrusmoke is False
    assert k.penetrated is False


def test_range_profile_bands_medians_and_flags():
    scripts = [
        _round(
            1,
            "T",
            "full_buy",
            [
                _kill("p1", "T", 200.0),
                _kill("p1", "T", 500.0, thrusmoke=True),
                _kill("p2", "T", 1000.0),
                _kill("p2", "T", 2000.0, penetrated=True),
                _kill("e1", "CT", 3000.0),  # opponent kill: excluded
            ],
        ),
    ]
    prof = build_range_profile(scripts, TEAM)
    assert prof.team_key == TEAM and prof.map_name == "de_anubis"
    t = prof.sides["T"]
    assert t.n == 4
    assert t.close_share == pytest.approx(0.5)
    assert t.medium_share == pytest.approx(0.25)
    assert t.long_share == pytest.approx(0.25)
    assert t.median_dist == pytest.approx(750.0)
    assert t.smoke_share == pytest.approx(0.25)
    assert t.wallbang_share == pytest.approx(0.25)
    assert "CT" not in prof.sides  # team never played CT here


def test_range_profile_buy_split_and_players():
    scripts = [
        _round(1, "T", "full_buy", [_kill("p1", "T", 1800.0), _kill("p1", "T", 1600.0)]),
        _round(2, "T", "full_buy", [_kill("p1", "T", 1900.0)]),
        _round(3, "T", "semi_eco", [_kill("p2", "T", 150.0), _kill("p2", "T", 250.0)]),
        _round(4, "T", "full_eco", [_kill("p2", "T", 100.0)]),
    ]
    prof = build_range_profile(scripts, TEAM)
    assert prof.full_buy is not None and prof.full_buy.n == 3
    assert prof.full_buy.long_share == pytest.approx(1.0)
    assert prof.low_buy is not None and prof.low_buy.n == 3
    assert prof.low_buy.close_share == pytest.approx(1.0)
    p1 = next(p for p in prof.players if p.player == "p1")
    assert p1.n == 3 and p1.median_dist == pytest.approx(1800.0)
    assert p1.long_share == pytest.approx(1.0)


def test_range_profile_skips_none_distance_and_unknown_team():
    scripts = [_round(1, "T", "full_buy", [_kill("p1", "T", None), _kill("p1", "T", 300.0)])]
    prof = build_range_profile(scripts, TEAM)
    assert prof.sides["T"].n == 1  # None-distance kill (pre-upgrade script) skipped
    assert prof.kills_total == 2 and prof.kills_with_distance == 1

    empty = build_range_profile(scripts, "nobody")
    assert empty.sides == {} and empty.players == []


def test_range_profile_prompt_lines_render():
    scripts = [
        _round(1, "T", "full_buy", [_kill("p1", "T", 200.0), _kill("p1", "T", 1800.0)]),
    ]
    lines = build_range_profile(scripts, TEAM).to_prompt_lines()
    text = "\n".join(lines)
    assert "T:" in text and "close" in text and "long" in text
    assert "n=2" in text
    # empty profile renders nothing rather than noise
    assert build_range_profile(scripts, "nobody").to_prompt_lines() == []


def test_range_profile_surfaces_in_prompts():
    from counterstrat.llm.insights import build_insights_user
    from counterstrat.llm.prompts import build_chat_system, build_user
    from counterstrat.mining.econ_policy import build_econ_policy
    from counterstrat.mining.gaps import build_gap_report
    from counterstrat.mining.tendencies import build_teambook
    from counterstrat.mining.utility_book import build_utility_book

    scripts = [
        _round(1, "T", "full_buy", [_kill("p1", "T", 200.0), _kill("p1", "T", 1800.0)]),
        _round(2, "T", "full_buy", [_kill("p1", "T", 900.0)]),
    ]
    prof = build_range_profile(scripts, TEAM)
    tb = build_teambook(scripts, TEAM)

    chat = build_chat_system("map block", tb, range_profile=prof)
    assert "<engagement_range>" in chat and "median" in chat

    dossier_user = build_user(tb, scripts, range_profile=prof)
    assert "## Engagement Range Profile" in dossier_user

    insights_user = build_insights_user(
        teambook=tb,
        utility_book=build_utility_book(scripts, TEAM),
        gap_report=build_gap_report(scripts, TEAM),
        econ_policy=build_econ_policy(scripts, TEAM),
        scripts=scripts,
    )
    assert "## Engagement Range Profile" in insights_user

    # no profile -> no empty section headers
    assert "<engagement_range>" not in build_chat_system("map block", tb)
    assert "## Engagement Range Profile" not in build_user(tb, scripts)


@pytest.mark.demo
def test_serialize_propagates_range_fields(anubis_bundle_range):
    scripts = serialize_match(**anubis_bundle_range)
    kills = [k for s in scripts for k in s.kills]
    with_dist = [k for k in kills if k.distance is not None]
    assert with_dist, "real lake carries distance for every kill"
    assert all(k.distance >= 0 for k in with_dist)
    # flags exist and at least parse as bools on a real corpus
    assert any(isinstance(k.thrusmoke, bool) for k in kills)
    team_key = scripts[0].t_team_key
    prof = build_range_profile(scripts, team_key)
    assert prof.sides and all(s.n > 0 for s in prof.sides.values())


@pytest.fixture(scope="module")
def anubis_bundle_range(anubis_lake, anubis_assets):
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
