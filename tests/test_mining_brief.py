"""Scout brief miner tests (plan Task 8)."""

from conftest import SYNTHETIC_TEAM

from counterstrat.mining.brief import build_scout_brief
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.tendencies import build_teambook
from counterstrat.mining.utility_book import build_utility_book

ALLOWED_KINDS = {"site_lean", "tempo", "utility_crutch", "gap", "econ_tell", "opener", "pistol"}


def _brief(scripts, team_key=SYNTHETIC_TEAM):
    return build_scout_brief(
        scripts,
        team_key,
        teambook=build_teambook(scripts, team_key),
        utility_book=build_utility_book(scripts, team_key),
        gap_report=build_gap_report(scripts, team_key),
        econ_policy=build_econ_policy(scripts, team_key),
    )


def test_scout_brief_headlines(synthetic_scripts):
    brief = _brief(synthetic_scripts)
    assert brief.team_key == SYNTHETIC_TEAM and brief.map_name == "de_anubis"
    assert 1 <= len(brief.items) <= 8
    kinds = {i.kind for i in brief.items}
    assert kinds <= ALLOWED_KINDS
    for item in brief.items:
        assert item.text
        assert item.n >= 1
        assert item.confidence in ("high", "medium")
        assert item.evidence, item.kind

    # The fixture's strongest reads must surface.
    assert "utility_crutch" in kinds  # smoke Middle every round
    assert "gap" in kinds  # CT formation never holds BombsiteB
    assert "opener" in kinds  # p1 takes every opening duel

    # Zone-bearing items backtick their zones for the UI/linter conventions.
    for item in brief.items:
        if item.kind in ("utility_crutch", "gap"):
            assert "`" in item.text, item.text


def test_scout_brief_deterministic(synthetic_scripts):
    a = _brief(synthetic_scripts).model_dump_json()
    b = _brief(synthetic_scripts).model_dump_json()
    assert a == b


def test_scout_brief_empty_for_unknown_team(synthetic_scripts):
    brief = _brief(synthetic_scripts, team_key="nobody")
    assert brief.items == []


def test_scout_brief_generated_from(synthetic_scripts):
    brief = _brief(synthetic_scripts)
    assert brief.generated_from == ["m1"]
