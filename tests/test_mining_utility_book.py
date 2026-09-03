"""Utility book miner tests (plan Task 2)."""

import pytest
from conftest import SYNTHETIC_TEAM

from counterstrat.mining.utility_book import build_utility_book
from counterstrat.roundscript.serialize import serialize_match


def test_utility_book_groups_by_nade_and_target(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, SYNTHETIC_TEAM)
    assert book.team_key == SYNTHETIC_TEAM
    assert book.map_name == "de_anubis"
    smokes = [p for p in book.patterns if p.nade == "smoke"]
    assert smokes, "synthetic fixture throws a smoke every round"

    t_smoke = next(p for p in smokes if p.side == "T")
    assert t_smoke.to_zone == "Middle"
    assert t_smoke.count == 4  # all four T rounds
    assert t_smoke.rounds_seen == 4
    assert abs(t_smoke.share - 1.0) < 1e-9
    assert abs(t_smoke.median_t - 8.0) < 1e-9
    assert abs(t_smoke.early_share - 1.0) < 1e-9  # 8s < 25s
    assert t_smoke.lineup_id == "Mid-Smoke"
    assert t_smoke.from_zones == {"TSpawn": 4}
    assert t_smoke.evidence and len(t_smoke.evidence) <= 6
    assert all(ev.count(":") == 1 for ev in t_smoke.evidence)


def test_utility_book_respects_side_and_opponent_exclusion(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, SYNTHETIC_TEAM)
    assert {p.side for p in book.patterns} <= {"T", "CT"}
    # The opponent's book must not inherit our throws: xyz threw nothing.
    opp = build_utility_book(synthetic_scripts, "xyz")
    assert opp.patterns == []


def test_utility_book_dump_windows(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, SYNTHETIC_TEAM)
    t_first = next(w for w in book.dump_windows if w["side"] == "T" and w["kth"] == 1)
    assert abs(t_first["median_t"] - 8.0) < 1e-9
    assert t_first["n"] == 4
    # Only one nade per round in the fixture: no k=2 window exists.
    assert not any(w for w in book.dump_windows if w["kth"] == 2)


def test_utility_book_empty_for_unknown_team(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, "nobody")
    assert book.patterns == [] and book.dump_windows == []


@pytest.mark.demo
def test_utility_book_demo_smoke(anubis_bundle_util):
    scripts = serialize_match(**anubis_bundle_util)
    team_key = scripts[0].t_team_key
    book = build_utility_book(scripts, team_key)
    assert book.patterns, "a real match must yield utility patterns"
    assert all(0 < p.share <= 1.0 for p in book.patterns)
    assert all(p.count >= 1 for p in book.patterns)
    assert len(book.patterns) <= 40


@pytest.fixture(scope="module")
def anubis_bundle_util(anubis_lake, anubis_assets):
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
