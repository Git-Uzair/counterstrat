import polars as pl
import pytest

from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.mining.tendencies import (
    RoleCard,
    TeamBook,
    Tendency,
    TendencyKey,
    build_teambook,
)
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import (
    Beat,
    Formation,
    KillEvent,
    MovementLine,
    PlantEvent,
    RoundScript,
    UtilEvent,
)
from counterstrat.roundscript.serialize import serialize_match


def _mk_scripts(
    n: int = 12,
    first_contact_zone: str = "Middle",
    minority_zone: str = "Water",
    minority: int = 3,
    team_key: str = "abc",
) -> list[RoundScript]:
    scripts: list[RoundScript] = []
    for i in range(n):
        match_id = f"m{i + 1}"
        round_num = 1
        is_minority = i >= (n - minority)
        zone = minority_zone if is_minority else first_contact_zone
        fc = KillEvent(
            t=15.0 + float(i),
            killer="p1",
            victim="e1",
            killer_side="T",
            zone=zone,
            weapon="ak47",
            headshot=True,
            traded_within_4s=False,
        )
        beat = Beat(
            label="B+15",
            t=15.0,
            t_form=Formation(zones=[(2, "Middle"), (3, "TSpawn")]),
            ct_form=Formation(zones=[(5, "CTSpawn")]),
        )
        util = UtilEvent(
            t=5.0,
            thrower="p1",
            side="T",
            nade="smoke",
            from_zone="TSpawn",
            to_zone="Middle",
            lineup_id="Mid-Smoke",
        )
        plant = PlantEvent(
            t=45.0,
            site="BombsiteA",
            planter="p1",
            alive_t=3,
            alive_ct=2,
        )
        mov = [
            MovementLine(
                player="p1",
                side="T",
                role_hint="pack",
                sentence="p1(T): TSpawn > Middle k(e1)",
            ),
            MovementLine(
                player="p2",
                side="T",
                role_hint="lurk",
                sentence="p2(T): TSpawn > Water",
            ),
        ]
        s = RoundScript(
            match_id=match_id,
            map_name="de_anubis",
            card_checksum="chk123",
            round_num=round_num,
            score_t=0,
            score_ct=1,  # T is behind
            t_team_key=team_key,
            ct_team_key="xyz",
            economy={
                "T": EconSummary(
                    buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0
                )
            },
            beats=[beat],
            kills=[fc],
            utility=[util],
            plant=plant,
            first_contact=fc,
            winner="T",
            reason="ct_killed",
            clock_used_s=50.0,
            movements=mov,
        )
        scripts.append(s)
    return scripts


@pytest.fixture(scope="module")
def anubis_bundle(anubis_lake, anubis_assets):
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


def test_miner_finds_planted_tendency():
    scripts = _mk_scripts(
        n=12,
        first_contact_zone="Middle",
        minority_zone="Water",
        minority=3,
    )
    tb = build_teambook(scripts, team_key="abc")
    assert isinstance(tb, TeamBook)
    t = next(t for t in tb.tendencies if t.key.side == "T" and t.key.buy_class == "full_buy")
    assert isinstance(t, Tendency)
    assert isinstance(t.key, TendencyKey)
    assert abs(t.first_contact_zone["Middle"] - 0.75) < 1e-9
    assert abs(t.first_contact_zone["Water"] - 0.25) < 1e-9
    assert t.n == 12 and len(t.evidence) == 12
    assert all(ev.count(":") == 1 for ev in t.evidence)
    assert t.low_n is False
    assert t.site_committed.get("A") == 1.0
    assert "Mid-Smoke" in t.lineup_sets
    assert t.median_first_contact_s is not None


def test_miner_deterministic():
    scripts = _mk_scripts(n=12, first_contact_zone="Middle", minority_zone="Water", minority=3)
    a = build_teambook(scripts, "abc").model_dump_json()
    b = build_teambook(scripts, "abc").model_dump_json()
    assert a == b


def test_role_cards_calculation():
    scripts = _mk_scripts(n=12, first_contact_zone="Middle", minority_zone="Water", minority=3)
    tb = build_teambook(scripts, "abc")
    assert len(tb.roles) == 2
    p1 = next(r for r in tb.roles if r.player == "p1")
    p2 = next(r for r in tb.roles if r.player == "p2")
    assert isinstance(p1, RoleCard)

    # p1 was the killer in all 12 first contacts -> opening_duel_rate = 1.0
    assert abs(p1.opening_duel_rate - 1.0) < 1e-9
    assert p1.lurk_rate == 0.0
    assert p1.modal_zone_fe15.get("T") == "Middle"

    # p2 was never in first contact, was lurk in all 12 rounds
    assert abs(p2.opening_duel_rate - 0.0) < 1e-9
    assert abs(p2.lurk_rate - 1.0) < 1e-9
    assert p2.modal_zone_fe15.get("T") == "Water"


def test_teambook_to_table_and_sentences():
    scripts = _mk_scripts(n=12, first_contact_zone="Middle", minority_zone="Water", minority=3)
    tb = build_teambook(scripts, "abc")
    table = tb.to_table_text()
    assert "Tendencies" in table
    assert "Middle" in table
    sentences = tb.to_sentences()
    assert len(sentences) >= 1
    assert any("Middle 75%" in s for s in sentences)


def test_low_n_flag():
    scripts = _mk_scripts(n=2, first_contact_zone="Middle", minority_zone="Water", minority=0)
    tb = build_teambook(scripts, "abc")
    # One group mined at three aggregation levels; every row is low_n.
    assert {t.level for t in tb.tendencies} == {0, 1, 2}
    assert all(t.low_n for t in tb.tendencies)


def test_empty_scripts_and_no_participating():
    tb_empty = build_teambook([], "abc")
    assert tb_empty.tendencies == []
    assert tb_empty.roles == []
    assert tb_empty.generated_from == []

    scripts = _mk_scripts(n=2)
    tb_other = build_teambook(scripts, "other_team")
    assert tb_other.tendencies == []
    assert tb_other.roles == []


def test_multiround_prev_outcome_tracking():
    # Build match with rounds 1, 2, 3, 13, 25, 26, 28
    rounds = [1, 2, 3, 13, 25, 26, 28]
    scripts: list[RoundScript] = []
    for rn in rounds:
        # Team abc wins round 1, loses round 2, wins round 25
        winner = "T" if rn in (1, 25) else "CT"
        s = RoundScript(
            match_id="match_single",
            map_name="de_anubis",
            card_checksum="chk123",
            round_num=rn,
            score_t=0,
            score_ct=0,
            t_team_key="abc",
            ct_team_key="xyz",
            economy={
                "T": EconSummary(
                    buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0
                )
            },
            beats=[],
            kills=[],
            utility=[],
            plant=None,
            first_contact=None,
            winner=winner,
            reason="ct_killed",
            clock_used_s=40.0,
            movements=[],
        )
        scripts.append(s)

    tb = build_teambook(scripts, "abc")

    def t_for(rn: int) -> Tendency:
        target = f"match_single:{rn}"
        return next(t for t in tb.tendencies if t.level == 2 and target in t.evidence)

    # Round 1 is first in match
    assert t_for(1).key.prev_outcome == "first"
    # Round 2 is after won round 1
    assert t_for(2).key.prev_outcome == "won"
    # Round 3 is after lost round 2
    assert t_for(3).key.prev_outcome == "lost"
    # Round 13 is halftime
    assert t_for(13).key.prev_outcome == "first"
    # Round 25 is OT start
    assert t_for(25).key.prev_outcome == "first"
    # Round 26 is after won round 25
    assert t_for(26).key.prev_outcome == "won"
    # Round 28 is OT second-half start (25 + 3)
    assert t_for(28).key.prev_outcome == "first"


# --- Task 1: hierarchical levels + signal filtering ---------------------------


def test_teambook_has_aggregate_levels(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    levels = {t.level for t in tb.tendencies}
    assert levels == {0, 1, 2}
    l0_t = [t for t in tb.tendencies if t.level == 0 and t.key.side == "T"]
    assert len(l0_t) == 1
    assert l0_t[0].n == 4  # all four synthetic T rounds
    assert l0_t[0].key.buy_class == "any"
    assert l0_t[0].key.score_bucket == "any"
    assert l0_t[0].key.prev_outcome == "any"


def test_concentration_and_signal_flags(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    for t in tb.tendencies:
        top = max(t.first_contact_zone.values(), default=0.0)
        assert abs(t.fc_concentration - top) < 1e-9
        top_site = max(t.site_committed.values(), default=0.0)
        assert abs(t.site_concentration - top_site) < 1e-9
        assert t.signal == (t.n >= 3 and t.fc_concentration >= 0.5)


def test_table_text_hides_noisy_level2_rows(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    txt = tb.to_table_text()
    for t in tb.tendencies:
        row_key = f"| {t.key.buy_class} | {t.key.score_bucket} | {t.key.prev_outcome} |"
        if t.level == 2 and not t.signal:
            assert row_key not in txt
        if t.level < 2:
            assert row_key in txt


def test_sentences_prefer_specific_signal_rows(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    sentences = tb.to_sentences()
    assert sentences
    # No sentence may quote a sub-threshold sample as a read.
    signal_ns = {t.n for t in tb.tendencies if t.signal}
    for s in sentences:
        if "No concentrated" not in s:
            n = int(s.rsplit("(n=", 1)[1].rstrip(".)"))
            assert n in signal_ns and n >= 3


def test_old_teambook_json_still_loads(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    dumped = tb.model_dump()
    for t in dumped["tendencies"]:
        # Simulate a pre-upgrade artifact: no level/concentration/signal fields.
        t.pop("level", None)
        t.pop("fc_concentration", None)
        t.pop("site_concentration", None)
        t.pop("signal", None)
    old = TeamBook.model_validate(dumped)
    assert all(t.level == 2 for t in old.tendencies)
    assert all(t.signal is False for t in old.tendencies)


@pytest.mark.demo
def test_teambook_demo_smoke(anubis_bundle):
    scripts = serialize_match(**anubis_bundle)
    team_key = scripts[0].t_team_key
    tb = build_teambook(scripts, team_key=team_key)
    assert len(tb.tendencies) > 0
    assert len(tb.roles) > 0

    script_ids = {
        f"{s.match_id}:{s.round_num}"
        for s in scripts
        if s.t_team_key == team_key or s.ct_team_key == team_key
    }
    for t in tb.tendencies:
        for ev in t.evidence:
            assert ev in script_ids

    # sum of level-2 n for each side equals rounds played on that side
    t_rounds = sum(1 for s in scripts if s.t_team_key == team_key)
    ct_rounds = sum(1 for s in scripts if s.ct_team_key == team_key)
    n_t = sum(t.n for t in tb.tendencies if t.key.side == "T" and t.level == 2)
    n_ct = sum(t.n for t in tb.tendencies if t.key.side == "CT" and t.level == 2)
    assert n_t == t_rounds
    assert n_ct == ct_rounds
    # ... and the level-0 rollup carries the same totals in one row per side.
    for side, rounds in (("T", t_rounds), ("CT", ct_rounds)):
        l0 = [t for t in tb.tendencies if t.key.side == side and t.level == 0]
        assert len(l0) == 1 and l0[0].n == rounds

    table = tb.to_table_text()
    assert "Tendencies" in table
    sentences = tb.to_sentences()
    assert len(sentences) >= 1
    assert all("(n=" in s for s in sentences)
