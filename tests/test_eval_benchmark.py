"""Tests for per-round prediction labels, time-ordered split, baselines, and LLM arms."""

from typing import Any

import pytest

from counterstrat.constants import BUY_BINS
from counterstrat.eval.benchmark import (
    MatchRecord,
    extract_labels,
    run_arms,
    run_offline_arms,
    split,
)
from counterstrat.llm.base import LLMResult
from counterstrat.llm.predict import MatchState, PredictedRound, predict_round
from counterstrat.mapcard.compile import MapCard
from counterstrat.mining.tendencies import build_teambook
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import (
    Beat,
    Formation,
    KillEvent,
    PlantEvent,
    RoundScript,
    UtilEvent,
)

ECO_ROUNDS = (1, 4)  # planted: eco rounds contact Middle, buy rounds contact Main


def _mk_round(match_id: str, round_num: int) -> RoundScript:
    """One synthetic round; the tendency planted is buy-tier dependent."""
    is_eco = round_num in ECO_ROUNDS
    zone = "Middle" if is_eco else "Main"
    fc = KillEvent(
        t=20.0 if is_eco else 40.0,
        killer="p1",
        victim="e1",
        killer_side="T",
        zone=zone,
        weapon="deagle" if is_eco else "ak47",
        headshot=False,
        traded_within_4s=False,
    )
    utility = [
        UtilEvent(
            t=8.0,
            thrower="p1",
            side="T",
            nade="smoke",
            from_zone="TSpawn",
            to_zone="Middle",
            lineup_id="Mid-Smoke",
        )
    ]
    if not is_eco:
        utility = [
            UtilEvent(
                t=32.0,
                thrower="p1",
                side="T",
                nade="smoke",
                from_zone="Main",
                to_zone="BombsiteA",
                lineup_id="AExec-Smoke",
            ),
            UtilEvent(
                t=35.0,
                thrower="p2",
                side="T",
                nade="flashbang",
                from_zone="Main",
                to_zone="BombsiteA",
                lineup_id="AExec-Flash",
            ),
        ]
    return RoundScript(
        match_id=match_id,
        map_name="de_anubis",
        card_checksum="chk123",
        round_num=round_num,
        score_t=round_num - 1,
        score_ct=0,
        t_team_key="abc",
        ct_team_key="xyz",
        economy={
            "T": EconSummary(
                buy_type="full_eco" if is_eco else "full_buy",
                spend=1200 if is_eco else 21000,
                equip=3000 if is_eco else 25000,
                awps=0 if is_eco else 1,
                loss_streak=1 if is_eco else 0,
            ),
            "CT": EconSummary(buy_type="full_buy", spend=20000, equip=24000, awps=1, loss_streak=0),
        },
        beats=[
            Beat(
                label="B+15",
                t=15.0,
                t_form=Formation(zones=[(3, zone), (2, "TSpawn")]),
                ct_form=Formation(zones=[(5, "CTSpawn")]),
            )
        ],
        kills=[fc],
        utility=utility,
        plant=(
            None
            if is_eco
            else PlantEvent(t=45.0, site="BombsiteA", planter="p1", alive_t=4, alive_ct=2)
        ),
        first_contact=fc,
        winner="T",
        reason="ct_killed",
        clock_used_s=60.0,
    )


def _mk_corpus(n_matches: int = 10, rounds_per_match: int = 6) -> list[MatchRecord]:
    return [
        MatchRecord(
            match_id=f"m{i:02d}",
            registered_at=f"2026-01-{i:02d}T12:00:00+00:00",
            scripts=[_mk_round(f"m{i:02d}", r) for r in range(1, rounds_per_match + 1)],
        )
        for i in range(1, n_matches + 1)
    ]


def _mk_card() -> MapCard:
    return MapCard(
        map="de_anubis",
        card_version="1.0",
        game_version="14178",
        nav_source="nav",
        frame={},
        zones={
            "Middle": {"aliases": ["mid"], "tags": ["mid_control"]},
            "Main": {"aliases": ["A main"], "tags": ["t_side_entry"]},
            "BombsiteA": {"aliases": ["A site"], "tags": ["site"]},
            "BombsiteB": {"aliases": ["B site"], "tags": ["site"]},
            "TSpawn": {"aliases": [], "tags": ["spawn"]},
            "CTSpawn": {"aliases": [], "tags": ["spawn"]},
        },
        topology={},
        rotates=[],
        timings={},
        objectives={"sites": ["BombsiteA", "BombsiteB"]},
        sightlines=[],
        checksum="chk123",
    )


class FakeJSONClient:
    """Records prompts and replays one canned structured prediction."""

    def __init__(self, pred: PredictedRound):
        self.pred = pred
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 4096) -> LLMResult:
        raise NotImplementedError

    def complete_json(
        self, *, system: str, user: str, schema: Any, max_tokens: int = 4096
    ) -> tuple[Any, LLMResult]:
        self.calls.append({"system": system, "user": user})
        usage = LLMResult(
            text=self.pred.model_dump_json(),
            input_tokens=100,
            output_tokens=20,
            model="mock-model",
            provider="mock",
        )
        return schema.model_validate(self.pred.model_dump()), usage

    def chat(self, *, system: str, turns: Any, tools: Any, max_tokens: int = 4096) -> Any:
        raise NotImplementedError


@pytest.fixture
def synthetic_scripts() -> list[RoundScript]:
    return [_mk_round("m01", r) for r in range(1, 7)]


@pytest.fixture
def synthetic_corpus() -> list[MatchRecord]:
    return _mk_corpus()


def test_ground_truth_labels(synthetic_scripts):
    labels = extract_labels(synthetic_scripts[0])
    assert labels.buy_type in BUY_BINS
    assert labels.site in {"A", "B", "none"}

    eco = extract_labels(synthetic_scripts[0])  # round 1: eco, no plant, fast contact
    buy = extract_labels(synthetic_scripts[1])  # round 2: full buy, A execute, slow contact
    assert (eco.buy_type, eco.first_contact_zone, eco.site) == ("full_eco", "Middle", "none")
    assert eco.execute is False and eco.fast is True
    assert (buy.buy_type, buy.first_contact_zone, buy.site) == ("full_buy", "Main", "A")
    assert buy.execute is True and buy.fast is False


def test_execute_needs_clustered_lineups_and_commit(synthetic_scripts):
    script = synthetic_scripts[1]
    assert extract_labels(script).execute is True

    spread = script.model_copy(deep=True)
    spread.utility[1].t = script.utility[0].t + 20.0  # outside the 8 s window
    assert extract_labels(spread).execute is False

    no_commit = script.model_copy(deep=True)
    no_commit.plant = None
    no_commit.first_contact = script.first_contact.model_copy(update={"zone": "Middle"})
    assert extract_labels(no_commit).execute is False

    committed = no_commit.model_copy(deep=True)
    committed.first_contact = script.first_contact.model_copy(update={"zone": "BombsiteA"})
    assert extract_labels(committed).execute is True


def test_time_ordered_split(synthetic_corpus):
    train, held = split(synthetic_corpus)
    assert train and held
    assert len(train) + len(held) == len(synthetic_corpus)
    assert max(m.registered_at for m in train) <= min(m.registered_at for m in held)


def test_freq_table_baseline_beats_majority_on_planted(synthetic_corpus):
    rep = run_offline_arms(synthetic_corpus, arms=["majority", "freq_table"])
    assert rep.n_eval_rounds == 18
    assert (
        rep.heads["first_contact_zone"]["freq_table"].accuracy
        > rep.heads["first_contact_zone"]["majority"].accuracy
    )
    assert rep.heads["site"]["freq_table"].brier <= rep.heads["site"]["majority"].brier
    assert rep.arm_costs == {"majority": 0.0, "freq_table": 0.0}


def test_run_offline_arms_rejects_llm_arms(synthetic_corpus):
    with pytest.raises(ValueError, match="require an LLM client"):
        run_offline_arms(synthetic_corpus, arms=["majority", "full"])


def test_predict_round_offline(synthetic_corpus):
    scripts = [s for m in synthetic_corpus for s in m.scripts]
    teambook = build_teambook(scripts, "abc")
    card = _mk_card()
    canned = PredictedRound(
        buy_type="full_buy",
        first_contact_zone="Main",
        site="A",
        execute=True,
        fast=False,
        p_site={"A": 0.7, "B": 0.2, "none": 0.1},
    )
    client = FakeJSONClient(canned)
    state = MatchState(
        score_t=5,
        score_ct=3,
        side="T",
        prev_round_summaries=["R8: full_buy, first contact Main@40s"],
        economy_estimate="previous round T spend $21000 (full_buy)",
    )

    pred, usage = predict_round(client, card, teambook, state)
    assert pred.first_contact_zone == "Main" and pred.site == "A"
    assert pred.p_site["A"] == pytest.approx(0.7)
    assert usage.input_tokens == 100
    assert len(client.calls) == 1
    assert "de_anubis" in client.calls[0]["system"] and "side T" in client.calls[0]["system"]
    assert "TeamBook: abc" in client.calls[0]["user"]
    assert "score T 5 - CT 3" in client.calls[0]["user"]


def test_llm_arms_wired_offline(synthetic_corpus):
    canned = PredictedRound(
        buy_type="full_buy",
        first_contact_zone="Main",
        site="A",
        execute=True,
        fast=False,
        p_site={"A": 0.8, "B": 0.1, "none": 0.1},
    )
    client = FakeJSONClient(canned)
    rep = run_arms(
        synthetic_corpus,
        ["full", "llm_no_profile"],
        client=client,
        card=_mk_card(),
    )
    assert rep.n_eval_rounds == 18
    assert set(rep.heads["site"]) == {"full", "llm_no_profile"}
    assert rep.arm_costs["full"] == pytest.approx(18 * 120)
    assert rep.arm_costs["llm_no_profile"] == pytest.approx(18 * 120)
    # 12 of 18 held-out rounds are full-buy A executes; the canned answer nails those.
    assert rep.heads["site"]["full"].accuracy == pytest.approx(12 / 18)

    full_prompts = [c["user"] for c in client.calls[:18]]
    no_profile_prompts = [c["user"] for c in client.calls[18:]]
    assert any("| T | full_buy |" in p for p in full_prompts)
    assert all("| T | full_buy |" not in p for p in no_profile_prompts)
    assert any("FC: p1(T) killed e1 @Main" in p for p in no_profile_prompts)


def test_run_arms_requires_client_for_llm_arms(synthetic_corpus):
    with pytest.raises(ValueError, match="need an LLMClient"):
        run_arms(synthetic_corpus, ["full"])
    with pytest.raises(ValueError, match="unknown arms"):
        run_arms(synthetic_corpus, ["nonsense"])
