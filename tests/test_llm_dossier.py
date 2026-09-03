"""Tests for LLM dossier generation, anti-hallucination lint gate, and exemplar selection."""

import os
from typing import Any

import pytest

from counterstrat.config import AppConfig
from counterstrat.llm.base import LLMResult, make_client
from counterstrat.llm.dossier import Dossier, generate, lint_dossier
from counterstrat.llm.prompts import build_system, build_user, select_exemplars
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import build_lexicon
from counterstrat.mining.tendencies import build_teambook
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
            score_ct=1,
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


def _mk_synthetic_bundle():
    scripts = _mk_scripts(
        n=12, first_contact_zone="Middle", minority_zone="Water", minority=3, team_key="abc"
    )
    teambook = build_teambook(scripts, "abc")
    lexicon = build_lexicon(
        "de_anubis", ["Middle", "Water", "BombsiteA", "BombsiteB", "TSpawn", "CTSpawn"]
    )
    evidence_ids = {f"{s.match_id}:{s.round_num}" for s in scripts}
    evidence_ids.add("deadbeefcafe:7")
    return scripts, teambook, lexicon, evidence_ids


def _mk_synthetic_card() -> MapCard:
    return MapCard(
        map="de_anubis",
        card_version="1.0",
        game_version="14178",
        nav_source="nav",
        frame={},
        zones={
            "Middle": {"aliases": ["mid"], "tags": ["mid_control"]},
            "Water": {"aliases": ["canals"], "tags": ["mid_control"]},
            "BombsiteA": {"aliases": ["A site"], "tags": ["site"]},
            "BombsiteB": {"aliases": ["B site"], "tags": ["site"]},
            "TSpawn": {"aliases": [], "tags": ["spawn"]},
            "CTSpawn": {"aliases": [], "tags": ["spawn"]},
        },
        topology={},
        rotates=[],
        timings={},
        objectives={},
        sightlines=[],
        checksum="chk123",
    )


class ScriptedClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 4096) -> LLMResult:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        text = self.responses.pop(0) if self.responses else "fallback"
        return LLMResult(
            text=text,
            input_tokens=150,
            output_tokens=80,
            model="mock-model",
            provider="mock",
        )

    def complete_json(self, *, system: str, user: str, schema: Any, max_tokens: int = 4096) -> Any:
        raise NotImplementedError

    def chat(self, *, system: str, turns: Any, tools: Any, max_tokens: int = 4096) -> Any:
        raise NotImplementedError


def test_prompt_construction():
    scripts, teambook, _, _ = _mk_synthetic_bundle()
    card = _mk_synthetic_card()
    system_prompt = build_system(card.to_yaml())
    required_sections = [
        "Identity & Overview",
        "Defaults & Roles",
        "Execute Repertoire with Counters",
        "Gaps & Triggers",
        "Economy Policy with Exploit",
        "Player-Specific Weaknesses",
        "Round-State Playbook Table",
        "Confidence & Evidence Appendix",
    ]
    for sec in required_sections:
        assert sec in system_prompt
    assert "Exploit:" in system_prompt

    exemplars = select_exemplars(teambook, scripts, cap=6)
    user_prompt = build_user(teambook, exemplars)
    assert teambook.team_key in user_prompt
    assert "## Exemplar Round Scripts" in user_prompt
    assert f"Round {exemplars[0].match_id}:{exemplars[0].round_num}" in user_prompt


def test_build_user_includes_mined_artifacts():
    from counterstrat.mining.econ_policy import build_econ_policy
    from counterstrat.mining.gaps import build_gap_report
    from counterstrat.mining.utility_book import build_utility_book

    scripts, teambook, _, _ = _mk_synthetic_bundle()
    user_prompt = build_user(
        teambook,
        [],
        utility_book=build_utility_book(scripts, teambook.team_key),
        gap_report=build_gap_report(scripts, teambook.team_key),
        econ_policy=build_econ_policy(scripts, teambook.team_key),
    )
    assert "## Utility Book (top patterns)" in user_prompt
    assert "## Economy Policy" in user_prompt
    # Gap findings need >= 3 rounds per (side, window, zone); the synthetic
    # bundle carries plants so at least a base coverage section appears.
    assert "## Gap Findings" in user_prompt


def test_dossier_lint_catches_fabrication():
    _, teambook, lexicon, evidence_ids = _mk_synthetic_bundle()
    good = "…cites `Middle` (evidence deadbeefcafe:7) 75%…"
    bad = "…cites `Ghost` (evidence ffffffffffff:99) 12%…"
    lint_g = lint_dossier(good, teambook, lexicon, evidence_ids)
    lint_b = lint_dossier(bad, teambook, lexicon, evidence_ids)
    assert lint_g.ok
    assert not lint_b.ok and "Ghost" in lint_b.unknown_zones
    assert "ffffffffffff:99" in lint_b.bad_citations


def test_dossier_lint_freq_mismatch():
    _, teambook, lexicon, evidence_ids = _mk_synthetic_bundle()
    # In synthetic teambook, on full_buy attacks Middle 75%. 12% is a mismatch for Middle.
    bad_freq = "On full_buy attacks `Middle` (evidence deadbeefcafe:7) 12% of rounds"
    lint = lint_dossier(bad_freq, teambook, lexicon, evidence_ids)
    assert not lint.ok
    assert len(lint.freq_mismatches) > 0


def test_exemplar_selection_covers_top_tendencies():
    scripts, teambook, _, _ = _mk_synthetic_bundle()
    ex = select_exemplars(teambook, scripts, cap=12)
    assert len(ex) <= 12
    assert len(ex) > 0
    top = max(teambook.tendencies, key=lambda t: t.n)
    assert any(f"{s.match_id}:{s.round_num}" in top.evidence for s in ex)


def test_generate_dossier_offline():
    scripts, teambook, lexicon, _ = _mk_synthetic_bundle()
    card = _mk_synthetic_card()

    valid_text = (
        "# 1. Identity & Overview\n"
        "Team abc on `de_anubis`.\n\n"
        "# 2. Defaults & Roles\n"
        "p1 plays `Middle`.\n\n"
        "# 3. Execute Repertoire with Counters\n"
        "Attacks `Middle` (evidence m1:1) 75%.\n\n"
        "# 4. Economy Policy with Exploit\n"
        "Buys on full_buy.\n\n"
        "# 5. Player-Specific Weaknesses\n"
        "Player p1 opening duel tendencies.\n\n"
        "# 6. Round-State Playbook Table\n"
        "| State | Play |\n|---|---|\n\n"
        "# 7. Confidence & Evidence Appendix\n"
        "Sample size n=12 (evidence m1:1).\n"
    )
    client = ScriptedClient([valid_text])
    dossier = generate(client, card, teambook, scripts, lexicon)
    assert isinstance(dossier, Dossier)
    assert dossier.lint.ok
    assert dossier.text == valid_text
    assert len(client.calls) == 1
    assert dossier.usage.input_tokens == 150


def test_generate_dossier_retry_on_bad_lint():
    scripts, teambook, lexicon, _ = _mk_synthetic_bundle()
    card = _mk_synthetic_card()

    bad_text = "# 1. Identity & Overview\nAttacks `Ghost` (evidence fake:99) 50%.\n"
    good_text = "# 1. Identity & Overview\nAttacks `Middle` (evidence m1:1) 75%.\n"
    client = ScriptedClient([bad_text, good_text])
    dossier = generate(client, card, teambook, scripts, lexicon)
    assert len(client.calls) == 2
    assert "Previous draft had errors" in client.calls[1]["user"]
    assert "Ghost" in client.calls[1]["user"]
    assert "fake:99" in client.calls[1]["user"]
    assert dossier.lint.ok
    assert dossier.text == good_text
    assert dossier.usage.input_tokens == 300  # 150 + 150


def test_generate_dossier_retry_still_bad_returns_flagged():
    scripts, teambook, lexicon, _ = _mk_synthetic_bundle()
    card = _mk_synthetic_card()

    bad_text_1 = "Attacks `Ghost` (evidence fake:99) 50%."
    bad_text_2 = "Still attacks `Ghost` (evidence fake:99) 50%."
    client = ScriptedClient([bad_text_1, bad_text_2])
    dossier = generate(client, card, teambook, scripts, lexicon)
    assert len(client.calls) == 2
    assert not dossier.lint.ok
    assert "Ghost" in dossier.lint.unknown_zones
    assert "fake:99" in dossier.lint.bad_citations


@pytest.mark.live
def test_live_dossier():
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("no LLM API key configured in environment")

    cfg = AppConfig.load()
    client = make_client(cfg)
    scripts, teambook, lexicon, _ = _mk_synthetic_bundle()
    card = _mk_synthetic_card()

    dossier = generate(client, card, teambook, scripts, lexicon)
    assert dossier.lint.ok, f"Dossier lint failed: {dossier.lint}"
    required_sections = [
        "Identity & Overview",
        "Defaults & Roles",
        "Execute Repertoire with Counters",
        "Gaps & Triggers",
        "Economy Policy with Exploit",
        "Player-Specific Weaknesses",
        "Round-State Playbook Table",
        "Confidence & Evidence Appendix",
    ]
    for sec in required_sections:
        assert sec.lower() in dossier.text.lower(), f"Missing section: {sec}"
