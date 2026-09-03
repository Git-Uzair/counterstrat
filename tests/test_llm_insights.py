"""LLM First Read insights tests (feedback round: dynamic insights)."""

import json

from conftest import SYNTHETIC_TEAM, ScriptedToolClient, build_synthetic_card

from counterstrat.llm.insights import build_insights_user, generate_insights
from counterstrat.mapcard.lexicon import build_lexicon
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.tendencies import build_teambook
from counterstrat.mining.utility_book import build_utility_book


class _CompleteClient(ScriptedToolClient):
    """ScriptedToolClient with the complete() path the insights generator uses."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text
        self.completions: list[dict] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 4096):
        from counterstrat.llm.base import LLMResult

        self.completions.append({"system": system, "user": user})
        return LLMResult(
            text=self.text,
            input_tokens=1000,
            output_tokens=200,
            model="mock-model",
            provider="mock",
        )


def _bundle(synthetic_scripts):
    teambook = build_teambook(synthetic_scripts, SYNTHETIC_TEAM)
    return {
        "teambook": teambook,
        "utility_book": build_utility_book(synthetic_scripts, SYNTHETIC_TEAM),
        "gap_report": build_gap_report(synthetic_scripts, SYNTHETIC_TEAM),
        "econ_policy": build_econ_policy(synthetic_scripts, SYNTHETIC_TEAM),
    }


def test_insights_user_prompt_carries_everything(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    user = build_insights_user(scripts=synthetic_scripts, **b)
    # Coverage counts and every data artifact.
    assert "1 demo(s), 9 rounds" in user
    assert "## Tendencies" in user  # teambook table
    assert "## Utility Book" in user
    assert "## Gap Findings" in user
    assert "## Economy Policy" in user
    assert "## Player Profiles" in user
    assert "## All Round Scripts" in user
    # Full scripts, movements included, for every round.
    for s in synthetic_scripts:
        assert f"### {s.match_id}:{s.round_num}" in user
    assert "TSpawn > Water" in user  # a movement sentence made it in


def test_generate_insights_calls_llm_and_lints(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient(
        "## 1. Mid crutch\nThey smoke `Middle` every T round (m1:2, n=4) - "
        "hold `Water` for the wrap.\n"
    )
    out = generate_insights(
        client,
        card,
        scripts=synthetic_scripts,
        lexicon=lex,
        **b,
    )
    assert out.text.startswith("## 1. Mid crutch")
    assert out.warnings == []  # valid zone + citation -> clean lint
    assert client.completions, "the LLM must actually be called"
    system = client.completions[0]["system"]
    # The anti-tautology rule is the whole point of the dynamic read.
    assert "normal play" in system
    assert "causal" in system.lower()


def test_generate_insights_flags_fabrications(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient("They rush `GhostTown` (evidence fake:99).")
    out = generate_insights(client, card, scripts=synthetic_scripts, lexicon=lex, **b)
    assert any("GhostTown" in w for w in out.warnings)
    assert any("fake:99" in w for w in out.warnings)


def test_insights_payload_roundtrips_json(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient("## Read\nSolid `Middle` control (m1:1).")
    out = generate_insights(client, card, scripts=synthetic_scripts, lexicon=lex, **b)
    blob = json.loads(out.model_dump_json())
    assert blob["text"] and blob["generated_from"] == ["m1"]
