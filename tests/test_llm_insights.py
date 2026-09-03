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

    def complete(self, *, system: str, user: str, max_tokens: int | None = None):
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
    assert "1 game(s), 9 rounds" in user
    assert "## Games" in user
    assert "## Tendencies" in user  # teambook table
    assert "## Utility Book" in user
    assert "## Gap Findings" in user
    assert "## Economy Policy" in user
    assert "## Player Profiles" in user
    assert "## All Round Scripts" in user
    # Full scripts, movements included, for every round - in human game language.
    for s in synthetic_scripts:
        assert f"### Game 1, round {s.round_num}" in user
    assert "### Game 1, round 1 (pistol round)" in user
    assert "### Game 1, round 13 (pistol round)" in user
    assert "TSpawn > Water" in user  # a movement sentence made it in
    # Internal ids and window codes never reach the prompt.
    assert "m1:" not in user
    assert "post-PL" not in user and "post-FC" not in user


def test_generate_insights_calls_llm_and_lints(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient(
        "## T Full Buy\n**Read** - They smoke `Middle` every T round (Game 1, rounds 1, 2; "
        "n=4) - hold `Water` for the wrap.\n"
    )
    out = generate_insights(
        client,
        card,
        scripts=synthetic_scripts,
        lexicon=lex,
        **b,
    )
    assert out.text.startswith("## T Full Buy")
    assert out.warnings == []  # valid zone + real rounds -> clean
    assert out.games == [{"label": "Game 1", "match_id": "m1"}]
    assert client.completions, "the LLM must actually be called"
    system = client.completions[0]["system"]
    # The mandated structure and the anti-tautology rule.
    for anchor in (
        "## T Pistol",
        "## CT Pistol",
        "## Eco & Force Habits",
        "## T Full Buy",
        "## CT Full Buy",
        "## Gotchas",
    ):
        assert anchor in system, anchor
    assert "first contact" in system
    assert "NEVER" in system and "hash" in system.lower()
    assert "normal play" in system
    assert "causal" in system.lower()


def test_generate_insights_forwards_zone_map(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient("## T Full Buy\nfine.")
    generate_insights(
        client,
        card,
        scripts=synthetic_scripts,
        lexicon=lex,
        zone_map="zone_map:\n- `Middle` at (0.53, 0.10)",
        **b,
    )
    assert "`Middle` at (0.53, 0.10)" in client.completions[0]["system"]


def test_generate_insights_flags_fabrications(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient(
        "They rush `GhostTown` (Game 1, round 99). Also (Game 7, round 1). "
        "See deadbeefcafe1234 for details."
    )
    out = generate_insights(client, card, scripts=synthetic_scripts, lexicon=lex, **b)
    assert any("GhostTown" in w for w in out.warnings)
    assert any("Game 1 round 99" in w for w in out.warnings)
    assert any("Game 7 does not exist" in w for w in out.warnings)
    assert any("hash leaked" in w for w in out.warnings)


def test_generate_insights_warns_on_truncation(synthetic_scripts):
    from counterstrat.llm.base import LLMResult

    class _TruncatedClient(_CompleteClient):
        def complete(self, *, system, user, max_tokens=None):
            return LLMResult(
                text="## Read\nCut mid-",
                input_tokens=1,
                output_tokens=65536,
                model="m",
                provider="mock",
                truncated=True,
            )

    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    out = generate_insights(_TruncatedClient(""), card, scripts=synthetic_scripts, lexicon=lex, **b)
    assert any("token ceiling" in w for w in out.warnings)


def test_insights_payload_roundtrips_json(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    card = build_synthetic_card()
    lex = build_lexicon("de_anubis", list(card.zones.keys()))
    client = _CompleteClient("## Gotchas\nSolid `Middle` control (Game 1, round 1).")
    out = generate_insights(client, card, scripts=synthetic_scripts, lexicon=lex, **b)
    blob = json.loads(out.model_dump_json())
    assert blob["text"] and blob["generated_from"] == ["m1"]
    assert blob["games"] == [{"label": "Game 1", "match_id": "m1"}]


def test_custom_game_labels_flow_through(synthetic_scripts):
    b = _bundle(synthetic_scripts)
    user = build_insights_user(
        scripts=synthetic_scripts, game_labels={"m1": "Game 1 (vs team_xyz)"}, **b
    )
    assert "- Game 1 (vs team_xyz)" in user
    assert "### Game 1 (vs team_xyz), round 4" in user
