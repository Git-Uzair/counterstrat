from typing import Any

import polars as pl
import pytest

from counterstrat.llm.base import ChatTurn, LLMResult, ToolSpec
from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import (
    Lexicon,
    ZoneDef,
    _compute_checksum,
    build_lexicon,
    get_default_overlay_path,
)
from counterstrat.mapcard.quiz import generate_quiz, grade
from counterstrat.mapcard.transitions import EdgeStat, ZoneGraph, zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places


class BaseFakeClient:
    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
        raise NotImplementedError

    def complete_json(
        self, *, system: str, user: str, schema: Any, max_tokens: int = 4096
    ) -> tuple[Any, LLMResult]:
        raise NotImplementedError

    def chat(
        self,
        *,
        system: str,
        turns: list[ChatTurn],
        tools: list[ToolSpec],
        max_tokens: int = 4096,
    ) -> tuple[ChatTurn, LLMResult]:
        raise NotImplementedError


class ScriptedAnswerer(BaseFakeClient):
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
        for prompt, ans in self.mapping.items():
            if prompt in user:
                return LLMResult(text=ans)
        return LLMResult(text="A")


class ScriptedClient(BaseFakeClient):
    """Answers everything with option 'A'."""

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
        return LLMResult(text="A")


class MalformedClient(BaseFakeClient):
    """Answers with text that contains no choice letter."""

    def complete(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
        return LLMResult(text="I am unsure of the layout.")


@pytest.fixture(scope="session")
def anubis_card(anubis_assets, anubis_lake):
    places = unique_places(parse_places(anubis_assets.vents))
    overlay = get_default_overlay_path("de_anubis")
    lexicon = build_lexicon("de_anubis", places, overlay)
    ticks = pl.read_parquet(anubis_lake.ticks)
    rounds = pl.read_parquet(anubis_lake.rounds)
    graph = zone_graph(ticks)
    return compile_card(
        lexicon=lexicon,
        graph=graph,
        ticks=ticks,
        rounds=rounds,
        map_name="de_anubis",
        patch_version="14178",
    )


@pytest.fixture
def synthetic_card() -> MapCard:
    zones = {
        "BombsiteA": ZoneDef(
            id="BombsiteA", aliases=["A site"], tags=["site"], engine_place="BombsiteA"
        ),
        "BombsiteB": ZoneDef(
            id="BombsiteB", aliases=["B site"], tags=["site"], engine_place="BombsiteB"
        ),
        "Middle": ZoneDef(
            id="Middle", aliases=["mid"], tags=["mid_control"], engine_place="Middle"
        ),
        "CTSpawn": ZoneDef(id="CTSpawn", aliases=[], tags=[], engine_place="CTSpawn"),
        "TSpawn": ZoneDef(id="TSpawn", aliases=[], tags=[], engine_place="TSpawn"),
        "Ruins": ZoneDef(id="Ruins", aliases=[], tags=["choke"], engine_place="Ruins"),
    }
    lex = Lexicon(map_name="de_dust2", zones=zones, checksum=_compute_checksum("de_dust2", zones))
    graph = ZoneGraph(
        edges={
            ("BombsiteA", "Middle"): EdgeStat(n=10, median_transit_s=4.0),
            ("Middle", "BombsiteA"): EdgeStat(n=10, median_transit_s=4.0),
            ("Middle", "BombsiteB"): EdgeStat(n=10, median_transit_s=5.0),
            ("BombsiteB", "Middle"): EdgeStat(n=10, median_transit_s=5.0),
            ("BombsiteA", "CTSpawn"): EdgeStat(n=8, median_transit_s=6.0),
            ("CTSpawn", "BombsiteA"): EdgeStat(n=8, median_transit_s=6.0),
            ("CTSpawn", "BombsiteB"): EdgeStat(n=8, median_transit_s=7.0),
            ("BombsiteB", "CTSpawn"): EdgeStat(n=8, median_transit_s=7.0),
            ("Middle", "Ruins"): EdgeStat(n=6, median_transit_s=3.0),
            ("Ruins", "Middle"): EdgeStat(n=6, median_transit_s=3.0),
        }
    )
    ticks = pl.DataFrame(
        {
            "steamid": [1] * 6 + [2] * 6,
            "round_num": [1] * 12,
            "tick": list(range(0, 12 * 16, 16)),
            "clock_s": [1.0, 3.0, 5.0, 8.0, 10.0, 12.0, 2.0, 4.0, 6.0, 9.0, 11.0, 14.0],
            "last_place_name": [
                "CTSpawn",
                "Middle",
                "BombsiteA",
                "BombsiteB",
                "Ruins",
                "Middle",
                "TSpawn",
                "Middle",
                "BombsiteB",
                "BombsiteA",
                "Ruins",
                "Middle",
            ],
            "X": [100.0] * 12,
            "Y": [100.0] * 12,
            "Z": [10.0] * 12,
            "team_name": ["CT"] * 6 + ["TERRORIST"] * 6,
            "is_alive": [True] * 12,
        }
    )
    rounds = pl.DataFrame({"round_num": [1], "start": [0], "freeze_end": [0], "end": [200]})
    return compile_card(
        lexicon=lex,
        graph=graph,
        ticks=ticks,
        rounds=rounds,
        map_name="de_dust2",
        patch_version="14178",
    )


@pytest.mark.demo
def test_quiz_generation_deterministic(anubis_card):
    q1 = generate_quiz(anubis_card, n=20, seed=1)
    q2 = generate_quiz(anubis_card, n=20, seed=1)
    assert [q.prompt for q in q1] == [q.prompt for q in q2]
    assert all(q.answer in q.options for q in q1)
    assert len(q1) == 20
    assert all(q.kind in {"adjacency", "rotate", "timing"} for q in q1)


@pytest.mark.demo
def test_grading_counts_correct(anubis_card):
    quiz = generate_quiz(anubis_card, n=10, seed=1)
    always_right = ScriptedAnswerer({q.prompt: q.answer for q in quiz})
    res = grade(always_right, quiz, card_yaml=None)
    assert res.accuracy == 1.0
    assert res.correct == 10
    assert res.total == 10
    assert len(res.transcript) == 10
    assert all(t["correct"] for t in res.transcript)


def test_synthetic_card_deterministic(synthetic_card):
    q1 = generate_quiz(synthetic_card, n=15, seed=42)
    q2 = generate_quiz(synthetic_card, n=15, seed=42)
    assert [q.prompt for q in q1] == [q.prompt for q in q2]
    assert all(q.answer in q.options for q in q1)
    assert len(q1) == 15


def test_grading_with_scripted_client(synthetic_card):
    quiz = generate_quiz(synthetic_card, n=12, seed=42)
    client_a = ScriptedClient()
    res = grade(client_a, quiz, card_yaml=None)
    expected_correct = sum(1 for q in quiz if q.answer == "A")
    assert res.correct == expected_correct
    assert res.total == 12
    assert res.accuracy == round(expected_correct / 12, 4)


def test_grading_malformed_client(synthetic_card):
    quiz = generate_quiz(synthetic_card, n=6, seed=42)
    client_malformed = MalformedClient()
    res = grade(client_malformed, quiz, card_yaml=None)
    assert res.accuracy == 0.0
    assert res.correct == 0
    assert all(not t["correct"] for t in res.transcript)
    assert all(t["extracted"] is None for t in res.transcript)


def test_grade_with_card_yaml(synthetic_card):
    quiz = generate_quiz(synthetic_card, n=5, seed=7)
    card_yaml = synthetic_card.to_yaml()

    recorded_systems: list[str] = []

    class InspectingClient(BaseFakeClient):
        def complete(self, *, system: str, user: str, max_tokens: int = 512) -> LLMResult:
            recorded_systems.append(system)
            return LLMResult(text="A")

    res = grade(InspectingClient(), quiz, card_yaml=card_yaml)
    assert res.total == 5
    assert len(recorded_systems) == 5
    assert all("```yaml" in s and "de_dust2" in s for s in recorded_systems)


def test_quiz_kinds_breakdown(synthetic_card):
    quiz = generate_quiz(synthetic_card, n=3, seed=42)
    always_right = ScriptedAnswerer({q.prompt: q.answer for q in quiz})
    res = grade(always_right, quiz, card_yaml=None)
    assert res.accuracy == 1.0
    for kind in ["adjacency", "rotate", "timing"]:
        assert kind in res.breakdown
        assert res.breakdown[kind]["total"] == 1
        assert res.breakdown[kind]["correct"] == 1
        assert res.breakdown[kind]["accuracy"] == 1.0


def test_empty_card_returns_empty_quiz():
    empty_lex = Lexicon(map_name="empty", zones={}, checksum=_compute_checksum("empty", {}))
    empty_card = compile_card(
        lexicon=empty_lex,
        graph=ZoneGraph(edges={}),
        ticks=pl.DataFrame(),
        rounds=pl.DataFrame(),
        map_name="empty",
        patch_version="1.0",
    )
    quiz = generate_quiz(empty_card, n=10, seed=0)
    assert quiz == []
    empty_res = grade(ScriptedClient(), quiz)
    assert empty_res.total == 0
    assert empty_res.accuracy == 0.0
