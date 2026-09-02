"""Map Card quiz harness: auto-graded evaluation of map knowledge (spec Phase 2a, §6.7-6)."""

import random
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from counterstrat.llm.base import LLMClient
from counterstrat.mapcard.compile import MapCard


class OptionsList(list):
    """List of multiple-choice option strings that supports letter membership testing.

    Allows `"A" in options` to evaluate `True` if any option starts with `"A)"` or `"A."`.
    """

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if isinstance(item, str):
            for opt in self:
                if isinstance(opt, str) and (
                    opt == item or opt.startswith((f"{item})", f"{item}."))
                ):
                    return True
        return False


class QuizQ(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: str
    prompt: str
    options: OptionsList
    answer: str

    @field_validator("options", mode="before")
    @classmethod
    def wrap_options(cls, v: Any) -> OptionsList:
        if isinstance(v, list) and not isinstance(v, OptionsList):
            return OptionsList(v)
        return v


class QuizResult(BaseModel):
    total: int
    correct: int
    accuracy: float
    breakdown: dict[str, dict[str, Any]]
    transcript: list[dict[str, Any]]


def _gen_adjacency(card: MapCard, zones_all: list[str], rng: random.Random) -> QuizQ | None:
    candidate_zones: list[str] = []
    for u, nbrs in card.topology.items():
        if u in zones_all and len(nbrs) >= 1:
            candidate_zones.append(u)
    if not candidate_zones:
        return None

    candidate_zones.sort()

    # Prefer: zone with >= 3 neighbors and >= 1 non-neighbor ("Which zone is NOT directly connected?")
    ideal_neg: list[str] = []
    for u in candidate_zones:
        direct = [v for v in card.topology[u] if v in zones_all and v != u]
        non = [z for z in zones_all if z != u and z not in direct]
        if len(direct) >= 3 and len(non) >= 1:
            ideal_neg.append(u)

    if ideal_neg:
        u = rng.choice(ideal_neg)
        direct = sorted([v for v in card.topology[u] if v in zones_all and v != u])
        non = sorted([z for z in zones_all if z != u and z not in direct])
        pos_sample = rng.sample(direct, 3)
        neg_sample = rng.choice(non)
        choices = pos_sample + [neg_sample]
        rng.shuffle(choices)
        ans_idx = choices.index(neg_sample)
        options = OptionsList([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
        answer = chr(65 + ans_idx)
        prompt = f"Which zone is NOT directly connected to {u}?"
        return QuizQ(kind="adjacency", prompt=prompt, options=options, answer=answer)

    # Secondary: zone with >= 1 neighbor and >= 3 non-neighbors ("Which zone IS directly connected?")
    ideal_pos: list[str] = []
    for u in candidate_zones:
        direct = [v for v in card.topology[u] if v in zones_all and v != u]
        non = [z for z in zones_all if z != u and z not in direct]
        if len(direct) >= 1 and len(non) >= 3:
            ideal_pos.append(u)

    if ideal_pos:
        u = rng.choice(ideal_pos)
        direct = sorted([v for v in card.topology[u] if v in zones_all and v != u])
        non = sorted([z for z in zones_all if z != u and z not in direct])
        pos_sample = rng.choice(direct)
        neg_sample = rng.sample(non, 3)
        choices = [pos_sample] + neg_sample
        rng.shuffle(choices)
        ans_idx = choices.index(pos_sample)
        options = OptionsList([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
        answer = chr(65 + ans_idx)
        prompt = f"Which zone IS directly connected to {u}?"
        return QuizQ(kind="adjacency", prompt=prompt, options=options, answer=answer)

    # Fallback: 2-choice question
    fallback: list[str] = []
    for u in candidate_zones:
        direct = [v for v in card.topology[u] if v in zones_all and v != u]
        non = [z for z in zones_all if z != u and z not in direct]
        if len(direct) >= 1 and len(non) >= 1:
            fallback.append(u)
    if fallback:
        u = rng.choice(fallback)
        direct = sorted([v for v in card.topology[u] if v in zones_all and v != u])
        non = sorted([z for z in zones_all if z != u and z not in direct])
        pos_sample = rng.choice(direct)
        neg_sample = rng.choice(non)
        choices = [pos_sample, neg_sample]
        rng.shuffle(choices)
        ans_idx = choices.index(pos_sample)
        options = OptionsList([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
        answer = chr(65 + ans_idx)
        prompt = f"Which zone IS directly connected to {u}?"
        return QuizQ(kind="adjacency", prompt=prompt, options=options, answer=answer)

    return None


def _gen_rotate(card: MapCard, zones_all: list[str], rng: random.Random) -> QuizQ | None:
    if not card.rotates:
        return None

    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in card.rotates:
        s1 = r.get("from")
        s2 = r.get("to")
        via = r.get("via", [])
        if s1 and s2 and via:
            key = (s1, s2)
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(r)

    if not pairs:
        return None

    sorted_pair_keys = sorted(pairs.keys())
    s1, s2 = rng.choice(sorted_pair_keys)
    routes = sorted(pairs[(s1, s2)], key=lambda x: (x.get("run_s", 0.0), x.get("via", [])))
    fastest = routes[0]
    via_zones = fastest.get("via", [])
    if not via_zones:
        return None

    correct_via = rng.choice(sorted(via_zones))
    candidate_distractors = sorted(
        [z for z in zones_all if z not in via_zones and z != s1 and z != s2]
    )

    if len(candidate_distractors) >= 3:
        distractors = rng.sample(candidate_distractors, 3)
    elif candidate_distractors:
        distractors = list(candidate_distractors)
    else:
        distractors = []

    choices = [correct_via] + distractors
    rng.shuffle(choices)
    ans_idx = choices.index(correct_via)
    options = OptionsList([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
    answer = chr(65 + ans_idx)
    prompt = f"Fastest rotate {s1}->{s2} runs through which zone?"
    return QuizQ(kind="rotate", prompt=prompt, options=options, answer=answer)


def _gen_timing(card: MapCard, rng: random.Random) -> QuizQ | None:
    available_sides: list[str] = []
    pairs_per_side: dict[str, list[tuple[str, str]]] = {}

    for side in ["T", "CT"]:
        t_dict = card.timings.get(side, {})
        if len(t_dict) >= 2:
            zones = sorted(t_dict.keys())
            pairs: list[tuple[str, str]] = []
            for i in range(len(zones)):
                for j in range(i + 1, len(zones)):
                    z1, z2 = zones[i], zones[j]
                    if t_dict[z1] != t_dict[z2]:
                        pairs.append((z1, z2))
            if pairs:
                available_sides.append(side)
                pairs_per_side[side] = pairs

    if not available_sides:
        return None

    side = rng.choice(sorted(available_sides))
    pair = rng.choice(pairs_per_side[side])
    z1, z2 = pair
    t_dict = card.timings[side]
    faster = z1 if t_dict[z1] < t_dict[z2] else z2

    choices = [z1, z2]
    rng.shuffle(choices)
    ans_idx = choices.index(faster)
    options = OptionsList([f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)])
    answer = chr(65 + ans_idx)
    prompt = f"Which zone can {side} reach earlier: {z1} or {z2}?"
    return QuizQ(kind="timing", prompt=prompt, options=options, answer=answer)


def generate_quiz(card: MapCard, n: int = 50, seed: int = 0) -> list[QuizQ]:
    """Generates an auto-gradable multiple choice quiz of size n from a MapCard."""
    zones_all = sorted(card.zones.keys()) if card.zones else sorted(card.topology.keys())
    if not zones_all:
        return []

    rng = random.Random(seed)

    available_kinds: list[str] = []
    if _gen_adjacency(card, zones_all, random.Random(0)) is not None:
        available_kinds.append("adjacency")
    if _gen_rotate(card, zones_all, random.Random(0)) is not None:
        available_kinds.append("rotate")
    if _gen_timing(card, random.Random(0)) is not None:
        available_kinds.append("timing")

    if not available_kinds:
        return []

    quiz: list[QuizQ] = []
    kind_idx = 0
    attempts = 0
    max_attempts = n * 50

    while len(quiz) < n and attempts < max_attempts:
        attempts += 1
        kind = available_kinds[kind_idx % len(available_kinds)]
        kind_idx += 1

        q: QuizQ | None = None
        if kind == "adjacency":
            q = _gen_adjacency(card, zones_all, rng)
        elif kind == "rotate":
            q = _gen_rotate(card, zones_all, rng)
        elif kind == "timing":
            q = _gen_timing(card, rng)

        if q is not None:
            quiz.append(q)

    return quiz


def extract_letter(text: str) -> str | None:
    """Extracts option letter A-D from LLM response text."""
    text = text.strip()
    if not text:
        return None
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    m_lead = re.match(r"^[\(\[]?([A-D])[\)\]\.:\s]", text, re.IGNORECASE)
    if m_lead:
        return m_lead.group(1).upper()
    m_ans = re.search(r"(?:answer|option)[\s:]*([A-D])\b", text, re.IGNORECASE)
    if m_ans:
        return m_ans.group(1).upper()
    m_word = re.search(r"\b([A-D])\b", text)
    if m_word:
        return m_word.group(1).upper()
    return None


def grade(
    client: LLMClient,
    quiz: list[QuizQ],
    card_yaml: str | None = None,
) -> QuizResult:
    """Grades an LLM client on a quiz, optionally providing Map Card in YAML format."""
    system = (
        "You are an expert Counter-Strike 2 analyst taking a map knowledge quiz.\n"
        "Answer with ONLY the letter of the correct option (e.g. A, B, C, or D).\n"
        "Do not provide any explanation."
    )
    if card_yaml:
        system += f"\n\nHere is the map reference card:\n```yaml\n{card_yaml}\n```"

    transcript: list[dict[str, Any]] = []
    breakdown: dict[str, dict[str, Any]] = {}
    correct_count = 0

    for q in quiz:
        user = f"{q.prompt}\n" + "\n".join(q.options)
        res = client.complete(system=system, user=user)
        extracted = extract_letter(res.text)
        is_correct = bool(extracted and extracted == q.answer)
        if is_correct:
            correct_count += 1

        if q.kind not in breakdown:
            breakdown[q.kind] = {"total": 0, "correct": 0, "accuracy": 0.0}
        breakdown[q.kind]["total"] += 1
        if is_correct:
            breakdown[q.kind]["correct"] += 1

        transcript.append(
            {
                "kind": q.kind,
                "prompt": q.prompt,
                "options": list(q.options),
                "answer": q.answer,
                "response": res.text,
                "extracted": extracted,
                "correct": is_correct,
            }
        )

    for kind_stats in breakdown.values():
        tot = kind_stats["total"]
        kind_stats["accuracy"] = round(kind_stats["correct"] / tot, 4) if tot > 0 else 0.0

    total = len(quiz)
    accuracy = round(correct_count / total, 4) if total > 0 else 0.0

    return QuizResult(
        total=total,
        correct=correct_count,
        accuracy=accuracy,
        breakdown=breakdown,
        transcript=transcript,
    )


if __name__ == "__main__":
    import argparse
    import os
    import sys
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Map Card quiz harness")
    parser.add_argument("map", nargs="?", default="de_anubis", help="Map name (e.g. de_anubis)")
    parser.add_argument("-n", type=int, default=50, help="Number of questions (default: 50)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--card-path",
        type=str,
        default=None,
        help="Path to compiled card.yaml (defaults to data/mapcards/<map>/card.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quiz results JSON (defaults to data/mapcards/<map>/quiz_results.json)",
    )
    args = parser.parse_args()

    map_name = args.map
    card_path = (
        Path(args.card_path)
        if args.card_path
        else Path("data") / "mapcards" / map_name / "card.yaml"
    )

    if not card_path.exists():
        print(f"Error: Map card file not found at {card_path}", file=sys.stderr)
        sys.exit(1)

    with open(card_path, "r", encoding="utf-8") as f:
        card_yaml_content = f.read()

    loaded_card = MapCard.model_validate(yaml.safe_load(card_yaml_content))
    quiz_items = generate_quiz(loaded_card, n=args.n, seed=args.seed)
    print(f"Generated {len(quiz_items)} questions for {map_name} (seed={args.seed})")

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))

    if not (has_anthropic or has_gemini):
        print(
            "Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY is set. Skipping live model evaluation."
        )
        sys.exit(0)

    print("Live model evaluation requires Task 18 LLM adapters.")
