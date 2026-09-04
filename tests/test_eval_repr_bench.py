"""Representation benchmark: deterministic LLM-free question generation.

The three GT-artifact classes found in the 2026-09-04 lab are excluded by
construction: boundary joins (>=2s inside a stint), entry ties (>=2s lead),
and 'entered N seconds later' when someone was already inside the zone.
"""

import random

import pytest

from counterstrat.eval.repr_bench import (
    Question,
    RoundContext,
    gen_temporal_questions,
    grade_answer,
    run_arm,
)
from counterstrat.roundscript.models import ZoneStint


@pytest.fixture
def rounds(synthetic_scripts):
    rng = random.Random(3)
    out = []
    for i, s in enumerate(synthetic_scripts):
        zones = ["TSpawn", "Middle", "BombsiteA", "Water", "CTSpawn"]
        tracks = {}
        sides = {}
        for pi in range(4):
            p = f"pl{pi}"
            sides[p] = "T" if pi < 2 else "CT"
            t = 0
            stints = []
            for z in rng.sample(zones, 3):
                dur = rng.randint(5, 20)
                stints.append(ZoneStint(t0=t, t1=t + dur, zone=z))
                t += dur
            tracks[p] = stints
        script = s.model_copy(update={"tracks": tracks, "sides": sides})
        out.append(RoundContext(rid=f"m{i}", script=script, tracks=tracks, sides=sides))
    return out


def test_question_generation_deterministic(rounds):
    a = gen_temporal_questions(rounds, seed=11, per_kind=5)
    b = gen_temporal_questions(rounds, seed=11, per_kind=5)
    assert [(q.qid, q.answer, q.prompt) for q in a] == [(q.qid, q.answer, q.prompt) for q in b]
    assert a, "no questions generated"
    for q in a:
        if q.grade == "numeric":
            float(q.answer)  # parses
        else:
            assert q.answer in "ABCD"
            # exactly one option carries the answer letter
            assert f"{q.answer})" in q.prompt


def test_ground_truth_needs_no_llm(rounds):
    """Generation touches no client; run_arm is the only API surface."""

    class Boom:
        def complete(self, **_):
            raise AssertionError("LLM called during generation")

    qs = gen_temporal_questions(rounds, seed=11, per_kind=5)
    assert qs  # generated without any client existing
    # and run_arm surfaces client errors as row errors, not crashes
    out = run_arm(Boom(), "x", lambda q: "block", qs[:1], workers=1)
    assert out["errors"] == 1


def test_join_questions_have_margins(rounds):
    qs = gen_temporal_questions(rounds, seed=11, per_kind=10)
    by_kind: dict[str, list[Question]] = {}
    for q in qs:
        by_kind.setdefault(q.kind, []).append(q)
    rid = {r.rid: r for r in rounds}
    for q in by_kind.get("who_where_when", []):
        r = rid[q.meta["round"]]
        # reconstruct: the asked player's stint must contain a kill moment
        # with >= 2s margin on both sides
        import re

        m = re.search(r"which zone was (\S+) \(", q.prompt)
        assert m, q.prompt
        player = m.group(1)
        kills = {(k.killer, k.victim): k.t for k in r.script.kills}
        assert any(st.t0 + 2 <= t <= st.t1 - 2 for t in kills.values() for st in r.tracks[player])


def test_grade_answer_tolerance():
    q = Question(qid="x", kind="k", prompt="p", answer="25", grade="numeric", tol=3.0)
    assert grade_answer(q, "I think 27")
    assert not grade_answer(q, "30")
    m = Question(qid="y", kind="k", prompt="p", answer="B")
    assert grade_answer(m, "B) Banana")
    assert not grade_answer(m, "A")
