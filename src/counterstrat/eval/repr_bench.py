"""Representation benchmark: map/round prompt formats graded against ground truth.

Port of the 2026-09-04 lab (docs/plans/2026-09-04-llm-spatiotemporal-
representation.md) that selected the scene-graph map block and the timeline
round text. Question generation is deterministic and never calls an LLM:
answers come from Dijkstra over the card topology, anchor bearings, and the
scripts' own kill/utility/track data. Rerun whenever a prompt format changes:

    uv run python -m counterstrat.eval.repr_bench --suite spatial
    uv run python -m counterstrat.eval.repr_bench --suite temporal --arms current,timeline
"""

import heapq
import itertools
import json
import math
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

from counterstrat.config import AppConfig
from counterstrat.llm.base import LLMClient
from counterstrat.llm.prompts import format_map_scene_graph, format_zone_map
from counterstrat.mapcard.compile import MapCard
from counterstrat.roundscript.models import RoundScript, ZoneStint
from counterstrat.roundscript.movement import zone_stints

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
LETTER_RE = re.compile(r"\b([A-D])\b")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

SYSTEM_TMPL = """You are a Counter-Strike 2 tactical analyst. Answer questions about the map or a
played round using ONLY the reference data below. Be precise.

{block}

Answer format: reply with ONLY the final answer - a single option letter for multiple choice,
or a single number for numeric questions. No explanation."""


@dataclass
class Question:
    qid: str
    kind: str
    prompt: str
    answer: str  # "A".."D" or a number-string
    grade: str = "mcq"  # "mcq" | "numeric"
    tol: float = 0.0
    meta: dict = field(default_factory=dict)


def grade_answer(q: Question, text: str) -> bool:
    text = text.strip()
    if q.grade == "mcq":
        m = LETTER_RE.search(text)
        return bool(m and m.group(1) == q.answer)
    m = NUM_RE.search(text.replace(",", ""))
    return bool(m) and abs(float(m.group(0)) - float(q.answer)) <= q.tol


def shuffled_mcq(rng: random.Random, correct: str, distractors: list[str]) -> tuple[str, str]:
    opts = [correct] + distractors
    rng.shuffle(opts)
    letter = chr(65 + opts.index(correct))
    return "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(opts)), letter


# ------------------------------------------------------------------ spatial
class MapContext:
    """Card + anchors + the undirected seconds graph every spatial arm shares."""

    def __init__(self, cfg: AppConfig, map_name: str):
        from counterstrat.web.routes import map_zone_anchors

        card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
        self.card = MapCard(**yaml.safe_load(card_path.read_text(encoding="utf-8")))
        self.anchors: dict[str, tuple] = map_zone_anchors(cfg, map_name)
        g: dict[str, dict[str, float]] = {}
        for u, nbrs in (self.card.topology or {}).items():
            for v, w in nbrs.items():
                w = float(w)
                if w <= 0 or u == v:
                    continue
                cur = g.setdefault(u, {}).get(v)
                if cur is None or w < cur:
                    g.setdefault(u, {})[v] = w
                    g.setdefault(v, {})[u] = w
        self.graph = g
        self.zones = sorted(self.card.zones)
        self.graph_zones = sorted(z for z in self.zones if g.get(z))

    def dijkstra(self, src: str, banned: str | None = None) -> dict[str, float]:
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in self.graph.get(u, {}).items():
                if v == banned:
                    continue
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def shortest_path(
        self, src: str, dst: str, banned: str | None = None
    ) -> tuple[list[str], float]:
        dist = {src: 0.0}
        prev: dict[str, str] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in self.graph.get(u, {}).items():
                if v == banned:
                    continue
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return [], math.inf
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        return path[::-1], dist[dst]

    def bearing(self, a: str, b: str) -> tuple[str, float, float]:
        ua, va, _ = self.anchors[a]
        ub, vb, _ = self.anchors[b]
        du, dv = ub - ua, vb - va
        ang = math.degrees(math.atan2(du, -dv)) % 360
        idx = int(((ang + 22.5) % 360) // 45)
        off = abs(((ang - idx * 45) + 180) % 360 - 180)
        return SECTORS[idx], off, math.hypot(du, dv)


def spatial_arm_current(m: MapContext) -> str:
    """What the app shipped before the 2026-09-04 upgrade: frozen for regression."""
    return f"<map_card>\n{m.card.to_yaml()}{format_zone_map(m.anchors)}\n</map_card>"


def spatial_arm_scenegraph(m: MapContext) -> str:
    return format_map_scene_graph(m.card, m.anchors)


SPATIAL_ARMS: dict[str, Callable[[MapContext], str]] = {
    "current": spatial_arm_current,
    "scenegraph": spatial_arm_scenegraph,
}


def gen_spatial_questions(m: MapContext, seed: int = 7, per_kind: int = 10) -> list[Question]:
    rng = random.Random(seed)
    qs: list[Question] = []
    coord_note = (
        "u: 0=west edge -> 1=east edge, v: 0=NORTH edge -> 1=SOUTH edge (v grows southward)"
    )

    # nearest_zone: ground a coordinate into a callout (needs anchors)
    zone_items = [(z, a) for z, a in sorted(m.anchors.items()) if z in m.zones]
    rng.shuffle(zone_items)
    for i, (zone, (u, v, _lvl)) in enumerate(zone_items[:per_kind]):
        d = sorted(
            (math.hypot(u - a[0], v - a[1]), z2) for z2, a in m.anchors.items() if z2 != zone
        )
        if d[0][0] < 0.04:  # too close to another anchor: ambiguous point
            continue
        distractors = [z2 for _, z2 in d[:3]]
        opts, letter = shuffled_mcq(rng, zone, distractors)
        qs.append(
            Question(
                qid=f"nearest_{i}",
                kind="nearest_zone",
                prompt=(
                    f"A player is standing at (u={u:.2f}, v={v:.2f}) ({coord_note}). "
                    f"Which callout zone are they in?\n{opts}"
                ),
                answer=letter,
            )
        )

    # direction: compass bearing between distant zones
    pairs = []
    for a in m.zones:
        for b in m.zones:
            if a < b and a in m.anchors and b in m.anchors:
                sec, off, dist = m.bearing(a, b)
                if dist >= 0.30 and off <= 30:
                    pairs.append((a, b, sec))
    rng.shuffle(pairs)
    for i, (a, b, sec) in enumerate(pairs[:per_kind]):
        opp = SECTORS[(SECTORS.index(sec) + 4) % 8]
        others = [s for s in SECTORS if s not in (sec, opp)]
        opts, letter = shuffled_mcq(rng, sec, [opp] + rng.sample(others, 2))
        qs.append(
            Question(
                qid=f"dir_{i}",
                kind="direction",
                prompt=(
                    f"Standing in `{a}`, in which compass direction is `{b}`? "
                    f"(radar convention: north = top of map)\n{opts}"
                ),
                answer=letter,
            )
        )

    # on_path + route_time: fastest-route reasoning
    paths = []
    for src in m.graph_zones:
        for dst in m.graph_zones:
            if src < dst:
                path, cost = m.shortest_path(src, dst)
                if len(path) >= 4 and cost < math.inf:
                    paths.append((src, dst, path, cost))
    rng.shuffle(paths)
    n_path = 0
    for src, dst, path, cost in paths:
        if n_path >= per_kind:
            break
        mid = path[len(path) // 2]
        off_path = []
        for z in m.graph_zones:
            if z in path:
                continue
            _, c1 = m.shortest_path(src, z)
            _, c2 = m.shortest_path(z, dst)
            if c1 + c2 >= cost * 1.4:
                off_path.append(z)
        if len(off_path) < 3:
            continue
        opts, letter = shuffled_mcq(rng, mid, rng.sample(off_path, 3))
        qs.append(
            Question(
                qid=f"path_{n_path}",
                kind="on_path",
                prompt=(
                    f"Taking the FASTEST route from `{src}` to `{dst}`, which of these "
                    f"zones do you move through?\n{opts}"
                ),
                answer=letter,
            )
        )
        n_path += 1
    timed = [p for p in paths if 8 <= p[3] <= 45]
    rng.shuffle(timed)
    for i, (src, dst, _path, cost) in enumerate(timed[:per_kind]):
        qs.append(
            Question(
                qid=f"time_{i}",
                kind="route_time",
                prompt=(
                    f"How many seconds does it take to move from `{src}` to `{dst}` "
                    f"by the fastest route? Answer with a single number."
                ),
                answer=f"{cost:.0f}",
                grade="numeric",
                tol=max(3.0, cost * 0.3),
            )
        )

    # race: two players toward one target
    races = []
    for tgt in m.graph_zones:
        dist_to = m.dijkstra(tgt)
        starts = [z for z in m.graph_zones if z != tgt and z in dist_to]
        for _ in range(4):
            if len(starts) < 2:
                break
            a, b = rng.sample(starts, 2)
            da, db = dist_to[a], dist_to[b]
            if abs(da - db) >= max(3.0, 0.3 * min(da, db)) and min(da, db) > 3:
                races.append((a, b, tgt, da < db))
    rng.shuffle(races)
    for i, (a, b, tgt, t_first) in enumerate(races[:per_kind]):
        gt = "the T player" if t_first else "the CT player"
        opts, letter = shuffled_mcq(rng, gt, ["the CT player" if t_first else "the T player"])
        qs.append(
            Question(
                qid=f"race_{i}",
                kind="race",
                prompt=(
                    f"A T player in `{a}` and a CT player in `{b}` start moving toward "
                    f"`{tgt}` at the same moment, both at run speed along the fastest "
                    f"routes. Who arrives first?\n{opts}"
                ),
                answer=letter,
            )
        )

    # detour_time: hard constrained-routing numeric
    detours = []
    for src, dst, path, cost in paths:
        if len(path) < 3 or cost < 6:
            continue
        banned = path[len(path) // 2]
        d2 = m.dijkstra(src, banned=banned)
        if dst in d2 and d2[dst] >= cost + 4:
            detours.append((src, dst, banned, d2[dst]))
    rng.shuffle(detours)
    for i, (src, dst, banned, alt) in enumerate(detours[:per_kind]):
        qs.append(
            Question(
                qid=f"detour_{i}",
                kind="detour_time",
                prompt=(
                    f"`{banned}` is blocked and cannot be crossed. How many seconds does "
                    f"the fastest route from `{src}` to `{dst}` take now? "
                    f"Answer with a single number."
                ),
                answer=f"{alt:.0f}",
                grade="numeric",
                tol=max(3.0, alt * 0.3),
            )
        )
    return qs


# ------------------------------------------------------------------ temporal
@dataclass
class RoundContext:
    rid: str
    script: RoundScript
    tracks: dict[str, list[ZoneStint]]
    sides: dict[str, str]


def load_round_contexts(
    cfg: AppConfig, map_name: str, seed: int = 11, want: int = 12
) -> list[RoundContext]:
    """Rounds with plants and enough action, tracks recomputed from the lake
    when the on-disk script predates v2."""
    from counterstrat.corpus import load_manifest

    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    matches = [mid for mid, rec in sorted(manifest.items()) if rec.map_name == map_name]
    rng = random.Random(seed)
    cands: list[tuple[str, Path]] = []
    for mid in matches:
        for p in sorted((cfg.data_root / "scripts" / mid).glob("round_*.json")):
            cands.append((mid, p))
    rng.shuffle(cands)
    out: list[RoundContext] = []
    tick_cache: dict[str, pl.DataFrame] = {}
    for mid, path in cands:
        if len(out) >= want:
            break
        script = RoundScript.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if not script.plant or len(script.kills) < 4 or len(script.utility) < 6:
            continue
        tracks, sides = script.tracks, script.sides
        if not tracks:
            if mid not in tick_cache:
                tick_cache[mid] = pl.read_parquet(cfg.data_root / "lake" / mid / "ticks.parquet")
            tracks, sides = zone_stints(tick_cache[mid], round_num=script.round_num)
        if not tracks:
            continue
        out.append(
            RoundContext(
                rid=f"{mid[:6]}r{script.round_num}",
                script=script.model_copy(update={"tracks": tracks, "sides": sides}),
                tracks=tracks,
                sides=sides,
            )
        )
    return out


def temporal_arm_current(r: RoundContext) -> str:
    return r.script.to_text()


def temporal_arm_timeline(r: RoundContext) -> str:
    return r.script.to_timeline_text()


def temporal_arm_timeline_lite(r: RoundContext) -> str:
    return r.script.to_timeline_text(lite=True)


TEMPORAL_ARMS: dict[str, Callable[[RoundContext], str]] = {
    "current": temporal_arm_current,
    "timeline": temporal_arm_timeline,
    "timeline_lite": temporal_arm_timeline_lite,
}


def _zone_at(tracks: dict[str, list[ZoneStint]], player: str, t: float) -> ZoneStint | None:
    for st in tracks.get(player, []):
        if st.t0 <= t < st.t1:
            return st
    return None


def gen_temporal_questions(
    rounds: list[RoundContext], seed: int = 11, per_kind: int = 10
) -> list[Question]:
    rng = random.Random(seed)
    qs: list[Question] = []
    per: dict[str, int] = {}

    def want(kind: str) -> bool:
        return per.get(kind, 0) < per_kind

    def add(
        kind: str, r: RoundContext, prompt: str, answer: str, grade: str = "mcq", tol: float = 0.0
    ):
        per[kind] = per.get(kind, 0) + 1
        qs.append(
            Question(
                qid=f"{kind}_{r.rid}",
                kind=kind,
                prompt=prompt,
                answer=answer,
                grade=grade,
                tol=tol,
                meta={"round": r.rid},
            )
        )

    for r in rounds:
        s = r.script
        pl_t = s.plant.t if s.plant else None
        all_zones = {st.zone for tr in r.tracks.values() for st in tr}

        # order: which of two >=5s-apart events came first
        if want("order"):
            named = [(k.t, f"the kill of {k.victim} by {k.killer}") for k in s.kills]
            named += [
                (u.t, f"the {u.nade} by {u.thrower} landing `{u.to_zone}`") for u in s.utility
            ]
            pairs = [
                (a, b) for i, a in enumerate(named) for b in named[i + 1 :] if abs(b[0] - a[0]) >= 5
            ]
            if pairs:
                a, b = rng.choice(pairs)
                first, second = (a, b) if a[0] < b[0] else (b, a)
                items = [first[1], second[1]]
                rng.shuffle(items)
                add(
                    "order",
                    r,
                    f"Which event happened FIRST this round?\nA) {items[0]}\nB) {items[1]}",
                    "A" if items[0] == first[1] else "B",
                )

        # delta_t: first kill -> plant
        fc = s.first_contact
        if want("delta_t") and pl_t is not None and fc and abs(pl_t - fc.t) >= 5:
            add(
                "delta_t",
                r,
                "How many seconds passed between the FIRST KILL of the round and the "
                "bomb plant? Answer with a single number.",
                f"{pl_t - fc.t:.0f}",
                grade="numeric",
                tol=3.0,
            )

        # where_kill: zone of a non-first kill
        if want("where_kill") and len(s.kills) >= 3:
            k = rng.choice(s.kills[1:])
            distract = sorted(all_zones - {k.zone})
            if len(distract) >= 3:
                opts, letter = shuffled_mcq(rng, k.zone, rng.sample(distract, 3))
                add(
                    "where_kill",
                    r,
                    f"In which zone did {k.killer} kill {k.victim}?\n{opts}",
                    letter,
                )

        # count_util: T flashes before the plant
        if want("count_util") and pl_t is not None:
            n = sum(1 for u in s.utility if u.side == "T" and u.nade == "flash" and u.t < pl_t)
            add(
                "count_util",
                r,
                "How many FLASH grenades did the T side detonate BEFORE the bomb plant? "
                "Answer with a single number.",
                str(n),
                grade="numeric",
            )

        # track: a zone a player moved through early
        if want("track"):
            picks = []
            for p, stints in r.tracks.items():
                mids = [
                    st for st in stints[1:] if st.t1 - st.t0 >= 4 and 5 <= st.t0 and st.t1 <= 60
                ]
                if mids:
                    visited = {st.zone for st in stints}
                    others = sorted(
                        {
                            st.zone
                            for pp, tr in r.tracks.items()
                            if r.sides.get(pp) != r.sides.get(p)
                            for st in tr
                        }
                        - visited
                    )
                    if len(others) >= 3:
                        picks.append((p, rng.choice(mids), others))
            if picks:
                p, st, others = rng.choice(picks)
                opts, letter = shuffled_mcq(rng, st.zone, rng.sample(others, 3))
                add(
                    "track",
                    r,
                    f"Which of these zones did {p} ({r.sides.get(p, '?')}) move through "
                    f"during the first 60 seconds?\n{opts}",
                    letter,
                )

        # who_where_when: join a kill moment onto another player's track
        # (margin: the moment sits >= 2s inside the stint, lab GT-artifact fix)
        if want("who_where_when"):
            picks = []
            for k in s.kills:
                for p in r.tracks:
                    if p in (k.killer, k.victim):
                        continue
                    st = next((x for x in r.tracks[p] if x.t0 + 2 <= k.t <= x.t1 - 2), None)
                    if st is None:
                        continue
                    others = sorted(all_zones - {st.zone})
                    if len(others) >= 3:
                        picks.append((k, p, st.zone, others))
            if picks:
                k, p, z, others = rng.choice(picks)
                opts, letter = shuffled_mcq(rng, z, rng.sample(others, 3))
                add(
                    "who_where_when",
                    r,
                    f"At the moment {k.killer} killed {k.victim}, which zone was "
                    f"{p} ({r.sides.get(p, '?')}) in?\n{opts}",
                    letter,
                )

        # first_into: first same-side player into a zone, >= 2s lead
        if want("first_into"):
            entries: dict[str, list[tuple[int, str]]] = {}
            for p, stints in r.tracks.items():
                for st in stints[1:]:
                    entries.setdefault(st.zone, []).append((st.t0, p))
            picks = []
            for z, es in entries.items():
                es.sort()
                t0, p0 = es[0]
                side0 = r.sides.get(p0, "?")
                first_by_p: dict[str, int] = {}
                for t, p in es:
                    if r.sides.get(p) == side0:
                        first_by_p.setdefault(p, t)
                rest = sorted(t for p, t in first_by_p.items() if p != p0)
                pool = [p for p in first_by_p if p != p0]
                if len(pool) >= 3 and t0 >= 5 and rest and rest[0] - t0 >= 2:
                    picks.append((z, p0, pool, side0))
            if picks:
                z, first, pool, side0 = rng.choice(picks)
                opts, letter = shuffled_mcq(rng, first, rng.sample(pool, 3))
                add(
                    "first_into",
                    r,
                    f"Which {side0}-side player entered `{z}` FIRST this round?\n{opts}",
                    letter,
                )

        # util_to_entry: smoke/molly pop -> first same-side entry (nobody already in)
        if want("util_to_entry"):
            picks = []
            for u in s.utility:
                if u.nade not in ("smoke", "molly"):
                    continue
                side_players = [p for p, sd in r.sides.items() if sd == u.side]
                stints_at_pop = (_zone_at(r.tracks, p, u.t) for p in side_players)
                if any(st is not None and st.zone == u.to_zone for st in stints_at_pop):
                    continue  # someone already inside: "entered N s later" is ambiguous
                best = None
                for p in side_players:
                    for st in r.tracks.get(p, [])[1:]:
                        if (
                            st.zone == u.to_zone
                            and st.t0 >= u.t + 1
                            and (best is None or st.t0 < best)
                        ):
                            best = st.t0
                if best is not None and 2 <= best - u.t <= 40:
                    picks.append((u, best - u.t))
            if picks:
                u, gap = rng.choice(picks)
                add(
                    "util_to_entry",
                    r,
                    f"The {u.side} {u.nade} landed at `{u.to_zone}` at t={u.t:.0f}s. How many "
                    f"seconds later did the first {u.side} player enter `{u.to_zone}`? "
                    f"Answer with a single number.",
                    f"{gap:.0f}",
                    grade="numeric",
                    tol=2.0,
                )

        # kill_gap: longest gap between consecutive kills
        if want("kill_gap") and len(s.kills) >= 4:
            ts = sorted(k.t for k in s.kills)
            gt = max(b - a for a, b in itertools.pairwise(ts))
            if gt >= 8:
                add(
                    "kill_gap",
                    r,
                    "What was the LONGEST gap in seconds between two consecutive kills "
                    "this round? Answer with a single number.",
                    f"{gt:.0f}",
                    grade="numeric",
                    tol=2.0,
                )
    return qs


# ------------------------------------------------------------------ runner
def run_arm(
    client: LLMClient,
    arm: str,
    block_for_q: Callable[[Question], str],
    questions: list[Question],
    workers: int = 8,
    max_tokens: int = 64,
) -> dict:
    def one(q: Question) -> dict:
        system = SYSTEM_TMPL.format(block=block_for_q(q))
        last_err = ""
        for attempt in range(3):
            try:
                res = client.complete(system=system, user=q.prompt, max_tokens=max_tokens)
                return {
                    "qid": q.qid,
                    "kind": q.kind,
                    "ok": grade_answer(q, res.text),
                    "raw": res.text.strip()[:200],
                    "answer": q.answer,
                }
            except Exception as exc:  # noqa: BLE001 - record and retry
                last_err = f"{type(exc).__name__}: {exc}"
                time.sleep(2.0 * (attempt + 1))
        return {"qid": q.qid, "kind": q.kind, "ok": False, "error": last_err[:300]}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, questions))
    by_kind: dict[str, dict] = {}
    for r in rows:
        d = by_kind.setdefault(r["kind"], {"n": 0, "ok": 0})
        d["n"] += 1
        d["ok"] += int(r.get("ok", False))
    correct = sum(int(r.get("ok", False)) for r in rows)
    return {
        "arm": arm,
        "total": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "errors": sum(1 for r in rows if "error" in r),
        "took_s": round(time.time() - t0, 1),
        "by_kind": {
            k: {**v, "acc": round(v["ok"] / v["n"], 3)} for k, v in sorted(by_kind.items())
        },
        "rows": rows,
    }


def _print_summary(s: dict) -> None:
    kinds = "  ".join(f"{k}={v['ok']}/{v['n']}" for k, v in s["by_kind"].items())
    print(
        f"{s['arm']:<16} acc={s['accuracy']:.3f} ({s['correct']}/{s['total']})"
        f" err={s['errors']} {s['took_s']}s | {kinds}"
    )


if __name__ == "__main__":
    import argparse

    from counterstrat.llm.base import make_client

    parser = argparse.ArgumentParser(description="Representation benchmark (2026-09-04 plan)")
    parser.add_argument("--suite", required=True, choices=["spatial", "temporal"])
    parser.add_argument("--arms", default=None, help="comma-separated; default: all in suite")
    parser.add_argument("--map", dest="map_name", default="de_inferno")
    parser.add_argument("--per-kind", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", default=None, help="default: <data_root>/eval/repr_<suite>.json"
    )
    args = parser.parse_args()

    cfg = AppConfig.load()
    client = make_client(cfg)
    print(f"provider={cfg.provider}")

    if args.suite == "spatial":
        m = MapContext(cfg, args.map_name)
        questions = gen_spatial_questions(m, seed=args.seed, per_kind=args.per_kind)
        arm_names = (args.arms or ",".join(SPATIAL_ARMS)).split(",")
        blocks = {name: SPATIAL_ARMS[name](m) for name in arm_names}

        def block_fn(name: str):
            return lambda q, b=blocks[name]: b

    else:
        rounds = load_round_contexts(cfg, args.map_name, seed=11, want=12)
        by_rid = {r.rid: r for r in rounds}
        questions = gen_temporal_questions(rounds, seed=args.seed, per_kind=args.per_kind)
        arm_names = (args.arms or ",".join(TEMPORAL_ARMS)).split(",")

        def block_fn(name: str):
            builder = TEMPORAL_ARMS[name]
            return lambda q, b=builder: b(by_rid[q.meta["round"]])

    kinds: dict[str, int] = {}
    for q in questions:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    print(f"questions: {len(questions)} {kinds}")

    out_path = (
        Path(args.output) if args.output else cfg.data_root / "eval" / f"repr_{args.suite}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in arm_names:
        s = run_arm(client, name, block_fn(name), questions)
        _print_summary(s)
        summaries.append(s)
    out_path.write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    print(f"wrote {out_path}")
