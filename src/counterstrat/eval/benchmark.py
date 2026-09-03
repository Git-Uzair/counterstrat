"""Prediction benchmark: ground-truth labels, time-ordered split, baselines, LLM arms.

Research harness (spec Phase 5), not a product surface. Every arm answers the
same question -- "what does this team do next round?" -- from information that
was available *before* the round: the TeamBook is mined only on the training
matches, and the split is strictly ordered by demo registration date so no
future match can inform a held-out prediction (spec risk #4).
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from sklearn.metrics import f1_score

from counterstrat.config import AppConfig
from counterstrat.llm.base import LLMClient, LLMResult, make_client
from counterstrat.llm.predict import MatchState, predict_round
from counterstrat.mapcard.compile import MapCard
from counterstrat.mining.tendencies import TeamBook, Tendency, _normalize_site, build_teambook
from counterstrat.roundscript.models import RoundScript

HEADS = ("buy_type", "first_contact_zone", "site", "execute", "fast")
OFFLINE_ARMS = ("majority", "freq_table")
LLM_ARMS = ("llm_no_profile", "full")
SITE_CLASSES = ("A", "B", "none")

EXECUTE_UTIL_WINDOW_S = 8.0  # spec Phase 5: clustered lineups landing together
EXECUTE_MIN_UTIL = 2
FAST_FIRST_CONTACT_S = 25.0
DEFAULT_TRAIN_RATIO = 0.7


class MatchRecord(BaseModel):
    """One demo's RoundScripts plus the registration timestamp the split orders by."""

    match_id: str
    registered_at: str
    scripts: list[RoundScript]


class RoundLabels(BaseModel):
    buy_type: str
    first_contact_zone: str
    site: str  # "A" | "B" | "none"
    execute: bool
    fast: bool


class BenchScore(BaseModel):
    accuracy: float
    f1: float = 0.0
    brier: float = 0.0


class BenchReport(BaseModel):
    heads: dict[str, dict[str, BenchScore]]  # head -> arm -> score
    n_eval_rounds: int
    arm_costs: dict[str, float] = Field(default_factory=dict)  # arm -> tokens spent


# --- ground truth -----------------------------------------------------------


def _zone_site(zone: str) -> str:
    """Site whose tag group a zone belongs to: only bombsite zones commit a site."""
    m = re.fullmatch(r"(?:bombsite|site)[_ ]?([ab])", zone.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else "none"


def _executed(script: RoundScript, side: str) -> bool:
    """>=2 utility events sharing a lineup prefix within 8 s, plus a committed site."""
    by_prefix: dict[str, list[float]] = defaultdict(list)
    for u in script.utility:
        if u.side == side and u.lineup_id:
            by_prefix[u.lineup_id.split("-")[0]].append(u.t)

    clustered = False
    for times in by_prefix.values():
        times.sort()
        for i, t0 in enumerate(times):
            if sum(1 for t in times[i:] if t - t0 <= EXECUTE_UTIL_WINDOW_S) >= EXECUTE_MIN_UTIL:
                clustered = True
                break
        if clustered:
            break
    if not clustered:
        return False

    if script.plant is not None:
        return True
    fc = script.first_contact
    return fc is not None and _zone_site(fc.zone) != "none"


def extract_labels(script: RoundScript, side: str = "T") -> RoundLabels:
    """Ground truth for one round, read off the round's own RoundScript."""
    econ = script.economy.get(side) or script.economy.get("T")
    fc = script.first_contact
    return RoundLabels(
        buy_type=econ.buy_type if econ else "full_buy",
        first_contact_zone=fc.zone if fc else "none",
        site=_normalize_site(script.plant.site) if script.plant else "none",
        execute=_executed(script, side),
        fast=fc is not None and fc.t < FAST_FIRST_CONTACT_S,
    )


# --- time-ordered split -----------------------------------------------------


def split[T](corpus: list[T], train_ratio: float = DEFAULT_TRAIN_RATIO) -> tuple[list[T], list[T]]:
    """Split by ``registered_at`` so every train item precedes every held-out item."""

    def when(item: Any) -> str:
        return item.registered_at

    ordered = sorted(corpus, key=when)
    if len(ordered) < 2:
        return ordered, []
    n_train = min(max(1, round(len(ordered) * train_ratio)), len(ordered) - 1)
    return ordered[:n_train], ordered[n_train:]


# --- round state (mirrors mining.tendencies key derivation) ------------------


def round_key(script: RoundScript, side: str, prev: RoundScript | None) -> tuple[str, str, str]:
    """``(buy_class, score_bucket, prev_outcome)`` for one round, as the miner keys it."""
    team_key = script.t_team_key if side == "T" else script.ct_team_key
    is_ot_start = script.round_num >= 25 and (script.round_num - 25) % 3 == 0
    if prev is None or script.round_num in (1, 13) or is_ot_start:
        prev_outcome = "first"
    else:
        prev_side = "T" if prev.t_team_key == team_key else "CT"
        won = prev.winner in (prev_side, team_key)
        prev_outcome = "won" if won else "lost"

    team_score = script.score_t if side == "T" else script.score_ct
    opp_score = script.score_ct if side == "T" else script.score_t
    if team_score > opp_score:
        score_bucket = "ahead"
    elif team_score < opp_score:
        score_bucket = "behind"
    else:
        score_bucket = "even"

    econ = script.economy.get(side)
    buy_class = econ.buy_type if econ else "full_buy"
    return buy_class, score_bucket, prev_outcome


class EvalRound(BaseModel):
    """One held-out round: what is asked, and the truth it is scored against."""

    script: RoundScript
    prev: list[RoundScript] = Field(default_factory=list)
    team_key: str
    labels: RoundLabels
    key: tuple[str, str, str]


def eval_rounds(heldout: Sequence[MatchRecord], side: str = "T") -> list[EvalRound]:
    rounds: list[EvalRound] = []
    for match in heldout:
        ordered = sorted(match.scripts, key=lambda s: s.round_num)
        for i, script in enumerate(ordered):
            prev = ordered[:i]
            rounds.append(
                EvalRound(
                    script=script,
                    prev=prev,
                    team_key=script.t_team_key if side == "T" else script.ct_team_key,
                    labels=extract_labels(script, side),
                    key=round_key(script, side, prev[-1] if prev else None),
                )
            )
    return rounds


# --- arms -------------------------------------------------------------------

Prediction = tuple[RoundLabels, dict[str, float]]  # labels + p_site for Brier

_NEUTRAL_P_SITE = {"A": 1 / 3, "B": 1 / 3, "none": 1 / 3}


def _argmax(dist: dict[str, float], default: str) -> str:
    """Highest-probability key, alphabetical tie-break (as in mining.tendencies)."""
    if not dist:
        return default
    return min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _majority_arm(train_labels: list[RoundLabels]) -> Prediction:
    """Global modal class per head over the training rounds."""
    if not train_labels:
        return (
            RoundLabels(
                buy_type="full_buy",
                first_contact_zone="none",
                site="none",
                execute=False,
                fast=False,
            ),
            dict(_NEUTRAL_P_SITE),
        )
    modal: dict[str, object] = {}
    for head in HEADS:
        counts = Counter(str(getattr(lbl, head)) for lbl in train_labels)
        modal[head] = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    site_counts = Counter(lbl.site for lbl in train_labels)
    p_site = {s: site_counts.get(s, 0) / len(train_labels) for s in SITE_CLASSES}
    return (
        RoundLabels(
            buy_type=str(modal["buy_type"]),
            first_contact_zone=str(modal["first_contact_zone"]),
            site=str(modal["site"]),
            execute=modal["execute"] == "True",
            fast=modal["fast"] == "True",
        ),
        p_site,
    )


def _match_tendencies(
    tendencies: Sequence[Tendency],
    side: str,
    key: tuple[str, str, str],
    *,
    use_buy: bool = True,
) -> list[Tendency]:
    """Tendencies for the round's key, relaxing the key until something matches."""
    buy_class, score_bucket, prev_outcome = key
    same_side = [t for t in tendencies if t.key.side == side]
    if use_buy:
        ladder = [
            lambda t: (
                t.key.buy_class == buy_class
                and t.key.score_bucket == score_bucket
                and t.key.prev_outcome == prev_outcome
            ),
            lambda t: t.key.buy_class == buy_class and t.key.score_bucket == score_bucket,
            lambda t: t.key.buy_class == buy_class,
        ]
    else:
        # buy_type is a prediction head: never key the lookup on the true buy.
        ladder = [
            lambda t: t.key.score_bucket == score_bucket and t.key.prev_outcome == prev_outcome,
            lambda t: t.key.score_bucket == score_bucket,
        ]
    for predicate in ladder:
        hit = [t for t in same_side if predicate(t)]
        if hit:
            return hit
    return same_side


def _weighted_dist(tendencies: Sequence[Tendency], field: str) -> dict[str, float]:
    agg: dict[str, float] = defaultdict(float)
    for t in tendencies:
        for k, p in getattr(t, field).items():
            agg[k] += p * t.n
    total = sum(agg.values())
    return {k: v / total for k, v in agg.items()} if total else {}


def _evidence_rate(
    tendencies: Sequence[Tendency],
    index: dict[str, RoundScript],
    side: str,
    head: str,
) -> float:
    """Rate of a boolean head over the rounds the matched tendencies cite."""
    vals = [
        bool(getattr(extract_labels(index[e], side), head))
        for t in tendencies
        for e in t.evidence
        if e in index
    ]
    return sum(vals) / len(vals) if vals else 0.0


def _freq_table_arm(
    teambook: TeamBook | None,
    index: dict[str, RoundScript],
    side: str,
    key: tuple[str, str, str],
) -> Prediction:
    """Argmax of the matching TeamBook tendency -- no LLM, no future information."""
    tendencies = _match_tendencies(teambook.tendencies, side, key) if teambook else []
    if not tendencies:
        return (
            RoundLabels(
                buy_type="full_buy",
                first_contact_zone="none",
                site="none",
                execute=False,
                fast=False,
            ),
            dict(_NEUTRAL_P_SITE),
        )
    buy_group = _match_tendencies(teambook.tendencies, side, key, use_buy=False) if teambook else []
    buy_counts: dict[str, float] = defaultdict(float)
    for t in buy_group:
        buy_counts[t.key.buy_class] += t.n

    site_dist = _weighted_dist(tendencies, "site_committed")
    p_site = {s: site_dist.get(s, 0.0) for s in SITE_CLASSES}
    return (
        RoundLabels(
            buy_type=_argmax(dict(buy_counts), "full_buy"),
            first_contact_zone=_argmax(_weighted_dist(tendencies, "first_contact_zone"), "none"),
            site=_argmax(site_dist, "none"),
            execute=_evidence_rate(tendencies, index, side, "execute") >= 0.5,
            fast=_evidence_rate(tendencies, index, side, "fast") >= 0.5,
        ),
        p_site,
    )


def _round_summary(script: RoundScript, side: str) -> str:
    lbl = extract_labels(script, side)
    fc_t = f"@{script.first_contact.t:.0f}s" if script.first_contact else ""
    return (
        f"R{script.round_num}: {lbl.buy_type}, first contact {lbl.first_contact_zone}{fc_t}, "
        f"site {lbl.site}, {'execute' if lbl.execute else 'default'}, "
        f"{'fast' if lbl.fast else 'slow'}, won by {script.winner}"
    )


def _match_state(round_: EvalRound, side: str, *, raw: bool) -> MatchState:
    prev = round_.prev[-3:]
    summaries = [p.to_text() for p in prev] if raw else [_round_summary(p, side) for p in prev]
    prev_econ = round_.prev[-1].economy.get(side) if round_.prev else None
    economy = (
        f"previous round {side} spend ${prev_econ.spend} ({prev_econ.buy_type}), "
        f"loss streak {prev_econ.loss_streak}"
        if prev_econ
        else "unknown"
    )
    return MatchState(
        score_t=round_.script.score_t,
        score_ct=round_.script.score_ct,
        side=side,
        prev_round_summaries=summaries,
        economy_estimate=economy,
    )


def _empty_teambook(round_: EvalRound) -> TeamBook:
    """Profile-free control arm: same prompt shape, no mined tendencies or roles."""
    return TeamBook(
        team_key=round_.team_key,
        map_name=round_.script.map_name,
        card_checksum=round_.script.card_checksum,
        tendencies=[],
        roles=[],
        generated_from=[],
    )


def _llm_arm(
    arm: str,
    client: LLMClient,
    card: MapCard,
    teambook: TeamBook | None,
    round_: EvalRound,
    side: str,
) -> tuple[Prediction, LLMResult]:
    raw = arm == "llm_no_profile"
    book = _empty_teambook(round_) if (raw or teambook is None) else teambook
    state = _match_state(round_, side, raw=raw)
    pred, usage = predict_round(client, card, book, state)
    labels = RoundLabels(
        buy_type=pred.buy_type,
        first_contact_zone=pred.first_contact_zone,
        site=pred.site,
        execute=pred.execute,
        fast=pred.fast,
    )
    p_site = {s: float(pred.p_site.get(s, 0.0)) for s in SITE_CLASSES}
    return (labels, p_site), usage


# --- scoring ----------------------------------------------------------------


def _score_head(truth: list[RoundLabels], preds: list[Prediction], head: str) -> BenchScore:
    y_true = [str(getattr(lbl, head)) for lbl in truth]
    y_pred = [str(getattr(p[0], head)) for p in preds]
    if not y_true:
        return BenchScore(accuracy=0.0)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred, strict=True)) / len(y_true)
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    brier = 0.0
    if head == "site":
        totals = [
            sum((p_site.get(c, 0.0) - (1.0 if lbl.site == c else 0.0)) ** 2 for c in SITE_CLASSES)
            for lbl, (_, p_site) in zip(truth, preds, strict=True)
        ]
        brier = sum(totals) / len(totals)
    return BenchScore(accuracy=accuracy, f1=f1, brier=brier)


# --- drivers ----------------------------------------------------------------


def run_arms(
    corpus: Sequence[MatchRecord],
    arms: Sequence[str] = OFFLINE_ARMS,
    *,
    side: str = "T",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    client: LLMClient | None = None,
    card: MapCard | None = None,
) -> BenchReport:
    """Score every requested arm on the held-out half of a time-ordered corpus."""
    unknown = [a for a in arms if a not in OFFLINE_ARMS + LLM_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    needs_llm = [a for a in arms if a in LLM_ARMS]
    if needs_llm and (client is None or card is None):
        raise ValueError(f"arms {needs_llm} need an LLMClient and a MapCard")

    train, heldout = split(list(corpus), train_ratio)
    train_scripts = [s for m in train for s in m.scripts]
    train_index = {f"{s.match_id}:{s.round_num}": s for s in train_scripts}
    rounds = eval_rounds(heldout, side)
    truth = [r.labels for r in rounds]

    teambooks: dict[str, TeamBook] = {}
    for team_key in {r.team_key for r in rounds}:
        teambooks[team_key] = build_teambook(train_scripts, team_key)

    train_labels = [extract_labels(s, side) for s in train_scripts]
    majority = _majority_arm(train_labels)  # one global modal answer for every round
    heads: dict[str, dict[str, BenchScore]] = {head: {} for head in HEADS}
    arm_costs: dict[str, float] = {}

    for arm in arms:
        preds: list[Prediction] = []
        tokens = 0.0
        for r in rounds:
            if arm == "majority":
                preds.append(majority)
            elif arm == "freq_table":
                preds.append(_freq_table_arm(teambooks.get(r.team_key), train_index, side, r.key))
            else:
                assert client is not None and card is not None  # guarded above
                pred, usage = _llm_arm(arm, client, card, teambooks.get(r.team_key), r, side)
                preds.append(pred)
                tokens += usage.input_tokens + usage.output_tokens
        for head in HEADS:
            heads[head][arm] = _score_head(truth, preds, head)
        arm_costs[arm] = tokens

    return BenchReport(heads=heads, n_eval_rounds=len(rounds), arm_costs=arm_costs)


def run_offline_arms(
    corpus: Sequence[MatchRecord],
    arms: Sequence[str] = OFFLINE_ARMS,
    *,
    side: str = "T",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> BenchReport:
    """Offline (no-LLM) arms only; rejects arms that would spend tokens."""
    llm = [a for a in arms if a not in OFFLINE_ARMS]
    if llm:
        raise ValueError(f"arms {llm} require an LLM client -- use run_arms/run")
    return run_arms(corpus, arms, side=side, train_ratio=train_ratio)


def load_card(data_root: Path, map_name: str) -> MapCard | None:
    path = data_root / "mapcards" / map_name / "card.yaml"
    if not path.exists():
        return None
    return MapCard.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_corpus(data_root: Path) -> list[MatchRecord]:
    """Corpus from ``<data_root>/corpus.jsonl`` + ``<data_root>/scripts/<match_id>/*.json``."""
    from counterstrat.corpus import load_manifest

    records: list[MatchRecord] = []
    for match_id, rec in load_manifest(data_root / "corpus.jsonl").items():
        script_dir = data_root / "scripts" / match_id
        if not script_dir.is_dir():
            continue
        scripts = [
            RoundScript.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(script_dir.glob("*.json"))
        ]
        if scripts:
            records.append(
                MatchRecord(match_id=match_id, registered_at=rec.registered_at, scripts=scripts)
            )
    return records


def run(
    cfg: AppConfig,
    arms: list[str],
    corpus: Sequence[MatchRecord] | None = None,
    *,
    side: str = "T",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    card: MapCard | None = None,
) -> BenchReport:
    """Benchmark entry point: loads the corpus/card/client that the arms need."""
    records = list(corpus) if corpus is not None else load_corpus(cfg.data_root)
    if not records:
        raise ValueError(f"no corpus found under {cfg.data_root} (need serialized RoundScripts)")
    client = make_client(cfg) if any(a in LLM_ARMS for a in arms) else None
    if client is not None and card is None:
        card = load_card(cfg.data_root, records[-1].scripts[0].map_name)
    return run_arms(records, arms, side=side, train_ratio=train_ratio, client=client, card=card)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-round prediction benchmark (spec Phase 5)")
    parser.add_argument(
        "--arms",
        default=",".join(OFFLINE_ARMS),
        help=f"comma-separated arms from {OFFLINE_ARMS + LLM_ARMS}",
    )
    parser.add_argument("--side", default="T", choices=["T", "CT"], help="side to predict for")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--card-path", default=None, help="path to a compiled card.yaml")
    parser.add_argument("--output", default=None, help="default: <data_root>/eval/report.json")
    args = parser.parse_args()

    app_cfg = AppConfig.load()
    cli_card = (
        MapCard.model_validate(yaml.safe_load(Path(args.card_path).read_text(encoding="utf-8")))
        if args.card_path
        else None
    )
    report = run(
        app_cfg,
        [a.strip() for a in args.arms.split(",") if a.strip()],
        side=args.side,
        train_ratio=args.train_ratio,
        card=cli_card,
    )
    out_path = Path(args.output) if args.output else app_cfg.data_root / "eval" / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")

    print(f"n_eval_rounds={report.n_eval_rounds}")
    for head_name, per_arm in report.heads.items():
        for arm_name, score in per_arm.items():
            print(
                f"{head_name:<20} {arm_name:<15} acc={score.accuracy:.3f} "
                f"f1={score.f1:.3f} brier={score.brier:.3f}"
            )
    print(f"arm_costs (tokens): {report.arm_costs}")
    print(f"wrote {out_path}")
