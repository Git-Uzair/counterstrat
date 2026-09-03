"""Deterministic scout brief: instant post-ingest headline exploits (plan Task 8).

No LLM call: every item is template text over mined artifacts, each carrying
its sample size, a confidence grade, and citable evidence. Kinds whose
thresholds are unmet are skipped - an honest short brief beats a noisy one.
"""

from pydantic import BaseModel

from counterstrat.mining.econ_policy import EconPolicy
from counterstrat.mining.gaps import GapReport
from counterstrat.mining.tendencies import TeamBook
from counterstrat.mining.utility_book import UtilityBook
from counterstrat.roundscript.models import RoundScript

MAX_ITEMS = 8
HIGH_CONFIDENCE_N = 6

SITE_LEAN_MIN_N = 4
SITE_LEAN_MIN_SHARE = 0.6
TEMPO_MIN_N = 4
TEMPO_FAST_S = 25.0
TEMPO_SLOW_S = 45.0
CRUTCH_MIN_SHARE = 0.5
CRUTCH_MIN_COUNT = 3
BASE_GAP_MIN_RATE = 0.8
BASE_GAP_MIN_N = 4
ECON_TELL_MIN_SHARE = 0.75
ECON_TELL_MIN_N = 3
OPENER_MIN_DUEL_RATE = 0.3
PISTOL_MIN_N = 2

_BUY_LABEL = {
    "full_eco": "full-eco",
    "semi_eco": "semi-eco",
    "semi_buy": "force",
    "full_buy": "full-buy",
    "any": "all buys",
}
_STATE_LABEL = {
    "pistol": "on pistols",
    "after_win": "after a won round",
    "after_loss_1": "after one loss",
    "after_loss_2": "after two losses",
    "after_loss_3plus": "on a 3+ loss streak",
}
_TRIGGER_LABEL = {
    "after_loss": "after losing a round",
    "after_win": "after winning a round",
    "behind": "when behind",
    "ahead": "when ahead",
    "after_fc_win": "after winning first contact",
    "after_fc_loss": "after losing first contact",
    "after_util_dump": "once their utility is dumped",
}


class BriefItem(BaseModel):
    kind: str  # site_lean | tempo | utility_crutch | gap | econ_tell | opener | pistol
    text: str
    n: int
    confidence: str  # "high" | "medium"
    evidence: list[str]


class ScoutBrief(BaseModel):
    team_key: str
    map_name: str
    items: list[BriefItem]
    generated_from: list[str]


def _conf(n: int) -> str:
    return "high" if n >= HIGH_CONFIDENCE_N else "medium"


def _gap_items(gap_report: GapReport) -> list[BriefItem]:
    triggered = [f for f in gap_report.findings if f.trigger != "base"]
    if triggered:
        f = triggered[0]  # findings are sorted by lift desc
        label = _TRIGGER_LABEL.get(f.trigger, f.trigger)
        return [
            BriefItem(
                kind="gap",
                text=(
                    f"{f.side} leave `{f.zone}` unwatched at {f.window} {label} "
                    f"({f.vacancy_rate:.0%} of {f.n} rounds vs {f.baseline_rate:.0%} baseline)."
                ),
                n=f.n,
                confidence=_conf(f.n),
                evidence=f.evidence,
            )
        ]
    bases = [
        f
        for f in gap_report.findings
        if f.trigger == "base" and f.vacancy_rate >= BASE_GAP_MIN_RATE and f.n >= BASE_GAP_MIN_N
    ]
    if bases:
        f = max(bases, key=lambda x: (x.vacancy_rate, x.n))
        return [
            BriefItem(
                kind="gap",
                text=(
                    f"{f.side} leave `{f.zone}` unheld at {f.window} in "
                    f"{f.vacancy_rate:.0%} of rounds (n={f.n})."
                ),
                n=f.n,
                confidence=_conf(f.n),
                evidence=f.evidence,
            )
        ]
    return []


def _crutch_items(utility_book: UtilityBook) -> list[BriefItem]:
    picks = [
        p
        for p in utility_book.patterns
        if p.share >= CRUTCH_MIN_SHARE and p.count >= CRUTCH_MIN_COUNT
    ]
    if not picks:
        return []
    p = picks[0]  # patterns are sorted by count desc
    lineup = f" [{p.lineup_id}]" if p.lineup_id else ""
    return [
        BriefItem(
            kind="utility_crutch",
            text=(
                f"{p.side} {p.nade} `{p.to_zone}`{lineup} in {p.count}/{p.rounds_seen} rounds "
                f"(median {p.median_t:.0f}s) - their go-to; punish the zones it covers "
                f"after it fades."
            ),
            n=p.count,
            confidence=_conf(p.count),
            evidence=p.evidence,
        )
    ]


def _site_lean_items(teambook: TeamBook) -> list[BriefItem]:
    items: list[BriefItem] = []
    for t in teambook.tendencies:
        if t.level != 1 or t.key.side != "T" or t.n < SITE_LEAN_MIN_N:
            continue
        site, share = next(iter(t.site_committed.items()), ("none", 0.0))
        if site in ("A", "B") and share >= SITE_LEAN_MIN_SHARE:
            buy = _BUY_LABEL.get(t.key.buy_class, t.key.buy_class)
            items.append(
                BriefItem(
                    kind="site_lean",
                    text=(
                        f"T {buy}: {share:.0%} of plants land site {site} (n={t.n}) - "
                        f"weight the defense there."
                    ),
                    n=t.n,
                    confidence=_conf(t.n),
                    evidence=t.evidence[-4:],
                )
            )
    return items[:2]


def _tempo_items(teambook: TeamBook) -> list[BriefItem]:
    items: list[BriefItem] = []
    for t in teambook.tendencies:
        if t.level != 0 or t.n < TEMPO_MIN_N or t.median_first_contact_s is None:
            continue
        med = t.median_first_contact_s
        if med <= TEMPO_FAST_S:
            flavor = "fast tempo - expect early duels, set crossfires before"
        elif med >= TEMPO_SLOW_S:
            flavor = "slow defaults - utility can be saved for late, watch for"
        else:
            continue
        items.append(
            BriefItem(
                kind="tempo",
                text=(
                    f"{t.key.side} first contact median {med:.0f}s (n={t.n}) - "
                    f"{flavor} the {med:.0f}s mark."
                ),
                n=t.n,
                confidence=_conf(t.n),
                evidence=t.evidence[-4:],
            )
        )
    return items[:2]


def _econ_items(econ: EconPolicy) -> list[BriefItem]:
    items: list[BriefItem] = []
    for state, dist in econ.policy.items():
        if state == "pistol" or econ.ns.get(state, 0) < ECON_TELL_MIN_N:
            continue
        buy, share = next(iter(dist.items()), (None, 0.0))
        if buy is None or share < ECON_TELL_MIN_SHARE:
            continue
        items.append(
            BriefItem(
                kind="econ_tell",
                text=(
                    f"{_STATE_LABEL.get(state, state).capitalize()} they go "
                    f"{_BUY_LABEL.get(buy, buy)} {share:.0%} of the time "
                    f"(n={econ.ns[state]})."
                ),
                n=econ.ns[state],
                confidence=_conf(econ.ns[state]),
                evidence=econ.evidence.get(state, []),
            )
        )
    items.sort(key=lambda i: (-i.n, i.text))
    return items[:2]


def _opener_items(teambook: TeamBook) -> list[BriefItem]:
    candidates = [
        r for r in teambook.roles if r.opening_duel_rate >= OPENER_MIN_DUEL_RATE and r.opening_zones
    ]
    if not candidates:
        return []
    r = max(candidates, key=lambda r: (r.opening_duel_rate, r.player))
    n = sum(r.opening_zones.values())
    top_zone = next(iter(r.opening_zones))
    return [
        BriefItem(
            kind="opener",
            text=(
                f"{r.player} takes {r.opening_duel_rate:.0%} of their opening duels "
                f"(wins {r.opening_kill_rate:.0%}), usually at `{top_zone}` - deny or "
                f"trade that first peek."
            ),
            n=n,
            confidence=_conf(n),
            evidence=[],
        )
    ]


def _pistol_items(econ: EconPolicy) -> list[BriefItem]:
    n = econ.ns.get("pistol_t", 0)
    if n < PISTOL_MIN_N or not econ.pistol_round_sites:
        return []
    site, share = next(iter(econ.pistol_round_sites.items()))
    if site not in ("A", "B") or share < ECON_TELL_MIN_SHARE:
        return []
    return [
        BriefItem(
            kind="pistol",
            text=f"T pistols commit site {site} {share:.0%} of the time (n={n}).",
            n=n,
            confidence=_conf(n),
            evidence=econ.evidence.get("pistol", []),
        )
    ]


def build_scout_brief(
    scripts: list[RoundScript],
    team_key: str,
    *,
    teambook: TeamBook,
    utility_book: UtilityBook,
    gap_report: GapReport,
    econ_policy: EconPolicy,
) -> ScoutBrief:
    """Assemble the instant post-ingest headline list from mined artifacts."""
    items: list[BriefItem] = []
    items += _gap_items(gap_report)
    items += _crutch_items(utility_book)
    items += _site_lean_items(teambook)
    items += _econ_items(econ_policy)
    items += _opener_items(teambook)
    items += _tempo_items(teambook)
    items += _pistol_items(econ_policy)

    # Openers mined from FC involvement have no round-id evidence of their own;
    # fall back to the team's rollup evidence so every item stays citable.
    l0_evidence = next((t.evidence for t in teambook.tendencies if t.level == 0), [])
    for item in items:
        if not item.evidence:
            item.evidence = l0_evidence[-4:]

    return ScoutBrief(
        team_key=team_key,
        map_name=teambook.map_name,
        items=items[:MAX_ITEMS],
        generated_from=teambook.generated_from,
    )
