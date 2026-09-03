"""LLM-generated First Read: dynamic, causal insights over the full corpus.

Unlike the deterministic scout brief (template text over mined thresholds),
this hands the model EVERYTHING - every round script with movements, the full
tendency table including noise rows, utility book, gap findings, economy
policy, and player profiles - and asks for reasoned, conditioned insights.
Modern context windows fit multiple demos comfortably; the value is in the
model connecting causes, not in us pre-chewing the data.
"""

import re

from pydantic import BaseModel, Field

from counterstrat.llm.base import LLMClient, LLMResult
from counterstrat.llm.dossier import lint_dossier
from counterstrat.llm.prompts import format_map_scene_graph
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mining.econ_policy import EconPolicy
from counterstrat.mining.gaps import GapReport
from counterstrat.mining.tendencies import TeamBook
from counterstrat.mining.utility_book import UtilityBook
from counterstrat.roundscript.models import RoundScript


class Insights(BaseModel):
    """One generated First Read with its lint findings and provenance."""

    text: str
    warnings: list[str] = Field(default_factory=list)
    generated_from: list[str]
    games: list[dict] = Field(default_factory=list)  # [{label, match_id, opponent}]
    usage: LLMResult


# Internal window codes -> plain language for prompts and any rendered artifact.
_WINDOW_LABEL = {"post-FC": "after first contact", "post-PL": "after the plant"}


def _friendly_window(window: str) -> str:
    if window in _WINDOW_LABEL:
        return _WINDOW_LABEL[window]
    if window.startswith("B+"):
        seconds = int(window[2:])
        return "at round start" if seconds == 0 else f"{seconds}s into the round"
    return window


def default_game_labels(generated_from: list[str]) -> dict[str, str]:
    return {mid: f"Game {i}" for i, mid in enumerate(generated_from, start=1)}


def _friendly_round(round_id: str, labels: dict[str, str]) -> str:
    mid, _, rn = round_id.partition(":")
    return f"{labels.get(mid, mid[:8])} R{rn}"


def build_insights_system(map_block: str, zone_map: str = "") -> str:
    return f"""You are an elite CS2 anti-strat analyst writing the FIRST READ on an opponent
for an in-game leader. You reason about WHY a team does something - economy pressure,
momentum, role habits, utility dependencies - and you never confuse normal play with a
tendency.

<map_card>
{map_block}{zone_map}
</map_card>

Write the brief in markdown with EXACTLY these six sections, in this order:
## T Pistol - their T-side pistol round default, and the punish.
## CT Pistol - their CT-side pistol round setup, and the punish.
## Eco & Force Habits - how they play low-buy rounds, and how to farm them safely.
## T Full Buy - their full-buy attack (defaults, executes, tempo), and the counter.
## CT Full Buy - their full-buy defense (setup, rotations, utility), and the counter.
## Gotchas - 2-5 bullets of anything else worth knowing: player habits, utility
crutches, timing tells, gaps, momentum behavior. Non-obvious observations only.

Inside every section write:
- **Read** - the pattern in one or two sentences ("after X, they do Y").
- **Why** - the causal chain, reasoned from the rounds, not just the frequency.
- **Punish** - the exact counter-call a caller can lift word-for-word.
- **Evidence** - cite as (Game 1, rounds 3, 7) plus the sample size n.

Language rules - the reader is a player, not a database:
- Refer to games as Game 1, Game 2 (the Games list in the data maps them). NEVER
  print raw match ids or hashes.
- Say "first contact", "after the plant", "30s into the round" - never internal
  codes like FC, PL, B+30.
- Wrap every zone name in backticks and use only zones from the Map Card.

Hard rules:
- Never present a consequence of normal play as an insight. Banned tautologies:
  CTs leaving the other site post-plant to retake; Ts not standing on bombsites
  early; teams saving on a lost eco. A read must be something a different team in
  the same situation would plausibly do differently.
- Prefer conditioned reads (previous round, economy state, first-contact outcome,
  utility spent) over raw frequencies. Cross-reference the round scripts - they are
  the ground truth.
- Pistol sections have tiny samples (one T and one CT pistol per game): state n
  honestly and when the data is too thin for a read, say so in one line and give a
  solid default recommendation instead of forcing a pattern.
- Cite only rounds that exist in the data.
"""


def build_insights_user(
    *,
    teambook: TeamBook,
    utility_book: UtilityBook,
    gap_report: GapReport,
    econ_policy: EconPolicy,
    scripts: list[RoundScript],
    game_labels: dict[str, str] | None = None,
) -> str:
    labels = game_labels or default_game_labels(teambook.generated_from)
    total_rounds = sum(t.n for t in teambook.tendencies if t.level == 0)
    sections = [
        (
            f"Data coverage: {len(teambook.generated_from)} game(s), {total_rounds} rounds "
            f"of {teambook.team_key} on {teambook.map_name}."
        ),
        "",
        "## Games",
    ]
    for mid in teambook.generated_from:
        sections.append(f"- {labels.get(mid, mid[:8])}")
    sections += ["", teambook.to_table_text(), "", "## Utility Book"]
    for p in utility_book.patterns:
        lineup = f" [{p.lineup_id}]" if p.lineup_id else ""
        evidence = ", ".join(_friendly_round(e, labels) for e in p.evidence[:4])
        sections.append(
            f"- {p.side} {p.nade} -> `{p.to_zone}`{lineup}: {p.count}/{p.rounds_seen} rounds, "
            f"median {p.median_t:.0f}s, early-share {p.early_share:.0%} (evidence: {evidence})"
        )
    sections += ["", "## Gap Findings (15s formation windows; plant rows = the planted site only)"]
    for f in gap_report.findings:
        evidence = ", ".join(_friendly_round(e, labels) for e in f.evidence[:4])
        sections.append(
            f"- {f.side} vacate `{f.zone}` {_friendly_window(f.window)} on '{f.trigger}': "
            f"{f.vacancy_rate:.0%} of {f.n} (baseline {f.baseline_rate:.0%}, lift {f.lift:+.0%}; "
            f"evidence: {evidence})"
        )
    sections += ["", "## Economy Policy"]
    for state, dist in econ_policy.policy.items():
        dist_str = ", ".join(f"{b} {p:.0%}" for b, p in dist.items())
        sections.append(f"- {state} (n={econ_policy.ns.get(state, 0)}): {dist_str}")
    if econ_policy.pistol_round_sites:
        sections.append(f"- T pistol sites: {econ_policy.pistol_round_sites}")
    if econ_policy.post_pistol_loss_buy:
        sections.append(f"- post-pistol-loss buys: {econ_policy.post_pistol_loss_buy}")
    sections += ["", "## Player Profiles"]
    for r in teambook.roles:
        sections.append(
            f"- {r.player}: opening-duel rate {r.opening_duel_rate:.0%} "
            f"(wins {r.opening_kill_rate:.0%}, zones {r.opening_zones}), "
            f"lurk {r.lurk_rate:.0%}, awp rounds {r.awp_rounds}, "
            f"traded when dying {r.trade_discipline:.0%}, modal zones 15s in {r.modal_zone_fe15}"
        )
    sections += ["", "## All Round Scripts (ground truth; movements included)"]
    for s in sorted(scripts, key=lambda s: (s.match_id, s.round_num)):
        pistol = " (pistol round)" if s.round_num in (1, 13) else ""
        sections.append(
            f"### {labels.get(s.match_id, s.match_id[:8])}, round {s.round_num}{pistol}"
        )
        sections.append(s.to_text(include_movements=True))
        sections.append("")
    return "\n".join(sections)


_GAME_CITE_RE = re.compile(
    r"Game\s+(\d+)\s*,?\s*rounds?\s+(\d[\d,\s]*(?:and\s+\d+)?)", re.IGNORECASE
)
_HASH_RE = re.compile(r"\b[0-9a-f]{12,}\b")


def _check_friendly_citations(
    text: str, generated_from: list[str], scripts: list[RoundScript]
) -> list[str]:
    """Verify (Game N, rounds ...) cites against real rounds; flag leaked hashes."""
    rounds_by_match: dict[str, set[int]] = {}
    for s in scripts:
        rounds_by_match.setdefault(s.match_id, set()).add(s.round_num)
    by_game = {i: rounds_by_match.get(mid, set()) for i, mid in enumerate(generated_from, start=1)}
    warnings: list[str] = []
    for m in _GAME_CITE_RE.finditer(text):
        game_no = int(m.group(1))
        cited = [int(x) for x in re.findall(r"\d+", m.group(2))]
        valid = by_game.get(game_no)
        if valid is None:
            warnings.append(f"Cited Game {game_no} does not exist")
            continue
        for rn in cited:
            if rn not in valid:
                warnings.append(f"Cited Game {game_no} round {rn} does not exist")
    if _HASH_RE.search(text):
        warnings.append("Internal match-id hash leaked into the text")
    return warnings


def generate_insights(
    client: LLMClient,
    card: MapCard,
    *,
    teambook: TeamBook,
    utility_book: UtilityBook,
    gap_report: GapReport,
    econ_policy: EconPolicy,
    scripts: list[RoundScript],
    lexicon: Lexicon,
    game_labels: dict[str, str] | None = None,
    renamer=None,  # counterstrat.aliases.Renamer; applies the user's callout vocabulary
    max_tokens: int | None = None,  # None = the model's own maximum: never cut analysis short
    anchors: dict | None = None,  # web.routes.map_zone_anchors output for this map
) -> Insights:
    """One LLM call over the full corpus; fabrications surface as soft warnings."""
    labels = game_labels or default_game_labels(teambook.generated_from)
    system = build_insights_system(format_map_scene_graph(card, anchors or {}))
    user = build_insights_user(
        teambook=teambook,
        utility_book=utility_book,
        gap_report=gap_report,
        econ_policy=econ_policy,
        scripts=scripts,
        game_labels=labels,
    )
    if renamer:
        system = renamer.rename_text(system)
        user = renamer.rename_text(user)
    result = client.complete(system=system, user=user, max_tokens=max_tokens)

    valid_evidence = {f"{s.match_id}:{s.round_num}" for s in scripts}
    lint = lint_dossier(
        result.text,
        teambook,
        lexicon,
        valid_evidence,
        aliases=renamer.aliases if renamer else None,
    )
    warnings = [f"Unknown zone: {z}" for z in lint.unknown_zones]
    warnings += _check_friendly_citations(result.text, list(teambook.generated_from), scripts)
    if result.truncated:
        warnings.append(
            "Output hit the model's token ceiling and is cut short - regenerate "
            "(the model may think less on a retry) or raise max_tokens."
        )

    return Insights(
        text=result.text,
        warnings=warnings,
        generated_from=list(teambook.generated_from),
        games=[
            {"label": labels.get(mid, mid[:8]), "match_id": mid} for mid in teambook.generated_from
        ],
        usage=result,
    )
