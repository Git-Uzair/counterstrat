"""LLM-generated First Read: dynamic, causal insights over the full corpus.

Unlike the deterministic scout brief (template text over mined thresholds),
this hands the model EVERYTHING - every round script with movements, the full
tendency table including noise rows, utility book, gap findings, economy
policy, and player profiles - and asks for reasoned, conditioned insights.
Modern context windows fit multiple demos comfortably; the value is in the
model connecting causes, not in us pre-chewing the data.
"""

from pydantic import BaseModel, Field

from counterstrat.llm.base import LLMClient, LLMResult
from counterstrat.llm.dossier import lint_dossier
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
    usage: LLMResult


def build_insights_system(card_yaml: str) -> str:
    return f"""You are an elite CS2 anti-strat analyst writing the FIRST READ on an opponent
for an in-game leader. You reason about WHY a team does something - economy pressure,
momentum, role habits, utility dependencies - and you never confuse normal play with a
tendency.

<map_card>
{card_yaml}
</map_card>

Write 5-10 insights the IGL can use in the next match, ordered by expected impact,
as markdown with one `##` heading per insight. Each insight has:
- **Read** - the conditioned pattern in one sentence ("after X, they do Y").
- **Why** - the causal chain behind it, reasoned from the rounds, not just the frequency.
- **Punish** - the exact counter-call a caller can lift word-for-word.
- **Evidence** - `match_id:round_num` cites and the sample size n.

Hard rules:
- Never present a consequence of normal play as an insight. Examples of banned
  tautologies: CTs leaving the other site post-plant to retake; Ts not standing on
  bombsites early; teams saving on a lost eco. An insight must be something a
  different team in the same situation would plausibly do differently.
- Prefer conditioned reads (previous round, economy state, first-contact outcome,
  utility spent) over raw frequencies. Cross-reference the round scripts to verify
  any pattern you claim - the scripts are the ground truth.
- Wrap every zone name in backticks and use only zones from the Map Card. Cite only
  rounds that exist in the data. State n for every claim; when the data is too thin
  for a read, say exactly that in one line rather than forcing one.
"""


def build_insights_user(
    *,
    teambook: TeamBook,
    utility_book: UtilityBook,
    gap_report: GapReport,
    econ_policy: EconPolicy,
    scripts: list[RoundScript],
) -> str:
    total_rounds = sum(t.n for t in teambook.tendencies if t.level == 0)
    sections = [
        (
            f"Data coverage: {len(teambook.generated_from)} demo(s), {total_rounds} rounds "
            f"of {teambook.team_key} on {teambook.map_name} "
            f"(match ids: {', '.join(teambook.generated_from)})."
        ),
        "",
        teambook.to_table_text(),
        "",
        "## Utility Book",
    ]
    for p in utility_book.patterns:
        lineup = f" [{p.lineup_id}]" if p.lineup_id else ""
        sections.append(
            f"- {p.side} {p.nade} -> `{p.to_zone}`{lineup}: {p.count}/{p.rounds_seen} rounds, "
            f"median {p.median_t:.0f}s, early-share {p.early_share:.0%} "
            f"(evidence: {', '.join(p.evidence[:4])})"
        )
    sections += ["", "## Gap Findings (15s windows; post-PL rows are the planted site only)"]
    for f in gap_report.findings:
        sections.append(
            f"- {f.side} vacate `{f.zone}` at {f.window} on '{f.trigger}': "
            f"{f.vacancy_rate:.0%} of {f.n} (baseline {f.baseline_rate:.0%}, lift {f.lift:+.0%}; "
            f"evidence: {', '.join(f.evidence[:4])})"
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
            f"traded when dying {r.trade_discipline:.0%}, modal B+15 {r.modal_zone_fe15}"
        )
    sections += ["", "## All Round Scripts (ground truth; movements included)"]
    for s in sorted(scripts, key=lambda s: (s.match_id, s.round_num)):
        sections.append(f"### {s.match_id}:{s.round_num}")
        sections.append(s.to_text(include_movements=True))
        sections.append("")
    return "\n".join(sections)


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
    max_tokens: int | None = None,  # None = the model's own maximum: never cut analysis short
) -> Insights:
    """One LLM call over the full corpus; fabrications surface as soft warnings."""
    system = build_insights_system(card.to_yaml())
    user = build_insights_user(
        teambook=teambook,
        utility_book=utility_book,
        gap_report=gap_report,
        econ_policy=econ_policy,
        scripts=scripts,
    )
    result = client.complete(system=system, user=user, max_tokens=max_tokens)

    valid_evidence = {f"{s.match_id}:{s.round_num}" for s in scripts}
    lint = lint_dossier(result.text, teambook, lexicon, valid_evidence)
    warnings = [f"Unknown zone: {z}" for z in lint.unknown_zones]
    warnings += [f"Bad citation: {c}" for c in lint.bad_citations]
    if result.truncated:
        warnings.append(
            "Output hit the model's token ceiling and is cut short - regenerate "
            "(the model may think less on a retry) or raise max_tokens."
        )

    return Insights(
        text=result.text,
        warnings=warnings,
        generated_from=list(teambook.generated_from),
        usage=result,
    )
