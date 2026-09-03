"""System and user prompt construction and exemplar round selection for LLM scouting dossiers."""

from counterstrat.mining.tendencies import TeamBook
from counterstrat.roundscript.models import RoundScript


def build_system(card_yaml: str) -> str:
    """Build the system prompt containing the Map Card and dossier output contract."""
    return f"""You are an elite Counter-Strike 2 strategic analyst producing a comprehensive anti-strat scouting dossier.

<map_card>
{card_yaml}
</map_card>

Output Contract & Required Sections:
You must structure the dossier using the following exact seven sections:
1. Identity & Overview
2. Defaults & Roles
3. Execute Repertoire with Counters
4. Economy Policy with Exploit
5. Player-Specific Weaknesses
6. Round-State Playbook Table
7. Confidence & Evidence Appendix

Mandatory Rules:
- Zone formatting: Wrap EVERY zone name in backticks, e.g. `Middle`, `BombsiteA`. Only use valid zones defined in the Map Card. Do NOT invent zone names.
- Evidence citations: You must cite evidence as `match_id:round_num` (e.g. `deadbeefcafe1234:7`) for every specific pattern, round outcome, or tactical claim.
- Frequencies: Quote frequencies and percentages verbatim from the TeamBook profile. Do not round differently or hallucinate numbers.
- Sample sizes & confidence: Explicitly hedge any tendency marked `low_n` or based on a small sample size.
- Timings: Express round timings in seconds (e.g. 15s, 45s) rather than MM:SS notation to prevent citation ambiguity.
"""


def select_exemplars(
    teambook: TeamBook, scripts: list[RoundScript], cap: int = 12
) -> list[RoundScript]:
    """Select exemplar RoundScripts based on tendency coverage.

    For each of the top 6 tendencies by sample size `n`, takes up to 2 most recent
    evidence round IDs, deduplicating and maintaining order, capped at `cap` items.
    """
    script_map = {f"{s.match_id}:{s.round_num}": s for s in scripts}
    sorted_tendencies = sorted(teambook.tendencies, key=lambda t: t.n, reverse=True)

    selected_ids: list[str] = []
    seen: set[str] = set()

    for t in sorted_tendencies[:6]:
        recent_evidence = t.evidence[-2:]
        for eid in recent_evidence:
            if eid in script_map and eid not in seen:
                seen.add(eid)
                selected_ids.append(eid)
                if len(selected_ids) >= cap:
                    break
        if len(selected_ids) >= cap:
            break

    return [script_map[eid] for eid in selected_ids]


def build_user(teambook: TeamBook, exemplars: list[RoundScript]) -> str:
    """Build the user turn containing the TeamBook tables, sentences, and exemplar scripts."""
    sections = [
        "## Team Profile & Tendencies",
        teambook.to_table_text(),
        "",
        "## Key Tendency Summaries",
        "\n".join(f"- {s}" for s in teambook.to_sentences()),
        "",
        "## Exemplar Round Scripts",
    ]
    if exemplars:
        for s in exemplars:
            sections.append(f"### Round {s.match_id}:{s.round_num}")
            sections.append(s.to_text())
            sections.append("")
    else:
        sections.append("No exemplar round scripts available.")
    return "\n".join(sections)
