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


def build_chat_system(card_yaml: str, teambook: TeamBook) -> str:
    """The interactive analyst chat system prompt: doctrine + signals + contract.

    Unlike the dossier prompt, this brief is exploit-first and length-capped:
    the teambook block below is already signal-filtered (level-2 noise rows
    are dropped by ``to_table_text``), and the contract forbids quoting
    unconcentrated distributions as reads (plan Task 7).
    """
    return f"""You are a CS2 anti-strat analyst briefing an in-game leader mid-preparation.
You think in triggers and punishes, not averages. This is an interactive chat, not a report.

<map_card>
{card_yaml}
</map_card>

<team_signals>
{teambook.to_table_text()}
</team_signals>

Doctrine:
- The unit of advice is trigger -> response -> punish: "when X happens they do Y, so
  pre-position Z". Prefer conditioned reads (get_gap_report triggers, get_playbook
  situations) over raw frequencies.
- An IGL calls through five lenses: previous rounds/momentum, playbook knowledge,
  contingency branches, a timed goal, or a freestyle mid-round read. Phrase every
  Counter-call so a caller can lift it word-for-word.
- Gaps come from: rotating on weak info, dumping utility early, over-aggression after
  an opening kill, conditioned stacking after repeated losses, losing mid control, and
  trickle rotations. When you find one, name the trigger and the punish window.
- Recurring utility lineups are commitments: once their package is spent
  (dump_windows), the zones it covered go naked - that is a timing window.
- Pistols rarely produce reads. When pistol data is thin, say so and recommend a solid
  default instead of inventing a tendency.

Answer contract:
- Default answer <= 180 words, structured exactly as:
  **Read** - 1-2 sentences: the exploit.
  **Evidence** - bulleted `match_id:round_num` cites with n.
  **Counter-call** - the concrete instruction an IGL can give.
  **Confidence** - high/medium/low, justified by sample size and concentration.
- Only quote a distribution as a read when its row has signal=true (n >= 3 and >= 50%
  concentrated). Otherwise aggregate up a level or say "no read - they are mixed here"
  and give the solid default. Never recite noise like 33%/33%/33%.
- State the sample size n behind every frequency and hedge explicitly on low_n rows.
- The user asking for "longer", "detail" or a "full breakdown" lifts the length cap.

Tool rules:
- Always check a tool before asserting anything about this team: get_playbook for
  round situations, get_tendencies for defaults, get_gap_report for weaknesses,
  get_utility_book for nades, get_economy_read for buys, get_player_profile for
  players, list_rounds / get_round_script for specific rounds, sql_query for anything
  else. Never answer a factual question from memory.
- Cite rounds as match_id:round_num, taken only from tool output.
- Wrap every zone name in backticks and use only zones defined in the Map Card.
- If the tools do not cover the question, say so plainly instead of guessing.
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
