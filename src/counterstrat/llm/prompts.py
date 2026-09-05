"""System and user prompt construction and exemplar round selection for LLM scouting dossiers."""

import math
from typing import TYPE_CHECKING

from counterstrat.constants import RANGE_CLOSE_M, RANGE_LONG_M
from counterstrat.mining.range_profile import RangeProfile
from counterstrat.mining.tendencies import TeamBook

if TYPE_CHECKING:
    from counterstrat.mapcard.compile import MapCard

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

COORD_NOTE = "u: 0=west edge -> 1=east edge, v: 0=NORTH edge -> 1=SOUTH edge (v grows southward)"

# Space, lurker, and engagement-range doctrine (space-vision research §2-§3).
# Injected verbatim into the chat and insights system prompts so the model
# separates information-gathering positioning from duel-seeking positioning
# and reasons about which range a fight favors instead of treating every
# lone player as a peek threat.
TACTICAL_DOCTRINE = f"""Space and purpose:
- Map control is space plus information. A team holds an area when someone can punish
  entry (a body watching it, utility denying it, a trade setup covering it); an area
  nobody sees or punishes is conceded. Control only pays when converted - into an
  execute behind it, a confirmed-info call, or a pick. "They take `X` then never use
  it" and "they concede `Y` every round" are reads, not noise.
- Read a lone player's PURPOSE before prescribing the counter. Info lurk: watching a
  corridor the enemy must cross, repositions without contact, no teammate in trade
  range, holds fire. Duel lurk / angle hold: locked on one combat angle, still or
  re-peeking the same line, fires on first sight, often trade-supported.
- Counter an info lurk by denying and punishing information: do not cross his watched
  sightline, smoke it off when executing, and pre-aim his lurk-up corridor 2-3s after
  your first contact elsewhere - that is when info lurkers move. Rotate through paths
  he cannot see.
- Counter a duel lurk / angle hold by refusing the duel: clear with utility, peek in
  trade pairs, or ignore him - a held angle expires with the round clock.

Engagement range (close < {RANGE_CLOSE_M:.0f}m, long > {RANGE_LONG_M:.0f}m):
- Fight the range your weapons win and force the enemy to fight the range theirs
  lose. SMG/pistol buys win close: against them hold distance, refuse entries, take
  long first contacts. AWPs and scoped rifles win long: smoke the long sightline off
  and force the close fight instead of crossing it.
- A team's kill-distance profile is a read: kills clustered close mean tight
  positions that must close space - fight them long; kills clustered long mean
  static long-line holds - deny the line and flood the short route."""


def format_zone_map(anchors: dict[str, tuple]) -> str:
    """Label-anchor coordinates per zone, rendered into the map-card block.

    ``anchors`` is web.routes.map_zone_anchors output - the same source the
    callout editor draws, so the model reasons over exactly the positions the
    user sees and labels. Canonical names here; the Renamer swaps in user
    callouts at the prompt boundary.
    """
    if not anchors:
        return ""
    multi = any(a[2] == "lower" for a in anchors.values())
    lines = [
        (
            "zone_map:  # label anchor per zone on the radar; "
            "x: 0 = west edge -> 1 = east, y: 0 = north edge -> 1 = south"
        )
    ]
    for zone, (u, v, level) in sorted(anchors.items()):
        suffix = f", {'lower' if level == 'lower' else 'upper'} level" if multi else ""
        lines.append(f"- `{zone}` at ({u:.2f}, {v:.2f}){suffix}")
    return "\n".join(lines)


def _bearing(anchors: dict[str, tuple], a: str, b: str) -> str | None:
    """8-sector compass direction from zone a's anchor to zone b's."""
    pa, pb = anchors.get(a), anchors.get(b)
    if pa is None or pb is None:
        return None
    du, dv = pb[0] - pa[0], pb[1] - pa[1]
    if du == 0 and dv == 0:
        return None
    ang = math.degrees(math.atan2(du, -dv)) % 360  # 0=N, 90=E (v grows southward)
    return SECTORS[int(((ang + 22.5) % 360) // 45)]


def format_map_scene_graph(
    card: "MapCard", anchors: dict[str, tuple], sightlines: list[dict] | None = None
) -> str:
    """The measured spatial representation: topology backbone with labeled units.

    Per zone: radar anchor (u, v) plus tags; per undirected edge: move seconds
    and precomputed compass bearing. Rotates/timings render as derived route
    tables. Replaces the raw card.yaml dump (which lost route_time 1/10 vs
    10/10 in the 2026-09-04 representation lab) in every LLM prompt. Canonical
    zone names throughout; the Renamer swaps in user callouts at the boundary.
    """
    multi = any(a[2] == "lower" for a in anchors.values())

    def _pos(zone: str) -> str:
        a = anchors.get(zone)
        if a is None:
            return ""
        lvl = f", {'lower' if a[2] == 'lower' else 'upper'} level" if multi else ""
        return f" (u={a[0]:.2f}, v={a[1]:.2f}{lvl})"

    # undirected edges, min seconds of either direction
    edges: dict[str, dict[str, float]] = {}
    for u, nbrs in (card.topology or {}).items():
        for v, w in nbrs.items():
            w = float(w)
            if w <= 0 or u == v:
                continue
            cur = edges.setdefault(u, {}).get(v)
            if cur is None or w < cur:
                edges.setdefault(u, {})[v] = w
                edges.setdefault(v, {})[u] = w

    lines = [
        f"# Map: {card.map} - spatial scene graph",
        f"# {COORD_NOTE}",
        "# per zone: radar position, then '-> neighbor: move seconds, compass direction'",
        "zones:",
    ]
    for zone in sorted(card.zones):
        tags = (card.zones[zone] or {}).get("tags") or []
        tag_s = f" tags=[{', '.join(tags)}]" if tags else ""
        lines.append(f"`{zone}`{_pos(zone)}{tag_s}")
        for nb, sec in sorted(edges.get(zone, {}).items()):
            b = _bearing(anchors, zone, nb)
            lines.append(f"  -> `{nb}`: {sec:.1f}s{f' {b}' if b else ''}")

    sites = (card.objectives or {}).get("sites") or []
    if sites:
        obj = card.objectives
        lines.append(
            f"objectives: sites={sites}, round_seconds={obj.get('round_seconds')}, "
            f"bomb_seconds={obj.get('bomb_seconds')}"
        )
    if card.rotates:
        lines.append("rotates:  # site-to-site routes, seconds at run speed")
        for r in card.rotates:
            via = " > ".join(f"`{z}`" for z in (r.get("via") or []))
            lines.append(
                f"- `{r.get('from')}` -> `{r.get('to')}`: {float(r.get('run_s', 0)):.1f}s"
                + (f" via {via}" if via else "")
            )
    timing_lines = []
    for side in ("CT", "T"):
        rows = (card.timings or {}).get(side) or {}
        if rows:
            cells = ", ".join(f"`{z}` {s:.1f}s" for z, s in sorted(rows.items()))
            timing_lines.append(f"- {side}: {cells}")
    if timing_lines:
        lines.append("earliest_reach:  # seconds from spawn each side first reaches the zone")
        lines.extend(timing_lines)
    effective_sightlines = sightlines if sightlines is not None else card.sightlines
    if effective_sightlines:
        lines.append(
            "sightlines:  # zone pairs that SEE each other (observed kills; "
            "bidirectional; n = evidence count)"
        )
        for sl in effective_sightlines:
            lines.append(
                f"- `{sl['from']}` <-> `{sl['to']}`: {sl['range']} "
                f"(~{sl['median_dist']:.0f}m, n={sl['n']})"
            )
    return "\n".join(lines)


def build_chat_system(
    map_block: str,
    teambook: TeamBook,
    zone_map: str = "",
    range_profile: RangeProfile | None = None,
) -> str:
    """The interactive analyst chat system prompt: doctrine + signals + contract.

    Unlike the dossier prompt, this brief is exploit-first and length-capped:
    the teambook block below is already signal-filtered (level-2 noise rows
    are dropped by ``to_table_text``), and the contract forbids quoting
    unconcentrated distributions as reads (plan Task 7).
    """
    total_rounds = sum(t.n for t in teambook.tendencies if t.level == 0)
    demos = ", ".join(teambook.generated_from) or "none"
    range_block = ""
    if range_profile is not None and (range_lines := range_profile.to_prompt_lines()):
        range_block = "\n<engagement_range>\n" + "\n".join(range_lines) + "\n</engagement_range>\n"
    return f"""You are a CS2 anti-strat analyst briefing an in-game leader mid-preparation.
You think in triggers and punishes, not averages. This is an interactive chat, not a report.

Data coverage: {len(teambook.generated_from)} demo(s), {total_rounds} rounds of this team on
this map (match ids: {demos}). Every tool answer draws on all of them; more demos mean
stronger reads, so state the coverage when the analyst asks how reliable a read is.

<map_card>
{map_block}{zone_map}
</map_card>

<team_signals>
{teambook.to_table_text()}
</team_signals>
{range_block}
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

{TACTICAL_DOCTRINE}

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
- Advanced analytics: get_rotation_report for who breaks holds on what trigger and how
  fast, get_utility_roi for what each nade pattern buys, get_death_profiles for how
  players die (moving/crosshair/weapon), get_retake_report for post-plant conversion,
  get_matchup(opponent) to diff this team against another booked team on this map.
- Rotations are movement-derived correlations; CS2 demos carry no footstep or sound
  events, so never explain a rotation with audio cues.
- Cite rounds as match_id:round_num, taken only from tool output.
- Wrap every zone name in backticks and use only zones defined in the Map Card.
- If the tools do not cover the question, say so plainly instead of guessing.
"""
