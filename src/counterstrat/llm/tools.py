"""Analyst chat tools: TeamBook lookups, round-script fetches, guarded SQL over the lake.

Every tool returns a JSON string and never raises -- tool errors reach the model as
data (``{"error": ...}``) so the agent loop can recover instead of crashing the turn.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from counterstrat.llm.base import ToolCall, ToolSpec
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mining.deaths import build_death_profiles
from counterstrat.mining.econ_policy import EconPolicy
from counterstrat.mining.gaps import GapReport
from counterstrat.mining.range_profile import build_range_profile
from counterstrat.mining.retakes import build_retake_report
from counterstrat.mining.rotations import build_rotation_report

# _normalize_site is the same site normalisation the miner used to build the
# TeamBook, so `site` filters here agree with mined `site_committed` keys.
from counterstrat.mining.tendencies import TeamBook, _normalize_site
from counterstrat.mining.utility_book import UtilityBook, build_utility_book
from counterstrat.mining.utility_roi import build_utility_roi
from counterstrat.roundscript.models import RoundScript

SQL_ROW_LIMIT = 50
SQL_SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)
SQL_FORBIDDEN_KEYWORDS = (
    "attach",
    "copy",
    "pragma",
    "install",
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "create",
)

SIDE_SCHEMA = {"type": "string", "enum": ["T", "CT"]}
BUY_CLASS_SCHEMA = {
    "type": "string",
    "enum": ["full_eco", "semi_eco", "semi_buy", "full_buy"],
}
NADE_SCHEMA = {"type": "string", "enum": ["smoke", "flash", "he", "molotov", "decoy"]}
SITUATION_SCHEMA = {
    "type": "string",
    "enum": ["pistol", "eco", "force", "full_buy", "after_loss", "after_win"],
}
TRIGGER_SCHEMA = {
    "type": "string",
    "enum": [
        "after_loss",
        "after_win",
        "behind",
        "ahead",
        "after_fc_win",
        "after_fc_loss",
        "after_util_dump",
    ],
}
ROTATION_TRIGGER_SCHEMA = {
    "type": "string",
    "enum": ["first_blood", "utility_near", "plant", "shots", "visible_contact"],
}
# Every rotation answer carries this so the model never invents sound reads.
ROTATION_NOTE = (
    "Rotations are movement-derived correlations: a hold of >=4s broken within 8s of a "
    "trigger (first_blood, utility_near = any enemy detonation, plant, shots, "
    "visible_contact via the sightline matrix). CS2 demos contain no footstep or sound "
    "events - never attribute a rotation to audio. Empty rows on corpora ingested "
    "before this upgrade: Rebuild the map to enrich them."
)
# Situation -> (buy classes, prev outcomes, econ policy states) used by get_playbook.
SITUATION_FILTERS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "pistol": ((), (), ("pistol",)),
    "eco": (("full_eco", "semi_eco"), (), ()),
    "force": (("semi_buy",), (), ()),
    "full_buy": (("full_buy",), (), ()),
    "after_loss": ((), ("lost",), ("after_loss_1", "after_loss_2", "after_loss_3plus")),
    "after_win": ((), ("won",), ("after_win",)),
}


class SessionContext(BaseModel):
    """Everything the analyst tools may read for one (team, map) chat session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    team_key: str
    map_name: str
    teambook: TeamBook
    lexicon: Lexicon
    card: MapCard
    con: Any = None  # duckdb connection over the lake views; None when no lake exists
    scripts: dict[str, RoundScript] = Field(default_factory=dict)  # "match_id:round" -> script
    # Task 6 mined artifacts; None on degraded/pre-upgrade sessions.
    utility_book: UtilityBook | None = None
    gap_report: GapReport | None = None
    econ_policy: EconPolicy | None = None
    # Callout renamer (counterstrat.aliases.Renamer); None = canonical names.
    renamer: Any = None
    # Data root for cross-team lookups (get_matchup); None disables them.
    data_root: str | None = None


def tool_specs() -> list[ToolSpec]:
    """The analyst tools, JSON-schema'd for both provider adapters."""
    return [
        ToolSpec(
            name="get_tendencies",
            description=(
                "Mined tendencies for this team on one side: first-contact zones, opening "
                "formations, site commitment, median first-contact time, sample size n, and "
                "citable evidence round ids. Rows exist at aggregation levels 0 (side "
                "rollup), 1 (side+buy) and 2 (full situation); only rows with signal=true "
                "are concentrated enough to quote as a read - treat the rest as noise or "
                "aggregate up a level. Call this before asserting any default, setup or "
                "frequency."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "buy_class": BUY_CLASS_SCHEMA,
                    "level": {"type": "integer", "enum": [0, 1, 2]},
                },
                "required": ["side"],
            },
        ),
        ToolSpec(
            name="list_rounds",
            description=(
                "List this team's rounds as {id, summary} pairs, optionally filtered by side, "
                "buy class, outcome and planted site. Use it to find rounds worth reading in "
                "full and to get citable round ids."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "buy_class": BUY_CLASS_SCHEMA,
                    "won": {"type": "boolean"},
                    "site": {"type": "string", "enum": ["A", "B", "none"]},
                },
            },
        ),
        ToolSpec(
            name="get_round_script",
            description=(
                "Complete round timeline for one round id ('match_id:round_num'): anchors "
                "(first contact/plant/end with seconds), 15s occupancy states, and every "
                "kill, movement and grenade with timestamps. Quote times and zones from "
                "here rather than guessing."
            ),
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        ),
        ToolSpec(
            name="get_role_cards",
            description=(
                "Per-player role cards for this team: modal zone 15s after freeze end per side, "
                "opening-duel rate and lurk rate."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="get_playbook",
            description=(
                "One-call situational brief: for a round situation (pistol, eco, force, "
                "full_buy, after_loss, after_win) returns the matching signal tendencies, "
                "the team's buy policy in that money state, pistol site leans when "
                "relevant, utility dump timing windows, and any gap findings triggered by "
                "that situation. Use this first for 'what do they do on X rounds' "
                "questions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "situation": SITUATION_SCHEMA,
                    "side": SIDE_SCHEMA,
                },
                "required": ["situation"],
            },
        ),
        ToolSpec(
            name="get_utility_book",
            description=(
                "Mined recurring utility: per (side, nade, target zone) the round count, "
                "share of rounds, origin zones, dominant lineup id, median throw time and "
                "early-package share, plus dump windows (median time their k-th nade is "
                "spent). Use for 'where do their smokes go', 'what is their exec package' "
                "and 'when is their utility gone'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "nade": NADE_SCHEMA,
                    "to_zone": {"type": "string"},
                },
                "required": ["side"],
            },
        ),
        ToolSpec(
            name="get_gap_report",
            description=(
                "Where this team leaves key zones (bombsites) unoccupied, by 15s beat "
                "window, with the trigger that opens the gap (after_loss, after_win, "
                "behind, ahead, first contact won/lost, utility dump) and the lift over "
                "the unconditioned baseline. Windows come from 15s formation beats, so "
                "sub-15s rotations are invisible. Use for 'where are they weak and when'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "trigger": TRIGGER_SCHEMA,
                },
                "required": ["side"],
            },
        ),
        ToolSpec(
            name="get_economy_read",
            description=(
                "The team's buy policy by money state (pistol, after_win, after_loss "
                "streaks), T-side pistol site leans, and the buy split after losing a "
                "pistol. Use to predict their next buy and pick anti-eco or force-punish "
                "calls."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
        ToolSpec(
            name="get_player_profile",
            description=(
                "Extended per-player profile: modal zones, opening-duel rate and win "
                "rate, opening zones, median first-contact time, AWP rounds, lurk rate "
                "and trade discipline. Omit 'player' for the full roster."
            ),
            input_schema={
                "type": "object",
                "properties": {"player": {"type": "string"}},
            },
        ),
        ToolSpec(
            name="get_rotation_report",
            description=(
                "Who breaks their hold on what trigger, how fast, and how often the "
                "trigger was a fake (no follow-up contact at the trigger zone within "
                "10s). Per (player, trigger): median latency seconds, fake rate, n, "
                "evidence rounds. Movement-derived correlation only - CS2 demos carry "
                "no sound/footstep events, so never explain a rotation with audio. Use "
                "for 'who over-rotates' and 'how do they react to utility/plants'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "trigger": ROTATION_TRIGGER_SCHEMA,
                },
            },
        ),
        ToolSpec(
            name="get_utility_roi",
            description=(
                "What each recurring nade pattern (lineup or from>to pair) actually "
                "buys: avg enemy/team blind seconds (sums per flash), damage per "
                "HE/molly, kills through each smoke, unit cost, and cost-per-value "
                "verdicts (only at n>=5; below that quote the numbers and hedge). "
                "None fields mean 'not measured' (pre-upgrade corpus), never zero. "
                "Use for 'is their utility efficient' and 'which flashes hurt us'."
            ),
            input_schema={
                "type": "object",
                "properties": {"side": SIDE_SCHEMA},
            },
        ),
        ToolSpec(
            name="get_death_profiles",
            description=(
                "How each of their players dies: % of deaths while moving, median "
                "crosshair-off-killer degrees at death, weapon out when they died, "
                "split by engagement range band. Sub-sample n per field (old corpora "
                "carry unmeasured deaths). Use for 'who peeks dry', 'who gets caught "
                "repositioning' and 'whose crosshair is off'."
            ),
            input_schema={
                "type": "object",
                "properties": {"player": {"type": "string"}},
            },
        ),
        ToolSpec(
            name="get_retake_report",
            description=(
                "Post-plant conversion from both chairs: retake win rate per (site, "
                "man-diff at plant) when they were CT, post-plant hold win rate when "
                "they were T, plus approach-vector sets (zones the retakers entered "
                "the site from, when tracks exist). Small corpora mean small n - "
                "state it. Use for 'what's their B retake conversion' and 'how do "
                "they approach retakes'."
            ),
            input_schema={
                "type": "object",
                "properties": {"site": {"type": "string"}},
            },
        ),
        ToolSpec(
            name="get_matchup",
            description=(
                "Diff this team against ANOTHER booked team on the same map (every "
                "demo books both teams, so opponents you ingested are queryable by "
                "name). Returns their top utility patterns and dump windows vs our "
                "gap findings and timings, plus both engagement-range profiles. "
                "Unknown names return the list of available teams. Use when the "
                "analyst names the next opponent."
            ),
            input_schema={
                "type": "object",
                "properties": {"opponent": {"type": "string"}},
                "required": ["opponent"],
            },
        ),
        ToolSpec(
            name="sql_query",
            description=(
                "Read-only DuckDB SELECT over the lake views (rounds, kills, damages, shots, "
                "grenades, smokes, infernos, bomb, item_purchase, ticks, rosters). Single "
                f"statement, no semicolons; results are capped at {SQL_ROW_LIMIT} rows."
            ),
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


def _team_side(ctx: SessionContext, script: RoundScript) -> str | None:
    """Which side the session's team played in this round, or None if it did not."""
    if script.t_team_key == ctx.team_key:
        return "T"
    if script.ct_team_key == ctx.team_key:
        return "CT"
    return None


def _tool_get_tendencies(ctx: SessionContext, args: dict) -> str:
    side = str(args.get("side") or "").upper()
    buy_class = args.get("buy_class")
    level = args.get("level")
    rows = [
        {
            "key": t.key.model_dump(),
            "level": t.level,
            "first_contact_zone": t.first_contact_zone,
            "opening_formation": t.opening_formation,
            "site_committed": t.site_committed,
            "median_first_contact_s": t.median_first_contact_s,
            "n": t.n,
            "low_n": t.low_n,
            "fc_concentration": t.fc_concentration,
            "signal": t.signal,
            "evidence": t.evidence,
        }
        for t in ctx.teambook.tendencies
        if t.key.side == side
        and (buy_class is None or t.key.buy_class == buy_class)
        and (level is None or t.level == level)
    ]
    return json.dumps(
        {"team_key": ctx.team_key, "side": side, "buy_class": buy_class, "tendencies": rows}
    )


def _tool_list_rounds(ctx: SessionContext, args: dict) -> str:
    side = str(args["side"]).upper() if args.get("side") else None
    buy_class = args.get("buy_class")
    won = args.get("won")
    site = _normalize_site(str(args["site"])) if args.get("site") else None

    rows: list[dict[str, str]] = []
    for script in sorted(ctx.scripts.values(), key=lambda s: (s.match_id, s.round_num)):
        team_side = _team_side(ctx, script)
        if team_side is None or (side is not None and team_side != side):
            continue
        econ = script.economy.get(team_side)
        if buy_class is not None and (econ is None or econ.buy_type != buy_class):
            continue
        if won is not None and (script.winner in (team_side, ctx.team_key)) is not bool(won):
            continue
        if site is not None:
            planted = _normalize_site(script.plant.site) if script.plant else "none"
            if planted != site:
                continue
        rows.append(
            {
                "id": f"{script.match_id}:{script.round_num}",
                "summary": script.to_text().splitlines()[0],
            }
        )
    return json.dumps({"rounds": rows, "count": len(rows)})


def _tool_get_round_script(ctx: SessionContext, args: dict) -> str:
    round_id = str(args.get("id") or "")
    script = ctx.scripts.get(round_id)
    if script is None:
        return json.dumps({"error": f"Round {round_id} not found"})
    return json.dumps({"id": round_id, "text": script.to_timeline_text()})


def _tool_get_role_cards(ctx: SessionContext, args: dict) -> str:
    return json.dumps({"roles": [r.model_dump() for r in ctx.teambook.roles]})


def _tool_sql_query(ctx: SessionContext, args: dict) -> str:
    query = str(args.get("query") or "")
    if not SQL_SELECT_RE.match(query):
        return json.dumps({"error": "Only SELECT queries allowed"})
    if ";" in query:
        return json.dumps({"error": "Forbidden token in query: ';' (single statement only)"})
    lowered = query.lower()
    for keyword in SQL_FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            return json.dumps({"error": f"Forbidden keyword in query: {keyword}"})
    if ctx.con is None:
        return json.dumps({"error": "No lake data available for this session"})

    wrapped = f"SELECT * FROM ({query.strip()}) LIMIT {SQL_ROW_LIMIT}"
    try:
        rel = ctx.con.sql(wrapped)
        columns = list(rel.columns)
        rows = [dict(zip(columns, record, strict=True)) for record in rel.fetchall()]
    except Exception as exc:  # noqa: BLE001 - SQL failures are data for the model
        return json.dumps({"error": str(exc)})
    return json.dumps({"rows": rows}, default=str)


def _tendency_row(t) -> dict:
    return {
        "key": t.key.model_dump(),
        "level": t.level,
        "first_contact_zone": t.first_contact_zone,
        "site_committed": t.site_committed,
        "median_first_contact_s": t.median_first_contact_s,
        "n": t.n,
        "signal": t.signal,
        "evidence": t.evidence,
    }


def _tool_get_playbook(ctx: SessionContext, args: dict) -> str:
    if ctx.econ_policy is None or ctx.utility_book is None or ctx.gap_report is None:
        return json.dumps({"error": "No mined playbook artifacts for this session (re-ingest)"})
    situation = str(args.get("situation") or "")
    if situation not in SITUATION_FILTERS:
        return json.dumps({"error": f"Unknown situation: {situation!r}"})
    side = str(args["side"]).upper() if args.get("side") else None
    buys, prevs, econ_states = SITUATION_FILTERS[situation]

    tendencies = [
        _tendency_row(t)
        for t in ctx.teambook.tendencies
        if t.signal
        and (side is None or t.key.side == side)
        and (not buys or t.key.buy_class in buys)
        and (not prevs or t.key.prev_outcome in prevs)
        and (buys or prevs)  # pistol has no tendency dimension
    ]
    econ = {
        "states": {
            s: ctx.econ_policy.policy[s] for s in econ_states if s in ctx.econ_policy.policy
        },
        "n": sum(ctx.econ_policy.ns.get(s, 0) for s in econ_states),
        "evidence": {s: ctx.econ_policy.evidence.get(s, []) for s in econ_states},
    }
    out: dict[str, Any] = {
        "situation": situation,
        "side": side,
        "tendencies": tendencies,
        "econ": econ,
        "dump_windows": [
            w for w in ctx.utility_book.dump_windows if side is None or w["side"] == side
        ],
        "related_gaps": [
            f.model_dump()
            for f in ctx.gap_report.findings
            if f.trigger == situation and (side is None or f.side == side)
        ],
    }
    if situation == "pistol":
        out["pistol_sites"] = ctx.econ_policy.pistol_round_sites
        out["post_pistol_loss_buy"] = ctx.econ_policy.post_pistol_loss_buy
    return json.dumps(out)


def _tool_get_utility_book(ctx: SessionContext, args: dict) -> str:
    if ctx.utility_book is None:
        return json.dumps({"error": "No utility book for this session (re-ingest)"})
    side = str(args.get("side") or "").upper()
    nade = args.get("nade")
    to_zone = args.get("to_zone")
    patterns = [
        p.model_dump()
        for p in ctx.utility_book.patterns
        if p.side == side
        and (nade is None or p.nade == nade)
        and (to_zone is None or p.to_zone == to_zone)
    ]
    windows = [w for w in ctx.utility_book.dump_windows if w["side"] == side]
    return json.dumps({"side": side, "patterns": patterns, "dump_windows": windows})


def _tool_get_gap_report(ctx: SessionContext, args: dict) -> str:
    if ctx.gap_report is None:
        return json.dumps({"error": "No gap report for this session (re-ingest)"})
    side = str(args.get("side") or "").upper()
    trigger = args.get("trigger")
    findings = [
        f.model_dump()
        for f in ctx.gap_report.findings
        if f.side == side and (trigger is None or f.trigger in (trigger, "base"))
    ]
    return json.dumps(
        {
            "side": side,
            "key_zones": ctx.gap_report.key_zones,
            "window_note": "Windows are 15s formation beats; sub-15s rotations are invisible.",
            "findings": findings,
        }
    )


def _tool_get_economy_read(ctx: SessionContext, args: dict) -> str:
    if ctx.econ_policy is None:
        return json.dumps({"error": "No economy policy for this session (re-ingest)"})
    return json.dumps(ctx.econ_policy.model_dump())


def _tool_get_player_profile(ctx: SessionContext, args: dict) -> str:
    player = args.get("player")
    roles = ctx.teambook.roles
    if player:
        wanted = str(player).lower()
        roles = [r for r in roles if r.player.lower() == wanted]
        if not roles:
            known = ", ".join(r.player for r in ctx.teambook.roles)
            return json.dumps({"error": f"Unknown player {player!r}; roster: {known}"})
    return json.dumps({"roles": [r.model_dump() for r in roles]})


def _tool_get_rotation_report(ctx: SessionContext, args: dict) -> str:
    side = str(args["side"]).upper() if args.get("side") else None
    trigger = args.get("trigger")
    report = build_rotation_report(list(ctx.scripts.values()), ctx.team_key)
    rows = [
        r.model_dump()
        for r in report.rows
        if (side is None or r.side == side) and (trigger is None or r.trigger == trigger)
    ]
    return json.dumps({"note": ROTATION_NOTE, "side": side, "trigger": trigger, "rows": rows})


def _tool_get_utility_roi(ctx: SessionContext, args: dict) -> str:
    side = str(args["side"]).upper() if args.get("side") else None
    roi = build_utility_roi(list(ctx.scripts.values()), ctx.team_key)
    rows = [r.model_dump() for r in roi.rows if side is None or r.side == side]
    return json.dumps(
        {
            "note": (
                "Blind seconds are SUMS across victims per flash; None = not measured "
                "(pre-upgrade corpus), never zero. Cost-per-value verdicts only at n>=5. "
                "cost_per_* fields are efficiency RATIOS (unit price / avg measured "
                "effect), not prices: a $600 incendiary averaging 0.16 damage reads as "
                "$3750 per damage point while still costing $600. Zoning utility "
                "legitimately reads near-zero damage - its value is map control, which "
                "these numbers cannot see - so never call a nade wasted on damage alone."
            ),
            "side": side,
            "rows": rows,
        }
    )


def _tool_get_death_profiles(ctx: SessionContext, args: dict) -> str:
    profiles = build_death_profiles(list(ctx.scripts.values()), ctx.team_key)
    players = profiles.players
    if args.get("player"):
        wanted = str(args["player"]).lower()
        players = [p for p in players if p.player.lower() == wanted]
        if not players:
            known = ", ".join(p.player for p in profiles.players) or "none recorded"
            return json.dumps({"error": f"No deaths for {args['player']!r}; roster: {known}"})
    return json.dumps(
        {
            "note": (
                "moving_n/preaim_n are the measured sub-samples; deaths from "
                "pre-upgrade corpora carry no context fields."
            ),
            "players": [p.model_dump() for p in players],
        }
    )


def _tool_get_retake_report(ctx: SessionContext, args: dict) -> str:
    report = build_retake_report(list(ctx.scripts.values()), ctx.team_key)
    site = str(args["site"]) if args.get("site") else None
    rows = [r.model_dump() for r in report.rows if site is None or r.site == site]
    approaches = [a.model_dump() for a in report.approaches if site is None or a.site == site]
    return json.dumps(
        {
            "note": (
                "side=CT rows are their retakes, side=T rows their post-plant holds; "
                "man_diff is the acting side's advantage at the plant. Approach rows "
                "on T-side describe the ENEMY retake vectors they held against."
            ),
            "site": site,
            "rows": rows,
            "approaches": approaches,
        }
    )


def _tool_get_matchup(ctx: SessionContext, args: dict) -> str:
    opponent = str(args.get("opponent") or "").strip()
    if not opponent:
        return json.dumps({"error": "Name the opponent team to compare against"})
    if not ctx.data_root:
        return json.dumps({"error": "Matchup lookup is unavailable in this session"})
    from counterstrat.teams import load_or_build_clusters

    data_root = Path(ctx.data_root)
    clusters = {c.team_id: c for c in load_or_build_clusters(data_root).values()}
    booked = [
        c
        for c in clusters.values()
        if c.team_id != ctx.team_key and any(m.map_name == ctx.map_name for m in c.matches.values())
    ]
    target = next(
        (
            c
            for c in booked
            if c.name.lower() == opponent.lower()
            or c.team_id == opponent
            or opponent in c.lineup_keys
        ),
        None,
    )
    if target is None:
        names = sorted(f"{c.name} ({c.team_id})" for c in booked)
        return json.dumps(
            {
                "error": (
                    f"No booked team {opponent!r} on {ctx.map_name}. "
                    f"Available: {', '.join(names) if names else 'none'}"
                )
            }
        )

    keys = target.all_keys()
    their_scripts: list[RoundScript] = []
    match_ids = sorted(m for m, tm in target.matches.items() if tm.map_name == ctx.map_name)
    for mid in match_ids:
        for path in sorted((data_root / "scripts" / mid).glob("round_*.json")):
            try:
                s = RoundScript.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001, S112 - one bad script must not kill the diff
                continue
            update: dict[str, str] = {}
            if s.t_team_key in keys:
                update["t_team_key"] = target.team_id
            if s.ct_team_key in keys:
                update["ct_team_key"] = target.team_id
            their_scripts.append(s.model_copy(update=update) if update else s)
    if not their_scripts:
        return json.dumps(
            {"error": f"No round scripts on disk for {target.name} on {ctx.map_name}"}
        )

    theirs_book = build_utility_book(their_scripts, target.team_id)
    our_scripts = list(ctx.scripts.values())
    theirs_played = sum(1 for s in their_scripts if target.team_id in (s.t_team_key, s.ct_team_key))
    our_gaps = ctx.gap_report.findings if ctx.gap_report else []
    their_zones = {p.to_zone for p in theirs_book.patterns}
    return json.dumps(
        {
            "note": (
                "Diff brief: their patterns mined over their own rounds, ours over this "
                "session's scope. overlap_zones = zones their utility targets that also "
                "appear in our gap findings - the collision points to plan around."
            ),
            "opponent": {
                "name": target.name,
                "team_id": target.team_id,
                "matches": len(match_ids),
                "rounds": theirs_played,
            },
            "their_top_utility": [p.model_dump() for p in theirs_book.top_patterns(limit=12)],
            "their_dump_windows": theirs_book.dump_windows,
            "their_range_profile": build_range_profile(
                their_scripts, target.team_id
            ).to_prompt_lines(),
            "our_gap_findings": [f.model_dump() for f in our_gaps[:12]],
            "our_dump_windows": ctx.utility_book.dump_windows if ctx.utility_book else [],
            "our_range_profile": build_range_profile(our_scripts, ctx.team_key).to_prompt_lines(),
            "overlap_zones": sorted(their_zones & {f.zone for f in our_gaps}),
        }
    )


_HANDLERS: dict[str, Callable[[SessionContext, dict], str]] = {
    "get_tendencies": _tool_get_tendencies,
    "list_rounds": _tool_list_rounds,
    "get_round_script": _tool_get_round_script,
    "get_role_cards": _tool_get_role_cards,
    "get_playbook": _tool_get_playbook,
    "get_utility_book": _tool_get_utility_book,
    "get_gap_report": _tool_get_gap_report,
    "get_economy_read": _tool_get_economy_read,
    "get_player_profile": _tool_get_player_profile,
    "get_rotation_report": _tool_get_rotation_report,
    "get_utility_roi": _tool_get_utility_roi,
    "get_death_profiles": _tool_get_death_profiles,
    "get_retake_report": _tool_get_retake_report,
    "get_matchup": _tool_get_matchup,
    "sql_query": _tool_sql_query,
}


def execute_tool(ctx: SessionContext, call: ToolCall) -> str:
    """Run one tool call, returning its result as a JSON string. Never raises.

    With a renamer attached, the model lives entirely in the user's callout
    vocabulary: quoted SQL literals are mapped back to canonical zone names on
    the way in, and every result is renamed on the way out.
    """
    handler = _HANDLERS.get(call.name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {call.name}"})
    try:
        args = dict(call.arguments or {})
        if ctx.renamer:
            if call.name == "sql_query" and args.get("sql"):
                args["sql"] = ctx.renamer.unalias_sql(str(args["sql"]))
            for key in ("to_zone", "site"):  # zone-valued tool filters arrive as callouts
                if isinstance(args.get(key), str):
                    args[key] = ctx.renamer.unalias_sql(f"'{args[key]}'")[1:-1]
        out = handler(ctx, args)
        return ctx.renamer.rename_text(out) if ctx.renamer else out
    except Exception as exc:  # noqa: BLE001 - a bad argument must not end the conversation
        return json.dumps({"error": f"{call.name} failed: {exc}"})
