"""Analyst chat tools: TeamBook lookups, round-script fetches, guarded SQL over the lake.

Every tool returns a JSON string and never raises -- tool errors reach the model as
data (``{"error": ...}``) so the agent loop can recover instead of crashing the turn.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from counterstrat.llm.base import ToolCall, ToolSpec
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mining.econ_policy import EconPolicy
from counterstrat.mining.gaps import GapReport

# _normalize_site is the same site normalisation the miner used to build the
# TeamBook, so `site` filters here agree with mined `site_committed` keys.
from counterstrat.mining.tendencies import TeamBook, _normalize_site
from counterstrat.mining.utility_book import UtilityBook
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
            for key in ("to_zone",):  # zone-valued tool filters arrive as callouts
                if isinstance(args.get(key), str):
                    args[key] = ctx.renamer.unalias_sql(f"'{args[key]}'")[1:-1]
        out = handler(ctx, args)
        return ctx.renamer.rename_text(out) if ctx.renamer else out
    except Exception as exc:  # noqa: BLE001 - a bad argument must not end the conversation
        return json.dumps({"error": f"{call.name} failed: {exc}"})
