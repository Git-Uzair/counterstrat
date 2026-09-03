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

# _normalize_site is the same site normalisation the miner used to build the
# TeamBook, so `site` filters here agree with mined `site_committed` keys.
from counterstrat.mining.tendencies import TeamBook, _normalize_site
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


def tool_specs() -> list[ToolSpec]:
    """The five analyst tools, JSON-schema'd for both provider adapters."""
    return [
        ToolSpec(
            name="get_tendencies",
            description=(
                "Mined tendencies for this team on one side: first-contact zones, opening "
                "formations, site commitment, median first-contact time, sample size n, the "
                "low_n flag, and citable evidence round ids. Call this before asserting any "
                "default, setup or frequency."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "side": SIDE_SCHEMA,
                    "buy_class": BUY_CLASS_SCHEMA,
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
                "Full round script text for one round id ('match_id:round_num'), with beats, "
                "first contact, plant and utility. Quote beats from here rather than guessing."
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
        if t.key.side == side and (buy_class is None or t.key.buy_class == buy_class)
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
    return json.dumps({"id": round_id, "text": script.to_text()})


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


_HANDLERS: dict[str, Callable[[SessionContext, dict], str]] = {
    "get_tendencies": _tool_get_tendencies,
    "list_rounds": _tool_list_rounds,
    "get_round_script": _tool_get_round_script,
    "get_role_cards": _tool_get_role_cards,
    "sql_query": _tool_sql_query,
}


def execute_tool(ctx: SessionContext, call: ToolCall) -> str:
    """Run one tool call, returning its result as a JSON string. Never raises."""
    handler = _HANDLERS.get(call.name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {call.name}"})
    try:
        return handler(ctx, dict(call.arguments or {}))
    except Exception as exc:  # noqa: BLE001 - a bad argument must not end the conversation
        return json.dumps({"error": f"{call.name} failed: {exc}"})
