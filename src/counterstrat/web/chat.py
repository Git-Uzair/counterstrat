"""Analyst chat sessions: (team, map)-scoped, tool-grounded conversations."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from counterstrat.config import AppConfig
from counterstrat.lake.duck import connect_lake
from counterstrat.llm.agent import AgentReply, run_agent
from counterstrat.llm.base import ChatTurn, make_client
from counterstrat.llm.prompts import build_system
from counterstrat.llm.tools import SessionContext
from counterstrat.mapcard.compile import MapCard
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mining.tendencies import TeamBook
from counterstrat.roundscript.models import RoundScript
from counterstrat.web.routes import ConfigDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat")

CHAT_TOOL_RULES = """You are now in an interactive analyst chat, not writing a dossier: ignore
the seven-section dossier structure above and answer the analyst's question directly.

Tool-usage rules:
- Always check a tendency (get_tendencies), a round (list_rounds / get_round_script), a role card
  (get_role_cards) or the lake (sql_query) before asserting anything about this team. Never answer
  a factual question from memory.
- State the sample size n behind every frequency you quote, and hedge explicitly whenever the
  tendency is flagged low_n.
- Cite rounds as match_id:round_num, taken only from tool output.
- Wrap every zone name in backticks and use only zones from the Map Card above.
- If the tools do not cover the question, say so plainly instead of guessing.
"""


class ChatSession(BaseModel):
    """One live chat: its tool context, cached system block, and turn history."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    team_key: str
    map_name: str
    system: str
    ctx: SessionContext
    history: list[ChatTurn] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    team_key: str
    map_name: str


class CreateSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    text: str


class TranscriptResponse(BaseModel):
    session_id: str
    team_key: str
    map_name: str
    messages: list[dict[str, Any]]


def _sessions(request: Request) -> dict[str, ChatSession]:
    store = getattr(request.app.state, "chat_sessions", None)
    if store is None:
        store = {}
        request.app.state.chat_sessions = store
    return store


def _transcript_path(cfg: AppConfig, sid: str) -> Path:
    return cfg.data_root / "chats" / f"{sid}.jsonl"


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_transcript(cfg: AppConfig, sid: str) -> tuple[dict[str, Any] | None, list[ChatTurn]]:
    """Read a session's on-disk transcript as (metadata, replayable turns)."""
    path = _transcript_path(cfg, sid)
    if not path.exists():
        return None, []
    meta: dict[str, Any] | None = None
    turns: list[ChatTurn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # skip a torn trailing line
            continue
        if record.get("type") == "meta":
            meta = record
        elif record.get("role") in ("user", "assistant"):
            turns.append(ChatTurn(role=record["role"], text=record.get("text") or ""))
    return meta, turns


def _lake_connection(cfg: AppConfig) -> Any:
    """A duckdb connection over the lake views, or None when there is no lake yet."""
    lake_root = cfg.data_root / "lake"
    if not any(lake_root.glob("*/rounds.parquet")):
        return None
    try:
        return connect_lake(lake_root)
    except Exception as exc:  # noqa: BLE001 - chat still works without SQL access
        logger.warning("Lake connection unavailable for chat: %s", exc)
        return None


def _system_prompt(card: MapCard, teambook: TeamBook) -> str:
    """Map Card + TeamBook + tool rules, all in one cacheable system block."""
    return (
        f"{build_system(card.to_yaml())}\n"
        f"<teambook>\n{teambook.to_table_text()}\n</teambook>\n\n"
        f"{CHAT_TOOL_RULES}"
    )


def _build_session(cfg: AppConfig, session_id: str, team_key: str, map_name: str) -> ChatSession:
    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if not card_path.exists():
        raise HTTPException(status_code=404, detail=f"Map card for {map_name} not found")

    try:
        teambook = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
        card = MapCard(**yaml.safe_load(card_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load chat session data: {exc}"
        ) from exc

    overlay_path = get_default_overlay_path(map_name)
    lexicon = build_lexicon(
        map_name, list(card.zones.keys()), overlay_path if overlay_path.exists() else None
    )

    scripts: dict[str, RoundScript] = {}
    for match_id in teambook.generated_from:
        for script_path in sorted((cfg.data_root / "scripts" / match_id).glob("round_*.json")):
            try:
                script = RoundScript.model_validate_json(script_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - one bad script must not kill chat
                logger.warning("Skipping unreadable round script %s: %s", script_path, exc)
                continue
            scripts[f"{script.match_id}:{script.round_num}"] = script

    ctx = SessionContext(
        team_key=team_key,
        map_name=map_name,
        teambook=teambook,
        lexicon=lexicon,
        card=card,
        con=_lake_connection(cfg),
        scripts=scripts,
    )
    return ChatSession(
        session_id=session_id,
        team_key=team_key,
        map_name=map_name,
        system=_system_prompt(card, teambook),
        ctx=ctx,
    )


def _get_session(request: Request, cfg: AppConfig, sid: str) -> ChatSession:
    """In-memory session, rebuilt from the on-disk transcript after a restart."""
    store = _sessions(request)
    session = store.get(sid)
    if session is not None:
        return session
    meta, turns = _read_transcript(cfg, sid)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{sid}' not found")
    session = _build_session(cfg, sid, str(meta.get("team_key")), str(meta.get("map_name")))
    session.history = turns
    store[sid] = session
    return session


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    req: CreateSessionRequest, request: Request, cfg: ConfigDep
) -> CreateSessionResponse:
    session_id = uuid.uuid4().hex[:12]
    session = _build_session(cfg, session_id, req.team_key, req.map_name)
    _sessions(request)[session_id] = session
    _append_record(
        _transcript_path(cfg, session_id),
        {
            "type": "meta",
            "session_id": session_id,
            "team_key": req.team_key,
            "map_name": req.map_name,
        },
    )
    return CreateSessionResponse(session_id=session_id)


@router.post("/sessions/{sid}/messages", response_model=AgentReply)
def post_message(sid: str, req: MessageRequest, request: Request, cfg: ConfigDep) -> AgentReply:
    session = _get_session(request, cfg, sid)

    # Resolve the client before recording the turn so a key-less install cannot
    # leave a dangling user message in the transcript.
    factory = getattr(request.app.state, "client_factory", None)
    try:
        client = factory(cfg) if factory else make_client(cfg)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No API key configured for provider '{cfg.provider}'. Set it in Settings.",
        ) from exc

    path = _transcript_path(cfg, sid)
    session.history.append(ChatTurn(role="user", text=req.text))
    _append_record(path, {"role": "user", "text": req.text})

    try:
        reply = run_agent(client, system=session.system, history=session.history, ctx=session.ctx)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat completion failed: {exc}") from exc

    session.history.append(ChatTurn(role="assistant", text=reply.text))
    _append_record(
        path,
        {
            "role": "assistant",
            "text": reply.text,
            "warnings": reply.warnings,
            "tool_trace": reply.tool_trace,
        },
    )
    return reply


@router.get("/sessions/{sid}", response_model=TranscriptResponse)
def get_session(sid: str, request: Request, cfg: ConfigDep) -> TranscriptResponse:
    session = _sessions(request).get(sid)
    if session is not None:
        return TranscriptResponse(
            session_id=sid,
            team_key=session.team_key,
            map_name=session.map_name,
            messages=[{"role": t.role, "text": t.text or ""} for t in session.history],
        )

    meta, turns = _read_transcript(cfg, sid)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Chat session '{sid}' not found")
    return TranscriptResponse(
        session_id=sid,
        team_key=str(meta.get("team_key") or ""),
        map_name=str(meta.get("map_name") or ""),
        messages=[{"role": t.role, "text": t.text or ""} for t in turns],
    )
