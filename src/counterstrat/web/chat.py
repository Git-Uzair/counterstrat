"""Analyst chat sessions: (team, map)-scoped, tool-grounded conversations."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from counterstrat.aliases import load_renamer
from counterstrat.config import AppConfig
from counterstrat.lake.duck import connect_lake
from counterstrat.llm.agent import AgentReply, run_agent
from counterstrat.llm.base import ChatTurn, make_client
from counterstrat.llm.prompts import build_chat_system
from counterstrat.llm.tools import SessionContext
from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.transitions import zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.vrf import extract_map_assets
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.tendencies import TeamBook, build_teambook
from counterstrat.mining.utility_book import build_utility_book
from counterstrat.roundscript.models import RoundScript
from counterstrat.teams import load_or_build_clusters
from counterstrat.web.ingest import _find_vpk_path, _find_vrf_cli, _rekey
from counterstrat.web.routes import ConfigDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat")


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
    match_id: str | None = None  # scope the session to one match ("what happened THAT game")


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
    """Doctrine + map card + signal-filtered teambook, in one cacheable block."""
    return build_chat_system(card.to_yaml(), teambook)


def _build_session(
    cfg: AppConfig,
    session_id: str,
    team_key: str,
    map_name: str,
    match_id: str | None = None,
) -> ChatSession:
    # Any lineup key resolves to its team cluster; scripts below are re-keyed to
    # the canonical id so stand-in lineups analyze as one team.
    cluster = load_or_build_clusters(cfg.data_root).get(team_key)
    team_key = cluster.team_id if cluster else team_key
    cluster_keys = cluster.all_keys() if cluster else {team_key}

    tb_path = cfg.data_root / "teambooks" / team_key / map_name / "teambook.json"
    if not tb_path.exists():
        raise HTTPException(
            status_code=404, detail=f"TeamBook for {team_key} on {map_name} not found"
        )

    try:
        teambook = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to load chat session data: {exc}"
        ) from exc

    source_matches = list(teambook.generated_from)
    if match_id is not None:
        if match_id not in source_matches:
            raise HTTPException(
                status_code=404,
                detail=f"Match {match_id} has no rounds for {team_key} on {map_name}",
            )
        source_matches = [match_id]

    scripts: dict[str, RoundScript] = {}
    for mid in source_matches:
        for script_path in sorted((cfg.data_root / "scripts" / mid).glob("round_*.json")):
            try:
                script = RoundScript.model_validate_json(script_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - one bad script must not kill chat
                logger.warning("Skipping unreadable round script %s: %s", script_path, exc)
                continue
            script = _rekey(script, cluster_keys, team_key)
            scripts[f"{script.match_id}:{script.round_num}"] = script

    if match_id is not None:
        # Single-match scope: every artifact is re-mined from just that match so
        # tendencies, gaps and economy reads describe THIS game only.
        teambook = build_teambook(list(scripts.values()), team_key)

    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    card: MapCard | None = None
    if card_path.exists():
        try:
            card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
            if isinstance(card_data, dict):
                card = MapCard(**card_data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load map card from %s: %s", card_path, exc)
            card = None

    if card is None:
        vpk_path = _find_vpk_path(map_name, cfg)
        vrf_cli = _find_vrf_cli()
        if vpk_path and vrf_cli:
            ticks_df: pl.DataFrame | None = None
            rounds_df: pl.DataFrame | None = None
            for mid in teambook.generated_from:
                tp = cfg.data_root / "lake" / mid / "ticks.parquet"
                rp = cfg.data_root / "lake" / mid / "rounds.parquet"
                if tp.exists() and rp.exists():
                    try:
                        ticks_df = pl.read_parquet(tp)
                        rounds_df = pl.read_parquet(rp)
                        break
                    except Exception:  # noqa: BLE001, S112
                        continue
            if ticks_df is None or rounds_df is None:
                lake_root = cfg.data_root / "lake"
                for rp in lake_root.glob("*/rounds.parquet"):
                    tp = rp.parent / "ticks.parquet"
                    if tp.exists():
                        try:
                            tdf = pl.read_parquet(tp)
                            rdf = pl.read_parquet(rp)
                            if "map_name" in rdf.columns and map_name in rdf["map_name"].to_list():
                                ticks_df = tdf
                                rounds_df = rdf
                                break
                        except Exception:  # noqa: BLE001, S112
                            continue
            if ticks_df is not None and rounds_df is not None:
                try:
                    assets_dir = cfg.data_root / "tmp_assets" / map_name
                    assets = extract_map_assets(vpk_path, vrf_cli, assets_dir)
                    places = unique_places(parse_places(assets.vents))
                    overlay_path = get_default_overlay_path(map_name)
                    lexicon_card = build_lexicon(
                        map_name, places, overlay_path if overlay_path.exists() else None
                    )
                    graph = zone_graph(ticks_df)
                    card = compile_card(
                        lexicon=lexicon_card,
                        graph=graph,
                        ticks=ticks_df,
                        rounds=rounds_df,
                        map_name=map_name,
                        patch_version="unknown",
                    )
                    card_path.parent.mkdir(parents=True, exist_ok=True)
                    card_path.write_text(card.to_yaml(), encoding="utf-8")
                except Exception as exc:
                    logger.exception("Mapcard compilation failed: %s", exc)  # noqa: TRY401
                    card = None

    if card is None:
        card = MapCard(
            map=map_name,
            game_version="unknown",
            nav_source="none",
            frame={},
            zones={},
            topology={},
            rotates=[],
            timings={},
            objectives={},
            sightlines=[],
            checksum="degraded",
        )

    overlay_path = get_default_overlay_path(map_name)
    if card.zones:
        lexicon = build_lexicon(
            map_name, list(card.zones.keys()), overlay_path if overlay_path.exists() else None
        )
    else:
        places_set: set[str] = set()
        for s in scripts.values():
            for b in s.beats:
                for _, z in b.t_form.zones:
                    if z:
                        places_set.add(z)
                for _, z in b.ct_form.zones:
                    if z:
                        places_set.add(z)
            for k in s.kills:
                if k.zone:
                    places_set.add(k.zone)
            if s.first_contact and s.first_contact.zone:
                places_set.add(s.first_contact.zone)
            for u in s.utility:
                if u.from_zone:
                    places_set.add(u.from_zone)
                if u.to_zone:
                    places_set.add(u.to_zone)
            if s.plant and s.plant.site:
                places_set.add(s.plant.site)
        for mid in teambook.generated_from:
            tp = cfg.data_root / "lake" / mid / "ticks.parquet"
            if tp.exists():
                try:
                    tdf = pl.read_parquet(tp, columns=["last_place_name"])
                    places_set.update(
                        str(p)
                        for p in tdf["last_place_name"].drop_nulls().unique().to_list()
                        if str(p).strip()
                    )
                except Exception:  # noqa: BLE001, S110
                    pass

        overlay_zones: list[str] = []
        valid_overlay: Path | None = None
        if overlay_path.exists():
            try:
                overlay_data = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
                overlay_zones = list((overlay_data.get("zones") or {}).keys())
                valid_overlay = overlay_path
            except Exception:  # noqa: BLE001
                valid_overlay = None

        all_places = sorted(places_set | set(overlay_zones))
        if not all_places:
            all_places = ["Default"]

        try:
            lexicon = build_lexicon(map_name, all_places, valid_overlay)
        except Exception:  # noqa: BLE001
            lexicon = build_lexicon(map_name, all_places, None)

    script_list = list(scripts.values())
    renamer = load_renamer(cfg.data_root, map_name)
    ctx = SessionContext(
        team_key=team_key,
        map_name=map_name,
        teambook=teambook,
        lexicon=lexicon,
        card=card,
        con=_lake_connection(cfg),
        scripts=scripts,
        utility_book=build_utility_book(script_list, team_key),
        gap_report=build_gap_report(script_list, team_key),
        econ_policy=build_econ_policy(script_list, team_key),
        renamer=renamer if renamer else None,
    )
    return ChatSession(
        session_id=session_id,
        team_key=team_key,
        map_name=map_name,
        system=renamer.rename_text(_system_prompt(card, teambook)),
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
    restored_match = meta.get("match_id")
    session = _build_session(
        cfg,
        sid,
        str(meta.get("team_key")),
        str(meta.get("map_name")),
        str(restored_match) if restored_match else None,
    )
    session.history = turns
    store[sid] = session
    return session


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    req: CreateSessionRequest, request: Request, cfg: ConfigDep
) -> CreateSessionResponse:
    session_id = uuid.uuid4().hex[:12]
    session = _build_session(cfg, session_id, req.team_key, req.map_name, req.match_id)
    _sessions(request)[session_id] = session
    _append_record(
        _transcript_path(cfg, session_id),
        {
            "type": "meta",
            "session_id": session_id,
            "team_key": session.ctx.team_key,
            "map_name": req.map_name,
            "match_id": req.match_id,
        },
    )
    return CreateSessionResponse(session_id=session_id)


@router.post("/sessions/{sid}/messages", response_model=AgentReply)
def post_message(sid: str, req: MessageRequest, request: Request, cfg: ConfigDep) -> AgentReply:
    session = _get_session(request, cfg, sid)

    # Resolve the client before recording the turn so a key-less install cannot
    # leave a dangling user message in the transcript.
    factory = getattr(request.app.state, "client_factory", None)
    if factory is None and request.query_params.get("mock") == "1":
        from counterstrat.llm.base import ToolCall

        class ScriptedMockClient:
            def __init__(self) -> None:
                self.turns = [
                    ChatTurn(
                        role="assistant",
                        tool_calls=[
                            ToolCall(id="c1", name="get_tendencies", arguments={"side": "T"})
                        ],
                    ),
                    ChatTurn(
                        role="assistant",
                        text="On full buy rounds (n=12), they default toward `BombsiteA` with 67% frequency, executing late via `A_Main`.",
                    ),
                ]

            def chat(
                self, *, system: Any, turns: Any, tools: Any, max_tokens: int = 4096
            ) -> tuple[ChatTurn, Any]:
                from counterstrat.llm.base import LLMResult

                turn = (
                    self.turns.pop(0)
                    if self.turns
                    else ChatTurn(
                        role="assistant",
                        text="Checked data via tools. Evidence points to default setup on `BombsiteA` (n=10).",
                    )
                )
                return turn, LLMResult(
                    text=turn.text or "",
                    input_tokens=150,
                    output_tokens=35,
                    model="mock-client",
                    provider="mock",
                )

        client: Any = ScriptedMockClient()
    else:
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
