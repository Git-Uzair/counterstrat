"""Tests for the analyst chat API: session creation, message loop, transcript reload."""

import json
from pathlib import Path

import pytest
import yaml
from conftest import (
    SYNTHETIC_TEAM,
    ScriptedToolClient,
    build_synthetic_card,
    build_synthetic_scripts,
)
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.llm.base import ChatTurn, ToolCall
from counterstrat.mining.tendencies import build_teambook
from counterstrat.web.app import create_app

MAP = "de_anubis"


@pytest.fixture
def chat_cfg(tmp_path: Path) -> AppConfig:
    """A data root holding one team's teambook, the map card, and its round scripts."""
    cfg = AppConfig(data_root=tmp_path)
    scripts = build_synthetic_scripts()
    card = build_synthetic_card()

    card_path = cfg.data_root / "mapcards" / MAP / "card.yaml"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card.to_yaml(), encoding="utf-8")

    tb_path = cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(
        build_teambook(scripts, SYNTHETIC_TEAM).model_dump_json(indent=2), encoding="utf-8"
    )

    for s in scripts:
        s_path = cfg.data_root / "scripts" / s.match_id / f"round_{s.round_num}.json"
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(s.to_json(), encoding="utf-8")

    return cfg


def _scripted() -> ScriptedToolClient:
    return ScriptedToolClient(
        [
            ChatTurn(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="get_tendencies", arguments={"side": "CT"})],
            ),
            ChatTurn(role="assistant", text="They hold `BombsiteB` (n=3, evidence m1:13)."),
            ChatTurn(
                role="assistant",
                tool_calls=[ToolCall(id="c2", name="get_role_cards", arguments={})],
            ),
            ChatTurn(role="assistant", text="p2 lurks `Water` (evidence m1:3)."),
        ]
    )


def _client(cfg: AppConfig, scripted: ScriptedToolClient | None = None) -> TestClient:
    return TestClient(create_app(cfg, client_factory=(lambda _cfg: scripted) if scripted else None))


def test_web_chat_flow(chat_cfg: AppConfig):
    scripted = _scripted()
    client = _client(chat_cfg, scripted)

    created = client.post("/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP})
    assert created.status_code == 200
    sid = created.json()["session_id"]
    assert sid

    first = client.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"text": "What is their default CT setup on full buys?"},
    )
    assert first.status_code == 200
    reply = first.json()
    assert "BombsiteB" in reply["text"]
    assert reply["warnings"] == []
    assert reply["tool_trace"][0]["name"] == "get_tendencies"
    assert reply["usage"]["input_tokens"] == 200

    # Transcript is on disk: metadata line plus both turns.
    transcript = chat_cfg.data_root / "chats" / f"{sid}.jsonl"
    assert transcript.exists()
    records = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines()]
    assert records[0]["type"] == "meta" and records[0]["team_key"] == SYNTHETIC_TEAM
    assert [r["role"] for r in records[1:]] == ["user", "assistant"]

    # Second message sees the first exchange as history.
    second = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "and who lurks?"})
    assert second.status_code == 200
    assert "p2 lurks" in second.json()["text"]
    sent_turns = scripted.calls[-1]["turns"]
    assert (sent_turns[0].text or "").startswith("What is their default")
    assert any("BombsiteB" in (t.text or "") for t in sent_turns)

    # GET transcript for a page reload.
    got = client.get(f"/api/chat/sessions/{sid}")
    assert got.status_code == 200
    body = got.json()
    assert body["team_key"] == SYNTHETIC_TEAM and body["map_name"] == MAP
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "assistant"]

    assert client.get("/api/chat/sessions/nope").status_code == 404
    assert client.post("/api/chat/sessions/nope/messages", json={"text": "hi"}).status_code == 404


def test_chat_session_survives_process_restart(chat_cfg: AppConfig):
    client = _client(chat_cfg, _scripted())
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "first question"})

    # A brand-new app (empty in-memory store) replays the jsonl into history.
    fresh_scripted = _scripted()
    fresh = _client(chat_cfg, fresh_scripted)
    reloaded = fresh.get(f"/api/chat/sessions/{sid}")
    assert reloaded.status_code == 200
    assert [m["role"] for m in reloaded.json()["messages"]] == ["user", "assistant"]

    resumed = fresh.post(f"/api/chat/sessions/{sid}/messages", json={"text": "follow up"})
    assert resumed.status_code == 200
    replayed = fresh_scripted.calls[0]["turns"]
    assert (replayed[0].text or "") == "first question"
    assert replayed[-1].text == "follow up"


def test_create_session_unknown_team_404(chat_cfg: AppConfig):
    client = _client(chat_cfg, _scripted())
    r = client.post("/api/chat/sessions", json={"team_key": "ghost", "map_name": MAP})
    assert r.status_code == 404
    assert "TeamBook" in r.json()["detail"]


def test_create_session_missing_card_fallback_degraded(tmp_path: Path):
    cfg = AppConfig(data_root=tmp_path)
    tb_path = cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(
        build_teambook(build_synthetic_scripts(), SYNTHETIC_TEAM).model_dump_json(),
        encoding="utf-8",
    )
    client = _client(cfg, _scripted())
    r = client.post("/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP})
    assert r.status_code == 200
    sid = r.json()["session_id"]
    assert sid

    # Message exchange works in degraded session
    msg = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "default setup?"})
    assert msg.status_code == 200


def test_chat_message_503_without_api_key(chat_cfg: AppConfig):
    client = _client(chat_cfg)  # no client_factory -> real make_client, no keys configured
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    r = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "hi"})
    assert r.status_code == 503
    assert "API key" in r.json()["detail"]


def test_system_prompt_carries_card_teambook_and_tool_rules(chat_cfg: AppConfig):
    scripted = _scripted()
    client = _client(chat_cfg, scripted)
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "q"})

    system = scripted.calls[0]["system"]
    card_zones = yaml.safe_load(
        (chat_cfg.data_root / "mapcards" / MAP / "card.yaml").read_text(encoding="utf-8")
    )["zones"]
    for zone in card_zones:
        assert zone in system
    assert f"TeamBook: {SYNTHETIC_TEAM}" in system
    assert "get_tendencies" in system and "low_n" in system
