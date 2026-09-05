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


def test_delete_session_clears_transcript_and_memory(chat_cfg: AppConfig):
    client = _client(chat_cfg, _scripted())
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "hello?"})
    transcript = chat_cfg.data_root / "chats" / f"{sid}.jsonl"
    assert transcript.exists()

    deleted = client.delete(f"/api/chat/sessions/{sid}")
    assert deleted.status_code == 200
    assert deleted.json() == {"session_id": sid, "status": "deleted"}
    assert not transcript.exists()
    assert client.get(f"/api/chat/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/chat/sessions/{sid}").status_code == 404

    # Re-selecting the same scope starts a fresh conversation under the same id.
    again = client.post("/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP})
    assert again.json()["session_id"] == sid
    assert client.get(f"/api/chat/sessions/{sid}").json()["messages"] == []


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


def test_chat_session_match_scope(chat_cfg: AppConfig):
    """match_id scopes every artifact to that one game; merged mode sees all."""
    # Add a second demo's scripts and a merged teambook spanning both.
    scripts_m1 = build_synthetic_scripts()
    scripts_m2 = [s.model_copy(update={"match_id": "m2"}) for s in scripts_m1]
    for s in scripts_m2:
        p = chat_cfg.data_root / "scripts" / "m2" / f"round_{s.round_num}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s.to_json(), encoding="utf-8")
    tb_path = chat_cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.write_text(
        build_teambook(scripts_m1 + scripts_m2, SYNTHETIC_TEAM).model_dump_json(),
        encoding="utf-8",
    )

    scripted = _scripted()
    client = _client(chat_cfg, scripted)

    merged = client.post("/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP})
    assert merged.status_code == 200
    client.post(f"/api/chat/sessions/{merged.json()['session_id']}/messages", json={"text": "q"})
    # The agent loop may call the LLM more than once per message: sample the last.
    assert "Data coverage: 2 demo(s), 18 rounds" in scripted.calls[-1]["system"]

    scoped = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_id": "m2"},
    )
    assert scoped.status_code == 200
    client.post(f"/api/chat/sessions/{scoped.json()['session_id']}/messages", json={"text": "q"})
    assert "Data coverage: 1 demo(s), 9 rounds" in scripted.calls[-1]["system"]

    missing = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_id": "ghost"},
    )
    assert missing.status_code == 404


def test_chat_session_subset_scope(chat_cfg: AppConfig):
    """match_ids scopes the session to any subset of the team's matches."""
    scripts_m1 = build_synthetic_scripts()
    for extra in ("m2", "m3"):
        for s in scripts_m1:
            s2 = s.model_copy(update={"match_id": extra})
            p = chat_cfg.data_root / "scripts" / extra / f"round_{s2.round_num}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s2.to_json(), encoding="utf-8")
    all_scripts = scripts_m1 + [
        s.model_copy(update={"match_id": m}) for m in ("m2", "m3") for s in scripts_m1
    ]
    tb_path = chat_cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.write_text(
        build_teambook(all_scripts, SYNTHETIC_TEAM).model_dump_json(), encoding="utf-8"
    )

    scripted = _scripted()
    client = _client(chat_cfg, scripted)

    subset = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_ids": ["m1", "m3"]},
    )
    assert subset.status_code == 200
    sid = subset.json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "q"})
    assert "Data coverage: 2 demo(s), 18 rounds" in scripted.calls[-1]["system"]

    # The transcript meta records the subset; a fresh app restores the same scope.
    fresh_scripted = _scripted()
    fresh = _client(chat_cfg, fresh_scripted)
    resumed = fresh.post(f"/api/chat/sessions/{sid}/messages", json={"text": "again"})
    assert resumed.status_code == 200
    assert "Data coverage: 2 demo(s), 18 rounds" in fresh_scripted.calls[-1]["system"]

    # Any unknown id in the subset is a 404, naming the offender.
    missing = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_ids": ["m1", "ghost"]},
    )
    assert missing.status_code == 404
    assert "ghost" in missing.json()["detail"]


def test_chat_scope_hash_identity_and_reuse(chat_cfg: AppConfig):
    """The scope IS the session: same selection -> same id and transcript;
    different selection -> different id. The scoped First Read, when cached,
    reaches the system prompt."""
    import json as _json

    from counterstrat.web.scope import scope_hash

    scripts = build_synthetic_scripts()
    for s in scripts:
        s2 = s.model_copy(update={"match_id": "m2"})
        p = chat_cfg.data_root / "scripts" / "m2" / f"round_{s2.round_num}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s2.to_json(), encoding="utf-8")
    all_scripts = scripts + [s.model_copy(update={"match_id": "m2"}) for s in scripts]
    tb_path = chat_cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.write_text(
        build_teambook(all_scripts, SYNTHETIC_TEAM).model_dump_json(), encoding="utf-8"
    )

    scripted = _scripted()
    client = _client(chat_cfg, scripted)

    full = client.post("/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP})
    sid_full = full.json()["session_id"]
    assert sid_full == scope_hash(SYNTHETIC_TEAM, MAP, ["m1", "m2"])

    sub = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_ids": ["m1"]},
    )
    sid_sub = sub.json()["session_id"]
    assert sid_sub == scope_hash(SYNTHETIC_TEAM, MAP, ["m1"]) and sid_sub != sid_full

    # A conversation in the subset scope...
    client.post(f"/api/chat/sessions/{sid_sub}/messages", json={"text": "first question"})
    # ...resurfaces when the same selection is analyzed again.
    again = client.post(
        "/api/chat/sessions",
        json={"team_key": SYNTHETIC_TEAM, "map_name": MAP, "match_ids": ["m1"]},
    )
    assert again.json()["session_id"] == sid_sub
    transcript = client.get(f"/api/chat/sessions/{sid_sub}").json()
    texts = [m["text"] for m in transcript["messages"] if m["role"] == "user"]
    assert "first question" in texts

    # A scoped First Read cache reaches the system prompt of later messages.
    fr_dir = chat_cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "insights"
    fr_dir.mkdir(parents=True, exist_ok=True)
    (fr_dir / f"{sid_sub}.json").write_text(
        _json.dumps({"text": "## T Pistol\nThey rush `BombsiteB` every pistol."}),
        encoding="utf-8",
    )
    client.post(f"/api/chat/sessions/{sid_sub}/messages", json={"text": "expand on pistols"})
    system = scripted.calls[-1]["system"]
    assert "<first_read>" in system
    assert "They rush `BombsiteB` every pistol." in system


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


def test_chat_system_prompt_is_igl_grade(chat_cfg: AppConfig):
    """Task 7: exploit-first contract, doctrine, and noise suppression."""
    scripted = _scripted()
    client = _client(chat_cfg, scripted)
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "q"})

    system = scripted.calls[0]["system"]
    # The answer contract.
    for anchor in ("Read", "Evidence", "Counter-call", "Confidence", "180 words"):
        assert anchor in system, anchor
    # The anti-strat doctrine.
    assert "trigger -> response -> punish" in system
    assert "get_playbook" in system and "get_gap_report" in system
    # Noise suppression: the fixture's n=1 CT semi_eco situational row is not
    # signal, so its level-2 key must not be rendered into the prompt...
    assert "| semi_eco | ahead | won |" not in system
    # ...while its side+buy rollup still is.
    assert "| semi_eco | any | any |" in system
    # The model must know how much data backs the profile (1 demo, 9 rounds).
    assert "Data coverage: 1 demo(s), 9 rounds" in system
