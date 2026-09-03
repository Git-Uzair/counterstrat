from pathlib import Path

import pytest
from conftest import (
    SYNTHETIC_TEAM,
    build_synthetic_card,
    build_synthetic_scripts,
)
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.mining.tendencies import build_teambook
from counterstrat.web.app import create_app


@pytest.fixture
def client_app(tmp_path: Path) -> TestClient:
    cfg = AppConfig(data_root=tmp_path)
    app = create_app(cfg)
    return TestClient(app)


@pytest.fixture
def chat_client(tmp_path: Path) -> TestClient:
    cfg = AppConfig(data_root=tmp_path)
    scripts = build_synthetic_scripts()
    card = build_synthetic_card()

    card_path = cfg.data_root / "mapcards" / "de_anubis" / "card.yaml"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card.to_yaml(), encoding="utf-8")

    tb_path = cfg.data_root / "teambooks" / SYNTHETIC_TEAM / "de_anubis" / "teambook.json"
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(
        build_teambook(scripts, SYNTHETIC_TEAM).model_dump_json(indent=2), encoding="utf-8"
    )

    for s in scripts:
        s_path = cfg.data_root / "scripts" / s.match_id / f"round_{s.round_num}.json"
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(s.to_json(), encoding="utf-8")

    app = create_app(cfg)
    return TestClient(app)


def test_index_page(client_app: TestClient) -> None:
    resp = client_app.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "app.js" in resp.text
    assert "style.css" in resp.text
    assert "drop-zone" in resp.text
    assert "teams-list" in resp.text
    assert "messages-container" in resp.text
    assert "settings-modal" in resp.text
    assert "dossier-btn" in resp.text


def test_static_app_js(client_app: TestClient) -> None:
    resp = client_app.get("/static/app.js")
    assert resp.status_code == 200
    for endpoint in ["/api/demos", "/api/jobs", "/api/teams", "/api/chat/sessions"]:
        assert endpoint in resp.text
    # Task 8: the First Look scout brief renderer.
    assert "/brief" in resp.text
    assert "first-look" in resp.text
    # Feedback round: the AI First Read panel.
    assert "/insights" in resp.text
    assert "ai-first-read" in resp.text
    # Team clustering round: single-match drill-down.
    assert "match_id" in resp.text
    assert "demo-chip" in resp.text
    # Feature A: demo deletion from the sidebar.
    assert "deleteDemo" in resp.text
    assert 'method: "DELETE"' in resp.text


def test_callouts_view_static(client_app: TestClient) -> None:
    """Feature B: the callout editor view and its script."""
    html = client_app.get("/").text
    for token in [
        'id="tab-callouts"',
        'id="callouts-view"',
        'id="callouts-map"',
        'id="callouts-table"',
        "callouts.js",
    ]:
        assert token in html, f"missing {token}"
    js = client_app.get("/static/callouts.js").text
    assert "/api/maps" in js and "/aliases" in js
    assert "callout-label" in js


def test_static_style_has_first_look_panel(client_app: TestClient) -> None:
    resp = client_app.get("/static/style.css")
    assert resp.status_code == 200
    assert ".first-look-panel" in resp.text


def test_static_style_css(client_app: TestClient) -> None:
    resp = client_app.get("/static/style.css")
    assert resp.status_code == 200


def test_models_endpoint_provider_query(client_app: TestClient) -> None:
    resp_anthropic = client_app.get("/api/models?provider=anthropic")
    assert resp_anthropic.status_code == 200
    assert "claude-sonnet-5" in resp_anthropic.json()

    resp_gemini = client_app.get("/api/models?provider=gemini")
    assert resp_gemini.status_code == 200
    assert "gemini-2.5-flash" in resp_gemini.json()


def test_insights_endpoint_mock_flow(chat_client: TestClient) -> None:
    url = f"/api/teams/{SYNTHETIC_TEAM}/de_anubis/insights"
    # 1. Nothing cached yet and no generate flag -> 404 with guidance.
    r = chat_client.get(url)
    assert r.status_code == 404
    assert "generate=1" in r.json()["detail"]

    # 2. Unknown team -> 404 regardless.
    assert chat_client.get("/api/teams/ghost/de_anubis/insights").status_code == 404

    # 3. Mock generation (no API key configured) writes the cache.
    r = chat_client.get(f"{url}?generate=1&mock=1")
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("## 1. Offline mock read")
    assert body["generated_from"] == ["m1"]

    # 4. The cache now serves without the generate flag.
    r = chat_client.get(url)
    assert r.status_code == 200
    assert r.json()["model"] == "mock"

    # 5. generate=1 always regenerates; with no key and no mock that is a 503.
    assert chat_client.get(f"{url}?generate=1").status_code == 503

    # 6. A cache built from different demos is stale and must not be served.
    import json as _json
    from pathlib import Path as _Path

    cache = _Path(str(chat_client.app.state.cfg.data_root)) / (
        f"teambooks/{SYNTHETIC_TEAM}/de_anubis/insights.json"
    )
    stale = _json.loads(cache.read_text(encoding="utf-8"))
    stale["generated_from"] = ["old_match"]
    cache.write_text(_json.dumps(stale), encoding="utf-8")
    assert chat_client.get(url).status_code == 404


def test_mock_chat_and_report_flow(chat_client: TestClient) -> None:
    # 1. Create chat session
    created = chat_client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": "de_anubis"}
    )
    assert created.status_code == 200
    sid = created.json()["session_id"]

    # 2. Send message with ?mock=1 offline flag (no provider API keys configured)
    msg_resp = chat_client.post(
        f"/api/chat/sessions/{sid}/messages?mock=1",
        json={"text": "What are their default buy round setups?"},
    )
    assert msg_resp.status_code == 200
    data = msg_resp.json()
    assert "text" in data
    assert "tool_trace" in data
    assert len(data["tool_trace"]) >= 1
    assert data["tool_trace"][0]["name"] == "get_tendencies"

    # 3. Download dossier with ?mock=1 offline flag
    report_resp = chat_client.get(f"/api/reports/{SYNTHETIC_TEAM}/de_anubis?mock=1")
    assert report_resp.status_code == 200
    assert "Anti-Strat Dossier" in report_resp.text
