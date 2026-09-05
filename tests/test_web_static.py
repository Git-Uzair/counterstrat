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
    assert "dossier" not in resp.text.lower()  # the dossier feature is gone
    # Multi-demo ingestion: multi-select input + the per-file queue panel.
    assert "multiple" in resp.text
    assert "ingest-queue" in resp.text
    # The radar tab is retired; the callout editor keeps its own tab.
    assert 'id="tab-radar"' not in resp.text
    assert "radar.js" not in resp.text
    assert 'id="tab-callouts"' in resp.text


def test_static_app_js(client_app: TestClient) -> None:
    resp = client_app.get("/static/app.js")
    assert resp.status_code == 200
    for endpoint in ["/api/demos", "/api/jobs", "/api/teams", "/api/chat/sessions"]:
        assert endpoint in resp.text
    # The deterministic First Look panel is retired: no brief fetch, no renderer.
    assert "/brief" not in resp.text
    assert "renderScoutBrief" not in resp.text
    # Scoped AI First Read: cards in the pinned panel, per-selection scope,
    # generated instantly on Analyze, transcript replay per scope session.
    assert "/insights" in resp.text
    assert "renderFirstRead" in resp.text
    assert "first-read-card" in resp.text
    assert "replayTranscript" in resp.text
    assert 'params.set("matches"' in resp.text
    assert "clearFirstRead" in resp.text
    # Picking a scope (match title click) never auto-generates: it probes the
    # cache only, and the LLM call waits for Analyze or the panel's
    # Generate/Regenerate button.
    assert '"probe"' in resp.text
    assert "No AI First Read for this selection yet" in resp.text
    # Multi-upload queue: every file gets its own polled row; duplicates are
    # terminal; finished rows dismiss themselves.
    assert "uploadDemoFiles" in resp.text
    assert "duplicate" in resp.text
    assert "dismissQueueRow" in resp.text
    # Team-grouped catalog tree with per-match subset selection.
    assert "team-group" in resp.text
    assert "match-row" in resp.text and "match-check" in resp.text
    assert "match_ids" in resp.text
    assert "opponent_name" in resp.text and "score_won" in resp.text
    # Deletion: match rows, map cards, and whole teams - one DELETE call each.
    assert "deleteMatches" in resp.text
    assert "team-delete" in resp.text
    assert 'method: "DELETE"' in resp.text
    assert "/api/demos?matches=" in resp.text
    # The radar viewer handoff is gone with the tab.
    assert "CounterStratRadar" not in resp.text


def test_static_style_has_catalog_tree(client_app: TestClient) -> None:
    css = client_app.get("/static/style.css").text
    for cls in [".team-group", ".match-row", ".queue-item", ".ingest-queue", ".match-score"]:
        assert cls in css, f"missing {cls}"


def test_callouts_view_static(client_app: TestClient) -> None:
    """Feature B: the callout editor view and its script."""
    html = client_app.get("/").text
    for token in [
        'id="tab-callouts"',
        'id="callouts-view"',
        'id="callouts-map"',
        'id="callouts-table"',
        'id="callouts-level-toggle"',
        'id="callouts-reset-all"',
        'id="callouts-add"',
        "callouts.js",
    ]:
        assert token in html, f"missing {token}"
    js = client_app.get("/static/callouts.js").text
    assert "/api/maps" in js and "/aliases" in js
    assert "callout-label" in js
    assert "setLevel" in js  # nuke upper/lower switching
    # Reset controls: per-zone clear buttons plus a confirmed reset-all that
    # removes BOTH renames and placed zones (regression: it once only cleared
    # aliases while claiming to remove all custom callouts).
    assert "callout-reset-btn" in js
    assert "resetAll" in js and "confirm(" in js
    assert 'putZones([], "Removing all custom callouts")' in js
    # Custom zone placement: drag-a-rectangle, rebuild polling, styled labels.
    assert "is-placing" in js
    assert "/zones" in js and "pollJob" in js
    assert "is-user-zone" in js
    # Rect zones: drag ghost, corner payloads, and rendered footprints.
    assert "zone-ghost" in js
    assert "zone-footprint" in js
    assert "u2" in js and "v2" in js
    assert "radius" not in js  # sphere zones are gone
    # Zoom/pan for precision placement.
    assert "callouts-zoom" in js
    assert "onWheel" in js and "resetView" in js
    # Game-zone occupancy overlay (Areas toggle).
    assert "showAreas" in js and "zone-area" in js
    css = client_app.get("/static/style.css").text
    assert ".zone-footprint" in css and ".zone-ghost" in css
    assert ".callouts-zoom" in css
    assert ".zone-area" in css


def test_static_style_has_first_read_panel(client_app: TestClient) -> None:
    resp = client_app.get("/static/style.css")
    assert resp.status_code == 200
    assert ".first-read-panel" in resp.text
    assert ".first-read-card" in resp.text
    html = client_app.get("/").text
    assert 'id="first-read-panel"' in html


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

    # 2. Unknown team -> 404 regardless; so does a match outside the teambook.
    assert chat_client.get("/api/teams/ghost/de_anubis/insights").status_code == 404
    assert chat_client.get(f"{url}?matches=ghost_match").status_code == 404

    # 3. Mock generation (no API key configured) writes the scoped cache.
    r = chat_client.get(f"{url}?generate=1&mock=1")
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("## 1. Offline mock read")
    assert body["generated_from"] == ["m1"]
    scope = body["scope"]
    assert body["match_ids"] == ["m1"]

    # 4. The cache now serves without the generate flag, keyed by scope.
    r = chat_client.get(url)
    assert r.status_code == 200
    assert r.json()["model"] == "mock" and r.json()["scope"] == scope
    # Explicitly selecting the same matches is the same scope.
    assert chat_client.get(f"{url}?matches=m1").json()["scope"] == scope

    # 5. The cache file lives under the scope hash.
    from pathlib import Path as _Path

    cache = _Path(str(chat_client.app.state.cfg.data_root)) / (
        f"teambooks/{SYNTHETIC_TEAM}/de_anubis/insights/{scope}.json"
    )
    assert cache.exists()

    # 6. No key and no mock -> 503 when generation is required.
    cache.unlink()
    assert chat_client.get(f"{url}?generate=1").status_code == 503


def test_insights_force_regenerates_over_cache(chat_client: TestClient) -> None:
    import json as _json

    url = f"/api/teams/{SYNTHETIC_TEAM}/de_anubis/insights"
    first = chat_client.get(f"{url}?generate=1&mock=1")
    assert first.status_code == 200
    scope = first.json()["scope"]
    cache = Path(str(chat_client.app.state.cfg.data_root)) / (
        f"teambooks/{SYNTHETIC_TEAM}/de_anubis/insights/{scope}.json"
    )
    # Tamper the cache so the serving path is provable.
    blob = _json.loads(cache.read_text(encoding="utf-8"))
    blob["text"] = "## stale sentinel"
    cache.write_text(_json.dumps(blob), encoding="utf-8")

    # generate=1 still serves the cache; force=1 regenerates (and implies generate).
    assert chat_client.get(f"{url}?generate=1&mock=1").json()["text"] == "## stale sentinel"
    fresh = chat_client.get(f"{url}?force=1&mock=1")
    assert fresh.status_code == 200
    assert fresh.json()["text"].startswith("## 1. Offline mock read")
    # The cache was overwritten: a plain cache read now serves the fresh text.
    assert chat_client.get(url).json()["text"].startswith("## 1. Offline mock read")


def test_static_chat_controls(client_app: TestClient) -> None:
    """Delete-chat and First Read regenerate controls ship in the bundle."""
    html = client_app.get("/").text
    assert 'id="delete-chat-btn"' in html
    js = client_app.get("/static/app.js").text
    assert "first-read-regen" in js
    assert '"force"' in js
    assert "deleteCurrentChat" in js
    css = client_app.get("/static/style.css").text
    assert ".first-read-regen" in css
    assert ".delete-chat-btn" in css


def test_mock_chat_flow(chat_client: TestClient) -> None:
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
