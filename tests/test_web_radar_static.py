from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app


@pytest.fixture
def client_app(tmp_path: Path) -> TestClient:
    return TestClient(create_app(AppConfig(data_root=tmp_path)))


def test_index_exposes_radar_view(client_app: TestClient) -> None:
    html = client_app.get("/").text
    for token in [
        "radar.js",
        'id="tab-chat"',
        'id="tab-radar"',
        'id="radar-view"',
        'id="radar-image"',
        'id="radar-canvas"',
        'id="radar-status"',
        'id="radar-side"',
        'id="radar-round"',
        'id="radar-trail-rounds"',
        'id="radar-level-toggle"',
        'id="chat-input-bar"',
    ]:
        assert token in html, f"missing {token}"
    for layer in ["heatmap", "trails", "utility", "duels", "bombs"]:
        assert f'id="layer-{layer}"' in html


def test_radar_js_targets_the_radar_endpoints(client_app: TestClient) -> None:
    js = client_app.get("/static/radar.js").text
    assert js.startswith("/**")
    assert "/api/radar/" in js
    assert "/info" in js and "/layers" in js
    assert "window.CounterStratRadar" in js
    for fn in ["drawHeatmap", "drawTrails", "drawUtility", "drawDuels", "drawBombs"]:
        assert fn in js, f"missing renderer {fn}"


def test_radar_js_v2_interactions(client_app: TestClient) -> None:
    """Task 9: zoom/pan transform, player tracing, glyphs, tooltip."""
    js = client_app.get("/static/radar.js").text
    assert "setTransform" in js
    assert "onWheel" in js and "onPointerDown" in js
    assert "PLAYER_COLORS" in js and "#e69f00" in js  # Okabe-Ito anchor
    assert 'params.set("players"' in js
    assert "drawGlyph" in js
    assert "hitTest" in js and "radar-tooltip" in js


def test_index_exposes_radar_v2_controls(client_app: TestClient) -> None:
    html = client_app.get("/").text
    for token in ['id="radar-reset-view"', 'id="radar-players"', 'id="radar-tooltip"']:
        assert token in html, f"missing {token}"
    # The glyph legend replaces the color-only utility swatches.
    for glyph_class in ["glyph-flash", "glyph-he", "glyph-decoy", "glyph-bomb"]:
        assert glyph_class in html, f"missing {glyph_class}"


def test_app_js_notifies_the_radar_viewer(client_app: TestClient) -> None:
    js = client_app.get("/static/app.js").text
    assert "CounterStratRadar" in js
    assert "onTargetSelected" in js


def test_style_css_has_radar_rules(client_app: TestClient) -> None:
    css = client_app.get("/static/style.css").text
    for rule in [".radar-view", ".radar-frame", ".radar-canvas", ".view-tab", ".radar-legend"]:
        assert rule in css, f"missing rule {rule}"
