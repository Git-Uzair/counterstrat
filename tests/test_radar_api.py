import json
from pathlib import Path

import pytest
from conftest import build_radar_lake
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app

# 1x1 transparent PNG - enough to prove the bytes are served untouched.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001aa5c9c2d0000000049454e44ae4260"
    "82"
)
MINIMAL_OVERVIEW = '"de_anubis"\n{\n\t"pos_x" "-1024"\n\t"pos_y" "1024"\n\t"scale" "2"\n}\n'


def _seed_radar_cache(data_root: Path, map_name: str = "de_anubis", *, lower: bool = False) -> None:
    """Pre-populates the radar cache so no CS2 install is needed in tests."""
    cache = data_root / "radar" / map_name
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "radar.png").write_bytes(PNG_BYTES)
    (cache / "overview.txt").write_text(MINIMAL_OVERVIEW, encoding="utf-8")
    (cache / "calibration.json").write_text(
        json.dumps(
            {
                "map_name": map_name,
                "pos_x": -1024.0,
                "pos_y": 1024.0,
                "scale": 2.0,
                "image_px": 1024,
                "lower_altitude_max": 100.0 if lower else None,
            }
        ),
        encoding="utf-8",
    )
    if lower:
        (cache / "radar_lower.png").write_bytes(PNG_BYTES)


def _seed_corpus(data_root: Path, map_name: str = "de_anubis", match_id: str = "m1") -> None:
    from counterstrat.corpus import DemoRecord

    rec = DemoRecord(
        match_id=match_id,
        path="/fake/m1.dem",
        map_name=map_name,
        patch_version="14178",
        demo_version_guid="guid1",
        server_name="S",
        registered_at="2026-09-03T00:00:00Z",
    )
    (data_root / "corpus.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")


@pytest.fixture
def radar_client(tmp_path: Path) -> TestClient:
    _seed_radar_cache(tmp_path)
    _seed_corpus(tmp_path)
    build_radar_lake(tmp_path / "lake")
    # AppConfig built directly => cs2_install_path is None, so only the cache is used.
    return TestClient(create_app(AppConfig(data_root=tmp_path)))


def test_info_returns_calibration_and_levels(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_anubis/info")
    assert r.status_code == 200
    info = r.json()
    assert info["pos_x"] == -1024.0 and info["scale"] == 2.0
    assert info["image_px"] == 1024
    assert info["levels"] == ["default"]
    assert info["lower_image_url"] is None
    assert info["image_url"] == "/api/radar/de_anubis/image?level=default"


def test_image_serves_png_bytes(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_anubis/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == PNG_BYTES


def test_image_lower_level_missing_is_404(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/de_anubis/image?level=lower").status_code == 404


def test_image_lower_level_present_is_served(tmp_path: Path) -> None:
    _seed_radar_cache(tmp_path, "de_nuke", lower=True)
    client = TestClient(create_app(AppConfig(data_root=tmp_path)))
    r = client.get("/api/radar/de_nuke/image?level=lower")
    assert r.status_code == 200
    assert client.get("/api/radar/de_nuke/info").json()["levels"] == ["default", "lower"]


def test_uncached_map_without_cs2_path_is_503(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_mirage/info")
    assert r.status_code == 503
    assert "cs2" in r.json()["detail"].lower()


def test_invalid_map_name_is_400(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/DE_Bad!/info").status_code == 400


def test_layers_full_payload(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&stride=1&trail_rounds=2")
    assert r.status_code == 200
    body = r.json()
    assert body["team_key"] == "teamA"
    assert body["map_name"] == "de_anubis"
    assert body["image_px"] == 1024
    assert [(x["round_num"], x["side"]) for x in body["rounds"]] == [(1, "CT"), (2, "T")]
    layers = body["layers"]
    assert set(layers) == {"heatmap", "trails", "utility", "duels", "bombs"}
    assert layers["heatmap"]["samples"] == 16
    assert len(layers["trails"]) == 4
    assert [u["kind"] for u in layers["utility"]] == ["smoke", "molotov"]
    assert len(layers["duels"]) == 3
    assert [b["event"] for b in layers["bombs"]] == ["defuse", "plant"]
    # steamids must be strings: JS cannot hold a 17-digit steamid exactly.
    assert isinstance(layers["trails"][0]["steamid"], str)


def test_layers_side_and_round_filters(radar_client: TestClient) -> None:
    ct = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&side=CT").json()
    assert [x["round_num"] for x in ct["rounds"]] == [1]
    assert ct["layers"]["heatmap"]["samples"] == 8

    r2 = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&rounds=2").json()
    assert [x["round_num"] for x in r2["rounds"]] == [2]


def test_layers_rejects_bad_rounds_and_bad_side(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/teamA/de_anubis/layers?rounds=abc").status_code == 400
    assert radar_client.get("/api/radar/teamA/de_anubis/layers?side=X").status_code == 422


def test_layers_matches_filter(radar_client: TestClient) -> None:
    """matches=<id> scopes the whole payload to one demo; unknown id -> 404."""
    full = radar_client.get("/api/radar/teamA/de_anubis/layers").json()
    scoped = radar_client.get("/api/radar/teamA/de_anubis/layers?matches=m1").json()
    assert scoped["rounds"] == full["rounds"]  # fixture has exactly one match
    assert radar_client.get("/api/radar/teamA/de_anubis/layers?matches=ghost").status_code == 404


def test_layers_unknown_map_is_404(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/teamA/de_dust2/layers")
    assert r.status_code == 404


def test_layers_unknown_team_returns_empty_rounds(radar_client: TestClient) -> None:
    body = radar_client.get("/api/radar/ghost/de_anubis/layers").json()
    assert body["rounds"] == []
    assert body["layers"]["trails"] == []
