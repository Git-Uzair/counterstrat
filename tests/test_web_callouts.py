"""Callout endpoints and vocabulary-boundary integration (feature B)."""

import json
from pathlib import Path

import polars as pl
import pytest
from conftest import SYNTHETIC_TEAM, build_synthetic_card, build_synthetic_scripts
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.mining.tendencies import build_teambook
from counterstrat.web.app import create_app

MAP = "de_anubis"


@pytest.fixture(autouse=True)
def isolated_repo_root(tmp_path: Path, monkeypatch):
    """Keep the real maps/ VPKs out of these tests; each test seeds its own.

    _find_vpk_path searches the working directory too, so chdir into the tmp
    tree or the repo's real de_anubis.vpk leaks in and VRF runs mid-test.
    """
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    return fake_repo


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(data_root=tmp_path / "data")
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    card = build_synthetic_card()
    card_path = cfg.data_root / "mapcards" / MAP / "card.yaml"
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card.to_yaml(), encoding="utf-8")
    return cfg


@pytest.fixture
def client(cfg: AppConfig) -> TestClient:
    return TestClient(create_app(cfg))


def _seed_vents(cfg: AppConfig, map_name: str, places: dict[str, tuple]) -> None:
    """Pre-populate the tmp_assets cache so no VRF run is needed."""
    vents = cfg.data_root / "tmp_assets" / map_name / "maps" / map_name / "entities"
    vents.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i, (name, origin) in enumerate(places.items()):
        blocks.append(
            f"===={i}====\n"
            'classname "env_cs_place"\n'
            f'place_name "{name}"\n'
            f"origin [{origin[0]}, {origin[1]}, {origin[2]}]\n"
            f'hammeruniqueid "{i}"\n'
        )
    (vents / "default_ents.vents").write_text("\n".join(blocks), encoding="utf-8")


def _seed_calibration(cfg: AppConfig, map_name: str, lower_max: float | None = None) -> None:
    radar_dir = cfg.data_root / "radar" / map_name
    radar_dir.mkdir(parents=True, exist_ok=True)
    (radar_dir / "radar.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (radar_dir / "overview.txt").write_text("x", encoding="utf-8")
    (radar_dir / "calibration.json").write_text(
        json.dumps(
            {
                "map_name": map_name,
                "pos_x": -1024.0,
                "pos_y": 1024.0,
                "scale": 2.0,
                "image_px": 1024,
                "lower_altitude_max": lower_max,
            }
        ),
        encoding="utf-8",
    )


def test_list_maps_unions_cards_and_vpks(client: TestClient, isolated_repo_root: Path):
    vpk_dir = isolated_repo_root / "maps" / "de_mirage"
    vpk_dir.mkdir(parents=True)
    (vpk_dir / "de_mirage.vpk").write_bytes(b"vpk")
    # A directory without a vpk must not appear.
    (isolated_repo_root / "maps" / "junk").mkdir()
    assert client.get("/api/maps").json() == [MAP, "de_mirage"]


def test_callouts_for_vpk_only_map(cfg: AppConfig, client: TestClient):
    """A map with no ingested demo is editable from its VPK place volumes."""
    _seed_vents(cfg, "de_mirage", {"AMain": (100.0, 100.0, 0.0), "BSite": (-200.0, 50.0, 0.0)})
    _seed_calibration(cfg, "de_mirage")
    body = client.get("/api/maps/de_mirage/callouts").json()
    zones = {z["name"]: z for z in body["zones"]}
    assert set(zones) == {"AMain", "BSite"}
    assert zones["AMain"]["u"] is not None and zones["AMain"]["level"] == "default"
    assert body["levels"] == ["default"]
    # Aliases save against the VPK vocabulary too.
    r = client.put("/api/maps/de_mirage/aliases", json={"aliases": {"AMain": "A Ramp"}})
    assert r.status_code == 200 and r.json()["aliases"] == {"AMain": "A Ramp"}


def test_callouts_levels_split_upper_and_lower(cfg: AppConfig, client: TestClient):
    """Nuke-style maps: zones classify to the level their volumes sit on."""
    _seed_vents(
        cfg,
        "de_nuke2",
        {"BombsiteA": (0.0, 0.0, 0.0), "BombsiteB": (10.0, 10.0, -600.0)},
    )
    _seed_calibration(cfg, "de_nuke2", lower_max=-450.0)
    body = client.get("/api/maps/de_nuke2/callouts").json()
    assert body["levels"] == ["default", "lower"]
    zones = {z["name"]: z for z in body["zones"]}
    assert zones["BombsiteA"]["level"] == "default"
    assert zones["BombsiteB"]["level"] == "lower"


def test_get_callouts_without_radar_or_lake(client: TestClient):
    r = client.get(f"/api/maps/{MAP}/callouts")
    assert r.status_code == 200
    body = r.json()
    zones = {z["name"]: z for z in body["zones"]}
    assert "Middle" in zones
    assert zones["Middle"]["alias"] is None
    assert zones["Middle"]["u"] is None  # no radar cache -> table-only editor
    assert client.get("/api/maps/ghost_map/callouts").status_code == 404


def test_put_aliases_roundtrip_and_validation(client: TestClient):
    r = client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Middle": "Mid"}})
    assert r.status_code == 200
    assert r.json()["aliases"] == {"Middle": "Mid"}

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    assert zones["Middle"]["alias"] == "Mid"

    # Removal via empty value.
    r = client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Middle": ""}})
    assert r.json()["aliases"] == {}

    assert (
        client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Ghost": "X"}}).status_code == 400
    )
    assert (
        client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Middle": "Water"}}).status_code
        == 400
    )


def test_callout_positions_from_lake(cfg: AppConfig, client: TestClient):
    # Radar calibration cache + one lake match with ticks in two zones.
    radar_dir = cfg.data_root / "radar" / MAP
    radar_dir.mkdir(parents=True)
    (radar_dir / "radar.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (radar_dir / "overview.txt").write_text("x", encoding="utf-8")
    (radar_dir / "calibration.json").write_text(
        json.dumps(
            {
                "map_name": MAP,
                "pos_x": -1024.0,
                "pos_y": 1024.0,
                "scale": 2.0,
                "image_px": 1024,
                "lower_altitude_max": None,
            }
        ),
        encoding="utf-8",
    )
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": "m1",
                "path": "demos/m1.dem",
                "map_name": MAP,
                "patch_version": "1",
                "demo_version_guid": "g",
                "server_name": "s",
                "registered_at": "2026-09-03T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lake = cfg.data_root / "lake" / "m1"
    lake.mkdir(parents=True)
    pl.DataFrame(
        {
            "X": [0.0, 10.0, -500.0],
            "Y": [0.0, 10.0, 500.0],
            "Z": [0.0, 0.0, 0.0],
            "last_place_name": ["Middle", "Middle", "BombsiteA"],
            "is_alive": [True, True, True],
        }
    ).write_parquet(lake / "ticks.parquet")

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    mid = zones["Middle"]
    # Mean (5, 5) world -> u=(5+1024)/2048, v=(1024-5)/2048.
    assert abs(mid["u"] - (1029 / 2048)) < 1e-3
    assert abs(mid["v"] - (1019 / 2048)) < 1e-3
    assert zones["BombsiteA"]["u"] is not None


def test_chat_session_speaks_user_callouts(cfg: AppConfig):
    """The model sees one vocabulary: user names, defaults only where unnamed."""
    from conftest import ScriptedToolClient

    from counterstrat.llm.base import ChatTurn, ToolCall

    scripts = build_synthetic_scripts()
    tb_path = cfg.data_root / "teambooks" / SYNTHETIC_TEAM / MAP / "teambook.json"
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(build_teambook(scripts, SYNTHETIC_TEAM).model_dump_json(), encoding="utf-8")
    for s in scripts:
        p = cfg.data_root / "scripts" / s.match_id / f"round_{s.round_num}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s.to_json(), encoding="utf-8")
    (cfg.data_root / "mapcards" / MAP / "aliases.json").write_text(
        json.dumps({"Middle": "Mid"}), encoding="utf-8"
    )

    scripted = ScriptedToolClient(
        turns=[
            ChatTurn(
                role="assistant",
                tool_calls=[ToolCall(id="t1", name="get_tendencies", arguments={"side": "T"})],
            )
        ]
    )
    app = create_app(cfg, client_factory=lambda _cfg: scripted)
    client = TestClient(app)
    sid = client.post(
        "/api/chat/sessions", json={"team_key": SYNTHETIC_TEAM, "map_name": MAP}
    ).json()["session_id"]
    reply = client.post(f"/api/chat/sessions/{sid}/messages", json={"text": "q"}).json()

    system = scripted.calls[0]["system"]
    assert "Mid" in system
    assert "Middle" not in system  # single vocabulary: the canonical never appears
    # Tool results reaching the model are renamed too.
    tool_turn_texts = [
        t.text for call in scripted.calls for t in call.get("turns", []) if t.role == "tool"
    ]
    assert tool_turn_texts and all("Middle" not in (t or "") for t in tool_turn_texts)
    assert any("Mid" in (t or "") for t in tool_turn_texts)
    assert isinstance(reply["text"], str)
