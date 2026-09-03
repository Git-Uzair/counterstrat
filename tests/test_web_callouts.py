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


def _seed_match(cfg: AppConfig, map_name: str, ticks: pl.DataFrame, match_id: str = "m1") -> None:
    """Register one corpus match whose lake holds the given ticks."""
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": match_id,
                "path": f"demos/{match_id}.dem",
                "map_name": map_name,
                "patch_version": "1",
                "demo_version_guid": "g",
                "server_name": "s",
                "registered_at": "2026-09-03T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lake = cfg.data_root / "lake" / match_id
    lake.mkdir(parents=True)
    ticks.write_parquet(lake / "ticks.parquet")


def test_tick_centroids_beat_volume_origins(cfg: AppConfig, client: TestClient):
    """Played maps anchor labels where players stand, not at entity pivots."""
    _seed_vents(cfg, MAP, {"Middle": (500.0, -500.0, 0.0)})
    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [0.0, 10.0, 20.0],
                "Y": [0.0, 10.0, 20.0],
                "Z": [0.0, 0.0, 0.0],
                "last_place_name": ["Middle", "Middle", "Middle"],
                "is_alive": [True, True, True],
            }
        ),
    )

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    # Tick median (10, 10), not the volume origin (500, -500).
    assert abs(zones["Middle"]["u"] - (1034 / 2048)) < 1e-3
    assert abs(zones["Middle"]["v"] - (1014 / 2048)) < 1e-3


def test_anchor_sits_on_occupied_ground_not_ring_center(cfg: AppConfig, client: TestClient):
    """Ring/L-shaped zones: a mean (or raw per-axis median) can land where nobody
    ever stands; the anchor must snap to a real tick so the label is on-zone."""
    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [100.0, -100.0, 0.0, 10.0, 80.0],
                "Y": [0.0, 10.0, 120.0, -100.0, 80.0],
                "Z": [0.0, 0.0, 0.0, 0.0, 0.0],
                "last_place_name": ["Middle"] * 5,
                "is_alive": [True] * 5,
            }
        ),
    )

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    # Per-axis median (10, 10) is the unoccupied ring center; the closest real
    # tick is (100, 0) and that is where the label must sit.
    assert abs(zones["Middle"]["u"] - (1124 / 2048)) < 1e-3
    assert abs(zones["Middle"]["v"] - (1024 / 2048)) < 1e-3


def test_anchor_follows_zones_dominant_level(cfg: AppConfig, client: TestClient):
    """A zone straddling nuke's two levels labels the level most of its ticks
    are on, and its anchor snaps to ground on THAT level - never to a stray
    tick from the other radar image."""
    _seed_calibration(cfg, MAP, lower_max=-450.0)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                # 6 lower-level ticks vs an upper cluster whose x is the
                # whole-zone median: without the dominant-level filter the
                # anchor snaps upper and the level flips.
                "X": [0.0, 60.0, 90.0, 91.0, 320.0, 350.0, 200.0, 205.0, 210.0],
                "Y": [0.0] * 9,
                "Z": [-600.0] * 6 + [0.0] * 3,
                "last_place_name": ["Middle"] * 9,
                "is_alive": [True] * 9,
            }
        ),
    )

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    mid = zones["Middle"]
    assert mid["level"] == "lower"
    # Median of the lower ticks' x is 90.5 -> snaps to the tick at (90, 0).
    assert abs(mid["u"] - (1114 / 2048)) < 1e-3
    assert abs(mid["v"] - (1024 / 2048)) < 1e-3


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
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [0.0, 10.0, 20.0, -500.0],
                "Y": [0.0, 10.0, 20.0, 500.0],
                "Z": [0.0, 0.0, 0.0, 0.0],
                "last_place_name": ["Middle", "Middle", "Middle", "BombsiteA"],
                "is_alive": [True, True, True, True],
            }
        ),
    )

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    mid = zones["Middle"]
    # Median (10, 10) world -> u=(10+1024)/2048, v=(1024-10)/2048.
    assert abs(mid["u"] - (1034 / 2048)) < 1e-3
    assert abs(mid["v"] - (1014 / 2048)) < 1e-3
    assert zones["BombsiteA"]["u"] is not None


def _seed_empty_tables(cfg: AppConfig, match_id: str = "m1") -> None:
    # No rosters.parquet: build_team_clusters skips missing files but would
    # crash on a schemaless one. serialize_match needs rounds.round_num.
    lake = cfg.data_root / "lake" / match_id
    pl.DataFrame(
        {"match_id": pl.Series([], dtype=pl.String), "round_num": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(lake / "rounds.parquet")
    pl.DataFrame({"match_id": pl.Series([], dtype=pl.String)}).write_parquet(lake / "kills.parquet")


def test_put_zones_places_zone_rebuilds_and_anchors_at_user_point(
    cfg: AppConfig, client: TestClient
):
    """Pointing at the radar creates a zone; the rebuild bakes it into the
    lake; its label anchors exactly where the user clicked."""
    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [190.0, 200.0, 210.0, 900.0],
                "Y": [-310.0, -300.0, -290.0, 900.0],
                "Z": [0.0] * 4,
                "last_place_name": ["Middle", "Middle", "Middle", "BombsiteA"],
                "is_alive": [True] * 4,
                "steamid": [1, 1, 1, 2],
                "round_num": [1] * 4,
                "tick": [0, 4, 8, 12],
                "clock_s": [0.0, 1.0, 2.0, 3.0],
                "team_name": ["CT", "CT", "CT", "TERRORIST"],
            }
        ),
    )
    _seed_empty_tables(cfg)

    # Click at world (200, -300): u=(200+1024)/2048, v=(1024+300)/2048.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Sandbags", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["zones"][0]["name"] == "Sandbags"
    # TestClient runs the background rebuild before returning control.
    job = client.get(f"/api/jobs/{body['job_id']}").json()
    assert job["stage"] == "done", job

    ticks = pl.read_parquet(cfg.data_root / "lake" / "m1" / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == [
        "Sandbags",
        "Sandbags",
        "Sandbags",
        "BombsiteA",
    ]
    assert ticks["place_default"].to_list()[:3] == ["Middle"] * 3

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    sb = zones["Sandbags"]
    assert sb["custom"] is True and sb["radius"] == 100.0
    # Anchor = the user's click, not a tick medoid.
    assert abs(sb["u"] - 612 / 1024) < 1e-3 and abs(sb["v"] - 662 / 1024) < 1e-3
    assert zones["Middle"]["custom"] is False

    # The LLM zone map speaks it at the same point.
    from counterstrat.llm.prompts import format_zone_map
    from counterstrat.web.routes import map_zone_anchors

    zm = format_zone_map(map_zone_anchors(cfg, MAP))
    assert "`Sandbags` at (0.60, 0.65)" in zm

    # Deleting via empty collection folds the ticks back.
    r = client.put(f"/api/maps/{MAP}/zones", json={"zones": []})
    assert r.status_code == 200
    ticks = pl.read_parquet(cfg.data_root / "lake" / "m1" / "ticks.parquet")
    assert ticks["last_place_name"].to_list()[:3] == ["Middle"] * 3


def test_put_zones_validation(cfg: AppConfig, client: TestClient):
    # No radar calibration -> cannot place.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "A", "u": 0.5, "v": 0.5, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "calibration" in r.json()["detail"]

    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [200.0],
                "Y": [-300.0],
                "Z": [0.0],
                "last_place_name": ["Middle"],
                "is_alive": [True],
                "steamid": [1],
                "round_num": [1],
                "tick": [0],
                "clock_s": [0.0],
                "team_name": ["CT"],
            }
        ),
    )
    _seed_empty_tables(cfg)
    # Far from any player data -> rejected (cannot ground the zone).
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "A", "u": 0.01, "v": 0.01, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "player data" in r.json()["detail"]
    # Name collision with a game zone.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Middle", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    # Name collision with an alias value.
    client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Water": "Pond"}})
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Pond", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    # And the reverse: an alias may not take a custom zone's name.
    client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Sandbags", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    r = client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Water": "Sandbags"}})
    assert r.status_code == 400


def test_chat_session_speaks_user_callouts(cfg: AppConfig):
    """The model sees one vocabulary: user names, defaults only where unnamed -
    and the zone map hands it the editor's exact label coordinates."""
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
    # Anchors for the zone map: median tick (200, -300) -> (0.60, 0.65).
    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [190.0, 200.0, 210.0],
                "Y": [-310.0, -300.0, -290.0],
                "Z": [0.0, 0.0, 0.0],
                "last_place_name": ["Middle"] * 3,
                "is_alive": [True] * 3,
            }
        ),
        match_id="anchor_match",
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
    # The zone map speaks the user's callout at the editor's exact position.
    assert "`Mid` at (0.60, 0.65)" in system
    # Tool results reaching the model are renamed too.
    tool_turn_texts = [
        t.text for call in scripted.calls for t in call.get("turns", []) if t.role == "tool"
    ]
    assert tool_turn_texts and all("Middle" not in (t or "") for t in tool_turn_texts)
    assert any("Mid" in (t or "") for t in tool_turn_texts)
    assert isinstance(reply["text"], str)
