import gzip
import json
import time
from pathlib import Path

import polars as pl
import pytest
import zstandard
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app


@pytest.fixture
def test_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(data_root=tmp_path)


@pytest.fixture
def client_app(test_cfg: AppConfig) -> TestClient:
    app = create_app(test_cfg)
    return TestClient(app)


def _poll_until_done(
    client: TestClient, job_id: str, timeout_s: float = 180, interval: float = 0.1
) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        state = r.json()
        if state["stage"] in ("done", "error"):
            return state
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")


def test_upload_rejects_garbage(client_app: TestClient):
    r = client_app.post("/api/demos", files={"demo": ("x.dem", b"NOTADEMO" * 4)})
    assert r.status_code == 400


def test_job_unknown_404(client_app: TestClient):
    assert client_app.get("/api/jobs/nope").status_code == 404


def test_teams_catalog_merges_stand_in_lineups(client_app: TestClient, test_cfg: AppConfig):
    """One catalog entry per team cluster with per-map demo lists."""
    from test_teams_clusters import SQUAD_A, SQUAD_B, SQUAD_C, SQUAD_D, _write_match

    _write_match(
        test_cfg.data_root, "m_one", "de_ancient", [(SQUAD_A, "team_7yrant"), (SQUAD_C, "opp1")]
    )
    _write_match(
        test_cfg.data_root, "m_two", "de_ancient", [(SQUAD_B, "team_7yrant"), (SQUAD_D, "opp2")]
    )

    teams = client_app.get("/api/teams").json()
    assert len(teams) == 3  # 7yrant merged + two opponents
    seven = next(t for t in teams if t["names"] == ["team_7yrant"])
    assert seven["demos"] == 2 and seven["rounds"] == 8
    ancient = seven["map_stats"]["de_ancient"]
    assert ancient["demos"] == 2 and ancient["rounds"] == 8
    assert [m["match_id"] for m in ancient["matches"]] == ["m_one", "m_two"]
    assert all(m["rounds"] == 4 for m in ancient["matches"])


def test_teams_catalog_matches_carry_opponent_score_and_date(
    client_app: TestClient, test_cfg: AppConfig
):
    """Match entries answer who-vs-who at first look: opponent, score, added."""
    from test_teams_clusters import SQUAD_A, SQUAD_C, _lineup_key, _write_match

    key_a, key_c = _lineup_key(SQUAD_A), _lineup_key(SQUAD_C)
    _write_match(
        test_cfg.data_root, "m_one", "de_ancient", [(SQUAD_A, "alpha"), (SQUAD_C, "bravo")]
    )
    # alpha played CT and won 3 of 4 rounds (winner values are lowercase in the lake).
    pl.DataFrame(
        {
            "round_num": [1, 2, 3, 4],
            "winner": ["ct", "ct", "t", "ct"],
            "t_team_key": [key_c] * 4,
            "ct_team_key": [key_a] * 4,
        }
    ).write_parquet(test_cfg.data_root / "lake" / "m_one" / "rounds.parquet")
    # A second match with NO rounds.parquet: scores must be None, not a crash.
    _write_match(
        test_cfg.data_root, "m_two", "de_ancient", [(SQUAD_A, "alpha"), (SQUAD_C, "bravo")]
    )

    teams = client_app.get("/api/teams").json()
    alpha = next(t for t in teams if t["names"] == ["alpha"])
    matches = {m["match_id"]: m for m in alpha["map_stats"]["de_ancient"]["matches"]}

    m1 = matches["m_one"]
    assert m1["opponent_name"] == "bravo"
    assert m1["opponent_id"] == key_c
    assert m1["score_won"] == 3 and m1["score_lost"] == 1
    assert m1["added_at"] == "2026-09-03T00:00:00+00:00"

    m2 = matches["m_two"]
    assert m2["score_won"] is None and m2["score_lost"] is None
    assert m2["opponent_name"] == "bravo"

    # The opponent's view mirrors the score.
    bravo = next(t for t in teams if t["names"] == ["bravo"])
    b1 = {m["match_id"]: m for m in bravo["map_stats"]["de_ancient"]["matches"]}["m_one"]
    assert b1["score_won"] == 1 and b1["score_lost"] == 3
    assert b1["opponent_name"] == "alpha"


def test_scout_brief_endpoint(client_app: TestClient, test_cfg: AppConfig):
    assert client_app.get("/api/teams/ghost/de_anubis/brief").status_code == 404

    brief_dir = test_cfg.data_root / "teambooks" / "abc" / "de_anubis"
    brief_dir.mkdir(parents=True)
    (brief_dir / "scout_brief.json").write_text(
        json.dumps(
            {
                "team_key": "abc",
                "map_name": "de_anubis",
                "items": [
                    {
                        "kind": "gap",
                        "text": "CT leave `BombsiteB` unheld at B+15 in 100% of rounds (n=5).",
                        "n": 5,
                        "confidence": "medium",
                        "evidence": ["m1:13"],
                    }
                ],
                "generated_from": ["m1"],
            }
        ),
        encoding="utf-8",
    )
    r = client_app.get("/api/teams/abc/de_anubis/brief")
    assert r.status_code == 200
    body = r.json()
    assert body["items"][0]["kind"] == "gap"
    assert body["generated_from"] == ["m1"]


@pytest.mark.demo
def test_upload_ingest_end_to_end(client_app: TestClient, demo_path: Path):
    with demo_path.open("rb") as f:
        r = client_app.post("/api/demos", files={"demo": (demo_path.name, f)})
    assert r.status_code == 200
    job = r.json()["job_id"]
    state = _poll_until_done(client_app, job, timeout_s=180)
    assert state["stage"] == "done" and state["map_name"] == "de_anubis"
    teams = client_app.get("/api/teams").json()
    assert len(teams) == 2 and all(t["rounds"] > 0 for t in teams)
    assert all("map_stats" in t and "de_anubis" in t["map_stats"] for t in teams)
    assert all(t["map_stats"]["de_anubis"]["rounds"] == t["rounds"] for t in teams)
    # Task 8: every ingested team gets an instant scout brief.
    for t in teams:
        brief = client_app.get(f"/api/teams/{t['team_key']}/de_anubis/brief")
        assert brief.status_code == 200
        body = brief.json()
        assert body["team_key"] == t["team_key"]
        assert isinstance(body["items"], list)
        for item in body["items"]:
            assert item["kind"] and item["text"] and item["n"] >= 1
    assert all(t["map_stats"]["de_anubis"]["demos"] == t["demos"] for t in teams)
    tb = client_app.get(f"/api/teams/{teams[0]['team_key']}/de_anubis/teambook")
    assert tb.status_code == 200


def test_settings_roundtrip_never_echoes_key(client_app: TestClient):
    r = client_app.post(
        "/api/settings",
        json={"provider": "gemini", "model": "gemini-2.5-pro", "api_key": "sec"},
    )
    assert r.status_code == 200
    got = client_app.get("/api/settings").json()
    assert got["provider"] == "gemini" and got["model"] == "gemini-2.5-pro"
    assert "sec" not in json.dumps(got)
    assert got["keys_present"]["gemini"] is True


def test_shipped_cards_cover_the_calibrated_pool():
    """Every calibrated map ships a real card: fresh clones must get zones,
    measured topology and rotates without a CS2 install or the VRF CLI."""
    from counterstrat.mapcard.cards import load_shipped_card

    pool = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
    for map_name in pool:
        card = load_shipped_card(map_name)
        assert card is not None, f"missing shipped card for {map_name}"
        assert card.map == map_name
        assert card.zones and card.topology and card.rotates
        assert card.objectives.get("sites") == ["BombsiteA", "BombsiteB"]
        # Engine vocabulary only: an operator's custom zones must never ship.
        assert all(z[0].isupper() for z in card.zones), f"non-engine zone in {map_name}"
    assert load_shipped_card("de_overpass") is None  # off the calibrated pool


def test_resolve_card_degrades_off_the_pool(tmp_path: Path):
    """No user card, no shipped card, no lake to compile from: analysis still
    gets a usable (degraded) card instead of an exception."""
    from conftest import SYNTHETIC_TEAM, build_synthetic_scripts

    from counterstrat.mining.tendencies import build_teambook
    from counterstrat.web.cards import resolve_card

    cfg = AppConfig(data_root=tmp_path)
    teambook = build_teambook(build_synthetic_scripts(), SYNTHETIC_TEAM)
    card = resolve_card(cfg, "de_overpass", teambook)
    assert card.map == "de_overpass"
    assert card.checksum == "degraded"
    assert card.zones == {}


def test_insights_bundle_survives_missing_card(tmp_path: Path):
    """Fresh clone: no compiled card, no CS2 install, no lake. The First Read
    bundle must resolve a usable card (shipped or degraded) instead of the
    404 a cloning user hit ('Map card for de_ancient not found')."""
    from conftest import SYNTHETIC_TEAM, build_synthetic_scripts

    from counterstrat.mining.tendencies import build_teambook
    from counterstrat.web.routes import _load_team_bundle

    cfg = AppConfig(data_root=tmp_path)
    scripts = build_synthetic_scripts()
    tb_path = cfg.data_root / "teambooks" / SYNTHETIC_TEAM / "de_anubis" / "teambook.json"
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(build_teambook(scripts, SYNTHETIC_TEAM).model_dump_json(), encoding="utf-8")
    for s in scripts:
        s_path = cfg.data_root / "scripts" / s.match_id / f"round_{s.round_num}.json"
        s_path.parent.mkdir(parents=True, exist_ok=True)
        s_path.write_text(s.to_json(), encoding="utf-8")

    card, teambook, loaded, lex = _load_team_bundle(cfg, SYNTHETIC_TEAM, "de_anubis")
    assert card.map == "de_anubis"
    assert teambook.team_key == SYNTHETIC_TEAM
    assert loaded, "scripts must load without a card"
    assert lex.zones, "lexicon must fall back to script/shipped vocabulary"


def test_settings_cs2_path_roundtrip(client_app: TestClient, tmp_path: Path):
    """The CS2 install folder is a first-class setting: reported, validated,
    persisted, and clearable."""
    got = client_app.get("/api/settings").json()
    assert got["cs2_install_path"] is None
    assert got["cs2_path_valid"] is False

    install = tmp_path / "cs2"
    (install / "game" / "csgo").mkdir(parents=True)
    r = client_app.post(
        "/api/settings",
        json={
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "cs2_install_path": str(install),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["cs2_install_path"] == str(install)
    assert body["cs2_path_valid"] is True

    got = client_app.get("/api/settings").json()
    assert got["cs2_install_path"] == str(install)
    assert got["cs2_path_valid"] is True

    # Explicit empty string clears the setting.
    r = client_app.post(
        "/api/settings",
        json={"provider": "gemini", "model": "gemini-2.5-pro", "cs2_install_path": ""},
    )
    assert r.status_code == 200
    assert r.json()["cs2_install_path"] is None

    # Omitting the field leaves the stored value untouched.
    client_app.post(
        "/api/settings",
        json={
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "cs2_install_path": str(install),
        },
    )
    r = client_app.post("/api/settings", json={"provider": "gemini", "model": "gemini-2.5-pro"})
    assert r.json()["cs2_install_path"] == str(install)


def test_settings_rejects_bogus_cs2_path(client_app: TestClient, tmp_path: Path):
    """A missing folder or one without game/csgo is refused with a reason."""
    r = client_app.post(
        "/api/settings",
        json={
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "cs2_install_path": str(tmp_path / "nope"),
        },
    )
    assert r.status_code == 400
    assert "does not exist" in r.json()["detail"]

    not_cs2 = tmp_path / "docs"
    not_cs2.mkdir()
    r = client_app.post(
        "/api/settings",
        json={
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "cs2_install_path": str(not_cs2),
        },
    )
    assert r.status_code == 400
    assert "game" in r.json()["detail"]

    # Quotes pasted from Explorer's "Copy as path" are tolerated.
    install = tmp_path / "cs2"
    (install / "game" / "csgo").mkdir(parents=True)
    r = client_app.post(
        "/api/settings",
        json={
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "cs2_install_path": f'"{install}"',
        },
    )
    assert r.status_code == 200
    assert r.json()["cs2_install_path"] == str(install)


def test_models_fallback_without_key(client_app: TestClient):
    r = client_app.get("/api/models")
    assert r.status_code == 200
    models = r.json()
    assert isinstance(models, list)
    assert len(models) > 0


def test_job_persistence_survives_reload(client_app: TestClient, test_cfg: AppConfig):
    from counterstrat.web.ingest import JobState, load_job_state, save_job_state

    state = JobState(
        job_id="persist123",
        stage="mining",
        match_id="m1",
        map_name="de_anubis",
        detail="testing persistence",
    )
    save_job_state(test_cfg.data_root, state)

    # Read from API
    r = client_app.get("/api/jobs/persist123")
    assert r.status_code == 200
    loaded = r.json()
    assert loaded["job_id"] == "persist123"
    assert loaded["stage"] == "mining"
    assert loaded["map_name"] == "de_anubis"
    assert loaded["detail"] == "testing persistence"

    # Read directly
    reloaded = load_job_state(test_cfg.data_root, "persist123")
    assert reloaded is not None
    assert reloaded.job_id == state.job_id
    assert reloaded.stage == state.stage


def test_reports_endpoint_removed(client_app: TestClient):
    """The dossier feature is gone: its endpoint must not resurface."""
    assert client_app.get("/api/reports/team123/de_anubis").status_code == 404


def test_upload_rejects_unsupported_extension(client_app: TestClient):
    r = client_app.post("/api/demos", files={"demo": ("match.zip", b"PBDEMS2\0garbage")})
    assert r.status_code == 400
    assert "Unsupported file format" in r.json()["detail"]


def test_upload_compressed_zst_and_gz(client_app: TestClient, monkeypatch):
    # Prevent background task from executing heavy pipeline on synthetic file
    monkeypatch.setattr("counterstrat.web.routes.run_ingest", lambda *args, **kwargs: None)

    # .dem.zst
    raw_demo = b"PBDEMS2\0" + b"\x00" * 64
    cctx = zstandard.ZstdCompressor()
    zst_data = cctx.compress(raw_demo)
    r1 = client_app.post("/api/demos", files={"demo": ("test_match.dem.zst", zst_data)})
    assert r1.status_code == 200
    assert "job_id" in r1.json()

    # .dem.gz
    gz_data = gzip.compress(raw_demo)
    r2 = client_app.post("/api/demos", files={"demo": ("test_match.dem.gz", gz_data)})
    assert r2.status_code == 200
    assert "job_id" in r2.json()


def test_upload_duplicate_short_circuits(client_app: TestClient, test_cfg: AppConfig, monkeypatch):
    """Re-uploading already-ingested content must not re-run the pipeline."""
    import hashlib

    monkeypatch.setattr("counterstrat.web.routes.run_ingest", lambda *args, **kwargs: None)
    content = b"PBDEMS2\0" + b"x" * 64
    mid = hashlib.sha256(content).hexdigest()[:16]

    r1 = client_app.post("/api/demos", files={"demo": ("dup.dem", content)})
    assert r1.status_code == 200

    # Simulate that first ingest COMPLETED: manifest entry + round scripts exist.
    from counterstrat.corpus import DemoRecord

    rec = DemoRecord(
        match_id=mid,
        path=str(test_cfg.data_root / "uploads" / "dup.dem"),
        map_name="de_anubis",
        patch_version="14178",
        demo_version_guid="g",
        server_name="s",
        registered_at="2026-09-05T00:00:00Z",
    )
    (test_cfg.data_root / "corpus.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    scripts_dir = test_cfg.data_root / "scripts" / mid
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "round_1.json").write_text("{}", encoding="utf-8")

    r2 = client_app.post("/api/demos", files={"demo": ("dup2.dem", content)})
    assert r2.status_code == 200
    job = client_app.get(f"/api/jobs/{r2.json()['job_id']}").json()
    assert job["stage"] == "duplicate"
    assert job["match_id"] == mid
    assert job["map_name"] == "de_anubis"
    assert "dup.dem" in job["detail"] or mid in job["detail"]
    # The redundant copy is removed; the original upload survives.
    assert not (test_cfg.data_root / "uploads" / "dup2.dem").exists()
    assert (test_cfg.data_root / "uploads" / "dup.dem").exists()


def test_upload_incomplete_prior_ingest_reruns(
    client_app: TestClient, test_cfg: AppConfig, monkeypatch
):
    """Registered but script-less (a crashed ingest) is NOT a duplicate: re-run."""
    import hashlib

    ran = []
    monkeypatch.setattr("counterstrat.web.routes.run_ingest", lambda *args, **kwargs: ran.append(1))
    content = b"PBDEMS2\0" + b"y" * 64
    mid = hashlib.sha256(content).hexdigest()[:16]
    from counterstrat.corpus import DemoRecord

    rec = DemoRecord(
        match_id=mid,
        path=str(test_cfg.data_root / "uploads" / "crashed.dem"),
        map_name="de_dust2",
        patch_version="14140",
        demo_version_guid="g",
        server_name="s",
        registered_at="2026-09-05T00:00:00Z",
    )
    (test_cfg.data_root / "corpus.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")
    r = client_app.post("/api/demos", files={"demo": ("crashed.dem", content)})
    assert r.status_code == 200
    job = client_app.get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["stage"] == "queued"
    assert ran == [1]


def test_upload_same_filename_never_overwrites(
    client_app: TestClient, test_cfg: AppConfig, monkeypatch
):
    monkeypatch.setattr("counterstrat.web.routes.run_ingest", lambda *args, **kwargs: None)
    a = b"PBDEMS2\0" + b"a" * 32
    b = b"PBDEMS2\0" + b"b" * 32
    assert client_app.post("/api/demos", files={"demo": ("same.dem", a)}).status_code == 200
    assert client_app.post("/api/demos", files={"demo": ("same.dem", b)}).status_code == 200
    uploaded = list((test_cfg.data_root / "uploads").glob("*.dem"))
    assert len(uploaded) == 2
    assert {p.read_bytes() for p in uploaded} == {a, b}


def test_ingest_rejects_retired_maps(test_cfg: AppConfig, tmp_path, monkeypatch):
    """A retired-map demo fails the job with a clear message and never touches
    the corpus manifest."""
    from counterstrat.web.ingest import load_job_state, run_ingest

    class FakeParser:
        def __init__(self, _path: str): ...

        def parse_header(self):
            return {"map_name": "de_overpass"}

    monkeypatch.setattr("demoparser2.DemoParser", FakeParser)
    demo = tmp_path / "retired.dem"
    demo.write_bytes(b"PBDEMS2\0" + b"x" * 32)

    run_ingest("job_retired", demo, test_cfg)

    job = load_job_state(test_cfg.data_root, "job_retired")
    assert job is not None and job.stage == "error"
    assert "retired" in job.detail and "de_overpass" in job.detail
    manifest = test_cfg.data_root / "corpus.jsonl"
    assert not manifest.exists() or "de_overpass" not in manifest.read_text(encoding="utf-8")


def test_calibrate_supported_maps_exclude_retired():
    from counterstrat.mapcard.calibrate import _supported_maps

    supported = _supported_maps()
    assert {"de_overpass", "de_train", "de_vertigo"}.isdisjoint(supported)


def test_list_demos_empty_and_populated(client_app: TestClient, test_cfg: AppConfig):
    # Initially empty
    r = client_app.get("/api/demos")
    assert r.status_code == 200
    assert r.json() == []

    # Write a record into corpus.jsonl
    from counterstrat.corpus import DemoRecord

    rec = DemoRecord(
        match_id="demo123",
        path="/fake/path.dem",
        map_name="de_dust2",
        patch_version="14178",
        demo_version_guid="guid123",
        server_name="Valve Server",
        registered_at="2026-09-03T00:00:00Z",
    )
    corpus_file = test_cfg.data_root / "corpus.jsonl"
    corpus_file.write_text(rec.model_dump_json() + "\n", encoding="utf-8")

    r2 = client_app.get("/api/demos")
    assert r2.status_code == 200
    demos = r2.json()
    assert len(demos) == 1
    assert demos[0]["match_id"] == "demo123"
    assert demos[0]["map_name"] == "de_dust2"
    assert demos[0]["team_keys"] == []


def test_find_vpk_and_vrf_from_other_working_directory(
    monkeypatch, tmp_path: Path, test_cfg: AppConfig
):
    from counterstrat.web.ingest import REPO_ROOT, _find_vpk_path, _find_vrf_cli

    assert REPO_ROOT.exists()

    # Create dummy maps and tools inside a fake repo root
    fake_repo = tmp_path / "fake_repo"
    vpk_file = fake_repo / "maps" / "de_anubis" / "de_anubis.vpk"
    vpk_file.parent.mkdir(parents=True, exist_ok=True)
    vpk_file.write_bytes(b"vpk")

    vrf_file = fake_repo / "tools" / "vrf" / "Source2Viewer-CLI.exe"
    vrf_file.parent.mkdir(parents=True, exist_ok=True)
    vrf_file.write_bytes(b"vrf")

    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)

    # Change current working directory to somewhere else
    other_dir = tmp_path / "somewhere_else"
    other_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(other_dir)

    found_vpk = _find_vpk_path("de_anubis", test_cfg)
    assert found_vpk == vpk_file

    found_vrf = _find_vrf_cli()
    assert found_vrf == vrf_file


def test_list_teams_map_stats(client_app: TestClient, test_cfg: AppConfig):
    from counterstrat.corpus import DemoRecord

    rec1 = DemoRecord(
        match_id="m1",
        path="/fake/m1.dem",
        map_name="de_anubis",
        patch_version="14178",
        demo_version_guid="guid1",
        server_name="Server1",
        registered_at="2026-09-03T00:00:00Z",
    )
    rec2 = DemoRecord(
        match_id="m2",
        path="/fake/m2.dem",
        map_name="de_mirage",
        patch_version="14178",
        demo_version_guid="guid2",
        server_name="Server2",
        registered_at="2026-09-03T00:00:00Z",
    )
    corpus_file = test_cfg.data_root / "corpus.jsonl"
    corpus_file.write_text(
        rec1.model_dump_json() + "\n" + rec2.model_dump_json() + "\n",
        encoding="utf-8",
    )

    lake_m1 = test_cfg.data_root / "lake" / "m1"
    lake_m1.mkdir(parents=True, exist_ok=True)
    df_m1 = pl.DataFrame(
        {
            "team_key": ["team_a", "team_a", "team_b"],
            "clan_name": ["Team Alpha", "Team Alpha", "Team Beta"],
            "round_num": [1, 2, 1],
        }
    )
    df_m1.write_parquet(lake_m1 / "rosters.parquet")

    lake_m2 = test_cfg.data_root / "lake" / "m2"
    lake_m2.mkdir(parents=True, exist_ok=True)
    df_m2 = pl.DataFrame(
        {
            "team_key": ["team_a", "team_c"],
            "clan_name": ["Team Alpha", "Team Gamma"],
            "round_num": [1, 1],
        }
    )
    df_m2.write_parquet(lake_m2 / "rosters.parquet")

    r = client_app.get("/api/teams")
    assert r.status_code == 200
    teams_by_key = {t["team_key"]: t for t in r.json()}

    team_a = teams_by_key["team_a"]
    assert team_a["demos"] == 2
    assert team_a["rounds"] == 3
    assert "map_stats" in team_a
    anubis = team_a["map_stats"]["de_anubis"]
    assert anubis["demos"] == 1 and anubis["rounds"] == 2
    assert [m["match_id"] for m in anubis["matches"]] == ["m1"]
    mirage = team_a["map_stats"]["de_mirage"]
    assert mirage["demos"] == 1 and mirage["rounds"] == 1

    team_b = teams_by_key["team_b"]
    assert team_b["demos"] == 1
    assert team_b["rounds"] == 1
    assert team_b["map_stats"]["de_anubis"]["demos"] == 1
    assert team_b["map_stats"]["de_anubis"]["rounds"] == 1
    assert "de_mirage" not in team_b["map_stats"]
