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


def test_reports_503_without_key(client_app: TestClient):
    r = client_app.get("/api/reports/team123/de_anubis")
    assert r.status_code == 503
    assert "API key" in r.json()["detail"]


def test_reports_cached_markdown(client_app: TestClient, test_cfg: AppConfig):
    report_file = test_cfg.data_root / "teambooks" / "team1" / "de_anubis" / "dossier.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("# Anti-Strat Dossier\n\nCached content.", encoding="utf-8")

    r = client_app.get("/api/reports/team1/de_anubis")
    assert r.status_code == 200
    assert "Cached content." in r.text
    assert "text/markdown" in r.headers["content-type"]


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
    assert team_a["map_stats"]["de_anubis"] == {"demos": 1, "rounds": 2}
    assert team_a["map_stats"]["de_mirage"] == {"demos": 1, "rounds": 1}

    team_b = teams_by_key["team_b"]
    assert team_b["demos"] == 1
    assert team_b["rounds"] == 1
    assert "map_stats" in team_b
    assert team_b["map_stats"]["de_anubis"] == {"demos": 1, "rounds": 1}
    assert "de_mirage" not in team_b["map_stats"]
