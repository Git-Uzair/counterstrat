"""Demo deletion: files, manifest, and derived artifacts (feature A)."""

import json
from pathlib import Path

import pytest
from conftest import build_synthetic_scripts
from fastapi.testclient import TestClient
from test_teams_clusters import SQUAD_A, SQUAD_B, SQUAD_C, SQUAD_D, _lineup_key, _write_match

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app
from counterstrat.web.maintenance import rebuild_artifacts

KEY_A = _lineup_key(SQUAD_A)  # team lineup 1
KEY_B = _lineup_key(SQUAD_B)  # same team, stand-in lineup
KEY_C = _lineup_key(SQUAD_C)  # opponent 1
KEY_D = _lineup_key(SQUAD_D)  # opponent 2


def _write_scripts(data_root: Path, match_id: str, t_key: str, ct_key: str) -> None:
    for s in build_synthetic_scripts():
        s2 = s.model_copy(
            update={
                "match_id": match_id,
                "t_team_key": t_key if s.t_team_key == "abc" else ct_key,
                "ct_team_key": t_key if s.ct_team_key == "abc" else ct_key,
            }
        )
        p = data_root / "scripts" / match_id / f"round_{s2.round_num}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s2.to_json(), encoding="utf-8")


@pytest.fixture
def populated(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(data_root=tmp_path)
    # Two demos of the same team (stand-in lineups) on de_anubis vs two opponents.
    _write_match(tmp_path, "m_one", "de_anubis", [(SQUAD_A, "team_x"), (SQUAD_C, "opp1")])
    _write_match(tmp_path, "m_two", "de_anubis", [(SQUAD_B, "team_x"), (SQUAD_D, "opp2")])
    _write_scripts(tmp_path, "m_one", KEY_A, KEY_C)
    _write_scripts(tmp_path, "m_two", KEY_B, KEY_D)
    # Fake uploaded demo files referenced by the manifest.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    manifest_lines = []
    for line in (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        demo_file = uploads / f"{rec['match_id']}.dem"
        demo_file.write_bytes(b"PBDEMS2\0fake")
        rec["path"] = str(demo_file)
        manifest_lines.append(json.dumps(rec))
    (tmp_path / "corpus.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    rebuild_artifacts(cfg)
    return cfg


def test_delete_demo_removes_everything_and_rebuilds(populated: AppConfig):
    cfg = populated
    client = TestClient(create_app(cfg))
    team_id = min(KEY_A, KEY_B)  # merged cluster id before deletion

    tb_path = cfg.data_root / "teambooks" / team_id / "de_anubis" / "teambook.json"
    assert tb_path.exists()
    assert json.loads(tb_path.read_text(encoding="utf-8"))["generated_from"] == [
        "m_one",
        "m_two",
    ]

    r = client.delete("/api/demos/m_one")
    assert r.status_code == 200
    assert r.json() == {"deleted": "m_one", "map_name": "de_anubis"}

    # Manifest, lake, scripts, and the uploaded file are gone.
    assert "m_one" not in (cfg.data_root / "corpus.jsonl").read_text(encoding="utf-8")
    assert not (cfg.data_root / "lake" / "m_one").exists()
    assert not (cfg.data_root / "scripts" / "m_one").exists()
    assert not (cfg.data_root / "uploads" / "m_one.dem").exists()
    assert (cfg.data_root / "uploads" / "m_two.dem").exists()

    # The surviving lineup re-clusters under its own id; artifacts follow.
    teams = client.get("/api/teams").json()
    keys = {t["team_key"] for t in teams}
    assert KEY_B in keys and KEY_D in keys
    assert team_id not in keys or team_id == KEY_B
    new_tb = cfg.data_root / "teambooks" / KEY_B / "de_anubis" / "teambook.json"
    assert json.loads(new_tb.read_text(encoding="utf-8"))["generated_from"] == ["m_two"]
    # Dead teambook dirs are pruned.
    live_dirs = {p.parent.parent.name for p in cfg.data_root.glob("teambooks/*/*/teambook.json")}
    assert KEY_A not in live_dirs and KEY_C not in live_dirs


def test_batch_delete_removes_all_and_rebuilds_once(populated: AppConfig, monkeypatch):
    """One card-level delete call removes N demos with a single artifact rebuild."""
    cfg = populated
    client = TestClient(create_app(cfg))

    calls = []
    from counterstrat.web import maintenance

    real = maintenance.rebuild_artifacts
    monkeypatch.setattr(maintenance, "rebuild_artifacts", lambda c: (calls.append(1), real(c))[1])

    r = client.delete("/api/demos", params={"matches": "m_one,m_two"})
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["deleted"]) == ["m_one", "m_two"]
    assert calls == [1]

    manifest_text = (cfg.data_root / "corpus.jsonl").read_text(encoding="utf-8")
    for mid in ("m_one", "m_two"):
        assert mid not in manifest_text
        assert not (cfg.data_root / "lake" / mid).exists()
        assert not (cfg.data_root / "scripts" / mid).exists()
        assert not (cfg.data_root / "uploads" / f"{mid}.dem").exists()
    # No teambooks survive an emptied corpus.
    assert not list(cfg.data_root.glob("teambooks/*/*/teambook.json"))


def test_batch_delete_unknown_id_rejects_whole_batch(populated: AppConfig):
    cfg = populated
    client = TestClient(create_app(cfg))
    r = client.delete("/api/demos", params={"matches": "m_one,ghost"})
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]
    # Nothing was deleted.
    assert "m_one" in (cfg.data_root / "corpus.jsonl").read_text(encoding="utf-8")
    assert (cfg.data_root / "lake" / "m_one").exists()


def test_batch_delete_requires_ids(populated: AppConfig):
    client = TestClient(create_app(populated))
    assert client.delete("/api/demos", params={"matches": " , "}).status_code == 400


def test_delete_unknown_demo_404(populated: AppConfig):
    client = TestClient(create_app(populated))
    assert client.delete("/api/demos/ghost").status_code == 404


def test_delete_leaves_external_demo_files_alone(populated: AppConfig, tmp_path: Path):
    cfg = populated
    external = tmp_path.parent / f"{tmp_path.name}_external.dem"
    external.write_bytes(b"PBDEMS2\0external")
    lines = []
    for line in (cfg.data_root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["match_id"] == "m_two":
            rec["path"] = str(external)
        lines.append(json.dumps(rec))
    (cfg.data_root / "corpus.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    client = TestClient(create_app(cfg))
    assert client.delete("/api/demos/m_two").status_code == 200
    assert external.exists(), "files outside the data root must never be deleted"
