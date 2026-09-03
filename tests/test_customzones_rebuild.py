"""Zone rebuild: ticks re-zoned in place, card recompiled, caches pruned."""

import json
from pathlib import Path

import polars as pl
import pytest

from counterstrat.config import AppConfig
from counterstrat.customzones import CustomZone, save_custom_zones
from counterstrat.web.maintenance import compile_map_card, rebuild_map_zones

MAP = "de_test"


@pytest.fixture(autouse=True)
def isolated_repo_root(tmp_path: Path, monkeypatch):
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    return fake_repo


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(data_root=tmp_path / "data")
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    return cfg


def _seed_corpus_and_lake(cfg: AppConfig, match_id: str = "m1") -> Path:
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": match_id,
                "path": f"demos/{match_id}.dem",
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
    lake = cfg.data_root / "lake" / match_id
    lake.mkdir(parents=True)
    pl.DataFrame(
        {
            "X": [100.0, 105.0, 900.0],
            "Y": [100.0, 105.0, 900.0],
            "Z": [0.0, 0.0, 0.0],
            "last_place_name": ["Middle", "Middle", "BombsiteA"],
            "is_alive": [True, True, True],
            "steamid": [1, 1, 2],
            "round_num": [1, 1, 1],
            "tick": [0, 4, 8],
            "clock_s": [0.0, 1.0, 2.0],
            "team_name": ["CT", "CT", "TERRORIST"],
            "match_id": [match_id] * 3,
        }
    ).write_parquet(lake / "ticks.parquet")
    # Empty companion tables: serialize_match sorts rounds by round_num and
    # then loops zero rounds, so kills/grenades/rosters are never touched
    # (rosters.parquet must NOT exist - build_team_clusters skips a missing
    # file but crashes on one without team_key). Real-demo chain is the e2e.
    pl.DataFrame(
        {"match_id": pl.Series([], dtype=pl.String), "round_num": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(lake / "rounds.parquet")
    pl.DataFrame({"match_id": pl.Series([], dtype=pl.String)}).write_parquet(lake / "kills.parquet")
    return lake


def test_compile_map_card_includes_custom_zones():
    ticks = pl.DataFrame(
        {
            "X": [0.0, 10.0],
            "Y": [0.0, 10.0],
            "Z": [0.0, 0.0],
            "last_place_name": ["Sandbags", "Middle"],
            "is_alive": [True, True],
            "steamid": [1, 1],
            "round_num": [1, 1],
            "tick": [0, 4],
            "clock_s": [0.0, 1.0],
            "team_name": ["CT", "CT"],
        }
    )
    card = compile_map_card(MAP, ["Middle", "BombsiteA"], ticks, pl.DataFrame(), "1")
    assert card is not None
    assert {"Sandbags", "Middle", "BombsiteA"} <= set(card.zones.keys())
    assert compile_map_card(MAP, [], pl.DataFrame(), pl.DataFrame(), "1") is None


def test_rebuild_map_zones_bakes_ticks_card_and_prunes_caches(cfg: AppConfig):
    lake = _seed_corpus_and_lake(cfg)
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=102.0, y=102.0, z=0.0, radius=100.0)],
        reserved=set(),
    )
    # Stale LLM caches that must die with the rebuild.
    tb_dir = cfg.data_root / "teambooks" / "team1" / MAP
    tb_dir.mkdir(parents=True)
    (tb_dir / "dossier.md").write_text("stale", encoding="utf-8")
    (tb_dir / "insights.json").write_text("{}", encoding="utf-8")
    other_map = cfg.data_root / "teambooks" / "team1" / "de_other"
    other_map.mkdir(parents=True)
    (other_map / "dossier.md").write_text("keep", encoding="utf-8")

    result = rebuild_map_zones(cfg, MAP)
    assert result["matches"] == 1 and result["zones"] == 1

    ticks = pl.read_parquet(lake / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == ["Sandbags", "Sandbags", "BombsiteA"]
    assert ticks["place_default"].to_list() == ["Middle", "Middle", "BombsiteA"]

    card = (cfg.data_root / "mapcards" / MAP / "card.yaml").read_text(encoding="utf-8")
    assert "Sandbags" in card

    assert not (tb_dir / "dossier.md").exists()
    assert not (tb_dir / "insights.json").exists()
    assert (other_map / "dossier.md").exists()  # other maps untouched

    # Reversibility: delete the zone, rebuild, game names return.
    save_custom_zones(cfg.data_root, MAP, [], reserved=set())
    rebuild_map_zones(cfg, MAP)
    ticks = pl.read_parquet(lake / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == ["Middle", "Middle", "BombsiteA"]


def test_rebuild_is_idempotent(cfg: AppConfig):
    lake = _seed_corpus_and_lake(cfg)
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=102.0, y=102.0, z=0.0, radius=100.0)],
        reserved=set(),
    )
    rebuild_map_zones(cfg, MAP)
    first = pl.read_parquet(lake / "ticks.parquet")
    rebuild_map_zones(cfg, MAP)
    second = pl.read_parquet(lake / "ticks.parquet")
    assert first.equals(second)
