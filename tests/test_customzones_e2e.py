"""Real-demo proof: a pointed callout propagates into scripts and teambooks."""

import json
import shutil
import time
from pathlib import Path

import polars as pl
import pytest

from counterstrat.config import AppConfig
from counterstrat.customzones import CustomZone, save_custom_zones
from counterstrat.web.maintenance import rebuild_map_zones

MAP = "de_anubis"


@pytest.fixture(autouse=True)
def isolated_repo_root(tmp_path: Path, monkeypatch):
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    return fake_repo


def test_custom_zone_reaches_scripts_and_teambook(tmp_path: Path, anubis_lake):
    cfg = AppConfig(data_root=tmp_path / "data")
    match_dir = Path(anubis_lake.root)
    match_id = match_dir.name
    dest = cfg.data_root / "lake" / match_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(match_dir, dest)
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": match_id,
                "path": "demos/real.dem",
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

    # Ground the zone at the densest real spot of Middle.
    ticks = pl.read_parquet(dest / "ticks.parquet")
    mid = ticks.filter(pl.col("is_alive") & (pl.col("last_place_name") == "Middle"))
    cx, cy, cz = (float(mid["X"].median()), float(mid["Y"].median()), float(mid["Z"].median()))
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=cx, y=cy, z=cz, half_x=200.0, half_y=200.0)],
        reserved=set(),
    )

    t0 = time.perf_counter()
    result = rebuild_map_zones(cfg, MAP)
    elapsed = time.perf_counter() - t0
    print(f"rebuild_map_zones({MAP}): {elapsed:.1f}s for {result['matches']} match(es)")
    assert elapsed < 120.0, "rebuild must stay interactive-job fast for one match"

    # 1. Lake speaks it, reversibly.
    zoned = pl.read_parquet(dest / "ticks.parquet")
    assert "Sandbags" in zoned["last_place_name"].unique().to_list()
    assert "place_default" in zoned.columns

    # 2. Card speaks it (compiled from the tick vocabulary; no VPK here).
    card_text = (cfg.data_root / "mapcards" / MAP / "card.yaml").read_text(encoding="utf-8")
    assert "Sandbags" in card_text

    # 3. Scripts speak it (movements/beats cite the zone players stood in).
    scripts_text = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((cfg.data_root / "scripts" / match_id).glob("round_*.json"))
    )
    assert "Sandbags" in scripts_text

    # 4. Teambooks (the miners' vocabulary) speak it.
    teambooks = "".join(
        p.read_text(encoding="utf-8")
        for p in (cfg.data_root / "teambooks").glob(f"*/{MAP}/teambook.json")
    )
    assert "Sandbags" in teambooks
