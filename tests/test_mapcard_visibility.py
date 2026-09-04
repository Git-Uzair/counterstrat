"""Empirical zone visibility matrix tests (space-vision research item 3, Route A).

Clean kills prove sightlines; smoke/wallbang kills prove nothing. The matrix
lands on the card's long-empty ``sightlines`` field and renders into the
scene-graph prompt block.
"""

import json

import polars as pl
import yaml

from counterstrat.mapcard.visibility import (
    MAX_SIGHTLINES,
    build_sightlines,
    refresh_card_sightlines,
)


def _kills(rows: list[tuple]) -> pl.DataFrame:
    """rows: (attacker_place, victim_place, distance, thrusmoke, penetrated)."""
    return pl.DataFrame(
        {
            "attacker_place": [r[0] for r in rows],
            "victim_place": [r[1] for r in rows],
            "distance": [r[2] for r in rows],
            "thrusmoke": [r[3] for r in rows],
            "penetrated": [r[4] for r in rows],
        },
        schema_overrides={"distance": pl.Float32, "penetrated": pl.Int32},
    )


def test_sightlines_aggregate_symmetric_pairs_with_bands():
    kills = _kills(
        [
            ("Middle", "BombsiteA", 40.0, False, 0),
            ("BombsiteA", "Middle", 50.0, False, 0),  # reverse direction merges
            ("Middle", "BombsiteA", 45.0, False, 0),
            ("Tunnel", "BombsiteB", 5.0, False, 0),
            ("Tunnel", "BombsiteB", 10.0, False, 0),
        ]
    )
    lines = build_sightlines(kills)
    assert len(lines) == 2
    top = lines[0]  # sorted by n desc
    assert (top["from"], top["to"]) == ("BombsiteA", "Middle")  # alphabetical pair
    assert top["n"] == 3
    assert top["median_dist"] == 45.0
    assert top["range"] == "long"
    close = lines[1]
    assert close["n"] == 2 and close["range"] == "close"


def test_sightlines_exclude_dirty_and_same_zone_evidence():
    kills = _kills(
        [
            ("Middle", "BombsiteA", 20.0, True, 0),  # through smoke: no sightline
            ("Middle", "BombsiteA", 20.0, False, 2),  # wallbang: no sightline
            ("Middle", "Middle", 5.0, False, 0),  # same zone: says nothing
            ("Middle", None, 12.0, False, 0),  # unknown endpoint
            ("Middle", "", 12.0, False, 0),  # empty endpoint
            ("Middle", "BombsiteA", None, False, 0),  # no distance recorded
        ]
    )
    assert build_sightlines(kills) == []
    assert build_sightlines(pl.DataFrame()) == []


def test_sightlines_min_events_floor_and_zone_filter():
    kills = _kills(
        [
            ("Middle", "BombsiteA", 20.0, False, 0),  # n=1: anecdote, dropped
            ("Alley", "Water", 20.0, False, 0),
            ("Alley", "Water", 25.0, False, 0),
            ("Ghost", "Water", 20.0, False, 0),  # Ghost not a card zone
            ("Ghost", "Water", 25.0, False, 0),
        ]
    )
    lines = build_sightlines(kills, valid_zones={"Middle", "BombsiteA", "Alley", "Water"})
    assert [(li["from"], li["to"]) for li in lines] == [("Alley", "Water")]


def test_sightlines_capped():
    rows = []
    for i in range(MAX_SIGHTLINES + 20):
        rows += [(f"Z{i:03d}", f"Y{i:03d}", 12.0, False, 0)] * 2
    lines = build_sightlines(_kills(rows))
    assert len(lines) == MAX_SIGHTLINES


def test_refresh_card_sightlines_end_to_end(tmp_path):
    data_root = tmp_path
    # manifest with two matches on the map, one on another map
    rec = {
        "path": "x.dem",
        "patch_version": "1",
        "demo_version_guid": "g",
        "server_name": "",
        "registered_at": "now",
    }
    manifest_lines = [
        json.dumps({"match_id": "m1", "map_name": "de_test", **rec}),
        json.dumps({"match_id": "m2", "map_name": "de_test", **rec}),
        json.dumps({"match_id": "m3", "map_name": "de_other", **rec}),
    ]
    (data_root / "corpus.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    for mid, dist in (("m1", 8.0), ("m2", 12.0)):
        lake_dir = data_root / "lake" / mid
        lake_dir.mkdir(parents=True)
        _kills([("Middle", "BombsiteA", dist, False, 0)]).write_parquet(lake_dir / "kills.parquet")
    # m3 (other map) would poison the matrix if included
    other = data_root / "lake" / "m3"
    other.mkdir(parents=True)
    _kills([("Nowhere", "Elsewhere", 5.0, False, 0)] * 5).write_parquet(other / "kills.parquet")

    card_dir = data_root / "mapcards" / "de_test"
    card_dir.mkdir(parents=True)
    card = {
        "map": "de_test",
        "card_version": "1.0",
        "game_version": "1",
        "nav_source": "nav",
        "frame": {},
        "zones": {"Middle": {}, "BombsiteA": {}},
        "topology": {},
        "rotates": [],
        "timings": {},
        "objectives": {},
        "sightlines": [],
        "checksum": "chk",
    }
    (card_dir / "card.yaml").write_text(yaml.dump(card), encoding="utf-8")

    written = refresh_card_sightlines(data_root, "de_test")
    assert len(written) == 1  # one pair, n=2 across the two matches

    saved = yaml.safe_load((card_dir / "card.yaml").read_text(encoding="utf-8"))
    assert saved["sightlines"] == [
        {"from": "BombsiteA", "to": "Middle", "n": 2, "median_dist": 10.0, "range": "close"}
    ]
    assert saved["checksum"] == "chk"  # compile provenance untouched

    # maps without a card are a no-op, not a crash
    assert refresh_card_sightlines(data_root, "de_missing") == []


def test_scene_graph_renders_sightlines():
    from counterstrat.llm.prompts import format_map_scene_graph
    from counterstrat.mapcard.compile import MapCard

    card = MapCard(
        map="de_test",
        game_version="1",
        frame={},
        zones={"Middle": {}, "BombsiteA": {}},
        topology={},
        rotates=[],
        timings={},
        objectives={},
        sightlines=[
            {"from": "BombsiteA", "to": "Middle", "n": 4, "median_dist": 45.0, "range": "long"}
        ],
        checksum="chk",
    )
    block = format_map_scene_graph(card, {})
    assert "sightlines:" in block
    assert "`BombsiteA` <-> `Middle`: long (~45m, n=4)" in block
