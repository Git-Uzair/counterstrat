"""Shipped zone topologies: engine-vocabulary snapshots and the fresh-clone guarantee."""

import json

import polars as pl

from counterstrat.mapcard.topologies import (
    export_shipped_topologies,
    load_shipped_topology,
    merge_topologies,
    save_shipped_topology,
    shipped_topology_maps,
)
from counterstrat.mining.gaps import hold_complex

MAP_POOL = (
    "de_ancient",
    "de_anubis",
    "de_cache",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
)


def test_save_and_load_roundtrip(tmp_path):
    topo = {"BombsiteA": {"Main": 3.1, "Heaven": 6.2}}
    save_shipped_topology("de_test", topo, game_version="1", generated_from=["m1"], root=tmp_path)
    assert load_shipped_topology("de_test", root=tmp_path) == topo
    assert shipped_topology_maps(root=tmp_path) == {"de_test"}
    # Absent and torn files load as empty, never raise.
    assert load_shipped_topology("de_missing", root=tmp_path) == {}
    (tmp_path / "de_torn.json").write_text("{nope", encoding="utf-8")
    assert load_shipped_topology("de_torn", root=tmp_path) == {}


def _walk_ticks(mid: str) -> pl.DataFrame:
    """One player pacing EngineA <-> EngineB; a custom zone renames EngineA."""
    rows = []
    tick = 0
    for lap in range(8):
        for zone in ("EngineA", "EngineB"):
            for _ in range(3):
                tick += 16  # 4 Hz sampling at 64 tick
                rows.append(
                    {
                        "match_id": mid,
                        "steamid": 76500000000000001,
                        "round_num": 1,
                        "tick": tick,
                        "clock_s": tick / 64.0,
                        "is_alive": True,
                        # The operator carved 'myspot' out of EngineA: effective
                        # vocabulary is custom, the engine's own name survives
                        # in place_default.
                        "last_place_name": "myspot" if zone == "EngineA" else zone,
                        "place_default": zone,
                    }
                )
        _ = lap
    return pl.DataFrame(rows)


def test_export_rebuilds_engine_vocabulary_from_the_lake(tmp_path):
    """Snapshots must carry the map's default callouts, never the operator's."""
    data_root = tmp_path / "data"
    mid = "m0000000000000001"
    (data_root / "lake" / mid).mkdir(parents=True)
    _walk_ticks(mid).write_parquet(data_root / "lake" / mid / "ticks.parquet")
    data_root.joinpath("corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": mid,
                "path": "x.dem",
                "map_name": "de_x",
                "patch_version": "9",
                "demo_version_guid": "g",
                "server_name": "",
                "registered_at": "2026-09-06T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    shipped_root = tmp_path / "shipped"
    written = export_shipped_topologies(data_root, root=shipped_root)
    assert [p.name for p in written] == ["de_x.json"]
    topo = load_shipped_topology("de_x", root=shipped_root)
    nodes = set(topo) | {v for nbrs in topo.values() for v in nbrs}
    assert nodes == {"EngineA", "EngineB"}, f"custom vocabulary leaked: {nodes}"
    assert topo["EngineA"]["EngineB"] > 0
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["generated_from"] == [mid]
    assert payload["game_version"] == "9"


def test_export_card_source_renames_operator_customs(tmp_path):
    """No lake for a map: the card exports with custom zones renamed to their
    engine parents (calibration-cache evidence); unresolvable ones are dropped."""
    import yaml

    data_root = tmp_path / "data"
    card_dir = data_root / "mapcards" / "de_y"
    card_dir.mkdir(parents=True)
    card_dir.joinpath("card.yaml").write_text(
        yaml.safe_dump(
            {
                "map": "de_y",
                "game_version": "9",
                "checksum": "cafe",
                "topology": {
                    "myspot": {"EngineB": 2.0},
                    "EngineB": {"myspot": 2.0, "EngineC": 3.0},
                    "ghost": {"EngineC": 1.0},  # custom with no calibration evidence
                },
            }
        ),
        encoding="utf-8",
    )
    card_dir.joinpath("zones.json").write_text(
        json.dumps(
            [
                {"name": "myspot", "x": 0.0, "y": 0.0, "z": 0.0, "half_x": 100.0, "half_y": 100.0},
                {"name": "ghost", "x": 9e4, "y": 9e4, "z": 0.0, "half_x": 50.0, "half_y": 50.0},
            ]
        ),
        encoding="utf-8",
    )
    cache = data_root / "calibration" / ".cache"
    cache.mkdir(parents=True)
    pl.DataFrame(
        {
            "X": [10.0, -20.0, 30.0],
            "Y": [5.0, 15.0, -25.0],
            "Z": [0.0, 0.0, 0.0],
            "last_place_name": ["EngineA", "EngineA", "EngineA"],
        }
    ).write_parquet(cache / "abc.parquet")
    cache.joinpath("index.json").write_text(
        json.dumps({"abc": {"file": "d.dem", "map": "de_y", "rows": 3}}), encoding="utf-8"
    )

    shipped_root = tmp_path / "shipped"
    written = export_shipped_topologies(data_root, root=shipped_root)
    assert [p.name for p in written] == ["de_y.json"]
    assert load_shipped_topology("de_y", root=shipped_root) == {
        "EngineA": {"EngineB": 2.0},
        "EngineB": {"EngineA": 2.0, "EngineC": 3.0},
    }
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["generated_from"] == ["card:cafe"]


def test_merge_topologies_live_wins_per_edge():
    shipped = {"A": {"B": 3.0, "C": 4.0}}
    live = {"A": {"B": 2.0, "Custom": 1.0}, "Custom": {"A": 1.0}}
    assert merge_topologies(shipped, live) == {
        "A": {"B": 2.0, "C": 4.0, "Custom": 1.0},
        "Custom": {"A": 1.0},
    }
    # Either side may be empty.
    assert merge_topologies({}, live) == live
    assert merge_topologies(shipped, {}) == shipped


def test_shipped_topologies_cover_the_map_pool():
    """A fresh clone must mine real hold complexes for every calibrated map."""
    assert set(MAP_POOL) <= shipped_topology_maps()
    for map_name in MAP_POOL:
        topo = load_shipped_topology(map_name)
        assert topo, f"no shipped topology for {map_name}"
        complex_a = hold_complex("BombsiteA", topo)
        assert len(complex_a) > 1, f"degenerate BombsiteA complex on {map_name}"
    # Spot-check the calibration that motivated all of this (Anubis A holds).
    anubis_a = hold_complex("BombsiteA", load_shipped_topology("de_anubis"))
    assert {"Heaven", "Main", "Walkway"} <= anubis_a
    assert "Middle" not in anubis_a


def test_shipped_topologies_speak_engine_vocabulary():
    """Regression for the operator-custom leak: shipped graphs carry the maps'
    own callouts only (the operator's zones.json names must never ship)."""

    def nodes_of(map_name: str) -> set[str]:
        topo = load_shipped_topology(map_name)
        return set(topo) | {v for nbrs in topo.values() for v in nbrs}

    anubis = nodes_of("de_anubis")
    assert "Bridge" in anubis and "bridge" not in anubis
    assert not {"broky", "temple", "africa", "ninja", "underheaven"} & nodes_of("de_ancient")
    assert not {"tripple", "coffin", "half wall", "sandbags", "a short"} & nodes_of("de_inferno")
