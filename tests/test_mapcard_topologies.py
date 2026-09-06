"""Shipped zone topologies: snapshot, load, and the fresh-clone guarantee."""

import json

import yaml

from counterstrat.mapcard.topologies import (
    export_shipped_topologies,
    load_shipped_topology,
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
    save_shipped_topology("de_test", topo, card_checksum="abc", game_version="1", root=tmp_path)
    assert load_shipped_topology("de_test", root=tmp_path) == topo
    assert shipped_topology_maps(root=tmp_path) == {"de_test"}
    # Absent and torn files load as empty, never raise.
    assert load_shipped_topology("de_missing", root=tmp_path) == {}
    (tmp_path / "de_torn.json").write_text("{nope", encoding="utf-8")
    assert load_shipped_topology("de_torn", root=tmp_path) == {}


def test_export_snapshots_every_compiled_card(tmp_path):
    data_root = tmp_path / "data"
    card_dir = data_root / "mapcards" / "de_x"
    card_dir.mkdir(parents=True)
    card_dir.joinpath("card.yaml").write_text(
        yaml.safe_dump(
            {
                "map": "de_x",
                "game_version": "9",
                "checksum": "deadbeef",
                "topology": {"BombsiteA": {"Main": 2.5}},
            }
        ),
        encoding="utf-8",
    )
    shipped_root = tmp_path / "shipped"
    written = export_shipped_topologies(data_root, root=shipped_root)
    assert [p.name for p in written] == ["de_x.json"]
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["card_checksum"] == "deadbeef"
    assert load_shipped_topology("de_x", root=shipped_root) == {"BombsiteA": {"Main": 2.5}}


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
