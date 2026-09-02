from pathlib import Path

import pytest

from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.vrf import extract_map_assets


def test_parse_places_mini():
    vols = parse_places(Path("tests/fixtures/mini.vents"))
    assert len(vols) == 2  # only env_cs_place blocks
    assert vols[0].place_name == "TSpawn"
    assert vols[0].origin == (-16.0, -1520.0, 104.0)
    assert unique_places(vols) == ["Street", "TSpawn"]


@pytest.mark.demo
def test_extract_anubis(anubis_vpk, vrf_cli, tmp_path):
    assets = extract_map_assets(anubis_vpk, vrf_cli, tmp_path)
    vols = parse_places(assets.vents)
    names = unique_places(vols)
    assert len(vols) == 62 and len(names) == 28  # verified counts
    assert assets.nav.stat().st_size == 495_530  # verified size


@pytest.mark.demo
def test_extract_ancient(ancient_vpk, vrf_cli, tmp_path):
    assets = extract_map_assets(ancient_vpk, vrf_cli, tmp_path)
    vols = parse_places(assets.vents)
    names = unique_places(vols)
    assert len(names) > 0
    assert assets.nav.stat().st_size == 371_251
