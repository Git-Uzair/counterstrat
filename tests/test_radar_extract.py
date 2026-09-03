from pathlib import Path

import pytest

from counterstrat.radar.extract import (
    RADAR_IMAGE_PX,
    parse_calibration,
    radar_cache_dir,
)

# Verbatim from pak01: resource/overviews/de_anubis.txt
ANUBIS_OVERVIEW = """// TAVR - AUTO RADAR. v 2.5.0a
"de_anubis"
{
\t"CTSpawn_x" "0.610000"
\t"CTSpawn_y" "0.220000"
\t"TSpawn_x" "0.580000"
\t"TSpawn_y" "0.930000"
\t"material" "overviews/de_anubis"
\t"pos_x" "-2796.000000"
\t"pos_y" "3328.000000"
\t"scale" "5.220000"
}
"""

# Verbatim from pak01: resource/overviews/de_nuke.txt (trailing // comments kept)
NUKE_OVERVIEW = """// HLTV overview description file for de_nuke.bsp

"de_nuke"
{
\t"material"\t"overviews/de_nuke"\t// texture file
\t"pos_x"\t\t"-3453"\t// upper left world coordinate
\t"pos_y"\t\t"2887"
\t"scale"\t\t"7" 

\t"verticalsections"
\t{
\t\t"default" // use the primary radar image
\t\t{
\t\t\t"AltitudeMax" "10000"
\t\t\t"AltitudeMin" "-495"
\t\t}
\t\t"lower" // i.e. de_nuke_lower_radar.dds
\t\t{
\t\t\t"AltitudeMax" "-495"
\t\t\t"AltitudeMin" "-10000"
\t\t}
\t}

\t"CTSpawn_x"\t"0.82"
\t"inset_left"\t\t"0.33"
}
"""

# Verbatim from pak01: resource/overviews/de_ancient.txt (has rotate/zoom)
ANCIENT_OVERVIEW = """// HLTV overview description file for de_ancient.bsp

"de_ancient"
{
\t"material"\t"overviews/de_ancient"\t// texture file
\t"pos_x"\t\t"-2953"\t// upper left world coordinate
\t"pos_y"\t\t"2164"
\t"scale"\t\t"5"
\t"rotate"\t"0"
\t"zoom"\t\t"0"
\t"CTSpawn_x"\t"0.51"
}
"""


def test_parse_calibration_single_level() -> None:
    cal = parse_calibration(ANUBIS_OVERVIEW, "de_anubis")
    assert cal.map_name == "de_anubis"
    assert cal.pos_x == -2796.0
    assert cal.pos_y == 3328.0
    assert cal.scale == 5.22
    assert cal.image_px == RADAR_IMAGE_PX == 1024
    assert cal.lower_altitude_max is None


def test_parse_calibration_multi_level_reads_lower_block() -> None:
    cal = parse_calibration(NUKE_OVERVIEW, "de_nuke")
    assert (cal.pos_x, cal.pos_y, cal.scale) == (-3453.0, 2887.0, 7.0)
    # -495 comes from verticalsections."lower".AltitudeMax, NOT "default".
    assert cal.lower_altitude_max == -495.0


def test_parse_calibration_ignores_rotate_and_zoom() -> None:
    cal = parse_calibration(ANCIENT_OVERVIEW, "de_ancient")
    assert (cal.pos_x, cal.pos_y, cal.scale) == (-2953.0, 2164.0, 5.0)
    assert cal.lower_altitude_max is None


def test_parse_calibration_rejects_missing_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        parse_calibration('"de_x"\n{\n\t"pos_x" "1"\n\t"pos_y" "2"\n}\n', "de_x")


def test_radar_cache_dir_layout(tmp_path: Path) -> None:
    assert radar_cache_dir(tmp_path, "de_anubis") == tmp_path / "radar" / "de_anubis"


def test_load_cached_assets_missing_returns_none(tmp_path: Path) -> None:
    from counterstrat.radar.extract import load_cached_assets

    assert load_cached_assets(tmp_path, "de_anubis") is None


@pytest.mark.demo
def test_extract_radar_assets_from_real_vpk(
    tmp_path: Path, cs2_install: Path, vrf_cli: Path
) -> None:
    from counterstrat.radar.extract import extract_radar_assets, load_cached_assets

    assets = extract_radar_assets(cs2_install, vrf_cli, "de_anubis", tmp_path)
    assert assets.image.exists() and assets.image.stat().st_size > 10_000
    assert assets.image.name == "radar.png"
    assert assets.overview.exists()
    assert assets.lower_image is None  # anubis is single-level
    assert assets.levels == ["default"]
    assert assets.calibration.pos_x == -2796.0
    assert assets.calibration.scale == 5.22
    # calibration.json is written so later requests skip re-parsing
    assert (tmp_path / "radar" / "de_anubis" / "calibration.json").exists()
    # cache hit path returns an equivalent bundle
    cached = load_cached_assets(tmp_path, "de_anubis")
    assert cached is not None
    assert cached.calibration == assets.calibration


@pytest.mark.demo
def test_extract_radar_assets_multi_level(tmp_path: Path, cs2_install: Path, vrf_cli: Path) -> None:
    from counterstrat.radar.extract import extract_radar_assets

    assets = extract_radar_assets(cs2_install, vrf_cli, "de_nuke", tmp_path)
    assert assets.lower_image is not None and assets.lower_image.exists()
    assert assets.levels == ["default", "lower"]
    assert assets.calibration.lower_altitude_max == -495.0


@pytest.mark.demo
def test_extract_radar_assets_unknown_map_raises(
    tmp_path: Path, cs2_install: Path, vrf_cli: Path
) -> None:
    from counterstrat.radar.extract import extract_radar_assets

    with pytest.raises(FileNotFoundError, match="de_bogusmap"):
        extract_radar_assets(cs2_install, vrf_cli, "de_bogusmap", tmp_path)
