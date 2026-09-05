"""User-defined callout zones: storage, validation, and lake re-zoning."""

from pathlib import Path

import polars as pl
import pytest

from counterstrat.customzones import (
    CustomZone,
    load_custom_zones,
    rezone_ticks,
    save_custom_zones,
)


def _rect(name: str = "Sandbags", **kw) -> CustomZone:
    base = {"x": 100.0, "y": 100.0, "z": 0.0, "half_x": 80.0, "half_y": 120.0}
    base.update(kw)
    return CustomZone(name=name, **base)


def test_rezone_assigns_inside_preserves_default_and_restores():
    ticks = pl.DataFrame(
        {
            "X": [100.0, 100.0, 900.0],
            "Y": [100.0, 100.0, 900.0],
            "Z": [0.0, 250.0, 0.0],
            "last_place_name": ["Middle", "MidRoof", "BombsiteA"],
            "is_alive": [True, True, True],
        }
    )
    out = rezone_ticks(ticks, [_rect()])
    assert out["last_place_name"].to_list() == ["Sandbags", "MidRoof", "BombsiteA"]
    # The game vocabulary survives in place_default.
    assert out["place_default"].to_list() == ["Middle", "MidRoof", "BombsiteA"]
    # Idempotent: re-zoning re-zoned ticks changes nothing.
    again = rezone_ticks(out, [_rect()])
    assert again["last_place_name"].to_list() == out["last_place_name"].to_list()
    # Reversible: an empty zone list restores the game names.
    restored = rezone_ticks(out, [])
    assert restored["last_place_name"].to_list() == ["Middle", "MidRoof", "BombsiteA"]


def test_rezone_without_coords_or_rows_is_a_noop():
    empty = pl.DataFrame({"last_place_name": [], "X": [], "Y": [], "Z": []})
    assert rezone_ticks(empty, [_rect()]).is_empty()
    no_coords = pl.DataFrame({"last_place_name": ["Middle"]})
    out = rezone_ticks(no_coords, [_rect()])
    assert out["last_place_name"].to_list() == ["Middle"]


def test_rezone_rect_membership_and_z_band():
    # Footprint x in [20,180], y in [-20,220], |dz| <= 200.
    ticks = pl.DataFrame(
        {
            "X": [100.0, 100.0, 100.0, 190.0, 100.0],
            "Y": [100.0, 210.0, 100.0, 100.0, 100.0],
            "Z": [0.0, 150.0, 250.0, 0.0, -150.0],
            "last_place_name": ["Mid", "Mid", "MidRoof", "Mid", "Under"],
            "is_alive": [True] * 5,
        }
    )
    out = rezone_ticks(ticks, [_rect()])
    assert out["last_place_name"].to_list() == [
        "Sandbags",  # center
        "Sandbags",  # inside footprint, z within band
        "MidRoof",  # 250 above: outside the +-200 band
        "Mid",  # x=190: outside half_x=80
        "Sandbags",  # 150 below: inside the band
    ]
    assert rezone_ticks(out, [])["last_place_name"].to_list()[0] == "Mid"


def test_save_and_load_roundtrip(tmp_path: Path):
    z = _rect(level="lower")
    saved = save_custom_zones(tmp_path, "de_test", [z], reserved={"Middle"})
    assert [s.name for s in saved] == ["Sandbags"]
    loaded = load_custom_zones(tmp_path, "de_test")
    assert loaded == saved
    assert loaded[0].half_x == 80.0 and loaded[0].half_y == 120.0
    # Torn/missing files load as empty, like aliases.
    (tmp_path / "mapcards" / "de_test" / "zones.json").write_text("{not json", encoding="utf-8")
    assert load_custom_zones(tmp_path, "de_test") == []
    assert load_custom_zones(tmp_path, "de_ghost") == []


def test_save_validation_matrix(tmp_path: Path):
    reserved = {"Middle", "Mid"}  # a game zone and an alias value
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [_rect("Middle")], reserved=reserved)
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [_rect("Mid")], reserved=reserved)
    with pytest.raises(ValueError, match="invalid characters"):
        save_custom_zones(tmp_path, "m", [_rect("a`b")], reserved=set())
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        save_custom_zones(
            tmp_path, "m", [_rect("A"), _rect("A", x=9000.0, y=9000.0)], reserved=set()
        )
    with pytest.raises(ValueError, match="half-extents"):
        save_custom_zones(tmp_path, "m", [_rect("A", half_x=2.0)], reserved=set())
    with pytest.raises(ValueError, match="half-extents"):
        save_custom_zones(tmp_path, "m", [_rect("A", half_y=700.0)], reserved=set())
    with pytest.raises(ValueError, match="level"):
        save_custom_zones(tmp_path, "m", [_rect("A", level="attic")], reserved=set())
    # Tiny hide-spot rects are legal down to a 5-unit half-extent.
    saved = save_custom_zones(
        tmp_path, "hide", [_rect("A", half_x=5.0, half_y=5.0)], reserved=set()
    )
    assert saved[0].half_x == 5.0


def test_rect_overlap_matrix(tmp_path: Path):
    # Footprints intersect -> rejected.
    with pytest.raises(ValueError, match="overlap"):
        save_custom_zones(
            tmp_path, "m", [_rect("A"), _rect("B", x=250.0, half_x=80.0)], reserved=set()
        )
    # Apart in XY -> fine.
    save_custom_zones(tmp_path, "m", [_rect("A"), _rect("B", x=400.0)], reserved=set())
    # Stacked levels never conflict: same footprint, 700 units apart.
    save_custom_zones(
        tmp_path, "m", [_rect("A"), _rect("B", z=-700.0, level="lower")], reserved=set()
    )
