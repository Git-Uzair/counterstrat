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


def _ticks() -> pl.DataFrame:
    # Two game zones; one tick sits inside the future "Sandbags" sphere,
    # one directly above it on a roof (same XY, Z +200), one far away.
    return pl.DataFrame(
        {
            "X": [100.0, 100.0, 900.0],
            "Y": [100.0, 100.0, 900.0],
            "Z": [0.0, 200.0, 0.0],
            "last_place_name": ["Middle", "MidRoof", "BombsiteA"],
            "is_alive": [True, True, True],
        }
    )


def test_rezone_assigns_inside_preserves_default_and_restores():
    zones = [CustomZone(name="Sandbags", x=110.0, y=110.0, z=0.0, radius=50.0)]
    out = rezone_ticks(_ticks(), zones)
    assert out["last_place_name"].to_list() == ["Sandbags", "MidRoof", "BombsiteA"]
    # The game vocabulary survives in place_default.
    assert out["place_default"].to_list() == ["Middle", "MidRoof", "BombsiteA"]
    # Idempotent: re-zoning re-zoned ticks changes nothing.
    again = rezone_ticks(out, zones)
    assert again["last_place_name"].to_list() == out["last_place_name"].to_list()
    # Reversible: an empty zone list restores the game names.
    restored = rezone_ticks(out, [])
    assert restored["last_place_name"].to_list() == ["Middle", "MidRoof", "BombsiteA"]


def test_rezone_z_scaled_membership_excludes_stacked_zones():
    # Radius 150 covers XY distance 14 but not the roof 200 above:
    # scaled dz = 2*200 = 400 > 150. Nuke Hut/HutRoof stay distinct.
    zones = [CustomZone(name="Sandbags", x=110.0, y=110.0, z=0.0, radius=150.0)]
    out = rezone_ticks(_ticks(), zones)
    assert out["last_place_name"].to_list() == ["Sandbags", "MidRoof", "BombsiteA"]


def test_rezone_without_coords_or_rows_is_a_noop():
    empty = pl.DataFrame({"last_place_name": [], "X": [], "Y": [], "Z": []})
    assert rezone_ticks(empty, [CustomZone(name="A", x=0, y=0, z=0)]).is_empty()
    no_coords = pl.DataFrame({"last_place_name": ["Middle"]})
    out = rezone_ticks(no_coords, [CustomZone(name="A", x=0, y=0, z=0)])
    assert out["last_place_name"].to_list() == ["Middle"]


def _rect(name: str = "Sandbags", **kw) -> CustomZone:
    base = {"x": 100.0, "y": 100.0, "z": 0.0, "shape": "rect", "half_x": 80.0, "half_y": 120.0}
    base.update(kw)
    return CustomZone(name=name, **base)


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
    # Reversible like spheres.
    assert rezone_ticks(out, [])["last_place_name"].to_list()[0] == "Mid"


def test_rect_validation(tmp_path: Path):
    ok = {"x": 0.0, "y": 0.0, "z": 0.0}
    with pytest.raises(ValueError, match="half_x"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="A", shape="rect", **ok)], reserved=set())
    with pytest.raises(ValueError, match="half-extents"):
        save_custom_zones(
            tmp_path,
            "m",
            [CustomZone(name="A", shape="rect", half_x=2.0, half_y=100.0, **ok)],
            reserved=set(),
        )
    # Tiny hide-spot rects are legal down to a 5-unit half-extent.
    save_custom_zones(
        tmp_path,
        "hide",
        [CustomZone(name="A", shape="rect", half_x=5.0, half_y=5.0, **ok)],
        reserved=set(),
    )
    # A valid rect saves and round-trips with its shape.
    saved = save_custom_zones(tmp_path, "m", [_rect()], reserved=set())
    assert saved[0].shape == "rect" and saved[0].half_x == 80.0
    loaded = load_custom_zones(tmp_path, "m")
    assert loaded[0].shape == "rect" and loaded[0].half_y == 120.0
    # Legacy sphere entries (no shape key) still load as spheres.
    (tmp_path / "mapcards" / "m2").mkdir(parents=True)
    (tmp_path / "mapcards" / "m2" / "zones.json").write_text(
        '[{"name": "Old", "x": 1.0, "y": 2.0, "z": 3.0, "radius": 100.0}]', encoding="utf-8"
    )
    legacy = load_custom_zones(tmp_path, "m2")
    assert legacy[0].shape == "sphere" and legacy[0].radius == 100.0


def test_rect_overlap_matrix(tmp_path: Path):
    # rect-rect: footprints intersect -> rejected.
    with pytest.raises(ValueError, match="overlap"):
        save_custom_zones(
            tmp_path, "m", [_rect("A"), _rect("B", x=250.0, half_x=80.0)], reserved=set()
        )
    # rect-rect apart in XY -> fine.
    save_custom_zones(tmp_path, "m", [_rect("A"), _rect("B", x=400.0)], reserved=set())
    # rect-sphere: sphere edge reaches the rect footprint -> rejected.
    with pytest.raises(ValueError, match="overlap"):
        save_custom_zones(
            tmp_path,
            "m",
            [_rect("A"), CustomZone(name="B", x=300.0, y=100.0, z=0.0, radius=150.0)],
            reserved=set(),
        )
    # Stacked levels never conflict: same footprint, 700 units apart.
    save_custom_zones(
        tmp_path, "m", [_rect("A"), _rect("B", z=-700.0, level="lower")], reserved=set()
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    z = CustomZone(name="Sandbags", x=1.0, y=2.0, z=3.0, level="lower", radius=100.0)
    saved = save_custom_zones(tmp_path, "de_test", [z], reserved={"Middle"})
    assert [s.name for s in saved] == ["Sandbags"]
    loaded = load_custom_zones(tmp_path, "de_test")
    assert loaded == saved
    # Torn/missing files load as empty, like aliases.
    (tmp_path / "mapcards" / "de_test" / "zones.json").write_text("{not json", encoding="utf-8")
    assert load_custom_zones(tmp_path, "de_test") == []
    assert load_custom_zones(tmp_path, "de_ghost") == []


def test_save_validation_matrix(tmp_path: Path):
    ok = {"x": 0.0, "y": 0.0, "z": 0.0}
    reserved = {"Middle", "Mid"}  # a game zone and an alias value
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="Middle", **ok)], reserved=reserved)
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="Mid", **ok)], reserved=reserved)
    with pytest.raises(ValueError, match="invalid characters"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="a`b", **ok)], reserved=set())
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        save_custom_zones(
            tmp_path,
            "m",
            [CustomZone(name="A", **ok), CustomZone(name="A", x=9000.0, y=9000.0, z=0.0)],
            reserved=set(),
        )
    with pytest.raises(ValueError, match="radius"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="A", radius=20.0, **ok)], reserved=set())
    with pytest.raises(ValueError, match="level"):
        save_custom_zones(
            tmp_path, "m", [CustomZone(name="A", level="attic", **ok)], reserved=set()
        )
    with pytest.raises(ValueError, match="overlap"):
        save_custom_zones(
            tmp_path,
            "m",
            [
                CustomZone(name="A", x=0.0, y=0.0, z=0.0, radius=150.0),
                CustomZone(name="B", x=200.0, y=0.0, z=0.0, radius=150.0),
            ],
            reserved=set(),
        )
    # Same XY but far apart vertically is NOT an overlap (scaled 3D).
    saved = save_custom_zones(
        tmp_path,
        "m",
        [
            CustomZone(name="A", x=0.0, y=0.0, z=0.0, radius=100.0),
            CustomZone(name="B", x=0.0, y=0.0, z=300.0, radius=100.0),
        ],
        reserved=set(),
    )
    assert [s.name for s in saved] == ["A", "B"]
