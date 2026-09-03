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
