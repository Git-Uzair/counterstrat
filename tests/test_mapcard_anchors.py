"""Shipped callout anchors: the calibration math and the load/save roundtrip."""

from pathlib import Path

import polars as pl

from counterstrat.mapcard.anchors import (
    compute_tick_anchors,
    load_shipped_anchors,
    save_shipped_anchors,
)
from counterstrat.radar.extract import RadarCalibration


def _cal(lower_max: float | None = None) -> RadarCalibration:
    return RadarCalibration(
        map_name="de_test",
        pos_x=-1024.0,
        pos_y=1024.0,
        scale=2.0,
        image_px=1024,
        lower_altitude_max=lower_max,
    )


def _frame(rows: dict) -> pl.DataFrame:
    df = pl.DataFrame(rows)
    if "is_alive" not in df.columns:
        df = df.with_columns(pl.lit(True).alias("is_alive"))
    return df


def test_anchor_is_median_of_occupancy_mass():
    df = _frame(
        {
            "X": [0.0, 10.0, 20.0],
            "Y": [0.0, 10.0, 20.0],
            "Z": [0.0, 0.0, 0.0],
            "last_place_name": ["Middle"] * 3,
        }
    )
    anchors = compute_tick_anchors(df, _cal())
    u, v, level, z = anchors["Middle"]
    # Median (10, 10) world -> u=(10+1024)/2048, v=(1024-10)/2048.
    assert abs(u - 1034 / 2048) < 1e-3
    assert abs(v - 1014 / 2048) < 1e-3
    assert level == "default"
    assert z == 0.0  # the snapped tick's ground Z ships with the anchor


def test_anchor_snaps_to_occupied_ground_not_ring_center():
    """Ring/L-shaped zones: a raw per-axis median can land where nobody ever
    stands; the anchor must snap to a real tick so the label is on-zone."""
    df = _frame(
        {
            "X": [100.0, -100.0, 0.0, 10.0, 80.0],
            "Y": [0.0, 10.0, 120.0, -100.0, 80.0],
            "Z": [0.0] * 5,
            "last_place_name": ["Middle"] * 5,
        }
    )
    anchors = compute_tick_anchors(df, _cal())
    u, v = anchors["Middle"][0], anchors["Middle"][1]
    # Per-axis median (10, 10) is the unoccupied ring center; the closest real
    # tick is (100, 0) and that is where the label must sit.
    assert abs(u - 1124 / 2048) < 1e-3
    assert abs(v - 1024 / 2048) < 1e-3


def test_anchor_follows_zones_dominant_level():
    """A zone straddling nuke's two levels labels the level most of its ticks
    are on, and its anchor snaps to ground on THAT level."""
    df = _frame(
        {
            "X": [0.0, 60.0, 90.0, 91.0, 320.0, 350.0, 200.0, 205.0, 210.0],
            "Y": [0.0] * 9,
            "Z": [-600.0] * 6 + [0.0] * 3,
            "last_place_name": ["Middle"] * 9,
        }
    )
    anchors = compute_tick_anchors(df, _cal(lower_max=-450.0))
    u, v, level, z = anchors["Middle"]
    assert level == "lower"
    # Median of the lower ticks' x is 90.5 -> snaps to the tick at (90, 0).
    assert abs(u - 1114 / 2048) < 1e-3
    assert abs(v - 1024 / 2048) < 1e-3
    assert z == -600.0  # ground Z of the dominant (lower) level's snap tick


def test_compute_ignores_dead_unnamed_and_offmap_ticks():
    df = pl.DataFrame(
        {
            "X": [0.0, 5.0, 99999.0],
            "Y": [0.0, 5.0, 99999.0],
            "Z": [0.0] * 3,
            "last_place_name": ["Mid", None, "OffMap"],
            "is_alive": [False, True, True],
        }
    )
    # Mid's only tick is a dead player; the null name never counts; OffMap
    # projects outside the radar image and is dropped.
    assert compute_tick_anchors(df, _cal()) == {}


def test_compute_empty_or_missing_columns():
    assert compute_tick_anchors(pl.DataFrame(), _cal()) == {}
    assert compute_tick_anchors(pl.DataFrame({"X": [1.0]}), _cal()) == {}


def test_calibrate_decompress_roundtrip(tmp_path: Path):
    import gzip

    import zstandard

    from counterstrat.mapcard.calibrate import _decompress

    payload = b"PBDEMS2\0" + b"x" * 256
    (tmp_path / "plain.dem").write_bytes(payload)
    (tmp_path / "a.dem.zst").write_bytes(zstandard.ZstdCompressor().compress(payload))
    (tmp_path / "b.dem.gz").write_bytes(gzip.compress(payload))

    work = tmp_path / "work"
    assert _decompress(tmp_path / "plain.dem", work) == tmp_path / "plain.dem"
    for name in ("a.dem.zst", "b.dem.gz"):
        out = _decompress(tmp_path / name, work)
        assert out.suffix == ".dem" and out.read_bytes() == payload


def test_calibrate_run_dedupes_resumes_and_writes(tmp_path: Path, monkeypatch):
    """Duplicate archives fold; processed demos are cached and skipped on the
    next run; pooled anchors land in the out dir with provenance."""
    from counterstrat.mapcard import calibrate
    from counterstrat.radar.extract import RadarCalibration

    demo_dir = tmp_path / "demos"
    demo_dir.mkdir()
    (demo_dir / "one.dem").write_bytes(b"PBDEMS2\0" + b"a" * 64)
    (demo_dir / "one_copy.dem").write_bytes(b"PBDEMS2\0" + b"a" * 64)  # same content
    (demo_dir / "two.dem").write_bytes(b"PBDEMS2\0" + b"b" * 64)

    frames = {
        "one.dem": pl.DataFrame(
            {"X": [0.0, 10.0], "Y": [0.0, 10.0], "Z": [0.0, 0.0], "last_place_name": ["Mid", "Mid"]}
        ),
        "two.dem": pl.DataFrame({"X": [20.0], "Y": [20.0], "Z": [0.0], "last_place_name": ["Mid"]}),
    }
    moves = pl.DataFrame(
        {
            "steamid": [1],
            "round_num": [1],
            "tick": [64],
            "clock_s": [1.0],
            "is_alive": [True],
            "last_place_name": ["Mid"],
            "team_name": ["CT"],  # current moves schema: shipped-card timings need the side
        }
    )
    parsed: list[str] = []

    def fake_frame(dem_path: Path):
        parsed.append(dem_path.name)
        return "de_test", "14000", frames[dem_path.name], moves

    monkeypatch.setattr(calibrate, "_extract_frames", fake_frame)
    monkeypatch.setattr(calibrate, "_kills_frame", lambda _dem: pl.DataFrame())
    monkeypatch.setattr(
        calibrate,
        "_radar_calibration",
        lambda _root, _map: RadarCalibration(
            map_name="de_test",
            pos_x=-1024.0,
            pos_y=1024.0,
            scale=2.0,
            image_px=1024,
            lower_altitude_max=None,
        ),
    )
    monkeypatch.setattr(calibrate, "_supported_maps", lambda: {"de_test", "de_missing"})

    out_dir = tmp_path / "shipped"
    result = calibrate.run(demo_dir, tmp_path / "data", out_dir)
    assert sorted(parsed) == ["one.dem", "two.dem"]  # the copy never parsed
    assert result["written"] == {"de_test": 1}
    assert result["missing"] == ["de_missing"]
    # Movement caches ride along for shipped-topology export.
    assert len(list((demo_dir / ".cache").glob("*.moves.parquet"))) == 2

    shipped = load_shipped_anchors("de_test", root=out_dir)
    # Pooled median of (0,0),(10,10),(20,20) is (10,10) -> u=1034/2048.
    u, v, level = shipped["Mid"]
    assert abs(u - 1034 / 2048) < 1e-3 and abs(v - 1014 / 2048) < 1e-3
    assert level == "default"

    # Second run: everything cached, nothing re-parsed, anchors rewritten.
    parsed.clear()
    result2 = calibrate.run(demo_dir, tmp_path / "data", out_dir)
    assert parsed == []
    assert result2["written"] == {"de_test": 1}


def test_compute_zone_bounds_boxes_the_occupancy():
    from counterstrat.mapcard.anchors import compute_zone_bounds

    df = _frame(
        {
            "X": [0.0, 25.0, 50.0, 75.0, 100.0],
            "Y": [0.0, 50.0, 100.0, 150.0, 200.0],
            "Z": [0.0] * 5,
            "last_place_name": ["Middle"] * 5,
        }
    )
    bounds = compute_zone_bounds(df, _cal())
    u0, v0, u1, v1 = bounds["Middle"]
    assert 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0
    # The box spans the data: x 0..100 -> u 0.5..~0.549, y 0..200 flips to v.
    assert abs(u0 - 1024 / 2048) < 0.02 and abs(u1 - 1124 / 2048) < 0.02
    assert abs(v0 - 824 / 2048) < 0.02 and abs(v1 - 1024 / 2048) < 0.02


def test_compute_zone_bounds_ignores_other_level_ticks():
    from counterstrat.mapcard.anchors import compute_zone_bounds

    df = _frame(
        {
            "X": [0.0, 50.0, 100.0, 5000.0],
            "Y": [0.0] * 4,
            "Z": [-600.0, -600.0, -600.0, 0.0],  # one stray upper tick far away
            "last_place_name": ["Ramp"] * 4,
        }
    )
    bounds = compute_zone_bounds(df, _cal(lower_max=-450.0))
    _u0, _v0, u1, _v1 = bounds["Ramp"]
    # The dominant (lower) level's box must not stretch toward x=5000.
    assert u1 < 0.6


def test_shipped_bounds_roundtrip(tmp_path: Path):
    from counterstrat.mapcard.anchors import load_shipped_zone_bounds

    anchors = {"Mid": (0.5, 0.5, "default", 10.0)}
    save_shipped_anchors(
        "de_test",
        anchors,
        generated_from=["a.dem"],
        bounds={"Mid": (0.4, 0.4, 0.6, 0.6)},
        root=tmp_path,
    )
    assert load_shipped_zone_bounds("de_test", root=tmp_path) == {"Mid": (0.4, 0.4, 0.6, 0.6)}
    # Files without bounds load as empty, and the anchor loaders still work.
    save_shipped_anchors("de_old", anchors, generated_from=["a.dem"], root=tmp_path)
    assert load_shipped_zone_bounds("de_old", root=tmp_path) == {}
    assert load_shipped_anchors("de_old", root=tmp_path)["Mid"] == (0.5, 0.5, "default")


def _kills_fixture() -> pl.DataFrame:
    # 4 clean kills Middle->Connector (two from inside the future custom rect
    # at (100+-50, 100+-50)), 2 kills Ramp->Site, plus one through-smoke kill
    # that must never count.
    return pl.DataFrame(
        {
            "attacker_place": ["Middle", "Middle", "Middle", "Middle", "Ramp", "Ramp", "Middle"],
            "victim_place": ["Connector"] * 4 + ["Site", "Site", "Connector"],
            "attacker_x": [100.0, 110.0, 400.0, 420.0, 900.0, 910.0, 105.0],
            "attacker_y": [100.0, 90.0, 400.0, 420.0, 900.0, 890.0, 95.0],
            "attacker_z": [0.0] * 7,
            "victim_x": [600.0, 610.0, 620.0, 630.0, 1200.0, 1210.0, 605.0],
            "victim_y": [600.0, 590.0, 610.0, 615.0, 1200.0, 1190.0, 595.0],
            "victim_z": [0.0] * 7,
            "distance": [18.0, 18.5, 20.0, 19.0, 11.0, 11.5, 18.2],
            "thrusmoke": [False, False, False, False, False, False, True],
            "penetrated": [0] * 7,
        }
    )


def test_shipped_sightlines_respect_custom_vocabulary(tmp_path: Path):
    """The shipped evidence is raw endpoints: a custom rect carved out of
    `Middle` re-labels the kills fired from inside it, so the LLM's sightline
    block speaks the user's callout."""
    from counterstrat.customzones import CustomZone, save_custom_zones
    from counterstrat.mapcard.anchors import shipped_kills_path, shipped_sightlines

    shipped = tmp_path / "shipped"
    shipped.mkdir()
    _kills_fixture().write_parquet(shipped_kills_path("de_test", shipped))
    data_root = tmp_path / "data"

    # No custom zones: engine vocabulary, smoke kill excluded (n=4 not 5).
    plain = {(s["from"], s["to"]): s for s in shipped_sightlines(data_root, "de_test", shipped)}
    assert plain[("Connector", "Middle")]["n"] == 4
    assert plain[("Ramp", "Site")]["n"] == 2

    # Carve "headshot" out of Middle around (100, 100): the two kills fired
    # from inside it move to the custom name, the other two stay Middle.
    save_custom_zones(
        data_root,
        "de_test",
        [CustomZone(name="headshot", x=100.0, y=100.0, z=0.0, half_x=50.0, half_y=50.0)],
        reserved=set(),
    )
    custom = {(s["from"], s["to"]): s for s in shipped_sightlines(data_root, "de_test", shipped)}
    assert custom[("Connector", "headshot")]["n"] == 2
    assert custom[("Connector", "Middle")]["n"] == 2
    assert custom[("Ramp", "Site")]["n"] == 2

    # Unshipped map: no evidence, no crash.
    assert shipped_sightlines(data_root, "de_ghost", shipped) == []


def test_shipped_anchor_roundtrip(tmp_path: Path):
    anchors = {"BombsiteA": (0.25, 0.75, "default"), "Ramp": (0.5, 0.5, "lower")}
    path = save_shipped_anchors(
        "de_test", anchors, generated_from=["a.dem", "b.dem"], root=tmp_path
    )
    assert path == tmp_path / "de_test.json"
    assert load_shipped_anchors("de_test", root=tmp_path) == anchors
    # Absent and torn files degrade to empty.
    assert load_shipped_anchors("de_ghost", root=tmp_path) == {}
    path.write_text("{torn", encoding="utf-8")
    assert load_shipped_anchors("de_test", root=tmp_path) == {}


def test_shipped_anchor_points_carry_ground_z(tmp_path: Path):
    with_z = {"Mid": (0.5, 0.5, "default", 64.0), "Ramp": (0.2, 0.2, "lower", -600.0)}
    save_shipped_anchors("de_test", with_z, generated_from=["a.dem"], root=tmp_path)
    from counterstrat.mapcard.anchors import load_shipped_anchor_points

    points = load_shipped_anchor_points("de_test", root=tmp_path)
    assert points == with_z
    # The plain loader keeps serving 3-tuples for the editor/prompts.
    assert load_shipped_anchors("de_test", root=tmp_path)["Mid"] == (0.5, 0.5, "default")
    # Legacy files without z: points loader skips, plain loader still works.
    save_shipped_anchors(
        "de_old", {"A": (0.1, 0.1, "default")}, generated_from=["b.dem"], root=tmp_path
    )
    assert load_shipped_anchor_points("de_old", root=tmp_path) == {}
    assert load_shipped_anchors("de_old", root=tmp_path)["A"] == (0.1, 0.1, "default")
