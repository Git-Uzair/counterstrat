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
    u, v, level = anchors["Middle"]
    # Median (10, 10) world -> u=(10+1024)/2048, v=(1024-10)/2048.
    assert abs(u - 1034 / 2048) < 1e-3
    assert abs(v - 1014 / 2048) < 1e-3
    assert level == "default"


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
    u, v, _ = anchors["Middle"]
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
    u, v, level = anchors["Middle"]
    assert level == "lower"
    # Median of the lower ticks' x is 90.5 -> snaps to the tick at (90, 0).
    assert abs(u - 1114 / 2048) < 1e-3
    assert abs(v - 1024 / 2048) < 1e-3


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
    parsed: list[str] = []

    def fake_frame(dem_path: Path):
        parsed.append(dem_path.name)
        return "de_test", frames[dem_path.name]

    monkeypatch.setattr(calibrate, "_occupancy_frame", fake_frame)
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
