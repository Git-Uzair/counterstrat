"""One-time callout anchor calibration from operator-curated demos.

Usage::

    uv run python -m counterstrat.mapcard.calibrate [demo_dir] [--limit N]

Scans ``demo_dir`` (default ``data/calibration``) for demos - plain ``.dem``
or ``.zst``/``.gz``/``.bz2`` archives - deduplicates them by content hash,
extracts each demo's occupancy ticks once into ``<demo_dir>/.cache/`` (so the
run is resumable and incremental), then pools every demo of a map and writes
the shipped anchor file ``src/counterstrat/mapcard/anchors/<map>.json``.

These shipped anchors are the calibration the app serves forever after;
user-uploaded demos never move a callout label (see
``counterstrat.mapcard.anchors``). Radar calibration (world -> image) comes
from the cached radar assets in ``<data_root>/radar/<map>/`` or, when absent,
a live extraction from the configured CS2 install.
"""

import argparse
import bz2
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import polars as pl
import zstandard

from counterstrat.mapcard.anchors import (
    SHIPPED_ANCHORS_DIR,
    compute_tick_anchors,
    compute_zone_bounds,
    save_shipped_anchors,
    shipped_kills_path,
)

logger = logging.getLogger(__name__)

TICK_STEP = 4  # mirror lake extraction sampling
CALIBRATION_PROPS = ["X", "Y", "Z", "last_place_name", "is_alive"]
DEMO_SUFFIXES = (".dem", ".zst", ".gz", ".bz2")


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _decompress(src: Path, work_dir: Path) -> Path:
    """A plain .dem path for the parser; archives inflate into work_dir."""
    suffix = src.suffix.lower()
    if suffix == ".dem":
        return src
    dest = work_dir / (src.stem if src.stem.endswith(".dem") else f"{src.stem}.dem")
    work_dir.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        if suffix == ".zst":
            with src.open("rb") as f:
                zstandard.ZstdDecompressor().copy_stream(f, out)
        elif suffix == ".gz":
            with gzip.open(src, "rb") as f:
                while chunk := f.read(1 << 20):
                    out.write(chunk)
        elif suffix == ".bz2":
            with bz2.open(src, "rb") as f:
                while chunk := f.read(1 << 20):
                    out.write(chunk)
        else:
            raise ValueError(f"Unsupported demo archive: {src.name}")
    return dest


def _occupancy_frame(dem_path: Path) -> tuple[str, pl.DataFrame]:
    """(map_name, reduced occupancy ticks) for one demo."""
    from demoparser2 import DemoParser

    parser = DemoParser(str(dem_path))
    header = parser.parse_header()
    map_name = str(header["map_name"])

    tick_args: dict = {}
    try:
        starts = parser.parse_event("round_start")
        ends = parser.parse_event("round_end")
        first = int(starts["tick"].min())
        last = int(ends["tick"].max())
        if last > first:
            tick_args["ticks"] = list(range(first, last, TICK_STEP))
    except Exception:  # noqa: BLE001 - no round events -> parse everything
        logger.warning("%s: no round events; sampling the whole demo", dem_path.name)

    df = pl.from_pandas(parser.parse_ticks(CALIBRATION_PROPS, **tick_args))
    if not tick_args and "tick" in df.columns:
        df = df.filter(pl.col("tick") % TICK_STEP == 0)
    df = df.filter(
        pl.col("is_alive")
        & pl.col("last_place_name").is_not_null()
        & (pl.col("last_place_name") != "")
    ).drop_nulls(["X", "Y", "Z"])
    return map_name, df.select(["X", "Y", "Z", "last_place_name"])


def _kills_frame(dem_path: Path) -> pl.DataFrame:
    """Kill endpoints (places + positions + distance + clean-shot flags).

    Positions ship so sightlines can be re-derived under any future custom
    callout vocabulary: an endpoint inside a user's rect re-labels to it.
    """
    from demoparser2 import DemoParser

    from counterstrat.constants import UNITS_PER_METER

    parser = DemoParser(str(dem_path))
    df = pl.from_pandas(
        parser.parse_event("player_death", player=["last_place_name", "X", "Y", "Z"])
    )
    needed = {
        "attacker_last_place_name",
        "user_last_place_name",
        "attacker_X",
        "attacker_Y",
        "attacker_Z",
        "user_X",
        "user_Y",
        "user_Z",
    }
    if df.is_empty() or not needed <= set(df.columns):
        return pl.DataFrame()

    euclid_m = (
        (pl.col("attacker_X") - pl.col("user_X")) ** 2
        + (pl.col("attacker_Y") - pl.col("user_Y")) ** 2
        + (pl.col("attacker_Z") - pl.col("user_Z")) ** 2
    ).sqrt() / UNITS_PER_METER
    distance = pl.col("distance").fill_null(euclid_m) if "distance" in df.columns else euclid_m
    out = df.select(
        pl.col("attacker_last_place_name").alias("attacker_place"),
        pl.col("user_last_place_name").alias("victim_place"),
        pl.col("attacker_X").alias("attacker_x"),
        pl.col("attacker_Y").alias("attacker_y"),
        pl.col("attacker_Z").alias("attacker_z"),
        pl.col("user_X").alias("victim_x"),
        pl.col("user_Y").alias("victim_y"),
        pl.col("user_Z").alias("victim_z"),
        distance.cast(pl.Float64).alias("distance"),
        *(pl.col(c) for c in ("thrusmoke", "penetrated") if c in df.columns),
    )
    return out.drop_nulls(["attacker_x", "victim_x", "distance"])


def _load_index(cache_dir: Path) -> dict:
    index_path = cache_dir / "index.json"
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Torn cache index; starting over")
    return {}


def _save_index(cache_dir: Path, index: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def _radar_calibration(data_root: Path, map_name: str):
    from counterstrat.config import AppConfig
    from counterstrat.radar.extract import extract_radar_assets, load_cached_assets
    from counterstrat.web.ingest import _find_vrf_cli

    assets = load_cached_assets(data_root, map_name)
    if assets is not None:
        return assets.calibration
    cfg = AppConfig.load()
    vrf = _find_vrf_cli()
    if cfg.cs2_install_path and vrf is not None:
        try:
            return extract_radar_assets(cfg.cs2_install_path, vrf, map_name, data_root).calibration
        except Exception as exc:  # noqa: BLE001
            logger.warning("Radar extraction failed for %s: %s", map_name, exc)
    return None


def _supported_maps() -> set[str]:
    from counterstrat.constants import RETIRED_MAPS
    from counterstrat.web.ingest import REPO_ROOT

    maps_dir = REPO_ROOT / "maps"
    if not maps_dir.exists():
        return set()
    return {
        d.name for d in maps_dir.iterdir() if d.is_dir() and (d / f"{d.name}.vpk").exists()
    } - RETIRED_MAPS


def run(demo_dir: Path, data_root: Path, out_dir: Path, limit: int | None = None) -> dict:
    """Process demos (resumable), then write anchors per fully-cached map."""
    cache_dir = demo_dir / ".cache"
    work_dir = demo_dir / ".tmp"
    index = _load_index(cache_dir)

    archives = sorted(
        p for p in demo_dir.iterdir() if p.is_file() and p.suffix.lower() in DEMO_SUFFIXES
    )
    print(f"Found {len(archives)} demo file(s) in {demo_dir}")

    seen_hashes: set[str] = set()
    pending: list[tuple[str, Path]] = []
    for arc in archives:
        sha = _sha16(arc)
        if sha in seen_hashes:
            print(f"  skip duplicate content: {arc.name}")
            continue
        seen_hashes.add(sha)
        if sha in index:
            continue
        pending.append((sha, arc))

    todo = pending[:limit] if limit else pending
    for i, (sha, arc) in enumerate(todo, 1):
        t0 = time.time()
        dem = None
        try:
            dem = _decompress(arc, work_dir)
            map_name, frame = _occupancy_frame(dem)
            kills = _kills_frame(dem)
            cache_dir.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache_dir / f"{sha}.parquet")
            kills.write_parquet(cache_dir / f"{sha}.kills.parquet")
            index[sha] = {"file": arc.name, "map": map_name, "rows": frame.height}
            _save_index(cache_dir, index)
            print(
                f"  [{i}/{len(todo)}] {arc.name} -> {map_name} "
                f"({frame.height} ticks, {kills.height} kills, {time.time() - t0:.0f}s)"
            )
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] {arc.name} FAILED: {exc}")
            logger.exception("Calibration parse failed for %s", arc)
        finally:
            if dem is not None and dem != arc and dem.exists():
                dem.unlink()

    # Backfill: demos cached before kill extraction existed get one more pass.
    backfill = [
        (sha, demo_dir / meta["file"])
        for sha, meta in index.items()
        if not (cache_dir / f"{sha}.kills.parquet").exists() and (demo_dir / meta["file"]).exists()
    ]
    for i, (sha, arc) in enumerate(backfill, 1):
        dem = None
        try:
            dem = _decompress(arc, work_dir)
            kills = _kills_frame(dem)
            kills.write_parquet(cache_dir / f"{sha}.kills.parquet")
            print(f"  [kills {i}/{len(backfill)}] {arc.name}: {kills.height} kills")
        except Exception as exc:
            print(f"  [kills {i}/{len(backfill)}] {arc.name} FAILED: {exc}")
            logger.exception("Kill extraction failed for %s", arc)
        finally:
            if dem is not None and dem != arc and dem.exists():
                dem.unlink()

    remaining = len(pending) - len(todo)
    if remaining > 0:
        print(f"{remaining} demo(s) still pending - rerun to continue; anchors not written yet")
        return {"pending": remaining}

    # Pool per map and write shipped anchors.
    by_map: dict[str, list[str]] = {}
    for sha, meta in index.items():
        if (cache_dir / f"{sha}.parquet").exists():
            by_map.setdefault(meta["map"], []).append(sha)

    written: dict[str, int] = {}
    uncalibratable: list[str] = []
    for map_name in sorted(by_map):
        shas = by_map[map_name]
        cal = _radar_calibration(data_root, map_name)
        if cal is None:
            uncalibratable.append(map_name)
            print(f"{map_name}: NO radar calibration (cs2_install_path unset?) - skipped")
            continue
        pooled = pl.concat(
            [pl.read_parquet(cache_dir / f"{sha}.parquet") for sha in shas]
        ).with_columns(pl.lit(value=True).alias("is_alive"))
        anchors = compute_tick_anchors(pooled, cal)
        if not anchors:
            uncalibratable.append(map_name)
            print(f"{map_name}: no anchors computable from {len(shas)} demo(s)")
            continue
        path = save_shipped_anchors(
            map_name,
            anchors,
            generated_from=[index[sha]["file"] for sha in shas],
            bounds=compute_zone_bounds(pooled, cal),
            root=out_dir,
        )
        # Ship the raw kill endpoints beside the anchors: sightlines derive
        # from them at serve time under the user's live callout vocabulary.
        kill_frames = [
            pl.read_parquet(cache_dir / f"{sha}.kills.parquet")
            for sha in shas
            if (cache_dir / f"{sha}.kills.parquet").exists()
        ]
        kill_frames = [k for k in kill_frames if not k.is_empty()]
        n_kills = 0
        if kill_frames:
            pooled_kills = pl.concat(kill_frames, how="vertical_relaxed")
            pooled_kills.write_parquet(shipped_kills_path(map_name, out_dir))
            n_kills = pooled_kills.height
        written[map_name] = len(anchors)
        print(
            f"{map_name}: {len(anchors)} zone anchors from {len(shas)} demo(s) "
            f"({pooled.height} pooled ticks, {n_kills} kills) -> {path}"
        )

    supported = _supported_maps()
    missing = sorted(supported - set(written))
    print(f"\nCalibrated: {', '.join(sorted(written)) or '(none)'}")
    if missing:
        print(f"Missing (supported by the app, no calibration): {', '.join(missing)}")
    return {"written": written, "missing": missing, "uncalibratable": uncalibratable}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("demo_dir", nargs="?", default="data/calibration", type=Path)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--out", default=SHIPPED_ANCHORS_DIR, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="demos to process this run")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if not args.demo_dir.exists():
        print(f"Demo directory {args.demo_dir} does not exist")
        return 1
    run(args.demo_dir, args.data_root, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
