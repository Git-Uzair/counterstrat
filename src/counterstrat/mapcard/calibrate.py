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


def _rounds_frame(parser) -> pl.DataFrame:
    """round_num/start/freeze_end/end from demoparser2 events (no awpy)."""

    def _ticks_of(event: str) -> pl.Series:
        try:
            df = pl.from_pandas(parser.parse_event(event))
        except Exception:  # noqa: BLE001 - event absent in this demo
            return pl.Series("tick", [], dtype=pl.Int64)
        if df.is_empty() or "tick" not in df.columns:
            return pl.Series("tick", [], dtype=pl.Int64)
        return df["tick"].cast(pl.Int64).unique().sort()

    starts = _ticks_of("round_start")
    ends = _ticks_of("round_end")
    freezes = _ticks_of("round_freeze_end")
    if starts.is_empty() or ends.is_empty():
        return pl.DataFrame()
    rounds = (
        pl.DataFrame({"start": starts})
        .sort("start")
        .with_row_index("round_num", offset=1)
        .join_asof(
            pl.DataFrame({"end": ends}).sort("end"),
            left_on="start",
            right_on="end",
            strategy="forward",
        )
        .join_asof(
            pl.DataFrame({"freeze_end": freezes}).sort("freeze_end"),
            left_on="start",
            right_on="freeze_end",
            strategy="forward",
        )
        .drop_nulls(["end"])
        .with_columns(
            pl.col("freeze_end").fill_null(pl.col("start")),
            pl.col("round_num").cast(pl.Int64),
        )
        # A freeze_end that belongs to the next round must not leak in.
        .filter(pl.col("freeze_end") <= pl.col("end"))
    )
    return rounds


def _extract_frames(dem_path: Path) -> tuple[str, str, pl.DataFrame, pl.DataFrame]:
    """(map_name, patch_version, occupancy ticks, movement ticks) - one parse.

    Occupancy feeds the anchor math (positions only). Movement keeps the
    trajectory keys (steamid/round_num/tick/clock_s) so shipped topologies can
    be measured from the same curated demos (see mapcard.topologies).
    """
    from demoparser2 import DemoParser

    parser = DemoParser(str(dem_path))
    header = parser.parse_header()
    map_name = str(header["map_name"])
    patch = str(header.get("patch_version") or "")

    rounds = _rounds_frame(parser)
    tick_args: dict = {}
    if not rounds.is_empty():
        first = int(rounds["start"].min())
        last = int(rounds["end"].max())
        if last > first:
            tick_args["ticks"] = list(range(first, last, TICK_STEP))
    if not tick_args:
        logger.warning("%s: no round events; sampling the whole demo", dem_path.name)

    df = pl.from_pandas(parser.parse_ticks(CALIBRATION_PROPS, **tick_args))
    if not tick_args and "tick" in df.columns:
        df = df.filter(pl.col("tick") % TICK_STEP == 0)

    occupancy = (
        df.filter(
            pl.col("is_alive")
            & pl.col("last_place_name").is_not_null()
            & (pl.col("last_place_name") != "")
        )
        .drop_nulls(["X", "Y", "Z"])
        .select(["X", "Y", "Z", "last_place_name"])
    )

    moves = pl.DataFrame()
    if not rounds.is_empty() and {"tick", "steamid"} <= set(df.columns):
        # Mirrors lake extraction: round via backward asof on start, clock_s
        # anchored at freeze end (counterstrat.lake.extract._sample_ticks).
        moves = (
            df.sort("tick")
            .join_asof(
                rounds.select(["round_num", "start", "freeze_end", "end"]).sort("start"),
                left_on="tick",
                right_on="start",
                strategy="backward",
            )
            .filter(pl.col("tick") <= pl.col("end"))
            .with_columns(((pl.col("tick") - pl.col("freeze_end")) / 64.0).alias("clock_s"))
            .select(["steamid", "round_num", "tick", "clock_s", "is_alive", "last_place_name"])
            .drop_nulls(["round_num"])
        )
    return map_name, patch, occupancy, moves


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


def backfill_moves(demo_dir: Path) -> int:
    """Movement caches for demos processed before topology extraction existed.

    Reparses each cached demo once and writes ``<sha>.moves.parquet`` beside
    its occupancy cache (plus the demo's patch_version into the index), so
    ``mapcard.topologies.export_shipped_topologies`` can measure shipped
    move-time graphs from the calibration set. Resumable and idempotent.
    """
    cache_dir = demo_dir / ".cache"
    work_dir = demo_dir / ".tmp"
    index = _load_index(cache_dir)
    pending = [
        (sha, demo_dir / meta["file"])
        for sha, meta in sorted(index.items())
        if not (cache_dir / f"{sha}.moves.parquet").exists() and (demo_dir / meta["file"]).exists()
    ]
    written = 0
    for i, (sha, arc) in enumerate(pending, 1):
        t0 = time.time()
        dem = None
        try:
            dem = _decompress(arc, work_dir)
            _, patch, _, moves = _extract_frames(dem)
            if moves.is_empty():
                print(f"  [moves {i}/{len(pending)}] {arc.name}: no round events, skipped")
                continue
            moves.write_parquet(cache_dir / f"{sha}.moves.parquet")
            index[sha]["patch"] = patch
            _save_index(cache_dir, index)
            written += 1
            print(
                f"  [moves {i}/{len(pending)}] {arc.name}: {moves.height} rows "
                f"({time.time() - t0:.0f}s)"
            )
        except Exception as exc:
            print(f"  [moves {i}/{len(pending)}] {arc.name} FAILED: {exc}")
            logger.exception("Movement extraction failed for %s", arc)
        finally:
            if dem is not None and dem != arc and dem.exists():
                dem.unlink()
    return written


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
            map_name, patch, frame, moves = _extract_frames(dem)
            kills = _kills_frame(dem)
            cache_dir.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(cache_dir / f"{sha}.parquet")
            kills.write_parquet(cache_dir / f"{sha}.kills.parquet")
            if not moves.is_empty():
                moves.write_parquet(cache_dir / f"{sha}.moves.parquet")
            index[sha] = {"file": arc.name, "map": map_name, "rows": frame.height, "patch": patch}
            _save_index(cache_dir, index)
            print(
                f"  [{i}/{len(todo)}] {arc.name} -> {map_name} "
                f"({frame.height} ticks, {moves.height} moves, {kills.height} kills, "
                f"{time.time() - t0:.0f}s)"
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

    backfill_moves(demo_dir)

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
    ap.add_argument(
        "--moves-only",
        action="store_true",
        help="only backfill movement caches (leaves shipped anchors untouched)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if not args.demo_dir.exists():
        print(f"Demo directory {args.demo_dir} does not exist")
        return 1
    if args.moves_only:
        print(f"movement caches written: {backfill_moves(args.demo_dir)}")
        return 0
    run(args.demo_dir, args.data_root, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
