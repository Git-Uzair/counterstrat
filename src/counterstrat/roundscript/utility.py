"""View C utility grammar: grenade lines, corpus lineup clusters, subdivision gate.

Spec §6.6 View C (`U+12 donk smoke TSpawn>Window [lineup Window-S1]`) and §6.7-2/-4
(the subdivision gate and the DBSCAN epsilon sweep).
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from counterstrat.constants import TICK_RATE
from counterstrat.lake.duck import TABLES
from counterstrat.lake.extract import LakePaths
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.models import UtilEvent
from counterstrat.roundscript.movement import _normalize_side

# nade vocabulary (spec §6.6 View C) -> cluster-name initial
NADE_INITIALS = {"smoke": "S", "flash": "F", "molly": "M", "he": "H"}

# awpy `grenades` trajectory rows: only the *Projectile* types carry coordinates.
PROJECTILE_NADES = {"CFlashbangProjectile": "flash", "CHEGrenadeProjectile": "he"}

BLIND_JOIN_TICKS = 64  # player_blind rows within +-1s of detonation
Z_SCALE = 2.0  # verticality weight, mirrors ZoneMapper.z_scale
XYZ_COLS = ("origin_x", "origin_y", "origin_z", "landing_x", "landing_y", "landing_z")
XYZ_SCHEMA = {c: pl.Float64 for c in XYZ_COLS}
DEFAULT_EPS = 150.0
DEFAULT_MIN_SAMPLES = 3
EPS_SWEEP = (50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 250.0, 300.0)


def _scan_round(path: str, round_num: int, extra: pl.Expr | None = None) -> pl.DataFrame:
    """Read one round out of a lake parquet, tolerating missing/empty tables."""
    if not path or not Path(path).exists():
        return pl.DataFrame()
    lf = pl.scan_parquet(path)
    if "round_num" not in lf.collect_schema().names():
        return pl.DataFrame()
    lf = lf.filter(pl.col("round_num") == round_num)
    if extra is not None:
        lf = lf.filter(extra)
    return lf.collect()


def _freeze_end(lake: LakePaths, round_num: int) -> int | None:
    rounds = _scan_round(lake.rounds, round_num)
    if rounds.is_empty() or "freeze_end" not in rounds.columns:
        return None
    return int(rounds["freeze_end"][0])


def _clock_s(tick: Any, freeze_end: int | None) -> float:
    if tick is None or freeze_end is None:
        return 0.0
    return (float(tick) - float(freeze_end)) / float(TICK_RATE)


def _sides_by_steamid(lake: LakePaths, round_num: int) -> dict[str, str]:
    """steamid (string) -> "T"/"CT" for one round, from the lake rosters table."""
    rosters = _scan_round(lake.rosters, round_num)
    out: dict[str, str] = {}
    if rosters.is_empty() or "steamids" not in rosters.columns:
        return out
    for row in rosters.iter_rows(named=True):
        side = _normalize_side(row.get("side"))
        for sid in row.get("steamids") or []:
            out[str(sid)] = side
    return out


def _blind_rows(lake: LakePaths) -> pl.DataFrame:
    """The whole player_blind table (207 rows on the fixture demo); tick-joined later."""
    if not lake.player_blind or not Path(lake.player_blind).exists():
        return pl.DataFrame()
    df = pl.read_parquet(lake.player_blind)
    needed = {"attacker_steamid", "user_name", "blind_duration", "tick"}
    if df.is_empty() or not needed <= set(df.columns):
        return pl.DataFrame()
    return df


def _blinded_for(
    blinds: pl.DataFrame, thrower_steamid: str, det_tick: int
) -> list[tuple[str, float]]:
    if blinds.is_empty():
        return []
    hits = blinds.filter(
        (pl.col("attacker_steamid").cast(pl.String) == thrower_steamid)
        & ((pl.col("tick").cast(pl.Int64) - det_tick).abs() <= BLIND_JOIN_TICKS)
    )
    victims = [
        (str(r["user_name"]), round(float(r["blind_duration"]), 2))
        for r in hits.iter_rows(named=True)
        if r["user_name"] is not None and r["blind_duration"] is not None
    ]
    victims.sort(key=lambda v: (-v[1], v[0]))
    return victims


def _bloom_rows(
    df: pl.DataFrame,
    nade: str,
    mapper: ZoneMapper,
    freeze_end: int | None,
) -> list[dict[str, Any]]:
    """Rows from the awpy `smokes`/`infernos` tables (thrower_* origin, X/Y/Z landing)."""
    to_zones = mapper.zones(df, "X", "Y", "Z").to_list()
    from_mapped = mapper.zones(df, "thrower_X", "thrower_Y", "thrower_Z").to_list()
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(df.iter_rows(named=True)):
        place = str(row.get("thrower_place") or "").strip()
        rows.append(
            {
                "t": _clock_s(row.get("start_tick"), freeze_end),
                "thrower": str(row.get("thrower_name") or ""),
                "side": _normalize_side(row.get("thrower_side")),
                "nade": nade,
                "from_zone": place or from_mapped[i],
                "to_zone": to_zones[i],
                "blinded": [],
                "origin_x": float(row["thrower_X"]),
                "origin_y": float(row["thrower_Y"]),
                "origin_z": float(row["thrower_Z"]),
                "landing_x": float(row["X"]),
                "landing_y": float(row["Y"]),
                "landing_z": float(row["Z"]),
            }
        )
    return rows


def _projectile_rows(
    lake: LakePaths,
    round_num: int,
    mapper: ZoneMapper,
    freeze_end: int | None,
    sides: dict[str, str],
) -> list[dict[str, Any]]:
    """Flashes/HEs from grenade trajectories: first tick = throw origin, last = detonation."""
    traj = _scan_round(
        lake.grenades,
        round_num,
        pl.col("grenade_type").is_in(list(PROJECTILE_NADES)) & pl.col("X").is_not_null(),
    )
    if traj.is_empty():
        return []

    per_nade = (
        traj.sort(["entity_id", "tick"])
        .group_by("entity_id", maintain_order=True)
        .agg(
            pl.first("grenade_type").alias("grenade_type"),
            pl.first("thrower").alias("thrower"),
            pl.first("thrower_steamid").alias("thrower_steamid"),
            pl.first("X").alias("origin_x"),
            pl.first("Y").alias("origin_y"),
            pl.first("Z").alias("origin_z"),
            pl.last("X").alias("landing_x"),
            pl.last("Y").alias("landing_y"),
            pl.last("Z").alias("landing_z"),
            pl.last("tick").alias("det_tick"),
        )
    )
    to_zones = mapper.zones(per_nade, "landing_x", "landing_y", "landing_z").to_list()
    from_zones = mapper.zones(per_nade, "origin_x", "origin_y", "origin_z").to_list()
    blinds = _blind_rows(lake)

    rows: list[dict[str, Any]] = []
    for i, row in enumerate(per_nade.iter_rows(named=True)):
        nade = PROJECTILE_NADES[str(row["grenade_type"])]
        steamid = str(row["thrower_steamid"])
        det_tick = int(row["det_tick"])
        rows.append(
            {
                "t": _clock_s(det_tick, freeze_end),
                "thrower": str(row.get("thrower") or ""),
                "side": sides.get(steamid, _normalize_side(None)),
                "nade": nade,
                "from_zone": from_zones[i],
                "to_zone": to_zones[i],
                "blinded": _blinded_for(blinds, steamid, det_tick) if nade == "flash" else [],
                "origin_x": float(row["origin_x"]),
                "origin_y": float(row["origin_y"]),
                "origin_z": float(row["origin_z"]),
                "landing_x": float(row["landing_x"]),
                "landing_y": float(row["landing_y"]),
                "landing_z": float(row["landing_z"]),
            }
        )
    return rows


def utility_events_with_xyz(
    lake: LakePaths,
    mapper: ZoneMapper,
    round_num: int,
) -> tuple[list[UtilEvent], pl.DataFrame]:
    """View C events for one round plus the raw (origin, landing) frame for clustering.

    The frame rows are in the same order as the events, which is what
    :func:`cluster_lineups` requires.
    """
    freeze_end = _freeze_end(lake, round_num)
    sides = _sides_by_steamid(lake, round_num)

    rows: list[dict[str, Any]] = []
    for path, nade in ((lake.smokes, "smoke"), (lake.infernos, "molly")):
        df = _scan_round(path, round_num)
        if not df.is_empty():
            rows.extend(_bloom_rows(df, nade, mapper, freeze_end))
    rows.extend(_projectile_rows(lake, round_num, mapper, freeze_end, sides))

    rows.sort(key=lambda r: (r["t"], r["nade"], r["thrower"], r["to_zone"]))
    events = [
        UtilEvent(
            t=r["t"],
            thrower=r["thrower"],
            side=r["side"],
            nade=r["nade"],
            from_zone=r["from_zone"],
            to_zone=r["to_zone"],
            blinded=r["blinded"],
        )
        for r in rows
    ]
    raw_xyz = pl.DataFrame(
        [{c: r[c] for c in XYZ_COLS} for r in rows],
        schema=XYZ_SCHEMA,
    )
    return events, raw_xyz


def utility_events(lake: LakePaths, mapper: ZoneMapper, round_num: int) -> list[UtilEvent]:
    """View C utility lines for one round, sorted by round clock."""
    return utility_events_with_xyz(lake, mapper, round_num)[0]


def _features(raw_xyz: pl.DataFrame) -> np.ndarray:
    missing = [c for c in XYZ_COLS if c not in raw_xyz.columns]
    if missing:
        raise ValueError(f"raw_xyz missing columns: {missing}")
    return raw_xyz.select(
        [
            pl.col("origin_x").cast(pl.Float64),
            pl.col("origin_y").cast(pl.Float64),
            pl.col("origin_z").cast(pl.Float64) * Z_SCALE,
            pl.col("landing_x").cast(pl.Float64),
            pl.col("landing_y").cast(pl.Float64),
            pl.col("landing_z").cast(pl.Float64) * Z_SCALE,
        ]
    ).to_numpy()


def _modal(values: list[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def _dbscan_labels(
    events: list[UtilEvent],
    feats: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN per nade type (a smoke and a flash are never the same lineup), global labels."""
    labels = np.full(len(events), -1, dtype=int)
    next_label = 0
    for nade in sorted({e.nade for e in events}):
        idx = [i for i, e in enumerate(events) if e.nade == nade]
        local = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats[idx])
        remap: dict[int, int] = {}
        for pos, lab in enumerate(local):
            if lab < 0:
                continue
            if lab not in remap:
                remap[int(lab)] = next_label
                next_label += 1
            labels[idx[pos]] = remap[int(lab)]
    return labels


def cluster_lineups(
    events: list[UtilEvent],
    raw_xyz: pl.DataFrame,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[int, str]:
    """Name (origin, landing) clusters and stamp `lineup_id` onto each event.

    Cluster name is ``{landing_zone}-{nade_initial}{ordinal}`` (e.g. ``Window-S1``);
    ordinals run by cluster size desc then centroid lexicographic order, so the same
    corpus always yields the same names. Noise points get ``lineup_id=None``.
    """
    for e in events:
        e.lineup_id = None
    if not events:
        return {}
    if raw_xyz.height != len(events):
        raise ValueError(f"raw_xyz has {raw_xyz.height} rows for {len(events)} events")

    feats = _features(raw_xyz)
    labels = _dbscan_labels(events, feats, eps, min_samples)

    members: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab >= 0:
            members.setdefault(int(lab), []).append(i)

    # (landing zone, nade initial) -> cluster labels competing for an ordinal
    groups: dict[tuple[str, str], list[int]] = {}
    meta: dict[int, tuple[int, tuple[float, ...]]] = {}
    for lab, idx in members.items():
        zone = _modal([events[i].to_zone for i in idx])
        nade = events[idx[0]].nade
        initial = NADE_INITIALS.get(nade, nade[:1].upper())
        meta[lab] = (len(idx), tuple(feats[idx].mean(axis=0).tolist()))
        groups.setdefault((zone, initial), []).append(lab)

    names: dict[int, str] = {}
    for (zone, initial), labs in groups.items():
        labs.sort(key=lambda lab: (-meta[lab][0], meta[lab][1]))
        for ordinal, lab in enumerate(labs, start=1):
            names[lab] = f"{zone}-{initial}{ordinal}"

    for lab, idx in members.items():
        for i in idx:
            events[i].lineup_id = names[lab]
    return dict(sorted(names.items()))


def subdivision_report(
    events: list[UtilEvent],
    mapper: ZoneMapper,
    out_path: Path | str | None = None,
) -> dict[str, int]:
    """Engine places holding >=2 distinct smoke-landing lineup clusters (spec §6.7-2).

    Requires `cluster_lineups` to have run first. A non-empty result is the measured
    argument for community subdivision of those places in the lexicon overlay.
    """
    known = {str(c) for c in mapper.classes_}
    per_place: dict[str, set[str]] = {}
    for e in events:
        if e.nade != "smoke" or e.lineup_id is None:
            continue
        if known and e.to_zone not in known:
            continue
        per_place.setdefault(e.to_zone, set()).add(e.lineup_id)

    report = {place: len(ids) for place, ids in sorted(per_place.items()) if len(ids) >= 2}
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _lake_from_root(root: Path) -> LakePaths:
    """Rebuild a LakePaths from an `extract_lake` output directory."""
    paths = {t: str(root / f"{t}.parquet") for t in TABLES}
    paths["player_blind"] = str(root / "player_blind.parquet")
    return LakePaths(root=str(root), **paths)


def _corpus_events(lake: LakePaths, mapper: ZoneMapper) -> tuple[list[UtilEvent], pl.DataFrame]:
    rounds = pl.read_parquet(lake.rounds)["round_num"].to_list()
    events: list[UtilEvent] = []
    frames: list[pl.DataFrame] = []
    for rn in rounds:
        evs, xyz = utility_events_with_xyz(lake, mapper, int(rn))
        events.extend(evs)
        frames.append(xyz)
    raw_xyz = pl.concat(frames) if frames else pl.DataFrame(schema=XYZ_SCHEMA)
    return events, raw_xyz


def eps_sweep(
    events: list[UtilEvent],
    raw_xyz: pl.DataFrame,
    eps_values: tuple[float, ...] = EPS_SWEEP,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[dict[str, Any]]:
    """Silhouette + cluster counts per epsilon (spec §6.7-4 unknown; needs a real corpus)."""
    feats = _features(raw_xyz)
    out: list[dict[str, Any]] = []
    for eps in eps_values:
        names = cluster_lineups(events, raw_xyz, eps=eps, min_samples=min_samples)
        labelled = [i for i, e in enumerate(events) if e.lineup_id is not None]
        ids = [events[i].lineup_id for i in labelled]
        silhouette: float | None = None
        if len(set(ids)) >= 2:
            silhouette = round(float(silhouette_score(feats[labelled], ids)), 4)
        out.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "clusters": len(names),
                "labelled": len(labelled),
                "noise": len(events) - len(labelled),
                "silhouette": silhouette,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="lineup clustering dev harness")
    ap.add_argument("--eps-sweep", action="store_true", help="sweep DBSCAN eps and write json")
    ap.add_argument("--lake-root", required=True, help="an extract_lake output directory")
    ap.add_argument("--map", default="de_anubis", dest="map_name")
    ap.add_argument("--out", default=None, help="defaults to data/mapcards/<map>/lineup_sweep.json")
    args = ap.parse_args()
    if not args.eps_sweep:
        ap.error("nothing to do: pass --eps-sweep")

    lake = _lake_from_root(Path(args.lake_root))
    mapper = ZoneMapper.fit(pl.read_parquet(lake.ticks))
    events, raw_xyz = _corpus_events(lake, mapper)
    sweep = eps_sweep(events, raw_xyz)

    out = (
        Path(args.out) if args.out else Path("data/mapcards") / args.map_name / "lineup_sweep.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"map": args.map_name, "n_events": len(events), "sweep": sweep}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(events)} events, {len(sweep)} eps values)")


if __name__ == "__main__":
    main()
