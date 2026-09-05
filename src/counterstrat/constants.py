"""Game + pipeline constants. Sources: CS2 defaults; spec §6.5 objectives."""

# Maps removed from the app (out of the current competitive rotation): they
# never appear in the map list, cannot be ingested, and the calibration
# coverage report ignores them - even when their VPK dirs are still on disk.
RETIRED_MAPS = frozenset({"de_overpass", "de_train", "de_vertigo"})

TICK_RATE = 64
ROUND_SECONDS = 115  # 1:55
BOMB_SECONDS = 40
DEFUSE_SECONDS = (10, 5)  # (no kit, kit)
TRADE_WINDOW_S = 4.0  # spec §6.2 timeline: traded_within_4s
BEAT_INTERVAL_S = 15  # spec §6.6 View A; re-decided in Task 15 gate
TICKS_HZ = 4  # lake tick sampling rate (spec Phase 1)

# Team freeze-end equipment value bins (MR12; calibrate in Task 12 gate).
# Mirrors awpy's CS:GO-era bins, team-of-5 totals.
BUY_BINS = {  # upper bounds, USD equip value
    "full_eco": 5_000,
    "semi_eco": 10_000,
    "semi_buy": 20_000,
    "full_buy": float("inf"),
}

# Engagement-range bands over kill distance in METERS - awpy's kills.distance
# unit, verified empirically: euclid(attacker_XYZ, victim_XYZ) / distance is
# uniformly ~39.37 (inches per meter; 1 Hammer unit = 1 inch). Thresholds from
# the space-vision research §3: SMG/pistol falloff and spread dominate under
# ~15m; scoped rifles and AWPs own lines past ~35m. Shared by the doctrine
# text, the range-profile miner, and sightline labels - one source.
RANGE_CLOSE_M = 15.0  # below: "close"
RANGE_LONG_M = 35.0  # above: "long"; between: "medium"
UNITS_PER_METER = 39.3701  # 1 Hammer unit = 1 inch; tick XYZ -> meters


def range_band(distance_m: float) -> str:
    """close / medium / long label for one kill distance in meters."""
    if distance_m < RANGE_CLOSE_M:
        return "close"
    if distance_m > RANGE_LONG_M:
        return "long"
    return "medium"
