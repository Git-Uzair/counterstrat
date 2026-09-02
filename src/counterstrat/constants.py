"""Game + pipeline constants. Sources: CS2 defaults; spec §6.5 objectives."""

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
