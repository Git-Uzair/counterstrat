"""Beat frame generation for RoundScript View A."""

from collections import Counter

import polars as pl

from counterstrat.constants import BEAT_INTERVAL_S
from counterstrat.roundscript.models import Beat, Formation


def formation_from_zones(zones: list[str]) -> Formation:
    """Run-length-compressed formation ordered by count descending then zone_id ascending."""
    counts = Counter(zones)
    sorted_items = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return Formation(zones=[(count, zone) for zone, count in sorted_items])


def _extract_zones(df: pl.DataFrame) -> list[str]:
    """Extract zone strings from last_place_name column, defaulting empty/null to Unknown."""
    if "last_place_name" not in df.columns or df.is_empty():
        return []
    zones: list[str] = []
    for val in df["last_place_name"].to_list():
        if val is None or str(val).strip() == "":
            zones.append("Unknown")
        else:
            zones.append(str(val).strip())
    return zones


def _label_priority(label: str) -> int:
    """Priority for deduplication: event beats take precedence over periodic beats."""
    if label.startswith("FC"):
        return 0
    if label.startswith("PL"):
        return 1
    return 2


def build_beats(
    ticks: pl.DataFrame,
    round_num: int,
    first_contact_t: float | None = None,
    plant_t: float | None = None,
    interval_s: int = BEAT_INTERVAL_S,
) -> list[Beat]:
    """Build periodic and event-anchored beat snapshots for a round."""
    round_ticks = ticks.filter(pl.col("round_num") == round_num)
    if round_ticks.is_empty() or "clock_s" not in round_ticks.columns:
        return []

    valid_clocks = round_ticks["clock_s"].drop_nulls()
    if valid_clocks.is_empty():
        return []
    max_clock_s = float(valid_clocks.max())
    if max_clock_s < 0.0:
        return []

    if interval_s <= 0:
        interval_s = BEAT_INTERVAL_S

    # Target periodic times: [0.0, interval_s, 2*interval_s, ...] while t <= max_clock_s
    candidates: list[tuple[float, str]] = []
    step = 0
    while step * interval_s <= max_clock_s:
        t_val = float(step * interval_s)
        candidates.append((t_val, f"B+{round(t_val):02d}"))
        step += 1

    # Add event times if provided and within range
    if first_contact_t is not None and 0.0 <= first_contact_t <= max_clock_s:
        candidates.append((float(first_contact_t), f"FC+{round(first_contact_t):02d}"))

    if plant_t is not None and 0.0 <= plant_t <= max_clock_s:
        candidates.append((float(plant_t), f"PL+{round(plant_t):02d}"))

    # Sort candidate beats by t, deduplicating identical times (event beats win ties)
    candidates.sort(key=lambda x: (round(x[0], 4), _label_priority(x[1])))
    deduped_candidates: list[tuple[float, str]] = []
    seen_times: set[float] = set()
    for t_val, label in candidates:
        t_key = round(t_val, 4)
        if t_key not in seen_times:
            seen_times.add(t_key)
            deduped_candidates.append((t_val, label))

    # Output beats strictly sorted by timestamp
    deduped_candidates.sort(key=lambda x: x[0])

    beats: list[Beat] = []
    for t_val, label in deduped_candidates:
        diff = (round_ticks["clock_s"] - t_val).abs()
        min_idx = diff.arg_min()
        if min_idx is None:
            continue

        if "tick" in round_ticks.columns:
            closest_tick = round_ticks["tick"][min_idx]
            tick_rows = round_ticks.filter(pl.col("tick") == closest_tick)
        else:
            closest_clock = round_ticks["clock_s"][min_idx]
            tick_rows = round_ticks.filter(pl.col("clock_s") == closest_clock)

        if "is_alive" in tick_rows.columns:
            alive_rows = tick_rows.filter(pl.col("is_alive"))
        else:
            alive_rows = tick_rows

        if "team_name" in alive_rows.columns:
            t_rows = alive_rows.filter(pl.col("team_name").is_in(["TERRORIST", "T"]))
            ct_rows = alive_rows.filter(pl.col("team_name").is_in(["CT"]))
        else:
            t_rows = pl.DataFrame(schema=alive_rows.schema)
            ct_rows = pl.DataFrame(schema=alive_rows.schema)

        t_places = _extract_zones(t_rows)
        ct_places = _extract_zones(ct_rows)

        t_form = formation_from_zones(t_places)
        ct_form = formation_from_zones(ct_places)

        beats.append(
            Beat(
                label=label,
                t=round(t_val, 1),
                t_form=t_form,
                ct_form=ct_form,
            )
        )

    return beats
