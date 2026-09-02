"""View B movement sentences: per-player zone sequences with dwell compression and event marks."""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from counterstrat.constants import BEAT_INTERVAL_S
from counterstrat.roundscript.models import KillEvent, MovementLine, PlantEvent


@dataclass
class _Visit:
    zone: str
    start_t: float
    end_t: float
    start_tick: int
    end_tick: int
    duration: float
    walk_ticks: int
    total_ticks: int
    is_defusing: bool
    kills: list[str] = field(default_factory=list)
    has_plant: bool = False
    has_defuse: bool = False
    has_death: bool = False


def _merge_visit(a: _Visit, b: _Visit) -> _Visit:
    a.start_t = min(a.start_t, b.start_t)
    a.end_t = max(a.end_t, b.end_t)
    a.start_tick = min(a.start_tick, b.start_tick)
    a.end_tick = max(a.end_tick, b.end_tick)
    a.duration += b.duration
    a.walk_ticks += b.walk_ticks
    a.total_ticks += b.total_ticks
    a.is_defusing = a.is_defusing or b.is_defusing
    a.kills.extend(b.kills)
    a.has_plant = a.has_plant or b.has_plant
    a.has_defuse = a.has_defuse or b.has_defuse
    a.has_death = a.has_death or b.has_death
    return a


def _combine_adjacent_same_zone(visits: list[_Visit]) -> list[_Visit]:
    if not visits:
        return []
    out = [visits[0]]
    for v in visits[1:]:
        if v.zone == out[-1].zone:
            _merge_visit(out[-1], v)
        else:
            out.append(v)
    return out


def _dwell_merge(visits: list[_Visit], min_dwell_s: float) -> list[_Visit]:
    if len(visits) <= 1:
        return visits

    visits = _combine_adjacent_same_zone(visits)
    changed = True
    while changed and len(visits) > 1:
        changed = False

        # 1. Look for a visit < min_dwell_s flanked by the same zone
        flanked_idx = -1
        for i in range(1, len(visits) - 1):
            if visits[i].duration < min_dwell_s and visits[i - 1].zone == visits[i + 1].zone:
                flanked_idx = i
                break

        if flanked_idx != -1:
            short_v = visits.pop(flanked_idx)
            _merge_visit(visits[flanked_idx - 1], short_v)
            visits = _combine_adjacent_same_zone(visits)
            changed = True
            continue

        # 2. More generally, if a visit < min_dwell_s, merge into preceding visit (or following if at start)
        short_idx = -1
        for i, v in enumerate(visits):
            if v.duration < min_dwell_s:
                short_idx = i
                break

        if short_idx != -1:
            short_v = visits.pop(short_idx)
            if short_idx > 0:
                _merge_visit(visits[short_idx - 1], short_v)
            else:
                _merge_visit(visits[0], short_v)
            visits = _combine_adjacent_same_zone(visits)
            changed = True

    return visits


def _normalize_side(side_val: Any) -> str:
    s = str(side_val or "").strip().upper()
    if "TERRORIST" in s or s == "T":
        return "T"
    return "CT"


def _calculate_role_hints(
    ticks_round: pl.DataFrame,
    interval_s: float = BEAT_INTERVAL_S,
) -> dict[Any, str]:
    """Calculate role hint ('lurk' vs 'pack') for each player based on beat snapshots."""
    if ticks_round.is_empty() or "clock_s" not in ticks_round.columns:
        return {}

    post_freeze = ticks_round.filter(pl.col("clock_s") >= 0.0)
    sample_source = post_freeze if not post_freeze.is_empty() else ticks_round

    if "is_alive" in sample_source.columns:
        alive_ticks = sample_source.filter(pl.col("is_alive"))
    else:
        alive_ticks = sample_source

    if alive_ticks.is_empty():
        return {}

    max_clock_s = float(alive_ticks["clock_s"].max())
    if max_clock_s < 0.0:
        return {}

    id_col = "steamid" if "steamid" in alive_ticks.columns else "name"
    if id_col not in alive_ticks.columns:
        return {}

    # Check total players per team across the round to avoid single-player false lurk
    team_player_counts: dict[str, set[Any]] = defaultdict(set)
    for row in sample_source.select([id_col, "team_name"]).iter_rows(named=True):
        side = _normalize_side(row.get("team_name"))
        team_player_counts[side].add(row[id_col])

    alone_beats: dict[Any, int] = defaultdict(int)

    step = 0
    while step * interval_s <= max_clock_s:
        t_val = float(step * interval_s)
        diff = (alive_ticks["clock_s"] - t_val).abs()
        min_idx = diff.arg_min()
        if min_idx is not None:
            closest_clock = alive_ticks["clock_s"][min_idx]
            snap = alive_ticks.filter(pl.col("clock_s") == closest_clock)
            for side in ("T", "CT"):
                side_snap = snap.filter(
                    pl.col("team_name").map_elements(_normalize_side, return_dtype=pl.String)
                    == side
                )
                if side_snap.is_empty():
                    continue

                player_zones: dict[Any, str] = {}
                for row in side_snap.iter_rows(named=True):
                    pid = row[id_col]
                    z = str(row.get("last_place_name") or "Unknown").strip()
                    player_zones[pid] = z if z else "Unknown"

                # If the team has teammates across the round
                if len(team_player_counts[side]) >= 2:
                    for pid, z in player_zones.items():
                        teammate_zones = [
                            other_z for other_id, other_z in player_zones.items() if other_id != pid
                        ]
                        if z not in teammate_zones:
                            alone_beats[pid] += 1

        step += 1

    role_hints: dict[Any, str] = {}
    for pid in alive_ticks[id_col].unique().to_list():
        role_hints[pid] = "lurk" if alone_beats.get(pid, 0) >= 3 else "pack"
    return role_hints


def movement_sentences(
    ticks: pl.DataFrame,
    kills: list[KillEvent] | pl.DataFrame | None = None,
    round_num: int = 1,
    min_dwell_s: float = 2.0,
    plants: list[PlantEvent] | PlantEvent | None = None,
) -> list[MovementLine]:
    """Generate per-player movement sentences for a given round."""
    if ticks.is_empty() or "round_num" not in ticks.columns:
        return []

    ticks_round = ticks.filter(pl.col("round_num") == round_num)
    if ticks_round.is_empty():
        return []

    # Parse kills
    kill_events: list[KillEvent] = []
    if isinstance(kills, list):
        kill_events = list(kills)
    elif kills is not None and hasattr(kills, "filter"):
        kills_r = kills.filter(pl.col("round_num") == round_num)
        freeze_sample = ticks_round.filter(pl.col("clock_s") == 0.0)
        freeze_end_tick = freeze_sample["tick"][0] if not freeze_sample.is_empty() else None
        for row in kills_r.iter_rows(named=True):
            if "clock_s" in row and row["clock_s"] is not None:
                t = float(row["clock_s"])
            elif freeze_end_tick is not None and "tick" in row:
                t = (float(row["tick"]) - float(freeze_end_tick)) / 64.0
            else:
                t = 0.0
            kill_events.append(
                KillEvent(
                    t=t,
                    killer=str(row.get("attacker_name") or row.get("killer") or ""),
                    victim=str(row.get("victim_name") or row.get("victim") or ""),
                    killer_side=_normalize_side(row.get("attacker_side") or row.get("killer_side")),
                    zone=str(row.get("attacker_place") or row.get("zone") or "Unknown"),
                    weapon=str(row.get("weapon") or ""),
                    headshot=bool(row.get("headshot", False)),
                    traded_within_4s=bool(row.get("traded_within_4s", False)),
                )
            )

    # Parse plants
    plant_events: list[PlantEvent] = []
    if isinstance(plants, PlantEvent):
        plant_events = [plants]
    elif isinstance(plants, list):
        plant_events = list(plants)
    elif "is_bomb_planted" in ticks_round.columns:
        planted_ticks = ticks_round.filter(pl.col("is_bomb_planted"))
        if not planted_ticks.is_empty() and (~ticks_round["is_bomb_planted"]).any():
            first_plant_tick = int(planted_ticks["tick"].min())
            clock_val = planted_ticks.filter(pl.col("tick") == first_plant_tick)["clock_s"][0]
            plant_clock_s = float(clock_val) if clock_val is not None else 0.0
            pre_plant = ticks_round.filter(
                (pl.col("tick") < first_plant_tick) & (pl.col("tick") >= first_plant_tick - 64 * 4)
            )
            has_in_bomb = "in_bomb_zone" in pre_plant.columns
            has_wpn = "active_weapon_name" in pre_plant.columns
            c4_filter = pl.lit(False)
            if has_in_bomb:
                c4_filter = c4_filter | pl.col("in_bomb_zone")
            if has_wpn:
                c4_filter = c4_filter | pl.col(
                    "active_weapon_name"
                ).str.to_lowercase().str.contains("c4")
            c4_holders = pre_plant.filter(c4_filter)
            if not c4_holders.is_empty() and "steamid" in c4_holders.columns:
                planter_id = c4_holders["steamid"].mode()[0]
                planter_rows = c4_holders.filter(pl.col("steamid") == planter_id)
                planter_name = (
                    str(planter_rows["name"][0])
                    if "name" in planter_rows.columns and planter_rows["name"][0]
                    else str(planter_id)
                )
                plant_events.append(
                    PlantEvent(
                        t=plant_clock_s,
                        site="",
                        planter=planter_name,
                        alive_t=0,
                        alive_ct=0,
                    )
                )

    role_hints = _calculate_role_hints(ticks_round)

    # Determine player grouping
    has_steamid = "steamid" in ticks_round.columns and ticks_round["steamid"].drop_nulls().len() > 0
    group_cols = ["steamid"] if has_steamid else (["name"] if "name" in ticks_round.columns else [])
    if not group_cols:
        return []

    lines: list[MovementLine] = []

    for _, df_player in ticks_round.group_by(group_cols):
        df_player = df_player.sort("tick" if "tick" in df_player.columns else "clock_s")
        if df_player.is_empty():
            continue

        steamid_val = df_player["steamid"][0] if has_steamid else None
        name_val = (
            df_player["name"][0]
            if "name" in df_player.columns and df_player["name"][0]
            else str(steamid_val)
        )
        player_name = str(name_val)
        pid_key = steamid_val if has_steamid else player_name

        team_val = df_player["team_name"][0] if "team_name" in df_player.columns else ""
        side = _normalize_side(team_val)

        # Check death
        died = False
        death_t: float | None = None
        for k in kill_events:
            if k.victim in (player_name, str(steamid_val)):
                died = True
                death_t = k.t
                break

        if not died and "is_alive" in df_player.columns:
            alive_vals = df_player["is_alive"].to_list()
            if False in alive_vals and True in alive_vals:
                died = True
                alive_sub = df_player.filter(pl.col("is_alive"))
                if "clock_s" in alive_sub.columns and not alive_sub.is_empty():
                    death_t = float(alive_sub["clock_s"][-1])

        # Filter to alive ticks and post-freeze
        if "is_alive" in df_player.columns:
            alive_df = df_player.filter(pl.col("is_alive"))
        else:
            alive_df = df_player

        if alive_df.is_empty():
            continue

        if "clock_s" in alive_df.columns and (alive_df["clock_s"] >= 0.0).any():
            post_freeze = alive_df.filter(pl.col("clock_s") >= 0.0)
            if not post_freeze.is_empty():
                alive_df = post_freeze

        # Extract zone series
        zones: list[str] = []
        for val in (
            alive_df["last_place_name"].to_list() if "last_place_name" in alive_df.columns else []
        ):
            if val is None or str(val).strip() == "":
                zones.append("Unknown")
            else:
                zones.append(str(val).strip())

        if not zones:
            continue

        clocks: list[float] = (
            [float(c) for c in alive_df["clock_s"].to_list()]
            if "clock_s" in alive_df.columns
            else [i * 0.25 for i in range(len(zones))]
        )
        raw_ticks: list[int] = (
            [int(t) for t in alive_df["tick"].to_list()]
            if "tick" in alive_df.columns
            else [i * 16 for i in range(len(zones))]
        )
        walks: list[bool] = (
            [bool(w) for w in alive_df["is_walking"].to_list()]
            if "is_walking" in alive_df.columns
            else [False] * len(zones)
        )
        defuses: list[bool] = (
            [bool(d) for d in alive_df["is_defusing"].to_list()]
            if "is_defusing" in alive_df.columns
            else [False] * len(zones)
        )

        dt = 0.25
        if len(clocks) > 1:
            step_dt = clocks[1] - clocks[0]
            if step_dt > 0.0:
                dt = step_dt

        # Group consecutive ticks into initial visits
        raw_visits: list[_Visit] = []
        i = 0
        n = len(zones)
        while i < n:
            j = i
            while j < n and zones[j] == zones[i]:
                j += 1
            dur = clocks[j] - clocks[i] if j < n else (clocks[j - 1] - clocks[i] + dt)
            v = _Visit(
                zone=zones[i],
                start_t=clocks[i],
                end_t=clocks[j] if j < n else (clocks[j - 1] + dt),
                start_tick=raw_ticks[i],
                end_tick=raw_ticks[j - 1],
                duration=dur,
                walk_ticks=sum(walks[i:j]),
                total_ticks=j - i,
                is_defusing=any(defuses[i:j]),
            )
            raw_visits.append(v)
            i = j

        merged = _dwell_merge(raw_visits, min_dwell_s)
        if not merged:
            continue

        # Match kills where this player is the killer
        my_kills = [k for k in kill_events if k.killer in (player_name, str(steamid_val))]
        for k in my_kills:
            best_v = min(merged, key=lambda v: max(0.0, v.start_t - k.t, k.t - v.end_t))
            best_v.kills.append(k.victim)

        # Match bomb plants
        my_plants = [p for p in plant_events if p.planter in (player_name, str(steamid_val))]
        for p in my_plants:
            best_v = min(merged, key=lambda v: max(0.0, v.start_t - p.t, p.t - v.end_t))
            best_v.has_plant = True

        # Assign defuse attempt
        for v in merged:
            if v.is_defusing:
                v.has_defuse = True

        # Assign death
        if died:
            if death_t is not None:
                d_t = float(death_t)
                best_v = min(merged, key=lambda v: max(0.0, v.start_t - d_t, d_t - v.end_t))
                best_v.has_death = True
            else:
                merged[-1].has_death = True

        tokens: list[str] = []
        for v in merged:
            prefix = "~" if (v.total_ticks > 0 and v.walk_ticks / v.total_ticks > 0.5) else ""
            dwell_str = f"({round(v.duration)}s)" if v.duration >= 20.0 else ""
            token = f"{prefix}{v.zone}{dwell_str}"

            marks: list[str] = []
            for victim in v.kills:
                marks.append(f"k({victim})")
            if v.has_plant:
                marks.append("p")
            if v.has_defuse:
                marks.append("x")
            if v.has_death:
                marks.append("d")

            if marks:
                token = f"{token} " + " ".join(marks)
            tokens.append(token)

        sentence = f"{player_name}({side}): " + " > ".join(tokens)
        role_hint = role_hints.get(pid_key, "pack")

        lines.append(
            MovementLine(
                player=player_name,
                side=side,
                role_hint=role_hint,
                sentence=sentence,
            )
        )

    lines.sort(key=lambda m: (m.side, m.player))
    return lines
