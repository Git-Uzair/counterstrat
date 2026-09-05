"""RoundScript serialization: match and round orchestrator."""

import math
from pathlib import Path

import polars as pl

from counterstrat.constants import TRADE_WINDOW_S
from counterstrat.lake.extract import LakePaths
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.beats import build_beats
from counterstrat.roundscript.econ import round_economy
from counterstrat.roundscript.models import KillEvent, PlantEvent, RoundScript, UtilEvent
from counterstrat.roundscript.movement import (
    VICTIM_MOVING_SPEED,
    _circ_diff_deg,
    _normalize_side,
    detect_rotations,
    movement_sentences,
    zone_stints,
)
from counterstrat.roundscript.utility import cluster_lineups, utility_events_with_xyz


def _death_context(
    ticks_r: pl.DataFrame, krow: dict
) -> tuple[bool | None, float | None, str | None]:
    """(victim_moving, victim_preaim_off_deg, victim_weapon) for one kill row.

    Read from the victim's last 16 Hz tick samples before the death tick; every
    field degrades to None when its inputs are missing (old lakes, bot rows) -
    None means "not measured", never "confirmed false".
    """
    vname = str(krow.get("victim_name") or "")
    ktick = krow.get("tick")
    if not vname or ktick is None or ticks_r.is_empty():
        return None, None, None
    if not {"name", "tick"} <= set(ticks_r.columns):
        return None, None, None
    vt = ticks_r.filter((pl.col("name") == vname) & (pl.col("tick") <= int(ktick))).sort("tick")
    if vt.is_empty():
        return None, None, None
    last = vt.row(-1, named=True)

    weapon = None
    if "active_weapon_name" in vt.columns:
        weapon = str(last.get("active_weapon_name") or "").strip() or None

    moving: bool | None = None
    if vt.height >= 2 and {"X", "Y"} <= set(vt.columns):
        prev = vt.row(-2, named=True)
        coords = (last.get("X"), last.get("Y"), prev.get("X"), prev.get("Y"))
        dt_ticks = int(last["tick"]) - int(prev["tick"])
        if all(c is not None for c in coords) and dt_ticks > 0:
            lx, ly, px, py = (float(c) for c in coords)
            speed = math.hypot(lx - px, ly - py) / (dt_ticks / 64.0)
            moving = speed > VICTIM_MOVING_SPEED

    preaim: float | None = None
    kx, ky = krow.get("attacker_X"), krow.get("attacker_Y")
    vx = krow.get("victim_X") if krow.get("victim_X") is not None else last.get("X")
    vy = krow.get("victim_Y") if krow.get("victim_Y") is not None else last.get("Y")
    yaw = last.get("yaw") if "yaw" in vt.columns else None
    if kx is not None and ky is not None and vx is not None and vy is not None and yaw is not None:
        bearing = math.degrees(math.atan2(float(ky) - float(vy), float(kx) - float(vx)))
        preaim = round(_circ_diff_deg(float(yaw), bearing), 1)
    return moving, preaim, weapon


def serialize_round(
    lake: LakePaths,
    mapper: ZoneMapper,
    lex: Lexicon,
    card_checksum: str,
    round_num: int,
    ticks_df: pl.DataFrame | None = None,
    rounds_df: pl.DataFrame | None = None,
    kills_df: pl.DataFrame | None = None,
    bomb_df: pl.DataFrame | None = None,
    utils: list[UtilEvent] | None = None,
    score_t: int = 0,
    score_ct: int = 0,
    sightlines: list[dict] | None = None,
    shots_df: pl.DataFrame | None = None,
) -> RoundScript:
    """Serialize a single round into a RoundScript model."""
    r_df = rounds_df if rounds_df is not None else pl.read_parquet(lake.rounds)
    t_df = ticks_df if ticks_df is not None else pl.read_parquet(lake.ticks)
    k_df = kills_df if kills_df is not None else pl.read_parquet(lake.kills)
    b_df = (
        bomb_df
        if bomb_df is not None
        else (
            pl.read_parquet(lake.bomb) if lake.bomb and Path(lake.bomb).exists() else pl.DataFrame()
        )
    )
    s_df = (
        shots_df
        if shots_df is not None
        else (
            pl.read_parquet(lake.shots)
            if lake.shots and Path(lake.shots).exists()
            else pl.DataFrame()
        )
    )

    r_row_df = r_df.filter(pl.col("round_num") == round_num)
    r_dict = next(r_row_df.iter_rows(named=True), {})

    t_key = str(r_dict.get("t_team_key") or "T")
    ct_key = str(r_dict.get("ct_team_key") or "CT")
    freeze_end = r_dict.get("freeze_end")
    round_end = r_dict.get("end")

    if round_end is not None and freeze_end is not None:
        clock_used_s = (float(round_end) - float(freeze_end)) / 64.0
    else:
        ticks_r = t_df.filter(pl.col("round_num") == round_num)
        clock_used_s = (
            float(ticks_r["clock_s"].max())
            if not ticks_r.is_empty() and "clock_s" in ticks_r.columns
            else 0.0
        )

    # Kills (death contexts read the victim's last tick samples, plan v3)
    ticks_round = t_df.filter(pl.col("round_num") == round_num) if not t_df.is_empty() else t_df
    r_kills = k_df.filter(pl.col("round_num") == round_num) if not k_df.is_empty() else k_df
    kill_events: list[KillEvent] = []
    for krow in r_kills.iter_rows(named=True):
        if krow.get("victim_name") is None or not str(krow["victim_name"]).strip():
            continue
        ktick = krow.get("tick")
        kt = (float(ktick) - float(freeze_end)) / 64.0 if ktick and freeze_end else 0.0

        ap = krow.get("attacker_place")
        if ap and lex.is_valid_zone(str(ap).strip()):
            kzone = str(ap).strip()
        elif krow.get("attacker_X") is not None:
            kzone = mapper.zone(krow["attacker_X"], krow["attacker_Y"], krow["attacker_Z"])
        elif krow.get("victim_place") and lex.is_valid_zone(str(krow["victim_place"]).strip()):
            kzone = str(krow["victim_place"]).strip()
        elif krow.get("victim_X") is not None:
            kzone = mapper.zone(krow["victim_X"], krow["victim_Y"], krow["victim_Z"])
        else:
            kzone = "Unknown"

        dist = krow.get("distance")
        victim_moving, victim_preaim, victim_weapon = _death_context(ticks_round, krow)
        kill_events.append(
            KillEvent(
                t=kt,
                killer=str(krow.get("attacker_name") or ""),
                victim=str(krow.get("victim_name") or ""),
                killer_side=_normalize_side(krow.get("attacker_side")),
                zone=kzone,
                weapon=str(krow.get("weapon") or ""),
                headshot=bool(krow.get("headshot", False)),
                traded_within_4s=False,
                distance=float(dist) if dist is not None else None,
                thrusmoke=bool(krow.get("thrusmoke") or False),
                penetrated=bool(krow.get("penetrated") or 0),
                victim_moving=victim_moving,
                victim_preaim_off_deg=victim_preaim,
                victim_weapon=victim_weapon,
            )
        )

    kill_events.sort(key=lambda k: k.t)
    for i, k1 in enumerate(kill_events):
        for k2 in kill_events[i + 1 :]:
            if k2.t - k1.t > TRADE_WINDOW_S:
                break
            if k2.victim == k1.killer and 0 <= k2.t - k1.t <= TRADE_WINDOW_S:
                k1.traded_within_4s = True
                break

    first_contact = kill_events[0] if kill_events else None

    # Plant
    plant: PlantEvent | None = None
    if not b_df.is_empty():
        r_bomb = b_df.filter((pl.col("round_num") == round_num) & (pl.col("event") == "plant"))
        if not r_bomb.is_empty():
            brow = r_bomb.iter_rows(named=True).__next__()
            ptick = brow.get("tick")
            pt = (float(ptick) - float(freeze_end)) / 64.0 if ptick and freeze_end else 0.0
            ticks_r = t_df.filter(pl.col("round_num") == round_num)
            at = 0
            act = 0
            if not ticks_r.is_empty() and ptick is not None:
                diff = (ticks_r["tick"] - ptick).abs()
                min_idx = diff.arg_min()
                if min_idx is not None:
                    closest_tick = ticks_r["tick"][min_idx]
                    snap = ticks_r.filter((pl.col("tick") == closest_tick) & pl.col("is_alive"))
                    at = snap.filter(pl.col("team_name").is_in(["TERRORIST", "T"])).height
                    act = snap.filter(pl.col("team_name").is_in(["CT"])).height
            bsite = str(brow.get("bombsite") or "")
            if not bsite or not lex.is_valid_zone(bsite):
                if brow.get("X") is not None:
                    bsite = mapper.zone(brow["X"], brow["Y"], brow["Z"])
                else:
                    bsite = "BombsiteA"

            plant = PlantEvent(
                t=pt,
                site=bsite,
                planter=str(brow.get("name") or ""),
                alive_t=at,
                alive_ct=act,
            )

    # Beats
    beats = build_beats(
        t_df,
        round_num=round_num,
        first_contact_t=first_contact.t if first_contact else None,
        plant_t=plant.t if plant else None,
    )

    # Economy
    economy = round_economy(t_df, round_num)

    # Movements
    movements = movement_sentences(t_df, kills=kill_events, round_num=round_num, plants=plant)

    # Per-player zone stints (timeline MOVE/HOLD lines, plan Task 3) with gaze
    # semantics restricted by the card's empirical sightlines when present.
    tracks, sides = zone_stints(t_df, round_num=round_num, sightlines=sightlines)

    # Rotations (2026-09-05 plan Task 1): trigger-conditioned hold-breaks.
    first_shot_t: float | None = None
    if not s_df.is_empty() and {"round_num", "tick"} <= set(s_df.columns) and freeze_end:
        r_shots = s_df.filter(pl.col("round_num") == round_num)
        if not r_shots.is_empty():
            first_shot_t = (float(r_shots["tick"].min()) - float(freeze_end)) / 64.0
    rotations = detect_rotations(
        tracks,
        sides,
        t_df,
        round_num,
        kills=kill_events,
        plant=plant,
        utility=utils or [],
        first_shot_t=first_shot_t,
        sightlines=sightlines,
    )

    match_id = str(r_dict.get("match_id") or Path(lake.root).name)
    winner_str = _normalize_side(r_dict.get("winner"))

    return RoundScript(
        match_id=match_id,
        map_name=lex.map_name,
        card_checksum=card_checksum,
        round_num=round_num,
        score_t=score_t,
        score_ct=score_ct,
        t_team_key=t_key,
        ct_team_key=ct_key,
        economy=economy,
        beats=beats,
        kills=kill_events,
        utility=utils or [],
        plant=plant,
        first_contact=first_contact,
        winner=winner_str,
        reason=str(r_dict.get("reason") or ""),
        clock_used_s=clock_used_s,
        movements=movements,
        tracks=tracks,
        sides=sides,
        rotations=rotations,
    )


def serialize_match(
    lake: LakePaths,
    mapper: ZoneMapper,
    lex: Lexicon,
    card_checksum: str,
    sightlines: list[dict] | None = None,
) -> list[RoundScript]:
    """Serialize all rounds of a match demo into RoundScript models."""
    rounds_df = pl.read_parquet(lake.rounds).sort("round_num")
    ticks_df = pl.read_parquet(lake.ticks)
    kills_df = pl.read_parquet(lake.kills)
    bomb_df = (
        pl.read_parquet(lake.bomb) if lake.bomb and Path(lake.bomb).exists() else pl.DataFrame()
    )
    shots_df = (
        pl.read_parquet(lake.shots) if lake.shots and Path(lake.shots).exists() else pl.DataFrame()
    )

    round_nums = [int(rn) for rn in rounds_df["round_num"].to_list()]

    # Extract all utility across rounds, cluster lineups match-wide, partition back
    all_utils: list[UtilEvent] = []
    raw_xyz_frames: list[pl.DataFrame] = []
    utils_by_round: dict[int, list[UtilEvent]] = {}
    for rn in round_nums:
        evs, xyz = utility_events_with_xyz(lake, mapper, rn)
        utils_by_round[rn] = evs
        all_utils.extend(evs)
        raw_xyz_frames.append(xyz)

    all_xyz = pl.concat(raw_xyz_frames) if raw_xyz_frames else pl.DataFrame()
    cluster_lineups(all_utils, all_xyz)

    team_scores: dict[str, int] = {}
    scripts: list[RoundScript] = []

    for r_dict in rounds_df.iter_rows(named=True):
        rn = int(r_dict["round_num"])
        t_key = str(r_dict.get("t_team_key") or "T")
        ct_key = str(r_dict.get("ct_team_key") or "CT")
        score_t = team_scores.get(t_key, 0)
        score_ct = team_scores.get(ct_key, 0)

        script = serialize_round(
            lake=lake,
            mapper=mapper,
            lex=lex,
            card_checksum=card_checksum,
            round_num=rn,
            ticks_df=ticks_df,
            rounds_df=rounds_df,
            kills_df=kills_df,
            bomb_df=bomb_df,
            utils=utils_by_round.get(rn, []),
            score_t=score_t,
            score_ct=score_ct,
            sightlines=sightlines,
            shots_df=shots_df,
        )
        scripts.append(script)

        # Update running match scores
        winner_str = _normalize_side(r_dict.get("winner"))
        if winner_str == "T":
            team_scores[t_key] = score_t + 1
        elif winner_str == "CT":
            team_scores[ct_key] = score_ct + 1

    return scripts
