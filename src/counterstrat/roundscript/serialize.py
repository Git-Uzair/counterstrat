"""RoundScript serialization: match and round orchestrator."""

from pathlib import Path

import polars as pl

from counterstrat.constants import TRADE_WINDOW_S
from counterstrat.lake.extract import LakePaths
from counterstrat.mapcard.lexicon import Lexicon
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.beats import build_beats
from counterstrat.roundscript.econ import round_economy
from counterstrat.roundscript.models import KillEvent, PlantEvent, RoundScript, UtilEvent
from counterstrat.roundscript.movement import _normalize_side, movement_sentences, zone_stints
from counterstrat.roundscript.utility import cluster_lineups, utility_events_with_xyz


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

    # Kills
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

    # Per-player zone stints (timeline MOVE lines, plan Task 3)
    tracks, sides = zone_stints(t_df, round_num=round_num)

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
    )


def serialize_match(
    lake: LakePaths,
    mapper: ZoneMapper,
    lex: Lexicon,
    card_checksum: str,
) -> list[RoundScript]:
    """Serialize all rounds of a match demo into RoundScript models."""
    rounds_df = pl.read_parquet(lake.rounds).sort("round_num")
    ticks_df = pl.read_parquet(lake.ticks)
    kills_df = pl.read_parquet(lake.kills)
    bomb_df = (
        pl.read_parquet(lake.bomb) if lake.bomb and Path(lake.bomb).exists() else pl.DataFrame()
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
        )
        scripts.append(script)

        # Update running match scores
        winner_str = _normalize_side(r_dict.get("winner"))
        if winner_str == "T":
            team_scores[t_key] = score_t + 1
        elif winner_str == "CT":
            team_scores[ct_key] = score_ct + 1

    return scripts
