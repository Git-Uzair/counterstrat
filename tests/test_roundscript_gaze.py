"""Gaze/stint semantics tests (space-vision research item 4).

A hold-stint gains a watched zone (yaw ray into zone centroids, restricted by
the empirical sightline matrix when present), a locked/scanning flag, and a
trade-support distance. The convention behind the math is verified against
real kills: attacker yaw == atan2(victim - attacker) within ~2 degrees median.
"""

import math

import polars as pl
import pytest

from counterstrat.roundscript.models import RoundScript, ZoneStint
from counterstrat.roundscript.movement import zone_stints

# Layout: p1 (T) holds "Hold" at (0,0) aiming east at "Watched" (1000,0) where
# e1 (CT) stands; teammate p2 (T) idles far north in "Team" (0,5000) aiming
# north at nothing.
_PLAYERS = [
    ("p1", "TERRORIST", "Hold", 0.0, 0.0, 0.0),
    ("p2", "TERRORIST", "Team", 0.0, 5000.0, 90.0),
    ("e1", "CT", "Watched", 1000.0, 0.0, 180.0),
]


def _ticks(seconds: int = 30, yaw_fn=None, with_view_cols: bool = True) -> pl.DataFrame:
    rows = []
    for s in range(seconds):
        for name, team, zone, x, y, yaw in _PLAYERS:
            if yaw_fn is not None and name == "p1":
                yaw = yaw_fn(s)
            row = {
                "round_num": 1,
                "clock_s": float(s),
                "tick": s * 64,
                "name": name,
                "team_name": team,
                "is_alive": True,
                "last_place_name": zone,
            }
            if with_view_cols:
                row |= {"X": x, "Y": y, "Z": 0.0, "yaw": yaw}
            rows.append(row)
    return pl.DataFrame(rows)


def test_hold_stint_gets_watched_zone_locked_and_support():
    tracks, _sides = zone_stints(_ticks(), round_num=1)
    st = tracks["p1"][0]
    assert st.zone == "Hold" and st.t1 - st.t0 >= 8
    assert st.watched == "Watched"
    assert st.locked is True
    # teammate p2 is 5000u away = 127m
    assert st.support_m == pytest.approx(127.0, abs=1.0)
    # p2 aims north at nothing: no watched zone, support is distance to p1
    st2 = tracks["p2"][0]
    assert st2.watched is None
    assert st2.support_m == pytest.approx(127.0, abs=1.0)


def test_scanning_hold_is_not_locked():
    tracks, _ = zone_stints(_ticks(yaw_fn=lambda s: 50.0 if s % 2 else -50.0), round_num=1)
    st = tracks["p1"][0]
    assert st.locked is False
    assert st.watched == "Watched"  # sweep still centers on the corridor


def test_sightline_matrix_restricts_candidates():
    # The matrix says Hold only sees Behind; Watched is occluded evidence-wise.
    sl = [{"from": "Behind", "to": "Hold", "n": 3, "median_dist": 20.0, "range": "medium"}]
    tracks, _ = zone_stints(_ticks(), round_num=1, sightlines=sl)
    st = tracks["p1"][0]
    assert st.watched is None  # Behind is not where he aims; Watched not visible


def test_short_stints_and_missing_view_columns_stay_unannotated():
    tracks, _ = zone_stints(_ticks(seconds=5), round_num=1)
    st = tracks["p1"][0]
    assert st.watched is None and st.locked is None and st.support_m is None

    tracks, _ = zone_stints(_ticks(with_view_cols=False), round_num=1)
    st = tracks["p1"][0]
    assert st.watched is None and st.locked is None and st.support_m is None


def test_timeline_renders_hold_and_move_annotations():
    rs = RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="chk",
        round_num=1,
        score_t=0,
        score_ct=0,
        t_team_key="abc",
        ct_team_key="xyz",
        economy={},
        beats=[],
        kills=[],
        utility=[],
        plant=None,
        first_contact=None,
        winner="T",
        reason="ct_killed",
        clock_used_s=60.0,
        tracks={
            "p1": [
                ZoneStint(
                    t0=0, t1=20, zone="Hold", watched="Watched", locked=True, support_m=127.0
                ),
                ZoneStint(
                    t0=20, t1=40, zone="Middle", watched="BombsiteA", locked=False, support_m=8.0
                ),
            ],
            "p2": [ZoneStint(t0=0, t1=40, zone="Team")],
        },
        sides={"p1": "T", "p2": "T"},
    )
    text = rs.to_timeline_text()
    assert "t=0s HOLD p1 (T) in `Hold` watching `Watched` (locked, nearest mate 127m)" in text
    assert (
        "t=20s MOVE p1 (T) enters `Middle`, watching `BombsiteA` "
        "(scanning, nearest mate 8m)" in text
    )
    # unannotated stints render exactly as before
    assert "HOLD p2" not in text

    # The First Read budget mode: every resolved gaze survives as a HOLD line
    # (even mid-round stints that would be MOVEs), plain traffic drops.
    lite = rs.to_timeline_text(lite=True, include_holds=True)
    assert "t=0s HOLD p1 (T) in `Hold` watching `Watched` (locked, nearest mate 127m)" in lite
    assert "t=20s HOLD p1 (T) in `Middle` watching `BombsiteA` (scanning, nearest mate 8m)" in lite
    assert "MOVE" not in lite and "SPAWNS" not in lite and "HOLD p2" not in lite
    # Plain lite stays gaze-free for callers that want the old shape.
    assert "watching" not in rs.to_timeline_text(lite=True)


@pytest.mark.demo
def test_yaw_convention_and_annotation_rate_on_real_match(anubis_lake):
    """Validation harness (research §7): yaw must equal bearing-to-victim on
    clean kills, and real holds must produce watched zones."""
    kills = pl.read_parquet(anubis_lake.kills).filter(
        pl.col("attacker_X").is_not_null() & pl.col("victim_X").is_not_null()
    )
    ticks = pl.read_parquet(anubis_lake.ticks)
    tsel = ticks.select(["tick", "name", "yaw", "X", "Y"])
    errs = []
    for row in kills.head(80).iter_rows(named=True):
        near = tsel.filter(
            (pl.col("name") == row["attacker_name"]) & ((pl.col("tick") - row["tick"]).abs() <= 16)
        )
        if near.is_empty():
            continue
        bearing = math.degrees(
            math.atan2(row["victim_Y"] - row["attacker_Y"], row["victim_X"] - row["attacker_X"])
        )
        d = (float(near["yaw"][0]) - bearing + 180.0) % 360.0 - 180.0
        errs.append(abs(d))
    errs.sort()
    assert errs and errs[len(errs) // 2] < 10.0  # median within 10 degrees

    tracks, _ = zone_stints(ticks, round_num=1)
    holds = [st for stints in tracks.values() for st in stints if st.t1 - st.t0 >= 8]
    assert holds, "a real round has hold-stints"
    assert any(st.watched for st in holds), "some holds must resolve a watched zone"
    assert any(st.support_m is not None for st in holds)
