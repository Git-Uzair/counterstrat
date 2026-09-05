"""RoundScript v2: per-player zone stints + the complete timeline text.

The 2026-09-04 representation lab measured today's to_text() at 83%/62%
(easy/hard) against 100%/96% for a complete chronological stream: the gaps are
kills after first contact, FC/plant timestamps, movement, and the utility cap.
to_timeline_text() closes them; these tests pin the contract.
"""

import polars as pl

from counterstrat.roundscript.models import (
    Beat,
    Formation,
    KillEvent,
    PlantEvent,
    RoundScript,
    UtilEvent,
    ZoneStint,
)
from counterstrat.roundscript.movement import zone_stints


def _ticks() -> pl.DataFrame:
    """Two players; p1 flickers into Mid for 1s (debounced), p2 dies at 8s."""
    rows = []
    for s in range(12):
        zone = "TSpawn" if s < 4 else ("Mid" if s == 4 else "Banana") if s < 10 else "BSite"
        rows.append((1, "p1", "TERRORIST", float(s), zone, True))
    # p1's s=4 'Mid' is a 1s blip between TSpawn and Banana -> merged away
    for s in range(12):
        alive = s < 8
        rows.append((1, "p2", "CT", float(s), "CTSpawn" if s < 5 else "ASite", alive))
    return pl.DataFrame(
        [
            {
                "round_num": r,
                "name": n,
                "team_name": t,
                "clock_s": c,
                "last_place_name": z,
                "is_alive": a,
                "tick": int(c * 64),
                "steamid": hash(n) % 1000,
            }
            for r, n, t, c, z, a in rows
        ]
    )


def test_zone_stints_debounce_and_death():
    tracks, sides = zone_stints(_ticks(), round_num=1, min_dwell_s=2.0)
    assert sides == {"p1": "T", "p2": "CT"}
    p1_zones = [s.zone for s in tracks["p1"]]
    assert "Mid" not in p1_zones  # 1s blip merged away
    assert p1_zones[0] == "TSpawn" and "Banana" in p1_zones
    # stints carry integer second bounds and are contiguous per player
    for st in tracks["p1"]:
        assert isinstance(st.t0, int) and isinstance(st.t1, int) and st.t1 > st.t0
    # p2 stops at death: no stint reaches past 8s
    assert max(s.t1 for s in tracks["p2"]) <= 9


def _script(**overrides) -> RoundScript:
    base = RoundScript(
        match_id="m1",
        map_name="de_test",
        card_checksum="chk",
        round_num=5,
        score_t=3,
        score_ct=2,
        t_team_key="tk1",
        ct_team_key="tk2",
        economy={},
        beats=[
            Beat(
                label="B+15",
                t=15.0,
                t_form=Formation(zones=[(2, "Banana")]),
                ct_form=Formation(zones=[(1, "ASite"), (1, "BSite")]),
            )
        ],
        kills=[
            KillEvent(
                t=20.0,
                killer="p1",
                victim="e1",
                killer_side="T",
                zone="Mid",
                weapon="ak47",
                headshot=False,
                traded_within_4s=False,
            ),
            KillEvent(
                t=44.0,
                killer="e2",
                victim="p1",
                killer_side="CT",
                zone="BSite",
                weapon="m4a1",
                headshot=True,
                traded_within_4s=True,
            ),
            KillEvent(
                t=61.0,
                killer="p2",
                victim="e2",
                killer_side="T",
                zone="BSite",
                weapon="deagle",
                headshot=False,
                traded_within_4s=False,
            ),
        ],
        utility=[
            UtilEvent(
                t=12.0,
                thrower="p1",
                side="T",
                nade="smoke",
                from_zone="TSpawn",
                to_zone="Mid",
                lineup_id="Mid-S1",
            ),
            UtilEvent(
                t=41.0,
                thrower="p2",
                side="T",
                nade="flash",
                from_zone="Banana",
                to_zone="BSite",
                blinded=[("e1", 2.3)],
            ),
        ],
        plant=PlantEvent(t=52.0, site="BSite", planter="p2", alive_t=3, alive_ct=2),
        first_contact=None,
        winner="T",
        reason="bomb_exploded",
        clock_used_s=97.0,
        movements=[],
        tracks={
            "p1": [ZoneStint(t0=0, t1=6, zone="TSpawn"), ZoneStint(t0=6, t1=44, zone="Banana")],
            "p2": [ZoneStint(t0=0, t1=9, zone="TSpawn"), ZoneStint(t0=9, t1=97, zone="BSite")],
        },
        sides={"p1": "T", "p2": "T"},
    )
    base.first_contact = base.kills[0]
    return base.model_copy(update=overrides) if overrides else base


def test_timeline_text_complete():
    text = _script().to_timeline_text()
    assert text.splitlines()[0].startswith("R5")
    # anchors header with precomputed deltas
    assert "first_contact=20s" in text
    assert "plant=52s at `BSite` (+32s after first contact)" in text
    assert "end=97s" in text
    # every kill renders, not just first contact
    assert "t=20s KILL p1 (T) kills e1 in `Mid` [ak47]" in text
    assert "t=44s KILL e2 (CT) kills p1 in `BSite` [m4a1] [traded]" in text
    assert "t=61s KILL p2 (T) kills e2 in `BSite` [deagle]" in text
    # utility with pop times, lineups, blinds
    assert "t=12s UTIL p1 (T) smoke from `TSpawn` lands `Mid` [Mid-S1]" in text
    assert "t=41s UTIL p2 (T) flash from `Banana` lands `BSite` (blinds e1 2.3s)" in text
    # plant with timestamp and alive counts
    assert "t=52s PLANT p2 plants at `BSite` (3v2 alive)" in text
    # movement stream + spawns roster
    assert "SPAWNS" in text and "p1:`TSpawn`" in text
    assert "t=6s MOVE p1 (T) enters `Banana`" in text
    # state snapshots survive from beats
    assert "state@15s: T[2x`Banana`] CT[1x`ASite`, 1x`BSite`]" in text
    assert "END: T wins (bomb_exploded) at 97s" in text


def test_enrichment_fields_never_change_the_timeline():
    """2026-09-05 plan budget rule: timelines dominate the First Read prompt,
    so the v3 enrichment (rotations / effect fields / death contexts) must add
    ZERO timeline lines - it renders via miners and tools only."""
    from counterstrat.roundscript.models import RotationEvent

    plain = _script()
    enriched = _script(
        kills=[
            k.model_copy(
                update={
                    "victim_moving": True,
                    "victim_preaim_off_deg": 38.5,
                    "victim_weapon": "AK-47",
                }
            )
            for k in plain.kills
        ],
        utility=[
            u.model_copy(
                update={
                    "enemy_blind_s": 3.1,
                    "team_blind_s": 0.2,
                    "damage": 34,
                    "kills_through": 1,
                }
            )
            for u in plain.utility
        ],
        rotations=[
            RotationEvent(
                t_trigger=12.0,
                trigger="utility_near",
                player="p1",
                side="T",
                from_zone="TSpawn",
                to_zone="Banana",
                latency_s=1.5,
            )
        ],
    )
    assert enriched.to_timeline_text() == plain.to_timeline_text()
    assert enriched.to_timeline_text(lite=True, include_holds=True) == plain.to_timeline_text(
        lite=True, include_holds=True
    )
    assert enriched.to_text() == plain.to_text()


def test_timeline_lite_drops_moves_keeps_kills():
    text = _script().to_timeline_text(lite=True)
    assert "MOVE" not in text and "SPAWNS" not in text
    assert "t=44s KILL e2 (CT) kills p1" in text
    assert "t=52s PLANT" in text
    assert "state@15s:" in text


def test_timeline_falls_back_without_tracks():
    s = _script(tracks={}, sides={})
    text = s.to_timeline_text()
    assert "MOVE" not in text and "SPAWNS" not in text
    assert "t=20s KILL p1 (T) kills e1" in text
    assert "END: T wins (bomb_exploded) at 97s" in text


def test_timeline_uncaps_utility():
    utils = [
        UtilEvent(
            t=float(5 + i),
            thrower="p1",
            side="T",
            nade="he",
            from_zone="A",
            to_zone="B",
        )
        for i in range(20)
    ]
    text = _script(utility=utils).to_timeline_text()
    assert text.count("UTIL") == 20  # to_text caps at 16; the timeline must not
