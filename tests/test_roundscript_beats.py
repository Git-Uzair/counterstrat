import polars as pl
import pytest

from counterstrat.roundscript.beats import build_beats, formation_from_zones
from counterstrat.roundscript.econ import EconSummary
from counterstrat.roundscript.models import (
    Beat,
    Formation,
    KillEvent,
    MovementLine,
    PlantEvent,
    RoundScript,
    UtilEvent,
)


def test_formation_rle_ordering():
    f = formation_from_zones(["Mid", "Mid", "Mid", "TSpawn", "ARamp"])
    assert f.zones == [(3, "Mid"), (1, "ARamp"), (1, "TSpawn")]


def test_formation_rle_empty_and_ties():
    assert formation_from_zones([]).zones == []
    # All count 1, alphabetical order
    f = formation_from_zones(["B", "A", "C"])
    assert f.zones == [(1, "A"), (1, "B"), (1, "C")]


def test_build_beats_synthetic():
    ticks = pl.DataFrame(
        {
            "round_num": [1, 1, 1, 1, 1, 1, 1, 1],
            "tick": [100, 100, 100, 100, 200, 200, 200, 200],
            "clock_s": [0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0],
            "team_name": [
                "TERRORIST",
                "TERRORIST",
                "CT",
                "CT",
                "TERRORIST",
                "TERRORIST",
                "CT",
                "CT",
            ],
            "is_alive": [True, True, True, False, True, False, True, True],
            "last_place_name": [
                "TSpawn",
                "TSpawn",
                "CTSpawn",
                "Mid",
                "ARamp",
                "TSpawn",
                "BDoors",
                "BDoors",
            ],
        }
    )
    beats = build_beats(ticks, round_num=1, first_contact_t=10.0, plant_t=None, interval_s=15)
    # Target times: 0.0, 15.0; event time: 10.0 -> order: 0.0, 10.0, 15.0
    assert len(beats) == 3
    assert beats[0].label == "B+00"
    assert beats[0].t == 0.0
    assert beats[0].t_form.zones == [(2, "TSpawn")]
    assert beats[0].ct_form.zones == [(1, "CTSpawn")]  # 1 alive, 1 dead ignored

    assert beats[1].label == "FC+10"
    assert beats[1].t == 10.0

    assert beats[2].label == "B+15"
    assert beats[2].t == 15.0
    assert beats[2].t_form.zones == [(1, "ARamp")]
    assert beats[2].ct_form.zones == [(2, "BDoors")]


@pytest.mark.demo
def test_beats_round1(anubis_lake):
    ticks = pl.read_parquet(anubis_lake.ticks)
    beats = build_beats(ticks, round_num=1, first_contact_t=None, plant_t=None)
    assert beats[0].label == "B+00"
    total0 = sum(c for c, _ in beats[0].t_form.zones)
    assert total0 == 5  # all five Ts alive at freeze end
    assert all(b.t <= 120 for b in beats)


@pytest.mark.demo
def test_beats_event_anchors(anubis_lake):
    ticks = pl.read_parquet(anubis_lake.ticks)
    beats = build_beats(ticks, round_num=1, first_contact_t=34.2, plant_t=52.0)
    labels = [b.label for b in beats]
    assert "FC+34" in labels
    assert "PL+52" in labels
    fc_beat = next(b for b in beats if b.label == "FC+34")
    assert fc_beat.t == 34.2
    pl_beat = next(b for b in beats if b.label == "PL+52")
    assert pl_beat.t == 52.0
    times = [b.t for b in beats]
    assert times == sorted(times)


def test_models_instantiation():
    t_form = Formation(zones=[(3, "Mid"), (2, "TSpawn")])
    ct_form = Formation(zones=[(5, "CTSpawn")])
    beat = Beat(label="B+00", t=0.0, t_form=t_form, ct_form=ct_form)
    kill = KillEvent(
        t=12.5,
        killer="player1",
        victim="player2",
        killer_side="T",
        zone="Mid",
        weapon="ak47",
        headshot=True,
        traded_within_4s=False,
    )
    util = UtilEvent(
        t=5.0,
        thrower="player1",
        side="T",
        nade="flashbang",
        from_zone="TSpawn",
        to_zone="Mid",
        lineup_id=None,
        blinded=[("player2", 1.8)],
    )
    plant = PlantEvent(t=45.0, site="A", planter="player1", alive_t=3, alive_ct=2)
    movement = MovementLine(
        player="player1",
        side="T",
        role_hint="entry",
        sentence="TSpawn > Mid k(player2)",
    )
    rs = RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="abc1234",
        round_num=1,
        score_t=0,
        score_ct=0,
        t_team_key="TeamA",
        ct_team_key="TeamB",
        economy={
            "T": EconSummary(buy_type="full_eco", spend=1000, equip=5000, awps=0, loss_streak=0),
            "CT": EconSummary(buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0),
        },
        beats=[beat],
        kills=[kill],
        utility=[util],
        plant=plant,
        first_contact=kill,
        winner="T",
        reason="target_bombed",
        clock_used_s=65.4,
        movements=[movement],
    )
    assert rs.match_id == "m1"
    assert rs.beats[0].label == "B+00"
    assert rs.plant is not None and rs.plant.site == "A"
    assert rs.first_contact is not None and rs.first_contact.killer == "player1"
    assert rs.to_text().splitlines()[0].startswith("R1 [")
    assert "END T target_bombed @1:05" in rs.to_text()
