import numpy as np
import polars as pl
import pytest

from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.zones import ZoneMapper
from counterstrat.roundscript.lint import lint_script
from counterstrat.roundscript.models import (
    Beat,
    Formation,
    KillEvent,
    MovementLine,
    PlantEvent,
    RoundScript,
    UtilEvent,
)
from counterstrat.roundscript.serialize import serialize_match, serialize_round


@pytest.fixture(scope="module")
def anubis_bundle(anubis_lake, anubis_assets):
    ticks = pl.read_parquet(anubis_lake.ticks)
    mapper = ZoneMapper.fit(ticks)
    places = unique_places(parse_places(anubis_assets.vents))
    overlay = get_default_overlay_path("de_anubis")
    lex = build_lexicon("de_anubis", places, overlay)
    return {
        "lake": anubis_lake,
        "mapper": mapper,
        "lex": lex,
        "card_checksum": lex.checksum,
    }


def test_lint_script_clean():
    lex = build_lexicon("de_anubis", ["BombsiteA", "BombsiteB", "Middle", "TSpawn", "CTSpawn"])
    kill = KillEvent(
        t=10.0,
        killer="p1",
        victim="p2",
        killer_side="T",
        zone="Middle",
        weapon="ak47",
        headshot=True,
        traded_within_4s=False,
    )
    rs = RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="c123",
        round_num=1,
        score_t=0,
        score_ct=0,
        t_team_key="TeamA",
        ct_team_key="TeamB",
        economy={},
        beats=[
            Beat(
                label="B+00",
                t=0.0,
                t_form=Formation(zones=[(1, "TSpawn")]),
                ct_form=Formation(zones=[(1, "CTSpawn")]),
            )
        ],
        kills=[kill],
        utility=[
            UtilEvent(
                t=5.0,
                thrower="p1",
                side="T",
                nade="smoke",
                from_zone="TSpawn",
                to_zone="Middle",
            )
        ],
        plant=None,
        first_contact=kill,
        winner="T",
        reason="ct_killed",
        clock_used_s=60.0,
        movements=[
            MovementLine(
                player="p1",
                side="T",
                role_hint="pack",
                sentence="p1(T): TSpawn > Middle k(p2)",
            )
        ],
    )
    problems = lint_script(rs, lex)
    assert problems == []


def test_lint_script_detects_invalid_zones_and_fc():
    lex = build_lexicon("de_anubis", ["BombsiteA", "BombsiteB", "Middle", "TSpawn", "CTSpawn"])
    kill1 = KillEvent(
        t=10.0,
        killer="p1",
        victim="p2",
        killer_side="T",
        zone="InvalidZoneKill",
        weapon="ak47",
        headshot=True,
        traded_within_4s=False,
    )
    kill2 = KillEvent(
        t=15.0,
        killer="p2",
        victim="p1",
        killer_side="CT",
        zone="Middle",
        weapon="m4a1",
        headshot=False,
        traded_within_4s=False,
    )
    rs = RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="c123",
        round_num=1,
        score_t=0,
        score_ct=0,
        t_team_key="TeamA",
        ct_team_key="TeamB",
        economy={},
        beats=[
            Beat(
                label="B+00",
                t=0.0,
                t_form=Formation(zones=[(1, "InvalidBeatZone")]),
                ct_form=Formation(zones=[(1, "CTSpawn")]),
            )
        ],
        kills=[kill1, kill2],
        utility=[
            UtilEvent(
                t=5.0,
                thrower="p1",
                side="T",
                nade="smoke",
                from_zone="TSpawn",
                to_zone="InvalidUtilZone",
            )
        ],
        plant=None,
        first_contact=kill2,  # mismatch: kill2 != kill1 (earliest kill)
        winner="T",
        reason="ct_killed",
        clock_used_s=60.0,
        movements=[
            MovementLine(
                player="p1",
                side="T",
                role_hint="pack",
                sentence="p1(T): TSpawn > InvalidMovementZone(20s)",
            )
        ],
    )
    problems = lint_script(rs, lex)
    assert any("InvalidBeatZone" in p for p in problems)
    assert any("InvalidZoneKill" in p for p in problems)
    assert any("InvalidUtilZone" in p for p in problems)
    assert any("InvalidMovementZone" in p for p in problems)
    assert any("First contact" in p for p in problems)


@pytest.mark.demo
def test_serialize_match_full(anubis_bundle):
    scripts = serialize_match(**anubis_bundle)
    assert len(scripts) == 30
    r1 = scripts[0]
    assert r1.first_contact is not None
    text = r1.to_text()
    assert text.splitlines()[0].startswith("R1 [")

    # determinism (spec §6.2: same demo -> same script)
    again = serialize_match(**anubis_bundle)
    assert [s.to_json() for s in scripts] == [s.to_json() for s in again]

    # linter: zero violations across the corpus
    problems = [p for s in scripts for p in lint_script(s, anubis_bundle["lex"])]
    assert problems == []

    # p95 token budget <= 450
    p95 = np.percentile([len(s.to_text()) / 4 for s in scripts], 95)
    assert p95 <= 450


def test_roundscript_to_text_options():
    kill = KillEvent(
        t=10.0,
        killer="donk",
        victim="b1",
        killer_side="T",
        zone="Middle",
        weapon="ak47",
        headshot=True,
        traded_within_4s=True,
    )
    plant = PlantEvent(
        t=45.0,
        site="BombsiteA",
        planter="magixx",
        alive_t=3,
        alive_ct=2,
    )
    rs = RoundScript(
        match_id="m1",
        map_name="de_anubis",
        card_checksum="c123",
        round_num=7,
        score_t=3,
        score_ct=3,
        t_team_key="TeamA",
        ct_team_key="TeamB",
        economy={},
        beats=[
            Beat(
                label="B+00",
                t=0.0,
                t_form=Formation(zones=[(2, "Middle"), (1, "TSpawn")]),
                ct_form=Formation(zones=[(1, "CTSpawn")]),
            )
        ],
        kills=[kill],
        utility=[
            UtilEvent(
                t=12.0,
                thrower="donk",
                side="T",
                nade="smoke",
                from_zone="TSpawn",
                to_zone="Window",
                lineup_id="Window-S1",
                blinded=[("b1", 1.8)],
            )
        ],
        plant=plant,
        first_contact=kill,
        winner="T",
        reason="target_bombed",
        clock_used_s=81.0,
        movements=[
            MovementLine(
                player="donk",
                side="T",
                role_hint="entry",
                sentence="donk(T): TSpawn > Middle k(b1)",
            )
        ],
    )
    text_default = rs.to_text()
    assert "[traded]" in text_default
    assert "PL: magixx planted @BombsiteA" in text_default
    assert "12s: donk(T) smoke TSpawn>Window [Window-S1] (blinded: b1 1.8s)" in text_default
    assert "END T target_bombed @1:21" in text_default
    assert "donk(T): TSpawn" not in text_default  # movements not included by default

    text_with_mov = rs.to_text(include_movements=True)
    assert "donk(T): TSpawn > Middle k(b1)" in text_with_mov


@pytest.mark.demo
def test_serialize_round_standalone(anubis_bundle):
    r1 = serialize_round(
        lake=anubis_bundle["lake"],
        mapper=anubis_bundle["mapper"],
        lex=anubis_bundle["lex"],
        card_checksum=anubis_bundle["card_checksum"],
        round_num=1,
    )
    assert r1.round_num == 1
    assert r1.first_contact is not None
    assert len(r1.beats) > 0
    assert len(r1.movements) == 10
