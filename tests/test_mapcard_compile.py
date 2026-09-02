import polars as pl
import pytest

from counterstrat.mapcard.compile import MapCard, compile_card
from counterstrat.mapcard.lexicon import ZoneDef, build_lexicon, get_default_overlay_path
from counterstrat.mapcard.transitions import EdgeStat, ZoneGraph, zone_graph
from counterstrat.mapcard.vents import parse_places, unique_places


@pytest.fixture
def synthetic_bundle():
    zones = {
        "BombsiteA": ZoneDef(
            id="BombsiteA", aliases=["A site"], tags=["site"], engine_place="BombsiteA"
        ),
        "BombsiteB": ZoneDef(
            id="BombsiteB", aliases=["B site"], tags=["site"], engine_place="BombsiteB"
        ),
        "Middle": ZoneDef(
            id="Middle", aliases=["mid"], tags=["mid_control"], engine_place="Middle"
        ),
        "CTSpawn": ZoneDef(id="CTSpawn", aliases=[], tags=[], engine_place="CTSpawn"),
        "TSpawn": ZoneDef(id="TSpawn", aliases=[], tags=[], engine_place="TSpawn"),
    }
    from counterstrat.mapcard.lexicon import Lexicon, _compute_checksum

    lex = Lexicon(map_name="de_dust2", zones=zones, checksum=_compute_checksum("de_dust2", zones))

    # Construct synthetic ticks
    # X in [0, 900], Y in [0, 900], Z in [0, 100]
    ticks = pl.DataFrame(
        {
            "steamid": [1] * 10 + [2] * 10,
            "round_num": [1] * 20,
            "tick": list(range(0, 20 * 16, 16)),
            "clock_s": [float(i) for i in range(20)],
            "last_place_name": (
                ["CTSpawn"] * 2
                + ["Middle"] * 3
                + ["BombsiteA"] * 5
                + ["TSpawn"] * 2
                + ["Middle"] * 3
                + ["BombsiteB"] * 5
            ),
            "X": [100.0] * 2 + [450.0] * 3 + [800.0] * 5 + [100.0] * 2 + [450.0] * 3 + [850.0] * 5,
            "Y": [100.0] * 2 + [450.0] * 3 + [800.0] * 5 + [100.0] * 2 + [450.0] * 3 + [850.0] * 5,
            "Z": [10.0] * 2 + [50.0] * 3 + [90.0] * 5 + [10.0] * 2 + [50.0] * 3 + [90.0] * 5,
            "team_name": ["CT"] * 10 + ["TERRORIST"] * 10,
            "is_alive": [True] * 20,
        }
    )
    rounds = pl.DataFrame(
        {
            "round_num": [1],
            "start": [0],
            "freeze_end": [0],
            "end": [320],
        }
    )
    # Graph with edges
    graph = ZoneGraph(
        edges={
            ("BombsiteA", "Middle"): EdgeStat(n=10, median_transit_s=4.0),
            ("Middle", "BombsiteB"): EdgeStat(n=10, median_transit_s=5.0),
            ("BombsiteA", "CTSpawn"): EdgeStat(n=8, median_transit_s=6.0),
            ("CTSpawn", "BombsiteB"): EdgeStat(n=8, median_transit_s=7.0),
            ("BombsiteB", "Middle"): EdgeStat(n=10, median_transit_s=5.0),
            ("Middle", "BombsiteA"): EdgeStat(n=10, median_transit_s=4.0),
            ("BombsiteB", "CTSpawn"): EdgeStat(n=8, median_transit_s=7.0),
            ("CTSpawn", "BombsiteA"): EdgeStat(n=8, median_transit_s=6.0),
        }
    )
    return {
        "lexicon": lex,
        "graph": graph,
        "ticks": ticks,
        "rounds": rounds,
        "map_name": "de_dust2",
        "patch_version": "14178",
    }


@pytest.fixture(scope="session")
def anubis_bundle(anubis_assets, anubis_lake):
    places = unique_places(parse_places(anubis_assets.vents))
    overlay = get_default_overlay_path("de_anubis")
    lexicon = build_lexicon("de_anubis", places, overlay)
    ticks = pl.read_parquet(anubis_lake.ticks)
    rounds = pl.read_parquet(anubis_lake.rounds)
    graph = zone_graph(ticks)
    return {
        "lexicon": lexicon,
        "graph": graph,
        "ticks": ticks,
        "rounds": rounds,
        "map_name": "de_anubis",
        "patch_version": "14178",
    }


@pytest.mark.demo
def test_compile_anubis_card(anubis_bundle):  # lexicon+graph+lake fixture
    card = compile_card(**anubis_bundle)
    assert isinstance(card, MapCard)
    y = card.to_yaml()
    assert "map: de_anubis" in y and "topology:" in y
    assert len(y) / 4 <= 4000  # spec §6.5 budget
    # incident encoding: a known heavy edge exists in some direction
    assert card.topology.get("Middle")
    # rotates precomputed between the two sites
    assert any(r["from"] == "BombsiteA" and r["to"] == "BombsiteB" for r in card.rotates)
    assert card.checksum and card.game_version == "14178"


def test_card_determinism_synthetic(synthetic_bundle):
    a = compile_card(**synthetic_bundle).to_yaml()
    b = compile_card(**synthetic_bundle).to_yaml()
    assert a == b


def test_card_structure_synthetic(synthetic_bundle):
    card = compile_card(**synthetic_bundle)
    assert card.map == "de_dust2"
    assert card.card_version == "1.0"
    assert card.game_version == "14178"
    assert card.nav_source == "nav"

    # Objectives
    assert card.objectives["sites"] == ["BombsiteA", "BombsiteB"]
    assert card.objectives["round_seconds"] == 115
    assert card.objectives["bomb_seconds"] == 40
    assert card.objectives["defuse_seconds"] == [10, 5]

    # Sightlines empty
    assert card.sightlines == []

    # Frame
    assert "bbox" in card.frame
    assert "z_median" in card.frame

    # Quadrant and elevation
    assert "quadrant" in card.zones["Middle"]
    assert "elevation" in card.zones["Middle"]

    # Topology has heavy edges (n >= 5)
    assert "BombsiteA" in card.topology
    assert card.topology["BombsiteA"]["Middle"] == 4.0

    # Rotates: 2 paths from BombsiteA to BombsiteB
    a_to_b = [r for r in card.rotates if r["from"] == "BombsiteA" and r["to"] == "BombsiteB"]
    assert len(a_to_b) == 2
    assert a_to_b[0]["run_s"] == 9.0  # via Middle: 4.0 + 5.0
    assert a_to_b[0]["via"] == ["Middle"]
    assert a_to_b[1]["run_s"] == 13.0  # via CTSpawn: 6.0 + 7.0
    assert a_to_b[1]["via"] == ["CTSpawn"]

    # Timings
    assert "CT" in card.timings and "T" in card.timings
    assert card.timings["CT"]["Middle"] == 2.0
    assert card.timings["T"]["Middle"] == 12.0


def test_card_empty_data():
    from counterstrat.mapcard.lexicon import Lexicon, _compute_checksum

    lex = Lexicon(map_name="de_empty", zones={}, checksum=_compute_checksum("de_empty", {}))
    card = compile_card(
        lexicon=lex,
        graph=ZoneGraph(edges={}),
        ticks=pl.DataFrame(),
        rounds=pl.DataFrame(),
        map_name="de_empty",
        patch_version="1.0",
    )
    assert card.map == "de_empty"
    assert card.zones == {}
    assert card.topology == {}
    assert card.rotates == []
    assert card.timings == {"CT": {}, "T": {}}
    assert card.objectives["sites"] == []
    y = card.to_yaml()
    assert "map: de_empty" in y
    assert len(card.checksum) == 12


def test_card_token_trimming(synthetic_bundle):
    # Create large alias lists to blow past 4000 tokens (16000 chars)
    bundle = dict(synthetic_bundle)
    long_aliases = [f"very_long_alias_name_{i}" * 10 for i in range(200)]
    zones = dict(bundle["lexicon"].zones)
    zones["Untagged"] = ZoneDef(
        id="Untagged",
        aliases=long_aliases,
        tags=[],
        engine_place="Untagged",
    )
    from counterstrat.mapcard.lexicon import Lexicon, _compute_checksum

    bundle["lexicon"] = Lexicon(
        map_name=bundle["map_name"],
        zones=zones,
        checksum=_compute_checksum(bundle["map_name"], zones),
    )
    card = compile_card(**bundle)
    y = card.to_yaml()
    assert len(y) / 4 <= 4000
