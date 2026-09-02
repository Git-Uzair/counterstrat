import polars as pl
import pytest

from counterstrat.mapcard.transitions import zone_graph


def test_zone_graph_synthetic():
    # one player walking A->B->C at 4 Hz, 2 s per zone
    df = pl.DataFrame(
        {
            "steamid": [1] * 24,
            "round_num": [1] * 24,
            "tick": list(range(0, 24 * 16, 16)),
            "clock_s": [t / 4 for t in range(24)],
            "last_place_name": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "is_alive": [True] * 24,
        }
    )
    g = zone_graph(df)
    assert ("A", "B") in g.edges and ("B", "C") in g.edges
    assert ("A", "C") not in g.edges  # no skip edges
    assert g.edges[("A", "B")].n == 1
    assert g.edges[("A", "B")].median_transit_s == 2.0


def test_zone_graph_ignores_dead_and_blank_zones():
    df = pl.DataFrame(
        {
            "steamid": [1] * 6,
            "round_num": [1] * 6,
            "tick": [0, 16, 32, 48, 64, 80],
            "clock_s": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
            "last_place_name": ["A", "A", None, "B", "", "C"],
            "is_alive": [True, True, True, True, True, False],
        }
    )
    g = zone_graph(df)
    # A -> B straddles a dropped sample, C's row is dead: nothing survives
    assert g.edges == {}


def test_zone_graph_empty():
    df = pl.DataFrame(
        schema={
            "steamid": pl.Int64,
            "round_num": pl.Int64,
            "tick": pl.Int64,
            "clock_s": pl.Float64,
            "last_place_name": pl.String,
            "is_alive": pl.Boolean,
        }
    )
    assert zone_graph(df).edges == {}


def test_zone_graph_caps_transit():
    df = pl.DataFrame(
        {
            "steamid": [1] * 4,
            "round_num": [1] * 4,
            "tick": [0, 16, 32, 48],
            "clock_s": [0.0, 100.0, 100.25, 100.5],
            "last_place_name": ["A", "A", "A", "B"],
            "is_alive": [True] * 4,
        }
    )
    g = zone_graph(df)
    assert g.edges[("A", "B")].median_transit_s == 30.0


@pytest.mark.demo
def test_zone_graph_anubis(anubis_lake):
    g = zone_graph(pl.read_parquet(anubis_lake.ticks))
    assert len(g.edges) > 20
    assert all(0.0 < e.median_transit_s <= 30.0 for e in g.edges.values())
    assert all(zone_from != zone_to for zone_from, zone_to in g.edges)
    # a walk across mid is a common, quick transition on de_anubis
    assert sum(e.n for e in g.edges.values()) > 1000
