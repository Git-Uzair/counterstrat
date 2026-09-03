import pytest

from counterstrat.corpus import register_demo
from counterstrat.lake.duck import connect_lake
from counterstrat.lake.extract import extract_lake


@pytest.mark.demo
def test_duck_views(demo_path, tmp_path):
    rec = register_demo(demo_path, tmp_path / "corpus.jsonl")
    extract_lake(rec, tmp_path / "lake")
    con = connect_lake(tmp_path / "lake")
    n = con.sql("select count(*) from rounds").fetchone()[0]
    assert n == 30
    places = con.sql("select count(distinct last_place_name) from ticks").fetchone()[0]
    assert places == 28  # verified inventory


def test_connect_lake_unions_mixed_schemas(tmp_path):
    """Re-zoned maps gain a place_default tick column; untouched maps don't.
    The glob views must survive the mixed schemas."""
    import polars as pl

    (tmp_path / "m1").mkdir()
    (tmp_path / "m2").mkdir()
    pl.DataFrame({"X": [1.0], "last_place_name": ["Middle"], "match_id": ["m1"]}).write_parquet(
        tmp_path / "m1" / "ticks.parquet"
    )
    pl.DataFrame(
        {
            "X": [2.0],
            "last_place_name": ["Sandbags"],
            "place_default": ["Middle"],
            "match_id": ["m2"],
        }
    ).write_parquet(tmp_path / "m2" / "ticks.parquet")
    for name in (
        "rounds",
        "kills",
        "damages",
        "shots",
        "grenades",
        "smokes",
        "infernos",
        "bomb",
        "item_purchase",
        "rosters",
    ):
        pl.DataFrame({"match_id": ["m1"]}).write_parquet(tmp_path / "m1" / f"{name}.parquet")

    con = connect_lake(tmp_path)
    rows = con.sql(
        "select last_place_name, place_default from ticks order by last_place_name"
    ).fetchall()
    assert rows == [("Middle", None), ("Sandbags", "Middle")]
