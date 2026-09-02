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
