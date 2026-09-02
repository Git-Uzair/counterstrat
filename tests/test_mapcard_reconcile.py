import duckdb
import pytest

from counterstrat.corpus import register_demo
from counterstrat.lake.duck import connect_lake
from counterstrat.lake.extract import extract_lake
from counterstrat.mapcard.reconcile import reconcile, reconcile_names
from counterstrat.mapcard.vents import parse_places, unique_places
from counterstrat.mapcard.vrf import extract_map_assets


def test_reconcile_policy():
    r = reconcile_names(["A", "B"], ["A"])
    assert r.ok and r.in_vpk_not_demo == ["B"]
    r2 = reconcile_names(["A"], ["A", "Ghost"])
    assert not r2.ok and r2.in_demo_not_vpk == ["Ghost"]


def test_reconcile_duckdb_synthetic():
    con = duckdb.connect()
    con.sql(
        "create table ticks as select 'A' as last_place_name "
        "union all select 'Ghost' as last_place_name "
        "union all select '' as last_place_name "
        "union all select null as last_place_name"
    )
    r = reconcile(["A"], con, "de_dust2")
    assert not r.ok
    assert r.in_demo_not_vpk == ["Ghost"]
    assert r.in_vpk_not_demo == []


@pytest.mark.demo
def test_reconcile_anubis(demo_path, anubis_vpk, vrf_cli, tmp_path):
    rec = register_demo(demo_path, tmp_path / "corpus.jsonl")
    extract_lake(rec, tmp_path / "lake")
    con = connect_lake(tmp_path / "lake")

    assets = extract_map_assets(anubis_vpk, vrf_cli, tmp_path / "vpk_out")
    vols = parse_places(assets.vents)
    names = unique_places(vols)

    report = reconcile(names, con, "de_anubis")
    assert report.ok and not report.in_vpk_not_demo and not report.in_demo_not_vpk
