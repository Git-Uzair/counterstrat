import duckdb
from pydantic import BaseModel


class ReconcileReport(BaseModel):
    in_vpk_not_demo: list[str]
    in_demo_not_vpk: list[str]
    ok: bool


def reconcile_names(vpk_names: list[str], demo_names: list[str]) -> ReconcileReport:
    in_demo_not_vpk = sorted(set(demo_names) - set(vpk_names))
    in_vpk_not_demo = sorted(set(vpk_names) - set(demo_names))
    ok = len(in_demo_not_vpk) == 0
    return ReconcileReport(
        in_vpk_not_demo=in_vpk_not_demo,
        in_demo_not_vpk=in_demo_not_vpk,
        ok=ok,
    )


def reconcile(
    lexicon_names: list[str],
    lake_con: duckdb.DuckDBPyConnection,
    map_name: str = "",
) -> ReconcileReport:
    rows = lake_con.sql(
        "select distinct last_place_name from ticks where last_place_name is not null and last_place_name != ''"
    ).fetchall()
    demo_places = [str(r[0]) for r in rows]
    return reconcile_names(lexicon_names, demo_places)
