from pathlib import Path

import duckdb

TABLES = [
    "rounds",
    "kills",
    "damages",
    "shots",
    "grenades",
    "smokes",
    "infernos",
    "bomb",
    "item_purchase",
    "ticks",
    "rosters",
]


def connect_lake(lake_root: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        pattern = str(lake_root / "*" / f"{t}.parquet").replace("\\", "/")
        con.sql(f"create or replace view {t} as select * from read_parquet('{pattern}')")
    return con
