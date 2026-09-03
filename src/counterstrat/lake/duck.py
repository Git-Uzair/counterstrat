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
        # union_by_name: re-zoned maps carry an extra place_default tick
        # column; matches missing a column read it as NULL instead of
        # breaking every view.
        con.sql(
            f"create or replace view {t} as "
            f"select * from read_parquet('{pattern}', union_by_name=true)"
        )
    return con
