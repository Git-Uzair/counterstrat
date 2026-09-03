# User-Defined Callout Zones Implementation Plan

> **For agentic workers:** execute with executing-plans (inline) or
> subagent-driven-development, task by task, TDD every task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Users create their own callouts by pointing at the radar; those
zones become first-class canonical zones in the data itself, so every demo
extraction, mined tendency, SQL query, label anchor, and LLM prompt speaks
the user's callout at the user's position - one vocabulary, one geometry,
zero translation layers.

**Architecture:** A custom zone is a named sphere (world-space center +
radius, Z scaled x2 like `ZoneMapper`) stored per map in
`data/mapcards/<map>/zones.json`. Saving zones triggers a background rebuild
that BAKES the effective zone into the lake: every tick inside a region gets
`last_place_name = <custom name>` (the game's original name is preserved in
a new `place_default` column, making re-zoning idempotent and reversible).
Everything downstream - RoundScripts, ZoneMapper, map card, miners, anchors,
`format_zone_map`, lint vocabulary, duckdb SQL - already reads
`last_place_name`, so correctness propagates structurally instead of by
per-consumer translation. This is the whole design bet: one writer, many
readers, no reader can be "forgotten".

**Tech Stack:** Python 3.13 + uv, polars, FastAPI + BackgroundTasks/JobState
(existing job infra), duckdb, vanilla JS editor. Tests: pytest
(`uv run python -m pytest -q`), lint: `uv run ruff check .` (line-length 100).

**Spec:** the request (2026-09-03 session): "add new callouts by pointing at
a location which also propagates to LLM calls and positions where the player
was in the demo... LLM prompt map, extractions, all should be as close as
possible. The anti-strat works only if the mappings are all correct."

## Global Constraints

- Never mutate raw demo files; the lake rewrite must be idempotent and
  reversible (`place_default` preserves the game vocabulary).
- Zone names share one namespace with game zones AND alias values; collisions
  rejected in both directions (`aliases.py` <-> `customzones.py`).
- All membership math uses scaled-3D distance with `Z_SCALE = 2.0`
  (mirrors `roundscript/utility.py:32` and `ZoneMapper.z_scale`) so stacked
  zones (nuke Hut/HutRoof) never bleed into each other.
- Rebuild scope is a single map; runs as a background job (existing
  `JobState` stages `queued -> serializing -> mining -> done|error`);
  the UI polls `/api/jobs/{job_id}`.
- `uv run python -m pytest -q` green and `uv run ruff check .` clean before
  every commit. One commit per task. Static assets cache-bust to `?v=5`
  when JS/CSS change.
- The user's server must be RESTARTED after landing (stale-server gotcha).

## Context (verified this session, path:line)

- Ticks land in `lake/<match>/ticks.parquet` via `lake/extract.py:154`
  (`extract_lake`); `last_place_name` is a parsed tick prop
  (`lake/extract.py:17`).
- RoundScripts read zones straight off ticks (`roundscript/movement.py:161`,
  `roundscript/beats.py:19`); grenade positions classify through
  `ZoneMapper.fit(ticks)` KNN (`mapcard/zones.py:20`, fitted at
  `web/ingest.py:207`); `serialize_match(lake, mapper, lex, card_checksum)`
  (`web/ingest.py:238`).
- Card compile: `compile_card(lexicon, graph, ticks, rounds, map_name,
  patch_version)` (`web/ingest.py:187`); lexicon places come from VPK vents
  when a card is compiled (`web/ingest.py:179`) - custom zones must be
  unioned in or the card (and therefore lint + prompts) never sees them.
- duckdb views glob ALL matches: `read_parquet('lake/*/ticks.parquet')`
  (`lake/duck.py:24`) - adding a column to only some files breaks the view
  unless `union_by_name=true`.
- Rebuild machinery exists: `web/maintenance.py:43` (`rebuild_artifacts`:
  recluster + re-mine + prune), `web/maintenance.py:21`
  (`mine_team_artifacts`).
- Jobs: `JobState` + `save_job_state`/`load_job_state`
  (`web/ingest.py:32-58`), poller `GET /api/jobs/{job_id}`
  (`web/routes.py:172`), started via `background_tasks.add_task`
  (`web/routes.py:168`). FastAPI TestClient runs background tasks
  synchronously after the response - tests can assert the completed state.
- Anchors: `_tick_positions` median+snap on dominant level
  (`web/routes.py:525`), `map_zone_anchors` (`web/routes.py:600`) feeds both
  the editor (`GET /api/maps/{map}/callouts`, `web/routes.py:610`) and
  `format_zone_map` (`llm/prompts.py:10`) into all three prompts.
- Aliases: `save_aliases(data_root, map_name, aliases, valid_zones)`
  validates names with `_ALIAS_RE` and rejects collisions
  (`src/counterstrat/aliases.py:34`).
- Dossier cache is a bare file served if present (`web/routes.py` get_report:
  `teambooks/<team>/<map>/dossier.md`); insights cache freshness =
  `generated_from` + `alias_fp` only - a zones rebuild MUST delete both per
  affected (team, map) or stale zone vocabulary survives.
- Editor: `static/callouts.js` renders labels at `(u*100%, v*100%)` from the
  callouts endpoint; assets at `?v=4` (`static/index.html:7,271`).
- Real-data fixtures: session-scoped `anubis_lake` (real demo -> lake) in
  `tests/conftest.py:59`; synthetic builders `build_synthetic_card`
  (zones incl. `Middle`, `Water`, `BombsiteA`), `build_synthetic_scripts`,
  `SYNTHETIC_TEAM = "abc"`; callout test seeds `_seed_calibration`,
  `_seed_match` in `tests/test_web_callouts.py`.
- Calibration: `pos_x/pos_y/scale/image_px/lower_altitude_max`;
  `game_to_norm`/`pixel_to_game`/`is_lower_level`/`level_expr`
  (`radar/coords.py`).

## Assumptions (stated, not asked)

1. Point + radius (a scaled sphere) is the v1 region shape; polygons are out
   of scope. Radius editable 64-600 units, default 150.
2. Custom zones require tick grounding: creation is rejected where no player
   data exists near the click (unplayed VPK-only maps get a clear 400).
   A callout the data cannot ground would break the "mappings all correct"
   requirement.
3. Overlapping custom regions are rejected at save (scaled-3D center
   distance < r1 + r2 on any level). Deterministic and simple beats
   precedence rules.
4. Renaming a custom zone = delete + create (name is identity in v1).
5. Aliases MAY target custom zones (harmless; renamer chain unchanged), but
   names may not collide across {game zones} ∪ {alias values} ∪ {custom
   zone names}.
6. Old chat sessions keep their pre-rebuild system prompt in memory until
   restored/rebuilt - same accepted staleness as demo deletion today.
7. `predict.py`/`quiz.py` (eval-only paths) are out of scope.

---
### Task 1: customzones core - model, storage, validation, re-zoning

**Files:**
- Create: `src/counterstrat/customzones.py`
- Test: `tests/test_customzones.py`

**Interfaces:**
- Produces (later tasks import these exact names from
  `counterstrat.customzones`):
  - `Z_SCALE: float = 2.0`
  - `class CustomZone(BaseModel)`: `name: str`, `x: float`, `y: float`,
    `z: float`, `level: str = "default"`, `radius: float = 150.0`
  - `load_custom_zones(data_root: Path, map_name: str) -> list[CustomZone]`
  - `save_custom_zones(data_root: Path, map_name: str,
    zones: list[CustomZone], *, reserved: set[str]) -> list[CustomZone]`
    (raises `ValueError` on any invalid input; `reserved` = game/card zone
    names ∪ alias values, built by the caller)
  - `rezone_ticks(df: pl.DataFrame, zones: list[CustomZone]) -> pl.DataFrame`
    (adds/uses `place_default`; recomputes `last_place_name`; idempotent;
    `zones=[]` restores the game vocabulary)
  - Storage file: `data/mapcards/<map>/zones.json` (sorted JSON list)

- [ ] **Step 1: Write the failing tests**

```python
"""User-defined callout zones: storage, validation, and lake re-zoning."""

import json
from pathlib import Path

import polars as pl
import pytest

from counterstrat.customzones import (
    CustomZone,
    load_custom_zones,
    rezone_ticks,
    save_custom_zones,
)


def _ticks() -> pl.DataFrame:
    # Two game zones; one tick sits inside the future "Sandbags" sphere,
    # one directly above it on a roof (same XY, Z +200), one far away.
    return pl.DataFrame(
        {
            "X": [100.0, 100.0, 900.0],
            "Y": [100.0, 100.0, 900.0],
            "Z": [0.0, 200.0, 0.0],
            "last_place_name": ["Middle", "MidRoof", "BombsiteA"],
            "is_alive": [True, True, True],
        }
    )


def test_rezone_assigns_inside_preserves_default_and_restores():
    zones = [CustomZone(name="Sandbags", x=110.0, y=110.0, z=0.0, radius=50.0)]
    out = rezone_ticks(_ticks(), zones)
    assert out["last_place_name"].to_list() == ["Sandbags", "MidRoof", "BombsiteA"]
    # The game vocabulary survives in place_default.
    assert out["place_default"].to_list() == ["Middle", "MidRoof", "BombsiteA"]
    # Idempotent: re-zoning re-zoned ticks changes nothing.
    again = rezone_ticks(out, zones)
    assert again["last_place_name"].to_list() == out["last_place_name"].to_list()
    # Reversible: an empty zone list restores the game names.
    restored = rezone_ticks(out, [])
    assert restored["last_place_name"].to_list() == ["Middle", "MidRoof", "BombsiteA"]


def test_rezone_z_scaled_membership_excludes_stacked_zones():
    # Radius 150 covers XY distance 14 but not the roof 200 above:
    # scaled dz = 2*200 = 400 > 150. Nuke Hut/HutRoof stay distinct.
    zones = [CustomZone(name="Sandbags", x=110.0, y=110.0, z=0.0, radius=150.0)]
    out = rezone_ticks(_ticks(), zones)
    assert out["last_place_name"].to_list() == ["Sandbags", "MidRoof", "BombsiteA"]


def test_rezone_without_coords_or_rows_is_a_noop():
    empty = pl.DataFrame({"last_place_name": [], "X": [], "Y": [], "Z": []})
    assert rezone_ticks(empty, [CustomZone(name="A", x=0, y=0, z=0)]).is_empty()
    no_coords = pl.DataFrame({"last_place_name": ["Middle"]})
    out = rezone_ticks(no_coords, [CustomZone(name="A", x=0, y=0, z=0)])
    assert out["last_place_name"].to_list() == ["Middle"]


def test_save_and_load_roundtrip(tmp_path: Path):
    z = CustomZone(name="Sandbags", x=1.0, y=2.0, z=3.0, level="lower", radius=100.0)
    saved = save_custom_zones(tmp_path, "de_test", [z], reserved={"Middle"})
    assert [s.name for s in saved] == ["Sandbags"]
    loaded = load_custom_zones(tmp_path, "de_test")
    assert loaded == saved
    # Torn/missing files load as empty, like aliases.
    (tmp_path / "mapcards" / "de_test" / "zones.json").write_text("{not json", encoding="utf-8")
    assert load_custom_zones(tmp_path, "de_test") == []
    assert load_custom_zones(tmp_path, "de_ghost") == []


def test_save_validation_matrix(tmp_path: Path):
    ok = dict(x=0.0, y=0.0, z=0.0)
    reserved = {"Middle", "Mid"}  # a game zone and an alias value
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="Middle", **ok)], reserved=reserved)
    with pytest.raises(ValueError, match="reserved"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="Mid", **ok)], reserved=reserved)
    with pytest.raises(ValueError, match="invalid characters"):
        save_custom_zones(tmp_path, "m", [CustomZone(name="a`b", **ok)], reserved=set())
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        save_custom_zones(
            tmp_path,
            "m",
            [CustomZone(name="A", **ok), CustomZone(name="A", x=9000.0, y=9000.0, z=0.0)],
            reserved=set(),
        )
    with pytest.raises(ValueError, match="radius"):
        save_custom_zones(
            tmp_path, "m", [CustomZone(name="A", radius=20.0, **ok)], reserved=set()
        )
    with pytest.raises(ValueError, match="level"):
        save_custom_zones(
            tmp_path, "m", [CustomZone(name="A", level="attic", **ok)], reserved=set()
        )
    with pytest.raises(ValueError, match="overlap"):
        save_custom_zones(
            tmp_path,
            "m",
            [
                CustomZone(name="A", x=0.0, y=0.0, z=0.0, radius=150.0),
                CustomZone(name="B", x=200.0, y=0.0, z=0.0, radius=150.0),
            ],
            reserved=set(),
        )
    # Same XY but far apart vertically is NOT an overlap (scaled 3D).
    saved = save_custom_zones(
        tmp_path,
        "m",
        [
            CustomZone(name="A", x=0.0, y=0.0, z=0.0, radius=100.0),
            CustomZone(name="B", x=0.0, y=0.0, z=300.0, radius=100.0),
        ],
        reserved=set(),
    )
    assert [s.name for s in saved] == ["A", "B"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_customzones.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'counterstrat.customzones'`

- [ ] **Step 3: Implement `src/counterstrat/customzones.py`**

```python
"""User-defined callout zones: spheres baked into the lake vocabulary.

A custom zone is a named world-space sphere (Z scaled x2, mirroring
ZoneMapper verticality weighting). ``rezone_ticks`` rewrites
``last_place_name`` for ticks inside a region while preserving the game's
own name in ``place_default`` - idempotent, reversible, and applied at the
data layer so every consumer (scripts, miners, card, anchors, SQL, prompts)
inherits the user's vocabulary without translation.
"""

import json
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from counterstrat.aliases import _ALIAS_RE

Z_SCALE = 2.0  # verticality weight, mirrors ZoneMapper.z_scale
MIN_RADIUS = 64.0
MAX_RADIUS = 600.0
_LEVELS = {"default", "lower"}


class CustomZone(BaseModel):
    name: str
    x: float
    y: float
    z: float
    level: str = "default"
    radius: float = 150.0


def _zones_path(data_root: Path, map_name: str) -> Path:
    return data_root / "mapcards" / map_name / "zones.json"


def load_custom_zones(data_root: Path, map_name: str) -> list[CustomZone]:
    path = _zones_path(data_root, map_name)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [CustomZone(**z) for z in raw] if isinstance(raw, list) else []
    except Exception:  # noqa: BLE001 - a torn file must not kill the editor
        return []


def save_custom_zones(
    data_root: Path, map_name: str, zones: list[CustomZone], *, reserved: set[str]
) -> list[CustomZone]:
    """Validate and persist; raises ValueError with a user-facing message."""
    seen: set[str] = set()
    for z in zones:
        if not _ALIAS_RE.match(z.name):
            raise ValueError(f"Zone name {z.name!r} contains invalid characters")
        if z.name in reserved:
            raise ValueError(f"Zone name {z.name!r} is reserved (existing zone or callout)")
        if z.name in seen:
            raise ValueError(f"Duplicate zone name {z.name!r}")
        seen.add(z.name)
        if not (MIN_RADIUS <= z.radius <= MAX_RADIUS):
            raise ValueError(f"radius must be {MIN_RADIUS:.0f}-{MAX_RADIUS:.0f} units")
        if z.level not in _LEVELS:
            raise ValueError(f"Unknown level {z.level!r}")
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z - b.z) * Z_SCALE) ** 2
            if d2 < (a.radius + b.radius) ** 2:
                raise ValueError(f"Zones {a.name!r} and {b.name!r} overlap - shrink or move one")

    ordered = sorted(zones, key=lambda z: z.name)
    path = _zones_path(data_root, map_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([z.model_dump() for z in ordered], indent=2), encoding="utf-8"
    )
    return ordered


def rezone_ticks(df: pl.DataFrame, zones: list[CustomZone]) -> pl.DataFrame:
    """Effective vocabulary: custom name inside a sphere, game name outside."""
    if "last_place_name" not in df.columns:
        return df
    if "place_default" not in df.columns:
        df = df.with_columns(pl.col("last_place_name").alias("place_default"))
    if not {"X", "Y", "Z"} <= set(df.columns) or df.is_empty():
        return df.with_columns(pl.col("place_default").alias("last_place_name"))
    expr = pl.col("place_default")
    # Descending name order wraps ascending names outermost: deterministic
    # precedence even though overlaps are rejected at save time.
    for z in sorted(zones, key=lambda z: z.name, reverse=True):
        d2 = (
            (pl.col("X") - z.x) ** 2
            + (pl.col("Y") - z.y) ** 2
            + ((pl.col("Z") - z.z) * Z_SCALE) ** 2
        )
        expr = pl.when(d2 <= z.radius**2).then(pl.lit(z.name)).otherwise(expr)
    return df.with_columns(expr.alias("last_place_name"))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/test_customzones.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
git add src/counterstrat/customzones.py tests/test_customzones.py
git commit -m "feat: custom callout zones - model, validated storage, lake re-zoning"
```

---

### Task 2: duckdb lake views survive per-map schema drift

Only edited maps gain the `place_default` column; the ticks view globs every
match (`lake/duck.py:24`) and would otherwise error on mixed schemas.

**Files:**
- Modify: `src/counterstrat/lake/duck.py:24`
- Test: `tests/test_lake_duck.py` (append one test)

**Interfaces:**
- Consumes: nothing new. Produces: `connect_lake` tolerant of parquet files
  whose column sets differ (missing columns read as NULL).

- [ ] **Step 1: Write the failing test** (append to `tests/test_lake_duck.py`)

```python
def test_connect_lake_unions_mixed_schemas(tmp_path):
    import polars as pl

    from counterstrat.lake.duck import connect_lake

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
        "rounds", "kills", "damages", "shots", "grenades",
        "smokes", "infernos", "bomb", "item_purchase", "rosters",
    ):
        pl.DataFrame({"match_id": ["m1"]}).write_parquet(tmp_path / "m1" / f"{name}.parquet")

    con = connect_lake(tmp_path)
    rows = con.sql(
        "select last_place_name, place_default from ticks order by last_place_name"
    ).fetchall()
    assert rows == [("Middle", None), ("Sandbags", "Middle")]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_lake_duck.py::test_connect_lake_unions_mixed_schemas -q`
Expected: FAIL (duckdb schema mismatch error or missing `place_default` binder error)

- [ ] **Step 3: Implement** - in `connect_lake`, change the view SQL to:

```python
        con.sql(
            f"create or replace view {t} as "
            f"select * from read_parquet('{pattern}', union_by_name=true)"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/test_lake_duck.py -q`
Expected: all PASS (existing tests too)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
git add src/counterstrat/lake/duck.py tests/test_lake_duck.py
git commit -m "fix: lake views union by name so per-map tick columns can differ"
```

---
### Task 3: rebuild pipeline - bake zones into the lake, rebuild everything derived

**Files:**
- Modify: `src/counterstrat/web/maintenance.py` (add `_lake_paths`,
  `compile_map_card`, `rebuild_map_zones`, `run_zone_rebuild`)
- Modify: `src/counterstrat/web/ingest.py:172-207` (re-zone fresh ticks at
  ingest; route card compile through `compile_map_card`)
- Test: `tests/test_customzones_rebuild.py` (new)

**Interfaces:**
- Consumes: Task 1 (`load_custom_zones`, `rezone_ticks`, `CustomZone`).
- Produces:
  - `maintenance._lake_paths(data_root: Path, match_id: str) -> LakePaths`
  - `maintenance.compile_map_card(map_name: str, places: list[str],
    ticks_df: pl.DataFrame, rounds_df: pl.DataFrame, patch_version: str)
    -> MapCard | None` (lexicon places = `places` ∪ effective tick places;
    returns None when both are empty)
  - `maintenance.rebuild_map_zones(cfg: AppConfig, map_name: str) -> dict`
    (re-zones every match's ticks atomically, recompiles the card,
    re-serializes scripts, calls `rebuild_artifacts`, deletes stale
    `dossier.md`/`insights.json` for that map)
  - `maintenance.run_zone_rebuild(job_id: str, map_name: str,
    cfg: AppConfig) -> None` (JobState wrapper:
    queued -> serializing -> mining -> done|error)
- Task 4 calls `run_zone_rebuild` via `background_tasks.add_task`.

- [ ] **Step 1: Write the failing tests**

```python
"""Zone rebuild: ticks re-zoned in place, card recompiled, caches pruned."""

import json
from pathlib import Path

import polars as pl
import pytest

from counterstrat.config import AppConfig
from counterstrat.customzones import CustomZone, save_custom_zones
from counterstrat.web.maintenance import compile_map_card, rebuild_map_zones

MAP = "de_test"


@pytest.fixture(autouse=True)
def isolated_repo_root(tmp_path: Path, monkeypatch):
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    return fake_repo


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(data_root=tmp_path / "data")
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    return cfg


def _seed_corpus_and_lake(cfg: AppConfig, match_id: str = "m1") -> Path:
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": match_id,
                "path": f"demos/{match_id}.dem",
                "map_name": MAP,
                "patch_version": "1",
                "demo_version_guid": "g",
                "server_name": "s",
                "registered_at": "2026-09-03T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lake = cfg.data_root / "lake" / match_id
    lake.mkdir(parents=True)
    pl.DataFrame(
        {
            "X": [100.0, 105.0, 900.0],
            "Y": [100.0, 105.0, 900.0],
            "Z": [0.0, 0.0, 0.0],
            "last_place_name": ["Middle", "Middle", "BombsiteA"],
            "is_alive": [True, True, True],
            "match_id": [match_id] * 3,
        }
    ).write_parquet(lake / "ticks.parquet")
    # Empty companion tables: serialize_match sorts rounds by round_num and
    # then loops zero rounds, so kills/grenades/rosters are never touched
    # (rosters.parquet must NOT exist - build_team_clusters skips a missing
    # file but crashes on one without team_key). Real-demo chain is Task 6.
    pl.DataFrame(
        {"match_id": pl.Series([], dtype=pl.String), "round_num": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(lake / "rounds.parquet")
    pl.DataFrame({"match_id": pl.Series([], dtype=pl.String)}).write_parquet(
        lake / "kills.parquet"
    )
    return lake


def test_compile_map_card_includes_custom_zones():
    ticks = pl.DataFrame(
        {
            "X": [0.0, 10.0],
            "Y": [0.0, 10.0],
            "Z": [0.0, 0.0],
            "last_place_name": ["Sandbags", "Middle"],
        }
    )
    card = compile_map_card(MAP, ["Middle", "BombsiteA"], ticks, pl.DataFrame(), "1")
    assert card is not None
    assert {"Sandbags", "Middle", "BombsiteA"} <= set(card.zones.keys())
    assert compile_map_card(MAP, [], pl.DataFrame(), pl.DataFrame(), "1") is None


def test_rebuild_map_zones_bakes_ticks_card_and_prunes_caches(cfg: AppConfig):
    lake = _seed_corpus_and_lake(cfg)
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=102.0, y=102.0, z=0.0, radius=100.0)],
        reserved=set(),
    )
    # Stale LLM caches that must die with the rebuild.
    tb_dir = cfg.data_root / "teambooks" / "team1" / MAP
    tb_dir.mkdir(parents=True)
    (tb_dir / "dossier.md").write_text("stale", encoding="utf-8")
    (tb_dir / "insights.json").write_text("{}", encoding="utf-8")
    other_map = cfg.data_root / "teambooks" / "team1" / "de_other"
    other_map.mkdir(parents=True)
    (other_map / "dossier.md").write_text("keep", encoding="utf-8")

    result = rebuild_map_zones(cfg, MAP)
    assert result["matches"] == 1 and result["zones"] == 1

    ticks = pl.read_parquet(lake / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == ["Sandbags", "Sandbags", "BombsiteA"]
    assert ticks["place_default"].to_list() == ["Middle", "Middle", "BombsiteA"]

    card = (cfg.data_root / "mapcards" / MAP / "card.yaml").read_text(encoding="utf-8")
    assert "Sandbags" in card

    assert not (tb_dir / "dossier.md").exists()
    assert not (tb_dir / "insights.json").exists()
    assert (other_map / "dossier.md").exists()  # other maps untouched

    # Reversibility: delete the zone, rebuild, game names return.
    save_custom_zones(cfg.data_root, MAP, [], reserved=set())
    rebuild_map_zones(cfg, MAP)
    ticks = pl.read_parquet(lake / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == ["Middle", "Middle", "BombsiteA"]


def test_rebuild_is_idempotent(cfg: AppConfig):
    lake = _seed_corpus_and_lake(cfg)
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=102.0, y=102.0, z=0.0, radius=100.0)],
        reserved=set(),
    )
    rebuild_map_zones(cfg, MAP)
    first = pl.read_parquet(lake / "ticks.parquet")
    rebuild_map_zones(cfg, MAP)
    second = pl.read_parquet(lake / "ticks.parquet")
    assert first.equals(second)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_customzones_rebuild.py -q`
Expected: FAIL with `ImportError: cannot import name 'compile_map_card'`

- [ ] **Step 3: Implement in `src/counterstrat/web/maintenance.py`**

Add imports at top: `import os`, `import polars as pl`,
`from counterstrat.customzones import load_custom_zones, rezone_ticks`,
`from counterstrat.lake.extract import LakePaths`,
`from counterstrat.mapcard.compile import MapCard, compile_card`,
`from counterstrat.mapcard.lexicon import build_lexicon, get_default_overlay_path`,
`from counterstrat.mapcard.transitions import zone_graph`,
`from counterstrat.mapcard.vents import parse_places, unique_places`,
`from counterstrat.mapcard.zones import ZoneMapper`,
`from counterstrat.roundscript.serialize import serialize_match`.

```python
_LAKE_TABLES = [
    "rounds", "kills", "damages", "shots", "grenades", "smokes",
    "infernos", "bomb", "item_purchase", "ticks", "rosters", "player_blind",
]


def _lake_paths(data_root: Path, match_id: str) -> LakePaths:
    """LakePaths for an already-extracted match directory."""
    root = data_root / "lake" / match_id
    return LakePaths(
        root=str(root), **{name: str(root / f"{name}.parquet") for name in _LAKE_TABLES}
    )


def _effective_tick_places(ticks_df: pl.DataFrame) -> set[str]:
    if "last_place_name" not in ticks_df.columns:
        return set()
    return {
        str(p)
        for p in ticks_df["last_place_name"].drop_nulls().unique().to_list()
        if str(p).strip()
    }


def compile_map_card(
    map_name: str,
    places: list[str],
    ticks_df: pl.DataFrame,
    rounds_df: pl.DataFrame,
    patch_version: str,
) -> MapCard | None:
    """Compile a card whose zone set is the VPK places UNION the effective
    tick vocabulary - custom zones ride into the card (and so into prompts,
    lint, and the editor) automatically."""
    all_places = sorted(set(places) | _effective_tick_places(ticks_df))
    if not all_places:
        return None
    overlay = get_default_overlay_path(map_name)
    lex = build_lexicon(map_name, all_places, overlay if overlay.exists() else None)
    return compile_card(
        lexicon=lex,
        graph=zone_graph(ticks_df),
        ticks=ticks_df,
        rounds=rounds_df,
        map_name=map_name,
        patch_version=patch_version,
    )


def _cached_vpk_places(cfg: AppConfig, map_name: str) -> list[str]:
    """VPK place names from the tmp_assets vents cache; [] when not extracted.
    (No VRF run here - the cache exists for any map that compiled a card.)"""
    vents = (
        cfg.data_root / "tmp_assets" / map_name / "maps" / map_name
        / "entities" / "default_ents.vents"
    )
    if not vents.exists():
        return []
    try:
        return unique_places(parse_places(vents))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unreadable vents for %s: %s", map_name, exc)
        return []


def rebuild_map_zones(cfg: AppConfig, map_name: str) -> dict[str, Any]:
    """Bake the map's custom zones into the lake and rebuild all derivatives.

    Order matters: ticks first (the vocabulary), then the card (zones/topology),
    then scripts (movements/beats/utility via a mapper trained on re-zoned
    ticks), then miners, then stale LLM caches.
    """
    zones = load_custom_zones(cfg.data_root, map_name)
    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    matches = [mid for mid, rec in sorted(manifest.items()) if rec.map_name == map_name]

    # 1. Re-zone every match's ticks, atomically.
    for mid in matches:
        ticks_path = cfg.data_root / "lake" / mid / "ticks.parquet"
        if not ticks_path.exists():
            continue
        df = rezone_ticks(pl.read_parquet(ticks_path), zones)
        tmp = ticks_path.with_suffix(f".tmp_{os.getpid()}")
        df.write_parquet(tmp)
        os.replace(tmp, ticks_path)

    # 2. Recompile the card from the first match (mirrors ingest).
    card: MapCard | None = None
    if matches:
        first = matches[0]
        first_lake = _lake_paths(cfg.data_root, first)
        ticks_df = pl.read_parquet(first_lake.ticks)
        rounds_df = (
            pl.read_parquet(first_lake.rounds)
            if Path(first_lake.rounds).exists()
            else pl.DataFrame()
        )
        card = compile_map_card(
            map_name,
            _cached_vpk_places(cfg, map_name),
            ticks_df,
            rounds_df,
            manifest[first].patch_version,
        )
        if card is not None:
            card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
            card_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.write_text(card.to_yaml(), encoding="utf-8")

    # 3. Re-serialize scripts per match with a mapper trained on the new
    # vocabulary.
    overlay = get_default_overlay_path(map_name)
    for mid in matches:
        lake = _lake_paths(cfg.data_root, mid)
        if not Path(lake.ticks).exists():
            continue
        ticks_df = pl.read_parquet(lake.ticks)
        try:
            mapper = ZoneMapper.fit(ticks_df)
        except ValueError:
            continue  # no valid tick rows - keep old scripts
        if card is not None:
            places = list(card.zones.keys())
        else:
            places = sorted(_effective_tick_places(ticks_df)) or ["Default"]
        lex = build_lexicon(map_name, places, overlay if overlay.exists() else None)
        scripts = serialize_match(lake, mapper, lex, card.checksum if card else "none")
        scripts_dir = cfg.data_root / "scripts" / mid
        if scripts:
            shutil.rmtree(scripts_dir, ignore_errors=True)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            for s in scripts:
                (scripts_dir / f"round_{s.round_num}.json").write_text(
                    s.to_json(), encoding="utf-8"
                )

    # 4. Re-mine every teambook (recluster + prune, existing machinery).
    stats = rebuild_artifacts(cfg)

    # 5. Stale LLM caches for THIS map speak the old vocabulary - delete.
    tb_root = cfg.data_root / "teambooks"
    if tb_root.exists():
        for team_dir in tb_root.iterdir():
            map_dir = team_dir / map_name
            for stale in ("dossier.md", "insights.json"):
                p = map_dir / stale
                if p.exists():
                    p.unlink()

    return {"matches": len(matches), "zones": len(zones), **stats}


def run_zone_rebuild(job_id: str, map_name: str, cfg: AppConfig) -> None:
    """Background-job wrapper mirroring run_ingest's state machine."""
    from counterstrat.web.ingest import JobState, load_job_state, save_job_state

    state = load_job_state(cfg.data_root, job_id) or JobState(job_id=job_id, stage="queued")
    state.map_name = map_name
    try:
        state.stage = "serializing"
        save_job_state(cfg.data_root, state)
        result = rebuild_map_zones(cfg, map_name)
        state.stage = "mining"
        save_job_state(cfg.data_root, state)
        state.stage = "done"
        state.detail = f"rebuilt {result['matches']} match(es), {result['zones']} custom zone(s)"
        save_job_state(cfg.data_root, state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Zone rebuild job %s failed", job_id)
        state.stage = "error"
        state.detail = str(exc)
        save_job_state(cfg.data_root, state)
```

- [ ] **Step 4: Wire ingest** (`src/counterstrat/web/ingest.py`)

4a. After `lake = extract_lake(...)` (line 156), bake existing zones into the
fresh match so new demos of an edited map join the same vocabulary:

```python
        from counterstrat.customzones import load_custom_zones, rezone_ticks

        custom = load_custom_zones(cfg.data_root, rec.map_name)
        if custom:
            zoned = rezone_ticks(pl.read_parquet(lake.ticks), custom)
            zoned.write_parquet(lake.ticks)
```

4b. Replace the inline card-compile block (lines 179-194, from
`places = unique_places(...)` through `card = compile_card(...)`) with:

```python
                    from counterstrat.web.maintenance import compile_map_card

                    places = unique_places(parse_places(assets.vents))
                    ticks_df = pl.read_parquet(lake.ticks)
                    rounds_df = pl.read_parquet(lake.rounds)
                    card = compile_map_card(
                        rec.map_name, places, ticks_df, rounds_df, rec.patch_version
                    )
```

(the lexicon/overlay/zone_graph lines move into `compile_map_card`; the
`build_lexicon` import stays - it is still used at lines 212/229.)

- [ ] **Step 5: Run to verify pass**

Run: `uv run python -m pytest tests/test_customzones_rebuild.py tests/test_ingest_accumulate.py tests/test_web_delete.py -q`
Expected: all PASS (ingest + delete regression-gate the refactor)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check .
git add src/counterstrat/web/maintenance.py src/counterstrat/web/ingest.py tests/test_customzones_rebuild.py
git commit -m "feat: zone rebuild bakes custom callouts into lake, card, scripts, and miners"
```

---
### Task 4: API - create/edit/delete zones by pointing, anchors honor the user's point

**Files:**
- Modify: `src/counterstrat/web/routes.py` (`_callout_zone_names`,
  `_zone_anchors`, `get_callouts`, new `PUT /maps/{map_name}/zones`,
  new `_infer_world_z`)
- Test: `tests/test_web_callouts.py` (append)

**Interfaces:**
- Consumes: Task 1 (`load_custom_zones`, `save_custom_zones`, `CustomZone`),
  Task 3 (`run_zone_rebuild`).
- Produces:
  - `PUT /api/maps/{map_name}/zones` with body
    `{"zones": [{"name": str, "u": float, "v": float, "level": "default"|"lower", "radius": float}]}`
    -> `{"map_name": str, "zones": [CustomZone dumps], "job_id": str}`.
    Full-collection semantics: absent = deleted. 400 on: no radar
    calibration, no player data near a point, any validation error.
  - `GET /api/maps/{map_name}/callouts` zone entries gain
    `"custom": bool` and `"radius": float | None`; custom zones' `u`/`v` is
    the USER'S stored center (projected), never a tick medoid - the user's
    point is authoritative for custom zones, in the editor and in
    `format_zone_map` alike (both flow from `_zone_anchors`).
  - Aliases: custom names ride into `_callout_zone_names`, so
    `put_aliases` accepts them as targets and rejects alias values that
    collide with them (existing checks, no new code).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_web_callouts.py`)

```python
def _seed_empty_tables(cfg: AppConfig, match_id: str = "m1") -> None:
    # No rosters.parquet: build_team_clusters skips missing files but would
    # crash on a schemaless one. serialize_match needs rounds.round_num.
    lake = cfg.data_root / "lake" / match_id
    pl.DataFrame(
        {"match_id": pl.Series([], dtype=pl.String), "round_num": pl.Series([], dtype=pl.Int64)}
    ).write_parquet(lake / "rounds.parquet")
    pl.DataFrame({"match_id": pl.Series([], dtype=pl.String)}).write_parquet(
        lake / "kills.parquet"
    )


def test_put_zones_places_zone_rebuilds_and_anchors_at_user_point(
    cfg: AppConfig, client: TestClient
):
    """Pointing at the radar creates a zone; the rebuild bakes it into the
    lake; its label anchors exactly where the user clicked."""
    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [190.0, 200.0, 210.0, 900.0],
                "Y": [-310.0, -300.0, -290.0, 900.0],
                "Z": [0.0] * 4,
                "last_place_name": ["Middle", "Middle", "Middle", "BombsiteA"],
                "is_alive": [True] * 4,
            }
        ),
    )
    _seed_empty_tables(cfg)

    # Click at world (200, -300): u=(200+1024)/2048, v=(1024+300)/2048.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={
            "zones": [
                {"name": "Sandbags", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}
            ]
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["zones"][0]["name"] == "Sandbags"
    # TestClient runs the background rebuild before returning control.
    job = client.get(f"/api/jobs/{body['job_id']}").json()
    assert job["stage"] == "done", job

    ticks = pl.read_parquet(cfg.data_root / "lake" / "m1" / "ticks.parquet")
    assert ticks["last_place_name"].to_list() == [
        "Sandbags", "Sandbags", "Sandbags", "BombsiteA",
    ]
    assert ticks["place_default"].to_list()[:3] == ["Middle"] * 3

    zones = {z["name"]: z for z in client.get(f"/api/maps/{MAP}/callouts").json()["zones"]}
    sb = zones["Sandbags"]
    assert sb["custom"] is True and sb["radius"] == 100.0
    # Anchor = the user's click, not a tick medoid.
    assert abs(sb["u"] - 612 / 1024) < 1e-3 and abs(sb["v"] - 662 / 1024) < 1e-3
    assert zones["Middle"]["custom"] is False

    # The LLM zone map speaks it at the same point.
    from counterstrat.llm.prompts import format_zone_map
    from counterstrat.web.routes import map_zone_anchors

    zm = format_zone_map(map_zone_anchors(cfg, MAP))
    assert "`Sandbags` at (0.60, 0.65)" in zm

    # Deleting via empty collection folds the ticks back.
    r = client.put(f"/api/maps/{MAP}/zones", json={"zones": []})
    assert r.status_code == 200
    ticks = pl.read_parquet(cfg.data_root / "lake" / "m1" / "ticks.parquet")
    assert ticks["last_place_name"].to_list()[:3] == ["Middle"] * 3


def test_put_zones_validation(cfg: AppConfig, client: TestClient):
    # No radar calibration -> cannot place.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "A", "u": 0.5, "v": 0.5, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "calibration" in r.json()["detail"]

    _seed_calibration(cfg, MAP)
    _seed_match(
        cfg,
        MAP,
        pl.DataFrame(
            {
                "X": [200.0],
                "Y": [-300.0],
                "Z": [0.0],
                "last_place_name": ["Middle"],
                "is_alive": [True],
            }
        ),
    )
    _seed_empty_tables(cfg)
    # Far from any player data -> rejected (cannot ground the zone).
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "A", "u": 0.01, "v": 0.01, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "player data" in r.json()["detail"]
    # Name collision with a game zone.
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Middle", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    # Name collision with an alias value.
    client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Water": "Pond"}})
    r = client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Pond", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    assert r.status_code == 400 and "reserved" in r.json()["detail"]
    # And the reverse: an alias may not take a custom zone's name.
    client.put(
        f"/api/maps/{MAP}/zones",
        json={"zones": [{"name": "Sandbags", "u": 612 / 1024, "v": 662 / 1024, "radius": 100.0}]},
    )
    r = client.put(f"/api/maps/{MAP}/aliases", json={"aliases": {"Water": "Sandbags"}})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_web_callouts.py -q`
Expected: new tests FAIL with 404/405 on the zones route

- [ ] **Step 3: Implement in `src/counterstrat/web/routes.py`**

3a. Imports: extend the aliases import line with
`from counterstrat.customzones import CustomZone, load_custom_zones, save_custom_zones`
and add `from counterstrat.web.maintenance import run_zone_rebuild` INSIDE
the endpoint (lazy, mirrors other maintenance imports; maintenance imports
ingest lazily so no cycle).

3b. `_callout_zone_names` unions custom names (custom zones exist even
before the card recompile lands):

```python
def _callout_zone_names(cfg: AppConfig, map_name: str, places: list) -> list[str] | None:
    """Card zones when compiled (the mined vocabulary), else VPK place names;
    the user's custom zones always join the list."""
    custom = [z.name for z in load_custom_zones(cfg.data_root, map_name)]
    card_path = cfg.data_root / "mapcards" / map_name / "card.yaml"
    if card_path.exists():
        card_data = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        return sorted(set((card_data.get("zones") or {}).keys()) | set(custom))
    if places or custom:
        return sorted({p.place_name for p in places} | set(custom))
    return None
```

3c. `_zone_anchors` - after the volume-origin fill loop, override custom
zones with the user's stored center (the click point is authoritative):

```python
    for cz in load_custom_zones(cfg.data_root, map_name):
        u, v = game_to_norm(cal, cz.x, cz.y)
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            anchors[cz.name] = (round(u, 4), round(v, 4), cz.level)
```

3d. `get_callouts` - add per-zone custom metadata:

```python
    custom = {z.name: z for z in load_custom_zones(cfg.data_root, map_name)}
    ...
        "zones": [
            {
                "name": z,
                "alias": aliases.get(z),
                "u": anchors.get(z, (None, None, None))[0],
                "v": anchors.get(z, (None, None, None))[1],
                "level": anchors.get(z, (None, None, None))[2],
                "custom": z in custom,
                "radius": custom[z].radius if z in custom else None,
            }
            for z in zones
        ],
```

3e. Z inference + the endpoint:

```python
def _infer_world_z(cfg: AppConfig, map_name: str, x: float, y: float, level: str, cal) -> float | None:
    """Median Z of the nearest same-level ticks; None when nobody ever
    played near the point (a zone the data cannot ground is rejected)."""
    from counterstrat.radar.coords import level_expr

    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    for match_id, rec in sorted(manifest.items()):
        if rec.map_name != map_name:
            continue
        ticks_path = cfg.data_root / "lake" / match_id / "ticks.parquet"
        if not ticks_path.exists():
            continue
        try:
            df = (
                pl.read_parquet(ticks_path, columns=["X", "Y", "Z", "is_alive"])
                .filter(pl.col("is_alive"))
                .drop_nulls(["X", "Y", "Z"])
            )
            if df.is_empty():
                continue
            df = df.with_columns(level_expr(cal, "Z").alias("_lvl")).filter(
                pl.col("_lvl") == level
            )
            near = (
                df.with_columns(
                    ((pl.col("X") - x) ** 2 + (pl.col("Y") - y) ** 2).alias("_d2")
                )
                .sort("_d2")
                .head(50)
            )
            if near.is_empty() or float(near["_d2"].min()) ** 0.5 > 300.0:
                return None
            return float(near["Z"].median())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Z inference unavailable from %s: %s", ticks_path, exc)
    return None


class ZonePlacement(BaseModel):
    name: str
    u: float
    v: float
    level: str = "default"
    radius: float = 150.0


class ZoneUpdateRequest(BaseModel):
    zones: list[ZonePlacement]


@router.put("/maps/{map_name}/zones")
def put_zones(
    map_name: str, req: ZoneUpdateRequest, cfg: ConfigDep, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Persist the user's custom zones and rebuild the map's derived data."""
    from counterstrat.radar.coords import pixel_to_game
    from counterstrat.web.maintenance import run_zone_rebuild

    cal = _radar_calibration(cfg, map_name)
    if cal is None:
        raise HTTPException(status_code=400, detail="No radar calibration for this map")
    places = _map_places(cfg, map_name)
    base = {
        z
        for z in (_callout_zone_names(cfg, map_name, places) or [])
        if z not in {c.name for c in load_custom_zones(cfg.data_root, map_name)}
    }
    reserved = base | set(load_aliases(cfg.data_root, map_name).values())

    zones: list[CustomZone] = []
    for p in req.zones:
        x, y = pixel_to_game(cal, p.u * cal.image_px, p.v * cal.image_px)
        z = _infer_world_z(cfg, map_name, x, y, p.level, cal)
        if z is None:
            raise HTTPException(
                status_code=400,
                detail=f"No player data near {p.name!r} - place it where players have been",
            )
        zones.append(
            CustomZone(name=p.name, x=round(x, 1), y=round(y, 1), z=round(z, 1),
                       level=p.level, radius=p.radius)
        )
    try:
        saved = save_custom_zones(cfg.data_root, map_name, zones, reserved=reserved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex[:12]
    save_job_state(cfg.data_root, JobState(job_id=job_id, stage="queued", map_name=map_name))
    background_tasks.add_task(run_zone_rebuild, job_id, map_name, cfg)
    return {
        "map_name": map_name,
        "zones": [z.model_dump() for z in saved],
        "job_id": job_id,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/test_web_callouts.py tests/test_web_api.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
git add src/counterstrat/web/routes.py tests/test_web_callouts.py
git commit -m "feat: place custom callout zones by pointing - API, grounded Z, user-point anchors"
```

---

### Task 5: editor UI - add-callout mode, custom labels, radius, delete, job polling

**Files:**
- Modify: `src/counterstrat/web/static/index.html` (add button; bump
  `callouts.js` and `style.css` to `?v=5`)
- Modify: `src/counterstrat/web/static/callouts.js`
- Modify: `src/counterstrat/web/static/style.css`
- Test: `tests/test_web_static.py::test_callouts_view_static` (extend)

**Interfaces:**
- Consumes: Task 4 endpoints (`PUT /api/maps/{map}/zones`,
  `GET /api/jobs/{job_id}`, callouts `custom`/`radius` fields).
- Produces: UI only.

- [ ] **Step 1: Extend the failing static pins**

```python
    # in test_callouts_view_static, extend the html token list with:
        'id="callouts-add"',
    # and after the existing js asserts add:
    assert "is-placing" in js  # add-callout placement mode
    assert "/zones" in js and "pollJob" in js
    assert "is-user-zone" in js
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run python -m pytest tests/test_web_static.py::test_callouts_view_static -q`
Expected: FAIL missing `id="callouts-add"`

- [ ] **Step 3: Implement**

3a. `index.html`: next to the Reset all button add
`<button type="button" id="callouts-add" class="btn btn-sm btn-outline">+ Add callout</button>`;
bump both asset versions to `?v=5`.

3b. `callouts.js` - state gains `placing: false` and `job: null`; cache
`el.addBtn = document.getElementById("callouts-add")` and
`el.frame = document.querySelector(".callouts-frame")`.

```javascript
  // ------------------------------------------------------------- custom zones
  function customZones() {
    return state.zones.filter(function (z) { return z.custom; });
  }

  function zonesPayload() {
    return customZones().map(function (z) {
      return { name: z.name, u: z.u, v: z.v, level: z.level || "default", radius: z.radius };
    });
  }

  function putZones(payload, verb) {
    setStatus(`${verb}...`, false);
    fetch(`/api/maps/${encodeURIComponent(state.mapName)}/zones`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zones: payload }),
    })
      .then(function (res) {
        if (!res.ok) {
          return res.json().then(function (body) {
            throw new Error((body && body.detail) || `Server returned ${res.status}`);
          });
        }
        return res.json();
      })
      .then(function (data) { pollJob(data.job_id); })
      .catch(function (err) { setStatus(`Not saved: ${err.message}`, true); });
  }

  function pollJob(jobId) {
    setStatus("Rebuilding map data - tendencies, scripts, and reads pick up your callout...", false);
    const timer = window.setInterval(function () {
      fetch(`/api/jobs/${encodeURIComponent(jobId)}`)
        .then(function (res) { return res.json(); })
        .then(function (job) {
          if (job.stage === "done") {
            window.clearInterval(timer);
            setStatus("Callouts rebuilt - the analyst speaks them from the next session or regenerated read.", false);
            loadCallouts();
          } else if (job.stage === "error") {
            window.clearInterval(timer);
            setStatus(`Rebuild failed: ${job.detail}`, true);
          }
        })
        .catch(function () { /* transient poll errors: keep polling */ });
    }, 1500);
  }

  function setPlacing(on) {
    state.placing = on;
    el.addBtn.classList.toggle("active", on);
    el.frame.classList.toggle("is-placing", on);
    if (on) setStatus("Click the radar where the new callout belongs. Esc cancels.", false);
  }

  function placeAt(ev) {
    if (!state.placing) return;
    const rect = el.labels.getBoundingClientRect();
    const u = (ev.clientX - rect.left) / rect.width;
    const v = (ev.clientY - rect.top) / rect.height;
    setPlacing(false);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "callout-edit-input";
    input.placeholder = "callout name";
    input.style.position = "absolute";
    input.style.left = `${u * 100}%`;
    input.style.top = `${v * 100}%`;
    el.labels.appendChild(input);
    input.focus();
    let done = false;
    function commit(save) {
      if (done) return;
      done = true;
      const name = (input.value || "").trim();
      input.remove();
      if (!save || !name) { render(); return; }
      const payload = zonesPayload();
      payload.push({ name: name, u: u, v: v, level: state.level, radius: 150 });
      putZones(payload, `Placing ${name}`);
    }
    input.addEventListener("keydown", function (kev) {
      if (kev.key === "Enter") commit(true);
      if (kev.key === "Escape") commit(false);
    });
    input.addEventListener("blur", function () { commit(true); });
  }

  function removeZone(name) {
    putZones(zonesPayload().filter(function (z) { return z.name !== name; }),
             `Removing ${name}`);
  }

  function setRadius(name, value) {
    const payload = zonesPayload();
    const target = payload.find(function (z) { return z.name === name; });
    if (!target) return;
    target.radius = Number(value) || target.radius;
    putZones(payload, `Resizing ${name}`);
  }
```

Wire in `render()`: give custom zones `btn.classList.add("is-user-zone")`;
in the table rows for custom zones replace the alias input cell with a
radius `<input type="number" min="64" max="600">` (change -> `setRadius`)
plus a delete button (click -> `removeZone`). In `DOMContentLoaded`:
`el.addBtn.addEventListener("click", function () { setPlacing(!state.placing); });`
`el.labels.addEventListener("click", placeAt, true);` (capture phase so
placement wins over label clicks) and a document `keydown` that exits
placement on Escape.

3c. `style.css`:

```css
.callouts-frame.is-placing { cursor: crosshair; }
.callout-label.is-user-zone { border-color: var(--color-accent, #6cf); color: var(--color-accent, #6cf); }
#callouts-add.active { background: var(--bg-tertiary); }
```

(use the accent variable the stylesheet already defines; check `:root` and
reuse the existing custom-label accent if one exists.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run python -m pytest tests/test_web_static.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check .
git add src/counterstrat/web/static/index.html src/counterstrat/web/static/callouts.js src/counterstrat/web/static/style.css tests/test_web_static.py
git commit -m "feat: point-and-name callout placement in the editor with rebuild polling"
```

---
### Task 6: real-demo end-to-end - the whole chain speaks the callout

**Files:**
- Test: `tests/test_customzones_e2e.py` (new; uses the session-scoped
  `anubis_lake` real-demo fixture, so it runs by default and skips when the
  demo file is absent)

**Interfaces:** consumes Tasks 1+3 only (no HTTP): proves
lake -> scripts -> teambook all adopt the custom vocabulary on REAL data,
and measures the rebuild cost.

- [ ] **Step 1: Write the test**

```python
"""Real-demo proof: a pointed callout propagates into scripts and teambooks."""

import json
import shutil
import time
from pathlib import Path

import polars as pl
import pytest

from counterstrat.config import AppConfig
from counterstrat.customzones import CustomZone, save_custom_zones
from counterstrat.web.maintenance import rebuild_map_zones

MAP = "de_anubis"


@pytest.fixture(autouse=True)
def isolated_repo_root(tmp_path: Path, monkeypatch):
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "maps").mkdir(parents=True)
    monkeypatch.setattr("counterstrat.web.ingest.REPO_ROOT", fake_repo)
    monkeypatch.chdir(tmp_path)
    return fake_repo


def test_custom_zone_reaches_scripts_and_teambook(tmp_path: Path, anubis_lake):
    cfg = AppConfig(data_root=tmp_path / "data")
    match_dir = Path(anubis_lake.root)
    match_id = match_dir.name
    dest = cfg.data_root / "lake" / match_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(match_dir, dest)
    (cfg.data_root / "corpus.jsonl").write_text(
        json.dumps(
            {
                "match_id": match_id,
                "path": "demos/real.dem",
                "map_name": MAP,
                "patch_version": "1",
                "demo_version_guid": "g",
                "server_name": "s",
                "registered_at": "2026-09-03T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Ground the zone at the densest real spot of Middle.
    ticks = pl.read_parquet(dest / "ticks.parquet")
    mid = ticks.filter(pl.col("is_alive") & (pl.col("last_place_name") == "Middle"))
    cx, cy, cz = (float(mid["X"].median()), float(mid["Y"].median()), float(mid["Z"].median()))
    save_custom_zones(
        cfg.data_root,
        MAP,
        [CustomZone(name="Sandbags", x=cx, y=cy, z=cz, radius=200.0)],
        reserved=set(),
    )

    t0 = time.perf_counter()
    result = rebuild_map_zones(cfg, MAP)
    elapsed = time.perf_counter() - t0
    print(f"rebuild_map_zones({MAP}): {elapsed:.1f}s for {result['matches']} match(es)")
    assert elapsed < 120.0, "rebuild must stay interactive-job fast for one match"

    # 1. Lake speaks it, reversibly.
    zoned = pl.read_parquet(dest / "ticks.parquet")
    assert "Sandbags" in zoned["last_place_name"].unique().to_list()
    assert "place_default" in zoned.columns

    # 2. Card speaks it (compiled from the tick vocabulary; no VPK here).
    card_text = (cfg.data_root / "mapcards" / MAP / "card.yaml").read_text(encoding="utf-8")
    assert "Sandbags" in card_text

    # 3. Scripts speak it (movements/beats cite the zone players stood in).
    scripts_text = "".join(
        p.read_text(encoding="utf-8")
        for p in sorted((cfg.data_root / "scripts" / match_id).glob("round_*.json"))
    )
    assert "Sandbags" in scripts_text

    # 4. Teambooks (the miners' vocabulary) speak it.
    teambooks = "".join(
        p.read_text(encoding="utf-8")
        for p in (cfg.data_root / "teambooks").glob("*/{}/teambook.json".format(MAP))
    )
    assert "Sandbags" in teambooks
```

- [ ] **Step 2: Run it**

Run: `uv run python -m pytest tests/test_customzones_e2e.py -q -s`
Expected: PASS, with the rebuild timing printed. If `Sandbags` misses any
of the four layers, the propagation chain is broken - fix the layer, not
the test.

- [ ] **Step 3: Commit**

```bash
uv run ruff check .
git add tests/test_customzones_e2e.py
git commit -m "test: real-demo e2e - pointed callout reaches lake, card, scripts, teambook"
```

---

### Task 7: full verification, handoff note

**Files:**
- Modify: `docs/plans/2026-09-03-session-handoff.md` (append a short update)

- [ ] **Step 1: Full suite + lint**

Run: `uv run python -m pytest -q` -> zero failures.
Run: `uv run ruff check .` -> zero errors.

- [ ] **Step 2: Handoff update**

Append to the update blockquote in
`docs/plans/2026-09-03-session-handoff.md`: custom callout zones landed
(point-and-name in the editor; spheres in `data/mapcards/<map>/zones.json`;
`rezone_ticks` bakes them into `last_place_name` with `place_default`
preserving game names; `rebuild_map_zones` rebuilds card/scripts/miners and
prunes per-map `dossier.md`/`insights.json`; duckdb views union_by_name).
Include the measured e2e rebuild timing.

- [ ] **Step 3: Commit and remind the user to RESTART the server**

```bash
git add docs/plans/2026-09-03-session-handoff.md
git commit -m "docs: handoff - custom callout zones landed"
```

---

## Risks

1. **Sample dilution** - a custom zone splits its parent's `n`; concentrated
   reads can fall under the signal thresholds and honestly disappear. This
   is correct behaviour, but surprising: the status line copy in Task 5
   says reads "pick up" the callout, not that they improve.
2. **Rebuild latency** - one background job per save; per match it is one
   parquet rewrite + serialize + mine. Task 6 measures it on a real demo
   and gates at <120s/match. If multi-demo maps get slow, the next lever is
   re-serializing only matches whose ticks actually changed rows (compare
   `last_place_name` hash before/after re-zone) - deferred, YAGNI now.
3. **Z inference at click** - median of the 50 nearest same-level ticks
   within 300u. A click on a balcony railing may ground to the floor below
   on the same level; the fix is the radius/level the user controls, and
   the reject-when-ungroundable rule keeps garbage out.
4. **Level classification of custom zones** - the zone's `level` is the
   user's toggle at placement, authoritative over tick majorities
   (`_zone_anchors` override). Ticks inside the sphere on the OTHER level
   are still captured by the scaled-3D predicate only if within radius -
   acceptable because Z_SCALE=2 makes vertical capture expensive.
5. **Schema drift** - `place_default` exists only on rebuilt maps' ticks;
   `union_by_name=true` (Task 2) keeps the duckdb views whole. Polars
   readers select columns explicitly and are unaffected.
6. **Stale in-memory chat sessions** after a rebuild keep the old system
   prompt until restored - same accepted staleness as demo deletion.
7. **User data root** - implementers must never run rebuilds against
   `data/` (the user's real corpus) during development; every test seeds
   its own `tmp_path` root and `isolated_repo_root`.

PLAN COMPLETE
