# Radar Visualizer (spec item N2) Implementation Plan

> **For agentic workers:** Use `executing-plans` or `subagent-driven-development`
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an interactive 2D radar overlay to the web UI that draws a team's
positional data (heatmaps, player trails, utility landings, kill duels, bomb
plants) on the real CS2 overhead map image, extracted from the local CS2 install.

**Architecture:** A new `counterstrat.radar` package with three focused modules —
`extract.py` (VRF asset extraction + overview KeyValues calibration parsing),
`coords.py` (pure world→radar projection math, scalar and polars-expression
forms), `layers.py` (polars LazyFrame queries turning lake tables into
normalized `[0,1]` coordinate layers). A new FastAPI router
(`web/radar_api.py`) serves the extracted PNG as-is plus JSON calibration and
layer payloads. The frontend draws overlays on an HTML `<canvas>` stacked over
an `<img>` of the radar — so **no server-side image rendering and no new Python
dependency**.

**Tech Stack:** Python 3.13, FastAPI, polars (lazy parquet scans), pydantic,
vendored Source2Viewer-CLI (VRF), vanilla JS + Canvas 2D.

**Spec:** `docs/NEEDS-FROM-YOU.md:42-58` (item N2 — "Radar images (optional;
evidence plots only)"), with background at
`docs/plans/cs2-demo-llm-anti-strat.md:217` and
`docs/plans/2026-09-03-cs2-anti-strat-implementation.md:183-185`.

---

## Context

- Repo: `counterstrat`, a CS2 demo → lake → LLM anti-strat analyst. Hatchling
  package under `src/counterstrat/`, `requires-python = ">=3.13"`
  (`pyproject.toml:4`). Ruff `line-length = 100` (`pyproject.toml:40`).
- **Test command (verified, 165 selected / 6 deselected, ~1.7 s collect):**
  `.venv\Scripts\python.exe -m pytest -q`
  `addopts = "-m 'not live'"` (`pyproject.toml:37`), so `demo`-marked tests
  **do** run by default; they self-skip via fixtures when assets are missing.
- **Lint command (verified, "All checks passed!"):**
  `.venv\Scripts\python.exe -m ruff check .`
- Single test: `.venv\Scripts\python.exe -m pytest tests/test_radar_coords.py -v`
- CS2 install is configured at
  `D:\Data\Gaming\Steam\steamapps\common\Counter-Strike Global Offensive`
  (`data/settings.json:5`), surfaced as `AppConfig.cs2_install_path`
  (`src/counterstrat/config.py:32`, `config.py:55`).
- VRF CLI vendored at `tools/vrf/Source2Viewer-CLI.exe`; the existing resolver
  `_find_vrf_cli()` lives at `src/counterstrat/web/ingest.py:74-87` — **reuse
  it, do not write a second one**.
- `data/*` is gitignored except `data/mapcards/` (`.gitignore:2-3`), so the new
  `data/radar/` cache needs no gitignore change.

## Verified facts (established this session — do not re-derive)

1. **VRF extraction works and is fast.** Two invocations against
   `<cs2>\game\csgo\pak01_dir.vpk`, exit code 0, ~0.6 s each:
   - `-f "panorama/images/overheadmaps/<map>_radar_psd.vtex_c" -d` writes
     `<out>\panorama\images\overheadmaps\<map>_radar_psd.png`.
   - `-f "resource/overviews/<map>.txt"` (**no** `-d`) writes
     `<out>\resource\overviews\<map>.txt`.
   - Comma-separated `-f` filters work, and **entries that do not exist in the
     VPK are silently skipped with exit code 0** (verified with
     `de_ancient_lower_radar_psd.vtex_c` and `de_bogusmap.txt`). So the same
     command can always ask for the `_lower_` variant.
2. **Radar PNGs are 1024x1024 RGBA** (verified for `de_anubis`, `de_ancient`).
3. **Overview KeyValues formats seen (all three verified verbatim):**
   - `de_anubis.txt`: `pos_x "-2796.000000"`, `pos_y "3328.000000"`,
     `scale "5.220000"`, no `verticalsections`.
   - `de_ancient.txt`: `pos_x "-2953"`, `pos_y "2164"`, `scale "5"`, plus
     `rotate "0"` and `zoom "0"`, no `verticalsections`.
   - `de_nuke.txt` / `de_vertigo.txt`: add a nested
     `verticalsections { "default" { AltitudeMax/AltitudeMin } "lower" { ... } }`
     block; nuke's `"lower"` has `AltitudeMax "-495"`, vertigo's `"11700"`.
   - Values are quoted, tab-separated, and lines carry trailing `// comments`
     (e.g. `"pos_x"  "-3453" // upper left world coordinate`).
4. **Projection validated against real data.** With
   `u = (X - pos_x) / (scale * 1024)`, `v = (pos_y - Y) / (scale * 1024)` and
   de_anubis calibration, all 487 k alive tick positions land inside
   `u ∈ [0.158, 0.861]`, `v ∈ [0.073, 0.945]`; A-site plants cluster at
   `(u≈0.764, v≈0.272)` and B-site plants at `(u≈0.323, v≈0.510)` — matching
   the Anubis radar. Exact anchor: bomb plant `X=1283.711304, Y=1873.848755`
   → `u=0.7632362, v=0.2720440`.
5. **awpy's own projection helpers are unusable here.**
   `awpy.plot.utils.game_to_pixel_axis` reads `awpy.data.map_data.MAP_DATA`,
   which is **empty** in this venv (`WARNING - Failed to load map data from
   C:\Users\Uzair\.awpy\maps\map-data.json`) and is populated only by a network
   download that the environment blocks. Hence our own `coords.py`.
6. **Lake column names and dtypes (read from real parquet, not guessed):**
   - `ticks`: `X/Y/Z` `Float32`, `steamid` `UInt64`, `round_num` `UInt32`,
     `clock_s` `Float64`, `is_alive` `Boolean`, `name` `String`,
     **`team_name` `String` (values `"CT"` / `"TERRORIST"` / null) — there is
     NO `side` column on `ticks`**, `match_id` `String`. Sampled every 4 ticks
     = **16 Hz**, ~487 k rows per match, so trails MUST be decimated.
   - `rosters`: `round_num` `Int64`, `side` `String` (`"CT"`/`"TERRORIST"`),
     `team_key` `String`, `steamids` `List(Int64)`, `clan_name`, `match_id`.
   - `kills`: `attacker_X/Y/Z`, `victim_X/Y/Z` `Float32`,
     `attacker_side`/`victim_side` `String` with **lowercase `"t"` / `"ct"` and
     nulls**, `attacker_steamid`/`victim_steamid` `UInt64`, `attacker_name`,
     `victim_name`, `weapon`, `headshot`, `tick` `Int32`, `round_num` `UInt32`.
   - `bomb`: `event` `String` with values `pickup, plant, drop, detonate,
     defuse`; `X/Y/Z`, `bombsite`, `steamid` `UInt64`, `name`, `round_num`.
   - `grenades`: `thrower_steamid` `UInt64`, `thrower` `String`,
     `grenade_type` `String`, `entity_id` `Int32`, `X/Y/Z`, `tick` `Int32`,
     `round_num` `UInt32`. **~2.5 M rows per match.**
   - `smokes` / `infernos`: `entity_id` `Int64`, `thrower_steamid` `Int64`,
     `thrower_side` lowercase `"t"`/`"ct"`, landing `X/Y/Z` + `thrower_X/Y/Z`,
     `start_tick`/`end_tick`.
7. **`grenade_type` has two families and only one is the thrown projectile.**
   Grouping `grenades` by `(round_num, entity_id)` over one match:
   `['CHEGrenadeProjectile','CFlashbangProjectile']` → **301 groups, 246 rows
   each** (≈10 thrown nades/round: the in-flight trajectory).
   `['CHEGrenade','CFlashbang']` → 439 groups, **2610 rows each** (the
   inventory-held entity, tracked all round). **Use only the `C*Projectile`
   values.** Full projectile set present in the data:
   `CSmokeGrenadeProjectile, CHEGrenadeProjectile, CFlashbangProjectile,
   CMolotovProjectile, CDecoyProjectile`. (Incendiary grenades also spawn a
   `CMolotovProjectile`, so both map to kind `molotov`.)
8. **steamid dtypes are inconsistent across tables** (`UInt64` on
   `ticks`/`kills`/`bomb`/`grenades`, `Int64` inside `rosters.steamids`), so
   every join must cast to a single type. All CS2 steamids fit in `Int64`.
9. `kopipasta` on this machine is the pre-0.70 build (no `map` verb), so the
   codebase-map skill fell back to narrow glob/grep. Not a blocker.

## Assumptions (chosen where the brief was ambiguous)

1. **No server-side plotting.** Pillow and matplotlib are importable (pulled in
   transitively by awpy) but are **not declared in `pyproject.toml`**. The plan
   therefore serves the VRF-produced PNG unmodified and draws all overlays in
   the browser. No dependency is added.
2. **Coordinates cross the wire normalized to `[0, 1]`**, rounded to 4 decimals,
   with `image_px` reported separately. The canvas is a fixed 1024x1024 backing
   store, so `px = norm * 1024` in the frontend.
3. **One layers endpoint, all layers in one payload**
   (`GET /api/radar/{team_key}/{map_name}/layers`). Toggling a layer is a pure
   client-side redraw, not a refetch. Cheaper than five endpoints and keeps the
   UI instant.
4. **The utility layer is derived from `grenades` projectile rows only** (throw
   origin = first position, landing = last position, per
   `(match_id, round_num, entity_id)`). The `smokes`/`infernos` tables are not
   used in this feature; they remain available for a future "effect radius"
   layer (see Risks).
5. **Utility shows only the analysed team's own throws.** Duels show every kill
   in the team's rounds, tagged `by_team`, because both directions are
   analytically useful.
6. **Radar assets are extracted lazily on first request** and cached under
   `<data_root>/radar/<map_name>/`. Ingest is not modified — the radar is a
   view-time concern and ingest must stay offline-capable.
7. **`rotate`/`zoom` overview keys are parsed and ignored** with a logged
   warning if non-zero. Every shipped competitive map has `rotate 0` or omits
   it; implementing rotation now would be speculative.
8. Multi-level maps expose a `level` filter (`default` / `lower` / `all`);
   membership is decided by `Z <= lower_altitude_max` from the `"lower"`
   `verticalsections` block. Single-level maps report
   `lower_altitude_max = null` and only `level=all`/`default` behave identically.
9. Frontend gets a **new** `static/radar.js` rather than growing the 716-line
   `app.js`; `app.js` receives only a 3-line hook.

## File structure

| File | Responsibility |
|---|---|
| `src/counterstrat/radar/__init__.py` (new) | Re-exports, mirroring `lake/__init__.py:1-6` |
| `src/counterstrat/radar/extract.py` (new) | VRF extraction, cache layout, overview KeyValues → `RadarCalibration` |
| `src/counterstrat/radar/coords.py` (new) | Pure projection math: scalar, polars-expression, level test, side normalization |
| `src/counterstrat/radar/layers.py` (new) | Lake LazyFrame → heatmap / trails / utility / duels / bombs payloads |
| `src/counterstrat/web/radar_api.py` (new) | `APIRouter(prefix="/api/radar")`: `/{map}/info`, `/{map}/image`, `/{team}/{map}/layers` |
| `src/counterstrat/web/app.py` (modify `:12`, `:26`) | Include the radar router |
| `src/counterstrat/web/static/radar.js` (new) | Canvas renderer, layer toggles, tab switching |
| `src/counterstrat/web/static/index.html` (modify `:66-113`, `:165`) | Radar tab + canvas + controls, `<script>` tag |
| `src/counterstrat/web/static/style.css` (modify, append) | Radar view styling using the existing token palette |
| `src/counterstrat/web/static/app.js` (modify `:298-321`) | Call the radar hook on target selection |
| `tests/conftest.py` (modify, append) | `build_radar_lake()` synthetic-lake helper + `TEST_CAL` |
| `tests/test_radar_extract.py` (new) | KeyValues parsing (offline) + real VRF extraction (`demo`-marked) |
| `tests/test_radar_coords.py` (new) | Projection math, round-trip, level test, side normalization |
| `tests/test_radar_layers.py` (new) | All five layer builders against the synthetic lake |
| `tests/test_radar_api.py` (new) | Endpoint contracts via `TestClient` |
| `tests/test_web_radar_static.py` (new) | Static asset + markup contract |
| `docs/NEEDS-FROM-YOU.md` (modify `:10`, `:42-58`) | Flip N2 from DEFERRED to implemented |

---

## Task 1: Radar asset extractor and calibration parser

**Goal:** Extract the CS2 overhead radar PNG(s) and the overview calibration file
out of `pak01_dir.vpk` into a per-map cache, and parse the overview KeyValues
into a typed `RadarCalibration`.

**Difficulty:** MEDIUM — **Route:** EASY
**Verify:** full

**Files:**
- Create: `src/counterstrat/radar/__init__.py`
- Create: `src/counterstrat/radar/extract.py`
- Create: `tests/test_radar_extract.py`
- Modify: `tests/conftest.py` (append fixtures at end of file)

**Interfaces — Produces (later tasks depend on these exact names):**
```python
RADAR_IMAGE_PX: int = 1024

class RadarCalibration(BaseModel):
    map_name: str
    pos_x: float
    pos_y: float
    scale: float
    image_px: int = RADAR_IMAGE_PX
    lower_altitude_max: float | None = None

class RadarAssets(BaseModel):
    map_name: str
    image: Path
    lower_image: Path | None
    overview: Path
    calibration: RadarCalibration
    @property
    def levels(self) -> list[str]: ...          # ["default"] or ["default", "lower"]

def radar_cache_dir(data_root: Path, map_name: str) -> Path: ...
def parse_calibration(text: str, map_name: str, image_px: int = RADAR_IMAGE_PX) -> RadarCalibration: ...
def load_cached_assets(data_root: Path, map_name: str) -> RadarAssets | None: ...
def extract_radar_assets(cs2_install: Path, vrf_cli: Path, map_name: str, data_root: Path,
                         *, timeout: float = 300.0) -> RadarAssets: ...
def get_radar_assets(data_root: Path, map_name: str, cs2_install: Path | None,
                     vrf_cli: Path | None, *, force: bool = False) -> RadarAssets: ...
```

- [x] **Step 1: Write the failing tests**

Create `tests/test_radar_extract.py`:

```python
from pathlib import Path

import pytest

from counterstrat.radar.extract import (
    RADAR_IMAGE_PX,
    parse_calibration,
    radar_cache_dir,
)

# Verbatim from pak01: resource/overviews/de_anubis.txt
ANUBIS_OVERVIEW = """// TAVR - AUTO RADAR. v 2.5.0a
"de_anubis"
{
\t"CTSpawn_x" "0.610000"
\t"CTSpawn_y" "0.220000"
\t"TSpawn_x" "0.580000"
\t"TSpawn_y" "0.930000"
\t"material" "overviews/de_anubis"
\t"pos_x" "-2796.000000"
\t"pos_y" "3328.000000"
\t"scale" "5.220000"
}
"""

# Verbatim from pak01: resource/overviews/de_nuke.txt (trailing // comments kept)
NUKE_OVERVIEW = """// HLTV overview description file for de_nuke.bsp

"de_nuke"
{
\t"material"\t"overviews/de_nuke"\t// texture file
\t"pos_x"\t\t"-3453"\t// upper left world coordinate
\t"pos_y"\t\t"2887"
\t"scale"\t\t"7"\x20

\t"verticalsections"
\t{
\t\t"default" // use the primary radar image
\t\t{
\t\t\t"AltitudeMax" "10000"
\t\t\t"AltitudeMin" "-495"
\t\t}
\t\t"lower" // i.e. de_nuke_lower_radar.dds
\t\t{
\t\t\t"AltitudeMax" "-495"
\t\t\t"AltitudeMin" "-10000"
\t\t}
\t}

\t"CTSpawn_x"\t"0.82"
\t"inset_left"\t\t"0.33"
}
"""

# Verbatim from pak01: resource/overviews/de_ancient.txt (has rotate/zoom)
ANCIENT_OVERVIEW = """// HLTV overview description file for de_ancient.bsp

"de_ancient"
{
\t"material"\t"overviews/de_ancient"\t// texture file
\t"pos_x"\t\t"-2953"\t// upper left world coordinate
\t"pos_y"\t\t"2164"
\t"scale"\t\t"5"
\t"rotate"\t"0"
\t"zoom"\t\t"0"
\t"CTSpawn_x"\t"0.51"
}
"""


def test_parse_calibration_single_level() -> None:
    cal = parse_calibration(ANUBIS_OVERVIEW, "de_anubis")
    assert cal.map_name == "de_anubis"
    assert cal.pos_x == -2796.0
    assert cal.pos_y == 3328.0
    assert cal.scale == 5.22
    assert cal.image_px == RADAR_IMAGE_PX == 1024
    assert cal.lower_altitude_max is None


def test_parse_calibration_multi_level_reads_lower_block() -> None:
    cal = parse_calibration(NUKE_OVERVIEW, "de_nuke")
    assert (cal.pos_x, cal.pos_y, cal.scale) == (-3453.0, 2887.0, 7.0)
    # -495 comes from verticalsections."lower".AltitudeMax, NOT "default".
    assert cal.lower_altitude_max == -495.0


def test_parse_calibration_ignores_rotate_and_zoom() -> None:
    cal = parse_calibration(ANCIENT_OVERVIEW, "de_ancient")
    assert (cal.pos_x, cal.pos_y, cal.scale) == (-2953.0, 2164.0, 5.0)
    assert cal.lower_altitude_max is None


def test_parse_calibration_rejects_missing_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        parse_calibration('"de_x"\n{\n\t"pos_x" "1"\n\t"pos_y" "2"\n}\n', "de_x")


def test_radar_cache_dir_layout(tmp_path: Path) -> None:
    assert radar_cache_dir(tmp_path, "de_anubis") == tmp_path / "radar" / "de_anubis"


def test_load_cached_assets_missing_returns_none(tmp_path: Path) -> None:
    from counterstrat.radar.extract import load_cached_assets

    assert load_cached_assets(tmp_path, "de_anubis") is None


@pytest.mark.demo
def test_extract_radar_assets_from_real_vpk(
    tmp_path: Path, cs2_install: Path, vrf_cli: Path
) -> None:
    from counterstrat.radar.extract import extract_radar_assets, load_cached_assets

    assets = extract_radar_assets(cs2_install, vrf_cli, "de_anubis", tmp_path)
    assert assets.image.exists() and assets.image.stat().st_size > 10_000
    assert assets.image.name == "radar.png"
    assert assets.overview.exists()
    assert assets.lower_image is None          # anubis is single-level
    assert assets.levels == ["default"]
    assert assets.calibration.pos_x == -2796.0
    assert assets.calibration.scale == 5.22
    # calibration.json is written so later requests skip re-parsing
    assert (tmp_path / "radar" / "de_anubis" / "calibration.json").exists()
    # cache hit path returns an equivalent bundle
    cached = load_cached_assets(tmp_path, "de_anubis")
    assert cached is not None
    assert cached.calibration == assets.calibration


@pytest.mark.demo
def test_extract_radar_assets_multi_level(
    tmp_path: Path, cs2_install: Path, vrf_cli: Path
) -> None:
    from counterstrat.radar.extract import extract_radar_assets

    assets = extract_radar_assets(cs2_install, vrf_cli, "de_nuke", tmp_path)
    assert assets.lower_image is not None and assets.lower_image.exists()
    assert assets.levels == ["default", "lower"]
    assert assets.calibration.lower_altitude_max == -495.0


@pytest.mark.demo
def test_extract_radar_assets_unknown_map_raises(
    tmp_path: Path, cs2_install: Path, vrf_cli: Path
) -> None:
    from counterstrat.radar.extract import extract_radar_assets

    with pytest.raises(FileNotFoundError, match="de_bogusmap"):
        extract_radar_assets(cs2_install, vrf_cli, "de_bogusmap", tmp_path)
```

Append to `tests/conftest.py` (the `vrf_cli` fixture already exists at
`tests/conftest.py:34-38` — do **not** duplicate it):

```python
@pytest.fixture(scope="session")
def cs2_install() -> Path:
    """The configured CS2 install root, or skip when N2 is unsatisfied."""
    from counterstrat.config import AppConfig

    root = AppConfig.load().cs2_install_path
    if root is None or not (root / "game" / "csgo" / "pak01_dir.vpk").exists():
        pytest.skip("CS2 install path not configured (docs/NEEDS-FROM-YOU.md item N2)")
    return root
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_extract.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'counterstrat.radar'`.

- [x] **Step 3: Write `src/counterstrat/radar/extract.py`**

```python
"""Radar asset extraction and overview calibration parsing (spec item N2).

The CS2 overhead radar art and its world->image calibration live in the game's
``pak01_dir.vpk``, not in the per-map VPKs this repo vendors. Both are pulled
out with the vendored Source2Viewer CLI and cached per map under
``<data_root>/radar/<map_name>/``.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Every shipped CS2 overhead radar decompiles to a 1024x1024 RGBA PNG.
RADAR_IMAGE_PX = 1024

_VPK_REL = Path("game") / "csgo" / "pak01_dir.vpk"

# KeyValues values are quoted and tab separated, with trailing "// comment" text.
_FULL_LINE_COMMENT = re.compile(r"(?m)^[ \t]*//.*$")
_TRAILING_COMMENT = re.compile(r"[ \t]//[^\n]*")
_PAIR = re.compile(r'"([^"]+)"[ \t\r\n]*"([^"]*)"')
_LOWER_BLOCK = re.compile(r'"lower"[^{]*\{(.*?)\}', re.DOTALL | re.IGNORECASE)


class RadarCalibration(BaseModel):
    """World->radar-image transform for one map, read from resource/overviews."""

    map_name: str
    pos_x: float          # world X at image column 0
    pos_y: float          # world Y at image row 0
    scale: float          # world units per image pixel
    image_px: int = RADAR_IMAGE_PX
    lower_altitude_max: float | None = None  # None => single-level map


class RadarAssets(BaseModel):
    """Cached on-disk radar bundle for one map."""

    map_name: str
    image: Path
    lower_image: Path | None = None
    overview: Path
    calibration: RadarCalibration

    @property
    def levels(self) -> list[str]:
        return ["default", "lower"] if self.lower_image else ["default"]


def radar_cache_dir(data_root: Path, map_name: str) -> Path:
    return Path(data_root) / "radar" / map_name


def _strip_comments(text: str) -> str:
    return _TRAILING_COMMENT.sub("", _FULL_LINE_COMMENT.sub("", text))


def parse_calibration(
    text: str, map_name: str, image_px: int = RADAR_IMAGE_PX
) -> RadarCalibration:
    """Parses an overview KeyValues file body into a :class:`RadarCalibration`.

    Only the keys this project needs are read: ``pos_x``, ``pos_y``, ``scale``
    and ``verticalsections."lower".AltitudeMax``. ``rotate``/``zoom`` are
    recognised and ignored (every shipped competitive map has them at 0).
    """
    body = _strip_comments(text)
    flat = {k.lower(): v.strip() for k, v in _PAIR.findall(body)}

    missing = [k for k in ("pos_x", "pos_y", "scale") if k not in flat]
    if missing:
        raise ValueError(f"overview for '{map_name}' is missing {', '.join(missing)}")

    for key in ("rotate", "zoom"):
        raw = flat.get(key, "0")
        try:
            if float(raw) != 0.0:
                logger.warning(
                    "overview for %s sets %s=%s; radar projection ignores it", map_name, key, raw
                )
        except ValueError:  # noqa: PERF203
            logger.warning("overview for %s has non-numeric %s=%r", map_name, key, raw)

    lower_max: float | None = None
    lower = _LOWER_BLOCK.search(body)
    if lower:
        inner = {k.lower(): v.strip() for k, v in _PAIR.findall(lower.group(1))}
        if "altitudemax" in inner:
            lower_max = float(inner["altitudemax"])

    scale = float(flat["scale"])
    if scale <= 0:
        raise ValueError(f"overview for '{map_name}' has non-positive scale {scale}")

    return RadarCalibration(
        map_name=map_name,
        pos_x=float(flat["pos_x"]),
        pos_y=float(flat["pos_y"]),
        scale=scale,
        image_px=image_px,
        lower_altitude_max=lower_max,
    )


def _run_vrf(
    vrf_cli: Path, vpk: Path, out_dir: Path, filters: list[str], *, decompile: bool, timeout: float
) -> None:
    """Runs one VRF extraction. Filter entries absent from the VPK are skipped."""
    args = [str(vrf_cli), "-i", str(vpk), "-o", str(out_dir), "-f", ",".join(filters)]
    if decompile:
        args.append("-d")
    subprocess.run(args, check=True, capture_output=True, timeout=timeout)


def load_cached_assets(data_root: Path, map_name: str) -> RadarAssets | None:
    """Returns the cached bundle for ``map_name``, or ``None`` when not extracted."""
    cache = radar_cache_dir(data_root, map_name)
    image, overview = cache / "radar.png", cache / "overview.txt"
    if not (image.exists() and overview.exists()):
        return None
    cal_json = cache / "calibration.json"
    if cal_json.exists():
        cal = RadarCalibration.model_validate_json(cal_json.read_text(encoding="utf-8"))
    else:
        cal = parse_calibration(overview.read_text(encoding="utf-8", errors="replace"), map_name)
    lower = cache / "radar_lower.png"
    return RadarAssets(
        map_name=map_name,
        image=image,
        lower_image=lower if lower.exists() else None,
        overview=overview,
        calibration=cal,
    )


def extract_radar_assets(
    cs2_install: Path, vrf_cli: Path, map_name: str, data_root: Path, *, timeout: float = 300.0
) -> RadarAssets:
    """Extracts radar PNG(s) + overview for ``map_name`` into the per-map cache."""
    vpk = Path(cs2_install) / _VPK_REL
    if not vpk.exists():
        raise FileNotFoundError(f"CS2 VPK index not found at {vpk}")

    cache = radar_cache_dir(data_root, map_name)
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="csradar_") as tmp:
        staged = Path(tmp)
        _run_vrf(
            vrf_cli,
            vpk,
            staged,
            [
                f"panorama/images/overheadmaps/{map_name}_radar_psd.vtex_c",
                f"panorama/images/overheadmaps/{map_name}_lower_radar_psd.vtex_c",
            ],
            decompile=True,
            timeout=timeout,
        )
        _run_vrf(
            vrf_cli, vpk, staged, [f"resource/overviews/{map_name}.txt"],
            decompile=False, timeout=timeout,
        )

        overheads = staged / "panorama" / "images" / "overheadmaps"
        src_image = overheads / f"{map_name}_radar_psd.png"
        src_lower = overheads / f"{map_name}_lower_radar_psd.png"
        src_overview = staged / "resource" / "overviews" / f"{map_name}.txt"
        if not src_image.exists() or not src_overview.exists():
            raise FileNotFoundError(
                f"pak01 has no radar assets for '{map_name}' "
                f"(image={src_image.exists()}, overview={src_overview.exists()})"
            )

        image = cache / "radar.png"
        overview = cache / "overview.txt"
        shutil.copyfile(src_image, image)
        shutil.copyfile(src_overview, overview)
        lower_image: Path | None = None
        if src_lower.exists():
            lower_image = cache / "radar_lower.png"
            shutil.copyfile(src_lower, lower_image)

    cal = parse_calibration(overview.read_text(encoding="utf-8", errors="replace"), map_name)
    (cache / "calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
    return RadarAssets(
        map_name=map_name,
        image=image,
        lower_image=lower_image,
        overview=overview,
        calibration=cal,
    )


def get_radar_assets(
    data_root: Path,
    map_name: str,
    cs2_install: Path | None,
    vrf_cli: Path | None,
    *,
    force: bool = False,
) -> RadarAssets:
    """Cache-first accessor: extracts on miss, raises when N2 is unsatisfied."""
    if not force:
        cached = load_cached_assets(data_root, map_name)
        if cached is not None:
            return cached
    if cs2_install is None:
        raise FileNotFoundError(
            "no cached radar for "
            f"'{map_name}' and cs2_install_path is not configured (see item N2)"
        )
    if vrf_cli is None:
        raise FileNotFoundError("Source2Viewer-CLI not found under tools/vrf/")
    return extract_radar_assets(cs2_install, vrf_cli, map_name, data_root)
```

Create `src/counterstrat/radar/__init__.py` (mirrors `lake/__init__.py:1-6`):

```python
"""Radar visualizer: overhead map assets, projection, and overlay layers."""

from counterstrat.radar.extract import (
    RADAR_IMAGE_PX,
    RadarAssets,
    RadarCalibration,
    get_radar_assets,
    parse_calibration,
)

__all__ = [
    "RADAR_IMAGE_PX",
    "RadarAssets",
    "RadarCalibration",
    "get_radar_assets",
    "parse_calibration",
]
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_extract.py -v`
Expected: all PASS. The three `demo`-marked tests pass on this machine
(CS2 install configured); on a machine without it they report SKIPPED with
"CS2 install path not configured".

- [x] **Step 5: Lint**

Run: `.venv\Scripts\python.exe -m ruff check .`
Expected: `All checks passed!`

- [x] **Step 6: Commit**

```bash
git add src/counterstrat/radar tests/test_radar_extract.py tests/conftest.py
git commit -m "feat(radar): extract CS2 overhead radar assets and parse overview calibration"
```

**Done when:**
- `.venv\Scripts\python.exe -m pytest tests/test_radar_extract.py -v` reports
  9 passed, 0 failed (or 6 passed / 3 skipped without a CS2 install).
- `.venv\Scripts\python.exe -m ruff check .` prints `All checks passed!`.
- After the run, `<tmp>/radar/de_anubis/` contained `radar.png` (>10 KB),
  `overview.txt`, and `calibration.json`.

---

## Task 2: Projection math

**Goal:** Convert world `(X, Y, Z)` into radar image pixels and normalized
`[0, 1]` coordinates, in both scalar and polars-expression form, plus the two
enum normalizers the lake needs.

**Difficulty:** EASY — **Route:** EASY
**Verify:** full

**Files:**
- Create: `src/counterstrat/radar/coords.py`
- Create: `tests/test_radar_coords.py`
- Modify: `src/counterstrat/radar/__init__.py` (extend re-exports)

**Interfaces:**
- Consumes: `RadarCalibration` from Task 1.
- Produces:
```python
def game_to_pixel(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]: ...
def game_to_norm(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]: ...
def pixel_to_game(cal: RadarCalibration, u: float, v: float) -> tuple[float, float]: ...
def norm_x_expr(cal: RadarCalibration, col: str = "X") -> pl.Expr: ...
def norm_y_expr(cal: RadarCalibration, col: str = "Y") -> pl.Expr: ...
def level_expr(cal: RadarCalibration, col: str = "Z") -> pl.Expr: ...   # "lower"/"default"
def is_lower_level(cal: RadarCalibration, z: float) -> bool: ...
def normalize_side(value: str | None) -> str | None: ...                # -> "T" | "CT" | None
def side_expr(col: str) -> pl.Expr: ...                                 # vectorized normalize_side
ROUND_DP: int = 4
```

- [x] **Step 1: Write the failing tests**

Create `tests/test_radar_coords.py`:

```python
import polars as pl
import pytest

from counterstrat.radar.coords import (
    game_to_norm,
    game_to_pixel,
    is_lower_level,
    level_expr,
    norm_x_expr,
    norm_y_expr,
    normalize_side,
    pixel_to_game,
    side_expr,
)
from counterstrat.radar.extract import RadarCalibration

# Deliberately round numbers: 2048 world units across 1024 px at 2 u/px, so
# world (0, 0) sits dead centre and the corners are exactly (0,0) and (1,1).
CAL = RadarCalibration(
    map_name="de_test", pos_x=-1024.0, pos_y=1024.0, scale=2.0, image_px=1024
)
NUKE = RadarCalibration(
    map_name="de_nuke", pos_x=-3453.0, pos_y=2887.0, scale=7.0, lower_altitude_max=-495.0
)
ANUBIS = RadarCalibration(map_name="de_anubis", pos_x=-2796.0, pos_y=3328.0, scale=5.22)


def test_game_to_pixel_origin_and_corners() -> None:
    assert game_to_pixel(CAL, -1024.0, 1024.0) == (0.0, 0.0)
    assert game_to_pixel(CAL, 0.0, 0.0) == (512.0, 512.0)
    assert game_to_pixel(CAL, 1024.0, -1024.0) == (1024.0, 1024.0)


def test_game_to_norm_is_pixel_over_image_px() -> None:
    assert game_to_norm(CAL, 0.0, 0.0) == (0.5, 0.5)
    assert game_to_norm(CAL, -1024.0, 1024.0) == (0.0, 0.0)
    assert game_to_norm(CAL, 512.0, -512.0) == (0.75, 0.75)


def test_game_to_norm_matches_real_anubis_bomb_plant() -> None:
    # Verified against data/lake/07d1db1b3671d660/bomb.parquet round 1 plant.
    u, v = game_to_norm(ANUBIS, 1283.711304, 1873.848755)
    assert u == pytest.approx(0.7632362, abs=1e-6)
    assert v == pytest.approx(0.2720440, abs=1e-6)


def test_pixel_to_game_round_trips() -> None:
    for x, y in [(0.0, 0.0), (-800.5, 733.25), (1024.0, -1024.0)]:
        u, v = game_to_pixel(CAL, x, y)
        rx, ry = pixel_to_game(CAL, u, v)
        assert rx == pytest.approx(x)
        assert ry == pytest.approx(y)


def test_norm_exprs_match_scalar_form() -> None:
    df = pl.DataFrame({"X": [0.0, 512.0, -1024.0], "Y": [0.0, -512.0, 1024.0]})
    got = df.select(norm_x_expr(CAL).alias("u"), norm_y_expr(CAL).alias("v"))
    assert got["u"].to_list() == [0.5, 0.75, 0.0]
    assert got["v"].to_list() == [0.5, 0.75, 0.0]


def test_norm_exprs_accept_alternate_columns() -> None:
    df = pl.DataFrame({"victim_X": [0.0], "victim_Y": [0.0]})
    got = df.select(
        norm_x_expr(CAL, "victim_X").alias("u"), norm_y_expr(CAL, "victim_Y").alias("v")
    )
    assert got.row(0) == (0.5, 0.5)


def test_level_classification_single_and_multi_level() -> None:
    assert is_lower_level(CAL, -9999.0) is False        # no lower_altitude_max
    assert is_lower_level(NUKE, -495.0) is True         # boundary is inclusive
    assert is_lower_level(NUKE, -494.9) is False
    df = pl.DataFrame({"Z": [-600.0, 0.0]})
    assert df.select(level_expr(NUKE))["Z"].to_list() == ["lower", "default"]
    assert df.select(level_expr(CAL))["Z"].to_list() == ["default", "default"]


def test_normalize_side_handles_every_lake_spelling() -> None:
    assert normalize_side("TERRORIST") == "T"
    assert normalize_side("t") == "T"
    assert normalize_side("T") == "T"
    assert normalize_side("ct") == "CT"
    assert normalize_side("CT") == "CT"
    assert normalize_side(None) is None
    assert normalize_side("") is None
    assert normalize_side("Spectator") is None


def test_side_expr_matches_scalar_form() -> None:
    df = pl.DataFrame({"team_name": ["CT", "TERRORIST", None, "weird"]})
    got = df.select(side_expr("team_name"))["team_name"].to_list()
    assert got == ["CT", "T", None, None]
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_coords.py -v`
Expected: `ModuleNotFoundError: No module named 'counterstrat.radar.coords'`.

- [x] **Step 3: Write `src/counterstrat/radar/coords.py`**

```python
"""World <-> radar-image projection.

awpy ships equivalent helpers (``awpy.plot.utils.game_to_pixel_axis``) but they
read ``awpy.data.map_data.MAP_DATA``, which is populated only by a network
download that is unavailable here (verified empty in this venv). These functions
take the calibration this project extracts from the game files instead.
"""

import polars as pl

from counterstrat.radar.extract import RadarCalibration

ROUND_DP = 4

_SIDES = {"t": "T", "terrorist": "T", "ct": "CT", "counter-terrorist": "CT"}


def game_to_pixel(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]:
    """World (X, Y) -> radar image pixel (u, v), origin top-left."""
    return ((x - cal.pos_x) / cal.scale, (cal.pos_y - y) / cal.scale)


def game_to_norm(cal: RadarCalibration, x: float, y: float) -> tuple[float, float]:
    """World (X, Y) -> image-relative (u, v) in [0, 1] for a 1:1 square radar."""
    u, v = game_to_pixel(cal, x, y)
    return (u / cal.image_px, v / cal.image_px)


def pixel_to_game(cal: RadarCalibration, u: float, v: float) -> tuple[float, float]:
    """Radar image pixel (u, v) -> world (X, Y). Inverse of :func:`game_to_pixel`."""
    return (u * cal.scale + cal.pos_x, cal.pos_y - v * cal.scale)


def norm_x_expr(cal: RadarCalibration, col: str = "X") -> pl.Expr:
    """Vectorised ``game_to_norm`` u-component over ``col``; keeps the column name."""
    return (pl.col(col) - cal.pos_x) / (cal.scale * cal.image_px)


def norm_y_expr(cal: RadarCalibration, col: str = "Y") -> pl.Expr:
    """Vectorised ``game_to_norm`` v-component over ``col``; keeps the column name."""
    return (cal.pos_y - pl.col(col)) / (cal.scale * cal.image_px)


def is_lower_level(cal: RadarCalibration, z: float) -> bool:
    """True when ``z`` belongs to the map's lower radar image (inclusive bound)."""
    if cal.lower_altitude_max is None:
        return False
    return z <= cal.lower_altitude_max


def level_expr(cal: RadarCalibration, col: str = "Z") -> pl.Expr:
    """Vectorised level label: ``"lower"`` or ``"default"``."""
    if cal.lower_altitude_max is None:
        return pl.lit("default").alias(col)
    return (
        pl.when(pl.col(col) <= cal.lower_altitude_max)
        .then(pl.lit("lower"))
        .otherwise(pl.lit("default"))
        .alias(col)
    )


def normalize_side(value: str | None) -> str | None:
    """Collapses the lake's side spellings (``TERRORIST``/``t``/``ct``) to T/CT."""
    if value is None:
        return None
    return _SIDES.get(str(value).strip().lower())


def side_expr(col: str) -> pl.Expr:
    """Vectorised :func:`normalize_side` over ``col``; keeps the column name."""
    return (
        pl.col(col)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(_SIDES, default=None)
        .alias(col)
    )
```

Extend `src/counterstrat/radar/__init__.py` re-exports with `game_to_norm`,
`game_to_pixel`, `normalize_side` (keep `__all__` sorted).

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_coords.py -v`
Expected: 9 passed.

If `replace_strict` rejects the `default=None` keyword on the installed polars,
substitute the equivalent chained form and re-run:
```python
pl.when(pl.col(col).cast(pl.String).str.to_lowercase().is_in(["t", "terrorist"]))
  .then(pl.lit("T"))
  .when(pl.col(col).cast(pl.String).str.to_lowercase() == "ct")
  .then(pl.lit("CT"))
  .otherwise(None)
  .alias(col)
```

- [x] **Step 5: Lint and commit**

```bash
.venv\Scripts\python.exe -m ruff check .
git add src/counterstrat/radar tests/test_radar_coords.py
git commit -m "feat(radar): world-to-radar projection math and side normalizers"
```

**Done when:**
- `.venv\Scripts\python.exe -m pytest tests/test_radar_coords.py -v` reports
  9 passed, 0 failed — including the real-data anchor asserting
  `game_to_norm(anubis, 1283.711304, 1873.848755) == (0.7632362, 0.2720440)`.
- `.venv\Scripts\python.exe -m ruff check .` prints `All checks passed!`.

---

## Task 3: Coordinate layer builders

**Goal:** Turn the lake's parquet tables into five JSON-ready overlay layers in
normalized radar coordinates, scoped to one team (and optionally one side, one
level, a round subset).

**Difficulty:** HARD — **Route:** HARD
**Verify:** isolated (verify this task immediately; every later task consumes
its payload shape)

Why HARD: the team/side scoping is a cross-cutting correctness invariant (a
wrong join silently shows the *opponent's* positions), steamid dtypes disagree
across tables, and `ticks` is 16 Hz / ~487 k rows per match so payload size and
decimation pull against fidelity.

**Files:**
- Create: `src/counterstrat/radar/layers.py`
- Create: `tests/test_radar_layers.py`
- Modify: `tests/conftest.py` (append `radar_test_cal()` + `build_radar_lake()`)
- Modify: `src/counterstrat/radar/__init__.py` (re-export `LayerFilters`, `build_layers`)

**Interfaces:**
- Consumes: `RadarCalibration` (Task 1); `norm_x_expr`, `norm_y_expr`,
  `level_expr`, `side_expr`, `ROUND_DP` (Task 2).
- Produces:
```python
PROJECTILE_KIND: dict[str, str]     # grenade_type -> "smoke"|"flash"|"he"|"molotov"|"decoy"
BOMB_EVENTS: tuple[str, ...]        # ("plant", "defuse")

class LayerFilters(BaseModel):
    side: Literal["T", "CT"] | None = None
    level: Literal["default", "lower", "all"] = "all"
    round_nums: list[int] | None = None
    trail_rounds: int | None = 4     # trails only; heatmap/duels use every scoped round
    stride: int = 8                  # tick decimation for trails (16 Hz / 8 = 2 Hz)
    grid: int = 128                  # heatmap cells per axis

@dataclass(slots=True)
class TeamScope:
    team_key: str
    rounds: pl.DataFrame     # match_id, round_num, side("T"/"CT")
    members: pl.DataFrame    # match_id, round_num, steamid(Int64)

def lake_frames(lake_root: Path, match_ids: list[str]) -> dict[str, pl.LazyFrame | None]: ...
def build_team_scope(rosters: pl.LazyFrame | None, team_key: str, f: LayerFilters) -> TeamScope: ...
def heatmap_layer(ticks, cal, scope, f) -> dict: ...
def trails_layer(ticks, cal, scope, f) -> list[dict]: ...
def utility_layer(grenades, cal, scope, f) -> list[dict]: ...
def duels_layer(kills, cal, scope, f) -> list[dict]: ...
def bomb_layer(bomb, cal, scope, f) -> list[dict]: ...
def build_layers(frames: dict[str, pl.LazyFrame | None], cal: RadarCalibration,
                 team_key: str, f: LayerFilters) -> dict: ...
```

Payload contract that Tasks 4 and 5 code against:
```json
{"map_name": "de_anubis", "team_key": "abc", "image_px": 1024,
 "lower_altitude_max": null,
 "filters": {"side": null, "level": "all", "round_nums": null,
             "trail_rounds": 4, "stride": 8, "grid": 128},
 "rounds": [{"match_id": "m1", "round_num": 1, "side": "CT"}],
 "layers": {
   "heatmap": {"grid": 128, "max": 4, "samples": 16, "cells": [[gx, gy, n]]},
   "trails":  [{"match_id": "m1", "round_num": 1, "steamid": "1", "name": "pA1",
                "side": "CT", "points": [[u, v, clock_s]]}],
   "utility": [{"kind": "smoke", "match_id": "m1", "round_num": 1, "thrower": "pA1",
                "steamid": "1", "u": 0.75, "v": 0.75, "from_u": 0.25, "from_v": 0.25,
                "level": "default"}],
   "duels":   [{"match_id": "m1", "round_num": 1, "tick": 10001, "weapon": "ak47",
                "headshot": true, "by_team": true, "level": "default",
                "victim": {"name": "pB1", "side": "T", "u": 0.75, "v": 0.5},
                "attacker": {"name": "pA1", "side": "CT", "u": 0.5, "v": 0.5}}],
   "bombs":   [{"match_id": "m1", "round_num": 2, "event": "plant", "bombsite": "BombsiteA",
                "name": "pA1", "steamid": "1", "tick": 20050, "u": 0.5, "v": 0.5,
                "level": "default"}]}}
```

**Non-obvious rules the implementation MUST follow:**
1. Every `steamid` crossing the wire is a **string** — raw CS2 steamids exceed
   JavaScript's 2^53 safe integer range.
2. Cast every steamid to `pl.Int64` before joining: `ticks`/`kills`/`bomb`/
   `grenades` store `UInt64`, `rosters.steamids` is `List(Int64)`.
3. Cast every `round_num` to `pl.Int64` before joining: `rosters` is `Int64`,
   every other table is `UInt32`.
4. Only `C*Projectile` `grenade_type` values are thrown nades (verified fact 7).
5. `ticks` has **no `side` column** — normalize `team_name` with `side_expr`.
6. Drop out-of-image points (`u`/`v` outside `[0, 1)`) from the heatmap only;
   other layers keep them so nothing silently disappears.
7. `trail_rounds` caps trails only, never the heatmap.

- [x] **Step 1: Append the synthetic-lake helper to `tests/conftest.py`**

```python
# --- Radar layer fixtures (spec item N2): a deterministic 2-round lake ---
#
# Geometry is chosen for exact assertions: with radar_test_cal(), world (0, 0)
# maps to normalized (0.5, 0.5) and every +-256 world units is exactly +-0.125.


def radar_test_cal():
    """A radar calibration spanning 2048 world units across 1024 px."""
    from counterstrat.radar.extract import RadarCalibration

    return RadarCalibration(
        map_name="de_test", pos_x=-1024.0, pos_y=1024.0, scale=2.0, image_px=1024
    )


def build_radar_lake(lake_root: Path, match_id: str = "m1") -> Path:
    """Writes rosters/ticks/kills/grenades/bomb parquet for one synthetic match.

    Roster: teamA = steamids 1,2 (CT in round 1, TERRORIST in round 2);
            teamB = steamids 3,4 (the mirror). Dtypes deliberately match the
            real lake, including the UInt64/Int64 steamid mismatch.
    """
    import polars as pl

    out = Path(lake_root) / match_id
    out.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "round_num": [1, 1, 2, 2],
            "side": ["CT", "TERRORIST", "TERRORIST", "CT"],
            "team_key": ["teamA", "teamB", "teamA", "teamB"],
            "steamids": [[1, 2], [3, 4], [1, 2], [3, 4]],
            "clan_name": ["Alpha", "Beta", "Alpha", "Beta"],
            "match_id": [match_id] * 4,
        },
        schema_overrides={"round_num": pl.Int64, "steamids": pl.List(pl.Int64)},
    ).write_parquet(out / "rosters.parquet")

    # steamid -> (dx, dy) walk direction; each of 4 samples steps 256 units.
    walks = {1: (256, 0), 2: (0, -256), 3: (-256, 0), 4: (0, 256)}
    names = {1: "pA1", 2: "pA2", 3: "pB1", 4: "pB2"}
    rows: list[dict] = []
    for rnd in (1, 2):
        for sid, (dx, dy) in walks.items():
            side = ("CT" if sid in (1, 2) else "TERRORIST") if rnd == 1 else (
                "TERRORIST" if sid in (1, 2) else "CT"
            )
            for i in range(4):
                rows.append(
                    {
                        "match_id": match_id,
                        "round_num": rnd,
                        "tick": rnd * 10000 + i,
                        "steamid": sid,
                        "name": names[sid],
                        "team_name": side,
                        "X": float(dx * i),
                        "Y": float(dy * i),
                        "Z": 0.0,
                        "clock_s": float(i),
                        "is_alive": True,
                    }
                )
    # One dead sample sharing steamid 1's last cell: proves is_alive filtering.
    rows.append(
        {
            "match_id": match_id, "round_num": 2, "tick": 20009, "steamid": 1,
            "name": "pA1", "team_name": "TERRORIST", "X": 768.0, "Y": 0.0, "Z": 0.0,
            "clock_s": 9.0, "is_alive": False,
        }
    )
    pl.DataFrame(
        rows,
        schema_overrides={
            "round_num": pl.UInt32, "tick": pl.Int32, "steamid": pl.UInt64,
            "X": pl.Float32, "Y": pl.Float32, "Z": pl.Float32, "clock_s": pl.Float64,
        },
    ).write_parquet(out / "ticks.parquet")

    pl.DataFrame(
        {
            "match_id": [match_id] * 3,
            "round_num": [1, 2, 2],
            "tick": [10001, 20001, 20002],
            "weapon": ["ak47", "m4a1", None],
            "headshot": [True, False, False],
            "attacker_X": [0.0, 0.0, None],
            "attacker_Y": [0.0, 0.0, None],
            "attacker_Z": [0.0, 0.0, None],
            "attacker_name": ["pA1", "pB1", None],
            "attacker_side": ["ct", "ct", None],       # lowercase, as awpy writes it
            "victim_X": [512.0, -512.0, 0.0],
            "victim_Y": [0.0, 0.0, 768.0],
            "victim_Z": [0.0, 0.0, 0.0],
            "victim_name": ["pB1", "pA1", "pB2"],
            "victim_side": ["t", "t", "ct"],
        },
        schema_overrides={
            "round_num": pl.UInt32, "tick": pl.Int32,
            "attacker_X": pl.Float32, "attacker_Y": pl.Float32, "attacker_Z": pl.Float32,
            "victim_X": pl.Float32, "victim_Y": pl.Float32, "victim_Z": pl.Float32,
        },
    ).write_parquet(out / "kills.parquet")

    # entity 10: teamA smoke (kept). 11: teamB flash (dropped, wrong team).
    # 12: teamA molotov (kept). 13: teamA "CFlashbang" held entity (dropped, not a projectile).
    gren = [
        (10, 1, 1, "CSmokeGrenadeProjectile", [(-512.0, 512.0), (0.0, 0.0), (512.0, -512.0)]),
        (11, 1, 3, "CFlashbangProjectile", [(0.0, 0.0), (256.0, 0.0)]),
        (12, 2, 2, "CMolotovProjectile", [(0.0, 0.0), (-256.0, 0.0)]),
        (13, 1, 1, "CFlashbang", [(0.0, 0.0), (256.0, 256.0)]),
    ]
    grows: list[dict] = []
    for entity_id, rnd, sid, gtype, path in gren:
        for i, (gx, gy) in enumerate(path):
            grows.append(
                {
                    "match_id": match_id, "round_num": rnd, "entity_id": entity_id,
                    "grenade_type": gtype, "thrower_steamid": sid, "thrower": names[sid],
                    "tick": rnd * 10000 + 100 + i, "X": gx, "Y": gy, "Z": 0.0,
                }
            )
    pl.DataFrame(
        grows,
        schema_overrides={
            "round_num": pl.UInt32, "entity_id": pl.Int32, "thrower_steamid": pl.UInt64,
            "tick": pl.Int32, "X": pl.Float32, "Y": pl.Float32, "Z": pl.Float32,
        },
    ).write_parquet(out / "grenades.parquet")

    pl.DataFrame(
        {
            "match_id": [match_id] * 3,
            "round_num": [1, 1, 2],
            "tick": [10020, 10050, 20050],
            "event": ["pickup", "defuse", "plant"],   # "pickup" must be dropped
            "X": [256.0, -512.0, 0.0],
            "Y": [0.0, 0.0, 0.0],
            "Z": [0.0, 0.0, 0.0],
            "steamid": [1, 3, 1],
            "name": ["pA1", "pB1", "pA1"],
            "bombsite": [None, "BombsiteA", "BombsiteA"],
        },
        schema_overrides={
            "round_num": pl.UInt32, "tick": pl.Int32, "steamid": pl.UInt64,
            "X": pl.Float32, "Y": pl.Float32, "Z": pl.Float32,
        },
    ).write_parquet(out / "bomb.parquet")

    return out
```

- [x] **Step 2: Write the failing tests**

Create `tests/test_radar_layers.py`:

```python
from pathlib import Path

import pytest
from conftest import build_radar_lake, radar_test_cal

from counterstrat.radar.layers import (
    LayerFilters,
    build_layers,
    build_team_scope,
    lake_frames,
)


@pytest.fixture
def frames(tmp_path: Path):
    build_radar_lake(tmp_path / "lake")
    return lake_frames(tmp_path / "lake", ["m1"])


@pytest.fixture
def cal():
    return radar_test_cal()


def test_lake_frames_reports_missing_tables_as_none(tmp_path: Path) -> None:
    got = lake_frames(tmp_path / "empty", ["m1"])
    assert set(got) == {"rosters", "ticks", "kills", "grenades", "bomb"}
    assert all(v is None for v in got.values())


def test_team_scope_tracks_side_swap(frames) -> None:
    scope = build_team_scope(frames["rosters"], "teamA", LayerFilters())
    rows = sorted(scope.rounds.iter_rows(named=True), key=lambda r: r["round_num"])
    assert [(r["round_num"], r["side"]) for r in rows] == [(1, "CT"), (2, "T")]
    assert sorted(scope.members["steamid"].unique().to_list()) == [1, 2]


def test_team_scope_side_filter_keeps_one_round(frames) -> None:
    scope = build_team_scope(frames["rosters"], "teamA", LayerFilters(side="CT"))
    assert scope.rounds.height == 1
    assert scope.rounds.row(0, named=True)["round_num"] == 1


def test_heatmap_counts_cells_and_excludes_dead_samples(frames, cal) -> None:
    payload = build_layers(frames, cal, "teamA", LayerFilters(grid=8))
    hm = payload["layers"]["heatmap"]
    cells = {(gx, gy): n for gx, gy, n in hm["cells"]}
    assert hm["grid"] == 8
    assert hm["samples"] == 16              # 2 rounds x 2 players x 4 alive samples
    assert len(cells) == 7
    assert cells[(4, 4)] == 4               # both players start dead centre, both rounds
    assert cells[(7, 4)] == 2               # NOT 3: the is_alive=False sample is excluded
    assert hm["max"] == 4


def test_heatmap_scoped_to_team_not_opponent(frames, cal) -> None:
    a = build_layers(frames, cal, "teamA", LayerFilters(grid=8))["layers"]["heatmap"]
    b = build_layers(frames, cal, "teamB", LayerFilters(grid=8))["layers"]["heatmap"]
    a_cells = {(gx, gy) for gx, gy, _ in a["cells"]}
    b_cells = {(gx, gy) for gx, gy, _ in b["cells"]}
    assert (7, 4) in a_cells and (7, 4) not in b_cells   # teamA walks +X
    assert (1, 4) in b_cells and (1, 4) not in a_cells   # teamB walks -X


def test_heatmap_side_filter_halves_samples(frames, cal) -> None:
    hm = build_layers(frames, cal, "teamA", LayerFilters(grid=8, side="CT"))
    assert hm["layers"]["heatmap"]["samples"] == 8
    assert hm["layers"]["heatmap"]["max"] == 2


def test_trails_one_per_player_round_with_normalized_points(frames, cal) -> None:
    trails = build_layers(
        frames, cal, "teamA", LayerFilters(stride=1, trail_rounds=None)
    )["layers"]["trails"]
    assert len(trails) == 4                              # 2 rounds x 2 players
    assert all(len(t["points"]) == 4 for t in trails)
    first = next(t for t in trails if t["round_num"] == 1 and t["steamid"] == "1")
    assert first["name"] == "pA1"
    assert first["side"] == "CT"
    assert first["points"][0] == [0.5, 0.5, 0.0]
    assert first["points"][3] == [0.875, 0.5, 3.0]
    second = next(t for t in trails if t["round_num"] == 2 and t["steamid"] == "1")
    assert second["side"] == "T"                         # side swapped at half


def test_trails_stride_decimates_and_trail_rounds_caps(frames, cal) -> None:
    strided = build_layers(
        frames, cal, "teamA", LayerFilters(stride=2, trail_rounds=None)
    )["layers"]["trails"]
    assert all(len(t["points"]) == 2 for t in strided)
    capped = build_layers(
        frames, cal, "teamA", LayerFilters(stride=1, trail_rounds=1)
    )["layers"]["trails"]
    assert len(capped) == 2
    assert {t["round_num"] for t in capped} == {1}


def test_utility_keeps_only_team_projectiles(frames, cal) -> None:
    util = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["utility"]
    assert [u["kind"] for u in util] == ["smoke", "molotov"]
    smoke = util[0]
    assert (smoke["from_u"], smoke["from_v"]) == (0.25, 0.25)   # throw origin
    assert (smoke["u"], smoke["v"]) == (0.75, 0.75)             # landing
    assert smoke["thrower"] == "pA1"
    assert smoke["steamid"] == "1"
    assert util[1]["u"] == 0.375
    # teamB's flashbang and the held (non-projectile) CFlashbang are both gone.
    assert all(u["kind"] != "flash" for u in util)


def test_duels_tag_by_team_and_tolerate_null_attacker(frames, cal) -> None:
    duels = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["duels"]
    assert len(duels) == 3
    assert [d["by_team"] for d in duels] == [True, False, False]
    assert duels[0]["victim"]["u"] == 0.75
    assert duels[0]["attacker"]["u"] == 0.5
    assert duels[0]["weapon"] == "ak47" and duels[0]["headshot"] is True
    assert duels[2]["attacker"] is None                 # null attacker_side/coords
    assert duels[2]["victim"]["v"] == 0.125


def test_bombs_keep_plant_and_defuse_only(frames, cal) -> None:
    bombs = build_layers(frames, cal, "teamA", LayerFilters())["layers"]["bombs"]
    assert [b["event"] for b in bombs] == ["defuse", "plant"]
    assert bombs[0]["u"] == 0.25
    assert bombs[1]["u"] == 0.5 and bombs[1]["bombsite"] == "BombsiteA"
    assert bombs[1]["steamid"] == "1"


def test_level_filter_uses_lower_altitude_max(tmp_path: Path) -> None:
    from counterstrat.radar.extract import RadarCalibration

    build_radar_lake(tmp_path / "lake")
    frames_ = lake_frames(tmp_path / "lake", ["m1"])
    multi = RadarCalibration(
        map_name="de_test", pos_x=-1024.0, pos_y=1024.0, scale=2.0,
        image_px=1024, lower_altitude_max=100.0,     # every synthetic Z=0 is "lower"
    )
    lower = build_layers(frames_, multi, "teamA", LayerFilters(grid=8, level="lower"))
    assert lower["layers"]["heatmap"]["samples"] == 16
    assert lower["layers"]["bombs"][0]["level"] == "lower"
    default = build_layers(frames_, multi, "teamA", LayerFilters(grid=8, level="default"))
    assert default["layers"]["heatmap"]["samples"] == 0
    assert default["layers"]["bombs"] == []


def test_build_layers_on_empty_lake_returns_empty_payload(tmp_path: Path, cal) -> None:
    payload = build_layers(lake_frames(tmp_path / "nope", ["m1"]), cal, "teamA", LayerFilters())
    assert payload["rounds"] == []
    assert payload["layers"]["heatmap"]["cells"] == []
    assert payload["layers"]["trails"] == []
    assert payload["layers"]["utility"] == []
    assert payload["layers"]["duels"] == []
    assert payload["layers"]["bombs"] == []


def test_unknown_team_key_yields_empty_layers(frames, cal) -> None:
    payload = build_layers(frames, cal, "nobody", LayerFilters())
    assert payload["rounds"] == []
    assert payload["layers"]["heatmap"]["samples"] == 0
    assert payload["layers"]["trails"] == []
```

- [x] **Step 3: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_layers.py -v`
Expected: `ModuleNotFoundError: No module named 'counterstrat.radar.layers'`.

- [x] **Step 4: Write `src/counterstrat/radar/layers.py`**

```python
"""Lake tables -> normalized radar overlay layers (spec item N2).

Every layer is scoped to one ``team_key`` through ``rosters``: the team's rounds
give the side it played, and its ``steamids`` give the players to keep. Getting
that scope wrong silently plots the opponent, so it is built once in
:func:`build_team_scope` and reused by every layer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel

from counterstrat.radar.coords import (
    ROUND_DP,
    level_expr,
    norm_x_expr,
    norm_y_expr,
    side_expr,
)
from counterstrat.radar.extract import RadarCalibration

logger = logging.getLogger(__name__)

LAKE_TABLES = ("rosters", "ticks", "kills", "grenades", "bomb")

# Only the *Projectile entities are thrown nades; the plain C*Grenade rows track
# the inventory-held entity for the whole round (~2600 rows each vs ~250).
PROJECTILE_KIND = {
    "CSmokeGrenadeProjectile": "smoke",
    "CFlashbangProjectile": "flash",
    "CHEGrenadeProjectile": "he",
    "CMolotovProjectile": "molotov",   # incendiaries spawn this too
    "CDecoyProjectile": "decoy",
}

BOMB_EVENTS = ("plant", "defuse")


class LayerFilters(BaseModel):
    side: Literal["T", "CT"] | None = None
    level: Literal["default", "lower", "all"] = "all"
    round_nums: list[int] | None = None
    trail_rounds: int | None = 4
    stride: int = 8
    grid: int = 128


@dataclass(slots=True)
class TeamScope:
    team_key: str
    rounds: pl.DataFrame
    members: pl.DataFrame


def _empty_scope(team_key: str) -> TeamScope:
    return TeamScope(
        team_key=team_key,
        rounds=pl.DataFrame(
            schema={"match_id": pl.String, "round_num": pl.Int64, "side": pl.String}
        ),
        members=pl.DataFrame(
            schema={"match_id": pl.String, "round_num": pl.Int64, "steamid": pl.Int64}
        ),
    )


def lake_frames(lake_root: Path, match_ids: list[str]) -> dict[str, pl.LazyFrame | None]:
    """Lazy scans of the radar-relevant tables for ``match_ids``; None when absent."""
    out: dict[str, pl.LazyFrame | None] = {}
    for table in LAKE_TABLES:
        paths = [
            p
            for m in match_ids
            if (p := Path(lake_root) / m / f"{table}.parquet").exists()
        ]
        out[table] = pl.scan_parquet(paths) if paths else None
    return out


def _keys(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Normalizes the join keys every table disagrees on."""
    return frame.with_columns(pl.col("round_num").cast(pl.Int64))


def _apply_level(frame: Any, f: LayerFilters) -> Any:
    return frame if f.level == "all" else frame.filter(pl.col("level") == f.level)


def _in_bounds() -> pl.Expr:
    return (
        (pl.col("u") >= 0) & (pl.col("u") < 1) & (pl.col("v") >= 0) & (pl.col("v") < 1)
    )


def _r(value: Any) -> float | None:
    return None if value is None else round(float(value), ROUND_DP)


def _sid(value: Any) -> str | None:
    """steamids cross the wire as strings: they exceed JS's 2**53 safe range."""
    return None if value is None else str(int(value))


def build_team_scope(
    rosters: pl.LazyFrame | None, team_key: str, f: LayerFilters
) -> TeamScope:
    """Resolves which (match, round, side) and which players belong to ``team_key``."""
    if rosters is None:
        return _empty_scope(team_key)

    rounds = (
        _keys(rosters.select("match_id", "round_num", "side", "team_key", "steamids"))
        .filter(pl.col("team_key") == team_key)
        .with_columns(side_expr("side"))
        .collect()
    )
    if f.side is not None:
        rounds = rounds.filter(pl.col("side") == f.side)
    if f.round_nums:
        rounds = rounds.filter(pl.col("round_num").is_in(f.round_nums))
    rounds = rounds.sort("match_id", "round_num")

    if rounds.is_empty():
        return _empty_scope(team_key)

    members = (
        rounds.explode("steamids")
        .rename({"steamids": "steamid"})
        .select("match_id", "round_num", pl.col("steamid").cast(pl.Int64))
        .drop_nulls()
        .unique()
    )
    return TeamScope(
        team_key=team_key,
        rounds=rounds.select("match_id", "round_num", "side"),
        members=members,
    )


def _team_ticks(
    ticks: pl.LazyFrame, cal: RadarCalibration, scope: TeamScope
) -> pl.LazyFrame:
    return (
        _keys(ticks)
        .with_columns(pl.col("steamid").cast(pl.Int64))
        .filter(pl.col("is_alive"))
        .join(scope.members.lazy(), on=["match_id", "round_num", "steamid"], how="semi")
        .with_columns(
            norm_x_expr(cal, "X").alias("u"),
            norm_y_expr(cal, "Y").alias("v"),
            level_expr(cal, "Z").alias("level"),
        )
    )


def heatmap_layer(
    ticks: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> dict[str, Any]:
    """Occupancy histogram over a ``f.grid`` x ``f.grid`` lattice of the radar image."""
    empty = {"grid": f.grid, "max": 0, "samples": 0, "cells": []}
    if ticks is None or scope.members.is_empty():
        return empty

    df = (
        _apply_level(_team_ticks(ticks, cal, scope), f)
        .filter(_in_bounds())
        .with_columns(
            (pl.col("u") * f.grid).floor().cast(pl.Int32).alias("gx"),
            (pl.col("v") * f.grid).floor().cast(pl.Int32).alias("gy"),
        )
        .group_by("gx", "gy")
        .agg(pl.len().alias("n"))
        .sort("gy", "gx")
        .collect()
    )
    cells = [[int(gx), int(gy), int(n)] for gx, gy, n in df.iter_rows()]
    return {
        "grid": f.grid,
        "max": max((c[2] for c in cells), default=0),
        "samples": sum(c[2] for c in cells),
        "cells": cells,
    }


def trails_layer(
    ticks: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """One decimated polyline per (round, player), capped at ``f.trail_rounds`` rounds."""
    if ticks is None or scope.members.is_empty():
        return []

    keep = scope.rounds.select("match_id", "round_num").unique().sort(
        "match_id", "round_num"
    )
    if f.trail_rounds is not None:
        keep = keep.head(f.trail_rounds)
    members = scope.members.join(keep, on=["match_id", "round_num"], how="semi")
    if members.is_empty():
        return []

    stride = max(1, f.stride)
    df = (
        _apply_level(
            _team_ticks(ticks, cal, TeamScope(scope.team_key, scope.rounds, members)), f
        )
        .with_columns(side_expr("team_name").alias("side"))
        .sort("match_id", "round_num", "steamid", "tick")
        .with_columns(
            pl.int_range(pl.len()).over("match_id", "round_num", "steamid").alias("i")
        )
        .filter(pl.col("i") % stride == 0)
        .group_by("match_id", "round_num", "steamid", maintain_order=True)
        .agg(
            pl.col("name").first(),
            pl.col("side").first(),
            pl.col("u").round(ROUND_DP),
            pl.col("v").round(ROUND_DP),
            pl.col("clock_s").round(1),
        )
        .sort("match_id", "round_num", "steamid")
        .collect()
    )
    return [
        {
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "steamid": _sid(r["steamid"]),
            "name": r["name"],
            "side": r["side"],
            "points": [
                [u, v, c]
                for u, v, c in zip(r["u"], r["v"], r["clock_s"], strict=True)
            ],
        }
        for r in df.iter_rows(named=True)
    ]


def utility_layer(
    grenades: pl.LazyFrame | None,
    cal: RadarCalibration,
    scope: TeamScope,
    f: LayerFilters,
) -> list[dict[str, Any]]:
    """The team's own thrown nades: throw origin + landing, per projectile entity."""
    if grenades is None or scope.members.is_empty():
        return []

    df = (
        _keys(
            grenades.select(
                "match_id", "round_num", "thrower_steamid", "thrower",
                "grenade_type", "entity_id", "X", "Y", "Z", "tick",
            )
        )
        .filter(pl.col("grenade_type").is_in(list(PROJECTILE_KIND)))
        .with_columns(pl.col("thrower_steamid").cast(pl.Int64).alias("steamid"))
        .join(scope.members.lazy(), on=["match_id", "round_num", "steamid"], how="semi")
        .group_by("match_id", "round_num", "entity_id")
        .agg(
            pl.col("grenade_type").first(),
            pl.col("thrower").first(),
            pl.col("steamid").first(),
            pl.col("tick").min().alias("throw_tick"),
            pl.col("X").sort_by("tick").first().alias("from_x"),
            pl.col("Y").sort_by("tick").first().alias("from_y"),
            pl.col("X").sort_by("tick").last().alias("land_x"),
            pl.col("Y").sort_by("tick").last().alias("land_y"),
            pl.col("Z").sort_by("tick").last().alias("land_z"),
        )
        .with_columns(
            norm_x_expr(cal, "land_x").alias("u"),
            norm_y_expr(cal, "land_y").alias("v"),
            norm_x_expr(cal, "from_x").alias("from_u"),
            norm_y_expr(cal, "from_y").alias("from_v"),
            level_expr(cal, "land_z").alias("level"),
        )
        .sort("match_id", "round_num", "throw_tick")
        .collect()
    )
    df = _apply_level(df, f)
    return [
        {
            "kind": PROJECTILE_KIND[r["grenade_type"]],
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "thrower": r["thrower"],
            "steamid": _sid(r["steamid"]),
            "u": _r(r["u"]),
            "v": _r(r["v"]),
            "from_u": _r(r["from_u"]),
            "from_v": _r(r["from_v"]),
            "level": r["level"],
        }
        for r in df.iter_rows(named=True)
    ]


def duels_layer(
    kills: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """Every kill in the team's rounds, tagged ``by_team`` for the team's own kills."""
    if kills is None or scope.rounds.is_empty():
        return []

    df = (
        _keys(
            kills.select(
                "match_id", "round_num", "tick", "weapon", "headshot",
                "attacker_X", "attacker_Y", "attacker_Z", "attacker_name", "attacker_side",
                "victim_X", "victim_Y", "victim_Z", "victim_name", "victim_side",
            )
        )
        .with_columns(side_expr("attacker_side"), side_expr("victim_side"))
        .join(scope.rounds.lazy(), on=["match_id", "round_num"], how="inner")
        .with_columns(
            norm_x_expr(cal, "victim_X").alias("u"),
            norm_y_expr(cal, "victim_Y").alias("v"),
            norm_x_expr(cal, "attacker_X").alias("au"),
            norm_y_expr(cal, "attacker_Y").alias("av"),
            level_expr(cal, "victim_Z").alias("level"),
            (pl.col("attacker_side") == pl.col("side")).fill_null(value=False).alias("by_team"),
        )
        .sort("match_id", "round_num", "tick")
        .collect()
    )
    df = _apply_level(df, f)
    out: list[dict[str, Any]] = []
    for r in df.iter_rows(named=True):
        attacker = None
        if r["au"] is not None and r["av"] is not None:
            attacker = {
                "name": r["attacker_name"],
                "side": r["attacker_side"],
                "u": _r(r["au"]),
                "v": _r(r["av"]),
            }
        out.append(
            {
                "match_id": r["match_id"],
                "round_num": int(r["round_num"]),
                "tick": int(r["tick"]),
                "weapon": r["weapon"],
                "headshot": bool(r["headshot"]),
                "by_team": bool(r["by_team"]),
                "level": r["level"],
                "victim": {
                    "name": r["victim_name"],
                    "side": r["victim_side"],
                    "u": _r(r["u"]),
                    "v": _r(r["v"]),
                },
                "attacker": attacker,
            }
        )
    return out


def bomb_layer(
    bomb: pl.LazyFrame | None, cal: RadarCalibration, scope: TeamScope, f: LayerFilters
) -> list[dict[str, Any]]:
    """Plant and defuse positions inside the team's rounds."""
    if bomb is None or scope.rounds.is_empty():
        return []

    df = (
        _keys(
            bomb.select(
                "match_id", "round_num", "tick", "event", "X", "Y", "Z",
                "steamid", "name", "bombsite",
            )
        )
        .filter(pl.col("event").is_in(list(BOMB_EVENTS)))
        .join(scope.rounds.lazy(), on=["match_id", "round_num"], how="inner")
        .with_columns(
            norm_x_expr(cal, "X").alias("u"),
            norm_y_expr(cal, "Y").alias("v"),
            level_expr(cal, "Z").alias("level"),
        )
        .sort("match_id", "round_num", "tick")
        .collect()
    )
    df = _apply_level(df, f)
    return [
        {
            "match_id": r["match_id"],
            "round_num": int(r["round_num"]),
            "event": r["event"],
            "bombsite": r["bombsite"],
            "name": r["name"],
            "steamid": _sid(r["steamid"]),
            "tick": int(r["tick"]),
            "u": _r(r["u"]),
            "v": _r(r["v"]),
            "level": r["level"],
        }
        for r in df.iter_rows(named=True)
    ]


def build_layers(
    frames: dict[str, pl.LazyFrame | None],
    cal: RadarCalibration,
    team_key: str,
    f: LayerFilters,
) -> dict[str, Any]:
    """Assembles the full radar payload for one (team, map) selection."""
    scope = build_team_scope(frames.get("rosters"), team_key, f)
    return {
        "map_name": cal.map_name,
        "team_key": team_key,
        "image_px": cal.image_px,
        "lower_altitude_max": cal.lower_altitude_max,
        "filters": f.model_dump(),
        "rounds": [
            {"match_id": r["match_id"], "round_num": int(r["round_num"]), "side": r["side"]}
            for r in scope.rounds.iter_rows(named=True)
        ],
        "layers": {
            "heatmap": heatmap_layer(frames.get("ticks"), cal, scope, f),
            "trails": trails_layer(frames.get("ticks"), cal, scope, f),
            "utility": utility_layer(frames.get("grenades"), cal, scope, f),
            "duels": duels_layer(frames.get("kills"), cal, scope, f),
            "bombs": bomb_layer(frames.get("bomb"), cal, scope, f),
        },
    }
```

Extend `src/counterstrat/radar/__init__.py` with `LayerFilters`, `build_layers`,
`lake_frames` (keep `__all__` sorted).

- [x] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_layers.py -v`
Expected: 14 passed.

Debugging notes if a test fails:
- Empty layer with a non-empty scope → a join key dtype mismatch. Print
  `scope.members.dtypes` and the frame's `collect_schema()`; every join key must
  be `String, Int64, Int64`.
- `pl.int_range(pl.len()).over(...)` unsupported → replace with
  `pl.cum_count("tick").over("match_id", "round_num", "steamid") - 1`.
- `fill_null(value=False)` signature error → use `fill_null(False)`.
- `_apply_level` on a `DataFrame` and a `LazyFrame` both work because both
  expose `.filter`; keep the `Any` annotation rather than overloading.

- [x] **Step 6: Lint and commit**

```bash
.venv\Scripts\python.exe -m ruff check .
git add src/counterstrat/radar tests/test_radar_layers.py tests/conftest.py
git commit -m "feat(radar): build heatmap, trail, utility, duel and bomb coordinate layers"
```

**Done when:**
- `.venv\Scripts\python.exe -m pytest tests/test_radar_layers.py -v` reports
  14 passed, 0 failed.
- `.venv\Scripts\python.exe -m pytest -q` still reports 0 failures (the whole
  suite, to prove the `conftest.py` append broke nothing).
- `.venv\Scripts\python.exe -m ruff check .` prints `All checks passed!`.
- `test_heatmap_scoped_to_team_not_opponent` passes — the proof that team
  scoping is real and not accidentally showing every player.

---

## Task 4: API endpoints

**Goal:** Serve the radar background image, its calibration, and the team layer
payload over HTTP.

**Difficulty:** MEDIUM — **Route:** EASY
**Verify:** full

**Files:**
- Create: `src/counterstrat/web/radar_api.py`
- Modify: `src/counterstrat/web/app.py` (import at `:12`, include at `:26`)
- Create: `tests/test_radar_api.py`

**Interfaces:**
- Consumes: `load_cached_assets`, `extract_radar_assets`, `RadarAssets`
  (Task 1); `LayerFilters`, `build_layers`, `lake_frames` (Task 3);
  `ConfigDep` from `counterstrat/web/routes.py:66`; `_find_vrf_cli` from
  `counterstrat/web/ingest.py:74`; `load_manifest` from
  `counterstrat/corpus.py:51` (returns `dict[str, DemoRecord]`, `{}` when the
  file is absent).
- Produces (the frontend in Task 5 calls exactly these):
  - `GET /api/radar/{map_name}/info` -> `RadarInfo`
  - `GET /api/radar/{map_name}/image?level=default|lower` -> `image/png`
  - `GET /api/radar/{team_key}/{map_name}/layers?side=&level=&rounds=&trail_rounds=&stride=&grid=`
    -> the Task 3 payload

**Status codes (the UI depends on the distinction):**
| Situation | Code |
|---|---|
| ok | 200 |
| `map_name` not `^[a-z0-9_]{1,64}$` | 400 |
| no cached radar **and** `cs2_install_path` unset, or VRF missing | 503 |
| map has no radar art in pak01 | 404 |
| `level=lower` on a single-level map | 404 |
| no ingested match on that map (layers) | 404 |
| unparseable `rounds` | 400 |

- [x] **Step 1: Write the failing tests**

Create `tests/test_radar_api.py`:

```python
import json
from pathlib import Path

import pytest
from conftest import build_radar_lake
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app

# 1x1 transparent PNG - enough to prove the bytes are served untouched.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6300010000050001aa5c9c2d0000000049454e44ae4260" "82"
)
MINIMAL_OVERVIEW = '"de_anubis"\n{\n\t"pos_x" "-1024"\n\t"pos_y" "1024"\n\t"scale" "2"\n}\n'


def _seed_radar_cache(data_root: Path, map_name: str = "de_anubis", *, lower: bool = False) -> None:
    """Pre-populates the radar cache so no CS2 install is needed in tests."""
    cache = data_root / "radar" / map_name
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "radar.png").write_bytes(PNG_BYTES)
    (cache / "overview.txt").write_text(MINIMAL_OVERVIEW, encoding="utf-8")
    (cache / "calibration.json").write_text(
        json.dumps(
            {
                "map_name": map_name, "pos_x": -1024.0, "pos_y": 1024.0,
                "scale": 2.0, "image_px": 1024, "lower_altitude_max": 100.0 if lower else None,
            }
        ),
        encoding="utf-8",
    )
    if lower:
        (cache / "radar_lower.png").write_bytes(PNG_BYTES)


def _seed_corpus(data_root: Path, map_name: str = "de_anubis", match_id: str = "m1") -> None:
    from counterstrat.corpus import DemoRecord

    rec = DemoRecord(
        match_id=match_id, path="/fake/m1.dem", map_name=map_name, patch_version="14178",
        demo_version_guid="guid1", server_name="S", registered_at="2026-09-03T00:00:00Z",
    )
    (data_root / "corpus.jsonl").write_text(rec.model_dump_json() + "\n", encoding="utf-8")


@pytest.fixture
def radar_client(tmp_path: Path) -> TestClient:
    _seed_radar_cache(tmp_path)
    _seed_corpus(tmp_path)
    build_radar_lake(tmp_path / "lake")
    # AppConfig built directly => cs2_install_path is None, so only the cache is used.
    return TestClient(create_app(AppConfig(data_root=tmp_path)))


def test_info_returns_calibration_and_levels(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_anubis/info")
    assert r.status_code == 200
    info = r.json()
    assert info["pos_x"] == -1024.0 and info["scale"] == 2.0
    assert info["image_px"] == 1024
    assert info["levels"] == ["default"]
    assert info["lower_image_url"] is None
    assert info["image_url"] == "/api/radar/de_anubis/image?level=default"


def test_image_serves_png_bytes(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_anubis/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == PNG_BYTES


def test_image_lower_level_missing_is_404(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/de_anubis/image?level=lower").status_code == 404


def test_image_lower_level_present_is_served(tmp_path: Path) -> None:
    _seed_radar_cache(tmp_path, "de_nuke", lower=True)
    client = TestClient(create_app(AppConfig(data_root=tmp_path)))
    r = client.get("/api/radar/de_nuke/image?level=lower")
    assert r.status_code == 200
    assert client.get("/api/radar/de_nuke/info").json()["levels"] == ["default", "lower"]


def test_uncached_map_without_cs2_path_is_503(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/de_mirage/info")
    assert r.status_code == 503
    assert "cs2" in r.json()["detail"].lower()


def test_invalid_map_name_is_400(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/DE_Bad!/info").status_code == 400


def test_layers_full_payload(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&stride=1&trail_rounds=2")
    assert r.status_code == 200
    body = r.json()
    assert body["team_key"] == "teamA"
    assert body["map_name"] == "de_anubis"
    assert body["image_px"] == 1024
    assert [(x["round_num"], x["side"]) for x in body["rounds"]] == [(1, "CT"), (2, "T")]
    layers = body["layers"]
    assert set(layers) == {"heatmap", "trails", "utility", "duels", "bombs"}
    assert layers["heatmap"]["samples"] == 16
    assert len(layers["trails"]) == 4
    assert [u["kind"] for u in layers["utility"]] == ["smoke", "molotov"]
    assert len(layers["duels"]) == 3
    assert [b["event"] for b in layers["bombs"]] == ["defuse", "plant"]
    # steamids must be strings: JS cannot hold a 17-digit steamid exactly.
    assert isinstance(layers["trails"][0]["steamid"], str)


def test_layers_side_and_round_filters(radar_client: TestClient) -> None:
    ct = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&side=CT").json()
    assert [x["round_num"] for x in ct["rounds"]] == [1]
    assert ct["layers"]["heatmap"]["samples"] == 8

    r2 = radar_client.get("/api/radar/teamA/de_anubis/layers?grid=8&rounds=2").json()
    assert [x["round_num"] for x in r2["rounds"]] == [2]


def test_layers_rejects_bad_rounds_and_bad_side(radar_client: TestClient) -> None:
    assert radar_client.get("/api/radar/teamA/de_anubis/layers?rounds=abc").status_code == 400
    assert radar_client.get("/api/radar/teamA/de_anubis/layers?side=X").status_code == 422


def test_layers_unknown_map_is_404(radar_client: TestClient) -> None:
    r = radar_client.get("/api/radar/teamA/de_dust2/layers")
    assert r.status_code == 404


def test_layers_unknown_team_returns_empty_rounds(radar_client: TestClient) -> None:
    body = radar_client.get("/api/radar/ghost/de_anubis/layers").json()
    assert body["rounds"] == []
    assert body["layers"]["trails"] == []
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_api.py -v`
Expected: `ModuleNotFoundError: No module named 'counterstrat.web.radar_api'`.

- [x] **Step 3: Write `src/counterstrat/web/radar_api.py`**

```python
"""Radar image, calibration, and coordinate-layer endpoints (spec item N2)."""

import logging
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from counterstrat.config import AppConfig
from counterstrat.corpus import load_manifest
from counterstrat.radar.extract import RadarAssets, extract_radar_assets, load_cached_assets
from counterstrat.radar.layers import LayerFilters, build_layers, lake_frames
from counterstrat.web.ingest import _find_vrf_cli
from counterstrat.web.routes import ConfigDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/radar")

# map_name lands in filesystem paths, so keep it to the CS2 naming convention.
_MAP_NAME = re.compile(r"^[a-z0-9_]{1,64}$")

Level = Literal["default", "lower"]
LayerLevel = Literal["default", "lower", "all"]


class RadarInfo(BaseModel):
    map_name: str
    pos_x: float
    pos_y: float
    scale: float
    image_px: int
    lower_altitude_max: float | None
    levels: list[str]
    image_url: str
    lower_image_url: str | None


def _check_map_name(map_name: str) -> str:
    if not _MAP_NAME.match(map_name):
        raise HTTPException(status_code=400, detail=f"Invalid map name '{map_name}'")
    return map_name


def _resolve_assets(map_name: str, cfg: AppConfig) -> RadarAssets:
    """Cache-first radar bundle, extracting from the CS2 install on a miss."""
    _check_map_name(map_name)
    cached = load_cached_assets(cfg.data_root, map_name)
    if cached is not None:
        return cached

    if cfg.cs2_install_path is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No cached radar for this map and no CS2 install path is configured. "
                "Set cs2_install_path in Settings to enable radar overlays "
                "(see docs/NEEDS-FROM-YOU.md item N2)."
            ),
        )
    vrf_cli = _find_vrf_cli()
    if vrf_cli is None:
        raise HTTPException(
            status_code=503, detail="Source2Viewer-CLI is not vendored under tools/vrf/"
        )
    try:
        return extract_radar_assets(cfg.cs2_install_path, vrf_cli, map_name, cfg.data_root)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"No radar art for map '{map_name}': {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Radar extraction failed for %s", map_name)
        raise HTTPException(
            status_code=500, detail=f"Radar extraction failed: {exc}"
        ) from exc


def _parse_rounds(rounds: str | None) -> list[int] | None:
    if not rounds:
        return None
    try:
        return [int(part) for part in rounds.split(",") if part.strip()]
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"'rounds' must be comma-separated integers, got {rounds!r}"
        ) from exc


@router.get("/{map_name}/info", response_model=RadarInfo)
def get_radar_info(map_name: str, cfg: ConfigDep) -> RadarInfo:
    """Calibration plus the image URLs the client should load."""
    assets = _resolve_assets(map_name, cfg)
    cal = assets.calibration
    return RadarInfo(
        map_name=cal.map_name,
        pos_x=cal.pos_x,
        pos_y=cal.pos_y,
        scale=cal.scale,
        image_px=cal.image_px,
        lower_altitude_max=cal.lower_altitude_max,
        levels=assets.levels,
        image_url=f"/api/radar/{map_name}/image?level=default",
        lower_image_url=(
            f"/api/radar/{map_name}/image?level=lower" if assets.lower_image else None
        ),
    )


@router.get("/{map_name}/image")
def get_radar_image(map_name: str, cfg: ConfigDep, level: Level = "default") -> FileResponse:
    """The VRF-decompiled overhead PNG, served unmodified."""
    assets = _resolve_assets(map_name, cfg)
    path: Path | None = assets.lower_image if level == "lower" else assets.image
    if path is None or not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Map '{map_name}' has no '{level}' radar image"
        )
    return FileResponse(path, media_type="image/png")


@router.get("/{team_key}/{map_name}/layers")
def get_radar_layers(
    team_key: str,
    map_name: str,
    cfg: ConfigDep,
    side: Literal["T", "CT"] | None = None,
    level: LayerLevel = "all",
    rounds: str | None = None,
    trail_rounds: Annotated[int | None, Query(ge=1, le=30)] = 4,
    stride: Annotated[int, Query(ge=1, le=64)] = 8,
    grid: Annotated[int, Query(ge=8, le=256)] = 128,
) -> dict[str, Any]:
    """Normalized coordinate layers for one (team, map) selection."""
    # Resolve the corpus BEFORE the radar assets: a map with no ingested match
    # must answer 404, not the 503 that an un-extractable radar would raise.
    _check_map_name(map_name)
    manifest = load_manifest(cfg.data_root / "corpus.jsonl")
    match_ids = sorted(mid for mid, rec in manifest.items() if rec.map_name == map_name)
    if not match_ids:
        raise HTTPException(
            status_code=404, detail=f"No ingested matches on map '{map_name}'"
        )

    cal = _resolve_assets(map_name, cfg).calibration
    filters = LayerFilters(
        side=side,
        level=level,
        round_nums=_parse_rounds(rounds),
        trail_rounds=trail_rounds,
        stride=stride,
        grid=grid,
    )
    frames = lake_frames(cfg.data_root / "lake", match_ids)
    return build_layers(frames, cal, team_key, filters)
```

- [x] **Step 4: Wire the router into `src/counterstrat/web/app.py`**

Add the import next to the existing router imports (after
`from counterstrat.web.chat import router as chat_router`, line 11):
```python
from counterstrat.web.radar_api import router as radar_router
```
Add the include after `app.include_router(chat_router)` (line 26):
```python
    app.include_router(radar_router)
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_radar_api.py -v`
Expected: 11 passed.

If `test_layers_rejects_bad_rounds_and_bad_side` sees 422 instead of 400 for
`rounds=abc`, FastAPI is coercing the type before the handler — keep `rounds`
annotated as `str | None` (not `list[int]`) so `_parse_rounds` owns the error.

- [x] **Step 6: Lint and commit**

```bash
.venv\Scripts\python.exe -m ruff check .
git add src/counterstrat/web/radar_api.py src/counterstrat/web/app.py tests/test_radar_api.py
git commit -m "feat(radar): add radar info, image and layer API endpoints"
```

**Done when:**
- `.venv\Scripts\python.exe -m pytest tests/test_radar_api.py -v` reports
  11 passed, 0 failed.
- `.venv\Scripts\python.exe -m pytest -q` reports 0 failures overall.
- `.venv\Scripts\python.exe -m ruff check .` prints `All checks passed!`.
- Manual smoke against the real install (optional but recommended): start
  `.venv\Scripts\python.exe -m counterstrat.web.app` and confirm
  `http://127.0.0.1:8710/api/radar/de_anubis/info` returns
  `"pos_x": -2796.0, "scale": 5.22, "levels": ["default"]`, and
  `http://127.0.0.1:8710/api/radar/de_anubis/image` renders the Anubis radar.

---

## Task 5: Frontend radar viewer

**Goal:** A Radar tab in the analysis pane that renders the selected team's
layers on a canvas stacked over the radar PNG, with per-layer toggles and
side/round/level filters.

**Difficulty:** MEDIUM — **Route:** EASY
**Verify:** full

**Files:**
- Create: `src/counterstrat/web/static/radar.js`
- Modify: `src/counterstrat/web/static/index.html` (tabs in `.chat-header`
  at `:67-77`; new section after `#messages-container` closes at `:92`; id on
  `.chat-input-bar` at `:94`; `<script>` at `:165`)
- Modify: `src/counterstrat/web/static/app.js` (hook inside `selectTarget`,
  after `:317`)
- Modify: `src/counterstrat/web/static/style.css` (append a radar section)
- Create: `tests/test_web_radar_static.py`

**Interfaces:**
- Consumes: the Task 4 endpoints and the Task 3 payload shape verbatim.
- Produces: `window.CounterStratRadar.onTargetSelected(teamKey, mapName, displayName)`
  — the single entry point `app.js` calls. Nothing else is global.

**Design notes (follow, do not re-derive):**
- The canvas backing store is fixed at 1024x1024 and CSS-scaled, so drawing is
  `x = u * 1024` with no resize math. `<img>` and `<canvas>` are stacked with
  `position: absolute; inset: 0` inside a square `.radar-frame`.
- Layer checkboxes trigger a **redraw only**; side/round/trail/level changes
  trigger a **refetch** (they change the server-side filter).
- Reuse the existing CSS tokens (`--accent-blue #58a6ff`,
  `--accent-orange #f0883e`, `--bg-*`, `--border-color`, `--radius-*`) — do not
  introduce a new palette. `.hidden { display: none !important }` already exists
  at `style.css:917-919`.
- `app.js` is a closed IIFE; do not restructure it. The hook is the only change.

- [x] **Step 1: Write the failing tests**

Create `tests/test_web_radar_static.py`:

```python
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from counterstrat.config import AppConfig
from counterstrat.web.app import create_app


@pytest.fixture
def client_app(tmp_path: Path) -> TestClient:
    return TestClient(create_app(AppConfig(data_root=tmp_path)))


def test_index_exposes_radar_view(client_app: TestClient) -> None:
    html = client_app.get("/").text
    for token in [
        "radar.js",
        'id="tab-chat"',
        'id="tab-radar"',
        'id="radar-view"',
        'id="radar-image"',
        'id="radar-canvas"',
        'id="radar-status"',
        'id="radar-side"',
        'id="radar-round"',
        'id="radar-trail-rounds"',
        'id="radar-level-toggle"',
        'id="chat-input-bar"',
    ]:
        assert token in html, f"missing {token}"
    for layer in ["heatmap", "trails", "utility", "duels", "bombs"]:
        assert f'id="layer-{layer}"' in html


def test_radar_js_targets_the_radar_endpoints(client_app: TestClient) -> None:
    js = client_app.get("/static/radar.js").text
    assert js.startswith("/**")
    assert "/api/radar/" in js
    assert "/info" in js and "/layers" in js
    assert "window.CounterStratRadar" in js
    for fn in ["drawHeatmap", "drawTrails", "drawUtility", "drawDuels", "drawBombs"]:
        assert fn in js, f"missing renderer {fn}"


def test_app_js_notifies_the_radar_viewer(client_app: TestClient) -> None:
    js = client_app.get("/static/app.js").text
    assert "CounterStratRadar" in js
    assert "onTargetSelected" in js


def test_style_css_has_radar_rules(client_app: TestClient) -> None:
    css = client_app.get("/static/style.css").text
    for rule in [".radar-view", ".radar-frame", ".radar-canvas", ".view-tab", ".radar-legend"]:
        assert rule in css, f"missing rule {rule}"
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_radar_static.py -v`
Expected: 4 failed — `missing radar.js`, 404 on `/static/radar.js`, etc.

- [x] **Step 3: Create `src/counterstrat/web/static/radar.js`**

```js
/**
 * Radar overlay viewer.
 * Draws the normalized [0,1] layer coordinates served by /api/radar/... onto a
 * 1024x1024 canvas stacked over the CS2 overhead radar PNG.
 */

(function () {
  "use strict";

  const CANVAS_PX = 1024;

  const SIDE_COLORS = { T: "#f0883e", CT: "#58a6ff" };
  const UTIL_COLORS = {
    smoke: "#c9d1d9",
    flash: "#e3b341",
    he: "#f85149",
    molotov: "#db6d28",
    decoy: "#8b949e",
  };
  const KILL_FOR = "#3fb950";
  const KILL_AGAINST = "#f85149";
  const NEUTRAL = "#8b949e";

  const state = {
    teamKey: null,
    mapName: null,
    displayName: null,
    info: null,
    payload: null,
    loading: false,
    level: "default",
    visible: { heatmap: true, trails: true, utility: true, duels: true, bombs: true },
  };

  const el = {};

  function cacheElements() {
    el.tabChat = document.getElementById("tab-chat");
    el.tabRadar = document.getElementById("tab-radar");
    el.radarView = document.getElementById("radar-view");
    el.messages = document.getElementById("messages-container");
    el.inputBar = document.getElementById("chat-input-bar");
    el.image = document.getElementById("radar-image");
    el.canvas = document.getElementById("radar-canvas");
    el.status = document.getElementById("radar-status");
    el.side = document.getElementById("radar-side");
    el.round = document.getElementById("radar-round");
    el.trailRounds = document.getElementById("radar-trail-rounds");
    el.levelToggle = document.getElementById("radar-level-toggle");
    el.levelDefault = document.getElementById("radar-level-default");
    el.levelLower = document.getElementById("radar-level-lower");
    el.refresh = document.getElementById("radar-refresh");
    el.checks = {
      heatmap: document.getElementById("layer-heatmap"),
      trails: document.getElementById("layer-trails"),
      utility: document.getElementById("layer-utility"),
      duels: document.getElementById("layer-duels"),
      bombs: document.getElementById("layer-bombs"),
    };
  }

  function setStatus(text, kind) {
    el.status.textContent = text;
    el.status.className = kind === "error" ? "radar-status is-error" : "radar-status";
  }

  function readJson(res) {
    if (!res.ok) {
      return res.json().then(
        function (body) {
          throw new Error((body && body.detail) || `Server returned ${res.status}`);
        },
        function () {
          throw new Error(`Server returned ${res.status}`);
        }
      );
    }
    return res.json();
  }

  // ---------------------------------------------------------------- fetching

  function layersUrl() {
    const params = new URLSearchParams();
    if (el.side.value) params.set("side", el.side.value);
    if (el.round.value) params.set("rounds", el.round.value);
    params.set("trail_rounds", el.trailRounds.value);
    const multiLevel = state.info && state.info.levels.length > 1;
    params.set("level", multiLevel ? state.level : "all");
    return (
      `/api/radar/${encodeURIComponent(state.teamKey)}` +
      `/${encodeURIComponent(state.mapName)}/layers?${params.toString()}`
    );
  }

  function load() {
    if (!state.teamKey || !state.mapName || state.loading) return;
    state.loading = true;
    setStatus("Loading radar...", null);

    fetch(`/api/radar/${encodeURIComponent(state.mapName)}/info`)
      .then(readJson)
      .then(function (info) {
        state.info = info;
        el.levelToggle.classList.toggle("hidden", info.levels.length < 2);
        const useLower = state.level === "lower" && info.lower_image_url;
        el.image.src = useLower ? info.lower_image_url : info.image_url;
        return fetch(layersUrl()).then(readJson);
      })
      .then(function (payload) {
        state.payload = payload;
        populateRounds(payload.rounds);
        setStatus(summarize(payload), null);
        draw();
      })
      .catch(function (err) {
        state.payload = null;
        clearCanvas();
        setStatus(err.message, "error");
      })
      .finally(function () {
        state.loading = false;
      });
  }

  function populateRounds(rounds) {
    const previous = el.round.value;
    el.round.innerHTML = '<option value="">All rounds</option>';
    (rounds || []).forEach(function (r) {
      const opt = document.createElement("option");
      opt.value = String(r.round_num);
      opt.textContent = `Round ${r.round_num} (${r.side})`;
      el.round.appendChild(opt);
    });
    el.round.value = previous;
  }

  function summarize(payload) {
    const L = payload.layers;
    return (
      `${payload.rounds.length} rounds | ` +
      `${L.heatmap.samples} position samples | ` +
      `${L.trails.length} trails | ` +
      `${L.utility.length} nades | ` +
      `${L.duels.length} duels | ` +
      `${L.bombs.length} bomb events`
    );
  }

  // ---------------------------------------------------------------- drawing

  function px(n) {
    return n * CANVAS_PX;
  }

  function clearCanvas() {
    el.canvas.getContext("2d").clearRect(0, 0, CANVAS_PX, CANVAS_PX);
  }

  function dot(ctx, x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function cross(ctx, x, y, size, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x - size, y - size);
    ctx.lineTo(x + size, y + size);
    ctx.moveTo(x + size, y - size);
    ctx.lineTo(x - size, y + size);
    ctx.stroke();
  }

  function heatColor(t) {
    // cold blue (#3c78f0-ish) -> warm orange, matching the app accents
    const r = Math.round(60 + 195 * t);
    const g = Math.round(120 + 16 * t);
    const b = Math.round(240 - 178 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function drawHeatmap(ctx, hm) {
    if (!hm || !hm.cells.length || !hm.max) return;
    const size = CANVAS_PX / hm.grid;
    hm.cells.forEach(function (cell) {
      const t = cell[2] / hm.max;
      ctx.fillStyle = heatColor(t);
      ctx.globalAlpha = 0.2 + 0.6 * t;
      ctx.fillRect(cell[0] * size, cell[1] * size, size, size);
    });
    ctx.globalAlpha = 1;
  }

  function drawTrails(ctx, trails) {
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    (trails || []).forEach(function (trail) {
      if (!trail.points || !trail.points.length) return;
      const color = SIDE_COLORS[trail.side] || NEUTRAL;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      trail.points.forEach(function (p, i) {
        if (i === 0) ctx.moveTo(px(p[0]), px(p[1]));
        else ctx.lineTo(px(p[0]), px(p[1]));
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
      const first = trail.points[0];
      const last = trail.points[trail.points.length - 1];
      dot(ctx, px(first[0]), px(first[1]), 3, color);
      dot(ctx, px(last[0]), px(last[1]), 5, color);
    });
  }

  function drawUtility(ctx, util) {
    (util || []).forEach(function (u) {
      if (u.u === null || u.v === null) return;
      const color = UTIL_COLORS[u.kind] || NEUTRAL;
      if (u.from_u !== null && u.from_v !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(px(u.from_u), px(u.from_v));
        ctx.lineTo(px(u.u), px(u.v));
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px(u.u), px(u.v), 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;
    });
  }

  function drawDuels(ctx, duels) {
    (duels || []).forEach(function (d) {
      if (!d.victim || d.victim.u === null) return;
      const color = d.by_team ? KILL_FOR : KILL_AGAINST;
      if (d.attacker && d.attacker.u !== null) {
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px(d.attacker.u), px(d.attacker.v));
        ctx.lineTo(px(d.victim.u), px(d.victim.v));
        ctx.stroke();
        ctx.globalAlpha = 1;
        dot(ctx, px(d.attacker.u), px(d.attacker.v), 3, color);
      }
      cross(ctx, px(d.victim.u), px(d.victim.v), 6, color);
    });
  }

  function drawBombs(ctx, bombs) {
    (bombs || []).forEach(function (b) {
      if (b.u === null || b.v === null) return;
      const color = b.event === "plant" ? "#f0883e" : "#58a6ff";
      const size = 12;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(px(b.u) - size / 2, px(b.v) - size / 2, size, size);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#0d1117";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px(b.u) - size / 2, px(b.v) - size / 2, size, size);
    });
  }

  function draw() {
    const ctx = el.canvas.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);
    if (!state.payload) return;
    const L = state.payload.layers;
    if (state.visible.heatmap) drawHeatmap(ctx, L.heatmap);
    if (state.visible.trails) drawTrails(ctx, L.trails);
    if (state.visible.utility) drawUtility(ctx, L.utility);
    if (state.visible.duels) drawDuels(ctx, L.duels);
    if (state.visible.bombs) drawBombs(ctx, L.bombs);
  }

  // ---------------------------------------------------------------- wiring

  function showView(view) {
    const radar = view === "radar";
    el.radarView.classList.toggle("hidden", !radar);
    el.messages.classList.toggle("hidden", radar);
    el.inputBar.classList.toggle("hidden", radar);
    el.tabChat.classList.toggle("active", !radar);
    el.tabRadar.classList.toggle("active", radar);
    if (radar && !state.payload) load();
  }

  function setLevel(level) {
    if (state.level === level) return;
    state.level = level;
    el.levelDefault.classList.toggle("active", level === "default");
    el.levelLower.classList.toggle("active", level === "lower");
    load();
  }

  function onTargetSelected(teamKey, mapName, displayName) {
    state.teamKey = teamKey;
    state.mapName = mapName;
    state.displayName = displayName;
    state.payload = null;
    state.info = null;
    state.level = "default";
    el.tabRadar.disabled = false;
    el.tabRadar.title = `Radar overlay for ${displayName} on ${mapName}`;
    el.round.value = "";
    clearCanvas();
    if (!el.radarView.classList.contains("hidden")) load();
    else setStatus("Open the Radar tab to plot this selection.", null);
  }

  function initEvents() {
    el.tabChat.addEventListener("click", function () {
      showView("chat");
    });
    el.tabRadar.addEventListener("click", function () {
      showView("radar");
    });
    Object.keys(el.checks).forEach(function (key) {
      el.checks[key].addEventListener("change", function () {
        state.visible[key] = el.checks[key].checked;
        draw();
      });
    });
    [el.side, el.round, el.trailRounds].forEach(function (control) {
      control.addEventListener("change", load);
    });
    el.levelDefault.addEventListener("click", function () {
      setLevel("default");
    });
    el.levelLower.addEventListener("click", function () {
      setLevel("lower");
    });
    el.refresh.addEventListener("click", load);
  }

  window.CounterStratRadar = {
    onTargetSelected: function (teamKey, mapName, displayName) {
      if (!el.radarView) return;
      onTargetSelected(teamKey, mapName, displayName);
    },
  };

  document.addEventListener("DOMContentLoaded", function () {
    cacheElements();
    if (!el.radarView) return;
    initEvents();
  });
})();
```

- [x] **Step 4: Edit `src/counterstrat/web/static/index.html`**

4a. Inside `.chat-header`, between the `</div>` that closes `.target-info`
(line 72) and `<div class="chat-header-actions">` (line 72), insert:

```html
          <div class="view-tabs" role="tablist">
            <button type="button" id="tab-chat" class="view-tab active" role="tab">Chat</button>
            <button type="button" id="tab-radar" class="view-tab" role="tab" disabled
                    title="Select a team and map first">Radar</button>
          </div>
```

4b. Immediately after the `</div>` that closes `#messages-container`
(line 92), insert the radar section:

```html
        <section id="radar-view" class="radar-view hidden" aria-label="Radar overlay">
          <div class="radar-controls">
            <div class="radar-control-group">
              <span class="radar-control-label">Layers</span>
              <label class="radar-check"><input type="checkbox" id="layer-heatmap" checked> Heatmap</label>
              <label class="radar-check"><input type="checkbox" id="layer-trails" checked> Trails</label>
              <label class="radar-check"><input type="checkbox" id="layer-utility" checked> Utility</label>
              <label class="radar-check"><input type="checkbox" id="layer-duels" checked> Duels</label>
              <label class="radar-check"><input type="checkbox" id="layer-bombs" checked> Bomb</label>
            </div>
            <div class="radar-control-group">
              <label class="radar-control-label" for="radar-side">Side</label>
              <select id="radar-side" class="form-control form-control-sm">
                <option value="">Both</option>
                <option value="T">T</option>
                <option value="CT">CT</option>
              </select>
              <label class="radar-control-label" for="radar-round">Round</label>
              <select id="radar-round" class="form-control form-control-sm">
                <option value="">All rounds</option>
              </select>
              <label class="radar-control-label" for="radar-trail-rounds">Trails</label>
              <select id="radar-trail-rounds" class="form-control form-control-sm">
                <option value="1">1 round</option>
                <option value="2">2 rounds</option>
                <option value="4" selected>4 rounds</option>
                <option value="8">8 rounds</option>
              </select>
              <div id="radar-level-toggle" class="radar-level-toggle hidden">
                <button type="button" id="radar-level-default" class="btn btn-sm btn-outline active">Upper</button>
                <button type="button" id="radar-level-lower" class="btn btn-sm btn-outline">Lower</button>
              </div>
              <button type="button" id="radar-refresh" class="btn btn-sm btn-outline">Reload</button>
            </div>
          </div>

          <div class="radar-stage">
            <div class="radar-frame">
              <img id="radar-image" class="radar-image" alt="Overhead radar">
              <canvas id="radar-canvas" class="radar-canvas" width="1024" height="1024"></canvas>
            </div>
          </div>

          <div class="radar-footer">
            <div id="radar-status" class="radar-status">
              Select a team and map, then open the Radar tab.
            </div>
            <ul class="radar-legend">
              <li><span class="swatch swatch-ct"></span>CT</li>
              <li><span class="swatch swatch-t"></span>T</li>
              <li><span class="swatch swatch-kill"></span>Kill by team</li>
              <li><span class="swatch swatch-death"></span>Kill against</li>
              <li><span class="swatch swatch-smoke"></span>Smoke</li>
              <li><span class="swatch swatch-fire"></span>Molotov</li>
              <li><span class="swatch swatch-bomb"></span>Plant</li>
            </ul>
          </div>
        </section>
```

4c. Give the input bar an id (line 94):
`<div class="chat-input-bar">` becomes
`<div class="chat-input-bar" id="chat-input-bar">`.

4d. Add the script tag after `<script src="/static/app.js"></script>` (line 165):
```html
  <script src="/static/radar.js"></script>
```

- [x] **Step 5: Add the hook to `src/counterstrat/web/static/app.js`**

In `selectTarget`, immediately after
`el.dossierBtn.title = "Download strategic dossier markdown report";` (line 317)
and before the `// Create session` comment, insert:

```js
    // Hand the selection to the radar viewer (static/radar.js), if present.
    if (window.CounterStratRadar) {
      window.CounterStratRadar.onTargetSelected(team.team_key, mapName, displayName);
    }
```

- [x] **Step 6: Append the radar styles to `src/counterstrat/web/static/style.css`**

Append at the end of the file (after the `.hidden` utility block):

```css
/* ==========================================================================
   Radar Overlay View
   ========================================================================== */

.view-tabs {
  display: flex;
  gap: 4px;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  padding: 3px;
}

.view-tab {
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  padding: 6px 14px;
}

.view-tab:hover:not(:disabled) {
  color: var(--text-primary);
}

.view-tab.active {
  background-color: var(--bg-tertiary);
  color: var(--accent-blue);
}

.view-tab:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

.radar-view {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 12px;
  min-height: 0;
  overflow-y: auto;
  padding: 16px 20px;
}

.radar-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 20px;
  justify-content: space-between;
}

.radar-control-group {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.radar-control-label {
  color: var(--text-muted);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.radar-check {
  align-items: center;
  color: var(--text-secondary);
  cursor: pointer;
  display: flex;
  font-size: 12px;
  gap: 5px;
}

.form-control-sm {
  font-size: 12px;
  padding: 4px 8px;
}

.radar-level-toggle {
  display: flex;
  gap: 4px;
}

.radar-level-toggle .btn.active {
  border-color: var(--accent-blue);
  color: var(--accent-blue);
}

.radar-stage {
  display: flex;
  justify-content: center;
}

.radar-frame {
  aspect-ratio: 1 / 1;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  overflow: hidden;
  position: relative;
  width: min(640px, 100%);
}

.radar-image,
.radar-canvas {
  height: 100%;
  inset: 0;
  position: absolute;
  width: 100%;
}

.radar-image {
  object-fit: contain;
}

.radar-canvas {
  pointer-events: none;
}

.radar-footer {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.radar-status {
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: 11px;
}

.radar-status.is-error {
  color: var(--accent-orange);
}

.radar-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  list-style: none;
  margin: 0;
  padding: 0;
}

.radar-legend li {
  align-items: center;
  color: var(--text-muted);
  display: flex;
  font-size: 11px;
  gap: 6px;
}

.radar-legend .swatch {
  border-radius: 2px;
  display: inline-block;
  height: 10px;
  width: 10px;
}

.swatch-ct { background-color: #58a6ff; }
.swatch-t { background-color: #f0883e; }
.swatch-kill { background-color: #3fb950; }
.swatch-death { background-color: #f85149; }
.swatch-smoke { background-color: #c9d1d9; }
.swatch-fire { background-color: #db6d28; }
.swatch-bomb { background-color: #f0883e; border: 1px solid #0d1117; }
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_web_radar_static.py tests/test_web_static.py -v`
Expected: all PASS (the pre-existing `test_index_page` and `test_static_app_js`
must still pass — the markup edits are additive).

- [x] **Step 8: Manual browser check (recommended, not an acceptance gate)**

Start the app, upload/select an already-ingested target, open the Radar tab:
```bash
.venv\Scripts\python.exe -m counterstrat.web.app
```
Then visit `http://127.0.0.1:8710/`, pick a team+map card, click **Radar**. The
Anubis radar image should appear with a heat overlay concentrated around mid and
the sites, trail polylines in orange/blue, dashed nade arcs, and orange plant
squares near A/B. Toggling a checkbox must redraw instantly with no network
request (check the Network tab). For an automated pass use the
`webapp-testing` skill.

- [x] **Step 9: Lint and commit**

```bash
.venv\Scripts\python.exe -m ruff check .
git add src/counterstrat/web/static tests/test_web_radar_static.py
git commit -m "feat(radar): interactive canvas radar viewer with toggleable layers"
```

**Done when:**
- `.venv\Scripts\python.exe -m pytest -q` reports 0 failures.
- `.venv\Scripts\python.exe -m ruff check .` prints `All checks passed!`.
- `GET /static/radar.js` returns 200 and `GET /` contains `id="radar-canvas"`.
- The Radar tab button is disabled until a team+map card is clicked, and
  enabled after.

---

## Task 6: Record N2 as satisfied

**Goal:** Update the needs-from-user register so item N2 no longer reads as
deferred.

**Difficulty:** EASY — **Route:** EASY
**Verify:** lite (docs-only diff)

**Files:**
- Modify: `docs/NEEDS-FROM-YOU.md` (row at `:10`, section at `:42-58`)

**Test first:** none — this task changes prose only, so its check is the diff.
Do not add a test framework or a test for a markdown edit.

**Change:**

1. Replace the N2 table row (`docs/NEEDS-FROM-YOU.md:10`) with:
```markdown
| **N2** | Radar images | **SATISFIED** (Closed) | CS2 install path supplied; radar art and overview calibration are extracted from `pak01_dir.vpk` on demand by `counterstrat.radar` and cached under `data/radar/<map>/`. Interactive radar overlay live in the web UI. |
```

2. Replace the body of the `## N2 — Radar images` section
(`docs/NEEDS-FROM-YOU.md:42-58`) with:

```markdown
## N2 — Radar images (satisfied)

**Status: SATISFIED**

CS2 install path: `D:\Data\Gaming\Steam\steamapps\common\Counter-Strike Global Offensive`
(stored in `data/settings.json` as `cs2_install_path`; also settable via
`.env` `CS2_INSTALL_PATH` or the Settings modal).

Verified asset locations inside `<install>\game\csgo\pak01_dir.vpk`:

- `panorama/images/overheadmaps/<map>_radar_psd.vtex_c` — the overhead art;
  decompiles to a 1024x1024 PNG with the VRF CLI's `-d` flag. Multi-level maps
  (nuke, vertigo, train) also ship `<map>_lower_radar_psd.vtex_c`.
- `resource/overviews/<map>.txt` — KeyValues calibration: `pos_x`, `pos_y`,
  `scale`, and for multi-level maps a `verticalsections."lower".AltitudeMax`
  altitude split.

Projection: `u = (X - pos_x) / scale`, `v = (pos_y - Y) / scale`, in pixels of
the 1024 px image.

Do **not** copy `pak01_dir.vpk` into the repo: it is only the directory index of
a split archive whose content lives in the `pak01_NNN.vpk` chunks beside it
(tens of GB). `counterstrat.radar.extract` reads the game files in place through
the vendored VRF CLI and caches the two small artefacts per map under
`data/radar/<map>/`.

awpy's hosted `map-data.json` is still unavailable, so
`awpy.plot.utils.game_to_pixel_axis` cannot be used — `counterstrat.radar.coords`
implements the transform against the calibration extracted above.
```

**Done when:**
- `docs/NEEDS-FROM-YOU.md` no longer contains the strings `OPTIONAL` or
  `DEFERRED` in the N2 row or section.
- `git diff --stat` shows `docs/NEEDS-FROM-YOU.md` as the only changed file for
  this task.
- Commit: `git commit -m "docs: close needs-from-user item N2 (radar images)"`

---

## Risks

- **Radar art moves on a CS2 update.** If `panorama/images/overheadmaps/` is
  renamed, Task 1's extraction returns no PNG and `_resolve_assets` answers 404
  with the map name in the detail. Recovery: re-run
  `tools\vrf\Source2Viewer-CLI.exe -i "<install>\game\csgo\pak01_dir.vpk" --vpk_dir`
  and grep the listing for `overheadmaps` to find the new prefix, then update
  the two filter strings in `extract_radar_assets`. No data loss — the cache
  under `data/radar/` keeps serving the last good extraction.
- **A stale cache survives a map art update.** `get_radar_assets(..., force=True)`
  re-extracts; the simplest manual fix is deleting `data/radar/<map>/`. The API
  deliberately does not expose `force` (no cache-busting endpoint was asked for).
- **`pl.int_range(pl.len()).over(...)` ordering** is the one polars idiom in
  Task 3 that could behave differently on a future version. The fallback
  (`pl.cum_count("tick").over(...) - 1`) is in the task's debugging notes, and
  `test_trails_stride_decimates_and_trail_rounds_caps` catches a regression.
- **Payload size on a large corpus.** Heatmap cells are bounded by
  `grid**2` (16 384 at the 128 default) and trails by `trail_rounds`, but
  `duels` and `utility` grow linearly with ingested rounds — roughly 220 duels
  and 300 nades per match. At ~20 matches that is a few hundred KB of JSON,
  which is acceptable; if it ever isn't, add a `matches=` filter to the layers
  endpoint rather than paginating.
- **`grenades` is ~2.5 M rows per match** and `utility_layer` scans it. Polars
  pushes the `grenade_type` filter into the parquet read, so this stays
  sub-second per match, but if the corpus grows past ~50 matches consider
  materialising a per-match `utility_landings.parquet` during ingest.
- **Multi-level maps are untested against real data** — no nuke/vertigo/train
  demo is ingested in this repo. Task 1's `demo`-marked
  `test_extract_radar_assets_multi_level` proves the *calibration* side against
  the real VPK; the *layer* side is proven only by
  `test_level_filter_uses_lower_altitude_max` on synthetic data. Flag any real
  nuke demo as a follow-up validation, not a blocker.
- **`AppConfig.load()` in the `cs2_install` test fixture is CWD-relative**
  (it reads `data/settings.json`). Running pytest from outside the repo root
  makes those tests skip rather than fail — acceptable, but do not "fix" a skip
  by hardcoding the install path.
- **Rollback:** every task is additive except the four edits in Task 5
  (`index.html`, `app.js`, `style.css`) and the two in Task 4 (`app.py`).
  Reverting the Task 4 commit removes the router and the whole feature goes
  dark with the rest of the app unaffected; nothing in this plan deletes or
  rewrites existing data.

PLAN COMPLETE
