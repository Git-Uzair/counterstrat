# CS2 Demo → LLM Anti-Strat: Verified Implementation Plan

> **For agentic workers:** execute with subagent-driven-development or
> executing-plans, task by task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the end-to-end product described in
`docs/plans/cs2-demo-llm-anti-strat.md` (the spec) as a **local web app**:
the user uploads CS2 demos, the app parses them into a parquet lake,
compiles per-map Map Cards from map VPKs, serializes rounds into the
RoundScript v2 grammar, mines team tendency profiles deterministically —
and then the user **chats** with an analyst agent ("what does team X run on
full buys?", "how do we anti-strat their B hits?", "what's their default CT
setup?") backed by Anthropic **or** Google Gemini (API-key switchable,
1M-token context on both), with tool-grounded, evidence-linked answers and
a downloadable scouting dossier.

**Architecture:** Python 3.13 monorepo (`counterstrat` package, src layout).
Six layers with hard boundaries: (L0) extraction lake — demoparser2/awpy →
parquet + DuckDB; (Map) Map Card compiler — VRF CLI → env_cs_place volumes +
nav v36 graph → versioned YAML atlas + `(x,y,z)→zone` mapper; (L1)
RoundScript serializer — deterministic demo→text grammar over the closed
zone lexicon; (L2) tendency miner — pure-code conditional frequencies with
evidence round-ids; (L3) LLM layer — provider-abstracted chat agent with
function tools (TeamBook lookup, round-script fetch, guarded SQL) plus
dossier generation, all behind anti-hallucination linting; (Web) FastAPI
backend (upload → background ingest jobs → chat sessions) + a no-build
static frontend. The LLM never counts, never pathfinds, and may only use
lexicon zone ids.

**Tech stack:** Python 3.13.14, uv, demoparser2 0.42.0, awpy 2.0.2, polars,
duckdb, pyarrow, pydantic v2, PyYAML, scikit-learn (DBSCAN only), FastAPI +
uvicorn (+ python-multipart, zstandard for FACEIT `.dem.zst` uploads),
pytest; `anthropic` and `google-genai` SDKs; Source2Viewer-CLI 20.0 (VRF,
vendored binary) for VPK/nav extraction. Frontend: vanilla HTML/JS/CSS
served by FastAPI — no node toolchain.

**Spec:** `docs/plans/cs2-demo-llm-anti-strat.md` (research report, rev. 2).
This plan implements its Phases 0–5 and converts its §6.7 unknowns into
tasks with measured exit gates. Phase 6 (productization) is out of scope
here except where noted.

---

## Context: verified facts (measured in this repo, 2026-09-03)

Everything below was empirically verified in this session on this machine.
Scratch evidence lives in `scratch/` (kept; regenerate with the listed
commands). Do not re-derive these; build on them.

### Environment
- Windows, Python 3.13.14, pip 26.1.2, uv 0.11.28, git 2.55.0. **No .NET
  SDK** — irrelevant, VRF CLI is self-contained.
- Network works for pip/uv installs, GitHub downloads, and API calls.
  `Invoke-WebRequest`/`Remove-Item` PowerShell cmdlets are policy-denied in
  the agent environment — use Python (`urllib.request`) for downloads and
  avoid delete-based cleanup in scripted steps.

### The provided demo (`demos/1-7065ab7c-...-1-1.dem`)
- FACEIT GOTV recording, `PBDEMS2`, `map_name=de_anubis`,
  `patch_version=14178`, `demo_version_guid=8e9d71ab-04a1-4c01-bb61-acfede27c046`
  (from `parse_header()`; see `scratch/inspect_demo.py`).
- Full demoparser2 pull (events list, round_ends, 2s-sampled ticks, kills
  with attached props, grenade trajectories, econ props): **6.0 s total**.
  awpy `Demo(...).parse()`: **4.5 s**. Spec's "seconds per demo" claim holds.
- awpy `rounds` table: **30 gameplay rounds** (12-12 regulation + OT),
  columns `round_num,start,freeze_end,end,official_end,winner,reason,
  bomb_plant,bomb_site`. Raw `round_end` events: 32 rows incl. warmup/stub —
  awpy's filtering works and must be used.
- 52 event types present. `list_game_events()` **under-reports**:
  `item_purchase` parses fine (2,298 rows) despite not being listed. Always
  probe `parse_event` directly.
- All 29 tick props needed by the serializer parse OK on this demo:
  `X,Y,Z,last_place_name,health,armor_value,has_helmet,has_defuser,is_alive,
  active_weapon_name,inventory,balance,cash_spent_this_round,
  current_equip_value,round_start_equip_value,is_walking,flash_duration,
  is_scoped,is_defusing,in_bomb_zone,team_name,team_clan_name,
  is_bomb_planted,total_rounds_played,ct_losing_streak,t_losing_streak,
  velocity,pitch,yaw` (see `scratch/inspect_props.py`).
- Table shapes (awpy): kills 221×47 (attacker/victim/assister `_place`
  columns included), damages 805×27, shots 5078×14, grenades **2,555,348**
  trajectory rows, smokes 173×15 (start/end ticks), infernos 131×15, bomb
  131×9, ticks **1,999,350×10** at 64 Hz.
- `dem.footsteps` raises `KeyError: player_sound` on this demo — optional
  awpy tables must be guarded with try/except.
- Teams are FACEIT PUG lobby names (`team_Budzik11`, `team_7yrant`) —
  **team identity must be keyed on roster steamid sets, never clan-name
  strings.**

### Map VPKs (provided in `maps/`)
- `de_anubis.vpk` contains `maps/de_anubis/entities/default_ents.vents_c`
  (41,759 B compiled) and `maps/de_anubis.nav` (495,530 B).
  `de_ancient.vpk` likewise (`default_ents.vents_c` 63,633 B,
  `de_ancient.nav` 371,251 B). The `*_vanity.vpk` files contain **no**
  radar/overview images — radar art is not in these VPKs.
- Source2Viewer-CLI 20.0 (sha256
  `d32ab327b8bbb42a2528866afb03bb582bdb779d0005488da32b90292afd3ff5`,
  vendored at `tools/vrf/Source2Viewer-CLI.exe`) decompiles the entity lump
  to text. Each `env_cs_place` block carries `place_name`, `classname`,
  `origin [x,y,z]`, `angles`, `scales`, `hammeruniqueid`, and a `model`
  resource (`maps/de_anubis/entities/unnamed_*.vmdl`) holding the volume
  geometry. Extraction command that worked (note: `-e` extension filter;
  single-file `-f`+`-o` mode is buggy — writes an empty dir entry):
  `tools\vrf\Source2Viewer-CLI.exe -i maps\de_anubis\de_anubis.vpk -e "vents_c" -d -o <outdir>`
  and `... -e "nav" -o <outdir>` for the raw nav.
- **Place reconciliation (spec §6.7-1) is settled for Anubis:** the VPK has
  **62 env_cs_place volumes → 28 unique place names**, and the demo-observed
  `last_place_name` set equals it **exactly** (0 mismatches both ways). See
  `scratch/places_vpk.json`, `scratch/places_demo.json`,
  `scratch/reconcile_places.py`. The 28 names: Alley, BackofB, BombsiteA,
  BombsiteB, Bricks, Bridge, CTSideUpper, CTSpawn, Canal, Connector,
  Fountain, Heaven, LowerTunnel, Main, MidDoors, Middle, OutsideLong,
  PalaceInterior, Ruins, SnipersNest, Street, TSideUpper, TSpawn, TStairs,
  Tunnel, TunnelStairs, Walkway, Water.
- **awpy 2.0.2 cannot parse current nav meshes**: `Nav.from_path()` raises
  `ValueError: Unsupported nav version: 36` (upstream issue
  pnxenopoulos/awpy#485, open). VRF 20.0 **does** parse v36: de_anubis nav =
  **2,633 areas**, tile size 128, cell size 1.5 (CLI summary output). The
  CLI does not export area/connection data to disk — a Python nav-v36
  reader is required (Task 8).
- **awpy hosted artifacts are unreliable**: `awpy get maps` / `get navs`
  404 for the current default patch (`https://awpycs.com/17595823/maps.zip`).
  Do not build anything load-bearing on awpy artifact downloads; radar
  images are a nice-to-have pending a source (see Needs-from-user).

### LLM providers (docs fetched 2026-09-03)
- **Anthropic** (platform.claude.com): Claude Opus 4.6
  (`claude-opus-4-6-20260205`) and Sonnet 4.6 (`claude-sonnet-4-6-20260217`)
  and newer (Opus 5 `claude-opus-5`, Sonnet 5) have **native 1M-token
  context — no beta header**; the old `context-1m-2025-08-07` header was
  retired 2026-04-30 and Sonnet 4.5 reverted to 200k. Long-context pricing
  applies above 200k input. Structured outputs are GA via
  `output_config.format = {"type":"json_schema","schema":...}` (constrained
  decoding; guaranteed schema compliance) with Python helper
  `client.messages.parse(..., output_format=PydanticModel)` →
  `response.parsed_output`; strict tool use via `strict: true`. Prompt
  caching: `cache_control {"type":"ephemeral"}` (5-min TTL; `"ttl":"1h"` at
  2× write cost), up to 4 explicit breakpoints, min cacheable **1,024
  tokens on Sonnet 4.5/4.6** (4,096 on Opus 4.6), reads at 0.1× input
  price; usage split across `input_tokens`/`cache_creation_input_tokens`/
  `cache_read_input_tokens`. Sonnet 4.6 $3/$15 per MTok, Opus 4.6 $5/$25.
- **Gemini** (ai.google.dev): `gemini-2.5-pro` (stable) and Gemini 3 series
  (`gemini-3-flash-preview`, `gemini-3.1-pro-preview`) have ~1M
  (1,048,576-token) context. SDK is **`google-genai`** (`from google import
  genai`). Structured output: `config={"response_mime_type":
  "application/json", "response_schema": PydanticModel}`; the API accepts an
  OpenAPI-3.0 subset (beware `$ref`/recursive schemas — flatten). Context
  caching (implicit + explicit) supported; Gemini 3 Pro pricing $2/$12 per
  MTok below 200k, doubles above 200k.
- Consequence for design: both adapters keep dossier inputs **< 200k
  tokens** by default (long-context surcharge on both providers); the 1M
  window is headroom for corpus-scale drill-downs, not the default posture.

## Assumptions

1. Product surface is a **locally hosted web app** (single analyst/team,
   `uvicorn` on localhost): upload demos → background ingest → per-team
   chat + downloadable dossier. No auth, no multi-tenant, no cloud deploy
   in v1 (spec Phase 6 territory). Uploads accept `.dem`, `.dem.zst`,
   `.dem.gz`, `.dem.bz2` (FACEIT/MM formats, spec §4).
2. Map pool for v1 = **de_anubis + de_ancient** (the provided VPKs); the
   provided FACEIT demo is the pipeline-development fixture. Pro-corpus
   acquisition is the user's manual task (ToS-safe) — see Needs-from-user.
3. Community callout subdivision (spec §6.5 source 2) ships as a
   hand-editable YAML overlay per map, seeded from totalcsgo.com names;
   the engine lexicon alone is the v1 default since reconciliation proved
   it complete. Subdivision decision (spec §6.7-2) becomes a measured gate
   in Task 15.
4. Sightline/visibility computation (spec §6.5 source 4) is **deferred**:
   awpy `.tri` artifacts 404 for the current patch and geometry LOS is not
   load-bearing for v1. Utility semantics = landing-zone only (spec risk #5
   fallback). The Map Card schema reserves the `sightlines:` block.
5. Voice, chat, veto history, POV demos: out of scope (spec §3.9, §3.10).
6. LLM defaults: Anthropic `claude-sonnet-5` (verified available on the
   user's key 2026-09-03, 1M ctx, structured outputs, $2/$10 per MTok),
   Gemini `gemini-2.5-pro`; both overridable via config/env. Model IDs are
   config data, never hardcoded in logic.
7. Repo gets `git init` in Task 1; one commit per task, direct to `main`.

## Needs from user (also in `docs/NEEDS-FROM-YOU.md`)

- **N1 — API keys:** `ANTHROPIC_API_KEY` and/or `GEMINI_API_KEY` in `.env`
  for the chat/dossier/quiz features. Everything up to mining runs offline.
- **N2 — radar images (optional):** CS2 install's `pak01_dir.vpk` path (VRF
  can extract overview art) or awpy artifacts once republished — only for
  human-facing evidence plots.
- **N3 — VPK refresh path:** on CS2 updates, re-supply the map VPKs (or a
  game install path) so Map Cards can be recompiled per `game_version`.
- **N4 — (optional, later) pro corpus for calibration/benchmark:** demos
  are the app's *runtime input* — users upload them; nothing is needed to
  build. But three research-grade items want a multi-demo pro corpus when
  convenient: lineup-cluster eps sweep (§6.7-4), buy-bin calibration, and
  the Phase-5 prediction benchmark (≥300 held-out rounds). These run on
  whatever accumulates through the app; they gate no product feature.

## Global constraints

- Python ≥3.13; deps pinned exactly in `pyproject.toml` +
  `uv.lock`: `demoparser2==0.42.0`, `awpy==2.0.2`, `polars`, `duckdb`,
  `pyarrow`, `pydantic>=2`, `PyYAML`, `fastapi`, `uvicorn`,
  `python-multipart`, `zstandard`, `scikit-learn`, `pytest`,
  `anthropic`, `google-genai`. Never upgrade parsers without re-running the
  Task 6 reconciliation gate.
- Every derived artifact is stamped with: demo sha256 (`match_id`), parser
  versions, `map_name`, `patch_version`, and Map Card checksum. Artifacts
  from different card versions must never mix (spec risk #10).
- The zone lexicon is closed: any generated zone id outside it is a lint
  failure (machine-detectable hallucination, spec §6.5).
- All LLM calls go through the provider abstraction; no SDK import outside
  `counterstrat/llm/`. No live-API calls in unit tests — record/replay
  fixtures only.
- **Development LLM policy (user directive, 2026-09-03):** every dev-time /
  live-test call resolves its model at call time — Anthropic: the newest
  sonnet-family id from `GET /v1/models` sorted by `created_at`
  (`resolve_latest_sonnet()`; today that is `claude-sonnet-5`); Gemini:
  `DEV_GEMINI_MODEL = "gemini-3.8-flash"` (verified available on the dev
  key). **Runtime model choice belongs to the app user**: they supply their
  own key and pick the model in the settings UI (Tasks 21/23); config
  defaults are first-run seeds only, never baked into logic.
- Tick rate is 64; round clock 115 s (1:55), bomb 40 s, defuse 10 s / 5 s
  with kit (CS2 constants, spec §6.5 objectives block).
- Windows-first paths (the dev box), but use `pathlib` everywhere; nothing
  may hardcode `D:\`.

## File structure (target)

```
counter-strat/
├── pyproject.toml            # uv-managed; console script `counterstrat`
├── .env.example              # ANTHROPIC_API_KEY=, GEMINI_API_KEY=
├── .gitignore                # .venv/, scratch/, tools/, data/, .env
├── docs/
│   ├── plans/                # spec + this plan
│   └── NEEDS-FROM-YOU.md
├── tools/vrf/                # vendored Source2Viewer-CLI (gitignored)
├── maps/<map>/<map>.vpk      # user-supplied inputs (gitignored, manifest tracked)
├── demos/*.dem               # user-supplied inputs (gitignored, manifest tracked)
├── data/                     # generated, gitignored
│   ├── uploads/              # raw uploaded demos (decompressed)
│   ├── jobs/<job_id>.json    # ingest job state
│   ├── lake/<match_id>/*.parquet
│   ├── mapcards/<map>/       # places.json, nav.json, card.yaml
│   ├── scripts/<match_id>/   # roundscripts (.txt + .json)
│   ├── teambooks/<team_key>/
│   └── chats/<session_id>.jsonl   # chat transcripts
├── src/counterstrat/
│   ├── __init__.py
│   ├── config.py             # AppConfig (pydantic-settings style, env + yaml)
│   ├── constants.py          # TICK_RATE, ROUND_SECONDS, BOMB_SECONDS, buy bins
│   ├── corpus.py             # manifest: register/hash/list demos
│   ├── lake/
│   │   ├── extract.py        # demo → parquet set
│   │   └── duck.py           # DuckDB view factory
│   ├── mapcard/
│   │   ├── vrf.py            # VRF CLI wrapper (extract vents/nav from VPK)
│   │   ├── vents.py          # .vents text → PlaceVolume list
│   │   ├── nav36.py          # nav v36 binary reader → areas + connections
│   │   ├── lexicon.py        # engine places + alias overlay → Lexicon
│   │   ├── zones.py          # ZoneMapper: (x,y,z) → zone id
│   │   ├── compile.py        # card.yaml compiler (topology/rotates/timings)
│   │   └── quiz.py           # map-quiz generation + grading
│   ├── roundscript/
│   │   ├── models.py         # pydantic: RoundScript, Beat, UtilEvent, ...
│   │   ├── econ.py           # buy classification (MR12 bins)
│   │   ├── beats.py          # View A beat frames
│   │   ├── movement.py       # View B movement sentences
│   │   ├── utility.py        # View C utility grammar + lineup clustering
│   │   ├── serialize.py      # orchestrator: lake round → v2 text + v1 JSON
│   │   └── lint.py           # closed-lexicon + citation linter
│   ├── mining/
│   │   └── tendencies.py     # TeamBook miner
│   ├── llm/
│   │   ├── base.py           # LLMClient protocol (+tool use) + usage acct
│   │   ├── anthropic_client.py
│   │   ├── gemini_client.py
│   │   ├── prompts.py        # system/user prompt assembly (Map Card cached)
│   │   ├── tools.py          # analyst tool specs + implementations
│   │   ├── agent.py          # provider-agnostic tool-call loop
│   │   ├── dossier.py        # dossier generation + post-lint
│   │   └── predict.py        # per-round predictor (structured output)
│   ├── eval/
│   │   └── benchmark.py      # Phase-5 heads, baselines, time-ordered split
│   └── web/
│       ├── app.py            # FastAPI factory + uvicorn main()
│       ├── ingest.py         # upload handling + background ingest job
│       ├── routes.py         # REST endpoints
│       ├── chat.py           # session store + agent wiring
│       └── static/           # index.html, app.js, style.css (no build)
└── tests/
    ├── conftest.py           # fixture demo/vpk paths, tiny synthetic lake
    ├── test_corpus.py … (one per module, named test_<module>.py)
    └── fixtures/             # recorded LLM responses, mini vents/nav blobs
```

Data-flow interfaces between layers (exact types defined in tasks):
upload → `corpus.register() → match_id` → `lake.extract(match_id) →
LakePaths` → `mapcard.compile(map) → MapCard` + `zones.ZoneMapper` →
`roundscript.serialize(match_id, mapper) → list[RoundScript]` →
`mining.build_teambook(scripts, team_key) → TeamBook` → chat session
(`llm.agent.run(client, system, history, tools) → answer`) and
`llm.dossier.generate(teambook, exemplars, card, client) → Dossier`;
offline: `eval.benchmark.run(corpus_split) → metrics.json`. The web ingest
job (Task 21) chains register→extract→(card if missing)→serialize→mine
automatically per upload.

---

## Tasks

Conventions for every task: run tests with `uv run pytest -q`, lint with
`uv run ruff check .`; both must be clean before the commit step. Tests that
need the real demo/VPK are marked `@pytest.mark.demo` and auto-skip when the
files are absent, so CI stays green without big binaries.

### Task 1: Repo scaffold + pinned environment

**Goal:** Installable `counterstrat` package with locked deps, pytest, ruff,
git history started.
**Difficulty:** EASY

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`,
  `src/counterstrat/__init__.py`, `src/counterstrat/constants.py`,
  `tests/conftest.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: `counterstrat.constants` — `TICK_RATE=64`, `ROUND_SECONDS=115`,
  `BOMB_SECONDS=40`, `BUY_BINS` (see code), used by every later task.

- [x] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "counterstrat"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
  "demoparser2==0.42.0",
  "awpy==2.0.2",
  "polars>=1.44",
  "duckdb>=1.5",
  "pyarrow>=25",
  "pandas>=2.2",
  "pydantic>=2.7",
  "PyYAML>=6",
  "scikit-learn>=1.5",
  "scipy>=1.14",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "zstandard>=0.23",
  "anthropic>=0.40",
  "google-genai>=1.0",
  "python-dotenv>=1.0",
]

[project.scripts]
counterstrat = "counterstrat.web.app:main"   # starts uvicorn on localhost

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
markers = ["demo: needs the real demo/VPK fixtures", "live: needs API keys"]
addopts = "-m 'not live'"

[tool.ruff]
line-length = 100
```

- [x] **Step 2: Write `.gitignore`** (`.venv/`, `data/`, `tools/`,
  `scratch/`, `demos/`, `maps/`, `.env`, `__pycache__/`, `*.parquet`) and
  `.env.example` with `ANTHROPIC_API_KEY=` and `GEMINI_API_KEY=` lines.

- [x] **Step 3: Write `src/counterstrat/constants.py`**

```python
"""Game + pipeline constants. Sources: CS2 defaults; spec §6.5 objectives."""
TICK_RATE = 64
ROUND_SECONDS = 115          # 1:55
BOMB_SECONDS = 40
DEFUSE_SECONDS = (10, 5)     # (no kit, kit)
TRADE_WINDOW_S = 4.0         # spec §6.2 timeline: traded_within_4s
BEAT_INTERVAL_S = 15         # spec §6.6 View A; re-decided in Task 15 gate
TICKS_HZ = 4                 # lake tick sampling rate (spec Phase 1)

# Team freeze-end equipment value bins (MR12; calibrate in Task 12 gate).
# Mirrors awpy's CS:GO-era bins, team-of-5 totals.
BUY_BINS = {                 # upper bounds, USD equip value
    "full_eco": 5_000,
    "semi_eco": 10_000,
    "semi_buy": 20_000,
    "full_buy": float("inf"),
}
```

- [x] **Step 4: Write `tests/conftest.py`**

```python
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demos" / "1-7065ab7c-bc8f-4995-adf1-ac774327c5db-1-1.dem"
ANUBIS_VPK = REPO / "maps" / "de_anubis" / "de_anubis.vpk"
VRF = REPO / "tools" / "vrf" / "Source2Viewer-CLI.exe"


@pytest.fixture(scope="session")
def demo_path() -> Path:
    if not DEMO.exists():
        pytest.skip("real demo fixture not present")
    return DEMO


@pytest.fixture(scope="session")
def anubis_vpk() -> Path:
    if not ANUBIS_VPK.exists():
        pytest.skip("anubis vpk not present")
    return ANUBIS_VPK


@pytest.fixture(scope="session")
def vrf_cli() -> Path:
    if not VRF.exists():
        pytest.skip("VRF CLI not vendored (see plan Context for URL+sha256)")
    return VRF
```

and `tests/test_smoke.py`:

```python
from counterstrat.constants import BUY_BINS, TICK_RATE


def test_constants():
    assert TICK_RATE == 64
    assert list(BUY_BINS) == ["full_eco", "semi_eco", "semi_buy", "full_buy"]
```

- [x] **Step 5: Install + run**
Run: `uv sync; uv run pytest -q` — expect 1 pass. `uv run ruff check .` clean.
- [x] **Step 6: Commit**
`git init; git add -A; git commit -m "feat: scaffold counterstrat package"`
(also commit the two docs under `docs/` and `docs/NEEDS-FROM-YOU.md`).

**Done when:** `uv run counterstrat` fails only because `web/app.py`
doesn't exist yet (expected until Task 21); pytest+ruff green.

---

### Task 2: Corpus manifest

**Goal:** Deterministic registry of demos: hash, header metadata, roster-
derived team keys — the join key for every downstream artifact.
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/corpus.py`, `tests/test_corpus.py`

**Interfaces:**
- Produces: `register_demo(dem: Path, manifest: Path) -> DemoRecord` and
  `DemoRecord` pydantic model with fields
  `match_id: str` (first 16 hex of sha256), `path: str`, `map_name: str`,
  `patch_version: str`, `demo_version_guid: str`, `server_name: str`,
  `registered_at: str` (iso). Manifest file: `data/corpus.jsonl`, one
  record per line, idempotent on re-register (same hash → no dup).
- Consumes: `demoparser2.DemoParser(path).parse_header()` (verified keys:
  `map_name`, `patch_version`, `demo_version_guid`, `server_name`).

- [x] **Step 1: Failing test** (`tests/test_corpus.py`)

```python
import json

from counterstrat.corpus import register_demo


@pytest.mark.demo
def test_register_demo_idempotent(demo_path, tmp_path):
    manifest = tmp_path / "corpus.jsonl"
    rec1 = register_demo(demo_path, manifest)
    rec2 = register_demo(demo_path, manifest)
    assert rec1.match_id == rec2.match_id
    assert rec1.map_name == "de_anubis"
    assert rec1.patch_version == "14178"
    lines = manifest.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["match_id"] == rec1.match_id
```

- [x] **Step 2: Run, verify fails** (`ModuleNotFoundError`).
- [x] **Step 3: Implement**

```python
"""Corpus manifest: content-addressed demo registry."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from demoparser2 import DemoParser
from pydantic import BaseModel


class DemoRecord(BaseModel):
    match_id: str
    path: str
    map_name: str
    patch_version: str
    demo_version_guid: str
    server_name: str
    registered_at: str


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def register_demo(dem: Path, manifest: Path) -> DemoRecord:
    match_id = _sha256_16(dem)
    existing = load_manifest(manifest)
    if match_id in existing:
        return existing[match_id]
    header = DemoParser(str(dem)).parse_header()
    rec = DemoRecord(
        match_id=match_id,
        path=str(dem),
        map_name=header["map_name"],
        patch_version=header["patch_version"],
        demo_version_guid=header["demo_version_guid"],
        server_name=header.get("server_name", ""),
        registered_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as f:
        f.write(rec.model_dump_json() + "\n")
    return rec


def load_manifest(manifest: Path) -> dict[str, DemoRecord]:
    if not manifest.exists():
        return {}
    out: dict[str, DemoRecord] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = DemoRecord(**json.loads(line))
            out[rec.match_id] = rec
    return out
```

- [x] **Step 4: Run test — PASS.** `uv run pytest tests/test_corpus.py -q`
- [x] **Step 5: Commit** `git commit -m "feat: corpus manifest"`

**Done when:** registering the fixture demo twice yields one manifest line
with `map_name=de_anubis`.

---

### Task 3: Extraction lake (demo → parquet)

**Goal:** One function turns a registered demo into the Level-0 parquet set
(spec Phase 1), combining awpy's analytics tables (verified) with a
demoparser2 4 Hz tick pull of the 29 verified props, plus roster-keyed team
identity per round.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/lake/__init__.py`,
  `src/counterstrat/lake/extract.py`, `tests/test_lake_extract.py`

**Interfaces:**
- Produces: `extract_lake(record: DemoRecord, out_root: Path) -> LakePaths`.
  `LakePaths` (pydantic): dir + one parquet per table:
  `rounds, kills, damages, shots, grenades, smokes, infernos, bomb,
  item_purchase, ticks, rosters`. All tables carry `match_id`.
  - `rounds` = awpy rounds + `t_team_key`/`ct_team_key` columns.
  - `ticks` = 4 Hz sample, all 10 players, 29 props + `round_num` +
    `clock_s` (float seconds since freeze_end; negative during freeze).
  - `rosters` = one row per (round_num, side): `team_key`
    (sha1[:12] of sorted steamids), `steamids: list`, `clan_name`.
- Consumes: Task 2 `DemoRecord`; awpy `Demo.parse()` tables (verified
  shapes in Context); demoparser2 `parse_ticks(props, ticks=[...])`.

Implementation notes (verified constraints):
- Guard optional awpy tables (`footsteps` KeyErrors on FACEIT demos) —
  only persist the 8 tables listed; wrap each `getattr` in try/except.
- awpy returns **polars** frames; demoparser2 returns **pandas** — convert
  pandas → polars via `pl.from_pandas`, write with `df.write_parquet`.
- 4 Hz tick list: `range(first_round_start_tick, last_end_tick, 16)`.
- `clock_s = (tick - freeze_end_tick_of_round) / 64.0`, joined by tick
  ranges from the rounds table (`join_asof` on tick against round starts).
- Team keys: at each round's `freeze_end`, group alive players by
  `team_name` (`CT`/`TERRORIST`), `team_key = sha1(",".join(sorted(ids)))[:12]`.
  Sides swap at halftime and in OT — derive per round, never assume.

- [x] **Step 1: Failing test**

```python
import polars as pl

from counterstrat.corpus import register_demo
from counterstrat.lake.extract import extract_lake


@pytest.mark.demo
def test_extract_lake(demo_path, tmp_path):
    rec = register_demo(demo_path, tmp_path / "corpus.jsonl")
    paths = extract_lake(rec, tmp_path / "lake")
    rounds = pl.read_parquet(paths.rounds)
    assert rounds.height == 30                      # verified count
    assert {"t_team_key", "ct_team_key"} <= set(rounds.columns)
    ticks = pl.read_parquet(paths.ticks)
    assert ticks.height > 400_000                   # ~2M/4 minus freeze gaps
    assert {"last_place_name", "clock_s", "round_num"} <= set(ticks.columns)
    # every gameplay tick maps into a round and clock is sane
    assert ticks["clock_s"].max() < 130
    rosters = pl.read_parquet(paths.rosters)
    assert rosters["team_key"].n_unique() == 2      # two teams, PUG-verified
```

- [x] **Step 2: Run — fails (module missing).**
- [x] **Step 3: Implement `extract.py`** — skeleton the implementer fills
  strictly within these verified APIs:

```python
import hashlib
from pathlib import Path

import polars as pl
from awpy import Demo
from demoparser2 import DemoParser
from pydantic import BaseModel

from counterstrat.corpus import DemoRecord

TICK_PROPS = [
    "X", "Y", "Z", "last_place_name", "health", "armor_value", "has_helmet",
    "has_defuser", "is_alive", "active_weapon_name", "balance",
    "cash_spent_this_round", "current_equip_value", "round_start_equip_value",
    "is_walking", "flash_duration", "is_scoped", "is_defusing",
    "in_bomb_zone", "team_name", "team_clan_name", "is_bomb_planted",
    "total_rounds_played", "ct_losing_streak", "t_losing_streak",
    "velocity", "pitch", "yaw",
]
AWPY_TABLES = ["rounds", "kills", "damages", "shots", "grenades",
               "smokes", "infernos", "bomb"]


class LakePaths(BaseModel):
    root: str
    rounds: str; kills: str; damages: str; shots: str; grenades: str
    smokes: str; infernos: str; bomb: str; item_purchase: str
    ticks: str; rosters: str


def extract_lake(record: DemoRecord, out_root: Path) -> LakePaths:
    out = out_root / record.match_id
    out.mkdir(parents=True, exist_ok=True)
    dem = Demo(record.path)
    dem.parse()
    tables: dict[str, pl.DataFrame] = {}
    for name in AWPY_TABLES:
        try:
            tables[name] = getattr(dem, name)
        except Exception:                       # optional-table guard
            tables[name] = pl.DataFrame()
    rounds = tables["rounds"]
    parser = DemoParser(record.path)
    purchases = pl.from_pandas(
        parser.parse_event("item_purchase", other=["total_rounds_played"])
    )
    ticks = _sample_ticks(parser, rounds)       # 4 Hz, clock_s, round_num
    rosters = _rosters(parser, rounds)          # team_key per (round, side)
    rounds = _attach_team_keys(rounds, rosters)
    tables |= {"item_purchase": purchases, "ticks": ticks, "rosters": rosters,
               "rounds": rounds}
    paths = {}
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        df.with_columns(pl.lit(record.match_id).alias("match_id")) \
          .write_parquet(p)
        paths[name] = str(p)
    return LakePaths(root=str(out), **paths)
```

  `_sample_ticks`: build tick list `range(rounds["start"].min(),
  rounds["end"].max(), 16)`, one `parser.parse_ticks(TICK_PROPS, ticks=...)`
  call, `pl.from_pandas`, then `join_asof` against
  `rounds.select("round_num", "start", "freeze_end", "end")` sorted by
  `start` to attach `round_num`, filter `tick <= end`, compute
  `clock_s = (tick - freeze_end) / 64.0`.
  `_rosters`: `parse_ticks(["team_name","team_clan_name"],
  ticks=rounds["freeze_end"].to_list())`, group by (tick→round_num,
  team_name), aggregate sorted steamid list → sha1[:12].

- [x] **Step 4: Run test — PASS** (allow ~30 s; it parses the demo twice).
- [x] **Step 5: Commit** `git commit -m "feat: extraction lake"`

**Done when:** the fixture demo yields 11 parquet files; rounds=30; two
distinct team_keys; `clock_s` bounded by round clock + bomb time.

---

### Task 4: DuckDB views over the lake

**Goal:** One-call analytical SQL surface (spec Phase 1 exit tooling; later
the text-to-SQL tool surface for the interactive analyst).
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/lake/duck.py`, `tests/test_lake_duck.py`

**Interfaces:**
- Produces: `connect_lake(lake_root: Path) -> duckdb.DuckDBPyConnection`
  registering views `rounds, kills, ticks, ...` (glob over
  `lake_root/*/<table>.parquet`, so multi-demo unions are automatic).

- [x] **Step 1: Failing test**

```python
from counterstrat.lake.duck import connect_lake


@pytest.mark.demo
def test_duck_views(demo_path, tmp_path):
    rec = register_demo(demo_path, tmp_path / "corpus.jsonl")
    extract_lake(rec, tmp_path / "lake")
    con = connect_lake(tmp_path / "lake")
    n = con.sql("select count(*) from rounds").fetchone()[0]
    assert n == 30
    places = con.sql(
        "select count(distinct last_place_name) from ticks"
    ).fetchone()[0]
    assert places == 28        # verified inventory
```

- [x] **Step 2: fails.**
- [x] **Step 3: Implement**

```python
import duckdb
from pathlib import Path

TABLES = ["rounds", "kills", "damages", "shots", "grenades", "smokes",
          "infernos", "bomb", "item_purchase", "ticks", "rosters"]


def connect_lake(lake_root: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for t in TABLES:
        pattern = str(lake_root / "*" / f"{t}.parquet").replace("\\", "/")
        con.sql(
            f"create or replace view {t} as "
            f"select * from read_parquet('{pattern}')"
        )
    return con
```

- [x] **Step 4: PASS.**  - [x] **Step 5: Commit** `"feat: duckdb lake views"`

**Done when:** the two asserts above hold against the fixture lake.

---

### Task 5: VRF wrapper + env_cs_place extraction

**Goal:** Programmatic `VPK → places.json` (the engine zone lexicon, spec
§6.5 source 1), wrapping the vendored Source2Viewer-CLI with the exact
invocation verified in Context.
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/mapcard/__init__.py`,
  `src/counterstrat/mapcard/vrf.py`, `src/counterstrat/mapcard/vents.py`,
  `tests/test_mapcard_vents.py`
- Test fixture: `tests/fixtures/mini.vents` — hand-write 3 entity blocks
  copied verbatim from the format in Context: `env_cs_place` "TSpawn"
  (origin `[ -16.0, -1520.0, 104.0 ]`), `env_cs_place` "Street", and one
  `func_nav_markup` block (must be ignored by the parser).

**Interfaces:**
- Produces:
  `vrf.extract_map_assets(vpk: Path, vrf_cli: Path, out_dir: Path) -> MapAssets`
  where `MapAssets` has `vents: Path` (decompiled text) and `nav: Path`
  (raw binary). `vents.parse_places(vents_path: Path) -> list[PlaceVolume]`
  with `PlaceVolume(place_name: str, origin: tuple[float, float, float],
  hammer_id: str, model: str)`; `vents.unique_places(vols) -> list[str]`
  (sorted unique names).

- [x] **Step 1: Failing unit test** (no VPK needed — uses `mini.vents`):

```python
from counterstrat.mapcard.vents import parse_places, unique_places


def test_parse_places_mini():
    vols = parse_places(Path("tests/fixtures/mini.vents"))
    assert len(vols) == 2                      # only env_cs_place blocks
    assert vols[0].place_name == "TSpawn"
    assert vols[0].origin == (-16.0, -1520.0, 104.0)
    assert unique_places(vols) == ["Street", "TSpawn"]
```

Plus the integration test:

```python
@pytest.mark.demo
def test_extract_anubis(anubis_vpk, vrf_cli, tmp_path):
    assets = extract_map_assets(anubis_vpk, vrf_cli, tmp_path)
    vols = parse_places(assets.vents)
    names = unique_places(vols)
    assert len(vols) == 62 and len(names) == 28    # verified counts
    assert assets.nav.stat().st_size == 495_530    # verified size
```

- [x] **Step 2: fails.**
- [x] **Step 3: Implement.** `vrf.py`:

```python
import subprocess
from pathlib import Path

from pydantic import BaseModel


class MapAssets(BaseModel):
    vents: Path
    nav: Path


def extract_map_assets(vpk: Path, vrf_cli: Path, out_dir: Path) -> MapAssets:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("vents_c", "nav"):
        args = [str(vrf_cli), "-i", str(vpk), "-e", ext, "-o", str(out_dir)]
        if ext == "vents_c":
            args.insert(5, "-d")               # decompile entities to text
        subprocess.run(args, check=True, capture_output=True, timeout=300)
    map_name = vpk.stem                        # de_anubis
    vents = out_dir / "maps" / map_name / "entities" / "default_ents.vents"
    nav = out_dir / "maps" / f"{map_name}.nav"
    if not vents.exists() or not nav.exists():
        raise FileNotFoundError(f"VRF extraction incomplete under {out_dir}")
    return MapAssets(vents=vents, nav=nav)
```

`vents.py` — the decompiled format is `====N====` separated blocks of
`key<spaces>value` lines (verified sample in Context):

```python
import re
from pathlib import Path

from pydantic import BaseModel

_BLOCK = re.compile(r"^====\d+====$", re.M)
_KV = re.compile(r"^(\w+)\s+(.*)$")


class PlaceVolume(BaseModel):
    place_name: str
    origin: tuple[float, float, float]
    hammer_id: str
    model: str


def parse_places(vents_path: Path) -> list[PlaceVolume]:
    text = vents_path.read_text(encoding="utf-8", errors="replace")
    out: list[PlaceVolume] = []
    for block in _BLOCK.split(text):
        kv = {}
        for line in block.splitlines():
            m = _KV.match(line.strip())
            if m:
                kv[m.group(1)] = m.group(2).strip()
        if kv.get("classname") != '"env_cs_place"':
            continue
        origin = tuple(
            float(x) for x in kv["origin"].strip("[] ").split(",")
        )
        out.append(PlaceVolume(
            place_name=kv["place_name"].strip('"'),
            origin=origin,  # type: ignore[arg-type]
            hammer_id=kv.get("hammeruniqueid", "").strip('"'),
            model=kv.get("model", ""),
        ))
    return out


def unique_places(vols: list[PlaceVolume]) -> list[str]:
    return sorted({v.place_name for v in vols})
```

- [x] **Step 4: PASS both tests.**
- [x] **Step 5: Commit** `"feat: VPK place extraction via VRF"`

**Done when:** Anubis yields 62 volumes / 28 names; Ancient VPK also
extracts (add a second `@pytest.mark.demo` assert: names non-empty and
`nav` size 371,251).

---

### Task 6: Place reconciliation gate (spec §6.7-1, Phase 2a exit)

**Goal:** Automated check that the VPK lexicon and the lake's observed
`last_place_name` values reconcile; hard-fails Map Card compilation on
game-version drift (spec risk #10).
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/mapcard/reconcile.py`,
  `tests/test_mapcard_reconcile.py`

**Interfaces:**
- Produces: `reconcile(lexicon_names: list[str], lake_con, map_name: str)
  -> ReconcileReport` with `in_vpk_not_demo: list[str]`,
  `in_demo_not_vpk: list[str]`, `ok: bool`. Policy: `in_demo_not_vpk`
  non-empty → `ok=False` (demo speaks names the card lacks = fatal);
  `in_vpk_not_demo` is a warning only (unvisited zones are legal).

- [x] **Step 1: Failing test** — unit (fake lists) + integration:

```python
def test_reconcile_policy():
    r = reconcile_names(["A", "B"], ["A"])
    assert r.ok and r.in_vpk_not_demo == ["B"]
    r2 = reconcile_names(["A"], ["A", "Ghost"])
    assert not r2.ok and r2.in_demo_not_vpk == ["Ghost"]


@pytest.mark.demo
def test_reconcile_anubis(demo_path, anubis_vpk, vrf_cli, tmp_path):
    ...  # build lake (Task 3) + places (Task 5), then:
    report = reconcile(names, con, "de_anubis")
    assert report.ok and not report.in_vpk_not_demo
```

- [x] **Step 2–4: implement `reconcile_names(vpk, demo)` (set diff) and the
  duckdb-backed `reconcile` (`select distinct last_place_name from ticks
  where last_place_name is not null and last_place_name != ''`); PASS.**
- [x] **Step 5: Commit** `"feat: place reconciliation gate"`

**Done when:** Anubis reconciles exactly (verified ground truth); a
synthetic ghost name flips `ok=False`.

---

### Task 7: Zone mapper — (x, y, z) → zone id

**Goal:** Classify arbitrary map points (grenade landings, plant spots)
into lexicon zones. Method: KNN over the corpus's own labeled player-tick
cloud (millions of `(x,y,z) → last_place_name` rows verified present),
z-weighted for multilevel correctness — the demo-derived mapper the spec's
§6.5.1 audit bucket enables, needing no volume-mesh math for v1.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/mapcard/zones.py`, `tests/test_mapcard_zones.py`

**Interfaces:**
- Produces: `ZoneMapper.fit(ticks: pl.DataFrame, z_scale: float = 2.0) ->
  ZoneMapper`; `ZoneMapper.zone(x, y, z) -> str`;
  `ZoneMapper.zones(df: pl.DataFrame, x="X", y="Y", z="Z") -> pl.Series`;
  `save(path)/load(path)` via joblib-free pickle of the fitted
  `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)`.
- Consumes: lake `ticks` (Task 3) filtered `is_alive & last_place_name != ''`.

- [x] **Step 1: Failing test** — synthetic + held-out accuracy:

```python
def test_zone_mapper_synthetic():
    df = pl.DataFrame({
        "X": [0.0] * 50 + [1000.0] * 50,
        "Y": [0.0] * 100,
        "Z": [0.0] * 100,
        "last_place_name": ["A"] * 50 + ["B"] * 50,
    })
    zm = ZoneMapper.fit(df)
    assert zm.zone(10, 0, 0) == "A" and zm.zone(990, 0, 0) == "B"


@pytest.mark.demo
def test_zone_mapper_holdout(anubis_lake):     # fixture built once per session
    ticks = pl.read_parquet(anubis_lake.ticks).filter(
        (pl.col("is_alive")) & (pl.col("last_place_name") != "")
    )
    train, test = ticks.head(ticks.height - 20_000), ticks.tail(20_000)
    zm = ZoneMapper.fit(train)
    acc = (zm.zones(test) == test["last_place_name"]).mean()
    assert acc >= 0.97       # spec Phase-2a exit: round-trip spot check
```

- [x] **Step 2: fails.**
- [x] **Step 3: Implement** — subsample fit input to ≤300k rows
  (deterministic seed 0) for tractability; scale z by `z_scale` before fit
  and query; store `z_scale` with the model.
- [x] **Step 4: PASS (holdout run ~1 min).**
- [x] **Step 5: Commit** `"feat: KNN zone mapper"`

**Done when:** held-out agreement ≥97% on the fixture lake; mapper
round-trips synthetic clusters exactly.

---

### Task 8: Nav v36 reader (adjacency + traversal), with demo fallback

**Goal:** Zone-graph inputs for the Map Card: walkable-area adjacency and
traversal seconds. Primary: Python reader for `.nav` version 36 (awpy 2.0.2
tops out at 35 — verified; VRF 20.0 parses 36 — verified, 2,633 areas on
Anubis). Fallback (always built, also the §6.7-5 calibrator): zone-level
adjacency + transit times observed directly from lake tick transitions.
**Difficulty:** HARD (reader) / MEDIUM (fallback)

**Files:**
- Create: `src/counterstrat/mapcard/nav36.py`,
  `src/counterstrat/mapcard/transitions.py`, `tests/test_mapcard_nav.py`,
  `tests/test_mapcard_transitions.py`

**Interfaces:**
- Produces:
  - `nav36.read_nav(path: Path) -> NavMesh` with
    `NavMesh(version: int, areas: dict[int, NavArea])`,
    `NavArea(area_id: int, corners: list[tuple[float,float,float]],
    connections: list[int])`. May raise `NavUnsupportedError` — callers
    must degrade to transitions-only mode.
  - `transitions.zone_graph(ticks: pl.DataFrame) -> ZoneGraph` with
    `edges: dict[tuple[str, str], EdgeStat]`,
    `EdgeStat(n: int, median_transit_s: float)` — built from consecutive
    4 Hz samples per player where `last_place_name` changes; transit time =
    dwell-boundary to dwell-boundary, capped at 30 s.
- Consumes: Task 5 `MapAssets.nav`; Task 3 `ticks`.

Implementation sources for the reader (do not invent the byte layout):
1. `.venv/Lib/site-packages/awpy/nav.py` — working v35 reader to port from
   (local, readable).
2. VRF's `ValveResourceFormat/NavMesh/*.cs` on GitHub at tag `20.0`
   (`NavMeshFile.cs`, `NavMeshArea.cs`) — the v36 delta (VRF 20.0 release
   notes: "Added nav mesh data previously skipped as unknown bytes").
If after porting, `read_nav` cannot round-trip Anubis (2,633 areas), commit
the reader behind `NavUnsupportedError` and proceed on the fallback — the
Map Card compiler (Task 10) accepts either source and records which one it
used in the card metadata.

- [x] **Step 1: Failing tests**

```python
@pytest.mark.demo
def test_read_nav_anubis(anubis_assets):
    mesh = read_nav(anubis_assets.nav)
    assert mesh.version == 36
    assert len(mesh.areas) == 2633             # verified via VRF CLI
    a = next(iter(mesh.areas.values()))
    assert len(a.corners) >= 3


def test_zone_graph_synthetic():
    # one player walking A→B→C at 4 Hz, 2 s per zone
    df = pl.DataFrame({
        "steamid": [1] * 24, "round_num": [1] * 24,
        "tick": list(range(0, 24 * 16, 16)),
        "clock_s": [t / 4 for t in range(24)],
        "last_place_name": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
        "is_alive": [True] * 24,
    })
    g = zone_graph(df)
    assert ("A", "B") in g.edges and ("B", "C") in g.edges
    assert ("A", "C") not in g.edges           # no skip edges
```

- [x] **Step 2: fails.**
- [x] **Step 3: Implement `transitions.py` first** (deterministic, pure
  polars: sort by steamid, round, tick; window over consecutive rows; emit
  edge when zone changes and gap == one sample; aggregate n + median).
  Sample interval is derived as the smallest positive per-player tick gap
  (4 ticks in the fixture lake), so gaps left by dropped samples are skipped
  rather than turned into skip edges.
- [x] **Step 4: Implement `nav36.py` by porting** (sources above). Track
  actual v36 layout findings as comments citing the C# lines used.
  **v36 delta found:** two binary KV3 v5 documents (`KV3\x05`), 8-byte
  aligned — one right after the `unk1` flags dword, one between the
  movable-mesh table and the area table. Skipped by size from the KV3 header
  (`BinaryKV3.ReadBuffer`) without decoding; area layout is unchanged from
  v35. Anubis: reader returns version 36, 2,633 areas — **nav is the primary
  source, no `NavUnsupportedError`.**
- [x] **Step 5: PASS; commit** `"feat: nav v36 reader + demo transition graph"`

**Done when:** synthetic graph test green; Anubis nav test green **or**
reader raises `NavUnsupportedError` and the transitions fallback is the
recorded source (both outcomes acceptable; record which in commit message).

---

### Task 9: Zone lexicon builder (engine base + alias overlay)

**Goal:** The closed, versioned zone lexicon every artifact must speak
(spec §6.5): engine places as base layer, plus a human-editable YAML overlay
adding aliases, role tags, and (later, if Task 15's gate demands)
subdivision zones. Checksummed so downstream artifacts can stamp it.
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/mapcard/lexicon.py`,
  `src/counterstrat/mapcard/overlays/de_anubis.yaml`,
  `src/counterstrat/mapcard/overlays/de_ancient.yaml`,
  `tests/test_mapcard_lexicon.py`

**Interfaces:**
- Produces: `build_lexicon(map_name: str, places: list[str],
  overlay_path: Path | None) -> Lexicon`;
  `Lexicon(map_name: str, zones: dict[str, ZoneDef], checksum: str)`;
  `ZoneDef(id: str, aliases: list[str], tags: list[str],
  engine_place: str)`; `Lexicon.is_valid_zone(zone_id) -> bool`;
  `Lexicon.checksum` = sha256[:12] of the canonical JSON dump.
- Consumes: Task 5 `unique_places`.

Overlay YAML format (seed `de_anubis.yaml` with exactly this — analyst
extends it later; names sourced from totalcsgo.com community callouts):

```yaml
map: de_anubis
zones:
  BombsiteA:  {aliases: [A site], tags: [site]}
  BombsiteB:  {aliases: [B site], tags: [site]}
  Middle:     {aliases: [mid], tags: [mid_control]}
  Connector:  {aliases: [connector, con], tags: [connector]}
  Water:      {aliases: [water, canal-water], tags: [mid_control]}
  Bridge:     {aliases: [mid bridge], tags: [choke]}
  MidDoors:   {aliases: [mid doors], tags: [choke]}
  Main:       {aliases: [A main], tags: [t_side_entry, choke]}
  Walkway:    {aliases: [A long, walkway], tags: [t_side_entry]}
  Heaven:     {aliases: [heaven], tags: [power_position, ct_side]}
  Street:     {aliases: [street], tags: [ct_side]}
  Canal:      {aliases: [canals], tags: [connector]}
  Tunnel:     {aliases: [B tunnels, upper tunnel], tags: [t_side_entry]}
  LowerTunnel: {aliases: [lower tunnel], tags: [connector]}
  Ruins:      {aliases: [ruins], tags: [b_side]}
  Alley:      {aliases: [alley], tags: [b_side, choke]}
  CTSpawn:    {aliases: [ct spawn], tags: [spawn, ct_side]}
  TSpawn:     {aliases: [t spawn], tags: [spawn]}
# zones omitted here inherit defaults: no aliases, no tags
```

- [x] **Step 1: Failing test**

```python
def test_lexicon_engine_base_plus_overlay(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text(
        "map: m\nzones:\n  A: {aliases: [a site], tags: [site]}\n"
    )
    lex = build_lexicon("m", ["A", "B"], overlay)
    assert set(lex.zones) == {"A", "B"}
    assert lex.zones["A"].tags == ["site"]
    assert lex.zones["B"].aliases == []
    assert lex.is_valid_zone("A") and not lex.is_valid_zone("Ghost")
    assert len(lex.checksum) == 12
    # determinism: same inputs → same checksum
    assert build_lexicon("m", ["B", "A"], overlay).checksum == lex.checksum


def test_lexicon_overlay_unknown_zone_rejected(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("map: m\nzones:\n  Ghost: {tags: [site]}\n")
    with pytest.raises(ValueError, match="Ghost"):
        build_lexicon("m", ["A"], overlay)
```

- [x] **Step 2: fails.** - [x] **Step 3: implement** (yaml.safe_load; overlay
  keys must be a subset of engine places — subdivision zones come later as
  `parent:` entries, rejected for now with a clear message; canonical dump =
  `json.dumps(..., sort_keys=True)`).
- [x] **Step 4: PASS.** - [x] **Step 5: Commit** `"feat: zone lexicon"`

**Done when:** Anubis lexicon builds from the 28 verified names + the
seeded overlay; unknown overlay zones raise.

---

### Task 10: Map Card compiler

**Goal:** Compile the versioned textual atlas (spec §6.5 card format):
zones with tags/quadrants/elevation, incident-encoded topology with
traversal seconds, precomputed rotates, earliest-arrival timings, objective
constants. Sightlines block reserved-empty (Assumption 4). ≤4k tokens.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/mapcard/compile.py`, `tests/test_mapcard_compile.py`

**Interfaces:**
- Produces: `compile_card(lexicon: Lexicon, graph: ZoneGraph,
  ticks: pl.DataFrame, rounds: pl.DataFrame, map_name: str,
  patch_version: str, nav_source: str) -> MapCard`;
  `MapCard.to_yaml() -> str`; `MapCard.checksum: str` (sha256[:12] of
  yaml); fields mirror the spec card: `map, card_version, game_version,
  frame, zones, topology, rotates, timings, objectives, sightlines (empty
  list), doctrine (absent in v1)`.
- Consumes: Tasks 3, 8, 9 outputs.

Computation rules (all deterministic, all precomputed — the LLM never
pathfinds, spec §5.2b):
- `quadrant`: from zone centroid (mean of that zone's tick positions) vs
  map bounding box thirds → one of nine labels (`north-west` … `center` …).
- `elevation`: `upper/lower/mixed` when a zone's tick Z interquartile range
  splits across the map-wide Z median; else omitted.
- `topology`: incident encoding — per zone, dict of neighbor → median
  transit seconds from `ZoneGraph`, edges with `n >= 5` only.
- `rotates`: for each site pair (zones tagged `site`), k-shortest paths
  (k=2) over the zone graph weighted by median transit seconds
  (`networkx.shortest_simple_paths`; networkx ships with awpy — verified
  in the installed dependency set), rendered as
  `{from, to, via: [zones], run_s}`.
- `timings`: per (side, zone): minimum observed `clock_s` of first entry
  by any player of that side across the corpus, rendered `earliest_s`,
  only for zones tagged `site|mid_control|choke` (budget control).
- `objectives`: sites from tags; clock constants from `constants.py`.
- Token budget: `len(yaml) / 4 <= 4000` enforced by test; trim `timings`
  first, then untagged-zone alias lists, if over.

- [x] **Step 1: Failing test** — build the card for the fixture corpus:

```python
@pytest.mark.demo
def test_compile_anubis_card(anubis_bundle):   # lexicon+graph+lake fixture
    card = compile_card(**anubis_bundle)
    y = card.to_yaml()
    assert "map: de_anubis" in y and "topology:" in y
    assert len(y) / 4 <= 4000                  # spec §6.5 budget
    # incident encoding: a known heavy edge exists in some direction
    assert "Middle" in card.topology and card.topology["Middle"]
    # rotates precomputed between the two sites
    assert any(r["from"] == "BombsiteA" and r["to"] == "BombsiteB"
               for r in card.rotates)
    assert card.checksum and card.game_version == "14178"


def test_card_determinism_synthetic(synthetic_bundle):
    a = compile_card(**synthetic_bundle).to_yaml()
    b = compile_card(**synthetic_bundle).to_yaml()
    assert a == b
```

- [x] **Step 2: fails.** - [x] **Step 3: implement.**
- [x] **Step 4: PASS.** - [x] **Step 5: commit** `"feat: map card compiler"`
  (the web ingest job persists cards to `data/mapcards/<map>/card.yaml` in
  Task 21).

**Done when:** Anubis card compiles ≤4k tokens with topology, ≥2 rotates,
timings for tagged zones; identical bytes on re-run.

---

### Task 11: Map quiz harness (spec Phase 2a, §6.7-6)

**Goal:** Auto-graded quiz measuring an LLM's map prior with/without the
card: adjacency, rotate ordering, earliest-arrival comparisons. Grades any
`LLMClient` (Task 18) — offline-testable with a scripted fake client.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/mapcard/quiz.py`, `tests/test_mapcard_quiz.py`,
  `src/counterstrat/llm/__init__.py`, `src/counterstrat/llm/base.py`
  (protocol stub only — the 3-method `LLMClient` Protocol from Task 18's
  Interfaces block; Task 18 fills in adapters, `LLMResult`, `make_client`)

**Interfaces:**
- Produces: `generate_quiz(card: MapCard, n: int = 50, seed: int = 0) ->
  list[QuizQ]` with `QuizQ(kind, prompt, options: list[str], answer: str)`,
  kinds: `adjacency` ("Which zone is NOT directly connected to X?"),
  `rotate` ("Fastest rotate A→B runs through?"), `timing` ("Which zone can
  a T reach earlier?") — all answerable from card fields alone;
  `grade(client: LLMClient, quiz, card_yaml: str | None) -> QuizResult`
  (accuracy, per-kind breakdown, raw transcript for audit).
- Consumes: Task 10 `MapCard`; Task 18 `LLMClient` protocol (quiz merges
  before 18 — define the 3-method protocol here in `llm/base.py` stub, see
  Task 18 Interfaces; Task 18 fills the real adapters in).

- [x] **Step 1: Failing test** — deterministic generation + fake grading:

```python
class ScriptedClient:                # answers everything with option "A"
    def complete(self, *, system, user, max_tokens=512): ...


def test_quiz_generation_deterministic(anubis_card):
    q1 = generate_quiz(anubis_card, n=20, seed=1)
    q2 = generate_quiz(anubis_card, n=20, seed=1)
    assert [q.prompt for q in q1] == [q.prompt for q in q2]
    assert all(q.answer in q.options for q in q1)


def test_grading_counts_correct(anubis_card):
    quiz = generate_quiz(anubis_card, n=10, seed=1)
    always_right = ScriptedAnswerer({q.prompt: q.answer for q in quiz})
    res = grade(always_right, quiz, card_yaml=None)
    assert res.accuracy == 1.0
```

- [x] **Step 2–4: implement + PASS.** Multiple-choice format, answers
  extracted by exact option-letter match; malformed reply = wrong.
- [x] **Step 5: Commit** `"feat: map quiz harness"`

**Done when:** quiz is deterministic per seed, self-grades correctly with
scripted clients. (Live model runs happen via
`python -m counterstrat.mapcard.quiz <map>` — a `__main__` block added in
this task, gated on N1 keys, models per dev policy (latest sonnet /
`gemini-3.8-flash`) — recording no-card vs card accuracies to
`data/mapcards/<map>/quiz_results.json`; spec exit: card closes ≥90% of the
no-card error gap. This is a research harness, not a product surface.)

---

### Task 12: Buy classifier + economy summarizer

**Goal:** Deterministic MR12 buy typing per (round, side) from freeze-end
equipment values, plus round economy summary for RoundScript headers
(spec §6.2 economy block).
**Difficulty:** EASY

**Files:**
- Create: `src/counterstrat/roundscript/__init__.py`,
  `src/counterstrat/roundscript/econ.py`, `tests/test_roundscript_econ.py`

**Interfaces:**
- Produces: `classify_buy(team_equip_value: float) -> str` (bin name from
  `constants.BUY_BINS`); `round_economy(ticks: pl.DataFrame,
  round_num: int) -> dict[str, EconSummary]` keyed `"T"|"CT"` with
  `EconSummary(buy_type: str, spend: int, equip: int, awps: int,
  loss_streak: int)` — computed at the round's freeze-end sample
  (min `clock_s >= 0` per player), `awps` = count of "awp" in
  `active_weapon_name`+`inventory` strings, spend = sum
  `cash_spent_this_round`.
- Consumes: Task 3 ticks.

- [x] **Step 1: Failing tests** — bin edges + fixture spot check:

```python
def test_classify_buy_bins():
    assert classify_buy(3_000) == "full_eco"
    assert classify_buy(5_000) == "semi_eco"     # boundary goes up
    assert classify_buy(19_999) == "semi_buy"
    assert classify_buy(26_000) == "full_buy"


@pytest.mark.demo
def test_round_economy_pistol(anubis_lake):
    econ = round_economy(pl.read_parquet(anubis_lake.ticks), round_num=1)
    assert econ["T"].buy_type in {"full_eco", "semi_eco"}   # pistol round
    assert econ["CT"].equip > 0
```

- [x] **Step 2–4: implement + PASS.**
- [x] **Step 5: Commit** `"feat: buy classifier + economy summary"`

**Done when:** pistol rounds classify eco-tier on the fixture; bins match
`constants.BUY_BINS` exactly. **Calibration note (spec §6.7 spirit):** when
an N4 corpus arrives, verify against 20 hand-labeled rounds and adjust bins in
one place (`constants.py`) — never inline.

---

### Task 13: RoundScript models + View A beat frames

**Goal:** The pydantic data model for RoundScript v2 and the beat-frame
renderer (spec §6.6 View A): side-grouped, run-length-compressed formation
snapshots at freeze-end + 0/15/…/90 s plus event-anchored beats
(first contact, plant).
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/roundscript/models.py`,
  `src/counterstrat/roundscript/beats.py`, `tests/test_roundscript_beats.py`

**Interfaces:**
- Produces (`models.py` — the single source of truth for later tasks):

```python
class Formation(BaseModel):          # one side at one beat
    zones: list[tuple[int, str]]     # [(count, zone_id)], count-desc order

class Beat(BaseModel):
    label: str                       # "B+00", "B+15", "FC+34", "PL+52"
    t: float                         # clock_s
    t_form: Formation
    ct_form: Formation

class KillEvent(BaseModel):
    t: float; killer: str; victim: str; killer_side: str
    zone: str; weapon: str; headshot: bool; traded_within_4s: bool

class UtilEvent(BaseModel):          # filled by Task 15
    t: float; thrower: str; side: str; nade: str
    from_zone: str; to_zone: str; lineup_id: str | None = None
    blinded: list[tuple[str, float]] = []

class PlantEvent(BaseModel):
    t: float; site: str; planter: str; alive_t: int; alive_ct: int

class MovementLine(BaseModel):       # filled by Task 14
    player: str; side: str; role_hint: str; sentence: str

class RoundScript(BaseModel):
    match_id: str; map_name: str; card_checksum: str; round_num: int
    score_t: int; score_ct: int
    t_team_key: str; ct_team_key: str
    economy: dict[str, "EconSummary"]
    beats: list[Beat]
    kills: list[KillEvent]
    utility: list[UtilEvent]
    plant: PlantEvent | None
    first_contact: KillEvent | None
    winner: str; reason: str; clock_used_s: float
    movements: list[MovementLine] = []

    def to_text(self) -> str: ...    # v2 text (Task 16 orchestrates)
```

- `beats.build_beats(ticks, round_num, first_contact_t, plant_t) ->
  list[Beat]`: samples at `clock_s ∈ {0,15,30,...}` (nearest 4 Hz sample,
  alive players only) plus `FC+t`/`PL+t` anchors; formations sorted
  count-desc then alpha; text form later renders `3×Middle` runs.
- Consumes: Task 3 ticks (columns `last_place_name, team_name, is_alive,
  clock_s, round_num, name`), Task 1 `BEAT_INTERVAL_S`.

- [x] **Step 1: Failing test** (synthetic 10-player frame + fixture):

```python
def test_formation_rle_ordering():
    f = formation_from_zones(["Mid", "Mid", "Mid", "TSpawn", "ARamp"])
    assert f.zones == [(3, "Mid"), (1, "ARamp"), (1, "TSpawn")]


@pytest.mark.demo
def test_beats_round1(anubis_lake):
    ticks = pl.read_parquet(anubis_lake.ticks)
    beats = build_beats(ticks, round_num=1, first_contact_t=None, plant_t=None)
    assert beats[0].label == "B+00"
    total0 = sum(c for c, _ in beats[0].t_form.zones)
    assert total0 == 5                       # all five Ts alive at freeze end
    assert all(b.t <= 120 for b in beats)
```

- [x] **Step 2: fails.** - [x] **Step 3: implement.** - [x] **Step 4: PASS.**
- [x] **Step 5: Commit** `"feat: roundscript models + beat frames"`

**Done when:** deterministic beats for fixture round 1; RLE ordering exact.

---

### Task 14: View B movement sentences

**Goal:** Per-player zone sequences with dwell compression and inline event
marks (spec §6.6 View B): `donk(T): TSpawn > Tunnel(20s) > ~Ruins > BombsiteB k(b1) d`.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/roundscript/movement.py`,
  `tests/test_roundscript_movement.py`

**Interfaces:**
- Produces: `movement_sentences(ticks, kills: list[KillEvent],
  round_num: int, min_dwell_s: float = 2.0) -> list[MovementLine]`.
  Grammar: zones joined by `" > "`; dwell `(Ns)` appended when ≥20 s;
  `~` prefix when >50% of the zone's samples have `is_walking`; marks:
  `k(victim)` per kill in that zone-visit, `d` at death, `p` plant,
  `x` defuse attempt. Zone visits shorter than `min_dwell_s` are merged
  into the neighbor (noise suppression at zone borders).
  `role_hint`: `"lurk"` if the player's max pairwise distance-in-zones from
  teammates (count of beats spent in a zone containing no teammate) ≥ 3,
  else `"pack"` — a heuristic the miner refines later.
- Consumes: Task 13 models, Task 3 ticks/kills tables.

- [x] **Step 1: Failing test** — synthetic walk with a border flicker:

```python
def test_movement_dwell_merge_and_marks():
    ticks = _synthetic_player_walk(          # helper in the test file
        zones=["TSpawn"] * 8 + ["Mid"] * 2 + ["TSpawn"] * 2 + ["Mid"] * 40,
        walking_from=10,
    )
    kills = [KillEvent(t=10.0, killer="p1", victim="e1", killer_side="T",
                       zone="Mid", weapon="ak47", headshot=True,
                       traded_within_4s=False)]
    lines = movement_sentences(ticks, kills, round_num=1)
    s = lines[0].sentence
    assert s.startswith("TSpawn > ")
    assert "TSpawn > Mid > TSpawn" not in s      # flicker merged
    assert "k(e1)" in s
```

- [x] **Step 2–4: implement + PASS.** - [x] **Step 5: Commit**
  `"feat: movement sentences"`

**Done when:** border flickers merge; marks land in the right visit;
fixture demo round renders 10 lines (add a `@pytest.mark.demo` smoke
assert).

---

### Task 15: View C utility grammar + lineup clustering + subdivision gate

**Goal:** Every grenade as a grammar line with throw-origin zone, landing
zone, per-victim blind durations, and corpus-level lineup ids from DBSCAN
over (origin, landing) pairs (spec §6.6 View C, §6.7-4). Also runs the
**subdivision gate** (spec §6.7-2): if distinct utility clusters collapse
into one engine place, flag which places need community subdivision.
**Difficulty:** HARD

**Files:**
- Create: `src/counterstrat/roundscript/utility.py`,
  `tests/test_roundscript_utility.py`

**Interfaces:**
- Produces:
  - `utility_events(lake: LakePaths, mapper: ZoneMapper, round_num) ->
    list[UtilEvent]` — smokes/infernos from their tables (landing = X,Y,Z;
    verified columns `thrower_place, X, Y, Z, start_tick`), flashes/HEs
    from `parse_event`-derived tables persisted in the lake grenades
    parquet (detonation row = last trajectory tick per entity_id);
    `blinded` joined from `player_blind` (add that event table to Task 3's
    extraction in this task — modify `lake/extract.py` AWPY_TABLES note:
    pull via `parser.parse_event("player_blind", other=[...])` into
    `player_blind.parquet`).
  - `cluster_lineups(events: list[UtilEvent], raw_xyz: pl.DataFrame,
    eps: float = 150.0, min_samples: int = 3) -> dict[int, str]` — DBSCAN
    in 6-D (origin xyz + landing xyz, z scaled 2×); cluster name =
    `{landing_zone}-{nade_initial}{ordinal}` (e.g. `Window-S1`); noise →
    `lineup_id=None`. eps is config; the §6.7-4 sweep is a dev harness
    (`python -m counterstrat.roundscript.utility --eps-sweep`, `__main__`
    block in this task) writing silhouette + cluster counts to
    `data/mapcards/<map>/lineup_sweep.json`, runnable whenever a multi-demo
    corpus exists (N4).
  - `subdivision_report(events, mapper) -> dict[str, int]` — engine places
    containing ≥2 distinct smoke-landing clusters (the measured answer to
    "are engine places enough?"; feeds Task 9 overlay work if needed).
- Consumes: Tasks 3, 7, 13.

- [x] **Step 1: Failing tests** — synthetic clusters, no demo needed:

```python
def test_cluster_lineups_two_smokes():
    evs, xyz = _two_synthetic_lineups(n_each=5, separation=800.0)
    names = cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    labelled = {e.lineup_id for e in evs if e.lineup_id}
    assert len(labelled) == 2                 # two distinct lineup ids
    assert all("-" in x for x in labelled)


def test_cluster_lineups_noise_unlabelled():
    evs, xyz = _two_synthetic_lineups(n_each=1, separation=800.0)
    cluster_lineups(evs, xyz, eps=150.0, min_samples=3)
    assert all(e.lineup_id is None for e in evs)
```

plus `@pytest.mark.demo`: fixture demo has 173 smokes / 131 infernos
(verified) → at least one lineup cluster must emerge with
`min_samples=3` on the single demo, and every event's `to_zone` must be a
valid lexicon zone (`lex.is_valid_zone`).

- [x] **Step 2: fails.** - [x] **Step 3: implement** (sklearn DBSCAN;
  deterministic ordinal assignment: clusters sorted by size desc then
  centroid lex order). - [x] **Step 4: PASS.**
- [x] **Step 5: Commit** `"feat: utility grammar + lineup clustering"`

**Done when:** synthetic clustering exact; fixture events all speak lexicon
zones; `subdivision_report` runs and its output is committed to
`data/mapcards/de_anubis/subdivision_report.json` for the analyst.

---

### Task 16: Serializer orchestrator + v1 JSON + grammar linter

**Goal:** `serialize_round` assembles Tasks 12–15 into one `RoundScript`,
renders the v2 text block (all three views) and the v1 JSON, and the
linter enforces the closed lexicon + budget (spec Phase 2b exits).
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/roundscript/serialize.py`,
  `src/counterstrat/roundscript/lint.py`, `tests/test_roundscript_serialize.py`

**Interfaces:**
- Produces:
  - `serialize_match(lake: LakePaths, mapper: ZoneMapper, lex: Lexicon,
    card_checksum: str) -> list[RoundScript]`
  - `RoundScript.to_text()` — exact line format (header/beats/events/end
    as in spec §6.6 View A example; movement sentences appended only when
    `movements` non-empty); `RoundScript.to_json()` = `model_dump_json()`.
  - `lint.lint_script(script: RoundScript, lex: Lexicon) -> list[str]`
    (empty = clean): every zone id in beats/kills/utility/movements ∈
    lexicon; `first_contact` equals earliest kill; token estimate
    `len(to_text())/4 ≤ 450` (spec p95 budget — assert p95 across the
    match, not per round).
  - First contact = earliest `KillEvent`; trade detection: a kill K2 trades
    K1 if `K2.victim == K1.killer` and `K2.t - K1.t <= TRADE_WINDOW_S`.
- Consumes: everything above.

- [x] **Step 1: Failing test**

```python
@pytest.mark.demo
def test_serialize_match_full(anubis_bundle):
    scripts = serialize_match(**anubis_bundle)
    assert len(scripts) == 30
    r1 = scripts[0]
    assert r1.first_contact is not None
    text = r1.to_text()
    assert text.splitlines()[0].startswith("R1 [")
    # determinism (spec §6.2: same demo → same script)
    again = serialize_match(**anubis_bundle)
    assert [s.to_json() for s in scripts] == [s.to_json() for s in again]
    # linter: zero violations across the corpus, p95 token budget
    problems = [p for s in scripts for p in lint_script(s, anubis_bundle["lex"])]
    assert problems == []
```

- [x] **Step 2: fails.** - [x] **Step 3: implement.** Text rendering rules:
  seconds as integers; `B+NN` zero-padded; economy header
  `R{n} [T {buy}(${spend/1000:.1f}k) | CT {buy}(${...}k)] score {t}-{ct}`;
  `END {winner} {reason} @{m:ss}`.
- [x] **Step 4: PASS.** - [x] **Step 5: Commit**
  `"feat: roundscript serializer + linter"`

**Done when:** 30/30 fixture rounds serialize deterministically, lint-clean,
p95 ≤ 450 tokens. **Deferred to N4 corpus (spec Phase 2b):** the 30-round
human validation set and the blinded round-reconstruction test — tracked in
`docs/NEEDS-FROM-YOU.md`, not blocking implementation.

---

### Task 17: Tendency miner (TeamBook)

**Goal:** Deterministic Level-2 profiles (spec §6.2/§6.6 TeamBook): keyed
conditional distributions with `n`, recency weight, and evidence round-ids;
plus per-player role cards from beat occupancy.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/mining/__init__.py`,
  `src/counterstrat/mining/tendencies.py`, `tests/test_mining_tendencies.py`

**Interfaces:**
- Produces: `build_teambook(scripts: list[RoundScript], team_key: str) ->
  TeamBook`:

```python
class TendencyKey(BaseModel):        # spec §6.6 key
    map_name: str; side: str; buy_class: str
    score_bucket: str                # "behind" | "even" | "ahead"
    prev_outcome: str                # "won" | "lost" | "first"

class Tendency(BaseModel):
    key: TendencyKey
    first_contact_zone: dict[str, float]      # P per zone
    opening_formation: dict[str, float]       # P per formation signature
    site_committed: dict[str, float]          # A/B/none
    lineup_sets: dict[str, float]             # frozenset→P, str-rendered
    median_first_contact_s: float | None
    n: int
    evidence: list[str]              # "match_id:round_num" refs

class RoleCard(BaseModel):
    player: str; steamid: str
    modal_zone_fe15: dict[str, str]  # side → modal zone at B+15
    opening_duel_rate: float         # share of team first-contacts involving them
    lurk_rate: float

class TeamBook(BaseModel):
    team_key: str; map_name: str; card_checksum: str
    tendencies: list[Tendency]
    roles: list[RoleCard]
    generated_from: list[str]        # match_ids, date-ordered
    def to_table_text(self) -> str   # compact table for prompts
    def to_sentences(self) -> list[str]  # one templated sentence per entry
```

  Formation signature = View A `B+15` formation rendered canonically
  (`"2×Middle 2×Water 1×TSpawn"`). `score_bucket` from round score diff
  (−: behind, 0: even, +: ahead). Every probability = count/n, no
  smoothing; entries with n < 3 are kept but flagged `low_n=True` (add the
  field) — the dossier prompt tells the model to hedge them (spec risk #2).
- Consumes: Task 16 scripts.

- [x] **Step 1: Failing test** — synthetic scripts with a planted pattern:

```python
def test_miner_finds_planted_tendency():
    scripts = _mk_scripts(                    # helper: 12 T-side full-buys,
        n=12, first_contact_zone="Middle",    # 9 with FC=Middle, 3 =Water
        minority_zone="Water", minority=3,
    )
    tb = build_teambook(scripts, team_key="abc")
    t = _find(tb, side="T", buy_class="full_buy")
    assert abs(t.first_contact_zone["Middle"] - 0.75) < 1e-9
    assert t.n == 12 and len(t.evidence) == 12
    assert all(ev.count(":") == 1 for ev in t.evidence)


def test_miner_deterministic(synthetic_scripts):
    a = build_teambook(synthetic_scripts, "abc").model_dump_json()
    b = build_teambook(synthetic_scripts, "abc").model_dump_json()
    assert a == b
```

- [x] **Step 2–4: implement + PASS** (pure dict math over scripts; polars
  optional). Include `@pytest.mark.demo` smoke: fixture demo yields ≥1
  tendency per side with correct `n` sums (= that team's rounds).
- [x] **Step 5: Commit** `"feat: tendency miner (TeamBook)"`

**Done when:** planted frequencies recovered exactly; identical JSON on
re-run; evidence ids resolve to real scripts. **Deferred to N4:** the
"rediscover 3–5 known tendencies of a famous team" sanity check (spec
Phase 3 exit) — requires a pro corpus.

---

### Task 18: LLM provider abstraction (Anthropic + Gemini, incl. tool use)

**Goal:** One `LLMClient` protocol, two adapters, config-driven model
selection via API keys — the user's stated requirement. Three capabilities:
plain completion, schema-validated JSON (`messages.parse` GA on Anthropic —
verified; `response_schema` on Gemini — verified), and **tool-use chat**
(Anthropic `tools` + `tool_use`/`tool_result` blocks; Gemini function
declarations + function responses — both providers' function calling is
documented; translate through one normalized shape). Prompt caching on the
Map Card system block. No live calls in tests: record/replay JSON fixtures.
**Difficulty:** MEDIUM-HARD

**Files:**
- Create: `src/counterstrat/llm/anthropic_client.py`,
  `src/counterstrat/llm/gemini_client.py`, `src/counterstrat/config.py`,
  `tests/test_llm_clients.py`, `tests/fixtures/llm/*.json`
- Modify: `src/counterstrat/llm/base.py` (protocol stub created by
  Task 11; this task adds `LLMResult`, `LLMBudgetError`, `make_client`)

**Interfaces:**
- Produces (`base.py`):

```python
class LLMResult(BaseModel):
    text: str
    input_tokens: int; output_tokens: int
    cache_read_tokens: int = 0
    model: str; provider: str

class ToolSpec(BaseModel):
    name: str; description: str
    input_schema: dict               # JSON schema (object)

class ToolCall(BaseModel):
    id: str; name: str; arguments: dict

class ChatTurn(BaseModel):
    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: list[ToolCall] = []  # assistant turns only
    tool_call_id: str | None = None  # tool turns only

class LLMClient(Protocol):
    def complete(self, *, system: str, user: str,
                 max_tokens: int = 4096) -> LLMResult: ...
    def complete_json[T: BaseModel](self, *, system: str, user: str,
                 schema: type[T], max_tokens: int = 4096) -> tuple[T, LLMResult]: ...
    def chat(self, *, system: str, turns: list[ChatTurn],
             tools: list[ToolSpec], max_tokens: int = 4096
             ) -> tuple[ChatTurn, LLMResult]: ...
    # chat returns the assistant turn: either .text (final answer) or
    # .tool_calls (caller executes tools, appends tool turns, calls again)

def make_client(cfg: AppConfig) -> LLMClient   # picks provider by cfg
```

- Produces (`config.py`):

```python
class AppConfig(BaseModel):
    provider: Literal["anthropic", "gemini"] = "anthropic"
    # First-run seeds only — the runtime source of truth is the settings
    # UI (user-provided key + user-selected model, Tasks 21/23):
    anthropic_model: str = "claude-sonnet-5"   # verified live on user's key
    gemini_model: str = "gemini-2.5-pro"       # verified live on user's key
    anthropic_api_key: str | None = None   # env ANTHROPIC_API_KEY
    gemini_api_key: str | None = None      # env, first hit wins:
    # GEMINI_API_KEY → GOOGLE_API_KEY → GOOGLE_GENERATIVE_AI_API_KEY
    # (the last is what exists on the dev machine — verified live
    #  2026-09-03, HTTP 200, gemini-2.5-pro + 3-series available)
    max_input_tokens: int = 190_000        # stay under 200k surcharge tier
    data_root: Path = Path("data")
    cs2_install_path: Path | None = None   # optional; future radar task
                                           # reads pak01 in place (N2)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig": ...
    # precedence: data/settings.json (written by the settings endpoint)
    # > env vars (dotenv) > defaults above


# --- Development policy (user directive; never used at app runtime) ---
DEV_GEMINI_MODEL = "gemini-3.8-flash"     # verified available 2026-09-03

def resolve_latest_sonnet(api_key: str) -> str:
    """Newest sonnet-family id from GET /v1/models (created_at desc).

    Used by live-marked tests and dev harnesses (quiz/benchmark) only.
    Today resolves to 'claude-sonnet-5'.
    """
```

- Adapter specifics (verified API shapes, Context section):
  - **Anthropic** (`anthropic` SDK): `client.messages.create(model=...,
    max_tokens=..., system=[{"type":"text","text":system,
    "cache_control":{"type":"ephemeral"}}], messages=[{"role":"user",
    "content":user}])`. `complete_json` uses `client.messages.parse(...,
    output_format=schema)` → `response.parsed_output`. Usage from
    `response.usage` (`input_tokens`, `output_tokens`,
    `cache_read_input_tokens`). System block gets the explicit cache
    breakpoint because the Map Card exceeds the 1,024-token Sonnet minimum.
  - **Gemini** (`google-genai` SDK): `client = genai.Client(api_key=...)`;
    `client.models.generate_content(model=..., contents=user, config={
    "system_instruction": system, "max_output_tokens": max_tokens})`;
    JSON mode adds `"response_mime_type": "application/json",
    "response_schema": schema` and parses `response.text` through the
    pydantic schema (`schema.model_validate_json`). **Note:** confirm the
    exact `system_instruction` config key against the installed
    `google-genai` version's `GenerateContentConfig` at implementation —
    the structured-output config keys are doc-verified, this one is not.
  - **Tool-use translation** (`chat`): Anthropic — `tools=[{name,
    description, input_schema}]`, assistant `tool_use` content blocks →
    `ToolCall`s, tool turns sent back as user-role `tool_result` blocks
    keyed by `tool_use_id`. Gemini — `types.Tool(function_declarations=
    [...])` in config, `function_call` parts → `ToolCall`s, tool turns as
    `function_response` parts. IDs: Gemini calls lack ids — synthesize
    `f"{name}:{index}"`. Verify exact google-genai types
    (`types.Tool`, `types.FunctionDeclaration`, `types.Part.
    from_function_response`) against the installed SDK at implementation.
  - Both: retry 429/5xx with exponential backoff (3 tries), raise
    `LLMBudgetError` if a pre-count estimate (`len(system+user)/3.5`)
    exceeds `cfg.max_input_tokens` — never silently spill into the
    long-context price tier.
- Record/replay: each adapter takes `transport: Callable | None`; tests
  inject fakes returning canned SDK-shaped objects (fixtures under
  `tests/fixtures/llm/`). A `@pytest.mark.live` test per adapter does one
  tiny real call (skipped without keys).

- [x] **Step 1: Failing tests**

```python
def test_make_client_provider_switch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = AppConfig.load()
    assert make_client(cfg).__class__.__name__ == "AnthropicClient"
    cfg2 = cfg.model_copy(update={"provider": "gemini", "gemini_api_key": "k"})
    assert make_client(cfg2).__class__.__name__ == "GeminiClient"


def test_missing_key_raises():
    cfg = AppConfig(provider="gemini", gemini_api_key=None)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make_client(cfg)


def test_anthropic_complete_json_replay(fake_anthropic_transport):
    client = AnthropicClient(api_key="k", model="m",
                             transport=fake_anthropic_transport)
    class Out(BaseModel):
        site: str
    out, usage = client.complete_json(system="s", user="u", schema=Out)
    assert out.site == "A" and usage.provider == "anthropic"


def test_budget_guard():
    client = AnthropicClient(api_key="k", model="m", transport=lambda *a: None)
    with pytest.raises(LLMBudgetError):
        client.complete(system="x" * 3_000_000, user="u")
```

(mirror the replay test for Gemini), plus a tool-round-trip replay test per
adapter:

```python
def test_anthropic_chat_tool_roundtrip(fake_anthropic_tool_transport):
    client = AnthropicClient(api_key="k", model="m",
                             transport=fake_anthropic_tool_transport)
    spec = ToolSpec(name="get_tendency", description="d",
                    input_schema={"type": "object", "properties": {}})
    turn, _ = client.chat(system="s",
                          turns=[ChatTurn(role="user", text="q")],
                          tools=[spec])
    assert turn.tool_calls and turn.tool_calls[0].name == "get_tendency"
    # feed the tool result back; fake transport then returns a final answer
    turns = [ChatTurn(role="user", text="q"), turn,
             ChatTurn(role="tool", tool_call_id=turn.tool_calls[0].id,
                      text='{"rows": []}')]
    final, _ = client.chat(system="s", turns=turns, tools=[spec])
    assert final.text and not final.tool_calls
```

- [x] **Step 2: fails.** - [x] **Step 3: implement both adapters + config.**
- [x] **Step 4: PASS offline suite; run `-m live` once if keys present and
  paste real usage numbers into the commit message. Live tests select
  models per the dev policy: `resolve_latest_sonnet()` for Anthropic,
  `DEV_GEMINI_MODEL` for Gemini — never the runtime config values.**
  (offline suite PASS: 14 replay/guard tests; live tests written and
  `-m live`-marked, not executed — no keys in this environment)
- [x] **Step 5: Commit** `"feat: LLM provider abstraction (anthropic+gemini)"`

**Done when:** provider switch by config/env verified; JSON mode returns
validated pydantic objects on both adapters via replay; tool calls
round-trip on both adapters via replay; budget guard trips.

---

### Task 19: Dossier generator + anti-hallucination gate

**Goal:** Spec Phase 4 mode 1: system prompt = Map Card (cached) + output
contract; user turn = TeamBook + exemplar RoundScripts; output = dossier
with the spec's section contract; post-generation lint rejects fabricated
evidence.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/llm/prompts.py`, `src/counterstrat/llm/dossier.py`,
  `tests/test_llm_dossier.py`

**Interfaces:**
- Produces:
  - `prompts.build_system(card_yaml: str) -> str` — card + doctrine-free
    output contract (spec §7 Phase 4 sections: Identity; Defaults & roles;
    Execute repertoire w/ counters; Economy policy w/ exploit;
    Player-specific weaknesses; Round-state playbook table;
    Confidence/evidence appendix) + hard rules: *only lexicon zone ids; cite
    `match_id:round` for every claim; quote frequencies verbatim from the
    TeamBook; hedge any `low_n` tendency.*
  - `prompts.build_user(teambook: TeamBook, exemplars: list[RoundScript])
    -> str` — TeamBook table text + sentences + `to_text()` of k exemplars
    selected by tendency coverage: for each of the top-6 tendencies by `n`,
    include its 2 most recent evidence rounds (dedup, cap 12 — spec §6.2 L3
    budget).
  - `dossier.generate(client, card, teambook, scripts) -> Dossier` with
    `Dossier(text: str, lint: DossierLint, usage: LLMResult)`;
    `DossierLint(unknown_zones: list[str], bad_citations: list[str],
    freq_mismatches: list[str], ok: bool)`:
    - zone check: scan for `Zone:`-tagged tokens? No — simpler and robust:
      require the model to wrap zone ids in backticks (contract), lint =
      every backtick token that case-matches `[A-Z][A-Za-z]+` must be in
      lexicon ∪ player names ∪ lineup ids.
    - citations: every `match_id:round` regex hit must exist in the input
      evidence set.
    - frequencies: every `NN%` adjacent to a tendency name must match the
      TeamBook within ±1pp (regex window of 40 chars).
    `ok=False` → `generate` retries once with lint failures appended to the
    user turn, then returns flagged.
- Consumes: Tasks 10, 16, 17, 18.

- [x] **Step 1: Failing tests** — all offline with a `ScriptedClient`:

```python
def test_dossier_lint_catches_fabrication(anubis_card, synthetic_teambook):
    good = "…cites `Middle` (evidence deadbeefcafe:7) 75%…"
    bad = "…cites `Ghost` (evidence ffffffffffff:99) 12%…"
    lint_g = lint_dossier(good, synthetic_teambook, lexicon, evidence_ids)
    lint_b = lint_dossier(bad, synthetic_teambook, lexicon, evidence_ids)
    assert lint_g.ok
    assert not lint_b.ok and "Ghost" in lint_b.unknown_zones
    assert "ffffffffffff:99" in lint_b.bad_citations


def test_exemplar_selection_covers_top_tendencies(synthetic_teambook, scripts):
    ex = select_exemplars(synthetic_teambook, scripts, cap=12)
    assert len(ex) <= 12
    top = max(synthetic_teambook.tendencies, key=lambda t: t.n)
    assert any(f"{s.match_id}:{s.round_num}" in top.evidence for s in ex)
```

- [x] **Step 2–4: implement + PASS.** Include one `@pytest.mark.live`
  end-to-end dossier on the fixture demo's team (single-demo TeamBook —
  content will be thin; the test asserts only lint-clean + all sections
  present; models per dev policy).
- [x] **Step 5: Commit** `"feat: dossier generation + lint gate"`

**Done when:** lint provably catches planted fabrications; exemplar
selection is coverage-driven; live run (when keys exist) produces a
lint-clean dossier. **Deferred to N4:** the two-pro-team expert review
exit (spec Phase 4) — needs real corpus + reviewers.

---

### Task 20: Per-round predictor + evaluation benchmark

**Goal:** Spec Phase 5: predict round k+1's `{buy_type, first_contact_zone,
site, execute_vs_default, fast_vs_slow}` from the TeamBook (mined on rounds
1..k of *other* matches, time-ordered) + live match state; score against
baselines. This is the falsifiable core and the ablation vehicle.
**Difficulty:** HARD

**Files:**
- Create: `src/counterstrat/llm/predict.py`,
  `src/counterstrat/eval/__init__.py`,
  `src/counterstrat/eval/benchmark.py`, `tests/test_eval_benchmark.py`

**Interfaces:**
- Produces:
  - `predict.PredictedRound(BaseModel)`: `buy_type:
    Literal["full_eco","semi_eco","semi_buy","full_buy"]`,
    `first_contact_zone: str`, `site: Literal["A","B","none"]`,
    `execute: bool`, `fast: bool` (first contact < 25 s), plus
    `p_site: dict[str, float]` for Brier scoring.
  - `predict.predict_round(client, card, teambook, match_state) ->
    PredictedRound` where `MatchState(score_t, score_ct, side,
    prev_round_summaries: list[str], economy_estimate)` — uses
    `complete_json` (both providers, structured).
  - `benchmark.split(corpus) -> (train_matches, heldout_matches)` strictly
    time-ordered by demo registration date (spec risk #4).
  - `benchmark.run(cfg, arms: list[str]) -> BenchReport` with per-head
    accuracy/F1, Brier for `p_site`, and baselines:
    (a) `majority` — global modal class per head;
    (b) `freq_table` — argmax from the TeamBook tendency matching the
    round's key (no LLM);
    (c) `llm_no_profile` — LLM with raw scripts, no TeamBook;
    (d) `full` — TeamBook + card. Ablation arms toggle card
    none/lexicon-only/full and encoding v1-JSON/v2-text (spec Phase 5
    list); each arm is a config value, not a code fork.
  - Ground truth per held-out round extracted from its own RoundScript
    (buy_type from economy; site from plant or `none`; execute = any
    lineup-clustered util set fired within 10 s window before site
    commit — define `execute` deterministically in code:
    ≥2 same-round `UtilEvent`s sharing a `lineup_id` prefix landing within
    8 s AND first_contact zone within the committed site's tag group;
    fast = `first_contact.t < 25`).
- Consumes: Tasks 16–18.

- [x] **Step 1: Failing tests** — offline, scripted client:

```python
def test_ground_truth_labels(synthetic_scripts):
    labels = extract_labels(synthetic_scripts[0])
    assert labels.buy_type in BUY_BINS
    assert labels.site in {"A", "B", "none"}


def test_freq_table_baseline_beats_majority_on_planted(synthetic_corpus):
    rep = run_offline_arms(synthetic_corpus, arms=["majority", "freq_table"])
    assert rep.heads["first_contact_zone"]["freq_table"].accuracy > \
           rep.heads["first_contact_zone"]["majority"].accuracy


def test_time_ordered_split(synthetic_corpus):
    train, held = split(synthetic_corpus)
    assert max(m.registered_at for m in train) <= \
           min(m.registered_at for m in held)
```

- [x] **Step 2–4: implement + PASS offline arms.** LLM arms run only via
  `python -m counterstrat.eval.benchmark` with keys (models per dev policy;
  writes `data/eval/report.json`; per-arm cost logged from `LLMResult`
  usage). This is a research harness, not a product surface.
- [x] **Step 5: Commit** `"feat: per-round predictor + eval benchmark"`

**Done when:** offline arms + labels + split proven on synthetic corpus;
LLM arms wired and runnable. **Deferred to N4:** the ≥300-held-out-round
significance run (spec Phase 5 exit) — impossible with one demo; the
harness must accept any corpus size and report `n` honestly.

---

### Task 21: Web backend — upload + background ingest + catalog API

**Goal:** The FastAPI service: demo upload (incl. FACEIT `.dem.zst`),
background ingest job chaining the whole offline pipeline
(register → extract → card-if-missing → serialize → mine), and the catalog
endpoints the frontend browses. No business logic here — thin orchestration
of Tasks 2–17 functions.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/web/__init__.py`, `src/counterstrat/web/app.py`,
  `src/counterstrat/web/ingest.py`, `src/counterstrat/web/routes.py`,
  `tests/test_web_api.py`

**Interfaces:**
- `app.create_app(cfg: AppConfig) -> FastAPI`; `app.main()` runs
  `uvicorn.run(create_app(AppConfig.load()), host="127.0.0.1", port=8710)`.
- Endpoints (all JSON, prefix `/api`):
  - `POST /api/demos` — multipart file field `demo`. Accepts `.dem`,
    `.dem.zst`, `.dem.gz`, `.dem.bz2`; decompresses to
    `data/uploads/<orig_stem>.dem` (zstandard/gzip/bz2 by suffix);
    rejects other suffixes and non-`PBDEMS2` magic (first 8 bytes) with
    400. Returns `{job_id, filename}` and schedules
    `ingest.run_ingest(job_id, path, cfg)` via `BackgroundTasks`.
  - `GET /api/jobs/{job_id}` — `JobState(job_id, stage:
    Literal["queued","extracting","mapcard","serializing","mining",
    "done","error"], match_id: str | None, map_name: str | None,
    detail: str)` read from `data/jobs/<job_id>.json` (written atomically
    at each stage transition — survives restarts).
  - `GET /api/demos` — corpus manifest records + per-demo team keys.
  - `GET /api/teams` — `[{team_key, names: [clan names seen], maps:
    [...], demos: int, rounds: int}]` aggregated from lake rosters.
  - `GET /api/teams/{team_key}/{map_name}/teambook` — TeamBook JSON
    (404 if not mined).
  - `GET /api/reports/{team_key}/{map_name}` — dossier markdown; generates
    via Task 19 on first request (needs keys; 503 with a clear message when
    no provider configured), caches to
    `data/teambooks/<team_key>/<map>/dossier.md`.
  - `GET /api/settings` — `{provider, model, keys_present: {anthropic:
    bool, gemini: bool}}`; **never returns key material**.
  - `POST /api/settings {provider, model, api_key?}` — persists to
    `data/settings.json` (plaintext is acceptable: localhost, single user,
    gitignored; key optional — env fallback). New chat sessions pick it up
    immediately; `AppConfig.load()` reads it first (see Task 18 precedence).
  - `GET /api/models` — live model-id list fetched from the active
    provider's models endpoint (populates the UI picker; 5-min in-memory
    cache; 502 with detail on upstream failure).
- `ingest.run_ingest`: per stage, update job file; on exception write
  `stage="error", detail=str(exc)` — never a silent hang. Map Card step:
  if `data/mapcards/<map>/card.yaml` missing and `maps/<map>/<map>.vpk`
  exists → compile (Tasks 5–10); if VPK missing → proceed without card,
  `detail="card missing: supply maps/<map>/<map>.vpk"` (chat still works,
  degraded: no topology/rotate answers — the system prompt says so).
  Mining runs for **both** team_keys found in the demo.

- [x] **Step 1: Failing tests** (httpx `TestClient`):

```python
def test_upload_rejects_garbage(client_app):
    r = client_app.post("/api/demos",
                        files={"demo": ("x.dem", b"NOTADEMO" * 4)})
    assert r.status_code == 400


def test_job_unknown_404(client_app):
    assert client_app.get("/api/jobs/nope").status_code == 404


@pytest.mark.demo
def test_upload_ingest_end_to_end(client_app, demo_path):
    with demo_path.open("rb") as f:
        r = client_app.post("/api/demos",
                            files={"demo": (demo_path.name, f)})
    job = r.json()["job_id"]
    state = _poll_until_done(client_app, job, timeout_s=180)
    assert state["stage"] == "done" and state["map_name"] == "de_anubis"
    teams = client_app.get("/api/teams").json()
    assert len(teams) == 2 and all(t["rounds"] > 0 for t in teams)
    tb = client_app.get(
        f"/api/teams/{teams[0]['team_key']}/de_anubis/teambook")
    assert tb.status_code == 200


def test_settings_roundtrip_never_echoes_key(client_app):
    r = client_app.post("/api/settings", json={
        "provider": "gemini", "model": "gemini-2.5-pro", "api_key": "sec"})
    assert r.status_code == 200
    got = client_app.get("/api/settings").json()
    assert got["provider"] == "gemini" and got["model"] == "gemini-2.5-pro"
    assert "sec" not in json.dumps(got)
    assert got["keys_present"]["gemini"] is True
```

(`TestClient` runs BackgroundTasks synchronously on response completion —
`_poll_until_done` is then a bounded loop reading the job file.)

- [x] **Step 2: fails.** - [x] **Step 3: implement.**
- [x] **Step 4: PASS.** - [x] **Step 5: Commit** `"feat: web backend + ingest"`

**Done when:** uploading the fixture demo through HTTP produces lake +
scripts + two TeamBooks and a `done` job; garbage uploads 400; job state
survives process restart (read from disk).

---

### Task 22: Analyst tools + agent loop + chat API

**Goal:** The product's core interaction (spec Phase 4 mode 2): a chat
session scoped to (team, map) where the model answers from **tools** —
TeamBook lookups, round-script fetches, guarded SQL over the lake — with
every reply soft-linted for fabricated zones/citations. "What's their
default CT setup on full buys?" must resolve to mined data, not vibes.
**Difficulty:** HARD

**Files:**
- Create: `src/counterstrat/llm/tools.py`, `src/counterstrat/llm/agent.py`,
  `src/counterstrat/web/chat.py`, `tests/test_llm_agent.py`,
  `tests/test_web_chat.py`

**Interfaces:**
- `tools.py`:

```python
class SessionContext(BaseModel, arbitrary_types_allowed=True):
    team_key: str; map_name: str
    teambook: TeamBook; lexicon: Lexicon; card: MapCard
    con: Any                          # duckdb connection (lake views)
    scripts: dict[str, RoundScript]   # "match_id:round" → script

def tool_specs() -> list[ToolSpec]    # the five below, JSON-schema'd
def execute_tool(ctx: SessionContext, call: ToolCall) -> str  # JSON string
```

  Tools (names are the contract; keep them):
  - `get_tendencies(side, buy_class=None)` → matching `Tendency` rows as
    compact table text + `n` + `low_n` flags.
  - `list_rounds(side=None, buy_class=None, won=None, site=None)` →
    `[{id: "match:round", summary: header line of the script}]`.
  - `get_round_script(id)` → full v2 text (the model quotes beats from it).
  - `get_role_cards()` → TeamBook.roles rendered.
  - `sql_query(query)` → guarded DuckDB: regex `^select\b` (case-fold),
    reject `;`, `attach`, `copy`, `pragma`, `install`; wrap as
    `select * from (<q>) limit 50`; return rows as JSON. Errors return as
    `{"error": ...}` strings — never exceptions into the loop.
- `agent.py`:

```python
class AgentReply(BaseModel):
    text: str
    tool_trace: list[dict]            # {name, arguments, result_preview}
    warnings: list[str]               # soft lint findings
    usage: LLMResult                  # summed across iterations

def run_agent(client: LLMClient, system: str, history: list[ChatTurn],
              ctx: SessionContext, max_iters: int = 8) -> AgentReply
```

  Loop: call `client.chat`; while `tool_calls`, execute via
  `execute_tool`, append assistant turn + tool turns, re-call. At
  `max_iters`, append a user turn "answer now from what you have" and take
  the text. Final text → Task 19's linter in **soft mode** (warnings, not
  retry). Sum usage across iterations.
- `web/chat.py` endpoints:
  - `POST /api/chat/sessions {team_key, map_name}` → `{session_id}`;
    builds `SessionContext` (loads teambook/card/scripts, opens duck con);
    system prompt = `prompts.build_system(card_yaml)` + TeamBook table
    (both in the cached system block) + tool-usage rules (*always check a
    tendency or round before asserting; state `n`; hedge `low_n`*).
  - `POST /api/chat/sessions/{sid}/messages {text}` → `AgentReply` JSON;
    appends both turns to `data/chats/<sid>.jsonl`.
  - `GET /api/chat/sessions/{sid}` → transcript (for page reload).
  - Sessions: in-memory dict + on-disk transcript; recreating a session
    after restart replays the jsonl into `history`.

- [ ] **Step 1: Failing tests** — all offline via scripted clients:

```python
def test_agent_executes_tool_then_answers(session_ctx):
    scripted = ScriptedToolClient([
        _turn_with_tool_call("get_tendencies", {"side": "T"}),
        _final_turn("They hit B on 75% of full buys (n=12)."),
    ])
    reply = run_agent(scripted, "sys", [ChatTurn(role="user",
                      text="what do they do on T full buys?")], session_ctx)
    assert reply.tool_trace[0]["name"] == "get_tendencies"
    assert "75%" in reply.text


def test_sql_tool_guard(session_ctx):
    bad = execute_tool(session_ctx, ToolCall(
        id="1", name="sql_query",
        arguments={"query": "insert into rounds values (1)"}))
    assert "error" in bad
    ok = execute_tool(session_ctx, ToolCall(
        id="2", name="sql_query",
        arguments={"query": "select count(*) as n from rounds"}))
    assert '"n"' in ok


def test_agent_iteration_cap(session_ctx):
    looping = ScriptedToolClient([_turn_with_tool_call("get_role_cards", {})] * 20)
    reply = run_agent(looping, "sys", [_user("q")], session_ctx, max_iters=3)
    assert reply.text                      # forced answer, no infinite loop
```

  plus `test_web_chat.py`: create session → post message with the scripted
  client injected (app factory takes `client_factory` for tests) → reply
  JSON has `text`/`warnings`; transcript file exists; second message sees
  history; unknown session 404.

- [ ] **Step 2: fails.** - [ ] **Step 3: implement.**
- [ ] **Step 4: PASS; one `@pytest.mark.live` conversation on the fixture
  team ("what is their default CT setup on full buys?") asserting the reply
  cites at least one `match_id:round` and zero lint warnings — run once per
  provider, models per dev policy.**
- [ ] **Step 5: Commit** `"feat: analyst chat agent + API"`

**Done when:** scripted-loop tests green on both adapters' shared loop; SQL
guard provably read-only; live conversation (keys present) answers the
user's canonical questions with tool-grounded, cited claims.

---

### Task 23: Frontend (no-build static UI)

**Goal:** The user-facing app: drag-drop demo upload with job progress, a
team/map picker, the chat pane, and dossier download. Vanilla HTML/JS/CSS
served by FastAPI — no node toolchain, no framework.
**Difficulty:** MEDIUM

**Files:**
- Create: `src/counterstrat/web/static/index.html`,
  `src/counterstrat/web/static/app.js`,
  `src/counterstrat/web/static/style.css`
- Modify: `src/counterstrat/web/app.py` (mount `StaticFiles`, `/` →
  `index.html`)
- Test: `tests/test_web_static.py`

**Behaviour contract (app.js, ~250 lines, fetch-based):**
- Upload zone: POST to `/api/demos`, then poll `/api/jobs/{id}` every 2 s
  rendering the stage; on `done`, refresh the team list; on `error`, show
  `detail` verbatim.
- Sidebar: `GET /api/teams` → clickable (team, map) pairs showing demo and
  round counts (the `n` honesty surface).
- Chat: create session on pick; textarea + send; render user/assistant
  bubbles; assistant `tool_trace` collapsible ("checked: get_tendencies(T,
  full_buy)"); `warnings` as an amber badge listing lint findings;
  round-script quotes rendered in `<pre>` (detect fenced blocks).
- Dossier button: `GET /api/reports/...` → download as `.md`; disabled
  with tooltip when 503 (no API key).
- Settings panel: provider toggle (Anthropic/Gemini), model dropdown
  populated from `GET /api/models`, API-key field (password input,
  write-only — posted, never read back; `keys_present` drives a "key set ✓"
  indicator). Saves via `POST /api/settings`.
- No streaming in v1 (tool loops make it complex); a spinner + "thinking,
  N tool calls so far" via short-poll on a lightweight
  `GET /api/chat/sessions/{sid}/pending` counter is acceptable if trivial,
  else spinner only.
- Style: single dark theme stylesheet, system fonts, monospace for scripts;
  no CSS framework.

- [ ] **Step 1: Failing test** — `GET /` returns 200 html containing
  `app.js`; `GET /static/app.js` 200; smoke-parse: response text contains
  the four endpoint paths (cheap contract that JS targets the real API).
- [ ] **Step 2: fails.** - [ ] **Step 3: implement.** - [ ] **Step 4: PASS +
  manual browser pass with the webapp-testing skill (Playwright) over the
  fixture flow: upload → job done → pick team → ask the canonical buy-round
  question (scripted client injectable via `?mock=1` app flag for offline
  UI testing).**
- [ ] **Step 5: Commit** `"feat: web UI"`

**Done when:** `uv run counterstrat` → browser at `127.0.0.1:8710` walks
upload→chat on the fixture demo without touching a terminal again; UI
surfaces `n`, tool trace, and lint warnings rather than hiding them.

---

### Task 24: Runbook + needs-file refresh

**Goal:** Operating docs so the user can run the product and knows exactly
what remains gated on their inputs.
**Difficulty:** EASY

**Files:**
- Create: `README.md` — quickstart: `uv sync`; vendoring VRF CLI (URL +
  sha256 from Context); `.env` with either provider key; `uv run
  counterstrat`; browser walkthrough (upload the fixture demo → chat);
  where map VPKs go and when to refresh them (N3); how the research
  harnesses run (`python -m counterstrat.mapcard.quiz`,
  `python -m counterstrat.eval.benchmark`).
- Modify: `docs/NEEDS-FROM-YOU.md` — mark which N-items remain open at
  completion time.

- [ ] **Step 1: write README; Step 2: execute every command in it verbatim
  on a clean checkout and paste real output snippets; Step 3: commit**
  `"docs: runbook"`.

**Done when:** a fresh clone + the README reaches a working chat session
about the fixture demo without reading this plan.

---

## Risks

1. **Nav v36 reader stalls (Task 8).** Mitigated by design: the
   demo-transition graph is a full substitute for topology/rotates (and is
   *better* calibrated, spec §6.5.1 bucket C); the card records its source.
2. **Parser breakage on CS2 updates** (spec risk #1). Pins + the Task 6
   reconciliation gate turn silent drift into hard failures; raw demos and
   parquet are both archived.
3. **Single-demo development corpus.** Several spec exit criteria (human
   validation, expert review, ≥300-round eval) are corpus-gated; this plan
   ships the machinery + honest `n` reporting, with the gates tracked in
   `docs/NEEDS-FROM-YOU.md`. Risk: thresholds (buy bins, DBSCAN eps, beat
   rate) tuned on one PUG demo may shift on pro corpora — all are single-
   point config constants by construction.
4. **Provider API drift.** Model IDs and the one unverified SDK key
   (`system_instruction` in google-genai config) are isolated in
   `config.py`/adapters; the `live`-marked smoke tests catch drift in
   minutes. Anthropic structured outputs + caching and Gemini
   response_schema are doc-verified as of 2026-09-03.
5. **KNN zone mapper edge error** near boundaries/verticality (Anubis has
   Water/Bridge stacking). Bounded by the ≥97% holdout gate; if grenade
   landing zones look wrong in Task 15's demo assertions, raise `z_scale`
   or fall back to per-zone z-band filters before touching mesh extraction.
6. **Token-budget creep** in dossier inputs as corpora grow. The budget
   guard (Task 18) fails closed under the 200k surcharge line; exemplar
   selection is capped by construction.
7. **LLM zone hallucination.** Triple defence: closed lexicon in the
   contract, backtick lint, structured outputs for the predictor (enum'd
   heads); dossier retry-once-then-flag, chat soft-warnings surfaced in the
   UI rather than hidden.
8. **Agent tool-loop runaway cost.** Hard `max_iters=8`, per-session usage
   summing in `AgentReply.usage`, budget guard on every call; tool results
   truncated to 50 rows / previews in the trace.
9. **Background-job fragility (single process, no queue).** Deliberate
   YAGNI for a localhost single-analyst app: job state is atomic-on-disk
   and re-readable after restart; an interrupted job re-runs by re-uploading
   (idempotent via content hash). Revisit only if multi-user hosting ever
   becomes a requirement (spec Phase 6).
10. **Uploads are large** (100–400 MB raw; ~40–60% smaller zst — spec §1).
   Localhost keeps this cheap; decompression is streamed to disk, never
   into memory; reject non-`PBDEMS2` payloads before parsing.

## Self-review notes

- Spec coverage: Phases 0–5 → Tasks 2–3 (P0/P1), 5–11 (P2a), 12–16 (P2b),
  17 (P3), 18–19 + 22 (P4: dossier = mode 1, chat agent + tools = mode 2),
  20 (P5), 21/23/24 (product surface + glue). Spec §6.7 unknowns: 1 =
  settled (Context), 2 = Task 15 gate, 3 = Task 16 budget assert + Task 20
  ablation arm, 4 = Task 15 eps-sweep, 5 = Task 8 transitions, 6 = Task 11.
  Doctrine block, sightlines, voice: explicitly deferred (Assumptions 4–5,
  spec §6.5 optional blocks). Text-to-SQL drill-down (spec L3 mode 2) =
  `sql_query` tool, Task 22.
- Type consistency pass done: `DemoRecord/LakePaths/PlaceVolume/Lexicon/
  ZoneGraph/MapCard/RoundScript/TeamBook/LLMClient/ToolSpec/ToolCall/
  ChatTurn/SessionContext/AgentReply/JobState/AppConfig` names match across
  all task Interfaces blocks.
- No placeholder patterns remain; every code step is concrete or names its
  exact verified source (nav v36 port cites its two reference files).

PLAN COMPLETE
