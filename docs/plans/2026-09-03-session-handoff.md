# Session Handoff - 2026-09-03

State dump for continuing in a fresh session. Everything below was landed on
`master` today; the working tree is clean apart from `scratch/` debris and an
untracked `map.json` (kopipasta leftover, deletion blocked for agents).

## Where the product stands

One day of work, 22 commits, 214 -> 304 tests (`uv run python -m pytest -q`,
6 live-marked deselected; lint: `uv run ruff check .`). The user runs the
server themselves (`uv run counterstrat`, port 8710); agents must remind them
to RESTART it after commits - a stale server has burned API calls on old code
once already.

| Commit | What |
|---|---|
| 225f90c | Task 0: teambooks accumulate across demos of one (team, map) |
| 44b395e | Hierarchical tendencies + signal flags (kills 33%/33%/33% noise) |
| 5d3489f | Utility book miner |
| 51ef02e | Gap/vacancy miner with triggers + lift |
| dfe8ee0 | Economy policy miner |
| 9d2890c | Extended player profiles on RoleCard |
| ace4e37 | Five situational analyst tools (get_playbook, gaps, utility, econ, profile) |
| 8c46c94 | IGL-grade chat prompt (Read/Evidence/Counter-call/Confidence, <=180 words) |
| 3d5a236 | Deterministic First Look scout brief + UI panel |
| 8a65591 | Radar v2: zoom/pan, player tracing, utility glyphs, tooltips |
| 24c1d11 | Dossier: Gaps & Triggers section, grounded in mined artifacts |
| 85b5cb1 | Gap post-plant tautology fix (planted site only) |
| f6de044 | Radar trails span both halves; Reset clears filters; solo = full match |
| 0ba9249 | AI First Read (LLM insights over the full corpus) + coverage line |
| 1774129 | Team identity clustering (>=3/5 steamid overlap) + single-match scope |
| 98132e5 | Gemini thinking-token truncation fix + LLMResult.truncated |
| 2db0e55 | Uncapped output by default (None = model max; Anthropic default 32k) |
| 752750f | Structured human-readable First Read (6 sections) + scrollable panel |
| 7b6ae89 | Demo deletion from the sidebar + full artifact rebuild |
| c2f8cb0 | User callouts with single-vocabulary LLM boundary (Renamer) |
| 698235a | Callout editor covers all VPK maps + nuke upper/lower levels |
| 9fffc1a | Callout label positions: tick centroids primary again (regression fix) |

## Architecture added today (key files)

- **Team identity**: `src/counterstrat/teams.py` - lineups cluster by >=3/5
  shared steamids (union-find), persisted to `data/teams.json`. Everything
  resolves lineup keys -> cluster id; scripts are re-keyed to the canonical id
  at load (`web/ingest.py:_scripts_for_keys`, `web/routes.py:_load_team_bundle`,
  `web/chat.py:_build_session`). Cluster ids can CHANGE when the smallest
  lineup's matches are deleted - `web/maintenance.py:rebuild_artifacts` handles
  that (recluster, re-mine, prune dead teambook dirs).
- **Mining**: `mining/tendencies.py` (3 aggregation levels, `signal` flag,
  `iter_round_states` shared by all miners), `mining/utility_book.py`,
  `mining/gaps.py` (beat-window vacancies x triggers, post-PL = planted site
  only), `mining/econ_policy.py`, `mining/brief.py` (deterministic scout brief).
- **LLM**: `llm/insights.py` (AI First Read: six mandated sections - T/CT
  Pistol, Eco & Force, T/CT Full Buy, Gotchas; human language, Game N labels,
  friendly-citation lint); `llm/tools.py` (10 tools; `SessionContext.renamer`);
  `llm/prompts.py:build_chat_system` (doctrine + answer contract + Data
  coverage line). Output is UNCAPPED by default (`max_tokens=None`); Gemini
  adds THINKING_HEADROOM when a cap is given; `LLMResult.truncated` flags
  provider-ceiling cuts.
- **Callouts / single vocabulary**: `src/counterstrat/aliases.py` - aliases in
  `data/mapcards/<map>/aliases.json`; `Renamer.rename_text` applies user names
  at every outbound surface (chat system, tool results, brief, insights,
  dossier), `unalias_sql` maps quoted literals back for the lake; the lint
  vocabulary flips (canonical-with-alias is rejected). Insight caches carry
  `alias_fp` for invalidation. Editor: Callouts tab (`static/callouts.js`),
  zones+anchors from `GET /api/maps/{map}/callouts`, saves via
  `PUT /api/maps/{map}/aliases`.
- **Anchors**: lake-tick centroids primary; `env_cs_place` volume origins
  (VRF-extracted to `data/tmp_assets/<map>`) fill gaps so never-ingested VPK
  maps are fully labeled. Levels classify by Z vs
  `calibration.lower_altitude_max` (nuke: BombsiteB/Decon/Observation/Tunnels/
  Vents on lower - verified on real data).

> **Update (same day, later session):** open issues 1-3 below are DONE
> (commits 9ec6011 anchors, 323fcdb zone map in prompts, 12b6989 reset
> buttons; 308 tests green). Anchors now: per-axis median snapped to the
> closest real tick, on the zone's dominant level (lower only when >=60%
> of ticks are below). Verified on real data via
> `scratch/check_anchor_accuracy.py`: 75/75 zones across
> ancient/anubis/nuke anchor on own-zone ground; nuke lower set unchanged.
> Every LLM prompt (chat/dossier/First Read) now embeds
> `format_zone_map(map_zone_anchors(...))` - the editor's exact source -
> renamed to user callouts at the existing boundary. Item 4 debts remain.
>
> **Update 2 (same day): custom callout zones landed** (plan
> `docs/plans/2026-09-03-custom-callout-zones.md`, commits 5bb0f66..adb4647,
> 320 tests green). Users point at the radar (+ Add callout) and name a
> spot; the zone is a world-space sphere in `data/mapcards/<map>/zones.json`
> (`customzones.py`), grounded by nearby tick Z, validated against game
> zones/aliases/overlaps. Saving triggers a background job
> (`rebuild_map_zones`): ticks re-zoned in place (`last_place_name` =
> effective, `place_default` preserves game names - idempotent, reversible),
> card recompiled (vocabulary unions VPK places + effective + default tick
> places), scripts re-serialized, teambooks re-mined, per-map
> `dossier.md`/`insights.json` deleted. duckdb views now
> `union_by_name=true`. Custom zones anchor at the USER'S click point in
> both the editor and `format_zone_map`. Measured rebuild: 5.1s per match
> (real anubis demo, e2e test prints it).

## OPEN ISSUES - pick up here

1. **Callouts reset buttons (user-requested, not built)**
   - Per-zone "reset to default" plus a "Reset all" with confirm.
   - Plumbing exists: PUT with empty string removes an alias; reset-all = PUT
     `{aliases: {zone: "" for all}}`. Pure `static/callouts.js` + a style
     tweak + static-pin test. Small task.
2. **Callout position accuracy (user: "must be correct always")**
   - Regression fixed in 9fffc1a (tick centroids primary). Remaining work:
     - Mean tick position can still sit off-center for L-shaped zones; a
       density-weighted MEDIAN (per-axis) would be sturdier. One-line change
       in `routes.py:_tick_positions` (`.median()` instead of `.mean()`),
       verify visually on ancient/anubis/nuke.
     - Volume-origin fallback is an entity pivot, not a box center; fine for
       unplayed maps, but if precision matters there too, nav-mesh area
       centroids could be matched to places (nav36 currently parses areas
       without place names - would need the place-table part of the format).
   - User acceptance: labels visually on-zone across all 10 maps, both nuke
     levels.
3. **"The LLM must know exactly where every callout is"**
   - Design (agreed direction, not built): embed per-zone normalized
     coordinates + level into the map-card block of every prompt, e.g.
     "`Mid` at (0.53, 0.10) upper" - generated at prompt-build time so the
     renamer applies and coordinates come from the same `_zone_anchors`
     source the editor shows. Gives the model true spatial reasoning
     (directions, distances) instead of adjacency-only. Touches
     `build_chat_system`, `build_insights_system/user`, dossier `build_system`;
     add a compact "## Zone map" section; test-pin that coordinates and user
     names appear and canonical names do not.
4. Minor debts: chat still cites raw `match_id:round` (First Read went human;
   chat could adopt Game-N labels too); README still lists 7-section dossier
   prose in one place; `scratch/` one-offs are uncommitted debris; radar
   playback scrubber remains deferred (docs/plans/2026-09-03 upgrade plan,
   Out of scope).

## Environment gotchas for the next agent

- Restart reminder: the user's server keeps old code loaded; every "it still
  does X" report - first ask whether the server was restarted.
- `Invoke-RestMethod`/`Remove-Item` are DENIED by permission rules; verify via
  in-process `TestClient` scratch scripts instead of live HTTP; never try to
  delete files from the shell.
- Playwright is unavailable (no package, network blocked for browser
  downloads): browser behavior is verified via static pins + TestClient +
  `scratch/radar_smoke.py` exists if the user ever installs Playwright.
- kopipasta on PATH is too old for `map --json`; explore with glob/grep.
- `_find_vpk_path` searches the CWD as well as `REPO_ROOT` - tests touching
  callouts must `monkeypatch.chdir(tmp_path)` or the real VPKs leak in and
  VRF runs mid-test (see `tests/test_web_callouts.py:isolated_repo_root`).
- Gemini thinking models: never trust visible-text length; check
  `LLMResult.truncated`.
- Static assets are cache-busted via `?v=3` query params in `index.html` -
  bump when editing app.js/radar.js/callouts.js/style.css.

## Verification commands

- Tests: `uv run python -m pytest -q` (full, ~80s; demo-marked tests parse a
  real demo and run by default; `-m live` only for paid-API spot checks).
- Lint: `uv run ruff check .` (line-length 100).
- Real-data smokes live in `scratch/`: `verify_callouts.py`,
  `regen_artifacts.py` (cluster-aware rebuild), `regen_insights.py` (paid),
  `patch_games.py`, `inspect_rosters.py`.

FACTS:
- repo: Python 3.13 + uv; FastAPI web app `uv run counterstrat` (127.0.0.1:8710)
- test: `uv run python -m pytest -q`; lint: `uv run ruff check .`
- data flows: demos -> lake parquet -> RoundScripts -> miners -> teambooks/briefs; clusters in data/teams.json; aliases in data/mapcards/<map>/aliases.json
- LLM: provider-agnostic clients in src/counterstrat/llm; output uncapped by default; renamer enforces user callout vocabulary at every boundary
- UI: static vanilla JS (app.js, radar.js, callouts.js) served from src/counterstrat/web/static
- open: callout reset buttons; median anchors; zone coordinates in prompts
- 304 tests green at 9fffc1a; all artifacts on disk regenerated post-clustering
