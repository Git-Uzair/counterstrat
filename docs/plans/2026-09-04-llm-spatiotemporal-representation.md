# LLM spatio-temporal representation upgrade

Date: 2026-09-04
Author: boss session (research + measured experiments)
Status: PROPOSED - every design decision below is backed by a measured
experiment on this repo's real demo data; raw results in
`scratch/repr_lab/results/*.json`, harness in `scratch/repr_lab/*.py`.

## Context

### The question

Two representation problems decide how accurate the analyst LLM can be:

1. **Space** - how to hand a 3D CS2 map (zones, connectivity, elevation,
   distances) to a text model so it reasons about positions, directions,
   travel times, and routes without hallucinating.
2. **Time** - how to hand a dynamic round (10 players moving, utility,
   kills, plant) to a text model so it knows exactly *what happened and
   when*, including joins ("where was X when Y died?").

### What the app ships today (measured baseline)

- Spatial: the raw `card.yaml` dump plus `format_zone_map` anchors
  (`src/counterstrat/llm/prompts.py:33-61`, `:10-30`). The yaml `topology:`
  section carries transit seconds **with no unit label anywhere**, no
  bearings, no coordinate/route derivations.
- Temporal: `RoundScript.to_text()`
  (`src/counterstrat/roundscript/models.py:79-116`) - 15s occupancy beats,
  first-contact kill only, utility log, END line. **It omits: every kill
  after first contact, all timestamps for FC/plant, player movement, and
  utility beyond 16 events.**

### Experiment design (what was actually run)

Built `scratch/repr_lab/`: same underlying data rendered into competing
text representations, then auto-graded MCQ/numeric questions whose ground
truth is computed programmatically (Dijkstra over the mined zone graph,
anchor bearings, real tick positions, real kill/utility timelines) - never
by an LLM. Map: `de_inferno` (the 4 ingested matches). Models:
`gemini-3.8-flash` (repo dev/eval policy) with cross-validation on
`claude-sonnet-5` (app default). Same questions, same options, same seed
for every arm.

Spatial arms: `current` (card.yaml+zone_map), `adjacency` (edges+seconds
only), `coords` (anchors only), `ascii` (grid radar), `scenegraph`
(topology backbone + per-edge seconds + compass bearing + per-zone (u,v)),
`scenegraph_matrix` (scenegraph + all-pairs seconds matrix).

Spatial tasks: easy = nearest_zone from a real tick coordinate, compass
direction between zones, fastest-to-reach ranking, waypoint on fastest
route, route seconds (numeric), two-player race to a target; hard =
position within 6s of two zones, route around a smoked-off zone,
detour cost when a zone is blocked (numeric).

Temporal arms: `current` (`to_text()` verbatim), `eventlog` (flat
chronological stream: SPAWNS, MOVE, KILL, DEATH, UTIL with blind list,
PLANT with alive counts, END - absolute seconds), `tracks` (per-player
zone stints + event list), `program` (phase-decomposed: anchors,
state@15s, per-phase events with relative offsets), `dual`
(eventlog+tracks), `dense` (5s-grid CSV of every player's zone + events).

Temporal tasks: easy = T-count in zone at t, event order, first-kill->plant
delta, kill zone, newly-occupied zone between beats, flash count before
plant, zone on a player's early track; hard = "where was P when K
happened" (track x event join), first player into a zone, CT alive at t,
seconds from smoke landing to first entry (join, 2s tolerance), longest
inter-kill gap.

### Results

Spatial, gemini-3.8-flash (n=58 easy / 25 hard per arm):

| arm | easy | hard | notes |
|---|---|---|---|
| current | 0.793 | 0.640 | route_time 1/10, detour_time 2/10 |
| adjacency | 0.845 | 1.000 | nearest_zone 4/10 (no coords = blind) |
| coords | 0.741 | - | on_path 6/10, race 7/10 (no graph) |
| ascii | 0.690 | - | worst; matches published grid findings |
| **scenegraph** | **0.983** | **1.000** | one miss: boundary-ambiguous tick |
| scenegraph_matrix | 0.983 | 1.000 | no accuracy gain over scenegraph |

Spatial, claude-sonnet-5 (same questions):

| arm | easy | hard |
|---|---|---|
| current | 0.759 | 0.440 |
| **scenegraph** | **0.966** | **0.800** |

Temporal, gemini-3.8-flash (n=60 easy / 47 hard per arm):

| arm | easy | hard | notes |
|---|---|---|---|
| current | 0.833 | 0.617 | count_util 6/10, kill_gap 1/10, util_to_entry 2/10 |
| **eventlog** | **1.000** | **0.957** | both hard misses = GT artifacts (see below) |
| tracks | 1.000 | 0.957 | same two artifact misses |
| program | 0.967 | 0.830 | util_to_entry 3/10: 15s snapshots lose the stream |
| dual | 1.000 | 0.957 | redundancy adds nothing |
| dense | 1.000 | 0.894 | 5s grid too coarse for 2s-tolerance joins |

Temporal, claude-sonnet-5:

| arm | easy | hard |
|---|---|---|
| current | 0.700 | 0.404 |
| **eventlog** | **0.950** | **0.936** |
| dual | 0.917 | 0.894 |

GT artifacts: the two hard-temporal questions missed by *every* complete
arm with the *same* wrong answer had a T player already inside the target
zone at throw time, making "entered N seconds later" ambiguous. On
well-formed questions eventlog/tracks/dual scored 45/45. Consensus-wrong
across arms = bad question, not bad representation.

### Findings (each one is load-bearing for the tasks below)

1. **Completeness beats format.** The baseline loses 17-60 points not
   because beats are a bad shape but because the text omits kills,
   timestamps, and movement. Any complete representation saturated the
   easy battery on flash.
2. **Precompute what LLMs are bad at; they handle the rest.** Unlabeled
   `topology` seconds produced systematic ~2x overestimates of travel
   time on both models (`route_time` 1/10 and 4/10; `detour_time` 0/10 on
   sonnet). One comment-level change - explicit "seconds to move" labels
   on edges - took the identical underlying numbers to 10/10. This is the
   T2SP thesis (docs/research.md) confirmed on our domain: shift structure
   extraction from the model into the representation.
3. **Topology is the backbone; coordinates are annotation.** Graph-only
   beat coords-only by +10 points easy and saturated hard; coords-only
   collapsed on path/race tasks. Matches the published navigation-
   serialization result (arXiv 2605.31404: topology = shield). But
   coordinate anchors are indispensable for grounding positions
   (nearest_zone: 4/10 without them, 9/10 with).
4. **Bearings must be precomputed.** Direction questions: 7/10 from raw
   coords on the graph-only arm vs 10/10 when each edge carries a compass
   tag. Cheap tokens, systematic gain, and consistent with LLM weakness
   at coordinate arithmetic (TRAM: "calculation slips" dominate temporal
   computation errors; same failure class).
5. **The all-pairs travel matrix is unnecessary.** Zero accuracy gain on
   either model, ~2x block tokens. Both models compute multi-hop routes
   reliably from labeled edge lists at this graph size (36 nodes). Skip
   it; revisit only if a 100+-zone map ever underperforms.
6. **ASCII maps lose.** Worst spatial arm (0.690). Consistent with
   PlanarBench/grid-probing literature. Do not ship a text radar.
7. **The flat chronological event log wins time.** Every event, absolute
   seconds, one line each. Entity-major (tracks) ties it; a hybrid adds
   nothing; 5s downsampling visibly hurts joins; phase summaries that
   *replace* the stream (program) lose 13 points on joins. Keep phase
   anchors as a *header*, never as a substitute for the stream. This is
   TISER's "explicit timeline" result, applied at the representation
   layer so the model doesn't have to build the timeline itself.
8. **Wall-clock is a proxy for representation quality.** Complete formats
   answered 3-4x faster (less thinking needed to reconstruct state).
   Cheaper AND more accurate.
9. **Latent bugs found while validating** (fix as part of this plan):
   - `AnthropicClient` has no thinking-token headroom: a small
     `max_tokens` silently returns empty text (`truncated=True`) on
     claude-sonnet-5 (`src/counterstrat/llm/anthropic_client.py:127`).
   - `DEFAULT_MAX_OUTPUT = 32_000` now trips the SDK's 10-minute
     non-streaming guard for sonnet-5: every uncapped `complete()` raises
     `ValueError: Streaming is required...`. The app's Anthropic path is
     currently broken for dossiers/insights; unnoticed because
     `data/settings.json` selects gemini.
   - `de_inferno` card has empty `rotates:` and `timings:` (the nav-based
     pass produced nothing for this map), so the current prompt carries
     zero route knowledge there.
   - Mined topology contains boundary artifacts (e.g.
     `Banana <-> Middle: 0.4s` on inferno) from zone-classifier flicker at
     zone seams.

## Assumptions

- The mined zone graph (`card.topology`) is the app's ground truth for
  connectivity; improving its hygiene is in scope, replacing it with nav
  pathfinding is not.
- User callouts stay canonical inside builders; the existing Renamer
  boundary (`src/counterstrat/aliases.py`) applies afterwards, unchanged.
- Miners consume structured `RoundScript` fields, not `to_text()`; text
  changes cannot affect mining outputs.
- `gemini-3.8-flash` remains the dev/eval model; `claude-sonnet-5` the
  Anthropic default (config.py:18-19).
- Token budgets: chat/dossier system prompts may grow to ~2.5k tokens for
  the map block; per-round timeline text may reach ~1.8k tokens (full) and
  ~0.8k (lite). All well inside `max_input_tokens=190_000`.

## Tasks

### Task 1 - Zone-graph hygiene + derived route fields on the card

**Goal**: the card's spatial numbers become trustworthy and complete
before any prompt embeds them: seam-artifact edges pruned, `rotates`/
`timings` derived from the mined graph whenever the nav pass produced
nothing (inferno today), units labeled in the yaml.

**Difficulty**: medium

**Verify**: `uv run python -m pytest -q tests/test_mapcard_transitions.py tests/test_mapcard_compile.py`

**Files**:
- `src/counterstrat/mapcard/transitions.py` (edge filter)
- `src/counterstrat/mapcard/compile.py` (derived rotates/timings fallback,
  yaml unit comment)
- `tests/test_mapcard_transitions.py`, `tests/test_mapcard_compile.py`

**Test first**:
- `test_seam_edges_pruned`: build a ZoneGraph from synthetic transitions
  where A<->B has 2 observations at 0.4s and A<->C has 5 at 3.0s; assert
  A-B absent, A-C present.
- `test_rotates_derived_when_nav_empty`: compile a card whose nav rotates
  are empty but whose topology connects both sites; assert `rotates`
  contains site->site entries with `run_s` equal to Dijkstra cost over
  topology and `via` equal to the shortest path interior.
- `test_timings_derived_when_nav_empty`: same for spawn->key-zone rows
  (T from `TSpawn`-tagged spawn zone, CT from CT spawn) covering both
  sites plus every `choke`-tagged zone.

**Change**:
- In `transitions.py`, where edges are accumulated, track observation
  counts; drop an edge iff `median_transit < 0.75s and observations < 3`.
  Record dropped pairs on the graph object; `compile.py` writes them under
  a new `topology_pruned:` card key (list of `"A|B"` strings) so pruning
  is inspectable.
- In `compile.py`, after nav-based rotates/timings: if `rotates` is empty
  and both sites are in the graph, emit the two site<->site shortest
  paths (both directions) with `run_s` = path seconds; if `timings` is
  empty, emit per-side earliest-reach seconds for sites + choke-tagged
  zones from the side's spawn zone via Dijkstra. Mark derived entries with
  `source: mined` so nav-derived and mined values stay distinguishable.
- Serialize the yaml `topology:` header with a comment line
  `# seconds to move between adjacent zones` (PyYAML: emit via a small
  post-processing string insert after dump - the card is written as text
  in `compile.py`, so insert the comment there).

**Done when**: recompiled inferno card has non-empty labeled rotates and
timings, `Banana <-> Middle: 0.4s`-class edges are gone, pruned pairs are
listed in the card, and the three new tests pass.

### Task 2 - `format_map_scene_graph`: the winning spatial block

**Goal**: replace the raw `card.yaml` dump in every LLM prompt with the
measured winner: per-zone entries carrying (u,v) anchors, tags, and
adjacency edges labeled `seconds + compass bearing`, plus an explicit
coordinate/unit convention header, objectives, and the (Task 1) rotate
and timing tables. No all-pairs matrix (Finding 5).

**Difficulty**: medium

**Verify**: `uv run python -m pytest -q tests/test_llm_prompts_scenegraph.py`

**Files**:
- `src/counterstrat/llm/prompts.py` (new builder + wire into
  `build_system`, `build_chat_system`)
- `src/counterstrat/llm/insights.py` (same block in the First Read prompt)
- `src/counterstrat/web/chat.py`, `src/counterstrat/web/routes.py`,
  `src/counterstrat/llm/dossier.py` (call sites: pass card object +
  anchors instead of raw yaml text)
- new `tests/test_llm_prompts_scenegraph.py`

**Test first** (pin the format exactly - this is the contract):
- `test_scene_graph_block_shape`: given a 3-zone toy card + anchors,
  output contains the convention header line
  (`u: 0=west edge -> 1=east edge, v: 0=NORTH edge -> 1=SOUTH edge`),
  one `ZoneName (u=0.NN, v=0.NN)` line per zone, edge lines of the form
  `-> Neighbor: 4.8s SE`, and NO raw yaml keys (`card_version`, `frame:`).
- `test_bearing_8way`: anchors placed due north/east/south-west of a hub
  produce `N`, `E`, `SW` on the hub's edges.
- `test_multi_level_suffix`: nuke-style anchors (a `lower` zone present)
  add `, lower level` / `, upper level` suffixes exactly like
  `format_zone_map` does today.
- `test_rotates_and_timings_sections`: cards with rotate/timing data get
  `rotates:` lines (`BombsiteA -> BombsiteB: 19.7s via Walkway > Middle >
  ...`) and `earliest_reach:` per-side tables; empty fields render nothing.
- `test_chat_system_embeds_scene_graph`: `build_chat_system` output
  contains the scene-graph header and not `topology:` raw yaml.

**Change** (core builder, adapted verbatim from the proven lab arm
`scratch/repr_lab/spatial.py:_scenegraph_lines` - bearings from anchor
deltas, `atan2(du, -dv)`, 8 sectors):

```python
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

def _bearing(anchors, a, b) -> str:
    ua, va, _ = anchors[a]; ub, vb, _ = anchors[b]
    ang = math.degrees(math.atan2(ub - ua, -(vb - va))) % 360
    return SECTORS[int(((ang + 22.5) % 360) // 45)]

def format_map_scene_graph(card: MapCard, anchors: dict[str, tuple]) -> str:
    # header: map name, coordinate + unit conventions (Finding 2, 4)
    # per zone: "Name (u=…, v=…)[, level][ tags=[…]]" then
    #   "  -> Neighbor: {sec:.1f}s {bearing}" per undirected edge
    # then objectives / rotates / earliest_reach sections (Task 1 data)
```

Wiring: `build_system(card_yaml, zone_map)` and
`build_chat_system(card_yaml, teambook, zone_map)` change signature to
accept the scene-graph block (call sites build it once per session/job);
the `<map_card>` tag stays so downstream lint/renamer behavior is
untouched. The Renamer already rewrites zone names at the boundary -
canonical names inside, exactly as today. Keep `format_zone_map` (the
editor block) - the scene graph *includes* anchors, so chat/dossier stop
passing `zone_map` separately.

**Done when**: all five pins pass; existing prompt tests updated; a chat
session on inferno answers the manual rotate question from the
Verification section with graph-consistent seconds.

### Task 3 - RoundScript v2: tracks + the timeline text

**Goal**: the round text carries *everything that happened, with absolute
seconds*: per-player zone stints (MOVE), every kill and death, utility
with pop times and blind lists, PLANT with timestamp and alive counts,
phase anchors as a header (never replacing the stream - Finding 7), END
with seconds.

**Difficulty**: hard (touches serializer, model, storage format)

**Verify**: `uv run python -m pytest -q tests/test_roundscript_serialize.py tests/test_roundscript_timeline.py`

**Files**:
- `src/counterstrat/roundscript/models.py` (new `ZoneStint` model,
  `RoundScript.tracks`, `to_timeline_text(lite: bool = False)`)
- `src/counterstrat/roundscript/movement.py` (expose
  `zone_stints(ticks, round_num, min_dwell_s=2.0) -> dict[str, list[ZoneStint]]`
  refactored out of the existing `_Visit`/`_dwell_merge` pipeline it
  already runs for sentences)
- `src/counterstrat/roundscript/serialize.py` (populate `tracks`)
- new `tests/test_roundscript_timeline.py`

**Test first**:
- `test_zone_stints_debounce`: synthetic ticks with a 1s flicker zone
  merge into surrounding stints; entries carry integer seconds
  `(t0, t1, zone)` and stop at death.
- `test_timeline_text_complete`: a synthetic round with 3 kills renders
  ALL of: a `SPAWNS` line, one `MOVE` line per stint after spawn,
  `KILL a (T) kills b in \`Zone\` [weapon]` for every kill (not just FC),
  `PLANT ... at NNs` with alive counts, `END ... at NNs`, and an
  `anchors:` header with `first_contact=`, `plant=` (with `+Ns after
  first contact`), `end=`.
- `test_timeline_lite_drops_moves_keeps_kills`: `lite=True` removes MOVE
  lines and the SPAWNS roster but keeps every KILL/UTIL/PLANT timestamp.
- `test_timeline_falls_back_without_tracks`: a v1 script JSON (no tracks)
  still renders (no MOVE lines, everything else intact) - old artifacts
  on disk keep working until the next rebuild.
- Extend `tests/test_roundscript_serialize.py` demo-marked e2e: the real
  round's timeline contains as many KILL lines as `script.kills`.

**Change**:
- `ZoneStint(BaseModel): t0: int; t1: int; zone: str`;
  `RoundScript.tracks: dict[str, list[ZoneStint]] = {}` plus
  `sides: dict[str, str] = {}` (player -> T/CT at that round) so the
  renderer never guesses sides. Pydantic default keeps old JSON loading.
- `serialize_round` fills both from `zone_stints(...)` (per-second zone
  sampling, alive only, 2s dwell merge - the identical algorithm the lab
  validated; movement.py already implements the merge for sentences).
- `to_timeline_text()`: header line (round, buys, spends, score) +
  `anchors:` line + merged chronological stream sorted by t
  (`t=NNs MOVE|KILL|DEATH|UTIL|PLANT ...`), zones in backticks for the
  renamer/lint, THEN the legacy 15s occupancy beat lines as a
  `state@NNs:` block per beat (state snapshots aid state_at recall -
  program arm's only strength - and cost ~10 lines). `lite=True` renders
  anchors + kills/utility/plant + states, no MOVE/SPAWNS (~0.8k tokens).
- `to_text()` stays untouched (miners/tests/back-compat).

**Done when**: new tests pass, demo e2e sees every kill in the timeline,
and a rebuilt script JSON round-trips `tracks` through disk.

### Task 4 - Wire the timeline into chat, dossier, and First Read

**Goal**: every surface that shows the LLM a round shows the complete
timeline: chat's `get_round_script` returns the full timeline, dossier
exemplars use it, insights (corpus-wide) uses the lite variant to respect
budget.

**Difficulty**: easy-medium

**Verify**: `uv run python -m pytest -q tests/test_llm_tools_v2.py tests/test_llm_dossier.py tests/test_llm_insights.py`

**Files**:
- `src/counterstrat/llm/tools.py` (`get_round_script` -> timeline text)
- `src/counterstrat/llm/prompts.py` (`build_user` exemplars -> timeline)
- `src/counterstrat/llm/insights.py` (corpus rounds -> `lite=True`)
- matching tests

**Test first**:
- `test_get_round_script_returns_timeline`: tool output for a scripted
  round contains `anchors:` and a `KILL` line for a non-FC kill.
- `test_dossier_exemplars_use_timeline`: `build_user` output contains
  `t=` event lines inside the exemplar section.
- `test_insights_rounds_are_lite`: insights user prompt contains KILL
  lines but no MOVE lines; total prompt estimate for the current corpus
  stays under `max_input_tokens` via `check_budget` (assert no raise).

**Change**: swap `s.to_text()` calls at the three call sites for
`s.to_timeline_text()` / `s.to_timeline_text(lite=True)`; exemplar cap in
`select_exemplars` stays 12. The Renamer path is untouched (it rewrites
the rendered text exactly as before; timeline keeps zones in backticks).

**Done when**: the three tests pass; a live chat "walk me through round N"
names mid-round kills with their times and zones (spot-check).

### Task 5 - Anthropic adapter: streaming + thinking headroom

**Goal**: the Anthropic path works again for analysis-grade calls
(`DEFAULT_MAX_OUTPUT` currently raises `ValueError: Streaming is
required...` on claude-sonnet-5), and small caps can never silently return
empty text on a thinking model.

**Difficulty**: medium

**Verify**: `uv run python -m pytest -q tests/test_llm_clients.py`; live
spot check `-m live` if keys present.

**Files**:
- `src/counterstrat/llm/anthropic_client.py`
- `tests/test_llm_clients.py` (+ a live-marked test)

**Test first**:
- `test_sdk_transport_streams_large_caps`: transport receives
  `stream=True`-shaped call (or `messages.stream` usage) when
  `max_tokens >= STREAM_THRESHOLD`; replay-transport asserts the request
  shape and returns a fixture.
- `test_small_cap_gets_thinking_headroom`: `complete(max_tokens=64)` sends
  a wire `max_tokens` of `64 + THINKING_HEADROOM` (mirroring the Gemini
  adapter's contract: caller caps VISIBLE text).
- live-marked: `complete(system="Reply with only the letter B", …,
  max_tokens=8)` returns non-empty text on claude-sonnet-5.

**Change**:
- Add `THINKING_HEADROOM` (start with 8_192 - sonnet-5 burned ~60 tokens
  on a trivial MCQ in the lab, but headroom must cover real reasoning;
  Gemini uses 24_576) and add it to the wire cap in `complete`,
  `complete_json`, `chat` exactly as `gemini_client.py:171` does.
- In `_sdk_transport`, when effective `max_tokens` exceeds a
  `STREAM_THRESHOLD` (16_000), use `client.messages.stream(...)` and
  assemble the final message via `get_final_message()`; below it, keep
  `create` (fast path). `parse` calls route through the SDK's streaming
  parse when over threshold.
- Keep `truncated` semantics (`stop_reason == "max_tokens"`).

**Done when**: replay tests pin both behaviors; live spot check returns
text; dossier generation on provider=anthropic completes without
ValueError.

### Task 6 - Permanent representation benchmark (`counterstrat.eval.repr_bench`)

**Goal**: the lab that produced this plan's numbers becomes a first-class,
reproducible harness (like `mapcard.quiz`), so any future format change is
measured against `current` and `scenegraph`/`timeline` instead of argued.

**Difficulty**: medium

**Verify**: `uv run python -m pytest -q tests/test_eval_repr_bench.py`
(question generation only - no API); manual paid run via CLI.

**Files**:
- new `src/counterstrat/eval/repr_bench.py` (port of
  `scratch/repr_lab/{common,spatial,temporal,hard}.py`, cleaned: builders
  import the Task 2/3 production formatters for their arms instead of
  duplicating them)
- new `tests/test_eval_repr_bench.py`

**Test first**:
- `test_question_generation_deterministic`: same seed -> identical qids
  and answers; every MCQ has exactly one correct option; numeric answers
  parse as floats.
- `test_ground_truth_no_llm`: generating 50 questions performs zero LLM
  calls (inject a client stub that raises).
- `test_join_questions_have_margins`: who_where_when joins sit >=2s inside
  stints; first_into winners lead by >=2s (the two GT-artifact classes
  found in the lab are structurally excluded).

**Change**: CLI
`uv run python -m counterstrat.eval.repr_bench --suite spatial|temporal
--arms current,scenegraph --provider gemini --out data/eval/repr_bench.json`
mirroring `eval.benchmark`'s argument style; arms resolve to the
production formatters (`format_map_scene_graph`, `to_timeline_text`) plus
the frozen `current` renderers for regression comparison; util_to_entry
generation adds the "nobody already inside the zone" constraint.

**Done when**: harness runs against the inferno corpus reproducing this
plan's ordering (current < scenegraph / current < timeline) and the three
tests pass.

### Task ordering and rollout

1 -> 2 -> 3 -> 4 ship the accuracy win (1 and 2 are independent of 3 and
4; run as two parallel tracks if desired). 5 is independent and unblocks
the Anthropic provider. 6 lands last, importing the production formatters.
After 3/4: remind the user to RESTART the server and trigger a rebuild so
scripts regain `tracks` (rebuild re-serializes all matches; measured 5.1s
per match on this corpus).

Expected end state, measured expectation from the lab: spatial prompt
block ~0.9k tokens (vs 1.9k today) scoring +19 easy / +36 hard points on
flash; round timeline ~1.6k tokens full / ~0.8k lite scoring +17 easy /
+34 hard vs today's text, with every kill, timestamp, and movement
available to the analyst.

## Research grounding (external)

- **T2SP** (user-provided, docs/research.md): structured symbolic
  decomposition beats raw serialization for time series across models and
  tasks; deterministic, training-free, representation-side. Adopted as the
  design stance for both suites; our "decomposition" is domain-specific
  (anchors/phases/stints), per its own Limitations section.
- **Topology vs geometry for LLM navigation** (arXiv 2605.31404): topology
  is the robust backbone; semantics must be correct; format effects vary
  with compression. Reproduced here: adjacency arms saturated hard tasks,
  coords-only collapsed.
- **3D scene graphs as LLM interfaces** (SayPlan arXiv 2307.06135,
  ConceptGraphs arXiv 2309.16650): text-serialized scene graphs are the
  standard grounding for large 3D spaces; hierarchy/collapse only needed
  at far larger scale than a CS2 map.
- **Grid/ASCII spatial probing** (PlanarBench arXiv 2606.02010; "Stuck in
  the Matrix" arXiv 2510.20198): text-grid spatial reasoning degrades
  sharply with size; reproduced (ascii arm worst).
- **TRAM** (arXiv 2310.00835): temporal "calculation slips" dominate LLM
  errors -> precompute deltas/anchors. **TISER** (arXiv 2504.05258):
  explicit timeline construction is the highest-leverage stage ->
  ship the timeline pre-built.

## Experiment inventory (for reproduction)

- Harness: `scratch/repr_lab/common.py` (clients, grading, parallel
  runner), `spatial.py` (6 arms, 6 task kinds), `temporal.py` (6 arms,
  7 task kinds), `hard.py` (3 + 5 hard kinds), `run_spatial.py`,
  `run_temporal.py`, `run_hard.py`, `smoke.py`.
- Results: `scratch/repr_lab/results/{spatial,temporal,hard_*}_{provider}_{arm}.json`
  (per-question rows: qid, expected, raw model text, tokens, truncated).
- Spend: ~1,300 gemini-3.8-flash calls + ~700 claude-sonnet-5 calls, all
  sub-2.5k-token prompts.
- Caveats: single map (de_inferno - the only map with ingested demos);
  n=25-60 per suite per arm; MCQ chance floor 25%; two GT artifacts in
  hard-temporal documented above and excluded by construction in Task 6.
  scratch/ stays uncommitted per repo convention; Task 6 is the durable
  home. An empty stray `map.json` sits at repo root (deletion is
  permission-blocked for agents).

## Risks

- **Prompt-cache invalidation**: changing system prompts invalidates
  Anthropic's ephemeral cache entries; first calls after deploy are
  full-price. One-time cost, no action.
- **Insights token growth**: corpus-wide prompts embedding many rounds
  grow ~2x with timeline-lite. Budget check already exists
  (`check_budget`); the lite variant plus exemplar capping keeps 70-round
  corpora near ~60k tokens. Verified against budget in Task 5 test.
- **Graph hygiene may prune a real edge** (fast rotate through a seam).
  Mitigation: prune only edges *both* faster than 0.75s *and* supported by
  fewer than 3 observed transitions; log what was pruned into the card for
  inspection.
- **Old scripts on disk** lack `tracks`; `to_timeline_text()` falls back
  to the legacy body for them until the next rebuild regenerates scripts
  (`web/maintenance.py:rebuild_artifacts` already re-serializes).
- **Anthropic streaming change** touches the transport layer; replay-
  transport tests keep it off the network, and the live-marked spot check
  covers the real wire.

## Verification commands

- `uv run python -m pytest -q` - full suite green (321 baseline).
- `uv run ruff check .` - zero errors.
- `uv run python -m counterstrat.eval.repr_bench --suite spatial --arms current,scenegraph`
  then `--suite temporal --arms current,timeline` - the new formats must
  beat `current` on both suites (expected: ~+0.15-0.20 easy, ~+0.30 hard).
- Manual: restart the server (`uv run counterstrat`), open a team on
  inferno, ask the chat "how long does a B->A rotate take and which zones
  does it pass?" - answer must quote seconds consistent with the scene
  graph and name on-path zones.

## Execution record (2026-09-04, all tasks DONE)

Commits, one per task, on master: 8afda4f (T1), 36d0ee0 (T2), then T3
(RoundScript v2), T4 (wiring), T5 (Anthropic adapter), T6 (repr_bench).
347 tests green, ruff clean, Anthropic live smoke passed.

Deviations from the written plan, both evidence-driven:

1. **Task 1 root cause was not seam flicker.** The mined graph was polluted
   by round-boundary teleports: post-round positions followed by the next
   round's freeze under the same round_num produced edges like
   `BombsiteB -> TSpawn n=2451 median=-70s`. Fixed at the source (freeze
   rows dropped, strictly positive transit, match_id in the grouping key).
   After the fix every remaining sub-second edge is genuine carved-
   vocabulary adjacency (`Middle -> Banana 0.44s n=68` with an asymmetric
   10.3s reverse - custom-zone seam, real). No pruning shipped: the planned
   rule (median < 0.75s AND n < 3) was dead code under the existing n >= 5
   gate and would have pruned nothing real.
2. **Inferno's empty rotates/timings were a missing-tags cascade, not a
   missing-derivation.** The mined-graph derivation already existed; the
   inferno lexicon ships no overlay so no zone carried the `site` tag.
   Sites now infer from Bombsite* names when untagged; rotates/timings/
   objectives.sites populate on rebuild.
3. Task 5 constants: THINKING_HEADROOM = 8_192, STREAM_THRESHOLD = 16_000;
   streaming routes both create- and parse-shaped requests through
   `messages.stream(...)` (SDK 1.3.0 accepts `output_format` there).

Production-format benchmark (gemini-3.8-flash, live inferno corpus):
spatial current 0.695 vs scenegraph 0.966 (n=59); temporal current 0.602
vs timeline 1.000 vs timeline_lite 0.855 (n=83; lite loses exactly the
movement-join kinds it intentionally drops). Reports:
`data/eval/repr_spatial.json`, `data/eval/repr_temporal.json`.

Operator actions still owed: restart the dev server and run a rebuild per
map (Rebuild in the UI or `rebuild_map_zones`) so cards lose the teleport
edges + gain sites/rotates/timings, and scripts gain `tracks`. Until the
rebuild, timelines render without MOVE lines (graceful fallback).

Known leftovers, out of scope: `tests/test_llm_agent.py:330` imports
`CHAT_TOOL_RULES` which 8c46c94 removed (pre-existing broken live test);
`llm/predict.py` (eval benchmark arm) still embeds raw card yaml by
design; `scratch/repr_lab/` and a stray empty `map.json` remain
uncommitted at the repo root.

FACTS:
- repo: Python 3.13 + uv; test `uv run python -m pytest -q` (321 green at 81fbc6b); lint `uv run ruff check .`
- experiment: scenegraph spatial block and eventlog timeline beat current app formats by +17-36 points on both gemini-3.8-flash and claude-sonnet-5; ascii and all-pairs matrix rejected by measurement
- current app gaps: round text omits non-FC kills, FC/plant timestamps, movement; card topology unlabeled units; inferno card rotates/timings empty
- latent bugs: AnthropicClient lacks thinking headroom AND uncapped calls raise SDK streaming ValueError on claude-sonnet-5; app currently on gemini so unnoticed
- keys: ANTHROPIC_API_KEY + GOOGLE_GENERATIVE_AI_API_KEY in env; dev/eval model gemini-3.8-flash
- plan: 6 tasks - graph hygiene, format_map_scene_graph, RoundScript tracks+timeline, wiring, anthropic streaming fix, permanent repr_bench
- lab code scratch/repr_lab (uncommitted); results JSONs carry per-question evidence

PLAN COMPLETE

