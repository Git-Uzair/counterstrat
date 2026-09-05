# Sidebar & ingestion UX: multi-upload, match-aware catalog, subset analysis, batch delete

Date: 2026-09-05
Status: APPROVED (user picked the team-grouped tree recommendation)

## Context

User pain (verbatim intent): upload 5-6 demos at once; sidebar currently
shows one card per (team, map) with opaque hash chips; wants to analyze 1..N
selected matches of a team on a map; wants who-vs-who visible at first look;
wants duplicate protection visible; wants card-level delete that removes all
demos inside.

Measured current state (all verified this session):

- `web/static/index.html:33` file input, no `multiple`; single `#job-status`
  box; `app.js:135-151` uploads `files[0]` only; `app.js:155-158` a new
  upload kills the previous job's poll timer.
- `POST /api/demos` (`routes.py:87`) accepts exactly one file; each request
  spawns `run_ingest` as a BackgroundTask; two parallel ingests race on
  corpus.jsonl append, teams.json overwrite, card sightline refresh, and
  teambook mining (`ingest.py:169,263,287-288,303`).
- Duplicate uploads: content-hash dedupe already exists in the manifest
  (`corpus.py:31-34`) but the pipeline re-runs anyway and reports success -
  invisible to the user.
- Catalog: `GET /api/teams` (`routes.py:229-254`) returns per-cluster
  `map_stats[map].matches = [{match_id, rounds}]`; `app.js:250-332` renders
  flat (team, map) cards with `hash8 · Nr` chips; chip click = single-match
  scope; no subset scope exists (`chat.py:57` `match_id: str | None`).
- Radar layers endpoint ALREADY accepts subset scope: `matches` csv param
  (`radar_api.py:145,160-162`).
- Deletion: per-demo only (`DELETE /api/demos/{id}`); each delete runs a
  full `rebuild_artifacts` (`maintenance.py:328-354`).
- Score data: `rounds.parquet` has `winner` ("t"/"ct" lowercase),
  `t_team_key`, `ct_team_key` per round (verified on real dust2 lake).
- Cluster names: most common clan name (`teams.py:106-115`).
- No played-at date exists; `registered_at` (upload time) is the only date.

## Assumptions

- "Select multiple matches" scope applies to chat + radar. Dossier and AI
  First Read remain corpus-wide artifacts (cached per (team, map)); First
  Read renders only when the selection is the full set - same rule as
  today's single-match behavior.
- Match date = upload date, labelled "added"; demos carry no played-at.
- Serializing ingests (one at a time) is acceptable; parsing is CPU-bound
  and the UI shows per-job stage. A `threading.Lock` in `run_ingest` beats
  a worker-queue: zero new infrastructure, BackgroundTasks stay
  synchronous under TestClient so existing tests keep their semantics.
- Deleting a (team, map) card deletes those demos globally (they also
  disappear from the opponents' cards) - stated in the confirm dialog.

## Tasks

### Task 1 - Ingest safety: serialize pipeline + duplicate short-circuit + filename collisions

Goal: N concurrent uploads never corrupt shared artifacts; duplicates are
instant and visible; same-name uploads don't overwrite each other.
Files: `src/counterstrat/web/ingest.py`, `src/counterstrat/web/routes.py`,
`tests/test_web_api.py`.
Change:
- `ingest.py`: module `_INGEST_LOCK = threading.Lock()`; `run_ingest` body
  wrapped in `with _INGEST_LOCK:` (job stays "queued" while waiting).
  `JobState.stage` Literal gains `"duplicate"`.
- `routes.py upload_demo`: unique-ify `dest_path` when it exists
  (`stem-2.dem`, `stem-3.dem`, ...); after magic validation compute
  `_sha256_16(dest_path)`; if match_id in manifest AND
  `data/scripts/<match_id>` has round files -> save JobState
  stage="duplicate" (match_id, map_name, detail naming the existing match),
  unlink the redundant new copy when its path differs from the recorded
  one, return job_id without scheduling ingest. Incomplete prior ingests
  (registered but no scripts - e.g. the dust2 NaN crash) re-run normally.
- Tests: duplicate upload -> terminal "duplicate" stage + no re-ingest;
  filename collision -> second file lands at a distinct path.
Done when: tests pass; `test_upload_compressed_zst_and_gz` (monkeypatches
`routes.run_ingest`) still passes unchanged.

### Task 2 - Catalog enrichment: opponent, score, added_at per match

Goal: `GET /api/teams` match entries carry who-vs-who at first look.
Files: `src/counterstrat/web/routes.py`, `tests/test_web_api.py`.
Change: in `list_teams`, build match_id -> [clusters] reverse index from
unique clusters; load manifest once; per match entry add `added_at`
(registered_at or None), `opponent_id`/`opponent_name` (the other cluster;
None if unknown), `score_won`/`score_lost` (winner rows in that match's
`rounds.parquet` where this cluster's keys held the winning/losing side;
None when parquet missing). Cache per-match score frames across teams.
Sort matches newest-first by added_at.
Test: fixture manifest + rounds.parquet -> assert enrichment; missing
rounds.parquet -> None scores; existing catalog test keeps passing.

### Task 3 - Subset scope: match_ids through chat sessions

Goal: a chat session scoped to any subset of a team's matches on a map.
Files: `src/counterstrat/web/chat.py`, `tests/test_web_chat.py`.
Change: `CreateSessionRequest.match_ids: list[str] | None` (keep `match_id`
accepted, normalized to a 1-list); `_build_session(match_ids)`: validate
every id against `teambook.generated_from` (404 otherwise), filter
source_matches, re-mine the teambook whenever match_ids is not None
(same rule as today's single-match); transcript meta stores `match_ids`;
`_get_session` restores from either meta key.
Test: subset of 2 of 3 matches -> session tools see only those matches'
scripts; unknown id -> 404; legacy `match_id` still works.

### Task 4 - Batch delete: N demos, one rebuild

Goal: delete a whole (team, map) card in one call.
Files: `src/counterstrat/web/maintenance.py`, `src/counterstrat/web/routes.py`,
`tests/test_web_delete.py`.
Change: `maintenance.delete_demos(cfg, match_ids) -> list[DemoRecord]`:
validate all first (KeyError lists missing), remove each (manifest entry,
lake/scripts trees, uploaded file inside data root), then ONE
`rebuild_artifacts`. `delete_demo` becomes a thin wrapper. New route
`DELETE /api/demos?matches=a,b,c` -> 404 naming missing ids, clears chat
sessions, returns deleted list + rebuild summary.
Test: two demos deleted in one call -> both gone, rebuild ran once
(monkeypatch counter); unknown id in list -> 404, nothing deleted.

### Task 5 - Frontend: multi-upload queue panel

Goal: drop/browse N files; per-file row with live stage; duplicate rows
visibly distinct; done rows refresh the catalog.
Files: `web/static/index.html`, `web/static/app.js`, `web/static/style.css`,
`tests/test_web_static.py`.
Change: `multiple` on the input; `#job-status` box replaced by
`#ingest-queue` list; uploads run sequentially (one POST at a time, each
row polls its own job every 2s; poller map replaces the single
`state.pollTimer`); stages done/error/duplicate are terminal (done ->
`loadTeams()`). Bump `?v=` on touched assets.
Done when: static pins assert `multiple`, `ingest-queue`, and no
`files[0]`-only handling.

### Task 6 - Frontend: team-grouped catalog tree + subset selection

Goal: the sidebar the user asked for.
Files: `web/static/app.js`, `web/static/style.css`,
`web/static/index.html` (only if container markup needs it),
`tests/test_web_static.py`.
Change: `renderTeams` renders team groups (collapsible header: name,
maps/matches counts) -> per-map card (map name, n rounds, match count,
map-delete button) -> match rows (checkbox default-checked; label
`vs <opponent> · <won>-<lost> · <rounds>r · added <date>`, hash tooltip;
row × delete) -> footer `Analyze` button whose label tracks the selection
("Analyze all 4 · n=87" / "Analyze 2 of 4 · n=45").
`selectTarget(team, displayName, mapName, cardEl, matchIds|null)`:
null = full scope; sends `match_ids` to `/api/chat/sessions`; passes the
array to `CounterStratRadar.onTargetSelected`; header meta names the scope;
First Read renders only at full scope. Row-label click = analyze that
match alone. `radar.js`: accept match array, emit `matches=` csv on the
layers URL.
Done when: static pins cover new classes/behaviors; manual TestClient
smoke renders enriched teams payload.

### Task 7 - Frontend: deletion UX

Goal: map-card delete removes all its demos; row delete stays.
Files: `web/static/app.js`, `web/static/style.css`.
Change: map-delete button -> confirm dialog naming the team, map, match
count, and the cross-team consequence -> `DELETE /api/demos?matches=csv`
-> `loadTeams()`; selection/session reset when the active target dies.
Row × keeps the existing single-demo flow.

### Task 8 - Full verification

`uv run python -m pytest -q` (0 failures), `uv run ruff check .` clean,
commits per task, remind user to restart the dev server.

## Risks

- Ingest lock holds one demo's full pipeline (~30-60s each); 6 demos ~5min
  serial - acceptable, visible per-row.
- Score enrichment reads one parquet per match per catalog load; fine at
  this corpus scale (cached within the request).
- Insight caches are (team, map)-wide; subset chat re-mines its own
  teambook in-session (already the single-match pattern), so no cache
  poisoning.
- `_build_session` re-mining for subsets uses only that subset's scripts -
  identical semantics to today's single-match path, just N>1.

## Execution record (2026-09-05, all tasks DONE)

| Task | Commit |
|---|---|
| 0 (pre-req) NaN ingest crash fix (orphan molly, dust2) | 118831e |
| 1 Ingest lock + duplicate stage + filename collisions | 889d4eb |
| 2 Catalog enrichment (opponent/score/added_at) | 5b6711d |
| 3 Chat subset scope (match_ids) | 41d5fbb |
| 4 Batch delete, one rebuild | a2ee9ff |
| 5-7 Frontend: queue panel, catalog tree, deletion UX | 97cc980 |

Suite at completion: 389 passed / 6 live-deselected; ruff clean; both JS
files pass `node --check`. Radar subset scope reuses the pre-existing
`matches` csv param (`radar_api.py:145`) - no backend change needed there.

Operator actions owed: RESTART the dev server (old code stays loaded
otherwise), then re-upload the dust2 demo that crashed
(`1-bb7908c0-...-1-1.dem`) - its earlier failure left it registered but
script-less, and the duplicate check deliberately re-runs such matches.

Notes:
- Dossier + AI First Read stay corpus-wide artifacts; First Read renders
  only when the full match set is selected (same rule as the old
  single-match behavior).
- `test_upload_incomplete_prior_ingest_reruns` pins the crash-recovery
  path the dust2 demo needs.

PLAN COMPLETE
