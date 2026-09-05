# Scoped First Read, scope-hash chat identity, sidebar scroll, dossier removal

Date: 2026-09-05
Status: APPROVED (user requirements list confirmed verbatim + "remove the Dossier")

## Context (verified)

- Sidebar squish: `.team-group` flex children shrink (default flex-shrink:1)
  inside `.teams-list`, so content compresses instead of overflowing - the
  scrollbar never engages.
- First Read today: one corpus-wide `insights.json` per (team, map)
  (`routes.py:457-556`), markdown with six prompt-mandated `##` sections
  (`insights.py:70-76`), rendered only at full-scope selection behind a
  button.
- Chat sessions: throwaway uuid ids (`chat.py:401`); transcripts in
  `data/chats/<sid>.jsonl` with meta {team_key, map_name, match_ids};
  replay endpoint exists (TranscriptResponse).
- Deletion: clears the in-memory session dict only; scoped caches would
  survive without new purging.

## Tasks

1. CSS: `.team-group { flex-shrink: 0 }` - scroll, never squish.
2. Remove the Dossier: `GET /reports/...` endpoint, `llm/dossier.py`,
   header button + app.js handler, tests/pins. Keep `dossier.md` in the
   stale-prune lists (legacy cleanup).
3. Shared scope identity: `scope_hash(team_id, map, sorted_ids) = sha1[:12]`
   in a tiny `web/scope.py`.
4. Scoped insights: `GET .../insights?matches=csv&generate=1` resolves the
   subset (404 on unknown ids), caches at
   `teambooks/<team>/<map>/insights/<hash>.json` with concrete match_ids +
   alias_fp inside; generation re-mines the subset teambook (chat's
   pattern); mock path preserved. Legacy insights.json superseded.
5. Chat scope identity: session_id = scope hash; POST /sessions with an
   existing transcript reuses it (no duplicate meta); meta always stores the
   RESOLVED concrete ids. Frontend replays via the transcript endpoint.
6. Chat knows the First Read: at message time, if the scoped insights cache
   exists, its text is appended to the system prompt as a first-read block.
7. Deletion invalidates: delete_demos purges chat transcripts and scoped
   insight files whose stored match_ids intersect the deleted set; frontend
   clears the analysis pane when the current scope is hit.
8. Frontend: `#first-read-panel` pinned above the scrolling messages;
   Analyze (any selection size) -> session create/reuse -> transcript
   replay -> insights fetch with generate=1 (loading skeleton -> six
   section cards + provenance/warnings card; 503 -> configure-key card);
   collapsible; cache-busters bumped; pins updated.
9. Full suite + ruff + commits per task group.

## Execution record (2026-09-05, all tasks DONE)

| Task | Commit |
|---|---|
| 1 sidebar flex-shrink + 2 dossier removal | 69332ec |
| 3-7 scope.py, scoped insights, scope-hash sessions, first-read block, deletion purge | 839064c |
| 8 frontend pinned panel, cards, replay, invalidation | b83565d |

Suite at completion: 402 passed / 5 live-deselected; ruff clean; node
--check clean on app.js + callouts.js. The lint gate (lint_dossier) moved
to llm/lint.py and stays on First Read generation.

PLAN COMPLETE
