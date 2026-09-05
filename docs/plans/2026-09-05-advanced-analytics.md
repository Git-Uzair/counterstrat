# Advanced analytics: rotation latency, utility ROI, death contexts, retake trees, matchup diff

Date: 2026-09-05
Status: EXECUTED (2026-09-05, all five tasks; see Outcome at the end)

## Context

The analyst model, asked what would sharpen its counter-calls, listed five
missing layers. Feasibility was audited against the actual lake schema and
the measured First Read budget; this plan implements all five in doable
form, exposed to BOTH the First Read and chat.

Measured baseline (Google countTokens, 2026-09-05, team_7yrant on
de_ancient, 6 matches / 116 rounds):

- Full First Read prompt: 226,073 tokens (system 3,606).
- All mined artifact sections combined: ~22.5k chars (~9k tokens).
- Round timelines: ~477k chars (~213k tokens) - 94% of the prompt.

Conclusion: new mined sections are budget-trivial. The timelines dominate;
nothing in this plan may add per-round timeline lines beyond what exists.

Data facts (verified in-session):

- Lake ticks: 16 Hz per player (TICK_STEP=4 of 64), X/Y/Z + view yaw
  sampled (yaw powers the existing gaze feature). No footstep/audio events
  exist in CS2 demos - sound triggers are out.
- `player_blind.parquet`: per-victim flash blind durations. `damages.parquet`:
  per-event damage with weapon attribution. `smokes.parquet`/`infernos.parquet`:
  bloom windows + positions. `kills` events carry `thrusmoke`/`penetrated`.
  `item_purchase`: spend.
- Miners (`mining/*.py`) take `list[RoundScript]` only; anything needing lake
  tables must be folded into scripts AT SERIALIZE TIME (the lake frames are
  in memory there) - this is the established pattern (gaze did the same).
- Scripts regenerate on ingest and on every zone rebuild
  (`rebuild_map_zones`), so schema additions reach old corpora via Rebuild.
- Every demo already books BOTH teams: opponent data ingestion needs no new
  pipeline - only the diff tool is missing.

## Assumptions

- Sound cues are unavailable (no such demo events); rotation triggers are:
  first blood, utility detonation in/adjacent to the defender's zone, plant,
  first shots fired, enemy entering a zone visible from the defender's zone
  (sightline matrix). Stated in tool docs so the model never claims audio.
- "Available to First Read and chat" means: each analytic renders as a
  compact prompt section in `build_insights_user` AND as a chat tool. The
  matchup diff is chat-only by nature (First Read is single-team; the
  opponent is named mid-conversation) - the First Read instead notes when
  other booked teams exist on the same map.
- Small-corpus honesty everywhere: every emitted row carries n and evidence
  ids; the sample-size doctrine already in prompts governs how the model
  may cite them. No minimum-n gates hidden inside miners beyond the
  existing signal conventions (state n, mark low_n).
- Victim-held-weapon at death needs a new tick prop; it applies to NEW
  ingests only (old lakes lack the column; miners must tolerate absence).

## Tasks

### Task 1 - Serialize-time enrichment: rotation responses, utility effects, death contexts

Goal: fold the lake-only facts into RoundScript so every miner and tool
downstream is pure-script, following the gaze pattern.
Difficulty: HARD (the load-bearing task; everything else consumes it).
Files: `roundscript/models.py`, `roundscript/serialize.py`,
`roundscript/movement.py` (helpers), `lake/extract.py` (one tick prop),
`tests/test_roundscript_rotations.py` (new),
`tests/test_roundscript_utility.py`, `tests/test_roundscript_timeline.py`.

Model additions (all optional-with-defaults so pre-v3 scripts load):

- `RotationEvent {t_trigger: float, trigger: str, player: str, side: str,
  from_zone: str, to_zone: str, latency_s: float}` and
  `RoundScript.rotations: list[RotationEvent] = []`.
  Trigger vocabulary: `first_blood`, `utility_near`, `plant`, `shots`,
  `visible_contact`.
- `UtilEvent` gains `enemy_blind_s: float | None`, `team_blind_s: float | None`
  (flashes; from player_blind joined on tick window + thrower),
  `damage: int | None` (HE/molly; damages rows attributed within the bloom
  window), `kills_through: int | None` (smokes; kills with thrusmoke across
  the smoked pair while active).
- `KillEvent` gains `victim_moving: bool | None` (speed from 16 Hz position
  deltas > walk threshold at death), `victim_preaim_off_deg: float | None`
  (victim yaw vs bearing to killer at death tick),
  `victim_weapon: str | None` (new tick prop `active_weapon`; None on old
  lakes).

Serialize changes (all inside `serialize_match`, lake frames in memory):

- Rotation detection: for each trigger event t, for each player whose stint
  at t had lasted >= 4s (a hold, not transit), find the first stint change
  within 8s after t; latency from 16 Hz ticks (first sustained displacement
  from hold position), rounded to 0.1s. One RotationEvent per (trigger,
  player) pair, nearest trigger wins.
- Utility effects: joins described above; each is a small polars window
  join; absent tables (old lakes) leave fields None.
- Death contexts: computed from the tick frame already loaded for movement.
- `extract.py`: add `active_weapon` to TICK_PROPS (new ingests only).

Test first: synthetic mini-lake fixtures per enrichment (mirror
`test_roundscript_utility.py` patterns): a defender breaking hold 1.5s after
a flash detonates two zones away -> RotationEvent(trigger="utility_near",
latency_s~1.5); a flash blinding 2 enemies 1.8s + 1 teammate 0.4s ->
enemy_blind_s=2.2/1.8-ish per convention chosen (document: SUM of enemy
blind seconds, MAX would hide multi-blinds); an HE dealing 34 damage ->
damage=34; a kill with victim stationary + yaw 40 degrees off killer ->
victim_moving=False, victim_preaim_off_deg~40.
Done when: new fixtures pass; existing serialize tests pass; a pre-v3
script JSON round-trips with empty/None new fields.

### Task 2 - Miners: rotation report, utility ROI, death profiles, retake trees

Goal: aggregate the enriched scripts into evidence-cited artifacts.
Difficulty: MEDIUM.
Files: `mining/rotations.py`, `mining/utility_roi.py`, `mining/deaths.py`,
`mining/retakes.py` (all new), `tests/test_mining_rotations.py`,
`tests/test_mining_utility_roi.py`, `tests/test_mining_deaths.py`,
`tests/test_mining_retakes.py`.

- `build_rotation_report(scripts, team) -> RotationReport`: per (player,
  trigger) median latency, over-rotation rate (rotated when trigger was a
  fake: no follow-up contact at trigger zone within 10s), n, evidence.
  Sentences like: "B4RT3Q breaks B anchor 2.1s after A-side utility (n=9,
  fake-follow rate 44%)".
- `build_utility_roi(scripts, team) -> UtilityROI`: per lineup/zone-pattern:
  avg enemy-blind seconds, team-flash seconds, damage per HE/molly, $ per
  point of value (spend from econ), smoke kills_through count. Rows carry n
  + evidence; no dollar verdicts below n=5 - emit the numbers, let the
  model hedge.
- `build_death_profiles(scripts, team) -> DeathProfiles`: per player: deaths
  while moving %, median preaim-off degrees, weapon-out-at-death counts
  (None-tolerant), split by range band (constants.range_band).
- `build_retake_report(scripts, team) -> RetakeReport`: per (site,
  man-diff): retake win rate, and per approach-vector-set (sorted tuple of
  zones the retakers entered the site from, from tracks) win rate, n,
  evidence. Both sides: retakes (CT post-plant) and post-plant holds (T).

Test first per miner with hand-built scripts; assert exact rates, n, and
evidence ids; assert empty inputs -> empty reports (never crash).
Done when: four miners green, ruff clean.

### Task 3 - Chat tools

Goal: the agent can pull each analytic on demand.
Difficulty: EASY-MEDIUM.
Files: `llm/tools.py`, `web/chat.py` (context if needed),
`tests/test_llm_agent.py` or the tools test file.

- `get_rotation_report(side?, trigger?)`, `get_utility_roi(side?)`,
  `get_death_profiles(player?)`, `get_retake_report(site?)` - all render
  the miners over the session's (scoped) scripts, same style as
  get_gap_report.
- `get_matchup(opponent: str)`: resolve the opponent team by name/key on
  the session's map (404 text listing available teams when absent), mine
  BOTH books over their own scripts, emit a diff brief: their top execute
  frequencies vs our gap findings at matching times/zones, their utility
  dumps vs our econ/utility timings, plus both engagement-range profiles.
  Chat-only.
- Tool-rules text in the chat system prompt gains one line each (which
  tool answers what), and the "never claim audio cues" caveat.

Test first: ScriptedToolClient invokes each tool; assert output contains
expected rates/evidence; get_matchup with unknown team returns the
available-team list.
Done when: tool tests green; tools appear in the registry test.

### Task 4 - First Read sections

Goal: the one-shot brief sees the same intelligence.
Difficulty: EASY.
Files: `llm/insights.py`, `tests/test_llm_insights.py`.

- `build_insights_user` gains four compact sections (each capped ~15 rows,
  mirroring existing section style): `## Rotation Responses`,
  `## Utility ROI`, `## Death Contexts`, `## Retake Book`.
- `build_insights_system` section contract UNCHANGED (six sections) - the
  new data informs the same six outputs; add one system line telling the
  model the new blocks exist and what they mean.
- A line after `## Games` noting other booked teams on this map ("matchup
  data available in chat via get_matchup: team_X, team_Y").
- Budget: measured headroom makes this trivial (~1-2k tokens on a 226k
  prompt); assert no timeline changes.

Test first: prompt-construction test asserting all four headers + content
render from enriched synthetic scripts, and that lite timelines are
untouched.
Done when: insights tests green.

### Task 5 - Rebuild + migration + docs

Goal: existing corpora gain the analytics without re-uploading.
Difficulty: EASY.
Files: `web/maintenance.py` (nothing expected - rebuild already
re-serializes; verify), `docs/NEEDS-FROM-YOU.md` note, plan status update.

- Verify `rebuild_map_zones` regenerates scripts with rotations/effects
  (it re-serializes from the lake; `active_weapon` stays None for old
  lakes - fine).
- Operator note: click Rebuild per map (or re-ingest) to enrich existing
  corpora; restart server.
Done when: rebuild e2e test still green; a rebuilt synthetic corpus shows
rotations in scripts.

### Task ordering

1 -> 2 -> 3 -> 4 -> 5. Task 1 is the risk concentrator: do it first, land
it green before any miner. Tasks 2-4 are parallelizable in principle but
land sequentially, one commit each.

## Risks

- Rotation latency semantics: distinguishing "rotated because of trigger"
  from "was already leaving" is heuristic (hold >= 4s precondition + 8s
  response window). The report words it as correlation ("breaks hold within
  Xs of...") - never causation. Documented in tool text.
- Blind-duration attribution joins on thrower + detonation window; team
  flashes through walls still count as enemy-blind if the game says so -
  acceptable, it IS effective blind time.
- `active_weapon` prop name must be verified against demoparser2 on a real
  demo before relying on it (probe first; fall back to omitting
  victim_weapon if unavailable).
- Retake trees on 6-demo corpora will mostly say n<=3; the value scales
  with corpus size. The model's honesty contract already handles this.
- Script schema grows; keep every field optional so old JSON loads; never
  bump a version gate that forces migrations.

## Verification

- `uv run python -m pytest -q` zero failures; `uv run ruff check .` clean.
- Re-run the token count script (scratch/count_first_read_tokens.py) after
  Task 4: total must stay within ~2% of the 226k baseline.
- Manual: rebuild ancient, ask chat "who over-rotates on utility?" and
  "what's their B retake conversion?" - answers cite the new tools.

PLAN COMPLETE

## Outcome (recorded 2026-09-05)

All five tasks landed, one commit each, on master. 449 tests pass
(404 at baseline), ruff clean. Deviations from the plan text, all verified
in-session:

- `extract.py` needed NO change: the tick prop is `active_weapon_name` and
  has been in TICK_PROPS since the first lake commit (60259a6), already
  proven by movement.py's C4-holder detection. Victim weapon therefore
  works on OLD lakes too, not just new ingests.
- `utility_near` triggers on ANY enemy detonation (nearest-in-time claims
  the response) rather than a spatial adjacency test: the plan's own
  fixture ("flash detonates two zones away") and the over-rotation read
  ("B anchor breaks on A-side utility") both require cross-map triggers.
  Documented in the model, the tool text, and both prompts.
- Utility ROI dollar verdicts gate on the MEASURED sub-sample (not raw
  throw count): five throws with one measured effect are one data point.
- Death-context speed uses 16 Hz position deltas as planned; the lake's
  `velocity` prop was probed and rejected (median 281 u/s, max 220k -
  not a usable speed scalar).
- Budget check: the four First Read sections add ~5.3k chars (~1.3k
  tokens) on a real single-match corpus and are hard-capped at 15 rows
  each, so the delta stays constant as the corpus grows: ~1% of the 226k
  baseline, within the ~2% bound. (The live countTokens script was not
  run - it spends API budget; measured in characters instead.)
- Real-lake probe of the full enrichment (match 7d3100eb84266589,
  de_ancient, 14 rounds): 156 rotations (utility_near 123 / first_blood
  27 / plant 6, median latency 0.6s), 81 flashes with blind splits,
  80 HEs and 82 mollies with damage, 3 smoke kills-through, and death
  contexts on all 98 kills (70 with a held weapon; the rest degrade to
  None as designed).
- Rebuild path verified by reading maintenance.py:204 (it calls
  serialize_match on the stored lake, which now emits the enrichment) and
  by the probe above running that exact call against a pre-upgrade lake.
  Operator note added to docs/NEEDS-FROM-YOU.md.
