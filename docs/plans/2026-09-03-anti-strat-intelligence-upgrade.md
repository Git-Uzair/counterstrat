# Anti-Strat Intelligence Upgrade

Date: 2026-09-03
Author: boss (research + design pass)
Status: EXECUTED 2026-09-03 - all 11 tasks landed on master

| Task | Commit |
|---|---|
| 0 multi-demo aggregation | 225f90c |
| 1 hierarchical tendencies | 44b395e |
| 2 utility book | 5d3489f |
| 3 gap miner | 51ef02e |
| 4 econ policy | dfe8ee0 |
| 5 player profiles | 9d2890c |
| 6 analyst tools | ace4e37 |
| 7 chat prompt | 8c46c94 |
| 8 scout brief | 3d5a236 |
| 9 radar v2 | 8a65591 |
| 10 dossier + README | 24c1d11 |

Final verification: `uv run python -m pytest -q` -> 262 passed, 6
deselected (live); `uv run ruff check .` -> clean. Live-data checks:
teambooks + briefs regenerated for all 4 stored (team, map) pairs;
brief and players-filtered layers endpoints verified against the
running server. Deviations from the written plan: `level` lives on
`Tendency` (not `TendencyKey`); playbook situation `anti_eco` shipped
as `eco` (teambook keys on the observed team's own buy); the radar
browser smoke was done at HTTP level plus static pins because
Playwright is unavailable in this environment.

## Context

The product works end-to-end but the intelligence it surfaces is not
IGL-grade. User complaints, each verified against the code this session:

1. **"Response is too long with useless 33%/33%/33% percentages."**
   Root cause is threefold:
   - `build_teambook` partitions rounds into up to 72 buckets per map:
     `k_tuple = (map, side, buy_class, score_bucket, prev_outcome)`
     (`src/counterstrat/mining/tendencies.py:218`). One demo (~24 rounds)
     gives n=1..3 per bucket, so every distribution is uniform noise.
   - `TeamBook.to_table_text()` dumps **every** bucket row into the chat
     system prompt (`src/counterstrat/web/chat.py:137`), so the model
     recites noise.
   - The chat system prompt is the dossier prompt ("comprehensive ...
     seven sections", "quote frequencies verbatim",
     `src/counterstrat/llm/prompts.py:9-31`) plus bolt-on tool rules
     (`src/counterstrat/web/chat.py:34-46`). Nothing constrains length,
     format, or actionability; nothing tells the model to *suppress* noise
     or to end with a counter-call.

2. **"First contact here and there is not the best information."** The
   only mined artifacts are first-contact zone, B+15 formation, site
   committed, lineup-id sets, median FC time
   (`src/counterstrat/mining/tendencies.py:22-32`). No utility book, no
   occupancy/gap analysis, no economy behavior, no per-player depth, no
   tempo profile. The tool surface is 5 tools
   (`src/counterstrat/llm/tools.py:59-129`): tendencies, list_rounds,
   round script, role cards, guarded SQL. (README line 73 advertises an
   `inspect_utility` tool that does not exist.)

3. **"Nothing gives me value before I start asking questions."** Ingest
   ends at TeamBook write (`src/counterstrat/web/ingest.py:205-213`); the
   UI welcome box is static copy (`index.html:85-96`). There is no
   auto-generated scout brief.

4. **Radar**: no zoom/pan (fixed 1024 canvas, `radar.js:10,152-157`);
   trails colored by side only so a single player cannot be traced
   (`radar.js:12,198-218`) even though the backend already ships
   `steamid`/`name`/`clock_s` per trail
   (`src/counterstrat/radar/layers.py:219-233`); utility marks are
   same-shape circles distinguished only by color (`radar.js:220-243`)
   with collisions: smoke `#c9d1d9` vs decoy `#8b949e`, HE `#f85149` ==
   kill-against red, molotov `#db6d28` ~= T orange `#f0883e`.

5. **"Where does the other team leave gaps, and what triggers it?"** Not
   mineable today by the LLM except via raw SQL. Beats already carry
   per-side zone formations at B+00/B+15/B+30/... plus FC/PL anchors
   (`src/counterstrat/roundscript/models.py:15-20`), so vacancy analysis
   is a pure mining problem over existing data.

## Research digest (what the answers should sound like)

Sources read this session: Dignitas "The Art of Anti-Stratting"
(dignitas.gg/articles/the-art-of-anti-stratting); NaToSaphiX pre-round
calling (readtldr.gg/csgo-news/pre-round-calls-from-a-pro-igl); Boosteria
"CS2 T-Side Fundamentals 2026: Takes, Fakes, Executes" and "How to Hold
Sites as CT in CS2 2026" (boosteria.org); Scope.gg / Leetify / noesis.gg
feature surveys (noesis is what G2's analyst uses for opponent tendency
prep: 2D replay, multi-round comparison, "punish predictability").

Framework distilled for the prompts and mining below:

- **Anti-strat atomic unit = trigger -> response -> punish.** "When they
  smoke mid doors, they run catwalk with two lurkers -> pre-aim catwalk,
  hold B tunnels for the lurk." Reads are indicator-conditioned, not
  frequency recitals. (Dignitas)
- **Five IGL calling lenses** (NaToSaphiX): previous rounds (momentum:
  won 3 fast-A -> they stack A -> go slow B), playbook knowledge
  ("I know they have this aggressive B push"), contingency (set play with
  a branch), goal-based (default with a timed objective), freestyle
  (default -> mid-round read). Pre-match prep = solidify your own first
  1-2 buy rounds; anti-strat material rarely exists for pistols.
- **T-side pillars**: information, pressure, space, timing, conversion.
  Site take = 6-stage chain (prep, vision isolation, entry, trade,
  stabilize, plant/post-plant). Fake taxonomy: micro / pressure / sell /
  delayed. Utility credibility decays if patterns never vary. Tempo
  change (still -> explode) wins duels by rhythm break. (Boosteria T)
- **CT-side structure**: roles anchor/support/rotator/lurker-catcher;
  default archetypes 2-1-2, 3-0-2, 1-2-2, passive-retake; rotation
  taxonomy lean/leave/freeze. **Gaps are created by**: (a) rotations
  triggered on weak info, (b) utility dumped early -> naked execute
  window when it fades, (c) over-aggression after an opening kill,
  (d) conditioned stacking after repeated losses to one site, (e) mid
  control lost without re-shaping the defense, (f) trickle rotations.
  (Boosteria CT) These six are exactly the trigger vocabulary the gap
  miner should condition on.
- **Economy reads**: post-pistol-loss force meta, anti-eco risk (SMG
  farm vs surprise force), loss-bonus rhythm eco->full. Buy prediction
  from loss streak + observed policy is a real edge. (ProSettings et al.)
- **What good answers look like**: short, exploit-first, evidence-cited,
  ending in a concrete counter-call with a confidence grade - the way a
  coach hands a card to an IGL, not the way a stats site prints a table.

## Assumptions

- Corpus stays small (1-10 demos per team) in v1; the design must be
  honest at n=9, not only at n=300. Noise suppression > more decimals.
- All new mining runs offline from existing `RoundScript` artifacts (and
  optionally lake parquet); no new dependencies. DBSCAN/lineups, beats,
  movements, blinded lists already exist in the scripts.
- No streaming, no auth, localhost single analyst (unchanged).
- Radar playback/scrubber, sightline analysis, and veto analysis are OUT
  of scope for this plan (noted in Risks/Future).
- Tests follow existing patterns: offline `synthetic_scripts` /
  `session_ctx` fixtures (`tests/conftest.py`), `demo`-marked tests for
  real fixtures, TestClient for web.

## Task overview

| # | Task | Difficulty |
|---|---|---|
| 0 | Fix multi-demo aggregation: teambook must accumulate | Easy-Medium |
| 1 | Hierarchical tendencies with signal filtering | Medium |
| 2 | Utility book miner | Medium |
| 3 | Gap/vacancy miner with triggers | Hard |
| 4 | Economy behavior miner | Easy-Medium |
| 5 | Player profile miner (extend RoleCard) | Medium |
| 6 | New analyst tools exposing 2-5 | Medium |
| 7 | IGL-grade chat prompt + trimmed system block | Medium |
| 8 | Instant scout brief (ingest + API + UI) | Medium |
| 9 | Radar: zoom/pan, player trace, utility glyphs | Medium-Hard |
| 10 | Dossier prompt alignment + README tool list fix | Easy |

Recommended execution order: 0 first (foundational - every miner and the
brief inherit it), then 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 (needs
1-5), 9 anytime (independent), 10 last.

---

## Task 0 - Fix multi-demo aggregation: teambook must accumulate

**Goal**: when two demos of TEAM A on the same map have been ingested,
the mined TeamBook (and everything downstream: tendencies, role cards,
chat scripts, dossier, brief) must cover BOTH demos. Today it does not.

**Bug trace** (verified 2026-09-03):
- `run_ingest` mines from only the current demo:
  `scripts = serialize_match(...)` (`src/counterstrat/web/ingest.py:185`)
  then `tb = build_teambook(scripts, tk)` (`ingest.py:206`) and
  **overwrites** `data/teambooks/<tk>/<map>/teambook.json`
  (`ingest.py:207-209`). The only other `build_teambook` call site is
  the eval harness (`eval/benchmark.py:439`).
- Chat then loads round scripts only for
  `teambook.generated_from` (`src/counterstrat/web/chat.py:157-158`),
  so the earlier demo's rounds are invisible to every tool except raw
  `sql_query` (which spans `lake/*`, `src/counterstrat/lake/duck.py:23`).
- Contradicts the documented promise: "the app auto-groups uploads by
  roster, so 'team X' accumulates across files"
  (`docs/NEEDS-FROM-YOU.md`, N4). The roster-derived `team_key` already
  lands both demos on the same teambook path - which is exactly why the
  second ingest clobbers the first.
- `build_teambook` itself already handles multi-match input correctly
  (groups by match_id for prev_outcome,
  `src/counterstrat/mining/tendencies.py:174-183`; `generated_from` is
  the set of match_ids, `tendencies.py:368`). The fix is purely what the
  ingest feeds it.

**Difficulty**: Easy-Medium

**Files**:
- `src/counterstrat/web/ingest.py`
- `tests/test_web_api.py` (or a new `tests/test_ingest_accumulate.py`)

**Test first** (offline; build two synthetic script sets with distinct
match_ids and the same team key, write them under a tmp
`data/scripts/<match_id>/`, then run the mining stage twice):

```python
def test_second_ingest_accumulates_teambook(tmp_path, synthetic_scripts):
    # persist match 1 scripts, mine, then mine match 2 and assert union
    ...
    tb = TeamBook.model_validate_json(tb_path.read_text(encoding="utf-8"))
    assert set(tb.generated_from) == {"match_one", "match_two"}
    t_rounds = sum(t.n for t in tb.tendencies if t.key.side == "T" and t.level == 2)
    assert t_rounds == expected_t_rounds_across_both
```

(Exercise the extracted helper below directly rather than full
`run_ingest`, which needs a real demo.)

**Change**: in `run_ingest`'s mining stage (`ingest.py:192-209`):

1. Extract a helper
   `_scripts_for_team(data_root: Path, map_name: str, team_key: str,
   current: list[RoundScript]) -> list[RoundScript]`:
   - read `corpus.jsonl` via `load_manifest`, take match_ids with
     `rec.map_name == map_name`, excluding the current match;
   - load each `data/scripts/<match_id>/round_*.json` (skip unreadable
     files with a warning, same pattern as `chat.py:159-164`);
   - keep scripts where the team participates
     (`s.t_team_key == team_key or s.ct_team_key == team_key`);
   - return prior + current.
2. `tb = build_teambook(_scripts_for_team(...), tk)`.
3. Tasks 2-5 miners and the Task 8 brief must be fed the same
   accumulated list inside `run_ingest` (one load, reused for all
   miners per team).
4. Chat needs no change: it already fans out from
   `teambook.generated_from`, which now lists every contributing match.

**Done when**: the new test passes; ingesting the fixture demo still
passes existing web tests; after two ingests of same-team demos (manual
or test-simulated), `generated_from` lists both match ids and tendency
`n`s sum across demos.

Verification commands (declared in `pyproject.toml`):
`uv run python -m pytest -q` (deselects `live` by default) and
`uv run ruff check .` (line-length 100).

---

## Task 1 - Hierarchical tendencies with signal filtering

**Goal**: kill the 33/33/33 problem at the source. Mine the same stats at
three aggregation levels and mark which rows carry signal, so both the
system prompt and the tools can prefer the coarsest level that has
meaningful n and only quote concentrated distributions.

**Difficulty**: Medium

**Files**:
- `src/counterstrat/mining/tendencies.py`
- `tests/test_mining_tendencies.py`

**Test first** (extend `tests/test_mining_tendencies.py`, reuse
`synthetic_scripts` fixture from `tests/conftest.py`):

```python
def test_teambook_has_aggregate_levels(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    levels = {t.level for t in tb.tendencies}
    assert levels == {0, 1, 2}
    l0_t = [t for t in tb.tendencies if t.level == 0 and t.key.side == "T"]
    assert len(l0_t) == 1                      # one side-wide rollup
    assert l0_t[0].n == 4                      # all 4 synthetic T rounds

def test_concentration_and_signal_flags(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    for t in tb.tendencies:
        if t.first_contact_zone:
            top = max(t.first_contact_zone.values())
            assert abs(t.fc_concentration - top) < 1e-9
        assert t.signal == (t.n >= 3 and t.fc_concentration >= 0.5)

def test_table_text_hides_noise(synthetic_scripts):
    tb = build_teambook(synthetic_scripts, "abc")
    txt = tb.to_table_text()
    for t in tb.tendencies:
        if t.level == 2 and not t.signal:
            # noisy fine-grained rows must not be rendered
            assert f"| {t.key.buy_class} | {t.key.score_bucket} | {t.key.prev_outcome} |" not in txt
```

Confirm each fails (`Tendency` has no `level` attr today) before
implementing.

**Change**:

1. `TendencyKey`: add `level: int = 2` plus make `score_bucket` and
   `prev_outcome` default to `"any"` so coarser keys reuse the model:
   - level 0 key = `(map, side, "any", "any", "any")`
   - level 1 key = `(map, side, buy_class, "any", "any")`
   - level 2 key = existing full key.
2. In `build_teambook`, after the existing level-2 grouping loop
   (`tendencies.py:180-219`), also group the same per-round records into
   level-1 (`(map, side, buy)`) and level-0 (`(map, side)`) buckets and
   emit a `Tendency` per bucket via the same aggregation block (factor
   lines 222-302 into `_mine_bucket(key, g_scripts) -> Tendency` so all
   three levels share it).
3. `Tendency`: add fields
   - `level: int`
   - `fc_concentration: float` (max share in `first_contact_zone`, 0.0
     when empty)
   - `site_concentration: float` (max share in `site_committed`)
   - `signal: bool` = `n >= 3 and fc_concentration >= 0.5`
   The 0.5 threshold is a named module constant
   `SIGNAL_MIN_CONCENTRATION = 0.5`, `SIGNAL_MIN_N = 3`.
4. `to_table_text()`: render level-0 and level-1 rows always, level-2
   rows only when `signal` is true. Add a `Level` column. Keep `low_n`.
5. `to_sentences()`: iterate signal rows only, most specific level first,
   and skip a coarser sentence when a more specific signal row for the
   same side+buy exists.
6. Keep JSON schema backward-compatible: old teambook.json files load
   because the new fields have defaults (`level=2`,
   concentrations computed... no - persisted books lack them). Give all
   new fields pydantic defaults (`level=2`, `fc_concentration=0.0`,
   `site_concentration=0.0`, `signal=False`) so stored books still
   validate; re-ingest regenerates them.

**Done when**: new tests pass; `test_web_chat.py` and
`test_llm_dossier.py` still pass (they consume `to_table_text`);
`uv run ruff check .` clean.

---

## Task 2 - Utility book miner

**Goal**: answer "where do they throw smokes on B site?" and "what is
their exec package?" from mined data instead of ad-hoc SQL. This is the
`inspect_utility` capability the README already promises.

**Difficulty**: Medium

**Files**:
- `src/counterstrat/mining/utility_book.py` (new)
- `src/counterstrat/mining/__init__.py` (export)
- `tests/test_mining_utility_book.py` (new)

**Test first**:

```python
from counterstrat.mining.utility_book import build_utility_book

def test_utility_book_groups_by_nade_and_target(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, "abc")
    smokes = [p for p in book.patterns if p.nade == "smoke"]
    assert smokes, "synthetic fixture throws smokes"
    p = smokes[0]
    assert p.count >= 1 and p.to_zone and p.evidence
    assert p.median_t >= 0
    assert 0 < p.share <= 1.0

def test_utility_book_respects_side(synthetic_scripts):
    book = build_utility_book(synthetic_scripts, "abc")
    assert {p.side for p in book.patterns} <= {"T", "CT"}
```

**Change**: new module with:

```python
class UtilityPattern(BaseModel):
    side: str                      # "T" | "CT"
    nade: str                      # smoke | flash | he | molotov | decoy
    to_zone: str
    from_zones: dict[str, int]     # origin zone -> count
    lineup_id: str | None          # dominant cluster id if any
    count: int                     # rounds containing this pattern
    rounds_seen: int               # rounds where side threw >=1 such nade
    share: float                   # count / rounds_seen
    median_t: float                # median throw clock_s
    early_share: float             # share thrown at t < 25s (opening package)
    evidence: list[str]            # "match:round", capped at 6

class UtilityBook(BaseModel):
    team_key: str
    map_name: str
    patterns: list[UtilityPattern]   # sorted count desc, capped at 40
    dump_windows: list[dict]         # see below

def build_utility_book(scripts: list[RoundScript], team_key: str) -> UtilityBook: ...
```

Implementation notes:
- Determine the team's side per script exactly as
  `build_teambook` does (`tendencies.py:187`).
- Group `s.utility` events with `u.side == team_side` by
  `(side, nade, to_zone)`; count a round at most once per pattern;
  `median_t` from event `t`s; `lineup_id` = most common non-null id in
  the group.
- `dump_windows`: per side, the median t of the k-th nade of the round
  (k=1..4) - this is the "their utility is spent by ~35s" read the gap
  miner and prompts reuse: `{"side": "T", "kth": 3, "median_t": 34.0,
  "n": 7}`.
- Cap `patterns` at 40 by count desc to bound prompt/tool payloads.

**Done when**: tests pass; book builds from the real fixture demo too
(`@pytest.mark.demo` smoke test asserting non-empty patterns).

---

## Task 3 - Gap/vacancy miner with triggers

**Goal**: answer "where do they leave gaps, when, and what triggers it"
deterministically: systematic zone vacancies by time window, conditioned
on the six research-derived triggers.

**Difficulty**: Hard (the most novel logic; keep v1 scope tight)

**Files**:
- `src/counterstrat/mining/gaps.py` (new)
- `src/counterstrat/mining/__init__.py` (export)
- `tests/test_mining_gaps.py` (new)

**Key design**: operate on `Beat` formations (already per-side zone
count lists at B+00/B+15/B+30/... plus FC+t/PL+t anchors,
`roundscript/models.py:15-20`). A "key zone" set comes from the MapCard
objectives/topology when available, else the union of zones observed in
plants + site names. v1 trigger vocabulary (all computable from
RoundScript alone):

- `after_fc_win` / `after_fc_loss`: first contact won or lost by the
  observed team before the beat.
- `after_util_dump`: >= 3 of the observed side's nades thrown before the
  beat time.
- `behind` / `ahead`: score bucket of the round.
- `after_loss` / `after_win`: previous-round outcome (reuse the
  prev_outcome computation from `build_teambook`).
- `base`: unconditioned.

```python
class GapFinding(BaseModel):
    side: str                 # side of the OBSERVED team (the opponent you attack)
    zone: str                 # vacated zone
    window: str               # beat label, e.g. "B+30"
    trigger: str              # vocabulary above
    vacancy_rate: float       # rounds vacant / rounds matching trigger
    n: int                    # rounds matching trigger with this beat present
    baseline_rate: float      # unconditioned vacancy rate for (zone, window)
    lift: float               # vacancy_rate - baseline_rate
    evidence: list[str]       # capped at 6

class GapReport(BaseModel):
    team_key: str
    map_name: str
    key_zones: list[str]
    findings: list[GapFinding]   # signal-filtered, sorted by lift desc

def build_gap_report(
    scripts: list[RoundScript],
    team_key: str,
    key_zones: list[str] | None = None,
) -> GapReport: ...
```

Implementation notes:
- A zone is *vacant at a beat* when it does not appear in that side's
  formation zones for that beat (count 0).
- Only emit findings with `n >= 3`, `vacancy_rate >= 0.6`, and
  `lift >= 0.15` for triggered rows (constants at module top); always
  emit `base` rows with `n >= 3` regardless of lift (baseline coverage
  map). Cap findings at 30.
- Beat labels beyond B+45 are sparse; normalize windows to
  `B+00/B+15/B+30/B+45/post-FC/post-PL` where post-FC = the FC-anchored
  beat, post-PL = plant-anchored beat.
- The report is about where the OBSERVED team's own side leaves holes:
  when scouting their CT half, `side == "CT"` findings are what a T-side
  caller attacks; the tool layer (Task 6) passes the side through.

**Test first** (synthetic fixture has deterministic formations):

```python
def test_gap_report_finds_uncovered_key_zone(synthetic_scripts, synthetic_card):
    rep = build_gap_report(synthetic_scripts, "abc",
                           key_zones=list(synthetic_card.zones.keys()))
    assert rep.findings, "expected at least a base coverage finding"
    f = rep.findings[0]
    assert 0 <= f.vacancy_rate <= 1 and f.n >= 3
    assert f.window.startswith(("B+", "post-"))

def test_gap_trigger_rows_report_lift(synthetic_scripts):
    rep = build_gap_report(synthetic_scripts, "abc")
    for f in rep.findings:
        if f.trigger != "base":
            assert f.lift >= 0.15 and f.vacancy_rate >= 0.6
```

If the 9-round synthetic fixture cannot produce a triggered finding,
extend the fixture with 2-3 extra scripts engineered to vacate a zone
after `after_fc_loss` - do that inside the test module, not by mutating
`conftest.py` (keep the shared fixture stable).

**Done when**: tests pass; a `@pytest.mark.demo` test builds a report
from the real demo without raising and with >= 1 base finding.

---

## Task 4 - Economy behavior miner

**Goal**: answer "what do they buy after losing pistol / on a loss
streak?" and let the analyst predict the next buy.

**Difficulty**: Easy-Medium

**Files**:
- `src/counterstrat/mining/econ_policy.py` (new)
- `tests/test_mining_econ_policy.py` (new)

**Change**:

```python
class EconPolicy(BaseModel):
    team_key: str
    map_name: str
    # state -> {buy_class -> share}; states below
    policy: dict[str, dict[str, float]]
    ns: dict[str, int]              # state -> n
    pistol_round_sites: dict[str, float]   # site committed on rounds 1 & 13 (T only)
    post_pistol_loss_buy: dict[str, float] # buy_class shares in rounds 2 & 14 after losing pistol
    evidence: dict[str, list[str]]  # state -> up to 4 round ids

def build_econ_policy(scripts: list[RoundScript], team_key: str) -> EconPolicy: ...
```

States (strings): `"after_loss_1"`, `"after_loss_2"`,
`"after_loss_3plus"`, `"after_win"`, `"pistol"`. Loss streak computed
per match from round order and `winner` exactly as
`build_teambook`'s prev_outcome logic (`tendencies.py:190-202`), extended
to a running streak counter. Buy class comes from
`s.economy[side].buy_type`.

**Test first**: feed a hand-built list of minimal `RoundScript`s (reuse
the constructor pattern from `tests/conftest.py::synthetic_scripts`)
where the team loses rounds 1-3 then full-buys round 4; assert
`policy["after_loss_3plus"]["full_buy"] == 1.0` and
`ns["after_loss_3plus"] == 1`.

**Done when**: tests pass; policy JSON is < 4 KB for the fixture demo.

---

## Task 5 - Player profile miner (extend RoleCard)

**Goal**: per-player reads an IGL can use: who entries where, who lurks,
who carries the AWP, who wins/loses their openers.

**Difficulty**: Medium

**Files**:
- `src/counterstrat/mining/tendencies.py` (extend `RoleCard` +
  `build_teambook` role loop, `tendencies.py:315-364`)
- `tests/test_mining_tendencies.py`

**Change**: add to `RoleCard` (all with defaults so old JSON loads):

```python
opening_kill_rate: float = 0.0      # FC involvements that were kills (not deaths)
opening_zones: dict[str, int] = {}  # zone -> count of FC involvements
awp_rounds: int = 0                 # rounds with an awp kill by this player
trade_discipline: float = 0.0       # share of their deaths traded within 4s
median_fc_t: float | None = None    # median FC time when involved
```

All computable from `s.kills` / `s.first_contact` (killer, victim,
weapon, traded_within_4s fields already exist,
`roundscript/models.py:22-30`). `awp_rounds` counts rounds where any
kill has `"awp" in KillEvent.weapon.lower()` with `killer == player`
(weapon strings in the lake are lowercase ids like `"ak47"`,
`tests/conftest.py:139` - match case-insensitively anyway).
Note: the synthetic fixture team key is `"abc"` and the opponent is
`"xyz"` (`tests/conftest.py:93-94`); all test snippets in this plan use
`"abc"`.

**Test first**: assert on the synthetic fixture that a player who takes
the first kill in 2 of 4 T rounds gets `opening_kill_rate == 1.0`,
`opening_zones` non-empty, and that `trade_discipline` is within [0,1].

**Done when**: tests pass; `get_role_cards` tool output (Task 6) carries
the new fields.

---

## Task 6 - New analyst tools

**Goal**: give the model niche, cheap, targeted retrieval so it stops
reasoning from one big table. Five new tools, each a thin JSON view over
Tasks 1-5 artifacts.

**Difficulty**: Medium

**Files**:
- `src/counterstrat/llm/tools.py`
- `src/counterstrat/web/chat.py` (build the new artifacts into
  `SessionContext` at `_build_session`, chat.py:305-313)
- `tests/test_llm_agent.py`, new `tests/test_llm_tools_v2.py`

**Change**:

1. Extend `SessionContext` with `utility_book: UtilityBook | None`,
   `gap_report: GapReport | None`, `econ_policy: EconPolicy | None`
   (defaults None; built in `_build_session` from the already-loaded
   `scripts` dict values - all three miners take `list[RoundScript]`).
2. New tool specs + handlers (mirror the existing pattern
   `tools.py:141-233`):
   - `get_playbook(situation)` - situation enum:
     `pistol | anti_eco | force | full_buy | after_loss | after_win`.
     Returns the *matching aggregation level* rows from Task 1 (signal
     rows only), the matching `EconPolicy` state, and matching utility
     `dump_windows`. This is the one-call answer for "what do they do on
     X rounds".
   - `get_utility_book(side, nade?, to_zone?)` - filtered
     `UtilityPattern` rows.
   - `get_gap_report(side, trigger?)` - filtered `GapFinding` rows,
     always including the matching base rows so the model can quote lift
     honestly.
   - `get_economy_read()` - the full `EconPolicy`.
   - `get_player_profile(player?)` - one or all extended RoleCards.
3. Each handler returns `{"error": ...}` JSON instead of raising when
   the artifact is None (degraded sessions), same convention as
   `_tool_sql_query` (`tools.py:214-215`).
4. Update the `get_tendencies` spec description to mention levels and
   the `signal` flag; add optional `level` arg (default: signal rows at
   any level).

**Test first** (offline, extend `session_ctx` fixture usage):

```python
def test_get_playbook_pistol_returns_signal_rows(session_ctx_v2):
    out = json.loads(execute_tool(session_ctx_v2,
        ToolCall(id="1", name="get_playbook", arguments={"situation": "pistol"})))
    assert "tendencies" in out and "econ" in out
    assert all(t["signal"] for t in out["tendencies"])

def test_gap_tool_filters_trigger(session_ctx_v2):
    out = json.loads(execute_tool(session_ctx_v2,
        ToolCall(id="1", name="get_gap_report",
                 arguments={"side": "CT", "trigger": "after_util_dump"})))
    assert set(f["trigger"] for f in out["findings"]) <= {"after_util_dump", "base"}
```

Add a `session_ctx_v2` fixture in `tests/test_llm_tools_v2.py` that
wraps the existing `session_ctx` and attaches the three new artifacts
built from `synthetic_scripts`.

**Done when**: tool tests pass; `run_agent` loop still passes
`tests/test_llm_agent.py`; tool count and names asserted in one test so
provider adapters stay in sync.

---

## Task 7 - IGL-grade chat prompt + trimmed system block

**Goal**: answers that read like a coach's card: exploit first, evidence,
counter-call, confidence - short by default, noise suppressed.

**Difficulty**: Medium (prose-heavy; the tests pin the contract)

**Files**:
- `src/counterstrat/llm/prompts.py` (new `build_chat_system(card_yaml,
  teambook)` function - keep `build_system` for the dossier)
- `src/counterstrat/web/chat.py` (`_system_prompt` and
  `CHAT_TOOL_RULES` replaced by the new builder)
- `tests/test_web_chat.py`

**Change**: `_system_prompt` becomes
`build_chat_system(card.to_yaml(), teambook)` producing:

1. Role: *"You are a CS2 anti-strat analyst briefing an in-game leader
   mid-preparation. You think in triggers and punishes, not averages."*
2. The map card block (unchanged, cacheable).
3. **Headline signals** instead of the full table: only
   `to_table_text()` output after Task 1 filtering (level-0/1 + signal
   level-2 rows).
4. Doctrine block (the research, ~25 lines): the five calling lenses;
   trigger->response->punish as the unit of advice; the six gap causes;
   fake taxonomy; utility credibility; tempo change; pistols rarely have
   reads (say so and give a solid-default recommendation instead).
5. Output contract:
   - Default answer <= 180 words. Structure: **Read** (1-2 sentences,
     the exploit), **Evidence** (bulleted `match:round` cites with n),
     **Counter-call** (the concrete instruction a caller can give),
     **Confidence** (high/medium/low + why).
   - Never quote a distribution whose top share < 50% or n < 3 as if it
     were signal: either aggregate up a level (tools support it) or say
     "no read - they are unpredictable here" and recommend the solid
     default.
   - "Longer" or "full breakdown" in the user message lifts the length
     cap.
   - Keep existing rules: backticked zones from the card only, cite ids
     from tool output only, state n, hedge low_n.

**Test first** (mocked client path already exists,
`chat.py:364-403`):

```python
def test_chat_system_prompt_contract(client_with_ingested_fixture):
    # build a session, inspect app.state session store
    ...
    assert "Counter-call" in session.system
    assert "180 words" in session.system
    assert "to_table_text" noise rows absent  # reuse Task 1 test idea
```

Practically: assert the system prompt contains the four section names
and does NOT contain a known noisy level-2 row from the fixture
teambook. Existing tests asserting the old CHAT_TOOL_RULES text must be
updated in the same commit.

**Done when**: `tests/test_web_chat.py` green; manual smoke: ask the
mock-mode chat (`?mock=1`) a question and see the reply recorded; a live
(`-m live`) question to Anthropic returns a <= ~180-word exploit-first
answer (spot-check, not CI).

---

## Task 8 - Instant scout brief

**Goal**: the moment ingest finishes (and whenever a target is selected),
the analyst sees a deterministic "First Look" card - top exploits with
n - before typing anything. No LLM call, so it is free and instant.

**Difficulty**: Medium

**Files**:
- `src/counterstrat/mining/brief.py` (new)
- `src/counterstrat/web/ingest.py` (hook after the teambook loop,
  `ingest.py:205-209`)
- `src/counterstrat/web/routes.py` (new endpoint)
- `src/counterstrat/web/static/app.js`, `index.html`, `style.css`
- `tests/test_mining_brief.py`, `tests/test_web_api.py`

**Change**:

1. `build_scout_brief(scripts, team_key, utility_book, gap_report,
   econ_policy, teambook) -> ScoutBrief` producing at most 8 headline
   items, each `{kind, text, n, confidence, evidence}` with kinds:
   `site_lean` (from level-1 signal tendencies), `tempo` (median FC/exec
   time vs 115s round), `utility_crutch` (top pattern with share >= 0.5),
   `gap` (top lift finding), `econ_tell` (e.g. "always force after
   losing pistol", share >= 0.75), `opener` (highest opening_kill_rate
   player), `pistol` (site lean on rounds 1/13 if n >= 2). Text is
   template-generated, e.g.:
   `"Full-buy T: 71% of executes hit BombsiteB (n=7) - stack B utility"`.
   Every item carries its evidence ids. Skip kinds whose thresholds are
   not met - an honest empty brief beats a noisy one.
2. Persist to
   `data/teambooks/<team_key>/<map>/scout_brief.json` in `run_ingest`
   right after each teambook write (build the three artifacts once per
   team from the already-serialized scripts).
3. `GET /api/teams/{team_key}/{map_name}/brief` -> the JSON (404 when
   missing, same shape as the teambook endpoint in `routes.py`).
4. UI: when a catalog target is selected (`app.js` target selection
   handler), fetch the brief and render a "First Look" panel above the
   chat welcome box: headline list with confidence badges and a
   "ask about this" affordance that pre-fills the chat input with a
   question about that item. Keep it visible until the first user
   message.

**Test first**:
- `test_scout_brief_headlines(synthetic_scripts, ...)`: brief has >= 1
  item, all items have n >= its kind threshold, evidence non-empty,
  text contains a backticked zone for zone-bearing kinds.
- Web test: after a mocked ingest (reuse the pattern in
  `tests/test_web_api.py`), `GET /api/teams/<tk>/<map>/brief` returns
  200 with the persisted items; unknown team -> 404.

**Done when**: brief JSON exists after fixture ingest
(`@pytest.mark.demo` path), endpoint tested, UI renders it (covered by
`tests/test_web_static.py`-style asset assertions: the new DOM ids exist
in `index.html`, `app.js` references the endpoint).

---

## Task 9 - Radar: zoom/pan, player tracing, utility glyphs

**Goal**: make the radar usable for actual film study: zoom into a site,
follow one player, tell nade types apart at a glance.

**Difficulty**: Medium-Hard (pure frontend except one query param)

**Files**:
- `src/counterstrat/web/static/radar.js` (bulk of the work)
- `src/counterstrat/web/static/index.html` (controls + legend)
- `src/counterstrat/web/static/style.css`
- `src/counterstrat/web/radar_api.py` + `src/counterstrat/radar/layers.py`
  (optional `players` filter)
- `tests/test_web_radar_static.py`, `tests/test_radar_layers.py`

**Change**:

1. **View transform**: introduce `state.view = {k: 1, tx: 0, ty: 0}`
   (scale + translation in canvas px). Every draw call goes through
   `ctx.setTransform(k, 0, 0, k, tx, ty)` once in `draw()`; hit
   coordinates stay normalized so only `draw()` changes. Interactions:
   - wheel: zoom toward cursor, `k` clamped to [1, 12]; when k==1 reset
     tx/ty to 0.
   - pointer drag: pan, clamped so the image never leaves the frame.
   - double-click or a "Reset view" button: back to identity.
   - the underlying `<img>` must transform identically: simplest is to
     draw the radar PNG onto the canvas itself (drawImage as the first
     layer) instead of stacking img+canvas; that removes the CSS/canvas
     sync problem. Keep the `<img>` element hidden as the decode source.
   - line widths and marker radii divide by `k` (or use
     `ctx.lineWidth = base / k`) so strokes do not fatten when zoomed.
2. **Player tracing**: build a roster panel from the trails payload
   (`name`/`steamid` already present, `layers.py:219-233`): checkbox per
   player plus a "solo" radio behavior (clicking a name isolates that
   player, click again to restore all). Selected players draw with a
   10-color colorblind-safe palette (Okabe-Ito: #E69F00 #56B4E9 #009E73
   #F0E442 #0072B2 #D55E00 #CC79A7 #999999 + two extras); unselected
   trails render at 0.12 alpha in side color. Trail endpoints get the
   player initial at the last point when k >= 2 (`ctx.fillText`).
   Optionally pass `players=<steamid,csv>` to the layers endpoint
   (add to `LayerFilters` + `build_team_scope` member filter) so heatmap
   and utility also scope to the selection - do this, it is ~15 lines
   server-side, and it directly answers "trace a single player alone".
3. **Utility glyphs**: one shape per kind, drawn in a `drawGlyph(ctx,
   kind, x, y, r)` helper: smoke = circle with inner dot; flash =
   4-point star; he = diamond; molotov = triangle; decoy = hollow
   square. New palette distinct from side/kill colors: smoke #9ecbff,
   flash #ffd24d, he #b07cff, molotov #ff7a45, decoy #7ee2b8. Kill
   markers stay green/red crosses; bomb plant becomes a bold "C4" label
   in a rounded rect so it can never be confused with a T trail dot.
   Legend in `index.html:146-154` updated to the new shapes (small
   inline SVGs), and a hover tooltip (`mousemove` hit-test within 12/k
   px) shows `kind - thrower, Rn @ mm:ss` for utility and
   `killer > victim (weapon)` for duels.
4. Keep all existing filters working; `summarize()` unchanged.

**Test first / verification**:
- `tests/test_radar_layers.py`: add a `players` filter test - scope
  members restricted to one steamid yields trails only for that player
  (pure polars, offline).
- `tests/test_web_radar_static.py`: assert new DOM ids
  (`radar-reset-view`, `radar-players`, legend glyph ids) exist and
  `radar.js` contains `setTransform` and the Okabe-Ito palette anchor
  (cheap regression pins, same style as the existing static tests).
- Manual: `uv run counterstrat`, load fixture team, zoom into B site,
  isolate one player, hover a smoke. (Playwright via webapp-testing
  skill if a scripted check is wanted, but do not add it to CI.)

**Done when**: static + layer tests green; manual zoom/trace/tooltip
smoke passes on the fixture demo.

---

## Task 10 - Dossier prompt alignment + README fix

**Goal**: the dossier reads like the chat: exploit-first sections, a Gaps
section, and the README stops advertising a tool that does not exist.

**Difficulty**: Easy

**Files**:
- `src/counterstrat/llm/prompts.py` (`build_system` section list)
- `src/counterstrat/llm/dossier.py` (section names if asserted)
- `README.md:73` (tool list -> actual names incl. new tools)
- `tests/test_llm_dossier.py`

**Change**: dossier sections become: 1. Identity & Overview, 2. Defaults
& Roles, 3. Execute Repertoire with Counters, 4. Gaps & Triggers (new -
from `get_gap_report` data), 5. Economy Policy with Exploit, 6.
Player-Specific Weaknesses, 7. Round-State Playbook Table, 8. Confidence
& Evidence Appendix. Each section must open with a one-line **Exploit**
sentence. Update `lint_dossier` section checks if they pin the old list;
update README's tool list to the real set. `build_user` (prompts.py:62)
additionally includes the utility book top-10 and gap findings so the
dossier is grounded in the new artifacts.

**Done when**: dossier tests green; README lists the real tools.

---

## Out of scope (explicitly deferred)

- Round playback/scrubber on the radar (big UI lift; layers already
  carry `clock_s` so it is a natural v2).
- Sightline/visibility analysis (per NEEDS-FROM-YOU locked decisions).
- Cross-map meta analysis, veto assistance (needs multi-map corpora).
- Live per-round "next buy" prediction UI (predict.py exists; wire it
  into chat later via a `predict_next_round` tool once EconPolicy lands).
- Voice/chat-log analysis (out of scope per locked decisions).

## Risks

- **Small-n honesty**: thresholds (n>=3, 50% concentration, 0.6 vacancy,
  0.15 lift) are opinionated defaults; with one demo many buckets stay
  silent. That is intended - the brief and prompts say "no read" rather
  than inventing one. Revisit thresholds when a real multi-demo corpus
  lands (NEEDS-FROM-YOU N4).
- **Beat granularity**: 15s windows can miss short rotations; v1 accepts
  this (documented in gap tool description) rather than re-mining ticks.
- **Prompt regressions**: chat/dossier tests pin structure, not wording;
  live spot-checks stay manual (`-m live`).
- **Canvas rewrite risk** (Task 9): drawing the PNG into the canvas
  changes the DOM contract; keep the `<img>` element (hidden) so
  existing tests that reference `radar-image` still pass, and adjust
  those that assert visibility.
- **Stored artifact compatibility**: all extended pydantic models take
  defaults so pre-upgrade JSON still validates; a re-ingest of the
  fixture refreshes artifacts (document in commit message).

PLAN COMPLETE
