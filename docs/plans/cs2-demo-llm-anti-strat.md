# CS2 Demo Files → LLM-Powered Anti-Strat: Research Report & Plan

Date: 2026-09-02 (rev. 2, same day)
Status: Research complete, plan proposed. Rev. 2 adds the demo→LLM
representation design in full: §5.2b (graph/space-encoding evidence),
§6.5 Map Card, §6.6 RoundScript v2 grammar, §6.7 what cannot be decided
without real demos, Phases 2a/2b, and the extended Phase-5 ablation suite.
Scope: How CS2 demos are parsed, what can be extracted, prior art (academic,
commercial, open source), and — the novel core — how to transform round-based
spatiotemporal match data into a representation a text LLM can reason over to
produce opponent weakness reports, counter-strats, and per-round play
predictions.

---

## TL;DR

1. **Parsing is a solved problem.** The CS2 `.dem` format (Source 2,
   `PBDEMS2`, protobuf network messages) is undocumented by Valve but fully
   reverse-engineered by mature open-source parsers. `demoparser2` (Rust core,
   Python/JS bindings) and `awpy` (built on it, adds CS-specific analytics)
   extract everything: per-tick positions, view angles, HP/armor, economy,
   inventory, grenade trajectories, kills/damages, bomb events, round
   metadata, even sub-tick mouse input and voice. Parse speed is seconds per
   demo. Do not write a parser; build on `demoparser2`/`awpy`.
2. **The demo even gives you callouts for free.** Every player carries
   `last_place_name` (`m_szLastPlaceName`) — the map region name ("BombsiteA",
   "LongDoors", "Palace"). This is the single most important property for
   LLM serialization: it converts (x, y, z) floats into the exact vocabulary
   human analysts and LLMs already speak.
3. **The LLM-representation problem is the actual research.** Raw ticks are
   ~10⁶ rows/match — useless to an LLM directly. The proven pattern
   (TextStarCraft II "Chain of Summarization", ESTA/ggViz state abstraction,
   soccer-analytics text pipelines) is a **semantic compression pyramid**:
   ticks → per-round symbolic "round scripts" (who bought what, opening
   setup, utility sequence with timings and landing callouts, first contact,
   site commitment, result) → per-team **tendency profiles** computed
   deterministically (frequencies, conditional probabilities) → LLM reasons
   only over scripts + profiles, optionally with SQL/retrieval tools for
   drill-down. The LLM interprets; it never counts.
4. **Nobody has nailed team-level anti-strat with LLMs yet.** Existing LLM
   projects (CS2-AI-coach etc.) do single-player coaching off aggregate
   stats. Commercial tools (Noesis, Scope.gg, Leetify) visualize tendencies
   but leave inference to a human analyst — G2's and Renegades' analysts are
   quoted doing exactly this workflow manually. Academic work (Xenopoulos'
   ESTA/ggViz/WPA line, 2025 CS2 temporal-GNN action valuation) provides
   representations and win-probability models but no natural-language
   tactical reasoning. The gap this project fills is real.
5. **A concrete, evaluable plan is at the end**: 6 phases from corpus
   building to a measurable benchmark ("predict the opponent's next-round
   behavior"), with the serialization format as the central design artifact.
6. **(Rev. 2) The map goes into the system prompt as a "Map Card"** — a
   compiled textual atlas: closed zone lexicon (engine `env_cs_place`
   volumes reconciled with community callouts), incident-encoded adjacency
   with traversal seconds, precomputed rotate-time matrix, sightline list,
   objective facts. Graph-encoding research (GraphQA/"Talk like a Graph",
   osmAG, TopoNav) dictates the format: per-zone grouped neighbor lists,
   semantic role tags, and **no task that requires the LLM to do graph
   algorithms** — paths and timings are precomputed. RoundScripts, mined
   tendencies, and the Map Card share one closed spatial vocabulary, so
   every artifact the LLM sees speaks the same language. **Honesty note:
   the architecture is fully specifiable today, but several parameters
   (zone granularity, beat rate, token budgets, engine-vs-community
   lexicon) can only be finalized by iterating on real parsed demos — see
   §6.7.**

---

## 1. The CS2 demo format — what a `.dem` actually is

A CS2 demo is **not video**. It is the server's recording of network state:
a stream of protobuf-encoded messages that lets the game client re-simulate
the match from any perspective.

- **Container**: file starts with magic `PBDEMS2\0` (CS:GO used `HL2DEMO`;
  the formats are incompatible — Source 1 vs Source 2). After the header,
  the file is a sequence of demo commands (packets, string tables, full
  snapshots every ~60s for seeking) whose payloads are protobuf messages.
  Valve ships the `.proto` definitions with the game files; the container
  framing itself is undocumented and community-reverse-engineered. CS2's
  format is nearly identical to Dota 2's (parsers descend from Dotabuff's
  `manta` and Skadistats' `clarity`).
- **Entity system (the hard part)**: game state lives in delta-compressed
  "sendtables" — a schema (`CDemoSendTables`/serializers) transmitted at
  demo start, then per-tick bit-packed entity updates (`CCSPlayerPawn`,
  `CCSPlayerController`, `CSmokeGrenadeProjectile`, `CCSGameRulesProxy`,
  …). Parsers decode these into per-tick property values. This is what
  demoparser2/awpy/demofile-net implement so you don't have to.
- **Game events (the easy part)**: discrete events (`player_death`,
  `weapon_fire`, `flashbang_detonate`, `bomb_planted`, `round_end`, …)
  arrive as `GameEvent` messages with typed fields.
- **Tick rate**: CS2 servers and demos run at **64 ticks/s** everywhere
  (Valve enforced uniform tickrate; FACEIT included). CS2's "sub-tick"
  system embeds exact input timestamps between ticks (`usercmd_subtick_moves`,
  mouse dx/dy are in the demo — demoparser2 exposes them).
- **Size/speed**: a full map is roughly 100–400 MB raw; FACEIT serves
  `.dem.zst` (~40–60% smaller), Valve MM serves `.bz2`, HLTV `.rar`/`.zip`.
  demoparser2 parses ~750 MB/s on a 12-core desktop; awpy fully parses a
  pro demo in ~4–10 s.
- **Demo flavors**: GOTV/server demos (HLTV, FACEIT, MM) contain all 10
  players and are reliable. **POV demos** (client-recorded) are lossy and
  parsers explicitly warn of higher error rates — avoid them for this
  project.
- **What is NOT in a demo**: player voice is present in some server demos
  (extractable — see §3.9) but pro/HLTV demos ship without team comms;
  no player screen/crosshair/settings; no cross-map series context (one
  demo = one map; a BO3 is ≥2 files).

Key sources: Valve Developer Community DEM page (legacy format),
healeycodes.com "Compressing CS2 Demos" (structure walkthrough),
demoparser2/awpy internals, pbdems2 format guide (docs.rs).

---

## 2. Parser ecosystem (state of the art, verified 2026)

| Parser | Language | API style | Notes |
|---|---|---|---|
| **demoparser2** (LaihoE/demoparser) | Rust core; Python (`pip install demoparser2`), Node, WASM | **Query-style**: `parse_event(name, player_props, other_props)`, `parse_events`, `parse_ticks(props, ticks=…)`, grenade-trajectory and voice extraction | The de-facto standard backend. MIT. Fastest option. Returns pandas DataFrames (Py) / JSON (JS). Battle-tested on MM+FACEIT+HLTV demos. |
| **awpy 2.x** (pnxenopoulos) | Rust core + Python ≥3.11 | `Demo("file.dem").parse()` → **Polars DataFrames**: `header, rounds, kills, damages, shots, grenades, smokes, infernos, bomb, footsteps, ticks` | The CS-analytics layer: round segmentation, ADR/KAST/HLTV-rating recipes, **map images + nav-mesh parsing (`awpy get maps navs`, `Nav` class, `find_path`)**, **visibility/line-of-sight from `.tri`**, plotting/heatmaps, CLI. 2.0.x used demoparser2 as backend; current stable moved to its own Rust core `pbdems2`. Authors of the ESTA dataset. |
| **demofile-net** (saul) | C# / .NET | Event hooks + entity access | Clean, actively maintained, good for services. |
| **demoinfocs-golang v4** (markus-wa) | Go | Streaming event/entity observer | Mature (CS:GO era), CS2 support since v4. |
| **source2-demo** (Rupas1k) | Rust | Observer macros over protobuf + entities | Dota/Deadlock/CS2; also demo *rewriting*. |
| **clarity** | Java | Observer | Primarily Dota; the sendtable reference impl. |

**Recommendation**: `demoparser2` for bulk feature extraction (it lets you
pull exactly the columns you need at exactly the ticks you need — e.g.
"positions of all players at every `flashbang_detonate` event"), `awpy` for
round bookkeeping, nav/callout/visibility utilities, and plotting. Both
from Python, which is also where the LLM tooling lives.

**Fragility warning**: every CS2 game update can change network protos;
parsers break and get patched within days. Pin parser versions per
demo-corpus snapshot, and record `demo_version_guid`/`network_protocol`
from the header with every parsed artifact.

---

## 3. What is extractable — the full inventory

Everything below is verified against demoparser2's exposed fields and awpy's
output dataframes.

### 3.1 Per-tick player state (the trajectory layer)
- Position `X, Y, Z`, velocity (+ components), eye angles `pitch/yaw`
- **`last_place_name` — engine-provided map callout region** ("BombsiteA",
  "Middle", "TSpawn", …)
- `health`, `armor_value`, `has_helmet`, `has_defuser`, `is_alive`,
  `life_state`, spawn/death times
- Movement state: `is_walking` (shift), `in_crouch`/`ducking`, `is_airborne`,
  `is_strafing`, `is_scoped`, `is_defusing`, `flash_duration` (blind time),
  `zoom_lvl`, `shots_fired`, `aim_punch_angle`
- Zone flags: `in_bomb_zone`, `in_buy_zone`, `which_bomb_zone`
- `active_weapon_name`, full `inventory`, ammo, silencer state
- Button presses (FORWARD/FIRE/WALK/USE/…), and **sub-tick usercmds
  including `usercmd_mouse_dx/dy`** (raw aim mechanics — useful for player
  fingerprinting, not needed for team anti-strat v1)
- `ping`, `team_num`, coach slot (`is_coach_team`), rank fields (MM)

### 3.2 Per-tick game/team state
- Round clock: `round_start_time`, freeze period flag, `game_phase`,
  `total_rounds_played`
- Score, team names/clan names, first/second-half scores, overtime score
- **Timeouts**: T/CT timeout active, remaining, count used — timeout-adjacent
  rounds are exactly where teams change plans (anti-strat signal)
- Bomb: `is_bomb_dropped`, `is_bomb_planted`
- Economy context: `ct_losing_streak`, `t_losing_streak`, `*_cant_buy`
- Round end: `round_win_status`, `round_win_reason` (elimination / defuse /
  detonation / time)

### 3.3 Events (the action layer)
`player_death` (attacker, victim, assister, weapon, headshot, penetration,
noscope, thrusmoke, attackerblind, distance), `player_hurt` (dmg to health &
armor, hitgroup, weapon), `weapon_fire`, `item_purchase`/`item_pickup`,
`bomb_planted/defused/exploded/dropped/pickup`, `bomb_begindefuse`,
`round_start/round_end/round_officially_ended/round_freeze_end`,
`smokegrenade_detonate` (x,y,z), `flashbang_detonate`, `hegrenade_detonate`,
`inferno_startburn/expire` (molotov area), `decoy_started/firing`,
`player_blind` (who blinded whom, duration), `player_footstep`,
`player_jump`, `player_spawn`, `player_team`, `cs_win_panel_match`, chat
messages, `server_cvar`, …  (demoparser2: `list_game_events()` enumerates
what a given demo actually contains; `parse_event` can attach *any* player
prop and *any* global prop to each event row — e.g. every kill row can carry
both players' positions, the round number, and current equipment values.)

### 3.4 Grenade trajectories (the utility layer)
- Full projectile paths per thrown grenade: thrower, grenade type, entity id,
  tick-by-tick x/y/z from hand to detonation (demoparser2 grenade API; awpy
  `dem.grenades`)
- Detonation events give landing point; `player_blind` gives per-enemy blind
  durations; `inferno_startburn` polygons give molly coverage; smoke entities
  give bloom position and duration (awpy `smokes`/`infernos` tables with
  start/end ticks)
- Utility damage and enemies-flashed aggregates per player per round
- → This supports: lineup identification (cluster throw-origin → landing
  pairs), utility timing patterns, "their A execute = these 4 nades within
  8s", flash-effectiveness, retake utility habits.

### 3.5 Economy (the macro layer)
- Per player per round: `balance`, `start_balance`, `cash_spent_this_round`,
  `total_cash_spent`, `round_start_equip_value`, `current_equip_value`,
  buys (`item_purchase` events; `weapon_purchases_this_round`)
- Team level: loss streaks (loss bonus tier), kill rewards, `money_saved`
- → Buy-type classification (full/force/semi/eco), save tendencies, "do they
  force after losing pistol?", AWP investment patterns, drop habits.

### 3.6 Round segmentation
awpy `rounds`: round number, start/freeze_end/end ticks, winner, win reason,
bomb plant tick & site. This is the temporal indexing backbone: every other
table joins to rounds via tick ranges. (Awpy handles warmup/restarts;
FACEIT knife rounds and MM warmups must be filtered — known gotcha.)

### 3.7 Derived spatial semantics (via awpy assets, not the demo itself)
- **Nav mesh** per map (`awpy get navs`): graph of walkable `NavArea` tiles
  with connectivity — supports region discretization finer than callouts,
  path-finding (rotate-time estimates), and chokepoint definitions
- Map radar images + world→radar transforms for plotting evidence images
- **Visibility**: awpy computes line-of-sight from map geometry (`.tri`) —
  supports "was holding an angle on X", crossfire detection
- Community callout polygon sets (e.g. cs2dak `@cs2dak/maps`, SimpleRadar)
  when `last_place_name` granularity is too coarse

### 3.8 Aggregate per-round scoreboard stats (free sanity checks)
`kills_total`, `deaths_total`, `assists_total`, `damage_total`,
`utility_damage_total`, `enemies_flashed_total`, `equipment_value_total`,
`headshot_kills_total`, 3k/4k/ace counts, `objective_total`, MVPs, score.

### 3.9 Voice & chat (edge, mind privacy/ToS)
Server demos may embed Opus voice per steamid (CS2VoiceData lineage;
demoparser2 supports extraction). Pro HLTV demos: no team comms. FACEIT:
subject to platform policy + GDPR — treat as out of scope for v1; chat
messages are parseable and occasionally informative (timeout calls, tilt).

### 3.10 What you cannot get from demos
- Anything cross-map without joining multiple files yourself (veto history
  comes from HLTV/FACEIT APIs, not the demo)
- Player intent/comms (pro demos), coach instructions
- True "roles" (IGL, anchor) — must be inferred from behavior
- Reliable data from POV demos

---

## 4. Getting demos (corpus building)

| Source | How | Retention | Notes |
|---|---|---|---|
| **HLTV** (pro matches) | Match page → "GOTV Demo" link (`.rar`/`.zip`) | Effectively forever | The anti-strat corpus for pro/semi-pro opponents. Downloads are Cloudflare-guarded; bulk scraping violates ToS — download targeted matches manually or via existing archives. |
| **FACEIT** (incl. ESEA leagues) | Match room → Watch Demo (`.dem.zst`/`.gz`); official Download API exists but is now restricted/private (server-side keys) | ~30 days | **You can download any room's demo, not just your own** — this is the standard scouting path for amateur/semi-pro opponents. Data API provides match history to find rooms. |
| **Valve MM/Premier** | In-game Watch tab; **share codes** (`CSGO-xxxxx-…`) + Game Coordinator. Tools: `boiler-writter`, CS Demo Manager, `GetNextMatchSharingCode` chaining via Steam API (needs the target account's match-history auth code — i.e. own-team only) | Days–weeks (link ~1 month) | Good for own-team review; useless for scouting arbitrary opponents (privacy by design). |
| Other platforms | 5EPlay, Gamers Club, MatchZy/Get5 community servers (tournament orgs give teams demo access) | varies | CS Demo Manager supports most; MatchZy demos = standard GOTV. |

Practical corpus for the product use case: **8–20 most recent HLTV/FACEIT
demos of the target team on the relevant map pool**, refreshed weekly.
Archive raw `.dem` + parsed parquet; demos expire, your parquet doesn't.

---

## 5. Prior art

### 5.1 Academic — CS-specific representations & models
- **Xenopoulos et al.** (the canonical line):
  - *Valuing Player Actions in CS:GO* (IEEE Big Data 2020): defines the
    game-state data model, XGBoost round-win-probability from (players
    alive, HP, equipment value, position summaries); players valued by
    win-prob deltas (WPA). Built `awpy`'s predecessor.
  - *ggViz* (2021): **discretizes game states into nav-region token sets**
    to retrieve similar rounds at scale; pro coaches/analysts interviewed
    endorsed the "query similar situations" workflow. → Direct precedent
    for tokenizing CS state; their discretization = our serialization's
    spatial vocabulary.
  - *Optimal Team Economic Decisions* (2021): game-level win prob;
    buy/save optimality analysis. (CS:GO MR15 economy — numbers must be
    redone for CS2 MR12.)
  - *ESTA dataset* (NeurIPS D&B 2022): 1,558 pro CS:GO demos parsed to
    hierarchical JSON (8.6M actions, 417k trajectories); win-prob
    benchmarks. Proves corpus-scale parsing pipelines and gives a schema
    reference. CS:GO-era; a CS2 equivalent does not exist publicly yet
    (opportunity).
  - *Graph NNs to Predict Sports Outcomes* (2022): permutation-invariant
    graph encoding of team game states (−20% loss vs. feature vectors on
    esports win prediction).
- **Szmida & Toka 2025 (CS2!)**: *Evaluating Player Actions … Temporal
  Heterogeneous Graph Neural Networks* — CS2 replays → dynamic heterogeneous
  graphs; frame-level win prob >76% acc; **Shapley values attribute win-prob
  swings to specific actions (kills, grenades)**. State of the art for
  CS2 action valuation; a perfect *numeric* companion to an LLM layer
  (feed "impactful moments" into round scripts).
- **Durst et al. 2024**, *Learning to Move Like Professional Counter-Strike
  Players* (CSKNOW/MLMOVE): behavior-cloned movement from pro demos —
  evidence that positioning tendencies are learnable/predictable at scale.
- **Wang, Ustun, McGroarty 2025**: discretized CS:GO simulation environment
  for strategic multi-agent planning — parallel confirmation that
  region-level discretization preserves strategy.
- Older: Bednárek et al. 2017 (death-location clustering), Hirota 2024
  (round-outcome prediction), SHAP-based pro-player analysis (2025).

### 5.2 Academic — representing games/time-series for LLMs
- **TextStarCraft II + Chain of Summarization (Ma et al., 2023/24)**: the
  key transferable design. Raw observations → text via an obs-to-text
  adapter; **L1 single-frame summarization** compresses each state; **L2
  multi-frame summarization** aggregates a window into "Situation Overview /
  Situation Analysis / Strategic Planning / Opponent Strategy Analysis /
  Recommendations / Decision". 10× fewer LLM calls; GPT-3.5-level models
  beat SC2's Harder AI. → Our per-round scripts = L1; per-team tendency
  profile = L2; the anti-strat report = L3.
- **Sports analytics with LLMs**: TacticalGPT (StatsBomb 2023, predicting
  football tactical decisions from event data), MatchTime (EMNLP 2024,
  soccer commentary from event streams), SoccerChat / multi-agent soccer
  understanding (2025), Wang & Yoshinaga 2024 (esports commentary from data
  records). Pattern in all of them: **structured event logs, not raw
  coordinates, are what LLMs consume**; numbers are precomputed.
- **Benchmarks that show the limits**: SportR (2025) and EgoEsportsQA
  (2026, FPS video-QA incl. CS2) — multimodal LLMs are still weak at
  fine tactical reasoning *from video*. → Strong argument for the
  **data-first (demo-parsing) route over the video route**: demos give
  perfect ground truth for a fraction of the compute.
- LLM time-series literature (LLMTime etc.) shows LLMs handle serialized
  numeric series zero-shot, but for this domain the semantic-event
  abstraction dominates raw-series approaches: CS rounds are naturally
  event-structured.

### 5.2b Academic — encoding *space and graphs* as text for LLMs
(Underpins the Map Card design in §6.5.)
- **"Talk like a Graph" (Fatemi et al., ICLR 2024) + GraphQA**: the first
  systematic study of graph→text encoding. Findings that transfer directly:
  (1) encoding choice changes task accuracy by **4.8–61.8%**; (2) the
  **incident encoding** — per-node grouped neighbor lists ("Jungle connects
  to: Connector, Stairs, CT-Spawn") — beats flat edge lists for local
  connectivity reasoning (53.8% vs 19.8% on connected-nodes); (3) LLMs are
  **poor at algorithmic graph tasks** (path existence, cycles) even when
  encoding is good; (4) semantically meaningful node names interact with
  model priors. Consequences: Map Card zones are named by callouts (prior-
  loaded names), topology is incident-encoded, and all shortest-path /
  rotate-time computation is done offline — the LLM reads results, never
  computes them.
- **NLGraph (Wang et al. 2023)**: benchmark confirming LLM graph-reasoning
  brittleness beyond toy sizes → keep the zone graph ≤ ~100 nodes (CS maps
  fit: ~25 engine places, ~50–70 community callouts per map).
- **osmAG (2024, robotics)**: hierarchical *area graph* (areas = polygon
  nodes, passages = edges) serialized as tagged text for LLM map
  comprehension — the closest structural template for a CS Map Card;
  they empirically compared connectivity phrasings for LLM comprehension.
- **TopoNav (2025) & Tag Map (2024)**: topological graphs as LLM spatial
  memory for navigation; TopoNav's ablation shows **semantic node metadata
  (room type, contained objects) measurably improves performance** →
  justify per-zone role tags (site/choke/connector/power-position).
- **MANGO (2024) + spatial-memory repair work (2026)**: LLMs *constructing*
  maps from experience accumulate structural errors that are hard to
  repair → never make the LLM infer the map from round data; always supply
  the authoritative Map Card up front.
- **ZorkGPT / TextWorld agents**: the working pattern for text-game agents
  is an externally maintained location graph fed into the prompt, with the
  LLM consuming (not building) spatial structure.

### 5.3 Existing products (competitive landscape)
- **Noesis.gg** — closest to the use case, *without* AI: upload demos,
  filter rounds ("all T-side buy rounds on Mirage"), compare rounds across
  matches, heatmaps. Marketing quotes **G2's analyst ("I use Noesis for
  every pre-game analysis to find tendencies about opponents")** and
  Renegades' coach; sells "punish predictability". The human does the
  induction — exactly the step an LLM layer would automate.
- **Scope.gg** — utility-centric 2D replay + utility effectiveness scores;
  match tracker. Player-improvement oriented.
- **Leetify** — per-category benchmarked scores (aim/utility/positioning/
  opening duels); "AI coaching" = scoring models + templated feedback, not
  generative tactical analysis. Team features are basic.
- **Refrag** — drills/practice servers; smartcoach.gg, cs2coach.gg — LLM
  grade-and-advice for individual players (thin analytics).
- **CS Demo Manager (akiver)** — OSS desktop demo library/downloader/
  analyzer; good reference implementation for acquisition plumbing.
- **DAK Studio / cs2-demo-analysis-kit (2026, OSS)** — local-first analysis
  workbench on demoparser2 with a "Coach Workbench: what will the opponent
  run, what do we prepare?" module explicitly marked *early/experimental* —
  independent confirmation the niche is open.
- Defunct: Shadow.gg (CS:GO anti-strat platform) — the concept pre-exists;
  it died on business model, not on data availability.

### 5.4 Existing open-source LLM×CS2 attempts (all player-coaching, not team anti-strat)
- `mohammed-alsaqqa/CS2-AI-coach`: demoparser2 → aggregate stats JSON →
  OpenAI Assistants Q&A. No spatial/temporal serialization.
- `faisalkhan07/cs2-ai-coach`: parsed metrics (economy, aim, utility) →
  LLM-narrated report + dashboard.
- `JBRKR000/cs2-demo-coach`: single-player report with ML round-impact.
- `afw825/cs2-antistrat`: opening positions → SQLite → radar overlays;
  **no LLM** — visual anti-strat only.
- `pr1malator/pr1maly`: local CS2 analyzer with an "AI match chat", a
  `callouts.py` coordinate→callout translator over hand-calibrated zone
  rectangles, and per-map "role zone" JSONs — the closest existing
  mechanism to our zone-grounding idea, but zones are ad-hoc rectangles,
  there is no shared closed vocabulary, no map topology given to the LLM,
  and no team-level tendency mining or prediction eval.
- Machine-readable zone data (inputs for §6.5): `mwridgway/CS2Callouts`
  and `xobust/CS2CalloutExtractor` extract the **engine's `env_cs_place`
  callout volumes** (the exact source of `last_place_name`) from map VPKs
  via ValveResourceFormat — e.g. 23/23 places recovered on Mirage with
  awpy-aligned radar overlays; totalcsgo.com documents ~383 community
  callout names with aliases across 8 maps; boltobserv ships per-map radar
  calibration configs.
- Take-away: everyone stops at either (a) aggregate stats → LLM prose, or
  (b) spatial visualization for humans. The **round-script serialization +
  tendency mining + LLM inference** middle layer is unclaimed — and no one
  gives the LLM a principled map representation at all.

---

## 6. The core design problem: serializing a match for an LLM

### 6.1 Why raw data fails
One map ≈ 2M player-tick rows × 50+ props. Even downsampled to 1 Hz it's
~35k rows of floats — token-hostile, and LLMs are unreliable at arithmetic
and 3D geometry over float soups. Every successful precedent (CoS, ggViz,
TacticalGPT, MatchTime) converts state to **domain-semantic symbols first**.

### 6.2 The semantic compression pyramid (proposed representation)

**Level 0 — Parquet lake (lossless).** demoparser2/awpy outputs, one parquet
set per demo: `rounds`, `kills`, `damages`, `grenades` (with trajectories),
`smokes`, `infernos`, `bomb`, `shots`, `ticks@4Hz` (positions + key flags),
`economy`. Keyed by (match_id, map, round_num, tick). This layer answers any
drill-down question and feeds the deterministic miners.

**Level 1 — Round script (the pivotal artifact).** One compact,
schema-validated JSON (or YAML) object per round, ~200–400 tokens, built by
pure code from Level 0. Contents:

```yaml
round: 7            # + score, half, side per team
economy:
  TeamA: {type: full_buy, spend: 24500, equip: 27k, awps: 1, streak_bonus: 2}
  TeamB: {type: force, spend: 9800, ...}
freeze_end_setup:            # positions at freeze end + ~15s, as callouts
  TeamA_CT: {AWP_kalle: MidDoors, b1: BombsiteB, ...}
  TeamB_T:  {spread: [TSpawn x4, LowerTunnels x1]}
timeline:                    # event-sequenced, seconds after freeze end
  - t: 12, type: util, side: T, nade: smoke, from: OutsideTunnels, lands: CTMid
  - t: 14, type: util, side: T, nade: flash, lands: BombsiteB, blinded: [b1 2.1s]
  - t: 16, type: contact, kill: {killer: donk, victim: b1, at: BombsiteB,
      weapon: ak47, hs: true, traded_within_4s: false}
  - t: 22, type: plant, site: B, planter: magixx, alive: {T: 4, CT: 3}
  - t: 41, type: retake_util, ...
outcome: {winner: T, reason: elimination, clock_used: 0:52,
          key_moment: "3v3 postplant, CT no util"}
labels:                      # miner-attached tags, closed vocabulary
  t_strategy: B_rush_lowers  # from opening-trajectory clustering
  first_contact: {zone: BombsiteB, t: 16, initiated_by: T}
```

Design rules learned from prior art:
- **Callouts, not coordinates** (`last_place_name`, refined by nav-region
  polygons where too coarse). Coordinates appear only in Level 0.
- **Relative time** (seconds after freeze end), not ticks.
- **Closed vocabularies** for buy types, strategy labels, win reasons —
  makes downstream counting/eval trivial and hallucination detectable.
- Deterministically generated: same demo → same script (testable).

**Level 2 — Team tendency profile (per map, per side).** Pure-code miner
aggregates N matches of round scripts into a stats object + evidence
pointers. Examples of mined tendencies (all are simple conditional
frequencies over Level-1 fields):
- Buy policy: P(force | lost pistol), save thresholds, AWP re-buy behavior
- Openings: distribution of first-contact zones by buy type and score
  state; default setups (modal callout per player at freeze_end+15s —
  **this yields role inference**: who anchors B, who lurks)
- Execute repertoire: clustered utility sequences (grenade landing-zone
  set + ordering + timing) with frequency and success rate, e.g.
  "A execute = CT smoke + jungle smoke + stairs molly + 2 flashes,
  avg commit at 1:15, used 9/31 T rounds, 67% success vs full buys"
- Timing profile: median first-contact time by side (fast vs default vs
  slow team), pace after timeouts, pace in must-win rounds
- Player habits: opening-duel seeker (who takes first fights, where),
  lurker paths, AWP positioning distribution
- Anti-eco/postplant/retake patterns; clutch tendencies
- Each tendency carries `n`, confidence, and round-script references
  (evidence links — non-negotiable for trust, cf. DAK's "every number
  links to the round").

**Level 3 — LLM layer.** Consumes Level 2 + selected Level-1 exemplars.
Three modes:
1. **Scouting dossier generation**: "given this profile + these 12
   representative rounds, write the anti-strat: weaknesses, exploit plans,
   utility counters, per-round-state predictions." Token budget: profile
   ~2–4k + 12 scripts ~4k + instructions ≈ well within any modern context.
2. **Interactive analyst (agent + tools)**: LLM with function tools —
   `query_rounds(filter)`, `get_round_script(id)`, `get_tendency(name)`,
   `render_heatmap(filter)` (returns image for the human) — over the
   parquet/DuckDB lake. Text-to-SQL over Level 0 for long-tail questions.
   This is the architecture the OpenAI-Assistants CS2 projects gesture at,
   but grounded in real spatial semantics instead of scoreboard stats.
3. **Per-round predictor (the evaluable claim)**: given profile + current
   match state (score, economy, previous rounds), predict next round:
   {buy type, first-contact zone, execute vs default, likely site,
   key utility}. This is scored against held-out rounds (§7 Phase 5).

### 6.3 Why hybrid (miner + LLM), not LLM-only
- LLMs miscount over long contexts; frequencies must come from code.
- Tendencies need `n` and dates (meta shifts after patches/roster changes;
  weight recent demos).
- Hallucination control: the LLM may only cite tendencies/rounds that exist
  in its input; evidence links let a human verify in a 2D viewer.
- Cost: mining is free CPU; LLM tokens are spent only on interpretation.

### 6.4 Alternatives considered and rejected for v1
- **Video/multimodal route** (feed replay video to a VLM): benchmarks
  (EgoEsportsQA, SportR) show current VLMs under-perform on exactly this
  tactical reasoning; demos already contain perfect state. Revisit later
  only for presentation (auto-generated clips).
- **Fine-tuning a bespoke model on round scripts**: premature — no labeled
  "correct anti-strat" corpus exists; prompted frontier models + solid
  serialization is the right first system. Fine-tuning becomes attractive
  later for the per-round predictor once Phase 5 produces a labeled set.
- **Pure embedding/GNN approach**: strong for prediction (see Szmida &
  Toka) but produces no natural-language, evidence-linked output — can be
  added as a numeric signal inside round scripts (win-prob swings) instead.

### 6.5 The Map Card: how the LLM knows what a position means

The question "how do we give the LLM the map" has two answers that must be
combined, not chosen between.

**Answer 1 — the prior is already in the weights (but must be measured).**
For legacy maps (Mirage, Dust2, Inferno, Nuke, Overpass), callout guides,
strat forums, and match commentary are massively represented in web
training data. Frontier LLMs can already discuss "smoking Jungle and
Stairs for an A hit" — vocabulary *and* tactical idiom. This prior is an
asset (semantically loaded zone names activate it — consistent with
"Talk like a Graph"'s finding that node-naming interacts with model
priors) and a liability (it may be stale: map reworks, CS2 layout changes,
renamed areas). Therefore: **never rely on it silently; measure it.** A
~50-question-per-map "map quiz" harness (adjacency, LOS, timing, standard
executes; graded against the compiled Map Card) quantifies per-model,
per-map prior quality and decides how much doctrine the Map Card must
carry. New or reworked maps (e.g. post-rework Train, fresh pool entries)
will fail the quiz — there the Map Card is the *only* source of truth.

**Answer 2 — the Map Card: a compiled, versioned textual atlas** placed in
the system prompt (static per map → prompt-caching friendly; budget
~2.5–4k tokens). It is *compiled by code* from four data sources, then
human-reviewed once per map (~1 hour of analyst time):

1. **Engine truth**: `env_cs_place` volumes extracted from the map's VPK
   (ValveResourceFormat; existing OSS extractors) — these are the *exact*
   regions the demo's `last_place_name` reports, so lake data and Map Card
   are aligned by construction (~20–30 places/map).
2. **Community refinement**: finer callout polygons + alias table
   (totalcsgo-style names; pr1maly-style calibration tooling) subdividing
   coarse engine places (~50–70 zones/map). Multilevel maps (Nuke,
   Vertigo, Overpass underpass areas) get **z-band annotations** so
   point-in-polygon lookup is 3D-correct.
3. **Nav mesh** (awpy): zone connectivity (which zones share walkable
   borders), traversal times (shortest nav path length ÷ 250 u/s run
   speed), chokepoint identification (narrow nav corridors between zones).
4. **Visibility geometry** (awpy `.tri`): the top-N long sightlines that
   cross zone boundaries (AWP lanes), and which zone a smoke must occupy
   to cut each one.

**Card format** (YAML; layers ordered by how often the LLM needs them):

```yaml
map: de_mirage            # card_version, game_version, checksum
frame: "radar top-down; +x east, +y north; coords normalized 0-100"
zones:                    # THE closed lexicon. Illustrative excerpt only —
  - id: TopMid            # final names come from the Phase-2a lexicon build
    aliases: [top of mid]
    engine_place: TopofMid
    tags: [t_side_entry, mid_control]
    quadrant: center-north
    centroid: [52, 71]    # reference only; prose uses zone ids
  - id: Window
    aliases: [sniper nest, window room]
    engine_place: Window
    tags: [power_position, ct_side, awp_anchor]
    quadrant: center
    elevation: upper      # z-band
topology:                 # incident encoding (per-zone grouped neighbors)
  TopMid: {Underpass: 3s, Mid: 2s, TRamp: 4s}
  Window: {Connector: 3s, Kitchen: 2s, Mid: "LOS only — drop 1-way"}
rotates:                  # precomputed; the LLM NEVER pathfinds
  - {from: ASite, to: BSite, via: CTSpawn, run: 22s, note: fastest CT rotate}
  - {from: ASite, to: BSite, via: Jungle-Connector-Mid, run: 26s}
timings:                  # spawn-to-zone at full run (meta anchors)
  - {side: T, zone: TopMid, earliest: "0:07 after freeze end"}
  - {side: CT, zone: Window, earliest: "0:05"}
sightlines:
  - {a: Window, b: TopMid, cut_by_smoke_in: [Window, TopMid], range: long}
  - {a: TicketBooth, b: MidDoors? , ...}       # curated top ~25 pairs
objectives:
  sites: {A: [ASite, Palace-exit, Jungle-exit], B: [BSite, BApts-exit]}
  plant_default: {A: default-box, B: back-van}  # named plant spots
  clock: {round: "1:55", bomb: "0:40", defuse: "10s / 5s with kit"}
doctrine:                 # OPTIONAL meta-priors block — versioned,
  standard_t_executes: [...]   # EXCLUDED in all evaluations (leakage),
  standard_ct_setups:  [...]   # included in production prompts
```

**Encoding rules (each grounded in §5.2b evidence):**
- Zones are *named* nodes (prior activation), incident-encoded adjacency
  (best encoding for neighbor reasoning), graph kept ≤ ~100 nodes.
- Anything algorithmic — shortest paths, rotate times, earliest-arrival
  timings, LOS pairs — is **precomputed into flat facts**, because LLMs
  demonstrably fail at graph algorithms even with good encodings.
- Qualitative spatial anchors (`quadrant`, `elevation`) beat raw floats;
  centroids are retained only as a reference frame of last resort.
- The zone lexicon is **closed and shared**: RoundScripts, TeamBook
  tendencies, LLM outputs may only use lexicon ids. Any out-of-lexicon
  zone name in a generation is a machine-detectable hallucination.

#### 6.5.1 Map Card build: inputs, and why the demo is not sufficient

The card is compiled from three input buckets. The demo is deliberately
the *least* authoritative of them — it is neither sufficient (it contains
no polygons, no community vocabulary, no wall geometry, no nav mesh, no
role semantics) nor strictly necessary (buckets A+B alone produce a valid
card). Demos audit and calibrate; they do not define.

| Card component | Source | Bucket |
|---|---|---|
| Zone volumes/polygons (base lexicon) | `env_cs_place` entities extracted from the map VPK — the same volumes the server consults when writing `last_place_name` into demos | A: game files |
| Zone adjacency, traversal seconds, chokepoints | Nav mesh (`.nav` via awpy): walkable-area graph, path length ÷ run speed | A: game files |
| Candidate sightlines | Visibility over physics geometry (`.tri`, awpy) | A: game files |
| Radar frame, normalized coords | Radar image + `map-data.json` transform (awpy) | A: game files |
| Community callout names, aliases, finer subdivision polygons | Community compendia + one calibration pass — **these names exist in no game file and no demo** | B: human |
| Role tags, named plant spots, curated top sightlines, optional doctrine | Analyst review (~1h/map) | B: human |
| Clock/bomb/defuse constants | Game rules | B: trivial |
| Place-inventory reconciliation | Diff distinct `last_place_name` values observed in the parsed corpus vs VPK extraction (catches game-version mismatch) | C: demos (audit) |
| Rotate/arrival time calibration | Observed zone transitions in demos — real distributions incl. walking/boosts, superior to idealized nav estimates | C: demos (calibrate) |
| Sightline validation | Kill/damage events = witnessed LOS pairs (attacker→victim positions); absence of kills proves nothing | C: demos (validate) |

Build order: extract A (mechanical, scriptable per game version) →
overlay B (once per map, versioned) → audit/calibrate with C (per corpus
refresh). The card is keyed to `(map, game_version)` and its checksum is
stamped into every RoundScript and TeamBook derived under it, so
artifacts from different map layouts can never silently mix.

**Demo-only fallback (degraded mode).** If VPK extraction breaks on a
game update, zone regions can be *approximated* from demos alone:
alpha-shapes over the labeled `(x, y, z, last_place_name)` point cloud of
player positions accumulated across the corpus. Limitations: only areas
players actually walked, fuzzy boundaries, engine granularity only, no
community subdivision, no sightlines, no nav timings. Acceptable to keep
the pipeline alive for days, not as a foundation.

### 6.6 RoundScript v2: the demo→text grammar (the novel transformation)

v1 (§6.2) was a JSON round summary. v2 formalizes it into a **three-view
grammar** over the shared lexicon — think *FEN + PGN for Counter-Strike*:
a positional snapshot notation, a per-player trajectory notation, and an
annotated event stream. All three are generated deterministically from the
Level-0 lake; all three are line-oriented so that rounds align vertically
across a match, because cross-round pattern induction over parallel
strings is exactly what LLMs are strong at (and what the anti-strat task
needs).

**View A — Beat frames (formation snapshots).** All 10 player zones
sampled at fixed beats: freeze-end +0/15/30/45/60/75/90s, plus
event-anchored beats at first-contact and plant±10s. Side-grouped,
run-length-compressed:

```
R7 [T full($24.5k) | CT full($26k)] score 3-3 (T:them)
B+00  T: 4×TSpawn 1×TRamp          | CT: Window Connector Jungle 2×BSite
B+15  T: 2×TopMid 2×Underpass 1×TRamp | CT: Window Connector Stairs BSite Bench
B+30  T: 3×Mid 1×Underpass 1×TRamp    | CT: Window Connector Jungle BSite Bench
FC+34 first_contact Mid: T donk k CT b1 (ak, hs, no-trade-4s)
B+45  T: 2×Connector 2×Mid 1×TRamp(lurk) | CT: Jungle Stairs BSite CTSpawn
PL+52 plant A(default-box) by magixx; alive T4 CT3
B+62  T: ASite Jungle Palace TRamp | CT: CTSpawn Stairs BSite→rotating
END   T win elimination @1:21; key: postplant crossfire Jungle+Palace
```

**View B — Movement sentences (per-player trajectories).** Zone sequence
with dwell compression and inline event marks (`k` kill, `d` death, `p`
plant, `x` defuse attempt, `~` walk/sneak, `!` util thrown):

```
donk(T,entry):  TSpawn > Underpass > TopMid! > Mid k(b1) > Connector > Jungle k(b2) d
magixx(T,plant): TSpawn > TRamp > Mid > Connector > ASite p
chopper(T,lurk): TSpawn > ~TRamp(45s) > ~Palace > ASite k(b3)
```

These are ESTA-style trajectories re-expressed in callout space — the
textual form of the token sequences ggViz used for numeric retrieval.

**View C — Utility grammar.** Every grenade: time, thrower, type,
throw-origin zone, landing zone, cluster id, and — joined from the Map
Card sightline table — the tactical effect:

```
U+12 donk    smoke  TSpawn>Window        [lineup W-1 "window from spawn"] cuts Window↔TopMid
U+14 sh1ro   flash  Underpass>TopMid(pop)  blinds: b1 1.8s, b4 0.6s
U+15 zont1x  molly  TopMid>CatwalkBoost    clears: cat boost
```

Lineup ids (`W-1`) come from clustering throw-origin→landing pairs in raw
coordinates (Level 0) across the whole corpus, then *naming* clusters by
landing zone + ordinal. Names are stable per map, so the miner can count
"they open 71% of A-executes with W-1 + CT-2 within 3s" and the LLM can
cite it.

**Grammar sketch (v2, EBNF-flavored):**

```
round      := header beatline+ eventline* endline
header     := "R" num "[" buys "]" "score" score "(" side-map ")"
beatline   := beat side-formation "|" side-formation
formation  := (count "×")? zone-id (annot)?     # zone-id ∈ Map Card lexicon
eventline  := (contact | util | plant | defuse | timeout)
movement   := player "(" side "," role ")" ":" zone-seq
zone-seq   := (dwell? marker? zone-id event-mark?)+
```

**Why this is the novel contribution** (and how it will be defended):
individually, pieces have precedent — region tokenization (ggViz),
multi-level summarization (CoS), area-graph-to-LLM (osmAG), callout
translation (pr1maly). Published nowhere: (1) a **closed, demo-derived
spatial vocabulary shared between a static map atlas and per-round
serializations**, (2) a **beat-aligned dual notation (formation snapshots
+ trajectory sentences)** designed for cross-round pattern induction by
LLMs, (3) grammar-constrained outputs that make spatial hallucinations
machine-detectable, and (4) an **anti-strat prediction benchmark** that
scores the whole stack. Phase 5's ablations turn each of these into a
measured claim rather than an assertion.

**TeamBook (Level 2) formalization.** Tendencies are keyed conditional
distributions over RoundScript fields:

```
key := (map, side, buy-class, score-bucket, prev-round-outcome)
val := P(opening-formation-cluster), P(first-contact-zone),
       P(execute-lineup-set), timing quantiles, n, recency-weight
```

plus per-player **role cards** (beat-occupancy histograms → "b1 anchors
BSite 78% of full-buy CT rounds; leaves site only after 0:45"). Both a
compact table and one templated sentence per entry are emitted — tables
for counting-free LLM consumption, sentences for the dossier.

### 6.7 What cannot be decided without real extracted demos (honesty)

The user asked for a flag if this design cannot be completed from
literature alone. It cannot — not the *design* (above, fully specified),
but the **parameter choices**. The following are empirically determined
and anyone claiming them without iterating on parsed demos is guessing:

1. **The actual `last_place_name` inventory per map** — the engine place
   list used by demos has not been dumped in this research session; the
   extraction tooling is verified to exist, the per-map inventories are
   not. First task of Phase 2a.
2. **Engine places vs community polygons** — whether ~25 engine places
   discriminate executes (e.g. distinguishing a Jungle smoke from a
   Stairs smoke on Mirage) or ~60 community zones are required. Expected:
   community zones are required; must be confirmed against real grenade
   landing scatter.
3. **Beat rate** (15s vs 10s vs event-anchored-only) and RoundScript token
   cost/information sufficiency — settled by the round-reconstruction test
   (Phase 2b) on real rounds, not by argument.
4. **Lineup-cluster granularity** — DBSCAN epsilon on throw/landing pairs
   needs real trajectory scatter; too fine invents fake lineups, too
   coarse merges distinct strats.
5. **Rotate-time table accuracy** — nav-path ÷ run-speed estimates must be
   calibrated against observed rotations in demos (walking, boosts, jump
   paths distort pure nav estimates).
6. **LLM map-prior quality per (model, map)** — measurable only by running
   the map quiz; determines doctrine-block size and how much the system
   leans on the card vs the weights.

Everything else in this document stands on verified sources: parser
capabilities from parser documentation, encoding findings from published
benchmarks, product claims from vendor pages. No demo file was parsed in
the making of this report — by design, this repo is research-only — and
the plan's Phase 0–2 exists precisely to convert these six unknowns into
measurements.

## 7. The concrete plan

Six phases; each has a falsifiable exit criterion. Python throughout
(demoparser2 + awpy + polars/duckdb; LLM-agnostic client).

### Phase 0 — Corpus & ground truth (week 1)
- Pick **one map** (Mirage or Inferno — richest public knowledge for sanity
  checks) and **two pro teams** with ≥10 recent HLTV demos on it.
- Download demos manually (ToS-safe), archive raw + header metadata
  (`demo_version_guid`, server, date, event).
- Deliverable: `corpus/` manifest (team, map, date, source, file hash).
- Exit: 20+ demos parse cleanly with pinned demoparser2 + awpy versions;
  parse failures documented.

### Phase 1 — Extraction lake (weeks 1–2)
- One script: demo → parquet set (rounds, kills, damages, grenades+trajectories,
  smokes, infernos, bomb, shots, economy snapshot per round, ticks@4Hz with
  `X,Y,Z,last_place_name,health,armor,active_weapon,is_alive,flash_duration,
  is_scoped,is_walking` for all 10 players).
- Normalize: half-aware team identity (steamid roster ↔ side per round),
  relative round clock, MR12 economy fields, filter warmup/knife/restarts.
- DuckDB views over the parquet lake.
- Exit: for 3 spot-checked rounds, every lake fact matches the demo replayed
  in the CS2 client / a 2D viewer (position at time t, buy values, nade
  landing spots).

### Phase 2a — Map Card compiler + LLM map-prior quiz (weeks 2–3)
- Dump the engine place inventory: extract `env_cs_place` volumes from the
  target map's VPK (CS2Callouts / CS2CalloutExtractor route) **and**
  cross-check by dumping the distinct `last_place_name` values that actually
  occur in the Phase-1 tick lake (resolves §6.7-1: the two must reconcile).
- Build the zone lexicon: engine places as the base layer; subdivide with
  community callout polygons where mining needs finer resolution
  (decided by §6.7-2 test: project the corpus' grenade landings and kill
  positions onto both lexicons; if two distinct utility clusters or
  opening routes collapse into one engine place, subdivision is required).
  Z-band annotations for multilevel areas.
- Compile topology/rotates/timings from awpy nav (§6.5 sources 3–4);
  calibrate rotate times against ≥20 observed rotations from the lake
  (§6.7-5); curate role tags + top sightlines (analyst hour).
- Build the **map quiz harness** (~50 auto-graded questions/map from the
  card: adjacency, rotate ordering, earliest timings, LOS) and run it on
  candidate LLMs with *no card* vs *card in system prompt* (§6.7-6).
- Exit: Map Card v1 (YAML, versioned, ≤4k tokens) checked in; quiz shows
  card closes ≥90% of the no-card error gap; zone mapper function
  (x,y,z)→zone id round-trips 100 random lake positions correctly on
  visual spot-check.

### Phase 2b — RoundScript v2 serializer (weeks 3–5, the core artifact)
- Implement the three-view grammar (§6.6): beat frames (start at 15s
  beats; re-decide per §6.7-3), movement sentences with dwell compression,
  utility grammar with corpus-level lineup clustering (epsilon sweep per
  §6.7-4) and sightline-effect joins; plus buy-type classifier
  (equip-value thresholds calibrated for MR12), contact/trade detection
  (4s window), plant/retake segmentation.
- Emit both v2 text and the v1 JSON (JSON stays the machine interface for
  the miner; v2 text is the LLM interface). Same underlying facts —
  property-test that they agree.
- Token budget check: p95 round ≤ 450 tokens (beat frames + events);
  movement sentences attached only for exemplar rounds.
- **Validation set**: hand-annotate 30 rounds from VOD review (strat run,
  first contact); serializer must agree ≥85% on first-contact zone and buy
  type; blinded-human round-reconstruction test ("round-script Turing
  test") ≥80%; **grammar linter**: 0 out-of-lexicon zone ids across the
  corpus.
- Exit: lexicon + grammar frozen at v2.0; agreement metrics hit; §6.7
  items 1–5 converted to recorded decisions with data.

### Phase 3 — Tendency miner (weeks 4–6)
- Deterministic miners over round scripts (§6.2 Level 2): buy policy table,
  opening/first-contact distributions, default-setup modal positions (role
  inference), utility-sequence clustering (DBSCAN/hierarchical over landing-
  zone multisets + timing; label clusters with callout names), timing
  profile, anti-eco/postplant/retake patterns. Every stat: `n`, window
  (last-K-demos weighting), evidence round-ids.
- Sanity check against public knowledge: for a famous team, miner output
  must rediscover 3–5 tendencies a human analyst already knows from
  watching (e.g. known fast-B tendencies, known AWP positions).
- Exit: profile JSON per (team, map, side) regenerates identically from the
  lake; spot-check confirms rediscovery.

### Phase 4 — LLM layer (weeks 6–8)
- Prompt architecture: **system prompt = Map Card (static, cached) +
  tactical doctrine + output contract**; user turn = TeamBook profile +
  exemplar RoundScripts. Output contract (dossier sections): Identity,
  Defaults & roles, Execute repertoire w/ counters, Economy policy w/
  exploit, Player-specific weaknesses, Round-state playbook table,
  Confidence/evidence appendix. All spatial language constrained to
  lexicon ids (linted post-generation).
- Inputs: Level-2 profile + top-k exemplar round scripts (retrieved by
  tendency coverage, not similarity fluff).
- Tool mode: DuckDB text-to-SQL + `get_round_script` for the interactive
  analyst; enforce "cite round-ids for every claim".
- Anti-hallucination gate: post-process — every cited round-id must exist;
  every frequency quoted must match the profile within rounding.
- Exit: two pro-team dossiers reviewed by ≥2 experienced CS players
  (Faceit 8+/analyst); Likert ≥4/5 on "actionable & accurate"; zero
  fabricated evidence links.

### Phase 5 — Evaluation benchmark (weeks 8–10, the research contribution)
- **Task**: given profile mined from demos 1..k and the live match state,
  predict round k+1 behaviors on a held-out match: buy type (4-class),
  first-contact zone (callout-set), site committed (A/B/none), execute vs
  default, fast (<25s) vs slow.
- Baselines: (a) majority class, (b) per-team frequency tables without LLM,
  (c) LLM without profile (raw scripts only), (d) full system.
- Metrics: accuracy/F1 per head, Brier score for probabilistic site calls;
  significance over ≥300 held-out rounds (≈15 matches).
- This doubles as the ablation suite for every representation decision —
  the publishable finding. Ablation arms:
  - **Map Card**: none / lexicon-only / full card / full + doctrine block
    (doctrine excluded from headline numbers — leakage; see §6.5)
  - **Zone lexicon**: engine places only vs community-refined zones
  - **Round encoding**: v1 JSON vs v2 beat-frames+events vs raw coordinates
    (control) — with/without movement sentences
  - **Beat rate**: 15s vs event-anchored-only
  - **Tendencies**: table vs prose vs both
- Also report the map-quiz→prediction correlation: does a model's measured
  map prior (Phase 2a) predict its anti-strat accuracy? (Cheap, novel,
  informs model selection.)
- Exit: full system beats frequency-table baseline on ≥3 of 5 heads, or we
  learn precisely where LLM inference adds nothing (also a result).

### Phase 6 — Productization sketch (later)
- Weekly auto-refresh per tracked opponent; dossier diffs ("new since last
  event: they now default B on pistols").
- 2D evidence renders (awpy plots) embedded next to each claim.
- Own-team blind-spot report (run the same pipeline on yourself).
- Optional: Szmida-style win-prob model to tag "highest-leverage moments"
  in scripts; voice ingestion for own-team demos; multi-map veto advisor
  (needs HLTV/FACEIT APIs).

### Cost/effort reality check
- Parsing: seconds/demo, laptop-class.
- Mining: trivial CPU.
- LLM: dossier ≈ 15–30k input tokens, a few dollars per team per week at
  frontier prices; per-round predictions are pennies. No training compute
  required until Phase 5 suggests fine-tuning.

---

## 8. Risks & open questions

1. **Parser breakage on CS2 updates** — pin versions, snapshot parquet,
   keep raw demos. (Historical precedent: parsers patch within days.)
2. **Small-N tendencies** — 10 demos ≈ 120 T rounds per map; some
   conditionals get n<10. Mitigation: confidence surfacing, hierarchical
   backoff (team → regional meta priors), last-K weighting for roster/meta
   shifts.
3. **Callout granularity** — `last_place_name` regions vary in size
   (e.g. giant "Middle"); refine with nav-tile clusters or community
   polygons where mining needs it.
4. **Anti-strat decay** — good teams change after being read; predictions
   must carry probabilities, and the dossier should flag "stale vs fresh"
   evidence. (Also why per-round prediction eval uses time-ordered splits.)
5. **Smokes/mollies as area denial** — landing point is easy; *effect*
   (what line is cut) needs visibility computation; v1 ships landing-zone
   semantics, visibility upgrade later.
6. **LLM spatial reasoning limits** — the design deliberately never asks
   the LLM to do geometry; if Phase 5 shows residual spatial errors, add
   more mined predicates instead of smarter prompts.
7. **ToS/legal** — HLTV bulk scraping prohibited; FACEIT demo API is
   restricted; voice is privacy-sensitive. Manual/targeted downloads for
   research; commercial product needs licensed data paths.
8. **Open research question worth owning**: which serialization of
   spatiotemporal esports rounds maximizes LLM tactical inference? Nobody
   has published this for CS2. Phase 2's grammar + Map Card + Phase 5's
   ablations are a paper ("RoundScript: grounding LLM tactical analysis in
   demo-derived symbolic round representations") and a defensible moat.
9. **Stale pretrained map priors** — models "know" pre-rework layouts;
   a confidently wrong prior is worse than no prior. Mitigations: map quiz
   per (model, map) before trusting; Map Card is authoritative on
   conflict; card carries `game_version` and layout-change notes.
10. **Lexicon drift across map updates** — Valve edits geometry and place
    volumes; lexicon and lineup-cluster names must be re-compiled per game
    version, and TeamBooks must not mix zones from different card versions
    (card checksum travels with every artifact).

---

## 9. Key sources

Format & parsers
- demoparser2: github.com/LaihoE/demoparser (README = full prop/event surface)
- awpy 2.x: github.com/pnxenopoulos/awpy · awpy.readthedocs.io (parser
  output primer, nav, visibility, plotting) · pbdems2 format guide (docs.rs)
- demofile-net (saul), demoinfocs-golang v4 (markus-wa), source2-demo
  (Rupas1k), clarity (skadistats)
- healeycodes.com/compressing-cs2-demos (format walkthrough)
- CS2VoiceData (DandrewsDev); CS Demo Manager docs (acquisition, sources)

Academic
- Xenopoulos et al.: arXiv:2011.01324 (WPA), arXiv:2107.06495 (ggViz),
  arXiv:2209.09861 (ESTA), Optimal Team Economic Decisions (IJCAI-W 2021),
  GNN sports outcomes (2022)
- Szmida & Toka 2025: Temporal Heterogeneous GNNs for CS2 action valuation
- Durst et al. 2024: Learning to Move Like Professional CS Players (MLMOVE)
- Ma et al.: arXiv:2312.11865 (TextStarCraft II, Chain of Summarization)
- TacticalGPT (StatsBomb 2023); MatchTime (EMNLP 2024); Wang & Yoshinaga
  2024 (esports commentary from data); SportR (arXiv:2511.06499);
  EgoEsportsQA (arXiv:2604.12320)

Graph/space encoding for LLMs (Map Card foundations)
- Fatemi, Halcrow, Perozzi: arXiv:2310.04560 "Talk like a Graph" (ICLR
  2024) + GraphQA benchmark (github.com/google-research/talk-like-a-graph)
- NLGraph (Wang et al. 2023, arXiv:2305.10037); MANGO benchmark (Ding et
  al. 2024); spatial-memory graph rectification (arXiv:2510.04195)
- osmAG map comprehension with LLMs (arXiv:2403.08228); TopoNav
  (arXiv:2509.01364); Tag Map (arXiv:2409.15451); ZorkGPT
  (github.com/stickystyle/ZorkGPT)

Map/zone data sources (Map Card inputs)
- mwridgway/CS2Callouts (env_cs_place polygon extraction, awpy-aligned)
- xobust/CS2CalloutExtractor (.NET, JSON/CSV via ValveResourceFormat)
- totalcsgo.com/callouts (community names + aliases, ~383 across 8 maps)
- boltgolt/boltobserv (radar calibration configs);
  pr1malator/pr1maly (callout rectangles, role zones, coordinate→callout)

Products & OSS attempts
- noesis.gg (pro-analyst tendency workflow, manual), scope.gg, leetify.com,
  cs-demo-manager.com, Refrag; smartcoach.gg, cs2coach.gg
- github.com: mohammed-alsaqqa/CS2-AI-coach, faisalkhan07/cs2-ai-coach,
  JBRKR000/cs2-demo-coach, afw825/cs2-antistrat,
  Starfie1d1272/cs2-demo-analysis-kit (DAK Studio)

REPORT COMPLETE
