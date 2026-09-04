# Space, vision, and engagement range: tactical research for the analyst LLM

Date: 2026-09-04
Status: RESEARCH ONLY - no code changes. Every "verified" claim below was
run against this repo's data this session; sources listed at the end.

## 1. The problem, concretely

First Read saw a lone player holding Apartments with vision onto another
part of the map and read it as duel-seeking. The player was gathering
information - watching the corridor CTs would rotate or push through. The
correct counter-strat is entirely different:

- Read as **duel lurk** -> "avoid the peek, set up a trade" (what the AI said).
- Read as **info lurk** -> "deny the information (smoke or don't cross his
  sightline), then punish his delayed lurk-up timing" (what it should say).

The model cannot make that distinction today because the data it sees
carries **position without purpose**: it knows the player was in
`Apartments` for 30 seconds, but not what he could see, what he was
watching, whether he was set up to fight, or whether anyone could trade
him. This document is the research base for closing that gap: what CS
tactical theory actually says, which signals separate the readings, what
our data already supports (verified), and how vision data could be
incorporated.

## 2. Tactical theory: space, control, and the lurker

Synthesis of coaching material (CSGold map-control guide, CS2Hype role and
map-control guides, BLAST lurker guide, csspot) plus the quantitative
esports literature.

### 2.1 What "taking space" means

Map control is **managing space and information**: strategically taking
and holding areas to (a) limit the enemy's movement options, (b) gather
information, (c) cut or delay rotations, and (d) create the geometry for
an execute (splits, crossfires, exit denial). Space is never valuable for
itself; the doctrine phrase that recurs across sources is **"always have a
purpose when taking map control"** and its mirror, **"know when to give
up control to avoid unfavorable fights."**

Operationally, a team "has" an area when they can contest anyone entering
it - a player watching its entrances, utility denying it, or a trade setup
covering it. A team **concedes** an area when nobody sees it and nobody
punishes entry. Control is therefore three observable things:

1. **Presence** - bodies in or covering the area (occupancy - we already
   mine this via beats/formations).
2. **Vision** - somebody can see the area's entrances (requires view
   direction + sightlines - we mine neither today).
3. **Utility denial** - smokes/mollies making the area uncrossable
   (we already mine utility with zones and timestamps).

Key control concepts worth encoding as doctrine for the LLM:

- **Layered control**: interconnected areas taken together (banana + car
  on inferno; outside + ramp-lobby on nuke). Holding one layer without
  the next is exposed control.
- **False control / pressure**: appearing to take space to pull rotations
  while the real hit happens elsewhere. Signature: presence + noise
  (utility, a peek) followed by movement away without commitment.
- **Control conversion**: control only pays when converted - into an
  execute (site hit behind the control), information (a confirmed CT
  position), or a pick. Unconverted control that expires with the round
  timer is a tendency worth flagging in scouting ("they take mid, then
  don't use it").
- **Concession patterns**: which areas a team never contests (CT side
  giving banana every round; T side never checking apartments). These are
  free-space maps for the opponent - exactly what an anti-strat wants.

### 2.2 The lurker: information vs duel

The role literature is unambiguous that lurking is primarily an
**information and rotation-cutting** role, not a fragging role:
primary objectives listed as "cut rotations, create pressure, gather
intel, secure exits" (CS2Hype); "gathering information, holding angles,
waiting for the moment the opponent exposes their back" (EGB). The BLAST
guide distinguishes **passive lurkers** (prevent CTs taking map control,
watch corridors) from **aggressive lurkers** (seek early solo duels) -
the exact distinction First Read failed to make.

Timing doctrine (CS2Hype's table, corroborated elsewhere):

| Situation | Lurker action |
|---|---|
| Team executes other site | push flank **2-3s after first contact** to catch rotators |
| Enemies rotate through his area | let the first pass, take the second |
| <30s on clock | commit or rejoin |
| Enemy stack confirmed by sound | call it, hit the weak site |

The **anti-lurker doctrine** follows directly, and is what the counter-
strat surface should emit:

1. **Deny the information**: don't cross his watched sightline; smoke it
   off when executing; clear him with utility, not bodies.
2. **Punish the timing**: an info-lurker moves up 2-3s after your fake or
   your first contact - pre-aim the corridor he must use, or hold the
   flank with one player exactly in that window.
3. **Starve him**: if he's cutting rotations, rotate through the path he
   does NOT see (requires knowing what he sees - vision data again).

### 2.3 Behavioural signatures that separate the readings

Every one of these is computable from our lake **today** (columns
verified in `ticks.parquet` and `kills.parquet` this session):

| Signal | Info lurk | Duel lurk / hold | Data |
|---|---|---|---|
| View scanning | sweeps a corridor or holds a *transit* line | locks a *combat* angle | `yaw` std over stint (verified: a real lone Lobby holder showed yaw_std=9° - locked) |
| Movement | shift-walking, deep off-angle, repositions without contact | still or jiggle-peeking the same angle | `is_walking`, position variance |
| Engagement | doesn't fire for long stints; kills come late or never | fires on first sight, early kills | kill/damage events per stint |
| Trade support | far from nearest teammate (untradeable by design) | teammate within trade range | inter-player distances per tick |
| Post-event motion | moves 2-5s after FC/execute starts elsewhere | holds through the execute | stint end vs FC/plant anchors (already in scripts) |
| Weapon | any; often rifle/scout at range | close-range weapon in close position, AWP on long angle | `active_weapon_name`, `is_scoped` |

A deterministic **stint-purpose classifier** (info-watch / angle-hold /
trade-support / exit-deny / plant-cover) over these features is the
missing semantic layer. It does not need to be perfect: even labelling
stints with the *watched zone* and *trade-support distance* would have
flipped the First Read example from "duels" to "information".

## 3. Long range vs close range

### 3.1 The mechanics (numbers worth encoding as doctrine)

- **Damage falloff** is per-weapon: M4A4 33 -> 30 dmg from 0m to 50m,
  while UMP-45 falls 35 -> under 15 over the same distance (cs.money
  fundamentals). Pistols/shotguns/SMGs fall off fastest; rifles slowly;
  AWP effectively not at all (97.5% armor pen, one-shot to 74km listed
  fatal HS range - i.e. unlimited in practice).
- **Accuracy falloff**: spread makes SMGs/shotguns miss at range;
  standing-still and crouching extend effective range (Tec-9 headshot
  range 14m running -> 22m standing -> 27m crouched). Scoped rifles and
  AWP dominate long sightlines.
- Therefore the core doctrine: **fight the range your weapon wins**.
  SMG/eco setups want close, tight angles and to deny distance; rifle/AWP
  setups want long sightlines and to deny closing routes. The counter is
  always to force the *other* range: smoke the long sightline to force
  close fights against AWPs; hold distance and refuse entries against
  SMG force-buys.

### 3.2 What to mine from our data (all columns already exist)

- `kills.distance` per team/player/zone: a team's **engagement-range
  profile** ("their kills on nuke: 62% under 300u - they fight close;
  force long-range duels") and per-player preferences (AWPer's average
  kill distance, entry player's sub-200u multi-kills).
- **Zone range character** (map property): median kill distance observed
  per zone pair -> label sightlines long/medium/close. Feeds "what fight
  does this position offer" reasoning.
- Context flags already present: `thrusmoke`, `attackerblind`,
  `penetrated`, `noscope`, `headshot`, `hitgroup` - smoke-abuse and
  wallbang tendencies are one groupby away.
- Buy-type cross-reference (economy already mined): eco rounds with
  close-range kill clusters = they play SMG positions on saves -> where.

## 4. Vision: can "what a player can see" be computed and used?

Yes - and it is the highest-leverage missing data. Three routes, ranked
by evidence gathered this session.

### 4.1 Route A - demo-derived empirical visibility (recommended first)

Every clean kill/damage event is a **proof of sightline**: attacker XYZ
saw victim XYZ (unless `thrusmoke` or `penetrated`). Our kills table
carries both endpoints plus those flags (verified). Aggregating thousands
of kill/damage pairs per map into zone-to-zone counts yields an
**empirical visibility matrix**: `from-zone x to-zone -> (n_events,
median_distance)`. Properties:

- Zero new dependencies, zero geometry, fully offline, uses only existing
  parquet. Confidence grows with corpus size.
- Survivorship caveat: sightlines nobody ever fought across stay unknown
  (absence of evidence). Fine for scouting: fights that never happen
  matter less to counter-stratting than ones that do.
- Doubles as the **distance labeller** for 3.2 (same events carry
  `distance`).

### 4.2 Route B - geometric raycasts via awpy (exact, needs one artifact)

`awpy 2.0.2` ships `VisibilityChecker(path=tri_file).is_visible(p1, p2)`
with a BVH over map triangles, and a `VphysParser` that builds triangles
from a decompiled `.vphys` (verified by inspection of the installed
package). Feasibility probe results on our inferno VPK (verified,
debris cleaned):

- Current CS2 VPKs no longer ship a standalone `world_physics.vphys_c`;
  physics is embedded in `maps/<map>/world_physics.vmdl_c` (VPK listing
  verified).
- VRF decompile of that model explodes into ~7,700 `.dmx` hull files -
  parseable in principle, real work in practice.
- glTF export of the physics model produces an **empty** GLB (physics
  models have no render mesh) - dead end.
- Practical path: **awpy's prebuilt `.tri` artifacts** for official maps
  (one-time download per map by the user; offline afterwards), then
  precompute a zone-anchor visibility matrix + per-pair distances, and
  validate against Route A: near-100% of clean kills must be `is_visible
  == True`; `thrusmoke`/`penetrated` kills must not be counted. The
  validation harness is free because the demos provide labelled pairs.

### 4.3 Route C - nav-mesh 2.5D approximation

We already extract `.nav` per map. Grid raycasts over walkable areas
approximate visibility but ignore vertical occluders and props; the
literature and our own map complexity (nuke) make this too crude.
Not recommended.

### 4.4 What vision data unlocks (why it is worth it)

1. **`sightlines` on the card** - the field exists and has been empty on
   every map since the beginning (`sightlines: []`; the health check now
   flags it). Populated as `from -> to (long/medium/close, n observed)`
   it slots straight into the scene-graph block: edges gain "sees" labels
   alongside "connects".
2. **Gaze semantics for stints**: with per-tick `yaw` (already stored)
   plus a visibility matrix, each hold-stint gets a *watched zone*: cast
   the view ray, take the nearest visible zone it enters. Timeline lines
   upgrade from `MOVE p1 (T) enters Apartments` to
   `HOLD p1 (T) in Apartments watching TopofMid (locked, no trade
   support)`. That single line is the difference between the wrong and
   right reading in the motivating example.
3. **Counter-strat language**: "smoke `X` to blind their info position",
   "rotate via `Y`, he cannot see it", "their AWP sits on the 1400u
   `Arch -> Middle` line - force the close fight through `boiler`".
4. **Agent tool**: a `get_vision(zone)` tool (what does a player there
   see; who saw zone X this round) grounds chat answers about holds,
   crossfires, and safe rotates.

## 5. What the analytics literature validates

- **Map control as a win-probability pillar**: a live CS2 win-prob model
  on de_inferno uses *Voronoi-derived map control* alongside economy and
  firepower (Henry1145141919810/CS2_Win_Probability_Model, building on
  Xenopoulos et al.'s ESTA/WPA line). Control is quantifiable and
  predictive - our occupancy beats are the coarse version; vision-aware
  control (area you *see*, not just stand in) is the refinement.
- **Action valuation frameworks** (Xenopoulos WPA; THGNN action
  evaluation on 114 pro inferno matches) explicitly value utility as
  map-control moves - e.g. a mid smoke countering an A-long smoke reads
  as control balance. Confirms our "utility denial = control" framing.
- Our differentiator stays the same as the repr work: those systems
  output numbers; this app must output **evidence-grounded language**, so
  the representation layer (timeline + scene graph + vision annotations)
  is where the value lands.

## 6. Verified data inventory (this session, this repo)

- `ticks.parquet` already carries `pitch`, `yaw`, `active_weapon_name`,
  `is_scoped`, `is_walking`, `is_defusing`, `flash_duration`, `velocity`,
  plus positions - view-direction analytics need **no re-parse and no new
  columns**.
- `kills.parquet` already carries `distance`, attacker/victim/assister
  XYZ + place, `thrusmoke`, `attackerblind`, `penetrated`, `noscope`,
  `headshot`, `hitgroup` - the entire engagement-range and
  visibility-evidence program runs on existing tables.
- Live probe on the current corpus (nuke, round 1): the only lone-zone T
  in the 20-45s window held Lobby for 25s with **yaw_std 9°** (locked
  angle), stationary, Glock out - the stint-signature features separate
  behaviours on real data at trivial compute cost.
- awpy `visibility` module present in the installed 2.0.2; constructor
  and parser signatures verified by inspection (Section 4.2).
- One-off probe scripts live in `scratch/` (`lurk_probe.py`); extraction
  debris was removed.

## 7. How to evaluate any of this (before believing it)

Same discipline as the representation lab - measured, not argued:

1. **Visibility matrix validation** (free labels): clean kills must land
   in visible zone pairs; `thrusmoke`/`penetrated` kills are excluded
   from matrix building and can serve as negative controls for Route B
   raycasts.
2. **Stint-purpose classifier sanity**: behavioural ground truth exists
   without hand labels - "did the player fire during/immediately after
   the stint", "was a teammate within trade distance", "did he move
   within 5s of FC elsewhere". A classifier must agree with these
   post-hoc outcomes on held-out rounds before its labels enter a prompt.
3. **LLM-facing A/B** (repr_bench pattern): generate auto-graded
   questions from rounds where ground truth is computable - "was the
   Apartments player set up to be traded?", "which zone was p1 watching
   at 30s?", "what range do their kills on this map favor?" - and compare
   prompt variants with/without vision + doctrine blocks on both models.
4. **First Read qualitative gate**: the lint layer can mechanically check
   that any "watching/holding X" claim in generated insights matches the
   computed gaze data - hallucinated purposes become lintable.

## 8. Candidate work items (ranked; NOT scheduled - no changes yet)

| # | Item | Effort | Value | Depends on |
|---|---|---|---|---|
| 1 | Doctrine blocks in chat/insights prompts: space-purpose taxonomy, anti-lurk counters, engagement-range table, control-conversion language | S | High - fixes the misread class immediately, zero new data | nothing |
| 2 | Engagement-range profiles in miners/teambook (team + player kill-distance distributions, smoke/wallbang rates, eco-range signatures) | S-M | High - new tendency class from existing columns | nothing |
| 3 | Empirical zone visibility matrix (Route A) + `sightlines` card population + scene-graph "sees" annotations | M | High - unlocks vision language with zero deps | nothing |
| 4 | Gaze/stint semantics: watched-zone per hold-stint via yaw + matrix; upgrade timeline HOLD lines and add trade-support distance | M | Very high - the exact missing semantics from the motivating example | 3 |
| 5 | Stint-purpose classifier (info/duel/support/exit/plant) surfaced in role cards + First Read | M | High | 3, 4 |
| 6 | `get_vision` agent tool + vision-aware lint of insights | S | Medium | 3 |
| 7 | Route B geometric raycasts (awpy tri) + validation harness | M-L | Medium - exactness; needs per-map tri artifact (user download or DMX parsing) | 3 (for validation labels) |
| 8 | Voronoi-style control metric per beat (contested/held/conceded areas) in scripts + teambook control tendencies | L | Medium-high - quantified space-taking reads | 3 useful, not required |

Suggested first slice when implementation is greenlit: items 1-3 (all
independent of new geometry), then 4, then re-measure First Read quality
on the corpus before touching 5-8.

## 9. Sources

- CSGold, "Map Control" mechanics guide - control definition, layered/
  false control, purpose doctrine. csgold.net/en/mechanics/map-control
- CS2Hype, "Lurker" role guide - objectives, timing table,
  lurking-vs-baiting. cs2hype.com/roles/lurker
- CS2Hype, "Map Control Mastery" - phase-by-phase control, CT hold styles.
- BLAST.tv, "CS2 lurker guide" - passive vs aggressive lurking, gap
  exploitation, pro references. blast.tv/article/cs2-lurker-guide
- EGB, "Lurker in CS2: When Solo Play Actually Works" - info-first
  framing of the role.
- csspot.org, "What Does Map Control Mean" - space + information framing.
- cs.money, "Game fundamentals: how distances work in CS2" - falloff and
  accuracy numbers (M4A4/UMP-45, Tec-9 ranges).
- cs2damage.com damage-falloff; tradeit.gg CS2 weapon stats - per-weapon
  range/armor tables.
- Henry1145141919810/CS2_Win_Probability_Model (GitHub) - per-second CS2
  win probability with Voronoi map-control pillar, on de_inferno;
  builds on Xenopoulos et al. (ESTA 2022, WPA).
- "Evaluating Player Actions in Professional Counter Strike using
  Temporal Heterogeneous Graph Neural Networks" (2025) - action valuation
  incl. utility-as-control on 114 pro inferno matches; SHAP-based event
  impact.
- Xenopoulos et al. - Valuing Player Actions in CS:GO (WPA), ggViz -
  game-state retrieval; the win-probability lineage.

RESEARCH COMPLETE - no code changed; implementation awaits your go.

## 10. Implementation status (addendum, 2026-09-04)

Items 1-4 are implemented, tested, and committed on master; 5-8 are not
started. Suite at completion: 379 passed / ruff clean.

| # | Item | Status | Commit |
|---|---|---|---|
| 1 | Doctrine blocks (chat + insights system prompts) | DONE | a21adb6 |
| 2 | Engagement-range profiles (KillEvent fields, miner, prompt surfaces) | DONE | c35e01c |
| 3 | Empirical visibility matrix -> card `sightlines` + scene graph | DONE | a9d464c |
| 4 | Gaze/stint semantics (watched/locked/support_m, HOLD timeline lines) | DONE | a094752 |
| 5 | Stint-purpose classifier in role cards + First Read | NOT STARTED | - |
| 6 | `get_vision` agent tool + vision-aware lint | NOT STARTED | - |
| 7 | Route B geometric raycasts + validation harness | NOT STARTED | - |
| 8 | Voronoi-style control metric per beat | NOT STARTED | - |

Gate before 5-8 (per §8): re-measure First Read quality with/without the
new blocks (§7.3 repr_bench pattern). NOT RUN yet.

Corrections discovered during implementation:
- awpy `kills.distance` is in METERS, not Hammer units (verified:
  euclid(attacker,victim)/distance ~= 39.37 in/m). Range bands live in
  `constants.py` as RANGE_CLOSE_M=15 / RANGE_LONG_M=35; §3.2's "300u"
  phrasing and the "1400u Arch->Middle" exemplar should be read in meters
  equivalents (~7.6m / ~35.5m).
- Yaw convention verified on real kills (median 1.6 deg vs
  bearing-to-victim): degrees, 0 = +X, CCW positive.

Known gaps inside the implemented slice:
- First Read corpus timelines use `to_timeline_text(lite=True)`, which
  skips MOVE/HOLD lines - per-stint gaze reaches the dossier exemplars and
  the chat `get_round_script` tool, but NOT the First Read prompt. Item 5
  ("surfaced in First Read") is the intended fix.
- Cards populate `sightlines` (and scripts gain gaze fields) only on the
  next ingest or zone rebuild per map; maps not re-processed since the
  change still carry `sightlines: []` and unannotated stints.
- The dossier system prompt has no doctrine block (item 1 was scoped to
  chat/insights); it gains sightlines via the shared scene graph only.
