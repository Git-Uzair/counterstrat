# Needs from you

Status as of 2026-09-03 (rev. 3 — implementation complete). The product is fully
functional end-to-end. The table below summarizes the status of each external
dependency item.

| Item | Title | Status | Impact / Notes |
|---|---|---|---|
| **N1** | API keys | **SATISFIED** (Closed) | Live Anthropic (`ANTHROPIC_API_KEY`) and Gemini (`GEMINI_API_KEY` / `GOOGLE_GENERATIVE_AI_API_KEY`) working. Configurable via `.env` or UI settings modal. |
| **N2** | Radar images | **OPTIONAL** (Deferred) | Optional for evidence radar image overlays; product operates fully without it using text/data citations. Supply CS2 install path when radar plots are desired. |
| **N3** | Map VPKs | **DOCUMENTED** (Satisfied for v1) | `de_anubis` and `de_ancient` VPKs provided in `maps/<map>/<map>.vpk`. Instructions documented for adding new maps and refreshing on CS2 updates. |
| **N4** | Multi-demo corpora | **OPTIONAL** (Open for >300 round eval) | Fixture demo provided for dev/test. Web app accepts runtime uploads. Multi-demo corpus optional whenever expanded benchmark evaluation (>300 rounds) is wanted. |

## What the app is (confirmed understanding)

Locally hosted web app: you upload demos (`.dem`, FACEIT `.dem.zst`, `.gz`,
`.bz2`), the app ingests them in the background (parse → round scripts →
team tendency mining, ~30–60 s per demo), then you pick a team and **chat**:
"what does team X run on full buys?", "how do we anti-strat their B hits?",
"what's their default CT setup?" — answers are tool-grounded in the mined
data, cite `match:round` evidence, and state sample sizes. A scouting
dossier is downloadable per (team, map). No CLI product surface.

## N1 — API keys (blocks chat/dossier/quiz)

Status 2026-09-03:
- **`ANTHROPIC_API_KEY`: verified working** (system env; live-checked
  against `/v1/models`, HTTP 200; `claude-sonnet-5`, `claude-opus-5`,
  `claude-fable-5-1` all available). Default model: `claude-sonnet-5`.
- **Gemini: verified working** under the env name
  `GOOGLE_GENERATIVE_AI_API_KEY` (User scope; live-checked against
  `/v1beta/models`, HTTP 200, 54 models; `gemini-2.5-pro`,
  `gemini-3-flash-preview`, `gemini-3.1-pro-preview` all available). The
  app's config loader reads `GEMINI_API_KEY` → `GOOGLE_API_KEY` →
  `GOOGLE_GENERATIVE_AI_API_KEY` in that order, so no action needed.
  Default model: `gemini-2.5-pro`.

**N1 is fully satisfied — nothing left to do here.**

Ingest and mining run fully offline; only chat/dossier/quiz call out.

## N2 — Radar images (optional; evidence plots only)

**Status: OPTIONAL / DEFERRED**

Not present in the map VPKs you provided (verified) and awpy's hosted
artifacts currently 404. If you want radar-overlay evidence images later:
provide the **path** to a CS2 install (folder containing
`game/csgo/pak01_dir.vpk`) here or in `.env` (`CS2_INSTALL_PATH`) / UI settings — do **not** copy `pak01_dir.vpk` into the
repo: it is only the directory index of a split archive; content lives in
the `pak01_NNN.vpk` chunks beside it (tens of GB). The radar task reads the
game files in place via the vendored VRF CLI.

CS2 install path: `<fill in when wanted>`

(Community-documented radar location inside pak01 is
`panorama/images/map_icons/…/overheadmaps`; verifying that is step 1 of the
radar task.)

## N3 — Map VPKs (documented; satisfied for v1)

**Status: SATISFIED FOR V1 & FULLY DOCUMENTED**

Placement: `maps/<map>/<map>.vpk` — e.g. `maps/de_anubis/de_anubis.vpk`, `maps/de_ancient/de_ancient.vpk`.
Source: `...\steamapps\common\Counter-Strike Global Offensive\game\csgo\maps\de_*.vpk`. The `de_<map>_vanity.vpk` files are **not needed**
(verified: no entities/nav/radar content we use). Any map whose demos you
plan to upload should get its VPK; extras are harmless. Demos on maps
without a VPK still ingest and mine — chat just runs degraded (no
topology/rotate answers) until the VPK lands.

Map Cards are keyed to `(map, game_version)`; current cards build from your
VPKs at patch 14178. After a CS2 update touching a map in your pool,
re-supply that `de_<map>.vpk` once. See `README.md` for map VPK placement details.

## N4 — Multi-demo corpora (optional; runtime input)

**Status: OPTIONAL / OPEN (Fixture provided; pro corpus optional for >300 round eval)**

**Placement (if providing now):** drop files anywhere under `demos/` —
flat or in any subfolders you like, or upload via the web UI drag-and-drop zone; the pipeline ignores folder layout
because map and teams are read from the file contents (header + roster
steamids), and duplicates are deduped by content hash. Raw `.dem` or
compressed `.dem.zst` / `.gz` / `.bz2` all work. They do **not** need to be
from the same team (every demo feeds both teams' books) and do **not** need
to be grouped by map. One real constraint: demos on maps other than
Anubis/Ancient will parse and mine fine but get no Map Card until that
map's VPK lands in `maps/<map>/<map>.vpk` (chat works degraded: no
topology/rotate answers). Inventory check after dropping files:
`uv run python scratch/scan_corpus.py` — prints map, rounds, and
rounds-of-signal per (team, map), and writes
`scratch/corpus_inventory.json`.

**You do not need to send demos to build or run anything** — demos are what users
upload at runtime, and the one provided demo covers all development
fixtures. Two honesty notes instead:

1. **Answer quality scales with uploads.** One demo = match review
   (~15 rounds/side; the UI shows `n` and the model hedges). 5–10 demos of
   the same roster on a map = real tendencies. The app auto-groups uploads
   by roster, so "team X" accumulates across files.
2. **Three research-grade calibrations** want a bigger corpus when you have
   one: lineup-cluster eps sweep, buy-bin thresholds, and the Phase-5
   prediction benchmark (≥300 held-out rounds). They run on whatever has
   been uploaded; none gates a product feature.

## Decisions directed by you (locked)

- **Dev-time LLM policy:** any development/live-test call uses the latest
  Sonnet available on the key (resolved dynamically from `/v1/models`;
  currently `claude-sonnet-5`) for Anthropic, and `gemini-3.8-flash`
  (verified available) for Gemini.
- **Runtime:** app users provide their own API key and select the model in
  the settings UI (provider toggle + live model dropdown + write-only key
  field); config defaults are first-run seeds only.

## Decisions assumed (correct me if wrong)

- Web app, localhost, single analyst — no auth/multi-tenant/cloud in v1.
- Map pool v1 = de_anubis + de_ancient (the provided VPKs).
- Sightline/visibility analysis deferred (utility semantics = landing zone).
- Voice/chat-log analysis out of scope.
- No streaming responses in v1 chat (spinner + tool-call trace instead).
