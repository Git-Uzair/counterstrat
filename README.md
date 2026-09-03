# Counter-Strat

Counter-Strat is a local CS2 tactical anti-stratting and opponent tendency analysis web application. It ingests match demos, compiles topological map cards from game VPKs, mines structured opponent books, and provides an interactive LLM analyst chat and downloadable scouting dossiers grounded in tick-level telemetry.

---

## Quickstart

### Environment & Prerequisites

- **Python**: 3.13 or newer
- **Package manager**: [`uv`](https://docs.astral.sh/uv/)
- **Platform**: Windows, Linux, macOS (ValveResourceFormat CLI is vendored for Windows x64; on Linux/macOS, provide the matching platform binary of `Source2Viewer-CLI` if compiling map cards directly from raw VPKs)

### Installation

Clone the repository and synchronize dependencies:

```bash
uv sync
```

### Vendoring VRF CLI

The ValveResourceFormat CLI decompiles Source 2 entity lumps (`.vents_c`) and extracts navigation meshes (`.nav`) from map VPKs.

- **Vendored binary path**: `tools/vrf/Source2Viewer-CLI.exe`
- **Upstream release**: ValveResourceFormat v20.0 ([GitHub Release](https://github.com/ValveResourceFormat/ValveResourceFormat/releases/tag/20.0))
- **SHA-256**: `d32ab327b8bbb42a2528866afb03bb582bdb779d0005488da32b90292afd3ff5`
- **Status**: Pre-vendored in the repository.

### Configuration

Create a `.env` file in the project root with your API key of choice:

```env
# Anthropic Claude (recommended default: claude-sonnet-5)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Or Google Gemini (default: gemini-2.5-pro; dev/eval: gemini-3.8-flash)
GEMINI_API_KEY=your_gemini_api_key
```

*Note:* API keys can also be configured dynamically in the web UI Settings dialog (gear icon in the navigation bar). Demo ingestion, lake extraction, and tendency mining run completely offline; only interactive chat, dossier compilation, and the map quiz require an LLM API key.

### Starting the App

Start the local server:

```bash
uv run counterstrat
```

The application starts on `http://127.0.0.1:8710`.

### Browser Walkthrough

1. Open `http://127.0.0.1:8710` in your web browser.
2. **Upload Demo**: Drag and drop one or more CS2 demos (`.dem`, FACEIT `.dem.zst`, `.dem.gz`, `.dem.bz2`) into the dropzone. A sample fixture demo is located at `demos/1-7065ab7c-bc8f-4995-adf1-ac774327c5db-1-1.dem`.
3. **Background Ingestion**: Monitor real-time status as the ingest worker processes the demo:
   - Registers demo in corpus manifest
   - Extracts Parquet telemetry lake and DuckDB views
   - Compiles MapCard from map VPK (if not already compiled)
   - Generates RoundScript DSL for each round
   - Mines opponent tendencies into `TeamBook`
4. **Select Team & Map**: Use the dropdown menus to select the team you want to analyze and the map.
5. **Interactive Analysis Chat**: Ask tactical anti-stratting questions:
   - *"What does this team do on full buys?"*
   - *"How do they execute onto A site?"*
   - *"What are their early round defaults on T side?"*
   - *"Where do they throw smokes on B site?"*
   
   The ReAct analyst agent uses inspection tools (`search_tendencies`, `get_round_script`, `sql_query`, `inspect_utility`) to retrieve verified evidence, cites specific `match:round` examples, and reports exact sample sizes.
6. **Download Dossier**: Click **Generate Dossier** to produce and download a structured Markdown scouting dossier (`.md`) summarizing defaults, setups, execute timing, economy patterns, and counter-strat recommendations.

---

## Map VPKs

Counter-Strat uses Valve Pak (`.vpk`) files to decompile map entities and extract official place names and walkable area geometry.

- **Placement**: `maps/<map>/<map>.vpk` (e.g. `maps/de_anubis/de_anubis.vpk`, `maps/de_ancient/de_ancient.vpk`)
- **Source**: `<SteamLibrary>/steamapps/common/Counter-Strike Global Offensive/game/csgo/maps/de_*.vpk`
- **Vanity files**: `de_*_vanity.vpk` files are **not needed** (they contain no entity or nav data used by the system).
- **Included maps**: VPKs for `de_anubis` and `de_ancient` are provided in the repository.
- **Game updates**: Map Cards are keyed to `(map, patch_version)`. When a CS2 patch alters map geometry or callouts, copy the updated `de_<map>.vpk` into `maps/<map>/<map>.vpk` to regenerate the card.

---

## Research Harnesses

Counter-Strat includes command-line research and evaluation harnesses:

### 1. Map Card Quiz Harness

Tests semantic callouts, area adjacency, and topological reasoning against compiled map cards:

```bash
uv run python -m counterstrat.mapcard.quiz de_anubis
# Options:
#   -n 50                 Number of questions (default: 50)
#   --seed 0              Random seed
#   --card-path <path>    Path to compiled card.yaml
#   --output <path>       Output path for results JSON
```

### 2. Evaluation Benchmark

Runs the multi-arm per-round prediction benchmark (evaluating baseline frequency models vs. LLM reasoning arms across round outcomes):

```bash
uv run python -m counterstrat.eval.benchmark --arms majority,freq_table --side T
# Options:
#   --arms majority,freq_table,llm_no_profile,full
#   --side T or CT
#   --train-ratio 0.8
#   --card-path <path>
#   --output data/eval/report.json
```

### 3. Utility Lineup Sweep

Runs a DBSCAN parameter sweep (`eps` clustering) over utility throws and landings from an extracted lake:

```bash
uv run python -m counterstrat.roundscript.utility --eps-sweep --lake-root data/lake/<match_id> --map de_anubis
# Options:
#   --out data/mapcards/<map>/lineup_sweep.json
```

---

## Architecture Overview

The repository is structured as a layered monorepo under `src/counterstrat`:

```
src/counterstrat/
├── lake/          # Telemetry ingestion: Parquet storage + DuckDB views
├── mapcard/       # Spatial semantics: VPK decompilation, nav graphs, KNN zone classifier
├── roundscript/   # Tactical DSL: round event serialization, utility clustering, AST linter
├── mining/        # Tendency mining: buy categories, defaults, executes, TeamBook
├── llm/           # Model integration: Anthropic + Gemini adapters, caching, dossier synthesis, eval
└── web/           # Application layer: FastAPI, background ingest worker, ReAct chat agent, web UI
```

### 1. Lake (`counterstrat.lake`)
Parses raw CS2 demos using `demoparser2` and `awpy` into a Parquet-backed data lake with DuckDB SQL views (`ticks`, `kills`, `rounds`, `grenades`, `damages`, `bomb_events`). Handles demo deduplication via content hashing and tracks metadata in `corpus.jsonl`.

### 2. MapCard (`counterstrat.mapcard`)
Compiles spatial ground truth for each map. Decompiles `default_ents.vents_c` via VRF CLI into `env_cs_place` volumes and parses `.nav` walkable areas. Builds a closed zone lexicon, an area adjacency graph, and a 3D KNN zone mapper that accurately classifies arbitrary `(x, y, z)` coordinates into callout zones.

### 3. RoundScript (`counterstrat.roundscript`)
Transforms granular telemetry into a compact, human- and LLM-readable RoundScript DSL. Serializes economy categories, opening setups, map control timings, grenade lineups (clustered via DBSCAN), and bombsite executions. A built-in linter validates syntax and prevents zone name hallucinations.

### 4. Mining (`counterstrat.mining`)
Aggregates round scripts across matches into structured opponent `TeamBook` records. Mines buy categories (eco, semi-eco, semi-buy, full-buy), opening default setups, bombsite execute timings, and utility tendencies with sample sizes (`n`) and frequencies.

### 5. LLM (`counterstrat.llm`)
Provides unified client abstraction for Anthropic Claude (with prompt caching and structured outputs) and Google Gemini. Manages token budget guards, per-round prediction evaluations, and automated scouting dossier generation verified against AST and markdown linters.

### 6. Web (`counterstrat.web`)
FastAPI application serving a responsive HTML5/CSS3 frontend. Includes background ingestion queues, REST endpoints for matches and dossiers, settings management, and a multi-turn ReAct analyst chat agent equipped with analytical tools (`search_tendencies`, `get_round_script`, `sql_query`, `inspect_utility`).
