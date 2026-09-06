# Counter-Strat

**Scout your CS2 opponents like a pro team - on your own PC.**

Drop in demos of the team you're about to face. Counter-Strat parses every
round tick by tick, mines the other team's habits - their defaults, utility
lineups, economy calls, rotations, and the holes in their setups - and then
lets you *talk to an AI analyst* that answers only from that evidence:

> *"When do they hit B on full buys?"*
> *"What does their mid player do after they lose a pistol?"*
> *"Where is their A site weakest at 30 seconds?"*

Everything runs locally. The only thing that ever leaves your PC is the chat
text sent to the AI provider **you** configure with **your own** key.

---

## What you get

- **AI First Read** - a scouting report per team and map: how they play
  pistols, ecos, and full buys on both sides, and how to punish it.
- **Analyst chat** - ask follow-ups; the AI checks mined stats, round
  timelines, and even runs read-only SQL over the parsed data before it
  answers. Sample sizes are always shown, so you know what's real.
- **Tendency books** - opening setups, per-player positions, utility lineups
  with timings, economy policy, rotation tells, retake habits.
- **Site-hold gaps** - where and when they leave a bombsite genuinely
  uncovered (post-plant chaos and man-down rounds don't pollute the numbers).
- **Your callouts** - rename any zone ("camera", "ninja", "topmid"...) in a
  radar editor; the analyst speaks *your* language, not the game's.
- **Downloadable dossier** - a full anti-strat document you can share with
  your team.

Seven maps work out of the box with bundled radar and calibration data:
**Ancient, Anubis, Cache, Dust2, Inferno, Mirage, Nuke.**

---

## What you need

| Thing | Why | Cost |
|---|---|---|
| Windows 10/11 PC | CS2 demos parse fastest where CS2 lives | - |
| [uv](https://docs.astral.sh/uv/) | Installs and runs the app (brings its own Python) | Free |
| An AI key: [Google Gemini](https://aistudio.google.com/apikey) or [Anthropic](https://console.anthropic.com/) | Powers the First Read and chat | Gemini has a free tier; typical use is pennies |
| Counter-Strike 2 installed | The app reads radar images and map data from the game files (the app reminds you until it's set) | You own it already |
| Demos of your opponents | The evidence | Free (see below) |

> No Python knowledge needed. You will copy-paste about three commands total.

---

## Install

### Step 1 - Install uv (one command)

Open **PowerShell** (press `Win`, type `powershell`, Enter) and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then check it worked:

```powershell
uv --version
```

You should see a version number. That's it - uv will download the right
Python and every dependency automatically on first run.

### Step 2 - Get Counter-Strat

**Option A - with Git:**

```powershell
git clone https://github.com/Git-Uzair/counterstrat.git
cd counterstrat
```

**Option B - no Git:** on the [GitHub page](https://github.com/Git-Uzair/counterstrat)
click the green **Code** button → **Download ZIP**, extract it somewhere
(e.g. `D:\counterstrat`), then in PowerShell:

```powershell
cd D:\counterstrat
```

(Use the folder you extracted to - it's the one containing `pyproject.toml`.)

### Step 3 - Run it

```powershell
uv run counterstrat
```

The first run downloads dependencies and takes a couple of minutes. When you
see the server line, open your browser at:

**http://localhost:8710**

Leave the PowerShell window open while you use the app. `Ctrl+C` in that
window stops it.

---

## First-time setup (the app walks you through it)

On first launch a banner and the Settings window will tell you what's missing:

1. **CS2 install folder** - the app extracts radar images and map data from
   the game's files. To find yours: Steam → Library → right-click
   **Counter-Strike 2** → **Manage** → **Browse local files**, then copy the
   path from the Explorer address bar. It usually looks like:
   `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive`
   Paste it into **Settings → CS2 Install Folder** and save. The banner stays
   until the folder checks out - this one is required.

2. **AI provider + key** - in the same Settings window pick **Google Gemini**
   (free tier, recommended to start) or **Anthropic (Claude)**, paste your API
   key, save. The key is stored only on your PC, in `data/settings.json`, and
   is never displayed again.

3. **Maps beyond the built-in seven** *(optional)* - to analyze other maps,
   download the **Source2Viewer CLI** (`cli-windows-x64.zip`) from
   [ValveResourceFormat releases](https://github.com/ValveResourceFormat/ValveResourceFormat/releases)
   and extract it so this file exists:
   `counterstrat\tools\vrf\Source2Viewer-CLI.exe`
   The built-in seven maps need none of this.

---

## Getting opponent demos

Any CS2 GOTV/match demo works: `.dem`, or compressed `.dem.zst`, `.dem.gz`,
`.dem.bz2` - no need to unpack them first.

- **Your own matches:** in CS2 go to **Watch → Your Matches → Download**.
  The file lands in `...\Counter-Strike Global Offensive\game\csgo\replays\`.
- **FACEIT / other platforms:** every match room has a "Download demo" link.
- **Scrims/leagues:** ask for the GOTV demos - most organizers keep them.

The more matches of the same team on the same map, the stronger every read.
Three or more is where it gets interesting.

---

## Using it

1. **Drag the demo files** onto the drop zone (top-left). Multiple at once is
   fine. Parsing takes a minute or two per demo - progress is shown per file.
2. Teams appear in the **Catalog**. Pick the enemy team, tick the matches you
   want (or keep all), and hit **Analyze**.
3. Wait for the progress bar - the app is mining every round - then read the
   **AI First Read** cards: pistols, ecos, full buys, gotchas, and how to
   punish each.
4. **Ask questions** in the chat. Good openers:
   - *"What's their T pistol default and how do we counter it?"*
   - *"Show their smoke lineups on mid and when they throw them."*
   - *"Which player takes the fights and who lurks?"*
   - *"What changes after they lose two rounds in a row?"*
5. **Set your callouts** in the Callouts tab (click a label on the radar,
   type your name for it). Do this once per map - every future answer and
   report uses your words.
6. Hit **Regenerate** on the First Read after ingesting new demos of the same
   team so the report covers the new games.

Everything mined is saved in the `data/` folder next to the app - your demos,
books, chats, and callouts survive restarts. Delete `data/` and you start
clean.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `uv: command not found` | Reopen PowerShell (the installer edits PATH), or reinstall uv (Step 1). |
| Browser shows nothing at localhost:8710 | Check the PowerShell window for errors; the app only listens on your own PC (`127.0.0.1`), which is by design. |
| "CS2 install folder required" banner won't go away | The folder you picked must contain `game\csgo`. Point it at the folder named `Counter-Strike Global Offensive`, not `steamapps`. |
| Upload says "card missing" for a map | That map isn't in the built-in seven - set the CS2 folder and add the Source2Viewer CLI (setup step 3), then re-ingest. Analysis still works meanwhile, just without map-graph extras. |
| First Read says "No API key configured" | Settings → pick provider → paste key → Save. |
| A demo fails to parse | Very old demos (pre-CS2 or from ancient game builds) aren't supported by the parser. Recent match demos work. |
| Antivirus flags `Source2Viewer-CLI.exe` | It's the standard open-source Valve-format tool; download only from the official releases page linked above. |

---

## Privacy & cost

- Demos, parsed data, mined books, callouts, chats: **all local**, under
  `data/`. Nothing is uploaded anywhere.
- Only the text of your chat/First Read requests goes to the AI provider you
  chose, using your key. With Gemini's free tier the bill is usually zero;
  paid usage is typically cents per scouting session.
- The app binds to `127.0.0.1` only - nothing on your network can reach it.

---

## For developers

```powershell
uv sync                       # create the environment
uv run python -m pytest -q    # full test suite; runs without any API keys
uv run ruff check src tests   # lint
uv run counterstrat           # run the app
```

- `uv run counterstrat` then `http://localhost:8710/?mock=1` exercises chat
  without spending LLM calls.
- Architecture: demos → Parquet lake (`lake/`) → map cards & zone lexicon
  (`mapcard/`) → per-round tactical scripts (`roundscript/`) → mined books
  (`mining/`) → tool-grounded LLM analyst (`llm/`) → FastAPI + vanilla JS UI
  (`web/`). Shipped per-map calibration (anchors, topologies, radar art)
  lives inside the package under `src/counterstrat/mapcard/anchors/` and
  `src/counterstrat/radar/assets/`.
- Tests marked `demo` need real demo fixtures in `demos/`; `live` tests need
  API keys. Both are excluded by default.

---

## License

**Free for personal use.** Analyze your matches, prep your hobby team, mod it,
share it - enjoy.

**Commercial use needs a license.** If you want Counter-Strat (or any part of
it) inside your commercial website, software, service, or product, contact me
first via [GitHub](https://github.com/Git-Uzair) and we'll sort out a
commercial license.

Full terms in [LICENSE](LICENSE). CS2, its map data and radar images belong to
Valve; demo files belong to their owners.
