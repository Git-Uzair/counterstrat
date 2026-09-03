from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "demos" / "1-7065ab7c-bc8f-4995-adf1-ac774327c5db-1-1.dem"
ANUBIS_VPK = REPO / "maps" / "de_anubis" / "de_anubis.vpk"
ANCIENT_VPK = REPO / "maps" / "de_ancient" / "de_ancient.vpk"
VRF = REPO / "tools" / "vrf" / "Source2Viewer-CLI.exe"


@pytest.fixture(scope="session")
def demo_path() -> Path:
    if not DEMO.exists():
        pytest.skip("real demo fixture not present")
    return DEMO


@pytest.fixture(scope="session")
def anubis_vpk() -> Path:
    if not ANUBIS_VPK.exists():
        pytest.skip("anubis vpk not present")
    return ANUBIS_VPK


@pytest.fixture(scope="session")
def ancient_vpk() -> Path:
    if not ANCIENT_VPK.exists():
        pytest.skip("ancient vpk not present")
    return ANCIENT_VPK


@pytest.fixture(scope="session")
def vrf_cli() -> Path:
    if not VRF.exists():
        pytest.skip("VRF CLI not vendored (see plan Context for URL+sha256)")
    return VRF


@pytest.fixture(scope="session")
def anubis_assets(tmp_path_factory, anubis_vpk, vrf_cli):
    from counterstrat.mapcard.vrf import extract_map_assets

    return extract_map_assets(anubis_vpk, vrf_cli, tmp_path_factory.mktemp("assets"))


@pytest.fixture(scope="session")
def anubis_lake(tmp_path_factory, demo_path):
    from counterstrat.corpus import register_demo
    from counterstrat.lake.extract import extract_lake

    bdir = tmp_path_factory.mktemp("lake_session")
    rec = register_demo(demo_path, bdir / "corpus.jsonl")
    return extract_lake(rec, bdir / "lake")


# --- Synthetic chat/agent fixtures (Task 22): fully offline, no demo needed ---

# The de_anubis zone set, matching src/counterstrat/mapcard/overlays/de_anubis.yaml so the
# synthetic card stays compatible with the real overlay the lexicon builder applies.
SYNTHETIC_ZONES = [
    "Alley",
    "BombsiteA",
    "BombsiteB",
    "Bridge",
    "CTSpawn",
    "Canal",
    "Connector",
    "Heaven",
    "LowerTunnel",
    "Main",
    "MidDoors",
    "Middle",
    "Ruins",
    "Street",
    "TSpawn",
    "Tunnel",
    "Walkway",
    "Water",
]
SYNTHETIC_TEAM = "abc"
SYNTHETIC_OPPONENT = "xyz"

# (round_num, team side, buy class, first-contact zone, plant site, winner side, score_t, score_ct)
# Scored so the miner yields two n=3 tendency groups -- (T, full_buy, ahead, won) over
# rounds 2-4 and (CT, full_buy, ahead, won) over rounds 14-16 -- plus low-n singletons.
SYNTHETIC_ROUNDS = [
    (1, "T", "full_buy", "Middle", "BombsiteA", "T", 0, 0),
    (2, "T", "full_buy", "Middle", "BombsiteA", "T", 1, 0),
    (3, "T", "full_buy", "Middle", "BombsiteA", "T", 2, 0),
    (4, "T", "full_buy", "Water", None, "T", 3, 0),
    (13, "CT", "full_buy", "BombsiteB", "BombsiteB", "CT", 6, 7),
    (14, "CT", "full_buy", "BombsiteB", "BombsiteB", "CT", 6, 8),
    (15, "CT", "full_buy", "BombsiteB", None, "CT", 6, 9),
    (16, "CT", "full_buy", "Middle", "BombsiteB", "CT", 6, 10),
    (17, "CT", "semi_eco", "BombsiteA", "BombsiteA", "T", 6, 11),
]


def build_synthetic_scripts(
    team_key: str = SYNTHETIC_TEAM, opponent: str = SYNTHETIC_OPPONENT
) -> list:
    """One match's RoundScripts: four T-side rounds then five CT-side rounds."""
    from counterstrat.roundscript.econ import EconSummary
    from counterstrat.roundscript.models import (
        Beat,
        Formation,
        KillEvent,
        MovementLine,
        PlantEvent,
        RoundScript,
        UtilEvent,
    )

    def econ(buy_type: str) -> EconSummary:
        return EconSummary(buy_type=buy_type, spend=20000, equip=25000, awps=1, loss_streak=0)

    scripts = []
    for round_num, team_side, buy, fc_zone, site, winner, score_t, score_ct in SYNTHETIC_ROUNDS:
        opp_side = "CT" if team_side == "T" else "T"
        fc = KillEvent(
            t=17.0,
            killer="p1",
            victim="e1",
            killer_side=team_side,
            zone=fc_zone,
            weapon="ak47",
            headshot=False,
            traded_within_4s=False,
        )
        scripts.append(
            RoundScript(
                match_id="m1",
                map_name="de_anubis",
                card_checksum="chk123",
                round_num=round_num,
                score_t=score_t,
                score_ct=score_ct,
                t_team_key=team_key if team_side == "T" else opponent,
                ct_team_key=team_key if team_side == "CT" else opponent,
                economy={team_side: econ(buy), opp_side: econ("full_buy")},
                beats=[
                    Beat(
                        label="B+15",
                        t=15.0,
                        t_form=Formation(zones=[(3, "Middle"), (2, "TSpawn")]),
                        ct_form=Formation(zones=[(3, "CTSpawn"), (2, "BombsiteA")]),
                    )
                ],
                kills=[fc],
                utility=[
                    UtilEvent(
                        t=8.0,
                        thrower="p1",
                        side=team_side,
                        nade="smoke",
                        from_zone="TSpawn",
                        to_zone="Middle",
                        lineup_id="Mid-Smoke",
                    )
                ],
                plant=(
                    PlantEvent(
                        t=45.0,
                        site=site,
                        planter="p1" if team_side == "T" else "e1",
                        alive_t=3,
                        alive_ct=2,
                    )
                    if site
                    else None
                ),
                first_contact=fc,
                winner=winner,
                reason="ct_killed" if winner == "T" else "t_killed",
                clock_used_s=55.0,
                movements=[
                    MovementLine(
                        player="p1",
                        side=team_side,
                        role_hint="pack",
                        sentence=f"p1({team_side}): TSpawn > Middle k(e1)",
                    ),
                    MovementLine(
                        player="p2",
                        side=team_side,
                        role_hint="lurk",
                        sentence=f"p2({team_side}): TSpawn > Water",
                    ),
                ],
            )
        )
    return scripts


def build_synthetic_card():
    """A MapCard covering exactly SYNTHETIC_ZONES."""
    from counterstrat.mapcard.compile import MapCard

    return MapCard(
        map="de_anubis",
        card_version="1.0",
        game_version="14178",
        nav_source="nav",
        frame={},
        zones={z: {"aliases": [], "tags": []} for z in SYNTHETIC_ZONES},
        topology={},
        rotates=[],
        timings={},
        objectives={},
        sightlines=[],
        checksum="chk123",
    )


class ScriptedToolClient:
    """Offline ``LLMClient`` stand-in that replays pre-baked assistant turns.

    Mirrors real provider behaviour in the one way the agent loop depends on: with
    no tools offered, the model cannot emit tool calls, so a text turn comes back.
    """

    def __init__(self, turns: list | None = None, fallback_text: str = "Final answer."):
        self.scripted = list(turns or [])
        self.fallback_text = fallback_text
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, user: str, max_tokens: int = 4096) -> Any:
        raise NotImplementedError

    def complete_json(self, *, system: str, user: str, schema: Any, max_tokens: int = 4096) -> Any:
        raise NotImplementedError

    def chat(self, *, system: str, turns: list, tools: list, max_tokens: int = 4096) -> tuple:
        from counterstrat.llm.base import ChatTurn, LLMResult

        self.calls.append({"system": system, "turns": list(turns), "tools": list(tools)})
        if tools and self.scripted:
            turn = self.scripted.pop(0)
        else:
            turn = ChatTurn(role="assistant", text=self.fallback_text)
        return turn, LLMResult(
            text=turn.text or "",
            input_tokens=100,
            output_tokens=20,
            model="mock-model",
            provider="mock",
        )


@pytest.fixture
def synthetic_scripts() -> list:
    return build_synthetic_scripts()


@pytest.fixture
def synthetic_card():
    return build_synthetic_card()


@pytest.fixture
def lake_con():
    """A duckdb connection with a tiny ``rounds`` table standing in for the lake views."""
    import duckdb

    con = duckdb.connect()
    con.sql(
        "create table rounds as "
        "select * from (values (1, 'T'), (2, 'CT'), (3, 'T')) v(round_num, winner)"
    )
    yield con
    con.close()


@pytest.fixture
def session_ctx(synthetic_scripts, synthetic_card, lake_con):
    from counterstrat.llm.tools import SessionContext
    from counterstrat.mapcard.lexicon import build_lexicon
    from counterstrat.mining.tendencies import build_teambook

    return SessionContext(
        team_key=SYNTHETIC_TEAM,
        map_name="de_anubis",
        teambook=build_teambook(synthetic_scripts, SYNTHETIC_TEAM),
        lexicon=build_lexicon("de_anubis", list(synthetic_card.zones.keys())),
        card=synthetic_card,
        con=lake_con,
        scripts={f"{s.match_id}:{s.round_num}": s for s in synthetic_scripts},
    )
