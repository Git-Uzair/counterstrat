"""Tests for the anti-hallucination lint gate and the zone map prompt block."""

from counterstrat.llm.lint import lint_dossier
from counterstrat.mapcard.lexicon import build_lexicon
from counterstrat.mining.tendencies import build_teambook
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


def _mk_scripts(
    n: int = 12,
    first_contact_zone: str = "Middle",
    minority_zone: str = "Water",
    minority: int = 3,
    team_key: str = "abc",
) -> list[RoundScript]:
    scripts: list[RoundScript] = []
    for i in range(n):
        match_id = f"m{i + 1}"
        round_num = 1
        is_minority = i >= (n - minority)
        zone = minority_zone if is_minority else first_contact_zone
        fc = KillEvent(
            t=15.0 + float(i),
            killer="p1",
            victim="e1",
            killer_side="T",
            zone=zone,
            weapon="ak47",
            headshot=True,
            traded_within_4s=False,
        )
        beat = Beat(
            label="B+15",
            t=15.0,
            t_form=Formation(zones=[(2, "Middle"), (3, "TSpawn")]),
            ct_form=Formation(zones=[(5, "CTSpawn")]),
        )
        util = UtilEvent(
            t=5.0,
            thrower="p1",
            side="T",
            nade="smoke",
            from_zone="TSpawn",
            to_zone="Middle",
            lineup_id="Mid-Smoke",
        )
        plant = PlantEvent(
            t=45.0,
            site="BombsiteA",
            planter="p1",
            alive_t=3,
            alive_ct=2,
        )
        mov = [
            MovementLine(
                player="p1",
                side="T",
                role_hint="pack",
                sentence="p1(T): TSpawn > Middle k(e1)",
            ),
            MovementLine(
                player="p2",
                side="T",
                role_hint="lurk",
                sentence="p2(T): TSpawn > Water",
            ),
        ]
        s = RoundScript(
            match_id=match_id,
            map_name="de_anubis",
            card_checksum="chk123",
            round_num=round_num,
            score_t=0,
            score_ct=1,
            t_team_key=team_key,
            ct_team_key="xyz",
            economy={
                "T": EconSummary(
                    buy_type="full_buy", spend=20000, equip=25000, awps=1, loss_streak=0
                )
            },
            beats=[beat],
            kills=[fc],
            utility=[util],
            plant=plant,
            first_contact=fc,
            winner="T",
            reason="ct_killed",
            clock_used_s=50.0,
            movements=mov,
        )
        scripts.append(s)
    return scripts


def _mk_synthetic_bundle():
    scripts = _mk_scripts(
        n=12, first_contact_zone="Middle", minority_zone="Water", minority=3, team_key="abc"
    )
    teambook = build_teambook(scripts, "abc")
    lexicon = build_lexicon(
        "de_anubis", ["Middle", "Water", "BombsiteA", "BombsiteB", "TSpawn", "CTSpawn"]
    )
    evidence_ids = {f"{s.match_id}:{s.round_num}" for s in scripts}
    evidence_ids.add("deadbeefcafe:7")
    return scripts, teambook, lexicon, evidence_ids


def test_lint_catches_fabrication():
    _, teambook, lexicon, evidence_ids = _mk_synthetic_bundle()
    good = "…cites `Middle` (evidence deadbeefcafe:7) 75%…"
    bad = "…cites `Ghost` (evidence ffffffffffff:99) 12%…"
    lint_g = lint_dossier(good, teambook, lexicon, evidence_ids)
    lint_b = lint_dossier(bad, teambook, lexicon, evidence_ids)
    assert lint_g.ok
    assert not lint_b.ok and "Ghost" in lint_b.unknown_zones
    assert "ffffffffffff:99" in lint_b.bad_citations


def test_lint_freq_mismatch():
    _, teambook, lexicon, evidence_ids = _mk_synthetic_bundle()
    # In synthetic teambook, on full_buy attacks Middle 75%. 12% is a mismatch for Middle.
    bad_freq = "On full_buy attacks `Middle` (evidence deadbeefcafe:7) 12% of rounds"
    lint = lint_dossier(bad_freq, teambook, lexicon, evidence_ids)
    assert not lint.ok
    assert len(lint.freq_mismatches) > 0


def test_zone_map_formats_and_reaches_prompts():
    """Anchor coordinates from the editor's source render into the map-card
    block of the insights and chat prompts; the builder speaks canonical names
    (the renamer swaps in user callouts at the existing boundary)."""
    from counterstrat.llm.insights import build_insights_system
    from counterstrat.llm.prompts import build_chat_system, format_zone_map

    anchors = {"Middle": (0.53, 0.1, "default"), "BombsiteB": (0.61, 0.34, "lower")}
    text = format_zone_map(anchors)
    assert "`Middle` at (0.53, 0.10), upper level" in text
    assert "`BombsiteB` at (0.61, 0.34), lower level" in text
    # Single-level maps: no level suffix. No anchors: no section at all.
    assert format_zone_map({"Middle": (0.5, 0.25, "default")}).endswith("`Middle` at (0.50, 0.25)")
    assert format_zone_map({}) == ""

    _, teambook, _lexicon, _ = _mk_synthetic_bundle()
    for system in (
        build_insights_system("zones: {}\n", zone_map=text),
        build_chat_system("zones: {}\n", teambook, zone_map=text),
    ):
        block = system.split("<map_card>")[1].split("</map_card>")[0]
        assert "`Middle` at (0.53, 0.10), upper level" in block
