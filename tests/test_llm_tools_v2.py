"""Tests for the Task 6 analyst tools: playbook, utility book, gaps, econ, profiles."""

import json

import pytest
from conftest import SYNTHETIC_TEAM

from counterstrat.llm.base import ToolCall
from counterstrat.llm.tools import SessionContext, execute_tool, tool_specs
from counterstrat.mining.econ_policy import build_econ_policy
from counterstrat.mining.gaps import build_gap_report
from counterstrat.mining.utility_book import build_utility_book


@pytest.fixture
def session_ctx_v2(session_ctx: SessionContext, synthetic_scripts) -> SessionContext:
    return session_ctx.model_copy(
        update={
            "utility_book": build_utility_book(synthetic_scripts, SYNTHETIC_TEAM),
            "gap_report": build_gap_report(synthetic_scripts, SYNTHETIC_TEAM),
            "econ_policy": build_econ_policy(synthetic_scripts, SYNTHETIC_TEAM),
        }
    )


def _run(ctx: SessionContext, name: str, arguments: dict) -> dict:
    return json.loads(execute_tool(ctx, ToolCall(id="t1", name=name, arguments=arguments)))


def test_tool_specs_include_v2_tools():
    names = [s.name for s in tool_specs()]
    assert names == [
        "get_tendencies",
        "list_rounds",
        "get_round_script",
        "get_role_cards",
        "get_playbook",
        "get_utility_book",
        "get_gap_report",
        "get_economy_read",
        "get_player_profile",
        "get_rotation_report",
        "get_utility_roi",
        "get_death_profiles",
        "get_retake_report",
        "get_matchup",
        "sql_query",
    ]


def test_get_playbook_pistol(session_ctx_v2):
    out = _run(session_ctx_v2, "get_playbook", {"situation": "pistol"})
    assert out["situation"] == "pistol"
    assert "econ" in out and "pistol_sites" in out
    assert out["econ"]["n"] >= 1


def test_get_playbook_full_buy_returns_signal_rows(session_ctx_v2):
    out = _run(session_ctx_v2, "get_playbook", {"situation": "full_buy", "side": "T"})
    assert out["tendencies"], "synthetic T full-buy group is signal"
    for row in out["tendencies"]:
        assert row["signal"] is True
        assert row["key"]["side"] == "T"
        assert row["key"]["buy_class"] in ("full_buy", "any")


def test_get_playbook_after_loss_links_gaps(session_ctx_v2):
    out = _run(session_ctx_v2, "get_playbook", {"situation": "after_loss"})
    assert "econ" in out
    assert isinstance(out.get("related_gaps"), list)
    for f in out["related_gaps"]:
        assert f["trigger"] == "after_loss"


def test_get_utility_book_filters(session_ctx_v2):
    out = _run(session_ctx_v2, "get_utility_book", {"side": "T"})
    assert out["patterns"]
    assert all(p["side"] == "T" for p in out["patterns"])
    assert all(w["side"] == "T" for w in out["dump_windows"])

    filtered = _run(session_ctx_v2, "get_utility_book", {"side": "T", "nade": "flash"})
    assert filtered["patterns"] == []  # fixture throws smokes only


def test_get_gap_report_filters_trigger(session_ctx_v2):
    out = _run(session_ctx_v2, "get_gap_report", {"side": "CT"})
    assert out["findings"]
    assert all(f["side"] == "CT" for f in out["findings"])
    assert out["key_zones"]

    trig = _run(session_ctx_v2, "get_gap_report", {"side": "CT", "trigger": "after_util_dump"})
    assert {f["trigger"] for f in trig["findings"]} <= {"after_util_dump", "base"}


def test_get_economy_read(session_ctx_v2):
    out = _run(session_ctx_v2, "get_economy_read", {})
    assert out["policy"]
    assert out["ns"]
    for state, dist in out["policy"].items():
        assert abs(sum(dist.values()) - 1.0) < 1e-9, state


def test_get_player_profile(session_ctx_v2):
    all_out = _run(session_ctx_v2, "get_player_profile", {})
    assert {r["player"] for r in all_out["roles"]} == {"p1", "p2"}
    one = _run(session_ctx_v2, "get_player_profile", {"player": "p1"})
    assert len(one["roles"]) == 1 and one["roles"][0]["player"] == "p1"
    assert "opening_kill_rate" in one["roles"][0]
    missing = _run(session_ctx_v2, "get_player_profile", {"player": "ghost"})
    assert "error" in missing


def test_v2_tools_error_when_artifacts_missing(session_ctx):
    # A pre-upgrade session has no mined artifacts attached.
    for name, args in (
        ("get_playbook", {"situation": "pistol"}),
        ("get_utility_book", {"side": "T"}),
        ("get_gap_report", {"side": "T"}),
        ("get_economy_read", {}),
    ):
        out = _run(session_ctx, name, args)
        assert "error" in out, name


def test_get_tendencies_level_filter(session_ctx_v2):
    out = _run(session_ctx_v2, "get_tendencies", {"side": "T", "level": 2})
    assert out["tendencies"]
    assert all(row["level"] == 2 for row in out["tendencies"])


# --- Advanced analytics tools (2026-09-05 plan Task 3) ---


@pytest.fixture
def session_ctx_v3(session_ctx_v2: SessionContext) -> SessionContext:
    """session_ctx_v2 with enriched scripts: rotations, effects, death contexts."""
    from counterstrat.roundscript.models import KillEvent, RotationEvent

    scripts = dict(session_ctx_v2.scripts)
    r13 = scripts["m1:13"]  # the team plays CT here
    scripts["m1:13"] = r13.model_copy(
        update={
            "rotations": [
                RotationEvent(
                    t_trigger=20.0,
                    trigger="utility_near",
                    player="p1",
                    side="CT",
                    from_zone="BombsiteB",
                    to_zone="Middle",
                    latency_s=2.1,
                )
            ],
            "utility": [
                u.model_copy(update={"enemy_blind_s": 2.5, "team_blind_s": 0.5})
                for u in r13.utility
            ],
            "kills": [
                *r13.kills,
                KillEvent(
                    t=30.0,
                    killer="e1",
                    victim="p1",
                    killer_side="T",
                    zone="BombsiteB",
                    weapon="ak47",
                    headshot=False,
                    traded_within_4s=False,
                    distance=12.0,
                    victim_moving=True,
                    victim_preaim_off_deg=35.0,
                    victim_weapon="AK-47",
                ),
            ],
        }
    )
    return session_ctx_v2.model_copy(update={"scripts": scripts})


def test_get_rotation_report_filters_and_warns_about_audio(session_ctx_v3):
    out = _run(session_ctx_v3, "get_rotation_report", {"side": "CT"})
    assert "no footstep or sound" in out["note"]
    assert len(out["rows"]) == 1
    row = out["rows"][0]
    assert row["player"] == "p1" and row["trigger"] == "utility_near"
    assert row["median_latency_s"] == 2.1 and row["n"] == 1
    assert row["evidence"] == ["m1:13"]

    none = _run(session_ctx_v3, "get_rotation_report", {"side": "CT", "trigger": "plant"})
    assert none["rows"] == []


def test_get_utility_roi_renders_measured_rows(session_ctx_v3):
    out = _run(session_ctx_v3, "get_utility_roi", {"side": "CT"})
    assert out["rows"], "the enriched CT flash pattern must appear"
    smoke = next(r for r in out["rows"] if r["pattern"] == "Mid-Smoke")
    assert smoke["avg_enemy_blind_s"] == 2.5 and smoke["avg_team_blind_s"] == 0.5
    assert smoke["cost_per_enemy_blind_s"] is None  # n < 5: no verdict
    t_side = _run(session_ctx_v3, "get_utility_roi", {"side": "T"})
    assert all(r["side"] == "T" for r in t_side["rows"])


def test_get_death_profiles_by_player(session_ctx_v3):
    out = _run(session_ctx_v3, "get_death_profiles", {"player": "p1"})
    assert len(out["players"]) == 1
    p = out["players"][0]
    assert p["n"] == 1 and p["moving_rate"] == 1.0
    assert p["median_preaim_off_deg"] == 35.0
    assert p["weapons"] == {"AK-47": 1}
    assert [(b["band"], b["n"]) for b in p["by_range"]] == [("close", 1)]

    missing = _run(session_ctx_v3, "get_death_profiles", {"player": "ghost"})
    assert "error" in missing and "p1" in missing["error"]


def test_get_retake_report_rows_and_site_filter(session_ctx_v3):
    out = _run(session_ctx_v3, "get_retake_report", {})
    assert out["rows"], "synthetic plants must yield conversion rows"
    ct_b = next(r for r in out["rows"] if r["side"] == "CT" and r["site"] == "BombsiteB")
    assert ct_b["man_diff"] == "-1" and ct_b["win_rate"] == 1.0 and ct_b["n"] == 3

    only_a = _run(session_ctx_v3, "get_retake_report", {"site": "BombsiteA"})
    assert all(r["site"] == "BombsiteA" for r in only_a["rows"])


def test_get_matchup_diffs_both_books(tmp_path, session_ctx_v3, synthetic_scripts):
    import json as _json

    from conftest import SYNTHETIC_OPPONENT

    # Every demo books both teams: m1's scripts ARE the opponent's rounds too.
    scripts_dir = tmp_path / "scripts" / "m1"
    scripts_dir.mkdir(parents=True)
    for s in synthetic_scripts:
        (scripts_dir / f"round_{s.round_num:03d}.json").write_text(s.to_json(), encoding="utf-8")
    (tmp_path / "teams.json").write_text(
        _json.dumps(
            {
                "built_from": [],
                "clusters": [
                    {
                        "team_id": SYNTHETIC_OPPONENT,
                        "name": "team_xyz",
                        "lineup_keys": [SYNTHETIC_OPPONENT],
                        "steamids": [],
                        "matches": {"m1": {"map_name": "de_anubis", "rounds": 9}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ctx = session_ctx_v3.model_copy(update={"data_root": str(tmp_path)})

    out = _run(ctx, "get_matchup", {"opponent": "team_xyz"})
    assert out["opponent"] == {
        "name": "team_xyz",
        "team_id": SYNTHETIC_OPPONENT,
        "matches": 1,
        "rounds": 9,
    }
    for key in (
        "their_top_utility",
        "their_dump_windows",
        "their_range_profile",
        "our_gap_findings",
        "our_dump_windows",
        "our_range_profile",
        "overlap_zones",
    ):
        assert key in out, key

    unknown = _run(ctx, "get_matchup", {"opponent": "ghosts"})
    assert "error" in unknown and "team_xyz" in unknown["error"]


def test_get_matchup_without_data_root_errors(session_ctx_v3):
    out = _run(session_ctx_v3, "get_matchup", {"opponent": "team_xyz"})
    assert "error" in out


def test_get_round_script_returns_full_timeline(session_ctx: SessionContext):
    """Non-FC kills and MOVE lines reach the agent (plan 2026-09-04 Task 4)."""
    from counterstrat.roundscript.models import KillEvent, ZoneStint

    base = session_ctx.scripts["m1:1"]
    second_kill = KillEvent(
        t=39.0,
        killer="e2",
        victim="p1",
        killer_side="CT",
        zone="BombsiteA",
        weapon="m4a1",
        headshot=False,
        traded_within_4s=False,
    )
    rich = base.model_copy(
        update={
            "kills": [*base.kills, second_kill],
            "tracks": {
                "p1": [
                    ZoneStint(t0=0, t1=9, zone="TSpawn"),
                    ZoneStint(t0=9, t1=39, zone="Middle"),
                ]
            },
            "sides": {"p1": "T"},
        }
    )
    ctx = session_ctx.model_copy(update={"scripts": {**session_ctx.scripts, "m1:77": rich}})
    out = _run(ctx, "get_round_script", {"id": "m1:77"})
    text = out["text"]
    assert "t=39s KILL e2 (CT) kills p1 in `BombsiteA` [m4a1]" in text
    assert "t=9s MOVE p1 (T) enters `Middle`" in text
    assert "anchors: first_contact=17s" in text
