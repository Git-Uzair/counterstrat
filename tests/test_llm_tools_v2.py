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
