"""Tests for the analyst tools and the tool-using agent loop (offline via scripted client)."""

import json
import os
import re

import pytest
from conftest import ScriptedToolClient
from test_llm_clients import ReplayTransport

from counterstrat.llm import AnthropicClient, GeminiClient
from counterstrat.llm.agent import AgentReply, run_agent
from counterstrat.llm.base import ChatTurn, ToolCall
from counterstrat.llm.tools import SessionContext, execute_tool, tool_specs


def _tool_turn(name: str, arguments: dict, call_id: str = "c1") -> ChatTurn:
    return ChatTurn(
        role="assistant",
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


def _final(text: str) -> ChatTurn:
    return ChatTurn(role="assistant", text=text)


def _user(text: str) -> ChatTurn:
    return ChatTurn(role="user", text=text)


def _run(ctx: SessionContext, name: str, arguments: dict) -> dict:
    raw = execute_tool(ctx, ToolCall(id="t", name=name, arguments=arguments))
    return json.loads(raw)


# --- Tool specs & tool execution ---


def test_tool_specs_expose_the_five_contract_tools():
    names = [s.name for s in tool_specs()]
    assert names == [
        "get_tendencies",
        "list_rounds",
        "get_round_script",
        "get_role_cards",
        "sql_query",
    ]
    for spec in tool_specs():
        assert spec.description
        assert spec.input_schema["type"] == "object"


def test_get_tendencies_filters_side_and_buy_class(session_ctx: SessionContext):
    t_side = _run(session_ctx, "get_tendencies", {"side": "T"})
    assert t_side["tendencies"]
    assert all(row["key"]["side"] == "T" for row in t_side["tendencies"])

    # The mined n=3 group: T full buys while ahead after a win.
    biggest = max(t_side["tendencies"], key=lambda row: row["n"])
    assert biggest["n"] == 3 and biggest["low_n"] is False
    assert biggest["evidence"] == ["m1:2", "m1:3", "m1:4"]
    assert round(biggest["first_contact_zone"]["Middle"] * 100) == 67
    assert round(biggest["site_committed"]["A"] * 100) == 67

    ct_semi = _run(session_ctx, "get_tendencies", {"side": "CT", "buy_class": "semi_eco"})
    assert ct_semi["tendencies"]
    assert all(
        row["key"]["side"] == "CT" and row["key"]["buy_class"] == "semi_eco"
        for row in ct_semi["tendencies"]
    )
    row = ct_semi["tendencies"][0]
    # The honesty surface: sample size, low-n flag and citable evidence always ride along.
    assert row["n"] == 1 and row["low_n"] is True
    assert row["evidence"] == ["m1:17"]

    assert (
        _run(session_ctx, "get_tendencies", {"side": "CT", "buy_class": "full_eco"})["tendencies"]
        == []
    )


def test_list_rounds_filters_side_outcome_and_site(session_ctx: SessionContext):
    everything = _run(session_ctx, "list_rounds", {})
    assert [r["id"] for r in everything["rounds"]] == [
        "m1:1",
        "m1:2",
        "m1:3",
        "m1:4",
        "m1:13",
        "m1:14",
        "m1:15",
        "m1:16",
        "m1:17",
    ]
    assert everything["count"] == 9
    assert everything["rounds"][0]["summary"].startswith("R1 [")

    assert [r["id"] for r in _run(session_ctx, "list_rounds", {"side": "T"})["rounds"]] == [
        "m1:1",
        "m1:2",
        "m1:3",
        "m1:4",
    ]
    # Round 17 is the only CT-side round the team lost.
    assert [
        r["id"] for r in _run(session_ctx, "list_rounds", {"side": "CT", "won": False})["rounds"]
    ] == ["m1:17"]
    assert [r["id"] for r in _run(session_ctx, "list_rounds", {"site": "B"})["rounds"]] == [
        "m1:13",
        "m1:14",
        "m1:16",
    ]
    assert [
        r["id"] for r in _run(session_ctx, "list_rounds", {"buy_class": "semi_eco"})["rounds"]
    ] == ["m1:17"]
    # Rounds where the session's team did not play are never listed.
    lone = session_ctx.model_copy(update={"team_key": "nobody"})
    assert _run(lone, "list_rounds", {})["rounds"] == []


def test_get_round_script_hit_and_miss(session_ctx: SessionContext):
    hit = _run(session_ctx, "get_round_script", {"id": "m1:1"})
    assert hit["id"] == "m1:1"
    assert "B+15:" in hit["text"] and "FC:" in hit["text"]

    miss = _run(session_ctx, "get_round_script", {"id": "m1:99"})
    assert miss == {"error": "Round m1:99 not found"}


def test_get_role_cards_returns_teambook_roles(session_ctx: SessionContext):
    roles = _run(session_ctx, "get_role_cards", {})["roles"]
    assert {r["player"] for r in roles} == {"p1", "p2"}
    assert all("lurk_rate" in r and "modal_zone_fe15" in r for r in roles)


def test_unknown_tool_reports_error_instead_of_raising(session_ctx: SessionContext):
    out = _run(session_ctx, "drop_everything", {})
    assert out["error"] == "Unknown tool: drop_everything"


# --- SQL guard ---


def test_sql_tool_guard(session_ctx: SessionContext):
    bad = execute_tool(
        session_ctx,
        ToolCall(id="1", name="sql_query", arguments={"query": "insert into rounds values (1)"}),
    )
    assert "error" in bad
    ok = execute_tool(
        session_ctx,
        ToolCall(id="2", name="sql_query", arguments={"query": "select count(*) as n from rounds"}),
    )
    assert '"n"' in ok
    assert json.loads(ok)["rows"] == [{"n": 3}]


@pytest.mark.parametrize(
    "query",
    [
        "insert into rounds values (9, 'T')",
        "update rounds set winner = 'CT'",
        "delete from rounds",
        "drop table rounds",
        "create table evil as select 1",
        "alter table rounds add column x int",
        "select 1; delete from rounds",
        "attach 'other.db' as other",
        "copy rounds to 'out.csv'",
        "pragma database_list",
        "install httpfs",
        "  WITH x AS (select 1) select * from x",
    ],
)
def test_sql_guard_is_read_only(session_ctx: SessionContext, query: str):
    out = _run(session_ctx, "sql_query", {"query": query})
    assert "error" in out, f"guard let through: {query}"
    # The lake is provably untouched after every rejected statement.
    assert session_ctx.con.sql("select count(*) from rounds").fetchall()[0][0] == 3


def test_sql_query_caps_rows_and_reports_errors_as_data(session_ctx: SessionContext):
    capped = _run(session_ctx, "sql_query", {"query": "select * from range(500) t(i)"})
    assert len(capped["rows"]) == 50

    broken = _run(session_ctx, "sql_query", {"query": "select * from no_such_table"})
    assert "error" in broken

    no_lake = session_ctx.model_copy(update={"con": None})
    assert "error" in _run(no_lake, "sql_query", {"query": "select 1"})


# --- Agent loop ---


def test_agent_executes_tool_then_answers(session_ctx: SessionContext):
    scripted = ScriptedToolClient(
        [
            _tool_turn("get_tendencies", {"side": "T"}),
            _final("They hit `BombsiteA` on 67% of full buys (n=3, evidence m1:1)."),
        ]
    )
    reply = run_agent(
        scripted,
        "sys",
        [_user("what do they do on T full buys?")],
        session_ctx,
    )
    assert isinstance(reply, AgentReply)
    assert reply.tool_trace[0]["name"] == "get_tendencies"
    assert reply.tool_trace[0]["arguments"] == {"side": "T"}
    assert "67%" in reply.text
    assert reply.warnings == []

    # Second call carries the assistant tool-call turn plus its tool result.
    second_turns = scripted.calls[1]["turns"]
    assert second_turns[-2].tool_calls[0].name == "get_tendencies"
    assert second_turns[-1].role == "tool" and second_turns[-1].tool_call_id == "c1"
    assert '"n": 3' in (second_turns[-1].text or "")


def test_agent_sums_usage_across_iterations(session_ctx: SessionContext):
    scripted = ScriptedToolClient(
        [_tool_turn("get_role_cards", {}), _final("p1 lurks (evidence m1:3).")]
    )
    reply = run_agent(scripted, "sys", [_user("q")], session_ctx)
    assert len(scripted.calls) == 2
    assert reply.usage.input_tokens == 200
    assert reply.usage.output_tokens == 40
    assert reply.usage.provider == "mock"


def test_agent_truncates_long_tool_results_in_the_trace(session_ctx: SessionContext):
    scripted = ScriptedToolClient([_tool_turn("list_rounds", {}), _final("ok")])
    reply = run_agent(scripted, "sys", [_user("q")], session_ctx)
    preview = reply.tool_trace[0]["result_preview"]
    assert len(preview) == 203 and preview.endswith("...")


def test_agent_iteration_cap(session_ctx: SessionContext):
    looping = ScriptedToolClient([_tool_turn("get_role_cards", {})] * 20)
    reply = run_agent(looping, "sys", [_user("q")], session_ctx, max_iters=3)
    assert reply.text  # forced answer, no infinite loop
    assert len(reply.tool_trace) == 3
    assert len(looping.calls) == 4  # 3 capped iterations + 1 forced, tool-free answer
    assert looping.calls[-1]["tools"] == []
    assert looping.calls[-1]["turns"][-1].role == "user"


def test_agent_soft_lints_fabricated_zones_and_citations(session_ctx: SessionContext):
    scripted = ScriptedToolClient(
        [_final("They stack `Catwalk` every round (evidence deadbeefcafe:7).")]
    )
    reply = run_agent(scripted, "sys", [_user("q")], session_ctx)
    # Soft mode: the answer still comes back, with findings attached as warnings.
    assert reply.text
    assert "Unknown zone: Catwalk" in reply.warnings
    assert "Bad citation: deadbeefcafe:7" in reply.warnings


def test_agent_tool_failure_does_not_break_the_loop(session_ctx: SessionContext):
    scripted = ScriptedToolClient(
        [
            _tool_turn("sql_query", {"query": "drop table rounds"}),
            _final("No data for that (evidence m1:1)."),
        ]
    )
    reply = run_agent(scripted, "sys", [_user("q")], session_ctx)
    assert "error" in reply.tool_trace[0]["result_preview"]
    assert reply.text.startswith("No data")


@pytest.mark.parametrize(
    ("client_cls", "fixtures", "provider"),
    [
        (AnthropicClient, ("anthropic_chat_tool_use", "anthropic_chat_final"), "anthropic"),
        (GeminiClient, ("gemini_chat_tool_use", "gemini_chat_final"), "gemini"),
    ],
)
def test_agent_loop_runs_on_both_adapters(
    session_ctx: SessionContext, client_cls, fixtures: tuple[str, str], provider: str
):
    """The loop is provider-agnostic: same code path over both adapters' replayed turns."""
    client = client_cls(api_key="k", model="m", transport=ReplayTransport(*fixtures))
    reply = run_agent(client, "sys", [_user("what do they do?")], session_ctx)
    assert reply.text == "They default to a slow A execute off mid control."
    # The fixtures call a tool outside this contract, so the loop reports and recovers.
    assert [t["name"] for t in reply.tool_trace] == ["get_tendency"]
    assert "Unknown tool" in reply.tool_trace[0]["result_preview"]
    assert reply.usage.provider == provider
    assert reply.usage.input_tokens > 0
    assert reply.warnings == []


def _live_client(provider: str):
    """Dev-policy live clients: newest sonnet for Anthropic, DEV_GEMINI_MODEL for Gemini."""
    from counterstrat.config import DEV_GEMINI_MODEL, resolve_latest_sonnet

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return AnthropicClient(api_key=key, model=resolve_latest_sonnet(key))
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    )
    if not key:
        pytest.skip("no Gemini API key set")
    return GeminiClient(api_key=key, model=DEV_GEMINI_MODEL)


@pytest.mark.live
@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
def test_live_agent_answers_from_tools(session_ctx: SessionContext, provider: str):
    """The canonical question must come back tool-grounded, cited, and lint-clean."""
    from counterstrat.llm.prompts import build_system
    from counterstrat.web.chat import CHAT_TOOL_RULES

    client = _live_client(provider)
    system = (
        f"{build_system(session_ctx.card.to_yaml())}\n"
        f"<teambook>\n{session_ctx.teambook.to_table_text()}\n</teambook>\n\n"
        f"{CHAT_TOOL_RULES}"
    )
    reply = run_agent(
        client,
        system,
        [_user("What is their default CT setup on full buys? Cite rounds.")],
        session_ctx,
    )
    assert reply.tool_trace, "model answered without consulting a tool"
    citations = set(re.findall(r"\bm1:\d+\b", reply.text))
    assert citations & set(session_ctx.scripts), f"no valid citation in: {reply.text}"
    assert reply.warnings == [], f"lint warnings: {reply.warnings}"
    print(f"{provider} live agent usage: {reply.usage.model_dump()}")
    print(f"{provider} live tools: {[t['name'] for t in reply.tool_trace]}")
    print(f"{provider} live answer: {reply.text}")
