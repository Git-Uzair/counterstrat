"""Tool-using analyst agent loop, shared by both provider adapters.

The loop is deliberately provider-agnostic: it only speaks ``LLMClient.chat`` plus
``ChatTurn``, so the Anthropic and Gemini adapters exercise the same code path.
"""

from pydantic import BaseModel, Field

from counterstrat.llm.base import ChatTurn, LLMClient, LLMResult
from counterstrat.llm.dossier import lint_dossier
from counterstrat.llm.tools import SessionContext, execute_tool, tool_specs

TRACE_PREVIEW_CHARS = 200
FORCE_ANSWER_NUDGE = "Answer now based on the information you have gathered so far."


class AgentReply(BaseModel):
    """One assistant answer plus what it consulted, what looked fabricated, and its cost."""

    text: str
    tool_trace: list[dict] = Field(default_factory=list)  # {name, arguments, result_preview}
    warnings: list[str] = Field(default_factory=list)  # soft lint findings
    usage: LLMResult  # summed across iterations


def run_agent(
    client: LLMClient,
    system: str,
    history: list[ChatTurn],
    ctx: SessionContext,
    max_iters: int = 8,
) -> AgentReply:
    """Answer the latest turn in ``history`` via tools, soft-linting the final text."""
    turns = list(history)
    tools = tool_specs()
    tool_trace: list[dict] = []
    final_text = ""
    input_tokens = output_tokens = cache_read_tokens = 0
    model = provider = "mock"

    def account(result: LLMResult) -> None:
        nonlocal input_tokens, output_tokens, cache_read_tokens, model, provider
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        cache_read_tokens += result.cache_read_tokens
        model, provider = result.model, result.provider

    for _ in range(max_iters):
        assistant_turn, result = client.chat(system=system, turns=turns, tools=tools)
        account(result)
        if not assistant_turn.tool_calls:
            final_text = assistant_turn.text or ""
            break
        turns.append(assistant_turn)
        for call in assistant_turn.tool_calls:
            res_str = execute_tool(ctx, call)
            preview = (
                res_str[:TRACE_PREVIEW_CHARS] + "..."
                if len(res_str) > TRACE_PREVIEW_CHARS
                else res_str
            )
            tool_trace.append(
                {"name": call.name, "arguments": call.arguments, "result_preview": preview}
            )
            turns.append(ChatTurn(role="tool", tool_call_id=call.id, text=res_str))
    else:
        # Iteration cap hit with the model still calling tools: force an answer by
        # re-asking with no tools offered (runaway tool loops are risk #8).
        turns.append(ChatTurn(role="user", text=FORCE_ANSWER_NUDGE))
        assistant_turn, result = client.chat(system=system, turns=turns, tools=[])
        account(result)
        final_text = assistant_turn.text or ""

    # Soft mode: surface fabricated zones/citations as warnings, never retry or block.
    lint = lint_dossier(
        final_text,
        ctx.teambook,
        ctx.lexicon,
        set(ctx.scripts.keys()),
        aliases=ctx.renamer.aliases if ctx.renamer else None,
    )
    warnings = [f"Unknown zone: {z}" for z in lint.unknown_zones]
    warnings += [f"Bad citation: {c}" for c in lint.bad_citations]

    return AgentReply(
        text=final_text,
        tool_trace=tool_trace,
        warnings=warnings,
        usage=LLMResult(
            text=final_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            model=model,
            provider=provider,
        ),
    )
