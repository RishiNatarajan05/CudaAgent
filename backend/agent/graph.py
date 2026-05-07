"""LangGraph ReAct agent with CUDA tools."""
from __future__ import annotations
import json
import re
from typing import AsyncIterator, Optional

from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from .tools import ALL_TOOLS
from .prompts import SYSTEM_PROMPT


MODEL_ID = "claude-sonnet-4-5"


def build_agent(max_steps: int = 12):
    llm = ChatAnthropic(model=MODEL_ID, max_tokens=4096, temperature=0)
    return create_react_agent(llm, tools=ALL_TOOLS)


def _initial_messages(repo_id: str, user_question: str) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT + f"\n\nActive repo_id: `{repo_id}`. Pass it as the repo_id argument to tools."),
        HumanMessage(content=user_question),
    ]


_ROOFLINE_MARK = re.compile(r"\[\[ROOFLINE_DATA\]\](\{.*?\})", re.DOTALL)
_ISSUES_MARK = re.compile(r"\[\[ISSUES_DATA\]\](\[.*?\])", re.DOTALL)


def _extract_special_payloads(tool_output: str):
    events = []
    for m in _ROOFLINE_MARK.finditer(tool_output):
        try:
            events.append({"type": "roofline_data", "data": json.loads(m.group(1))})
        except Exception:
            pass
    for m in _ISSUES_MARK.finditer(tool_output):
        try:
            events.append({"type": "issues_data", "issues": json.loads(m.group(1))})
        except Exception:
            pass
    return events


def _scrub_markers(text: str) -> str:
    text = _ROOFLINE_MARK.sub("", text)
    text = _ISSUES_MARK.sub("", text)
    return text


async def stream_agent(repo_id: str, user_question: str) -> AsyncIterator[dict]:
    """Async generator producing event dicts: agent_step / roofline_data / issues_data / final_answer / error."""
    try:
        agent = build_agent()
        msgs = _initial_messages(repo_id, user_question)
        last_ai_text = ""
        sources: list[dict] = []
        async for event in agent.astream({"messages": msgs}, stream_mode="updates"):
            for node, payload in event.items():
                msgs_out = payload.get("messages") if isinstance(payload, dict) else None
                if not msgs_out:
                    continue
                for m in msgs_out:
                    if isinstance(m, AIMessage):
                        # Tool calls invoked by the LLM
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                yield {
                                    "type": "agent_step",
                                    "tool": tc["name"],
                                    "input": tc.get("args", {}),
                                    "output": "",
                                }
                        if isinstance(m.content, str) and m.content.strip():
                            last_ai_text = _scrub_markers(m.content)
                        elif isinstance(m.content, list):
                            txt = "".join(b.get("text", "") for b in m.content if isinstance(b, dict))
                            if txt.strip():
                                last_ai_text = _scrub_markers(txt)
                    elif isinstance(m, ToolMessage):
                        out_text = m.content if isinstance(m.content, str) else str(m.content)
                        # Surface special payloads as their own events
                        for ev in _extract_special_payloads(out_text):
                            yield ev
                        # Track source previews for the SourcePanel
                        if m.name == "search_codebase":
                            sources.append({"tool": m.name, "preview": out_text[:1500]})
                        yield {
                            "type": "agent_step",
                            "tool": m.name,
                            "input": {},
                            "output": _scrub_markers(out_text)[:4000],
                        }
        yield {"type": "final_answer", "content": last_ai_text or "(no answer)", "sources": sources}
    except Exception as e:  # pragma: no cover
        yield {"type": "error", "message": str(e)}
