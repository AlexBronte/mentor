"""Agent choice followed by mandatory read -> summarize -> save graph nodes."""
import logging
import json
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from src.summarizer import summarize_article

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """Ты агент обработки статей. По компактному состоянию выбери
read_article с индексом next_index. Вызывай ровно один инструмент.
Не пропускай и не повторяй статьи. Граф сам суммаризирует и сохраняет статью.
Тебе не нужно формировать summary или выбирать пути файлов."""


class AgentState(TypedDict, total=False):
    total: int
    next_index: int
    selected_index: int
    attempts: int
    current_article: dict | None
    current_summary: str | None


async def call_tool(client, name, arguments=None):
    """Call MCP and reject tool errors or missing structured output."""
    result = await client.call_tool(name, arguments or {})
    if result.is_error:
        raise RuntimeError(f"MCP tool {name} failed: {result.content}")
    data = result.structured_content
    if data is None:
        texts = [block.text for block in result.content if block.type == "text"]
        if len(texts) != 1:
            raise RuntimeError(f"MCP tool {name} returned unexpected content")
        data = json.loads(texts[0])
    if not isinstance(data, dict):
        raise RuntimeError(f"MCP tool {name} returned no object")
    return data


def build_graph(client, llm, read_schema):
    """Build the graph using a tool schema discovered from the MCP server."""
    agent_llm = llm.bind_tools([read_schema], tool_choice="required")

    async def initialize_node(state):
        total = (await call_tool(client, "get_articles_count"))["count"]
        backup = await call_tool(client, "prepare_results")
        logger.info("Input: %s articles; backup: %s", total, backup.get("backup"))
        return {"total": total, "next_index": 0, "attempts": 0,
                "current_article": None, "current_summary": None}

    async def agent_node(state):
        response = await agent_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"total={state['total']}; next_index={state['next_index']}; "
                         f"saved={state['next_index']}. Вызови read_article для next_index.")])
        calls = response.tool_calls
        valid = (len(calls) == 1 and calls[0]["name"] == "read_article"
                 and calls[0]["args"] == {"index": state["next_index"]})
        if not valid:
            attempts = state["attempts"] + 1
            logger.warning("Invalid agent tool choice: %s/3", attempts)
            if attempts >= 3:
                raise RuntimeError("Agent failed to select the next article after 3 attempts")
            return {"selected_index": -1, "attempts": attempts}
        return {"selected_index": state["next_index"], "attempts": 0}

    async def read_node(state):
        article = await call_tool(client, "read_article", {"index": state["selected_index"]})
        logger.info("Reading article %s/%s", state["next_index"] + 1, state["total"])
        return {"current_article": article}

    async def summarize_node(state):
        return {"current_summary": await summarize_article(llm, state["current_article"])}

    async def save_node(state):
        saved = await call_tool(client, "append_result", {
            "index": state["selected_index"], "summary": state["current_summary"]})
        if saved["count"] != state["next_index"] + 1:
            raise RuntimeError("Saved count does not match graph progress")
        logger.info("Saved article %s/%s", saved["count"], state["total"])
        return {"next_index": state["next_index"] + 1,
                "current_article": None, "current_summary": None}

    def should_continue(state):
        return END if state["next_index"] >= state["total"] else "agent"

    def route_selection(state):
        return "read" if state["selected_index"] >= 0 else "agent"

    builder = StateGraph(AgentState)
    for name, node in (("initialize", initialize_node), ("agent", agent_node),
                       ("read", read_node), ("summarize", summarize_node), ("save", save_node)):
        builder.add_node(name, node)
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", should_continue)
    builder.add_conditional_edges("agent", route_selection)
    builder.add_edge("read", "summarize")
    builder.add_edge("summarize", "save")
    builder.add_conditional_edges("save", should_continue)
    return builder.compile()
