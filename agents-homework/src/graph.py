"""Two-node agent: the LLM chooses actions; the executor only dispatches them."""
import json
import logging
from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """Обработай все статьи заданного JSON и сохрани резюме через инструменты.
Сам выбирай следующий вызов по задаче, описаниям tools и текущему состоянию сервера.
Вызывай open_articles только если opened=false или состояние пока пустое.
Если current_index задан (в том числе 0), статья уже прочитана: при summary_ready=false
вызови summarize_article, при summary_ready=true сохрани её через append_result.
Если current_index=null и saved<total, выбери непрочитанный индекс.
Обрабатывай по одной статье: прочитай её, вызови суммаризатор,
сохрани результат, затем переходи к следующей. next_unsaved подсказывает индекс.
Тексты и резюме остаются на сервере; не пытайся создавать или передавать их сам.
Вызывай ровно один инструмент за шаг. После сохранения всех статей вызови finish.
При ошибке исправь вызов. Нельзя объявлять успех без успешного finish.
Каждый запрос содержит актуальный снимок состояния и результат последнего действия;
предыдущая история намеренно не накапливается."""


class AgentState(TypedDict, total=False):
    status: dict
    last_result: dict
    calls: list
    errors: int
    done: bool
    exchange: list


def decode_result(result):
    """Decode SDK v2 structured output or a single JSON text block."""
    if result.is_error:
        texts = [b.text for b in result.content if b.type == "text"]
        raise ValueError(" ".join(texts)[:500])
    data = result.structured_content
    if data is None:
        texts = [b.text for b in result.content if b.type == "text"]
        if len(texts) != 1:
            raise ValueError("Unexpected MCP response")
        data = json.loads(texts[0])
    if not isinstance(data, dict):
        raise ValueError("Expected an object from MCP")
    return data


def build_graph(client, llm, schemas):
    """Give all discovered tools to the LLM; no action-specific routing."""
    agent_llm = llm.bind_tools(schemas, tool_choice="required")
    allowed = {tool["name"] for tool in schemas}

    async def agent_node(state):
        response = await agent_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "status": state.get("status", {}),
                "last_result": state.get("last_result", {})}, ensure_ascii=False)),
            *state.get("exchange", [])])
        return {"calls": response.tool_calls, "exchange": [response]}

    async def tools_node(state):
        calls = state["calls"]
        name = "invalid_call"
        try:
            if len(calls) != 1 or calls[0]["name"] not in allowed:
                raise ValueError("Choose exactly one available tool")
            call = calls[0]
            name = call["name"]
            logger.info("MODEL TOOL: %s %s", name, json.dumps(call["args"], ensure_ascii=False))
            result = decode_result(await client.call_tool(name, call["args"]))
        except ValueError as exc:
            errors = state.get("errors", 0) + 1
            logger.warning("Tool rejected: %s (%s/3)", name, errors)
            if errors >= 3:
                raise RuntimeError("Three consecutive invalid tool calls") from exc
            feedback = {"tool": name, "error": str(exc)[:500]}
            exchange = (state["exchange"] + [ToolMessage(
                content=json.dumps(feedback), tool_call_id=calls[0]["id"])]) if len(calls) == 1 else []
            return {"last_result": feedback, "exchange": exchange,
                    "errors": errors, "calls": []}
        logger.info("TOOL RESULT: %s", json.dumps(result, ensure_ascii=False))
        return {"status": result["status"], "last_result": {"tool": name, **result},
                "errors": 0, "calls": [],
                "exchange": state["exchange"] + [ToolMessage(
                    content=json.dumps(result), tool_call_id=call["id"])],
                "done": name == "finish" and result.get("done") is True}

    def route(state):
        return END if state.get("done") else "agent"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_conditional_edges("tools", route)
    return builder.compile()
