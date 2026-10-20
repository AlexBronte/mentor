"""Regression tests: actual MCP protocol, deterministic planner and summarizer."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage
from mcp import Client, StdioServerParameters
from langgraph.errors import GraphRecursionError
import sys
from main import ROOT, run
from mcp_server import create_server
from src.graph import build_graph, decode_result
from src import file_tools
from src.summarizer import summarize_article


class FakeSummary:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated Groq failure")
        return AIMessage(content="Test summary")


class Planner:
    """A deterministic test double only; production decisions come from Groq."""
    def __init__(self, first_action=None, always=None, reverse=False):
        self.messages = []
        self.actions = []
        self.first_action = first_action
        self.always = always
        self.reverse = reverse

    def bind_tools(self, schemas, **kwargs):
        self.schemas = schemas
        return self

    async def ainvoke(self, messages):
        self.messages.append(messages)
        status = json.loads(messages[1].content)["status"]
        args = {}
        if self.always:
            name = self.always
        elif self.first_action and not self.actions:
            name = self.first_action
        elif not status.get("opened"):
            name = "open_articles"
        elif status["saved"] == status["total"]:
            name = "finish"
        elif status["current_index"] is None:
            name = "read_article"
            args = {"index": status["total"] - 1 - status["saved"] if self.reverse
                    else status["next_unsaved"]}
        elif not status["summary_ready"]:
            name = "summarize_article"
        else:
            name = "append_result"
            args = {"index": status["current_index"]}
        self.actions.append((name, args))
        return AIMessage(content="", tool_calls=[{"id": "test", "name": name, "args": args}])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "articles.json"
        self.output = Path(self.temp.name) / "result.json"
        self.articles = [{"id": str(i), "title": f"Title {i}", "text": f"UNIQUE_ARTICLE_{i}"}
                         for i in range(3)]
        self.source.write_text(json.dumps(self.articles), encoding="utf-8")

    async def execute_async(self, planner, summary=None, steps=100):
        server = create_server(self.source, self.output, summary or FakeSummary())
        async with Client(server, cache=None) as client:
            discovered = await client.list_tools()
            schemas = [{"name": t.name, "description": t.description, "parameters": t.input_schema}
                       for t in discovered.tools]
            return await build_graph(client, planner, schemas).ainvoke({}, {"recursion_limit": steps})

    def execute(self, planner, summary=None, steps=100):
        return asyncio.run(self.execute_async(planner, summary, steps))

    def test_agent_controls_all_actions_and_context_is_bounded(self):
        planner, summary = Planner(), FakeSummary()
        state = self.execute(planner, summary)
        self.assertTrue(state["done"])
        self.assertEqual(state["status"]["saved"], 3)
        self.assertEqual({s["name"] for s in planner.schemas},
                         {"open_articles", "read_article", "summarize_article", "append_result", "finish"})
        self.assertEqual([a[0] for a in planner.actions],
                         ["open_articles"] + ["read_article", "summarize_article", "append_result"] * 3 + ["finish"])
        for messages in planner.messages:
            self.assertLessEqual(len(messages), 4)
            self.assertNotIn("UNIQUE_ARTICLE", messages[-1].content)
            self.assertNotIn("Test summary", messages[-1].content)
            self.assertLess(len(messages[-1].content), 1500)
        self.assertEqual(len(summary.calls), 3)
        for messages in summary.calls:
            self.assertEqual(messages[-1].content.count("UNIQUE_ARTICLE"), 1)
        results = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in results], ["0", "1", "2"])

    def test_graph_does_not_force_article_order(self):
        self.execute(Planner(reverse=True))
        self.assertEqual([r["id"] for r in json.loads(self.output.read_text())], ["2", "1", "0"])

    def test_premature_finish_is_returned_to_model_for_correction(self):
        planner = Planner(first_action="finish")
        self.assertTrue(self.execute(planner)["done"])
        self.assertIn("error", json.loads(planner.messages[1][1].content)["last_result"])

    def test_repeated_bad_calls_are_bounded(self):
        planner = Planner(always="finish")
        with self.assertRaises(Exception):
            self.execute(planner)
        self.assertEqual(len(planner.actions), 3)
        self.assertFalse(self.output.exists())

    def test_repeated_valid_calls_hit_step_budget(self):
        with self.assertRaises(Exception):
            self.execute(Planner(always="open_articles"), steps=8)
        self.assertEqual(json.loads(self.output.read_text()), [])

    def test_empty_input_and_backup(self):
        self.source.write_text("[]")
        self.output.write_text('[{"old": true}]')
        planner = Planner()
        self.assertTrue(self.execute(planner)["done"])
        self.assertEqual([a[0] for a in planner.actions], ["open_articles", "finish"])
        backup = next(self.output.parent.glob("result.*.bak.json"))
        self.assertEqual(json.loads(backup.read_text()), [{"old": True}])

    def test_partial_results_survive_failed_summarization(self):
        # All attempts after the first summary fail, keeping the first record.
        class Failing(FakeSummary):
            async def ainvoke(self, messages):
                if self.calls:
                    raise RuntimeError("Groq unavailable")
                return await super().ainvoke(messages)
        with self.assertRaises(Exception):
            self.execute(Planner(), Failing())
        self.assertEqual(len(json.loads(self.output.read_text())), 1)

    def test_invalid_input_does_not_change_previous_output(self):
        self.source.write_text('[{"id":"bad"}]')
        self.output.write_text("[]")
        with self.assertRaises(Exception):
            self.execute(Planner())
        self.assertEqual(self.output.read_text(), "[]")

    def test_server_guards_and_idempotency(self):
        async def check():
            async with Client(create_server(self.source, self.output, FakeSummary()), cache=None) as client:
                self.assertTrue((await client.call_tool("summarize_article", {})).is_error)
                await client.call_tool("open_articles", {})
                self.assertTrue((await client.call_tool("append_result", {"index": 0})).is_error)
                self.assertTrue((await client.call_tool("read_article", {"index": -1})).is_error)
                await client.call_tool("read_article", {"index": 0})
                self.assertTrue((await client.call_tool("read_article", {"index": 1})).is_error)
                await client.call_tool("summarize_article", {})
                await client.call_tool("append_result", {"index": 0})
                repeated = decode_result(await client.call_tool("append_result", {"index": 0}))
                self.assertEqual(repeated["status"]["saved"], 1)
                await client.call_tool("open_articles", {})
                self.assertEqual(len(json.loads(self.output.read_text())), 1)
                self.assertTrue((await client.call_tool("finish", {})).is_error)
        asyncio.run(check())

    def test_stdio_subprocess_discovery_and_lazy_open(self):
        async def check():
            params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py"),
                "--input", str(self.source), "--output", str(self.output)])
            async with Client(params, cache=None) as client:
                tools = await client.list_tools()
                self.assertEqual(len(tools.tools), 5)
                self.assertFalse(self.output.exists())
                data = decode_result(await client.call_tool("open_articles", {}))
                self.assertEqual(data["status"]["total"], 3)
        asyncio.run(check())

    def test_atomic_failure_preserves_output(self):
        self.output.write_text("[]")
        with patch("src.file_tools.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                file_tools.atomic_write(self.output, [{"id": "new"}])
        self.assertEqual(self.output.read_text(), "[]")
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_missing_and_identical_paths(self):
        with self.assertRaises(ValueError):
            asyncio.run(run(self.source, self.source, Planner()))
        with self.assertRaises(FileNotFoundError):
            asyncio.run(run(self.source.parent / "missing.json", self.output, Planner()))

    def test_truncated_summary_is_rejected(self):
        class Truncated:
            async def ainvoke(self, messages):
                return AIMessage(content="partial", response_metadata={"finish_reason": "length"})
        with self.assertRaises(ValueError):
            asyncio.run(summarize_article(Truncated(), self.articles[0]))


if __name__ == "__main__":
    unittest.main()
