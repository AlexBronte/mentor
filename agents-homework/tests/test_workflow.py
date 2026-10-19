"""Offline regression tests: real MCP subprocess and graph, deterministic LLM."""
import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage
from main import run
from src import file_tools
from src.summarizer import summarize_article


class FakeModel:
    def __init__(self, fail_at=None, bad_selection=False):
        self.calls = []
        self.fail_at = fail_at
        self.bad_selection = bad_selection
        self.summaries = 0

    def bind_tools(self, tools, **kwargs):
        self.schema = tools[0]
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        text = messages[-1].content
        if "next_index=" in text:
            index = int(re.search(r"next_index=(\d+)", text).group(1))
            if self.bad_selection:
                index += 1
            return AIMessage(content="", tool_calls=[
                {"id": "test", "name": "read_article", "args": {"index": index}}])
        self.summaries += 1
        if self.summaries == self.fail_at:
            raise RuntimeError("simulated model failure")
        return AIMessage(content="Краткое резюме текущей статьи.")


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "articles.json"
        self.output = Path(self.temp.name) / "result.json"
        self.articles = [
            {"id": str(i), "title": f"Title {i}", "text": f"UNIQUE_ARTICLE_{i}: facts."}
            for i in range(3)]
        self.source.write_text(json.dumps(self.articles), encoding="utf-8")

    def execute(self, model):
        return asyncio.run(run(self.source, self.output, model))

    def test_full_graph_real_mcp_and_bounded_context(self):
        model = FakeModel()
        state = self.execute(model)
        results = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(len(results), 3)
        self.assertEqual([r["id"] for r in results], ["0", "1", "2"])
        self.assertEqual([r["title"] for r in results], [a["title"] for a in self.articles])
        self.assertIsNone(state["current_article"])
        self.assertIsNone(state["current_summary"])
        self.assertNotIn("messages", state)
        self.assertEqual(model.schema["name"], "read_article")
        self.assertEqual(len(model.calls), 6)
        for messages in model.calls:
            self.assertEqual(len(messages), 2)
            text = messages[-1].content
            self.assertLessEqual(text.count("UNIQUE_ARTICLE_"), 1)
            if "next_index=" in text:
                self.assertNotIn("UNIQUE_ARTICLE_", text)

    def test_partial_result_survives_model_failure(self):
        with self.assertRaises(Exception):
            self.execute(FakeModel(fail_at=2))
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))), 1)

    def test_bad_agent_choice_is_bounded(self):
        model = FakeModel(bad_selection=True)
        with self.assertRaises(Exception):
            self.execute(model)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), [])

    def test_empty_input_and_backup(self):
        self.source.write_text("[]", encoding="utf-8")
        self.output.write_text('[{"old": true}]', encoding="utf-8")
        model = FakeModel()
        self.execute(model)
        self.assertEqual(model.calls, [])
        self.assertEqual(self.output.read_text(encoding="utf-8").strip(), "[]")
        backup = next(self.output.parent.glob("result.*.bak.json"))
        self.assertEqual(json.loads(backup.read_text(encoding="utf-8")), [{"old": True}])

    def test_invalid_source_keeps_previous_output(self):
        self.source.write_text('[{"id": "1"}]', encoding="utf-8")
        self.output.write_text("[]", encoding="utf-8")
        with self.assertRaises(Exception):
            self.execute(FakeModel())
        self.assertEqual(self.output.read_text(encoding="utf-8"), "[]")

    def test_same_path_is_rejected(self):
        before = self.source.read_bytes()
        with self.assertRaises(ValueError):
            asyncio.run(run(self.source, self.source, FakeModel()))
        self.assertEqual(self.source.read_bytes(), before)

    def test_missing_input_is_reported_before_starting_mcp(self):
        missing = self.source.parent / "missing.json"
        with self.assertRaisesRegex(FileNotFoundError, "Articles JSON not found"):
            asyncio.run(run(missing, self.output, FakeModel()))
        self.assertFalse(self.output.exists())

    def test_atomic_failure_preserves_previous_result(self):
        self.output.write_text("[]", encoding="utf-8")
        with patch("src.file_tools.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                file_tools.atomic_write(self.output, [{"id": "new"}])
        self.assertEqual(self.output.read_text(encoding="utf-8"), "[]")
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_idempotent_save_and_validation(self):
        file_tools.prepare_results(self.output)
        for _ in range(2):
            file_tools.append_result(self.output, self.articles[0], "summary")
        self.assertEqual(len(json.loads(self.output.read_text(encoding="utf-8"))), 1)
        with self.assertRaises(ValueError):
            file_tools.append_result(self.output, self.articles[0], "different")
        with self.assertRaises(ValueError):
            file_tools.read_article(self.articles, -1)
        self.source.write_text(json.dumps([self.articles[0]] * 2), encoding="utf-8")
        with self.assertRaises(ValueError):
            file_tools.load_articles(self.source)

    def test_truncated_summary_is_rejected(self):
        class Truncated:
            async def ainvoke(self, messages):
                return AIMessage(content="partial", response_metadata={"finish_reason": "length"})
        with self.assertRaises(ValueError):
            asyncio.run(summarize_article(Truncated(), self.articles[0]))


if __name__ == "__main__":
    unittest.main()
