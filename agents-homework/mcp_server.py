"""Stateful MCP tools; every workflow action must be requested by the model."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from mcp.server import MCPServer
from src import file_tools
from src.summarizer import create_llm, summarize_article as generate_summary

ROOT = Path(__file__).resolve().parent


def create_server(input_path, output_path, llm=None):
    """Bind a server to explicit paths. File loading is deferred to open_articles."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must be different")
    server = MCPServer("articles-homework")
    articles = None
    current_index = None
    current_summary = None
    saved = set()
    model = llm

    def status():
        """Return bounded metadata, never article text or summaries."""
        return {
            "opened": articles is not None,
            "total": len(articles) if articles is not None else None,
            "saved": len(saved),
            "next_unsaved": next((i for i in range(len(articles)) if i not in saved), None)
                if articles is not None else None,
            "current_index": current_index,
            "summary_ready": current_summary is not None,
        }

    @server.tool()
    def open_articles() -> dict:
        """Open configured JSON, validate articles, report count. Initialize output with backup. Idempotent within session; call first."""
        nonlocal articles
        if articles is None:
            loaded = file_tools.load_articles(input_path)
            file_tools.prepare_results(output_path)
            articles = loaded
        return {"status": status()}

    @server.tool()
    def read_article(index: int) -> dict:
        """Read one unsaved article into server memory. Returns metadata; text remains available to summarize_article. Save current article before switching."""
        nonlocal current_index
        if articles is None:
            raise ValueError("Open articles first")
        if index in saved:
            raise ValueError("Article already saved; select an unsaved index")
        if current_index is not None and current_index != index:
            raise ValueError("Save the current article before reading another")
        file_tools.read_article(articles, index)
        current_index = index
        return {"status": status()}

    @server.tool()
    async def summarize_article() -> dict:
        """Ask Groq to summarize the currently read article. Keep the summary on server; call after read_article and before append_result."""
        nonlocal current_summary, model
        if current_index is None:
            raise ValueError("Read an article first")
        if current_summary is None:
            if model is None:
                model = create_llm()
            current_summary = await generate_summary(model, articles[current_index])
        return {"status": status()}

    @server.tool()
    def append_result(index: int) -> dict:
        """Save the prepared summary for index atomically, with original id/title. Requires summarize_article. Repeating a saved index is harmless."""
        nonlocal current_index, current_summary
        if index in saved:
            return {"status": status()}
        if current_index != index or current_summary is None:
            raise ValueError("Read and summarize this article before saving")
        file_tools.append_result(output_path, articles[index], current_summary)
        saved.add(index)
        current_index = None
        current_summary = None
        return {"status": status()}

    @server.tool()
    def finish() -> dict:
        """Verify the result file and finish only when every article is saved. Call also for empty input after open_articles."""
        if articles is None or len(saved) != len(articles):
            raise ValueError("Cannot finish: open the file and save every article")
        results = json.loads(Path(output_path).read_text(encoding="utf-8"))
        expected = {a["id"]: a["title"] for a in articles}
        if (not isinstance(results, list) or len(results) != len(articles)
                or any(not isinstance(r, dict) or set(r) != {"id", "title", "summary"}
                       or not isinstance(r["summary"], str) or not r["summary"].strip()
                       for r in results)
                or {r["id"]: r["title"] for r in results} != expected):
            raise ValueError("Result file failed validation")
        return {"status": status(), "done": True}

    return server


def main():
    """Start stdio server; load its own .env because this is a separate process."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_server(args.input, args.output).run(transport="stdio")


if __name__ == "__main__":
    main()
