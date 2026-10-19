"""CLI: open one MCP session, invoke the complete graph once."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from src.graph import build_graph, call_tool
from src.summarizer import create_llm

ROOT = Path(__file__).resolve().parent


def describe_error(exc):
    """Unwrap async task groups so logs show the actual cause."""
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_error(child) for child in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


async def run(input_path, output_path, llm=None):
    """Discover tool schema and execute LangGraph via a real MCP subprocess."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different")
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Articles JSON not found: {input_path.resolve()}. "
            "Restore this file or specify --input PATH_TO_JSON."
        )
    model = llm if llm is not None else create_llm()
    params = StdioServerParameters(command=sys.executable, args=[
        str(ROOT / "mcp_server.py"), "--input", str(input_path.resolve()),
        "--output", str(output_path.resolve())])
    async with Client(params, read_timeout_seconds=90, cache=None) as client:
        discovered = await client.list_tools()
        read = next(tool for tool in discovered.tools if tool.name == "read_article")
        schema = {"name": read.name, "description": read.description, "parameters": read.input_schema}
        total = (await call_tool(client, "get_articles_count"))["count"]
        graph = build_graph(client, model, schema)
        return await graph.ainvoke({}, config={"recursion_limit": max(10, total * 6 + 3)})


def main():
    """Load .env, configure logs, parse paths and return a process exit code."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Groq + MCP + LangGraph article agent")
    parser.add_argument("--input", type=Path, default=ROOT / "data/test_articles.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/result.json")
    args = parser.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            ROOT / "logs/run.log", encoding="utf-8")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        state = asyncio.run(run(args.input, args.output))
    except KeyboardInterrupt:
        logging.error("Interrupted; saved results remain on disk.")
        return 130
    except Exception as exc:
        logging.error("Run failed: %s. Saved results remain on disk.", describe_error(exc))
        return 1
    logging.info("Complete: %s/%s -> %s", state["next_index"], state["total"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
