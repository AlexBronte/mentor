"""Official MCP SDK v2 server; stdout is reserved for stdio protocol."""
import argparse
from pathlib import Path
from mcp.server import MCPServer
from src import file_tools


def create_server(input_path, output_path):
    """Expose tools scoped to two configured files."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output must be different")
    articles = file_tools.load_articles(input_path)
    server = MCPServer("articles-homework")
    prepared = False

    @server.tool()
    def get_articles_count() -> dict:
        """Return the number of available articles."""
        return {"count": file_tools.get_articles_count(articles)}

    @server.tool()
    def prepare_results() -> dict:
        """Initialize output once per session, preserving a backup."""
        nonlocal prepared
        if prepared:
            return {"already_prepared": True}
        result = file_tools.prepare_results(output_path)
        prepared = True
        return result

    @server.tool()
    def read_article(index: int) -> dict:
        """Read one article by its zero-based index."""
        return file_tools.read_article(articles, index)

    @server.tool()
    def append_result(index: int, summary: str) -> dict:
        """Save summary with id/title taken from the original article."""
        if not prepared:
            raise ValueError("Call prepare_results first")
        return file_tools.append_result(output_path, file_tools.read_article(articles, index), summary)

    return server


def main():
    """Start the local stdio server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    create_server(args.input, args.output).run(transport="stdio")


if __name__ == "__main__":
    main()
