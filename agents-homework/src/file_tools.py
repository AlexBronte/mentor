"""Validated, atomic JSON storage used by the MCP server."""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def load_articles(file_path):
    """Validate all records before changing output."""
    data = json.loads(Path(file_path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list")
    ids = set()
    for i, article in enumerate(data):
        if not isinstance(article, dict) or not all(
            isinstance(article.get(k), str) and article[k].strip() for k in ("id", "title", "text")
        ):
            raise ValueError(f"Article {i}: expected non-empty string id, title, text")
        if article["id"] in ids:
            raise ValueError(f"Duplicate id at article {i}")
        if len(article["text"]) + len(article["title"]) > 16000:
            raise ValueError(f"Article {i} exceeds 16000 characters")
        ids.add(article["id"])
    return data


def atomic_write(file_path, data):
    """Write a temporary file and atomically replace the result."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def prepare_results(file_path):
    """Preserve old output and initialize a fresh list."""
    path = Path(file_path)
    backup = None
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"{path.stem}.{stamp}.bak.json")
        backup.write_bytes(path.read_bytes())
    atomic_write(path, [])
    return {"backup": str(backup) if backup else None}


def get_articles_count(articles):
    """Return count without transferring article text."""
    return len(articles)


def read_article(articles, index):
    """Read exactly one article, rejecting invalid indices."""
    if type(index) is not int or not 0 <= index < len(articles):
        raise ValueError("Article index is out of range")
    return articles[index]


def append_result(file_path, article, summary):
    """Save id/title from source; repeated identical saves are idempotent."""
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Empty summary")
    results = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("Result must be a JSON list")
    record = {"id": article["id"], "title": article["title"], "summary": summary.strip()}
    previous = next((r for r in results if r["id"] == article["id"]), None)
    if previous is not None:
        if previous != record:
            raise ValueError("Conflicting summary for the same id")
    else:
        results.append(record)
        atomic_write(file_path, results)
    return {"saved": article["id"], "count": len(results)}
