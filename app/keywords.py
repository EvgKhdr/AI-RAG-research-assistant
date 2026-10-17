"""Keyword (BM25) search over chunk text, using SQLite FTS5 in DATA_DIR/keywords.db.

Mirrors the chunks stored in Chroma (vectorstore.py keeps the two in sync) so exact terms, names
and numbers can be found even when the embedding model misses them.
"""
import re
import sqlite3
from contextlib import closing

from db import DATA_DIR

DB_PATH = DATA_DIR / "keywords.db"
_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\w+(?:[-./]\w+)*")  # keeps ids like ABC-123 or v2.1 together
_STOPWORDS = set(
    "a an and are as at be been but by can could did do does for from had has have how i if in into is it "
    "its me my of on or our please she so than that the their them then there these they this to us was "
    "we were what when where which who why will with would you your about tell show find give list".split()
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
        "text, doc_id UNINDEXED, folder_id UNINDEXED, filename UNINDEXED, chunk UNINDEXED, page UNINDEXED, "
        "tokenize='porter unicode61 remove_diacritics 2')"
    )
    return conn


def count() -> int:
    with closing(_connect()) as conn:
        return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def add(rows: list[tuple[str, int, int, str, int, int]]) -> None:
    """rows: (text, doc_id, folder_id, filename, chunk, page)."""
    with closing(_connect()) as conn, conn:
        conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)


def rebuild(rows: list[tuple[str, int, int, str, int, int]]) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM chunks")
        conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", rows)


def delete(*, doc_id: int | None = None, folder_id: int | None = None) -> None:
    column, value = ("doc_id", doc_id) if doc_id is not None else ("folder_id", folder_id)
    with closing(_connect()) as conn, conn:
        conn.execute(f"DELETE FROM chunks WHERE {column} = ?", (value,))


def match_expression(question: str) -> str:
    """FTS5 query: "quoted phrases" stay exact, other words are OR-ed (BM25 ranks the best matches
    first). Stopwords are dropped. Every term is quoted, so user text cannot inject FTS syntax."""
    parts = []
    for phrase in re.findall(r'"([^"]+)"', question):
        words = _WORD.findall(phrase.lower())
        if words:
            parts.append('"' + " ".join(words) + '"')
    rest = re.sub(r'"[^"]*"', " ", question).lower()
    terms = [t for t in _TOKEN.findall(rest) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]
    parts += ['"' + " ".join(_WORD.findall(t)) + '"' for t in dict.fromkeys(terms)]
    return " OR ".join(parts)


def search(question: str, k: int, folder_ids: list[int] | None = None) -> list[dict]:
    """Top-k chunks by BM25, best first. Empty when the question has no usable keywords."""
    expr = match_expression(question)
    if not expr:
        return []
    sql = "SELECT text, doc_id, folder_id, filename, chunk, page FROM chunks WHERE chunks MATCH ?"
    args: list = [expr]
    if folder_ids:
        sql += f" AND folder_id IN ({','.join('?' * len(folder_ids))})"
        args += folder_ids
    sql += " ORDER BY bm25(chunks) LIMIT ?"
    args.append(k)
    with closing(_connect()) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [
        {"text": t, "doc_id": d, "folder_id": f, "filename": n, "chunk": c, "page": p or None}
        for t, d, f, n, c, p in rows
    ]
