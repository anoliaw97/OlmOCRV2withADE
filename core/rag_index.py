from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.json_chunker import TextChunk, chunk_package_content
from core.loaders import DocumentPackage


@dataclass(slots=True)
class IndexedSearchResult:
    package_id: str
    source_file: str
    source_type: str
    content: str
    score: float
    section: str = ""
    page: str = ""
    table_name: str = ""


class LocalRagIndex:
    """SQLite-backed lightweight index over extracted JSON/MD/TXT chunks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.fts_enabled = True
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.connection.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_type TEXT NOT NULL,
                section TEXT,
                page TEXT,
                table_name TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL
            )
            """
        )

        try:
            cur.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content)")
        except sqlite3.OperationalError:
            self.fts_enabled = False

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def is_ready(self) -> bool:
        cur = self.connection.execute("SELECT COUNT(*) AS n FROM chunks")
        row = cur.fetchone()
        return bool(row and row["n"] > 0)

    def build_or_update(self, packages: Iterable[DocumentPackage]) -> int:
        total = 0
        for package in packages:
            total += self._replace_package(package)
        return total

    def _replace_package(self, package: DocumentPackage) -> int:
        chunks = chunk_package_content(package)

        cur = self.connection.cursor()
        existing_ids = [
            row["id"]
            for row in cur.execute("SELECT id FROM chunks WHERE package_id = ?", (package.package_id,)).fetchall()
        ]

        if existing_ids:
            if self.fts_enabled:
                cur.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(row_id,) for row_id in existing_ids])
            cur.execute("DELETE FROM chunks WHERE package_id = ?", (package.package_id,))

        for chunk in chunks:
            self._insert_chunk(cur, chunk)

        self.connection.commit()
        return len(chunks)

    def _insert_chunk(self, cur: sqlite3.Cursor, chunk: TextChunk) -> None:
        cur.execute(
            """
            INSERT INTO chunks (package_id, source_file, source_type, section, page, table_name, chunk_index, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.package_id,
                chunk.source_file,
                chunk.source_type,
                chunk.metadata.get("section", ""),
                chunk.metadata.get("page", ""),
                chunk.metadata.get("table_name", ""),
                chunk.chunk_index,
                chunk.content,
            ),
        )
        rowid = cur.lastrowid
        if self.fts_enabled:
            cur.execute("INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)", (rowid, chunk.content))

    def search(
        self,
        query: str,
        limit: int = 6,
        package_id: str | None = None,
    ) -> list[IndexedSearchResult]:
        cleaned = query.strip()
        if not cleaned:
            return []

        if self.fts_enabled:
            return self._search_fts(cleaned, limit, package_id)
        return self._search_like(cleaned, limit, package_id)

    def _search_fts(self, query: str, limit: int, package_id: str | None) -> list[IndexedSearchResult]:
        terms = [token for token in query.replace('"', " ").split() if len(token) > 1]
        if not terms:
            terms = [query]
        fts_query = " OR ".join(terms)

        sql = (
            "SELECT c.*, bm25(chunks_fts) AS rank "
            "FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
        )
        params: list[object] = [fts_query]

        if package_id:
            sql += " AND c.package_id = ?"
            params.append(package_id)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            cur = self.connection.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return self._search_like(query, limit, package_id)

        return [
            IndexedSearchResult(
                package_id=row["package_id"],
                source_file=row["source_file"],
                source_type=row["source_type"],
                content=row["content"],
                score=float(-row["rank"]),
                section=row["section"] or "",
                page=row["page"] or "",
                table_name=row["table_name"] or "",
            )
            for row in rows
        ]

    def _search_like(self, query: str, limit: int, package_id: str | None) -> list[IndexedSearchResult]:
        sql = "SELECT * FROM chunks WHERE content LIKE ?"
        params: list[object] = [f"%{query}%"]
        if package_id:
            sql += " AND package_id = ?"
            params.append(package_id)
        sql += " LIMIT ?"
        params.append(limit)

        cur = self.connection.execute(sql, params)
        rows = cur.fetchall()

        return [
            IndexedSearchResult(
                package_id=row["package_id"],
                source_file=row["source_file"],
                source_type=row["source_type"],
                content=row["content"],
                score=1.0,
                section=row["section"] or "",
                page=row["page"] or "",
                table_name=row["table_name"] or "",
            )
            for row in rows
        ]
