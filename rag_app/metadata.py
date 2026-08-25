from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class SourceMeta:
    document_id: str
    source_kind: str
    title: str
    source_url: str
    raw_path: str
    text_path: str | None = None
    language: str | None = None
    page_count: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_source_metadata(data_dir: Path) -> dict[str, SourceMeta]:
    """Map crawler SHA/document id to URL/title/path metadata."""
    db_path = data_dir / "everlight.db"
    if db_path.exists():
        result = _load_from_db(db_path)
        if result:
            return result

    docs_path = data_dir / "rag_ready" / "documents.jsonl"
    if docs_path.exists():
        result: dict[str, SourceMeta] = {}
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            doc_id = d["document_id"]
            result[doc_id] = SourceMeta(
                document_id=doc_id,
                source_kind=d.get("source_kind", "unknown"),
                title=d.get("title") or d.get("source_title") or doc_id,
                source_url=d.get("source_url") or d.get("canonical_url") or "",
                raw_path=d.get("raw_path", ""),
                text_path=d.get("text_path"),
                language=d.get("language"),
                page_count=d.get("page_count"),
            )
        return result
    return {}


def _load_from_db(db_path: Path) -> dict[str, SourceMeta]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT dv.sha256, dv.title, dv.raw_path, dv.text_path, dv.page_count,
                   u.url, u.kind, u.language
            FROM document_versions dv
            JOIN urls u ON u.id = dv.url_id
            WHERE dv.is_current = 1
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        con.close()

    out: dict[str, SourceMeta] = {}
    for r in rows:
        doc_id = r["sha256"]
        out[doc_id] = SourceMeta(
            document_id=doc_id,
            source_kind=r["kind"],
            title=r["title"] or doc_id,
            source_url=r["url"] or "",
            raw_path=r["raw_path"] or "",
            text_path=r["text_path"],
            language=r["language"],
            page_count=r["page_count"],
        )
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def upsert_manifest(path: Path, records: Iterable[dict]) -> None:
    old = read_jsonl(path)
    merged = {str(x.get("md_path")): x for x in old if x.get("md_path")}
    for rec in records:
        key = str(rec.get("md_path"))
        if key:
            merged[key] = rec
    write_jsonl(path, sorted(merged.values(), key=lambda x: str(x.get("md_path", ""))))
