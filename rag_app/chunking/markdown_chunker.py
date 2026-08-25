from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from rag_app.config import Settings
from rag_app.metadata import write_jsonl
from rag_app.utils import meaningful_text, sha256_text


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HFTokenCounter:
    def __init__(self, model_id: str) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class ApproxTokenCounter:
    """Offline/test fallback; production should use HFTokenCounter."""
    def count(self, text: str) -> int:
        # Approximate English words + CJK characters + punctuation groups.
        return max(1, len(re.findall(r"[\u4e00-\u9fff]|\w+|[^\w\s]", text)))


@dataclass
class Section:
    heading_path: list[str]
    text: str


def parse_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    raw = text[4:end]
    body = text[end + 5 :]
    meta: dict[str, object] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            v = v[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        elif v.isdigit():
            v = int(v)
        meta[k.strip()] = v
    return meta, body.strip()


def split_sections(md: str) -> list[Section]:
    """Split by ATX headings while preserving a hierarchical heading path."""
    lines = md.splitlines()
    path: list[str] = []
    current: list[str] = []
    current_path: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip()
        if text:
            sections.append(Section(current_path.copy(), text))
        current = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            path = path[: level - 1]
            while len(path) < level - 1:
                path.append("")
            path.append(title)
            current_path = [x for x in path if x]
            current = [line]
        else:
            if not current and not current_path:
                current_path = []
            current.append(line)
    flush()
    return sections


def markdown_blocks(text: str) -> list[str]:
    """Paragraph/table/list-friendly block splitter.

    Blank lines define the primary block boundary. Markdown tables and consecutive
    list items naturally stay together because their rows/items are adjacent.
    """
    blocks = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    return blocks


def split_large_block(block: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    if counter.count(block) <= max_tokens:
        return [block]

    lines = [x for x in block.splitlines() if x.strip()]
    # Table: preserve header + separator for every row slice.
    if len(lines) >= 3 and "|" in lines[0] and re.match(r"^\s*\|?\s*:?-+", lines[1]):
        header = lines[:2]
        rows = lines[2:]
        out, cur = [], header.copy()
        for row in rows:
            candidate = "\n".join(cur + [row])
            if len(cur) > 2 and counter.count(candidate) > max_tokens:
                out.append("\n".join(cur))
                cur = header + [row]
            else:
                cur.append(row)
        if cur:
            out.append("\n".join(cur))
        return out

    # Prefer line/sentence boundaries before hard character slicing.
    units: list[str] = []
    for line in lines or [block]:
        parts = re.split(r"(?<=[.!?。！？；;])\s+", line)
        units.extend(x.strip() for x in parts if x.strip())

    out: list[str] = []
    cur: list[str] = []
    for unit in units:
        if counter.count(unit) > max_tokens:
            # Conservative fallback: character windows. CJK is often close to one
            # token/character; English is usually less, so this remains below max.
            char_window = max(200, max_tokens * 2)
            if cur:
                out.append(" ".join(cur))
                cur = []
            for i in range(0, len(unit), char_window):
                out.append(unit[i : i + char_window])
            continue
        candidate = " ".join(cur + [unit])
        if cur and counter.count(candidate) > max_tokens:
            out.append(" ".join(cur))
            cur = [unit]
        else:
            cur.append(unit)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_section(
    section: Section,
    counter: TokenCounter,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    blocks: list[str] = []
    for block in markdown_blocks(section.text):
        blocks.extend(split_large_block(block, counter, max_tokens))

    chunks: list[str] = []
    cur: list[str] = []
    for block in blocks:
        candidate = "\n\n".join(cur + [block])
        if cur and counter.count(candidate) > target_tokens:
            chunks.append("\n\n".join(cur).strip())
            # Block-level overlap; never crosses a heading section.
            overlap: list[str] = []
            for prev in reversed(cur):
                candidate_overlap = [prev] + overlap
                if counter.count("\n\n".join(candidate_overlap)) > overlap_tokens:
                    break
                overlap = candidate_overlap
            cur = overlap + [block]
            if counter.count("\n\n".join(cur)) > max_tokens:
                cur = [block]
        else:
            cur.append(block)

        if counter.count("\n\n".join(cur)) >= max_tokens:
            chunks.append("\n\n".join(cur).strip())
            cur = []
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return [x for x in chunks if x]


def _iter_md_files(settings: Settings) -> Iterable[Path]:
    yield from sorted(settings.md_html_dir.glob("*.md"))
    yield from sorted(settings.md_pdf_dir.glob("*/page_*.md"))


def build_chunks(settings: Settings, use_hf_tokenizer: bool = True) -> dict:
    settings.ensure_dirs()
    counter: TokenCounter = HFTokenCounter(settings.bge_model_id) if use_hf_tokenizer else ApproxTokenCounter()
    records: list[dict] = []
    doc_chunk_counter: dict[str, int] = {}

    for md_path in _iter_md_files(settings):
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_front_matter(text)
        if "no retrievable content" in body.lower() or not meaningful_text(body, min_chars=20):
            continue

        doc_id = str(meta.get("document_id") or md_path.stem)
        source_kind = str(meta.get("source_kind") or "unknown")
        title = str(meta.get("title") or doc_id)
        page_number = meta.get("page_number")
        page_image = meta.get("page_image")
        source_url = str(meta.get("source_url") or "")

        for section in split_sections(body):
            for content in chunk_section(
                section,
                counter,
                settings.chunk_target_tokens,
                settings.chunk_max_tokens,
                settings.chunk_overlap_tokens,
            ):
                token_count = counter.count(content)
                if token_count < settings.min_chunk_tokens and records:
                    # Do not merge across source files/pages/headings; simply keep short
                    # chunks if they contain strong structured information (tables/lists).
                    structured = "|" in content or re.search(r"(?m)^\s*[-*+]\s+", content)
                    if not structured:
                        continue

                doc_chunk_counter[doc_id] = doc_chunk_counter.get(doc_id, 0) + 1
                idx = doc_chunk_counter[doc_id]
                heading = " > ".join(section.heading_path)
                prefix_parts = [f"Title: {title}"]
                if heading:
                    prefix_parts.append(f"Section: {heading}")
                if source_kind == "pdf" and page_number:
                    prefix_parts.append(f"Page: {page_number}")
                embedding_text = "\n".join(prefix_parts) + "\n\n" + content
                chunk_id = sha256_text(f"{doc_id}|{page_number}|{idx}|{content}")
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "document_id": doc_id,
                        "source_kind": source_kind,
                        "title": title,
                        "source_url": source_url,
                        "md_path": str(md_path),
                        "page_number": page_number,
                        "page_image": page_image,
                        "heading_path": section.heading_path,
                        "token_count": token_count,
                        "content": content,
                        "embedding_text": embedding_text,
                    }
                )

    write_jsonl(settings.chunks_path, records)
    return {
        "markdown_files": sum(1 for _ in _iter_md_files(settings)),
        "chunks": len(records),
        "output": str(settings.chunks_path),
    }
