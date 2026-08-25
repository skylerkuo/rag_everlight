from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from rag_app.config import Settings
from rag_app.metadata import SourceMeta, load_source_metadata, upsert_manifest
from rag_app.models.gemma4_llamacpp import Gemma4GGUF
from rag_app.utils import meaningful_text, strip_code_fence, write_md_with_front_matter

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a deterministic document-cleaning engine for RAG ingestion.
Your job is to convert extracted web-page text into faithful Markdown.

Rules:
1. Preserve every technical fact, model number, numerical value, unit, product feature,
   application, and meaningful heading from the source.
2. Do NOT invent, infer, translate, or add knowledge that is not in the input.
3. Remove obvious navigation boilerplate, cookie text, repeated menu labels, empty
   download placeholders, and duplicated headers/footers when they carry no content.
4. Organize the remaining content with Markdown headings, paragraphs, bullet lists,
   and tables when the source clearly represents tabular data.
5. Keep the original language(s) of the source.
6. Return Markdown only. Do not wrap the result in a code fence.
7. If the input contains no substantive retrievable content, return exactly: __EMPTY__
"""


def _fallback_html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    return "\n".join(x.strip() for x in soup.stripped_strings if x.strip())


def _load_or_create_txt(settings: Settings, meta: SourceMeta) -> Path:
    """Use the crawler-generated TXT when present; otherwise create one from HTML.

    This preserves the requested legacy flow: one HTML source -> one TXT -> one MD.
    """
    if meta.text_path:
        crawler_txt = settings.data_dir / meta.text_path
        if crawler_txt.exists():
            return crawler_txt

    raw = settings.data_dir / meta.raw_path
    if not raw.exists():
        raise FileNotFoundError(raw)
    target = settings.txt_html_dir / f"{meta.document_id}.txt"
    if not target.exists():
        target.write_text(_fallback_html_to_text(raw), encoding="utf-8")
    return target


def prepare_html(
    settings: Settings,
    force: bool = False,
    limit: int | None = None,
) -> dict:
    settings.ensure_dirs()
    sources = [x for x in load_source_metadata(settings.data_dir).values() if x.source_kind == "html"]
    sources.sort(key=lambda x: x.document_id)
    if limit is not None:
        sources = sources[:limit]

    model = Gemma4GGUF(
        settings.gemma_model_path,
        n_ctx=settings.gemma_n_ctx,
        n_gpu_layers=settings.gemma_n_gpu_layers,
        temperature=settings.gemma_temperature,
        max_tokens=settings.gemma_max_tokens,
    )

    records: list[dict] = []
    counts = {"processed": 0, "written": 0, "skipped_empty": 0, "skipped_existing": 0, "failed": 0}

    for i, meta in enumerate(sources, 1):
        out_path = settings.md_html_dir / f"{meta.document_id}.md"
        try:
            if out_path.exists() and not force:
                counts["skipped_existing"] += 1
                LOGGER.info("HTML %s/%s skip existing: %s", i, len(sources), out_path.name)
                continue

            txt_path = _load_or_create_txt(settings, meta)
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            counts["processed"] += 1
            if not meaningful_text(raw_text):
                counts["skipped_empty"] += 1
                LOGGER.info("HTML %s/%s empty before LLM: %s", i, len(sources), meta.title)
                continue

            user_prompt = f"""SOURCE TITLE: {meta.title}
SOURCE URL: {meta.source_url}

EXTRACTED TEXT:
{raw_text}
"""
            md = strip_code_fence(model.invoke(SYSTEM_PROMPT, user_prompt)).strip()
            if md == "__EMPTY__" or not meaningful_text(re.sub(r"[#*`>|_-]", "", md)):
                counts["skipped_empty"] += 1
                LOGGER.info("HTML %s/%s removed as empty/noise: %s", i, len(sources), meta.title)
                continue

            write_md_with_front_matter(
                out_path,
                {
                    "document_id": meta.document_id,
                    "source_kind": "html",
                    "title": meta.title,
                    "source_url": meta.source_url,
                    "source_raw_path": meta.raw_path,
                    "source_txt_path": str(txt_path.relative_to(settings.data_dir)) if txt_path.is_relative_to(settings.data_dir) else str(txt_path),
                    "language": meta.language or "",
                },
                md,
            )
            counts["written"] += 1
            records.append(
                {
                    "document_id": meta.document_id,
                    "source_kind": "html",
                    "title": meta.title,
                    "source_url": meta.source_url,
                    "md_path": str(out_path),
                    "page_number": None,
                    "page_image": None,
                    "status": "ready",
                }
            )
            LOGGER.info("HTML %s/%s -> %s", i, len(sources), out_path.name)
        except Exception as exc:
            counts["failed"] += 1
            LOGGER.exception("HTML failed %s: %s", meta.document_id, exc)

    if records:
        upsert_manifest(settings.manifest_path, records)
    return counts
