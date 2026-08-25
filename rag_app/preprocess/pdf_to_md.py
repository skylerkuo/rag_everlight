from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from rag_app.config import Settings
from rag_app.metadata import load_source_metadata, upsert_manifest
from rag_app.utils import strip_code_fence, write_md_with_front_matter

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document-understanding engine that converts one TARGET PDF page image into faithful Markdown for retrieval.
You may also receive neighboring PDF pages as CONTEXT. Use context only to understand document continuity.
Do not answer questions. Do not summarize away technical details. Never merge neighboring-page content into the target page.
"""

BASE_PAGE_PROMPT = """Convert the TARGET PDF page into Markdown for a technical RAG system.

Requirements:
- Output content for the TARGET page only.
- Neighboring page images are CONTEXT ONLY. Use them to resolve section continuity, table headers,
  figure/caption relationships, and whether content continues across page boundaries.
- Never copy paragraphs, rows, values, captions, or other factual content that appears only on a context page.
- If the TARGET page clearly continues a section whose heading appears only on the previous page,
  you may repeat that heading and mark it `(continued)`.
- If the TARGET page continues a table whose column headers appear only on the previous page,
  you may repeat the table header strictly as structural context, but never repeat rows from the previous page.
- Transcribe all meaningful visible text on the TARGET page faithfully.
- Preserve headings, paragraphs, bullet lists, model numbers, formulas, symbols, units,
  part numbers, and numerical values.
- Reconstruct clear tables as Markdown tables. Never invent missing table cells.
- For charts, circuit diagrams, block diagrams, product drawings, or figures on the TARGET page,
  include a concise `Figure description:` describing what is visibly shown. Do not infer unseen values.
- Ignore repetitive page decorations only when they contain no useful information.
- Keep the source language(s); do not translate.
- Return Markdown only; no code fence.
- If the TARGET page is truly blank, return exactly `__EMPTY__` even if context pages contain content.
"""


def _render_page(page: fitz.Page, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    pix.save(str(out_path))


def _context_page_indices(page_idx: int, page_count: int, radius: int) -> list[int]:
    """Return ordered page indices around the target, bounded to the document.

    radius=1 gives [previous, target, next] when all three exist.
    """
    radius = max(0, int(radius))
    start = max(0, page_idx - radius)
    end = min(page_count - 1, page_idx + radius)
    return list(range(start, end + 1))


def _page_role_label(ctx_idx: int, target_idx: int) -> str:
    page_no = ctx_idx + 1
    if ctx_idx == target_idx:
        return f"TARGET PAGE — page {page_no}. Convert ONLY this page to Markdown."
    if ctx_idx < target_idx:
        distance = target_idx - ctx_idx
        prefix = "PREVIOUS CONTEXT PAGE" if distance == 1 else f"PREVIOUS CONTEXT PAGE (-{distance})"
        return f"{prefix} — page {page_no}. Context only; do NOT transcribe this page."
    distance = ctx_idx - target_idx
    prefix = "NEXT CONTEXT PAGE" if distance == 1 else f"NEXT CONTEXT PAGE (+{distance})"
    return f"{prefix} — page {page_no}. Context only; do NOT transcribe this page."


def _build_page_prompt(target_page_no: int, context_page_nos: list[int]) -> str:
    context_text = ", ".join(str(x) for x in context_page_nos)
    return (
        f"Document page context supplied: {context_text}.\n"
        f"The TARGET page is page {target_page_no}.\n\n"
        + BASE_PAGE_PROMPT
    )


def _ensure_image(
    doc: fitz.Document,
    page_idx: int,
    image_path: Path,
    dpi: int,
    force: bool,
    rendered_this_run: set[str],
) -> bool:
    """Render once per run. Existing cached page images are reused unless --force."""
    key = str(image_path.resolve())
    if key in rendered_this_run:
        return False
    if image_path.exists() and not force:
        return False
    _render_page(doc[page_idx], image_path, dpi)
    rendered_this_run.add(key)
    return True


def prepare_pdf(
    settings: Settings,
    force: bool = False,
    limit: int | None = None,
    page_limit: int | None = None,
    context_radius: int | None = None,
) -> dict:
    """Convert PDF pages to Markdown with sliding neighboring-page visual context.

    Default context_radius=1 means that a middle page is sent to Qwen together with
    its previous and next page.  Only the target page is written to Markdown.  This
    improves continuity for technical PDFs while retaining a strict 1 page -> 1 MD
    mapping for downstream retrieval and image grounding.
    """
    settings.ensure_dirs()
    if context_radius is None:
        context_radius = settings.pdf_context_radius
    context_radius = max(0, int(context_radius))

    sources = [x for x in load_source_metadata(settings.data_dir).values() if x.source_kind == "pdf"]
    sources.sort(key=lambda x: x.document_id)
    if limit is not None:
        sources = sources[:limit]

    from rag_app.models.qwen35_vl import Qwen35VL

    model = Qwen35VL(settings.qwen_model_id)
    records: list[dict] = []
    counts = {
        "documents": 0,
        "target_pages": 0,
        "written": 0,
        "skipped_existing": 0,
        "rendered_images": 0,
        "failed": 0,
        "context_radius": context_radius,
    }

    for di, meta in enumerate(sources, 1):
        pdf_path = settings.data_dir / meta.raw_path
        if not pdf_path.exists():
            LOGGER.error("Missing PDF: %s", pdf_path)
            counts["failed"] += 1
            continue
        counts["documents"] += 1
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            counts["failed"] += 1
            LOGGER.exception("Cannot open PDF: %s", pdf_path)
            continue

        total_pages = len(doc)
        n_target_pages = total_pages if page_limit is None else min(total_pages, page_limit)
        rendered_this_run: set[str] = set()

        for page_idx in range(n_target_pages):
            page_no = page_idx + 1
            counts["target_pages"] += 1
            md_dir = settings.md_pdf_dir / meta.document_id
            img_dir = settings.page_image_dir / meta.document_id
            md_path = md_dir / f"page_{page_no:04d}.md"
            target_image_path = img_dir / f"page_{page_no:04d}.png"

            try:
                if md_path.exists() and target_image_path.exists() and not force:
                    counts["skipped_existing"] += 1
                    LOGGER.info("PDF %s/%s p%s skip existing", di, len(sources), page_no)
                    continue

                # Build a sliding visual window.  With radius=1 this is normally
                # previous + target + next.  Context may extend beyond --page-limit,
                # because page_limit limits generated MD targets, not useful context.
                ctx_indices = _context_page_indices(page_idx, total_pages, context_radius)
                image_paths: list[Path] = []
                image_labels: list[str] = []
                for ctx_idx in ctx_indices:
                    ctx_path = img_dir / f"page_{ctx_idx + 1:04d}.png"
                    if _ensure_image(
                        doc,
                        ctx_idx,
                        ctx_path,
                        settings.pdf_render_dpi,
                        force,
                        rendered_this_run,
                    ):
                        counts["rendered_images"] += 1
                    image_paths.append(ctx_path)
                    image_labels.append(_page_role_label(ctx_idx, page_idx))

                prompt = _build_page_prompt(page_no, [x + 1 for x in ctx_indices])
                md = strip_code_fence(
                    model.generate(
                        prompt,
                        image_paths=image_paths,
                        image_labels=image_labels,
                        system=SYSTEM_PROMPT,
                        max_new_tokens=settings.qwen_max_new_tokens_page,
                    )
                ).strip()
                if md == "__EMPTY__":
                    md = "<!-- no retrievable content on this page -->"

                write_md_with_front_matter(
                    md_path,
                    {
                        "document_id": meta.document_id,
                        "source_kind": "pdf",
                        "title": meta.title,
                        "source_url": meta.source_url,
                        "source_raw_path": meta.raw_path,
                        "page_number": page_no,
                        "page_count": total_pages,
                        "page_image": str(target_image_path),
                        "vlm_context_pages": ",".join(str(x + 1) for x in ctx_indices),
                        "vlm_context_radius": context_radius,
                        "language": meta.language or "",
                    },
                    md,
                )
                counts["written"] += 1
                records.append(
                    {
                        "document_id": meta.document_id,
                        "source_kind": "pdf",
                        "title": meta.title,
                        "source_url": meta.source_url,
                        "md_path": str(md_path),
                        "page_number": page_no,
                        "page_image": str(target_image_path),
                        "vlm_context_pages": [x + 1 for x in ctx_indices],
                        "status": "ready",
                    }
                )
                LOGGER.info(
                    "PDF %s/%s p%s context=%s -> %s",
                    di,
                    len(sources),
                    page_no,
                    [x + 1 for x in ctx_indices],
                    md_path.name,
                )
            except Exception as exc:
                counts["failed"] += 1
                LOGGER.exception("PDF page failed %s p%s: %s", meta.document_id, page_no, exc)
        doc.close()

    if records:
        upsert_manifest(settings.manifest_path, records)
    return counts
