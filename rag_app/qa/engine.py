from __future__ import annotations

from pathlib import Path

from rag_app.config import Settings
from rag_app.models.qwen35_vl import Qwen35VL
from rag_app.retrieval.bge_m3_index import BGEM3Index, SearchResult


ANSWER_SYSTEM = """You are the final answer model in a retrieval-augmented generation system.
Use only the retrieved evidence supplied in the prompt and any attached PDF page images.
Do not invent facts. If the evidence is insufficient, say that the retrieved evidence is insufficient.
Answer in the same language as the user's question.
When making factual claims, cite the supplied evidence labels such as [S1], [S2].
"""


def _context(results: list[SearchResult]) -> str:
    parts: list[str] = []
    for r in results:
        c = r.chunk
        loc = []
        if c.get("source_kind"):
            loc.append(str(c["source_kind"]).upper())
        if c.get("page_number"):
            loc.append(f"page {c['page_number']}")
        heading = " > ".join(c.get("heading_path") or [])
        if heading:
            loc.append(heading)
        parts.append(
            f"[S{r.rank}]\n"
            f"Title: {c.get('title','')}\n"
            f"Location: {' | '.join(loc)}\n"
            f"URL: {c.get('source_url','')}\n"
            f"Evidence:\n{c.get('content','')}"
        )
    return "\n\n---\n\n".join(parts)


def _images(results: list[SearchResult], max_images: int) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for r in results:
        c = r.chunk
        if c.get("source_kind") != "pdf" or not c.get("page_image"):
            continue
        p = Path(c["page_image"])
        key = str(p.resolve())
        if key in seen or not p.exists():
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_images:
            break
    return out


class RAGEngine:
    """Current answer stage uses Qwen3.5-VL as the temporary Gemini replacement."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.index = BGEM3Index(settings, load_model=True)
        self.index.load()
        self.answer_model = Qwen35VL(settings.qwen_model_id)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        results = self.index.search(question, top_k=top_k)
        images = _images(results, self.settings.max_answer_images)
        prompt = f"""USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{_context(results)}

Instructions:
- Answer the user question using the evidence above.
- If PDF page images are attached, use them to verify tables, figures, values, labels,
  or layout details that may not be fully represented in the extracted Markdown.
- Cite evidence with [S1], [S2], etc.
"""
        answer = self.answer_model.generate(
            prompt,
            image_paths=images,
            system=ANSWER_SYSTEM,
            max_new_tokens=self.settings.qwen_max_new_tokens_answer,
        )
        return {
            "question": question,
            "attached_pdf_images": [str(x) for x in images],
            "results": [x.to_dict() for x in results],
            "answer": answer,
        }
