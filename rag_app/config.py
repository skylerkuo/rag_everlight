from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# UBUNTU PATH CONFIGURATION
# =============================================================================
# Only DATA_DIR needs to describe the dataset layout. All dataset input/output
# paths are derived from DATA_DIR below.
#
# Expected source layout:
#   DATA_DIR/
#   ├── raw/html/
#   ├── raw/pdf/
#   ├── text/
#   └── everlight.db
#
# Edit DATA_DIR if your dataset is elsewhere.
# =============================================================================

DATA_DIR = Path("/home/skyler/Desktop/rag_system/data_photo_coupler")

# Existing crawler dataset — derived from DATA_DIR.
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
CRAWLER_TEXT_DIR = DATA_DIR / "text"
DB_PATH = DATA_DIR / "everlight.db"
RAG_READY_DOCUMENTS_PATH = DATA_DIR / "rag_ready" / "documents.jsonl"

# Generated RAG artifacts — also derived from DATA_DIR.
WORK_DIR = DATA_DIR / "rag_v6"
TXT_HTML_DIR = WORK_DIR / "txt" / "html"
MD_HTML_DIR = WORK_DIR / "md" / "html"
MD_PDF_DIR = WORK_DIR / "md" / "pdf"
PAGE_IMAGE_DIR = WORK_DIR / "page_images"
MANIFEST_PATH = WORK_DIR / "manifest.jsonl"
CHUNKS_PATH = WORK_DIR / "chunks.jsonl"
INDEX_DIR = WORK_DIR / "index"
INDEX_DENSE_PATH = INDEX_DIR / "dense.npy"
INDEX_SPARSE_PATH = INDEX_DIR / "sparse.jsonl"
INDEX_CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
INDEX_META_PATH = INDEX_DIR / "index_meta.json"

# Local Gemma 4 E4B GGUF. This is outside DATA_DIR, so keep its own path.
GEMMA_MODEL_PATH = Path(
    "/home/skyler/Desktop/rag_system/gemma-4-E4B-it-Q4_K_M.gguf"
)

# Hugging Face uses its normal/default cache behavior. No HF cache path is set
# by this project.
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
BGE_MODEL_ID = "BAAI/bge-m3"


@dataclass(slots=True)
class Settings:
    """Central runtime settings for the multi-source RAG system."""

    data_dir: Path
    raw_html_dir: Path
    raw_pdf_dir: Path
    crawler_text_dir: Path
    db_path: Path
    rag_ready_documents_path: Path

    work_dir: Path
    txt_html_dir: Path
    md_html_dir: Path
    md_pdf_dir: Path
    page_image_dir: Path
    manifest_path: Path
    chunks_path: Path
    index_dir: Path
    index_dense_path: Path
    index_sparse_path: Path
    index_chunks_path: Path
    index_meta_path: Path

    gemma_model_path: Path

    # Gemma 4 E4B GGUF: same ChatLlamaCpp style as llm_eval-main.
    gemma_n_ctx: int = 8192
    gemma_n_gpu_layers: int = -1
    gemma_temperature: float = 0.0
    gemma_max_tokens: int = 1800

    # Qwen3.5 is multimodal. The same model is temporarily used as answer LLM.
    qwen_model_id: str = QWEN_MODEL_ID
    qwen_max_new_tokens_page: int = 2200
    qwen_max_new_tokens_answer: int = 1000

    # BGE-M3 retrieval.
    bge_model_id: str = BGE_MODEL_ID
    bge_use_fp16: bool = True
    bge_batch_size: int = 12
    bge_max_length: int = 1024
    candidate_k: int = 24
    top_k: int = 7
    use_bge_pair_rerank: bool = True

    # Markdown-aware chunking.
    chunk_target_tokens: int = 450
    chunk_max_tokens: int = 650
    chunk_overlap_tokens: int = 70
    min_chunk_tokens: int = 35

    # PDF rendering / multimodal answering.
    pdf_render_dpi: int = 150
    # 1 = previous + target + next. Only the target page is written to MD.
    pdf_context_radius: int = 1
    max_answer_images: int = 7

    @classmethod
    def fixed(cls) -> "Settings":
        return cls(
            data_dir=DATA_DIR,
            raw_html_dir=RAW_HTML_DIR,
            raw_pdf_dir=RAW_PDF_DIR,
            crawler_text_dir=CRAWLER_TEXT_DIR,
            db_path=DB_PATH,
            rag_ready_documents_path=RAG_READY_DOCUMENTS_PATH,
            work_dir=WORK_DIR,
            txt_html_dir=TXT_HTML_DIR,
            md_html_dir=MD_HTML_DIR,
            md_pdf_dir=MD_PDF_DIR,
            page_image_dir=PAGE_IMAGE_DIR,
            manifest_path=MANIFEST_PATH,
            chunks_path=CHUNKS_PATH,
            index_dir=INDEX_DIR,
            index_dense_path=INDEX_DENSE_PATH,
            index_sparse_path=INDEX_SPARSE_PATH,
            index_chunks_path=INDEX_CHUNKS_PATH,
            index_meta_path=INDEX_META_PATH,
            gemma_model_path=GEMMA_MODEL_PATH,
        )

    def ensure_dirs(self) -> None:
        for p in (
            self.work_dir,
            self.txt_html_dir,
            self.md_html_dir,
            self.md_pdf_dir,
            self.page_image_dir,
            self.index_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def validate_paths(self) -> list[str]:
        """Return human-readable path problems before expensive model loading."""
        problems: list[str] = []
        if not self.data_dir.exists():
            problems.append(f"DATA_DIR does not exist: {self.data_dir}")
        if not self.raw_html_dir.exists():
            problems.append(f"Missing raw HTML directory: {self.raw_html_dir}")
        if not self.raw_pdf_dir.exists():
            problems.append(f"Missing raw PDF directory: {self.raw_pdf_dir}")
        if not self.crawler_text_dir.exists():
            problems.append(f"Missing crawler text directory: {self.crawler_text_dir}")
        if not self.db_path.exists():
            problems.append(f"Missing crawler database: {self.db_path}")
        return problems
