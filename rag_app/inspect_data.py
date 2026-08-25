from __future__ import annotations

from collections import Counter
from pathlib import Path

from rag_app.metadata import load_source_metadata


def inspect_data(data_dir: Path) -> dict:
    meta = load_source_metadata(data_dir)
    kinds = Counter(x.source_kind for x in meta.values())
    html_files = list((data_dir / "raw" / "html").glob("*.html"))
    pdf_files = list((data_dir / "raw" / "pdf").glob("*.pdf"))
    txt_files = list((data_dir / "text").glob("*.txt"))
    pdf_pages = sum(int(x.page_count or 0) for x in meta.values() if x.source_kind == "pdf")
    return {
        "data_dir": str(data_dir),
        "raw_html_files": len(html_files),
        "raw_pdf_files": len(pdf_files),
        "crawler_txt_files": len(txt_files),
        "db_html_documents": kinds.get("html", 0),
        "db_pdf_documents": kinds.get("pdf", 0),
        "db_pdf_pages": pdf_pages,
        "database": str(data_dir / "everlight.db"),
        "note": "The DB links each SHA-named raw file/TXT file to its source URL and title.",
    }
