from __future__ import annotations

import argparse
import json

from rag_app.config import Settings
from rag_app.inspect_data import inspect_data
from rag_app.utils import release_accelerator_memory, setup_logging


def get_settings() -> Settings:
    return Settings.fixed()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-source PDF/HTML RAG pipeline (Ubuntu DATA_DIR-derived edition)"
    )
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("paths", help="Print DATA_DIR and all derived dataset/RAG paths")
    sub.add_parser("inspect", help="Inspect configured source data")

    ph = sub.add_parser("prepare-html", help="HTML/TXT -> one MD per HTML using Gemma 4 E4B")
    ph.add_argument("--force", action="store_true")
    ph.add_argument("--limit", type=int)

    pp = sub.add_parser("prepare-pdf", help="PDF pages -> page images + one MD per page using Qwen3.5-VL")
    pp.add_argument("--force", action="store_true")
    pp.add_argument("--limit", type=int, help="Limit number of PDFs for testing")
    pp.add_argument("--page-limit", type=int, help="Limit target pages per PDF for testing")
    pp.add_argument("--context-radius", type=int, help="Neighboring pages per side; default 1")

    prep = sub.add_parser("prepare", help="Run HTML then PDF preprocessing")
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--html-limit", type=int)
    prep.add_argument("--pdf-limit", type=int)
    prep.add_argument("--page-limit", type=int)
    prep.add_argument("--context-radius", type=int)

    pc = sub.add_parser("chunk", help="Split only from generated MD files")
    pc.add_argument("--approx-tokenizer", action="store_true", help="Debug only; avoid HF tokenizer download")

    sub.add_parser("index", help="Build BGE-M3 dense+sparse index")

    pb = sub.add_parser("build", help="Prepare HTML + PDF, chunk MD, build BGE-M3 index")
    pb.add_argument("--force", action="store_true")
    pb.add_argument("--html-limit", type=int)
    pb.add_argument("--pdf-limit", type=int)
    pb.add_argument("--page-limit", type=int)
    pb.add_argument("--context-radius", type=int)

    ps = sub.add_parser("search")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int)

    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--top-k", type=int)

    args = p.parse_args()
    setup_logging(args.verbose)
    s = get_settings()

    if args.command == "paths":
        print(json.dumps({
            "data_dir": str(s.data_dir),
            "raw_html_dir": str(s.raw_html_dir),
            "raw_pdf_dir": str(s.raw_pdf_dir),
            "crawler_text_dir": str(s.crawler_text_dir),
            "db_path": str(s.db_path),
            "rag_ready_documents_path": str(s.rag_ready_documents_path),
            "work_dir": str(s.work_dir),
            "txt_html_dir": str(s.txt_html_dir),
            "md_html_dir": str(s.md_html_dir),
            "md_pdf_dir": str(s.md_pdf_dir),
            "page_image_dir": str(s.page_image_dir),
            "manifest_path": str(s.manifest_path),
            "chunks_path": str(s.chunks_path),
            "index_dir": str(s.index_dir),
            "index_dense_path": str(s.index_dense_path),
            "index_sparse_path": str(s.index_sparse_path),
            "index_chunks_path": str(s.index_chunks_path),
            "index_meta_path": str(s.index_meta_path),
            "gemma_model_path": str(s.gemma_model_path),
            "huggingface_cache": "default Hugging Face cache behavior",
        }, ensure_ascii=False, indent=2))
        return

    path_problems = s.validate_paths()
    if path_problems:
        raise SystemExit(
            "Configured dataset paths are invalid. Edit DATA_DIR in rag_app/config.py first:\n- "
            + "\n- ".join(path_problems)
        )

    s.ensure_dirs()

    if args.command == "inspect":
        print(json.dumps(inspect_data(s.data_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-html":
        if not s.gemma_model_path.exists():
            raise SystemExit(f"Gemma GGUF not found: {s.gemma_model_path}\nEdit GEMMA_MODEL_PATH in rag_app/config.py")
        from rag_app.preprocess.html_to_md import prepare_html
        print(json.dumps(prepare_html(s, args.force, args.limit), ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-pdf":
        from rag_app.preprocess.pdf_to_md import prepare_pdf
        print(json.dumps(prepare_pdf(s, args.force, args.limit, args.page_limit, args.context_radius), ensure_ascii=False, indent=2))
        return

    if args.command == "prepare":
        if not s.gemma_model_path.exists():
            raise SystemExit(f"Gemma GGUF not found: {s.gemma_model_path}\nEdit GEMMA_MODEL_PATH in rag_app/config.py")
        from rag_app.preprocess.html_to_md import prepare_html
        from rag_app.preprocess.pdf_to_md import prepare_pdf
        a = prepare_html(s, args.force, args.html_limit)
        release_accelerator_memory()
        b = prepare_pdf(s, args.force, args.pdf_limit, args.page_limit, args.context_radius)
        print(json.dumps({"html": a, "pdf": b}, ensure_ascii=False, indent=2))
        return

    if args.command == "chunk":
        from rag_app.chunking.markdown_chunker import build_chunks
        print(json.dumps(build_chunks(s, use_hf_tokenizer=not args.approx_tokenizer), ensure_ascii=False, indent=2))
        return

    if args.command == "index":
        from rag_app.retrieval.bge_m3_index import BGEM3Index
        idx = BGEM3Index(s, load_model=True)
        print(json.dumps(idx.build(), ensure_ascii=False, indent=2))
        return

    if args.command == "build":
        if not s.gemma_model_path.exists():
            raise SystemExit(f"Gemma GGUF not found: {s.gemma_model_path}\nEdit GEMMA_MODEL_PATH in rag_app/config.py")
        from rag_app.preprocess.html_to_md import prepare_html
        from rag_app.preprocess.pdf_to_md import prepare_pdf
        from rag_app.chunking.markdown_chunker import build_chunks
        from rag_app.retrieval.bge_m3_index import BGEM3Index
        out = {}
        out["html"] = prepare_html(s, args.force, args.html_limit)
        release_accelerator_memory()
        out["pdf"] = prepare_pdf(s, args.force, args.pdf_limit, args.page_limit, args.context_radius)
        release_accelerator_memory()
        out["chunk"] = build_chunks(s, use_hf_tokenizer=True)
        out["index"] = BGEM3Index(s, load_model=True).build()
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if args.command == "search":
        from rag_app.retrieval.bge_m3_index import BGEM3Index
        idx = BGEM3Index(s, load_model=True)
        idx.load()
        results = idx.search(args.query, top_k=args.top_k)
        print(json.dumps([x.to_dict() for x in results], ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        from rag_app.qa.engine import RAGEngine
        engine = RAGEngine(s)
        print(json.dumps(engine.ask(args.question, top_k=args.top_k), ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
