# Multi-Source Document RAG System — Ubuntu DATA_DIR-Derived Edition (v6)

The path configuration is intentionally simple:

```python
DATA_DIR = Path("/home/gai/Desktop/data_photo_coupler")
```

All crawler input paths and all generated RAG paths are derived from `DATA_DIR`.
There is no `--data-dir` argument and this project does **not** set a custom Hugging Face cache directory.

The local Gemma GGUF remains a separate path because it is outside the dataset directory.

## Derived dataset paths

With:

```python
DATA_DIR = Path("/home/gai/Desktop/data_photo_coupler")
```

the project automatically uses:

```text
/home/gai/Desktop/data_photo_coupler/raw/html
/home/gai/Desktop/data_photo_coupler/raw/pdf
/home/gai/Desktop/data_photo_coupler/text
/home/gai/Desktop/data_photo_coupler/everlight.db
```

Generated files automatically go under:

```text
/home/gai/Desktop/data_photo_coupler/rag_v6/
```

The output includes:

```text
rag_v6/
├── txt/html/
├── md/html/
├── md/pdf/
├── page_images/
├── manifest.jsonl
├── chunks.jsonl
└── index/
```

## Pipeline

```text
HTML -> crawler TXT -> Gemma 4 E4B -> one MD per HTML -> chunks

PDF -> previous/target/next page images -> Qwen3.5-VL-4B
    -> one MD for TARGET page only
    -> preserve target-page image
    -> chunks that never cross page boundaries

chunks -> BAAI/bge-m3 retrieval -> Top-K
      -> temporary Qwen3.5-VL answer model
      -> if PDF evidence is retrieved, attach the exact corresponding page image
```

## PDF context strategy

Default:

```text
Previous page = context only
Target page   = output Markdown
Next page     = context only
```

This gives Qwen cross-page context for table headers, continued sections, captions, and related material while preserving exact page provenance.

## Chunking

Only generated Markdown is chunked.

- HTML: one HTML -> one MD -> Markdown heading-aware chunks
- PDF: one target page -> one MD -> chunks never cross page boundaries
- target: ~450 tokens
- hard maximum: ~650 tokens
- overlap: ~70 tokens within a section

## Basic commands

```bash
python rag.py paths
python rag.py inspect
python rag.py prepare-html --limit 1
python rag.py prepare-pdf --limit 1 --page-limit 3 --context-radius 1
python rag.py chunk
python rag.py index
python rag.py search "What is the isolation voltage?"
python rag.py ask "What is the isolation voltage?"
```

Full build:

```bash
python rag.py build --context-radius 1
```

See `UBUNTU_QUICKSTART.md` for the complete setup sequence.
