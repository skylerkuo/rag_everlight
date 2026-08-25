# rag_everlight — System Design

This document describes the architecture, data flow, retrieval method, multimodal PDF handling, and design decisions implemented in this repository.

The system is a local multi-source RAG pipeline for technical product documents. It ingests crawler-produced HTML/TXT data and PDF files, converts both sources into structured Markdown, chunks the Markdown, builds a BGE-M3 hybrid dense+sparse index, retrieves a small evidence set, and uses Qwen3.5-4B as the final multimodal answer model. When retrieved evidence comes from a PDF, the exact source page image can also be attached to the final answer stage so the model can verify tables, figures, formulas, and layout details against the original page.

---

## 1. Design goals

The implementation is built around the following goals:

1. **Preserve source provenance.** Every retrieved chunk retains its document ID, title, source URL, source kind, Markdown path, and PDF page number/image when applicable.
2. **Treat PDF pages as hard visual boundaries.** A PDF retrieval chunk never spans multiple source pages, so a retrieved chunk can be grounded to one exact page image.
3. **Use structure before embedding.** HTML and PDF content are first normalized into Markdown, then split by Markdown headings and blocks instead of fixed character windows.
4. **Support both semantic and exact technical retrieval.** BGE-M3 dense vectors handle semantic similarity, while BGE-M3 sparse lexical weights help preserve exact model names, symbols, units, and part numbers.
5. **Keep the final answer grounded.** The answer model is instructed to use only retrieved evidence and attached PDF images, and to state when the evidence is insufficient.
6. **Keep preprocessing and online QA separable.** Expensive document conversion and indexing are performed offline; search and QA load the already-built index.

---

## 2. High-level architecture

```text
                         CRAWLER DATA
                ┌──────────────────────────┐
                │ everlight.db             │
                │ raw/html/*.html          │
                │ raw/pdf/*.pdf            │
                │ text/*.txt               │
                └────────────┬─────────────┘
                             │
               source metadata / provenance
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
          ▼                                     ▼
   HTML / crawler TXT                         PDF
          │                                     │
          │ Gemma 4 E4B                         │ PyMuPDF render
          ▼                                     ▼
   one Markdown file                  page PNG images @ 150 DPI
   per HTML source                              │
                                                │ previous/target/next
                                                │ visual context by default
                                                ▼
                                     Qwen3.5-4B multimodal
                                                │
                                                ▼
                                     one Markdown file per
                                     TARGET PDF page only
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
                   Markdown-aware chunking
              heading / paragraph / list / table
                 PDF page boundary preserved
                             │
                             ▼
                        chunks.jsonl
                             │
                             ▼
                         BGE-M3
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          dense embeddings          sparse weights
                │                         │
                └────────────┬────────────┘
                             ▼
                   Dense + Sparse ranking
                             │
                      weighted RRF fusion
                             │
                 optional BGE-M3 pair score
                             │
                             ▼
                         final Top-K
                             │
             ┌───────────────┴────────────────┐
             │                                │
             ▼                                ▼
       retrieved text                exact PDF page images
                                             max 4
             │                                │
             └───────────────┬────────────────┘
                             ▼
                    Qwen3.5-4B answer
                             │
                             ▼
                 grounded answer + [S#]
```

---

## 3. Source metadata and dataset contract

The central path is configured in `rag_app/config.py`:

```python
DATA_DIR = Path("/path/to/data_photo_coupler")
```

The expected crawler layout is:

```text
DATA_DIR/
├── raw/
│   ├── html/
│   └── pdf/
├── text/
├── everlight.db
└── rag_ready/
    └── documents.jsonl      # fallback metadata source
```

`rag_app.metadata.load_source_metadata()` first attempts to load current document metadata from `everlight.db`. It joins `document_versions` and `urls` and uses the current SHA-256 document version as `document_id`. If the database cannot provide metadata, `rag_ready/documents.jsonl` is used as a fallback.

Each source record can carry:

- `document_id`
- `source_kind` (`html` or `pdf`)
- `title`
- `source_url`
- `raw_path`
- optional `text_path`
- language
- PDF page count

This metadata is propagated through Markdown front matter, chunks, the index snapshot, retrieval results, and the final answer context.

---

## 4. HTML preprocessing

Implementation: `rag_app/preprocess/html_to_md.py`

### 4.1 Input selection

For HTML sources, the pipeline prefers the TXT file already generated by the crawler. If a crawler TXT file is not available, BeautifulSoup is used as a fallback to extract visible text from the raw HTML.

The intended mapping is:

```text
1 HTML source -> 1 TXT source -> 1 Markdown document
```

### 4.2 Gemma cleanup stage

The extracted text is passed to the local Gemma 4 E4B GGUF through `ChatLlamaCpp`.

Gemma is instructed to:

- preserve technical facts, product/model numbers, values, units, features, and applications;
- preserve the source language;
- remove navigation and obvious boilerplate;
- organize content with Markdown headings, paragraphs, lists, and tables;
- avoid invention or outside knowledge.

The output is written to:

```text
rag_v6/md/html/<document_id>.md
```

This 1:1 source-to-Markdown mapping keeps the original URL/document identity stable while allowing chunking to change independently later.

---

## 5. PDF preprocessing and visual grounding

Implementation: `rag_app/preprocess/pdf_to_md.py`

PDF handling is deliberately multimodal because the corpus contains technical tables, figures, circuit diagrams, formulas, and layouts that are not always faithfully represented by plain PDF text extraction.

### 5.1 Rendering

PyMuPDF renders each PDF page to PNG using the configured DPI:

```python
pdf_render_dpi = 150
```

Images are stored under:

```text
rag_v6/page_images/<document_id>/page_0001.png
rag_v6/page_images/<document_id>/page_0002.png
...
```

### 5.2 Sliding visual context

The default context radius is:

```python
pdf_context_radius = 1
```

For a middle target page `p`, Qwen normally receives:

```text
page p-1  -> PREVIOUS CONTEXT PAGE
page p    -> TARGET PAGE
page p+1  -> NEXT CONTEXT PAGE
```

The neighboring pages are context only. Qwen is explicitly instructed to output Markdown for the target page only.

This strategy is used because technical documents commonly have:

- continued table headers;
- sections that begin on the preceding page;
- captions separated from figures;
- tables or lists continued across page boundaries.

The context pages help Qwen understand continuity without breaking the downstream invariant:

```text
1 source PDF page -> 1 Markdown file -> 1 exact page image
```

### 5.3 Target-page-only Markdown

For each target page, Qwen is instructed to preserve:

- visible text;
- headings;
- model and part numbers;
- formulas and symbols;
- numerical values and units;
- Markdown tables when the page clearly contains tabular data;
- concise visible figure descriptions for diagrams/figures.

It must not copy factual rows or paragraphs that exist only on neighboring context pages.

Output:

```text
rag_v6/md/pdf/<document_id>/page_0001.md
rag_v6/md/pdf/<document_id>/page_0002.md
...
```

The Markdown front matter records the target page number, target page image, original document metadata, and the context pages supplied to the VLM.

---

## 6. Markdown chunking

Implementation: `rag_app/chunking/markdown_chunker.py`

Only generated Markdown is chunked. Raw HTML/PDF files are never embedded directly.

### 6.1 Heading-aware sections

The chunker first parses ATX Markdown headings (`#` through `######`) and maintains a hierarchical `heading_path`.

Chunks are never merged across heading sections.

### 6.2 Block-aware splitting

Within a section, blank lines define the primary blocks. The splitter tries to preserve:

- paragraphs;
- lists;
- Markdown tables;
- line/sentence boundaries.

Large Markdown tables are sliced by rows while repeating the table header/separator in each slice.

### 6.3 Token limits

Default settings:

```text
Target chunk size : ~450 tokens
Hard maximum      : ~650 tokens
Overlap           : ~70 tokens
Minimum           : ~35 tokens
```

Production chunking uses the BGE-M3 Hugging Face tokenizer. `--approx-tokenizer` exists only as an offline/debug fallback.

### 6.4 PDF page boundary

PDF Markdown is already one file per source page, and `_iter_md_files()` processes each page independently. Therefore a chunk cannot cross from one PDF page to another.

This is important because the QA stage can attach the exact source page image for any retrieved PDF chunk.

### 6.5 `content` vs `embedding_text`

Each chunk stores both:

- `content`: evidence shown to the answer model;
- `embedding_text`: content enriched with title, heading path, and PDF page number.

Example:

```text
Title: EL3120 IGBT Gate Drive Optocoupler
Section: Driver Power Dissipation
Page: 6

PInternal = ...
```

Metadata enrichment improves retrieval without polluting the evidence text shown to the final answer model.

The resulting chunk set is written to:

```text
rag_v6/chunks.jsonl
```

---

## 7. BGE-M3 hybrid index

Implementation: `rag_app/retrieval/bge_m3_index.py`

The retrieval model is:

```text
BAAI/bge-m3
```

For every `embedding_text`, the index stores:

1. a normalized dense embedding;
2. BGE-M3 sparse lexical weights.

Generated files:

```text
rag_v6/index/
├── dense.npy
├── sparse.jsonl
├── chunks.jsonl
└── index_meta.json
```

### 7.1 Candidate generation

Default configuration:

```python
candidate_k = 24
top_k = 5
```

For a query, BGE-M3 produces both a dense vector and sparse lexical weights.

The index independently ranks chunks by:

- dense cosine-equivalent dot product on normalized vectors;
- sparse lexical dot product.

The candidate pool size is:

```python
candidate_k = min(max(config.candidate_k, requested_top_k), number_of_chunks)
```

Therefore the default plain search creates up to 24 candidates before the final Top-5 selection.

### 7.2 RRF fusion

Dense and sparse rank lists are combined with weighted Reciprocal Rank Fusion:

```text
Dense weight  = 0.55
Sparse weight = 0.45
RRF k0        = 60
```

RRF is used because dense and sparse raw score scales are not directly comparable.

### 7.3 BGE-M3 pair scoring

When enabled:

```python
use_bge_pair_rerank = True
```

The fused candidates are passed through `BGEM3FlagModel.compute_score()` using BGE-M3's combined modes:

```text
weights_for_different_modes = [0.4, 0.2, 0.4]
```

If the combined score is available, candidates are sorted by that score before the final Top-K is returned. If pair scoring fails, the implementation falls back to the hybrid RRF order.

This is an in-model BGE-M3 pair scoring stage, not a separately loaded cross-encoder reranker.

---

## 8. Online QA path (`rag.py ask`)

Implementation: `rag_app/qa/engine.py`

The standard QA path is intentionally simple:

```text
original question
      ↓
BGE-M3 hybrid search
      ↓
final Top-K chunks (default 5)
      ↓
text evidence [S1]...[S5]
      +
up to 4 unique retrieved PDF page images
      ↓
Qwen3.5-4B multimodal answer
```

### 8.1 Text evidence

Every returned `SearchResult` is converted to evidence containing:

- evidence label `[S#]`;
- title;
- source kind;
- page number when applicable;
- heading path;
- source URL;
- chunk content.

All final retrieval results are included in the answer prompt.

### 8.2 PDF images

For PDF results, `_images()` walks the final ranked results and attaches the corresponding `page_image`.

Images are:

- deduplicated by file path;
- limited by `max_answer_images` (default `4`);
- attached in retrieval rank order.

The Qwen wrapper scales images to 50% before inference.

### 8.3 Grounded answer policy

The answer-system prompt requires Qwen to:

- use only retrieved evidence and attached images;
- not invent facts;
- state when evidence is insufficient;
- answer in the language of the user question;
- cite factual claims with evidence labels such as `[S1]`, `[S2]`.

---

## 9. Entity-aware retrieval used by `rag_loop.py` and `rag_ans.py`

The repository also contains a second online retrieval path used for batch evaluation and the interactive terminal.

This path adds a query-analysis step before BGE-M3 retrieval.

```text
original question
      ↓
Qwen3.5 query analyzer
      ↓
keywords + proper_nouns
      ↓
expanded retrieval query
      ↓
BGE-M3 broad candidate retrieval
      ↓
Exact Product Filter when a product/model was detected
      ↓
final Top-K (default 5)
      ↓
Qwen3.5 grounded multimodal answer
```

### 9.1 Query analysis

The already-loaded Qwen answer model extracts two arrays:

```json
{
  "keywords": [],
  "proper_nouns": []
}
```

`proper_nouns` are intended for exact product/model/series names explicitly present in the user question.

### 9.2 Expanded retrieval query

The original question is preserved and augmented with extracted terms:

```text
<original question>
Search keywords: ...
Exact product/model names: ...
```

Product names are repeated to strengthen exact lexical matching.

### 9.3 Broader candidate pool

The entity-aware flow requests at least 30 retrieval results:

```python
candidate_top_k = max(final_top_k, 30)
```

The BGE-M3 index therefore searches/reranks a broader pool before entity filtering.

### 9.4 Exact Product Filter

If exact product/model names were detected, retrieved chunks are retained only when they contain at least one exact literal product name with alphanumeric boundaries.

Example behavior:

```text
EL827  matches EL827
EL827  does not match EL8270
```

If no candidate contains the extracted name, the system falls back to the unfiltered candidate list instead of returning no evidence.

This filter is useful for part-number-heavy product corpora but is only present in `rag_loop.py` / `rag_ans.py`; the basic `rag.py ask` path does not use it.

---

## 10. Interactive and evaluation entry points

### `rag.py`

Core pipeline CLI:

- inspect paths/data;
- preprocess HTML/PDF;
- chunk;
- build index;
- plain hybrid search;
- plain Top-K RAG answer.

### `rag_loop.py`

Batch evaluation runner:

- reads JSONL questions;
- uses entity-aware retrieval;
- writes one JSONL output record per question immediately;
- supports `--resume`;
- can optionally save full retrieval results.

### `rag_ans.py`

Stateless terminal QA:

- every question is independent;
- entity-aware retrieval is enabled;
- one retrieval pass per question;
- no conversation memory;
- no iterative evidence-sufficiency or query-rewrite loop;
- captures a generated-token probability from the final Qwen generation.

The generated-token probability is a raw generation confidence statistic; it is not a calibrated factual-correctness probability.

---

## 11. Generated artifact layout

All RAG-generated artifacts are derived from `DATA_DIR` and stored under `DATA_DIR/rag_v6/`:

```text
rag_v6/
├── txt/
│   └── html/                    # fallback extracted HTML text
├── md/
│   ├── html/                    # one MD per HTML source
│   └── pdf/
│       └── <document_id>/
│           ├── page_0001.md
│           └── ...
├── page_images/
│   └── <document_id>/
│       ├── page_0001.png
│       └── ...
├── manifest.jsonl
├── chunks.jsonl
└── index/
    ├── dense.npy
    ├── sparse.jsonl
    ├── chunks.jsonl
    └── index_meta.json
```

---

## 12. Key runtime settings

Current defaults in `rag_app/config.py`:

| Setting | Default | Purpose |
|---|---:|---|
| `gemma_n_ctx` | 8192 | HTML cleanup context |
| `gemma_temperature` | 0.0 | deterministic preprocessing |
| `qwen_max_new_tokens_page` | 2200 | PDF page-to-Markdown budget |
| `qwen_max_new_tokens_answer` | 1000 | final QA budget |
| `bge_batch_size` | 12 | BGE-M3 indexing batch |
| `bge_max_length` | 1024 | BGE-M3 max sequence length |
| `candidate_k` | 24 | plain hybrid candidate pool |
| `top_k` | 5 | default final retrieval results |
| `use_bge_pair_rerank` | `True` | BGE-M3 pair scoring |
| `chunk_target_tokens` | 450 | chunk target |
| `chunk_max_tokens` | 650 | hard chunk maximum |
| `chunk_overlap_tokens` | 70 | overlap inside one heading |
| `pdf_render_dpi` | 150 | PDF page rendering |
| `pdf_context_radius` | 1 | previous/target/next PDF context |
| `max_answer_images` | 4 | maximum PDF page images per answer |

---

## 13. Important invariants

The current implementation relies on several invariants:

### PDF page provenance

```text
one target page -> one MD file -> zero or more chunks -> one exact page image
```

A PDF chunk must not span pages.

### Evidence labels

Ranks assigned by retrieval are also used as final answer labels:

```text
rank 1 -> [S1]
rank 2 -> [S2]
...
```

### No raw-document answering

The final answer model does not search raw PDFs or HTML itself. It sees only:

1. the final retrieved Markdown chunks;
2. the PDF page images corresponding to retrieved PDF chunks.

---

## 14. Current limitations

These are implementation limitations of the current repository, not planned features:

1. **No QA-stage neighbor expansion.** Retrieving a chunk from page 7 does not automatically attach page 6 or page 8 unless those pages are also in the final results.
2. **No document/product metadata pre-filter in the core index.** Search is performed across the indexed corpus; the entity-aware path only applies an exact-name filter after broad retrieval.
3. **No explicit Chinese/English synonym expansion.** The query analyzer extracts terms already present in the question but is instructed not to invent or normalize names.
4. **No dedicated formula canonicalization layer.** Mathematical Unicode and PDF/VLM transcription are indexed as generated in the Markdown.
5. **No iterative evidence-sufficiency loop.** The terminal explicitly performs one retrieval pass per question.
6. **The basic `rag.py ask` path is not entity-aware.** Query analysis and Exact Product Filter are implemented in `rag_loop.py` and `rag_ans.py`.
7. **PDF image attachment is limited to the final ranked results and `max_answer_images`.** A relevant page outside final Top-K cannot be inspected by the answer VLM.

These limitations are especially important when evaluating retrieval failures separately from answer-model failures.

---

## 15. Recommended evaluation methodology

Final answer accuracy alone cannot tell whether an error came from retrieval or generation. For technical-document RAG, evaluate the stages separately.

Recommended metrics:

```text
Document Recall@K
Page Recall@K
Chunk Recall@K
MRR
Final QA Accuracy
```

For PDF questions, record the ground-truth answer page and compare it with the returned `page_number` values. A useful diagnostic record is:

```json
{
  "question": "...",
  "ground_truth_page": 6,
  "retrieved_pages": [7, 5, 8, 1, 10],
  "attached_image_pages": [7, 5, 8, 1],
  "ground_truth_page_in_top_k": false,
  "answer_correct": false
}
```

This separates three failure modes:

```text
1. correct evidence never retrieved       -> retrieval failure
2. correct text retrieved but image absent -> visual-evidence selection issue
3. correct evidence supplied but answer wrong -> answer/grounding failure
```

---

## 16. Possible future improvements

The following are sensible extensions but are **not implemented in the current code**:

- formula/Unicode normalization before indexing;
- Chinese/English technical synonym expansion;
- product/document metadata filtering before vector search;
- neighboring chunk/page expansion after retrieval;
- a dedicated answerability/cross-encoder reranker;
- separate table/formula/figure chunk types;
- retrieval diagnostics that automatically compute Page Recall@K;
- evidence-sufficiency checking followed by a controlled second retrieval pass.

The existing design intentionally keeps source preprocessing, chunking, retrieval, and answer generation modular so these components can be introduced independently.
