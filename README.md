# rag_everlight

A local multi-source Retrieval-Augmented Generation (RAG) system for Everlight technical documents.

The repository ingests crawler-produced HTML/TXT and PDF data, converts both sources into structured Markdown, creates heading-aware chunks, builds a BGE-M3 hybrid dense+sparse index, retrieves evidence, and uses Qwen3.5-4B as the final multimodal answer model.

For PDF evidence, the exact retrieved source page image can also be sent to Qwen so tables, figures, formulas, values, and layout can be checked against the original page.

The project currently provides three QA/evaluation paths:

- `rag.py ask`: basic hybrid RAG
- `rag_loop.py` / `rag_ans.py`: entity-aware retrieval with keyword and exact product extraction
- `rag_loop_v2.py`: entity-aware retrieval plus a dedicated `BAAI/bge-reranker-v2-m3` second-stage reranker

For implementation details, see [`DESIGN.md`](DESIGN.md).

---

## 1. Architecture

### 1.1 Offline document processing

```text
HTML / crawler TXT
        │
        ▼
   Gemma 4 E4B
        │
        ▼
     Markdown
        │
        ├──────────────────────────────┐
        │                              │
PDF     │                              │
 │      │                              │
 ▼      │                              │
Page images                           │
 │                                     │
 ▼                                     │
Qwen3.5-4B                             │
 │                                     │
 ▼                                     │
One Markdown file per PDF page         │
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Markdown-aware chunks
                       │
                       ▼
                 BGE-M3 index
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          Dense index       Sparse index
              │                 │
              └────────┬────────┘
                       ▼
                Weighted RRF
```

### 1.2 Online QA

Basic path:

```text
User Question
      │
      ▼
BGE-M3 Hybrid Retrieval
      │
      ▼
Final Top-K
      │
      ├── Text evidence
      └── PDF page images when applicable
      │
      ▼
Qwen3.5-4B
      │
      ▼
Grounded Answer + [S#]
```

Entity-aware path:

```text
User Question
      │
      ▼
Qwen3.5 Query Analysis
      │
      ├── Keywords
      └── Proper nouns / product names
      │
      ▼
Expanded Retrieval Query
      │
      ▼
BGE-M3 Broad Retrieval
      │
      ▼
Exact Product Filter
      │
      ▼
Final Top-K
      │
      ▼
Qwen3.5-4B
```

Reranker path (`rag_loop_v2.py`):

```text
User Question
      │
      ▼
Qwen3.5 Query Analysis
      │
      ▼
Expanded Retrieval Query
      │
      ▼
BGE-M3 Broad Retrieval
      │
      ▼
Exact Product Filter
      │
      ▼
BAAI/bge-reranker-v2-m3
      │
      ▼
Final Top-K
      │
      ├── Text evidence
      └── PDF page images when applicable
      │
      ▼
Qwen3.5-4B
      │
      ▼
Grounded Answer + [S#]
```

---

## 2. Repository structure

```text
rag_everlight/
├── rag.py                         # build / search / basic ask CLI
├── rag_ans.py                     # interactive entity-aware QA
├── rag_loop.py                    # batch entity-aware evaluation baseline
├── rag_loop_v2.py                 # batch evaluation with dedicated reranker
├── rag_app/
│   ├── config.py                  # paths and runtime settings
│   ├── metadata.py                # crawler DB / metadata loading
│   ├── inspect_data.py
│   ├── preprocess/
│   │   ├── html_to_md.py          # HTML/TXT -> Gemma -> Markdown
│   │   └── pdf_to_md.py           # PDF pages -> Qwen -> Markdown
│   ├── chunking/
│   │   └── markdown_chunker.py    # heading-aware chunking
│   ├── retrieval/
│   │   ├── bge_m3_index.py        # BGE-M3 dense+sparse retrieval
│   │   └── reranker.py            # bge-reranker-v2-m3 reranking
│   ├── qa/
│   │   └── engine.py              # evidence -> Qwen answer
│   └── models/
│       ├── gemma4_llamacpp.py
│       └── qwen35_vl.py
├── tests/
├── requirements.txt
├── questions.jsonl
├── DESIGN.md
└── README.md
```

Large datasets and local GGUF model files should not be committed to Git. The repository `.gitignore` should exclude paths such as `data_photo*` and files such as `*.gguf`.

---

## 3. Required external data

The project expects an existing crawler dataset configured through `DATA_DIR` in `rag_app/config.py`.

Expected structure:

```text
DATA_DIR/
├── raw/
│   ├── html/
│   │   └── *.html
│   └── pdf/
│       └── *.pdf
├── text/
│   └── *.txt
├── everlight.db
└── rag_ready/
    └── documents.jsonl
```

`everlight.db` is used to map document IDs to metadata such as:

- source URL
- title
- raw path
- TXT path
- language
- PDF page count

Generated RAG files are stored under:

```text
DATA_DIR/rag_v6/
```

---

## 4. Models

| Stage | Model |
|---|---|
| HTML/TXT cleanup | local Gemma 4 E4B GGUF through `ChatLlamaCpp` |
| PDF page understanding | `Qwen/Qwen3.5-4B` |
| Query keyword/product extraction | `Qwen/Qwen3.5-4B` |
| Retrieval | `BAAI/bge-m3` |
| Dedicated reranking in `rag_loop_v2.py` | `BAAI/bge-reranker-v2-m3` |
| Final answer | `Qwen/Qwen3.5-4B` |

The Gemma GGUF is provided locally.

Qwen, BGE-M3, and the dedicated reranker use the normal Hugging Face cache and are downloaded automatically when first loaded if they are not already cached.

The dedicated reranker requires:

```bash
pip install -U FlagEmbedding
```

---

## 5. Environment setup

Install the project dependencies from:

```bash
pip install -r requirements.txt
```

If the reranker dependency is not already included in `requirements.txt`, install it separately:

```bash
pip install -U FlagEmbedding
```

The Gemma path and dataset path must be configured before running the pipeline.

---

## 6. Configure paths

Open:

```text
rag_app/config.py
```

Set the dataset root:

```python
DATA_DIR = Path("/your/path/data_photo_coupler")
```

Set the local Gemma GGUF path:

```python
GEMMA_MODEL_PATH = Path(
    "/your/path/gemma-4-E4B-it-Q4_K_M.gguf"
)
```

The current model IDs are also defined in `config.py`:

```python
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
BGE_MODEL_ID = "BAAI/bge-m3"
```

All generated paths are derived from `DATA_DIR`.

---

## 7. Verify the configuration

Print resolved paths:

```bash
python rag.py paths
```

Inspect the crawler dataset:

```bash
python rag.py inspect
```

The following paths should exist before preprocessing:

```text
DATA_DIR
DATA_DIR/raw/html
DATA_DIR/raw/pdf
DATA_DIR/text
DATA_DIR/everlight.db
```

---

## 8. Smoke test

Before processing the complete corpus, test one HTML source and a few PDF pages.

### HTML

```bash
python rag.py prepare-html --limit 1
```

### PDF

```bash
python rag.py prepare-pdf \
  --limit 1 \
  --page-limit 3 \
  --context-radius 1
```

With:

```text
--context-radius 1
```

a middle PDF page is processed with:

```text
previous page + target page + next page
```

Only the target page is written to Markdown.

Check the generated files under:

```text
DATA_DIR/rag_v6/md/
DATA_DIR/rag_v6/page_images/
```

---

## 9. Build the complete RAG pipeline

The stages can be run separately.

### Step 1 — preprocess HTML/TXT

```bash
python rag.py prepare-html
```

Output:

```text
DATA_DIR/rag_v6/md/html/
```

### Step 2 — preprocess PDFs

```bash
python rag.py prepare-pdf --context-radius 1
```

Output:

```text
DATA_DIR/rag_v6/md/pdf/
DATA_DIR/rag_v6/page_images/
```

### Step 3 — chunk Markdown

```bash
python rag.py chunk
```

Output:

```text
DATA_DIR/rag_v6/chunks.jsonl
```

Production chunking uses the BGE-M3 tokenizer.

For lightweight debugging:

```bash
python rag.py chunk --approx-tokenizer
```

### Step 4 — build the BGE-M3 index

```bash
python rag.py index
```

Output:

```text
DATA_DIR/rag_v6/index/
├── dense.npy
├── sparse.jsonl
├── chunks.jsonl
└── index_meta.json
```

### One-command build

```bash
python rag.py build --context-radius 1
```

The build flow is:

```text
HTML preprocessing
-> release accelerator memory
-> PDF preprocessing
-> release accelerator memory
-> chunking
-> BGE-M3 indexing
```

Regenerate existing outputs with:

```bash
python rag.py build --force --context-radius 1
```

---

## 10. Search without answer generation

Use the search command to inspect retrieval results directly:

```bash
python rag.py search "EL3120 PInternal formula"
```

Override Top-K:

```bash
python rag.py search "EL3120 PInternal formula" --top-k 10
```

Each result contains retrieval scores and provenance such as:

- document ID
- title
- source kind
- page number
- heading path
- source URL
- PDF page image path

This is the first command to use when diagnosing whether an incorrect answer is caused by retrieval or by the final answer model.

---

## 11. Basic RAG QA

Ask a question:

```bash
python rag.py ask "EL3120 的 PInternal 如何計算？"
```

Override the final retrieval size:

```bash
python rag.py ask "EL3120 的 PInternal 如何計算？" --top-k 10
```

The basic path is:

```text
Question
-> BGE-M3 hybrid retrieval
-> final Top-K chunks
-> PDF page images when applicable
-> Qwen3.5-4B
-> grounded answer
```

Typical defaults:

```text
BGE candidate_k       = 24
final top_k           = 5
max PDF answer images = 4
```

The final answer model is instructed to:

- use only retrieved evidence and attached PDF page images;
- avoid unsupported facts;
- state when evidence is insufficient;
- answer in the user's language;
- cite evidence using `[S1]`, `[S2]`, etc.

---

## 12. Interactive entity-aware QA

Run:

```bash
python rag_ans.py
```

Optional Top-K:

```bash
python rag_ans.py --top-k 5
```

Before retrieval, Qwen extracts:

```json
{
  "keywords": [],
  "proper_nouns": []
}
```

The flow is:

1. keep the original question;
2. extract keywords and exact product/model names;
3. build an expanded retrieval query;
4. retrieve a broader candidate pool with BGE-M3;
5. apply an Exact Product Filter when possible;
6. keep the final Top-K;
7. attach PDF page images when applicable;
8. generate a grounded Qwen answer.

The interactive terminal is stateless. It does not currently maintain conversation memory or run an iterative evidence-sufficiency/query-rewrite loop.

---

## 13. Batch evaluation

Both `rag_loop.py` and `rag_loop_v2.py` read one JSON object per line.

Minimum input:

```json
{"question": "EL3120 的最大驅動電流是多少？"}
```

If an `answer` field is present, it is copied into the output as `ground_truth`.

### 13.1 Baseline entity-aware evaluation

```bash
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl
```

Useful options:

```bash
# Test only the first 10 questions
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --limit 10

# Resume an interrupted run
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --resume

# Save retrieval results
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --save-results
```

### 13.2 Evaluation with dedicated reranker

`rag_loop_v2.py` adds a dedicated `BAAI/bge-reranker-v2-m3` stage after BGE-M3 retrieval and Exact Product Filter.

Recommended run:

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --candidate-top-k 30 \
  --top-k 5 \
  --save-results
```

Useful options:

```bash
# Test only the first 10 questions
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --limit 10

# Resume an interrupted run
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --resume

# Disable dedicated reranking for an A/B baseline
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2_no_rerank.jsonl \
  --disable-reranker \
  --save-results
```

The v2 output can preserve:

- BGE-M3 rank before dedicated reranking;
- dedicated reranker score;
- final reranker rank;
- retrieval time;
- reranking time.

This makes it possible to distinguish:

```text
correct evidence missing from candidate pool
-> retrieval recall failure

correct evidence present but ranked too low
-> ranking failure

correct evidence supplied but answer still wrong
-> answer / grounding failure
```

---

## 14. Retrieval paths

| Entry point | Query analysis | Exact Product Filter | Dedicated reranker | Default final Top-K | Main use |
|---|---|---|---|---:|---|
| `rag.py ask` | No | No | No | 5 | basic QA |
| `rag_ans.py` | Yes | Yes | No | 5 | interactive product QA |
| `rag_loop.py` | Yes | Yes | No | 5 | baseline batch evaluation |
| `rag_loop_v2.py` | Yes | Yes | Yes | 5 | batch evaluation with reranking |

The entity-aware paths request a broader candidate set before filtering.

Typical logic:

```python
candidate_top_k = max(final_top_k, 30)
```

If no candidate contains an extracted exact product/model name, the system falls back to the unfiltered candidate list instead of returning an empty result set.

In `rag_loop_v2.py`, the surviving candidates are then reranked before selecting the final Top-K.

---

## 15. Retrieval method

### 15.1 BGE-M3 hybrid retrieval

The index uses BGE-M3 for both:

```text
Dense retrieval   -> semantic similarity
Sparse retrieval  -> exact / lexical matching
```

The two ranked lists are fused by weighted Reciprocal Rank Fusion.

Current weights:

```text
Dense RRF weight  = 0.55
Sparse RRF weight = 0.45
RRF k0            = 60
```

This allows semantic matching while preserving technical tokens such as:

```text
EL3120
EL827
CTR
Vrms
PInternal
Rg
VEE
```

### 15.2 BGE-M3 internal pair scoring

The core BGE-M3 index can also attempt its own BGE-M3 pair scoring before returning the final results.

This is part of the BGE-M3 retrieval implementation.

### 15.3 Dedicated reranker

`rag_loop_v2.py` additionally loads:

```text
BAAI/bge-reranker-v2-m3
```

This is a separate second-stage cross-encoder reranker.

Its role is different from BGE-M3 retrieval:

```text
BGE-M3
-> quickly retrieve a broad candidate set

bge-reranker-v2-m3
-> jointly score question + candidate
-> improve final ranking precision
```

The v2 flow is:

```text
Expanded Query
      ↓
BGE-M3 Dense + Sparse Retrieval
      ↓
RRF / BGE-M3 ranking
      ↓
Broad Candidate Pool
      ↓
Exact Product Filter
      ↓
bge-reranker-v2-m3
      ↓
Final Top-K
```

The dedicated reranker can only reorder candidates already retrieved by BGE-M3. It cannot recover a chunk that is completely absent from the candidate pool.

---

## 16. Generated data layout

After a complete build:

```text
DATA_DIR/rag_v6/
├── txt/
│   └── html/
├── md/
│   ├── html/
│   └── pdf/
├── page_images/
├── manifest.jsonl
├── chunks.jsonl
└── index/
    ├── dense.npy
    ├── sparse.jsonl
    ├── chunks.jsonl
    └── index_meta.json
```

| Path | Purpose |
|---|---|
| `md/html/` | cleaned Markdown generated from HTML/TXT |
| `md/pdf/` | one Markdown file per target PDF page |
| `page_images/` | rendered PDF pages for multimodal grounding |
| `manifest.jsonl` | generated-source manifest |
| `chunks.jsonl` | heading-aware retrieval chunks |
| `index/dense.npy` | normalized BGE-M3 dense vectors |
| `index/sparse.jsonl` | BGE-M3 sparse lexical weights |
| `index/chunks.jsonl` | indexed chunk snapshot |
| `index/index_meta.json` | index metadata |

---

## 17. Evaluation recommendations

Final answer accuracy alone is not enough to diagnose RAG failures.

Recommended metrics:

```text
Document Recall@K
Page Recall@K
Chunk Recall@K
Candidate Recall@30
MRR before reranking
MRR after reranking
Final QA Accuracy
```

For PDF questions, record the ground-truth page and compare it against retrieved page numbers.

Example:

```json
{
  "question": "How is PInternal calculated?",
  "ground_truth_page": 6,
  "bge_candidate_pages": [7, 5, 6, 8, 1],
  "reranked_pages": [6, 7, 5, 8, 1],
  "answer_correct": true
}
```

This makes it possible to tell whether the reranker actually moves answer-bearing evidence toward the top.

---

## 18. Run tests

```bash
pytest -q
```

The current test suite covers areas such as:

- heading-aware chunk splitting;
- PDF previous/target/next context selection;
- target/context page labeling.

Machine-specific helper scripts may contain absolute paths from earlier environments. Edit those paths if needed, or use the Python commands documented here directly.

---

## 19. Common issues

### `Configured dataset paths are invalid`

Check:

```text
rag_app/config.py
```

Then run:

```bash
python rag.py paths
python rag.py inspect
```

### `Gemma GGUF not found`

Set:

```python
GEMMA_MODEL_PATH = Path("/correct/path/model.gguf")
```

### `Index files are missing`

Run:

```bash
python rag.py chunk
python rag.py index
```

### `FlagEmbedding` is missing

Install:

```bash
pip install -U FlagEmbedding
```

### Out-of-memory during PDF preprocessing or QA

Start with a smaller workload:

```bash
python rag.py prepare-pdf \
  --limit 1 \
  --page-limit 1 \
  --context-radius 0
```

Then increase the workload gradually.

### The source contains the answer, but RAG answers incorrectly

First inspect basic retrieval:

```bash
python rag.py search "<same question>" --top-k 10
```

For reranker diagnostics:

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --candidate-top-k 30 \
  --top-k 5 \
  --save-results
```

Check:

1. whether the correct document/page entered the BGE-M3 candidate pool;
2. whether the reranker moved it upward;
3. whether the correct evidence was sent to Qwen;
4. whether Qwen still answered incorrectly despite correct evidence.

---

## 20. Current limitations

The current implementation does not yet include:

- QA-stage neighboring page/chunk expansion;
- metadata pre-filtering before vector search;
- Chinese/English synonym expansion;
- formula/Unicode canonicalization;
- iterative evidence-sufficiency checking;
- query rewrite and retry;
- full agentic retrieval;
- automatic Page Recall@K evaluation in the main pipeline.

These are possible future extensions.

---

## 21. Git and large files

Before pushing:

```bash
git status
```

Recommended `.gitignore` entries include:

```text
*.gguf
data_photo_coupler/
data_photo*/
```

If a large file was already tracked before being added to `.gitignore`, remove it only from Git's index:

```bash
git rm -r --cached <path>
```

Then commit again.

---

## 22. Minimal command sequence

```bash
source .venv/bin/activate

python rag.py paths
python rag.py inspect

python rag.py prepare-html
python rag.py prepare-pdf --context-radius 1
python rag.py chunk
python rag.py index

python rag.py search "EL3120 的最大驅動電流是多少？" --top-k 10
python rag.py ask "EL3120 的最大驅動電流是多少？"
```

Interactive entity-aware QA:

```bash
python rag_ans.py --top-k 5
```

Batch baseline:

```bash
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --save-results
```

Batch evaluation with dedicated reranker:

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --candidate-top-k 30 \
  --top-k 5 \
  --save-results
```
