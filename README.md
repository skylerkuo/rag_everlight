# rag_everlight

<<<<<<< HEAD
A local multi-source RAG system for Everlight technical documents. The repository ingests crawler-produced HTML/TXT and PDF data, converts the sources into structured Markdown, builds a BGE-M3 hybrid dense+sparse index, retrieves Top-K evidence, and uses Qwen3.5-4B as the final multimodal answer model.
=======
A local multi-source RAG system for Everlight technical documents. The repository ingests crawler-produced HTML/TXT and PDF data, converts the sources into structured Markdown, builds a BGE-M3 hybrid dense+sparse index, retrieves candidate evidence, and uses Qwen3.5-4B as the final multimodal answer model. The `rag_loop_v2.py` evaluation path additionally applies a dedicated `BAAI/bge-reranker-v2-m3` second-stage reranker before the final Top-K is sent to Qwen.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

For PDF evidence, the exact retrieved source page image can also be sent to Qwen so tables, figures, formulas, values, and layout can be checked against the original page.

For the implementation rationale and retrieval details, see [`DESIGN.md`](DESIGN.md).

---

## 1. Architecture at a glance

```text
HTML -> crawler TXT -> Gemma 4 E4B -> Markdown
                                         │
PDF -> page images -> Qwen3.5-4B -> page Markdown
                                         │
                                         ▼
                             Markdown-aware chunks
                                         │
                                         ▼
                         BGE-M3 dense + sparse index
                                         │
                            Dense/Sparse -> RRF
                                         │
                         optional BGE-M3 pair scoring
                                         │
                                         ▼
                                  final Top-K
                                default = 5
                                  │       │
                                  │       └-> retrieved PDF page images
                                  │           max 4 images
                                  ▼
                         Qwen3.5-4B final answer
                                  │
                                  ▼
                            answer + [S#]
```

<<<<<<< HEAD
The repository also contains an entity-aware query path (`rag_loop.py` and `rag_ans.py`) that uses Qwen to extract search keywords and exact product/model names before retrieval.
=======
The repository also contains an entity-aware query path (`rag_loop.py` and `rag_ans.py`) that uses Qwen to extract search keywords and exact product/model names before retrieval. `rag_loop_v2.py` extends the batch path with a dedicated cross-encoder reranker after BGE-M3 candidate retrieval and Exact Product Filter.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

---

## 2. Repository structure

```text
rag_everlight/
├── rag.py                         # main build/search/ask CLI
<<<<<<< HEAD
├── rag_loop.py                    # batch entity-aware evaluation
=======
├── rag_loop.py                    # batch entity-aware evaluation baseline
├── rag_loop_v2.py                 # batch evaluation + dedicated reranker
>>>>>>> f07dea0 (Add reranker and update RAG docs)
├── rag_ans.py                     # interactive entity-aware terminal
├── rag_app/
│   ├── config.py                  # paths and runtime defaults
│   ├── metadata.py                # crawler DB / metadata loading
│   ├── inspect_data.py
│   ├── preprocess/
│   │   ├── html_to_md.py          # TXT/HTML -> Gemma -> MD
│   │   └── pdf_to_md.py           # PDF images -> Qwen -> page MD
│   ├── chunking/
│   │   └── markdown_chunker.py    # heading-aware chunking
│   ├── retrieval/
<<<<<<< HEAD
│   │   └── bge_m3_index.py        # dense+sparse BGE-M3 retrieval
=======
│   │   ├── bge_m3_index.py        # dense+sparse BGE-M3 retrieval
│   │   └── reranker.py            # BAAI/bge-reranker-v2-m3 second-stage reranking
>>>>>>> f07dea0 (Add reranker and update RAG docs)
│   ├── qa/
│   │   └── engine.py              # Top-K evidence -> Qwen answer
│   └── models/
│       ├── gemma4_llamacpp.py
│       └── qwen35_vl.py
├── tests/
├── requirements.txt
├── questions.jsonl
├── DESIGN.md
└── README.md
```

The large dataset and local GGUF model are intentionally not included in Git. The repository `.gitignore` excludes `data_photo*` and `*.gguf`.

---

## 3. Required external data

The project expects an existing crawler dataset. Set its location through `DATA_DIR` in `rag_app/config.py`.

Expected layout:

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
    └── documents.jsonl            # optional metadata fallback
```

`everlight.db` is used to map SHA/document IDs to source URLs, titles, raw paths, TXT paths, languages, and PDF page counts.

Generated RAG data is created under:

```text
DATA_DIR/rag_v6/
```

---

## 4. Models

The current code uses:

| Stage | Model |
|---|---|
| HTML/TXT cleanup | local Gemma 4 E4B GGUF through `ChatLlamaCpp` |
| PDF page understanding | `Qwen/Qwen3.5-4B` |
| Retrieval | `BAAI/bge-m3` |
<<<<<<< HEAD
| Final answer | `Qwen/Qwen3.5-4B` |
| Entity/keyword extraction in `rag_loop.py` / `rag_ans.py` | same loaded Qwen3.5-4B |

The Gemma GGUF is local and must be downloaded/provided separately. Qwen and BGE-M3 use normal Hugging Face cache behavior and are downloaded by Hugging Face when first loaded if they are not already cached.
=======
| Dedicated reranking in `rag_loop_v2.py` | `BAAI/bge-reranker-v2-m3` |
| Final answer | `Qwen/Qwen3.5-4B` |
| Entity/keyword extraction in `rag_loop.py` / `rag_loop_v2.py` / `rag_ans.py` | same loaded Qwen3.5-4B |

The Gemma GGUF is local and must be downloaded/provided separately. Qwen, BGE-M3, and the dedicated reranker use normal Hugging Face cache behavior and are downloaded by Hugging Face when first loaded if they are not already cached. The reranker path also requires `FlagEmbedding`.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

---

## 5. Environment setup

refer to previous projects llama.cpp python and requirements.txt

## 6. Configure paths before running

Open:

```text
rag_app/config.py
```

Change at least these two paths.

### Dataset root

```python
DATA_DIR = Path("/your/path/data_photo_coupler")
```

### Local Gemma GGUF

```python
GEMMA_MODEL_PATH = Path(
    "/your/path/gemma-4-E4B-it-Q4_K_M.gguf"
)
```

All crawler and generated RAG paths are derived from `DATA_DIR`; there is no `--data-dir` CLI argument.

The current model IDs are also defined in `config.py`:

```python
QWEN_MODEL_ID = "Qwen/Qwen3.5-4B"
BGE_MODEL_ID = "BAAI/bge-m3"
```

---

## 7. Verify the configuration

Print all resolved paths:

```bash
python rag.py paths
```

Inspect the crawler dataset:

```bash
python rag.py inspect
```

`rag.py` validates that these exist before running expensive stages:

```text
DATA_DIR
DATA_DIR/raw/html
DATA_DIR/raw/pdf
DATA_DIR/text
DATA_DIR/everlight.db
```

If path validation fails, edit `DATA_DIR` in `rag_app/config.py`.

---

## 8. Recommended first smoke test

Do not run the complete corpus first. Test one HTML document and one PDF with a few pages.

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

`--context-radius 1` means a normal middle target page is parsed with:

```text
previous page + target page + next page
```

Only the target page is written to Markdown.

Inspect the generated files under:

```text
DATA_DIR/rag_v6/md/
DATA_DIR/rag_v6/page_images/
```

before building the complete corpus.

---

## 9. Build the complete RAG pipeline

You can run the stages separately, which is recommended while debugging.

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

### Step 3 — chunk generated Markdown

```bash
python rag.py chunk
```

Output:

```text
DATA_DIR/rag_v6/chunks.jsonl
```

Production chunking uses the BGE-M3 tokenizer. For a lightweight debug run that does not require the Hugging Face tokenizer download:

```bash
python rag.py chunk --approx-tokenizer
```

### Step 4 — build BGE-M3 index

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

After the configuration and smoke tests are correct, the entire pipeline can be run with:

```bash
python rag.py build --context-radius 1
```

`build` performs:

```text
HTML preprocessing
-> release accelerator memory
-> PDF preprocessing
-> release accelerator memory
-> chunking
-> BGE-M3 indexing
```

Use `--force` when you intentionally want existing generated Markdown/images to be regenerated:

```bash
python rag.py build --force --context-radius 1
```

---

## 10. Search the index without answering

Use `search` when debugging retrieval:

```bash
python rag.py search "EL3120 PInternal formula"
```

The default final retrieval size is configured as:

```python
top_k = 5
```

Override it from the command line:

```bash
python rag.py search "EL3120 PInternal formula" --top-k 10
```

Each result contains retrieval scores plus chunk provenance such as document ID, source kind, page number, heading path, and page image.

This command is the best first tool for deciding whether a wrong answer is caused by retrieval or by the answer model.

---

## 11. Ask a question with the basic RAG path

```bash
python rag.py ask "EL3120 的 PInternal 如何計算？"
```

Or specify the final retrieval size:

```bash
python rag.py ask "EL3120 的 PInternal 如何計算？" --top-k 10
```

The basic `rag.py ask` path is:

```text
question
-> BGE-M3 hybrid retrieval
-> final Top-K chunks
-> up to 4 unique PDF page images
-> Qwen3.5-4B
-> grounded answer
```

Important defaults:

```text
BGE candidate_k      = 24
final top_k          = 5
max PDF answer images = 4
```

The answer prompt instructs Qwen to use only the retrieved evidence and attached images and to cite evidence as `[S1]`, `[S2]`, etc.

---

## 12. Interactive entity-aware RAG terminal

For repeated interactive questions, use:

```bash
python rag_ans.py
```

Optional Top-K:

```bash
python rag_ans.py --top-k 5
```

This path is different from `rag.py ask`.

Before retrieval, Qwen extracts:

```json
{
  "keywords": [],
  "proper_nouns": []
}
```

Then the system:

1. keeps the original question;
2. appends extracted search terms;
3. retrieves a broader candidate pool;
4. applies an Exact Product Filter when a product/model name was detected;
5. keeps the final Top-K;
6. sends retrieved text plus PDF page images to Qwen.

Each terminal question is stateless. The current implementation does **not** keep conversation memory and does **not** run iterative evidence-sufficiency/query-rewrite loops.

Type one of the following to exit:

```text
exit
quit
q
```

The terminal also reports generated-token probability for the final answer. This is a raw generation-confidence statistic, not a calibrated probability that the answer is factually correct.

---

## 13. Batch evaluation

<<<<<<< HEAD
`rag_loop.py` reads one JSON object per line. Every line must contain at least:
=======
Both `rag_loop.py` and `rag_loop_v2.py` read one JSON object per line. Every line must contain at least:
>>>>>>> f07dea0 (Add reranker and update RAG docs)

```json
{"question": "EL3120 的最大驅動電流是多少？"}
```

If an `answer` field is present, it is copied into the output as `ground_truth`.

<<<<<<< HEAD
Run:
=======
### Baseline entity-aware batch path
>>>>>>> f07dea0 (Add reranker and update RAG docs)

```bash
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl
```

<<<<<<< HEAD
Useful options:

```bash
# Only test the first 10 questions
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --limit 10

# Resume an interrupted run
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --resume

# Save complete Top-K retrieval records for failure analysis
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --save-results

# Override final Top-K
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --top-k 10 \
  --save-results
```

The batch runner loads BGE-M3 and Qwen only once and writes each result immediately, so a later interruption does not discard earlier completed questions.

---

## 14. Plain RAG vs entity-aware RAG

There are currently two retrieval entry points.

| Command | Query analysis | Exact Product Filter | Default final Top-K | Use case |
|---|---|---|---:|---|
| `python rag.py ask ...` | No | No | 5 | simple QA / retrieval debugging |
| `python rag_ans.py` | Yes, Qwen | Yes | 5 | interactive product QA |
| `python rag_loop.py ...` | Yes, Qwen | Yes | config/default 5 | batch evaluation |
=======
### Reranker batch path

`rag_loop_v2.py` keeps the same query-analysis and Exact Product Filter flow, but retrieves a broader BGE-M3 candidate pool and then uses `BAAI/bge-reranker-v2-m3` to reorder the candidates before selecting the final Top-K.

Install the reranker dependency once:

```bash
pip install -U FlagEmbedding
```

Run:

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --candidate-top-k 30 \
  --top-k 5 \
  --save-results
```

Useful options for `rag_loop_v2.py`:

```bash
# Only test the first 10 questions
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --limit 10

# Resume an interrupted run
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --resume

# Disable the dedicated reranker for an A/B baseline run
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2_no_rerank.jsonl \
  --disable-reranker \
  --save-results
```

The v2 output can preserve both the BGE-M3 candidate rank and the final reranker score/rank, which is useful for determining whether an error is caused by retrieval recall or by ranking. Both batch runners write each completed result immediately so an interruption does not discard earlier outputs.

---

## 14. Plain RAG vs entity-aware / reranked RAG

There are currently multiple retrieval entry points.

| Command | Query analysis | Exact Product Filter | Dedicated reranker | Default final Top-K | Use case |
|---|---|---|---|---:|---|
| `python rag.py ask ...` | No | No | No | 5 | simple QA / retrieval debugging |
| `python rag_ans.py` | Yes, Qwen | Yes | No | 5 | interactive product QA |
| `python rag_loop.py ...` | Yes, Qwen | Yes | No | config/default 5 | batch baseline |
| `python rag_loop_v2.py ...` | Yes, Qwen | Yes | `bge-reranker-v2-m3` | 5 | batch evaluation with second-stage reranking |
>>>>>>> f07dea0 (Add reranker and update RAG docs)

The entity-aware path first retrieves at least 30 results before exact-name filtering:

```python
candidate_top_k = max(final_top_k, 30)
```

<<<<<<< HEAD
If no retrieved candidate contains the extracted exact product name, the code falls back to the unfiltered retrieval list instead of returning an empty result set.
=======
If no retrieved candidate contains the extracted exact product name, the code falls back to the unfiltered retrieval list instead of returning an empty result set. In `rag_loop_v2.py`, the surviving candidate pool is then scored by the dedicated reranker and only the highest-ranked results are sent to the final answer model.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

---

## 15. Retrieval method

The index uses BGE-M3 in two modes for every query:

```text
Dense retrieval  -> semantic similarity
Sparse retrieval -> exact/lexical matching
```

The two rank lists are fused by weighted Reciprocal Rank Fusion:

```text
Dense RRF weight  = 0.55
Sparse RRF weight = 0.45
```

<<<<<<< HEAD
By default, BGE-M3 pair scoring is then attempted on the candidate set before final Top-K selection.
=======
By default, the core BGE-M3 index can attempt its own BGE-M3 pair scoring before final Top-K selection. This is separate from the dedicated cross-encoder reranker used by `rag_loop_v2.py`.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

This is why technical tokens such as the following can contribute through the sparse branch while paraphrased questions can still match through dense retrieval:

```text
EL3120
EL827
CTR
Vrms
PInternal
Rg
VEE
```

<<<<<<< HEAD
See [`DESIGN.md`](DESIGN.md) for the complete scoring flow.
=======
In `rag_loop_v2.py`, the online ranking flow is therefore: BGE-M3 broad retrieval -> Exact Product Filter -> `bge-reranker-v2-m3` -> final Top-K. See [`DESIGN.md`](DESIGN.md) for the complete scoring flow.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

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

### What each item is for

| Path | Purpose |
|---|---|
| `md/html/` | cleaned one-MD-per-HTML content |
| `md/pdf/` | one MD per target PDF page |
| `page_images/` | exact rendered PDF pages used for multimodal grounding |
| `manifest.jsonl` | generated-source manifest |
| `chunks.jsonl` | heading-aware retrieval chunks |
| `index/dense.npy` | normalized BGE-M3 dense vectors |
| `index/sparse.jsonl` | BGE-M3 lexical weights |
| `index/chunks.jsonl` | indexed chunk snapshot |
| `index/index_meta.json` | index metadata |

---

## 17. Run tests

```bash
pytest -q
```

The current tests cover:

- heading-aware chunk splitting;
- PDF previous/target/next context selection;
- target/context page labeling.

The provided `run_test.sh` and `run_build_all.sh` contain machine-specific absolute paths from a previous environment. Either edit those paths before using the scripts or run the Python commands in this README directly.

---

## 18. Common issues

### `Configured dataset paths are invalid`

Check `DATA_DIR` in:

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

Gemma is required by `prepare-html`, `prepare`, and `build`.

### `Index files are missing`

Build chunks and the index first:

```bash
python rag.py chunk
python rag.py index
```

### Out-of-memory during PDF preprocessing or answer generation

Qwen3.5-4B is loaded with `device_map="auto"`. Reduce workload while testing:

```bash
python rag.py prepare-pdf --limit 1 --page-limit 1 --context-radius 0
```

Then increase the workload gradually.

### Answer is wrong even though the source PDF contains the answer

First inspect retrieval instead of immediately changing the answer model:

```bash
python rag.py search "<same question>" --top-k 10
```

Check whether the correct document/page appears in the retrieval results. If the ground-truth page is outside Top-K, the final Qwen model never receives that page's evidence.

<<<<<<< HEAD
For batch diagnosis, use:

```bash
python rag_loop.py \
  --input questions.jsonl \
  --output rag_model_outputs.jsonl \
  --save-results
```

This makes it possible to separate retrieval failures from final-answer failures.
=======
For batch diagnosis with the dedicated reranker, use:

```bash
python rag_loop_v2.py \
  --input questions.jsonl \
  --output rag_model_outputs_v2.jsonl \
  --candidate-top-k 30 \
  --top-k 5 \
  --save-results
```

This makes it possible to compare the BGE-M3 candidate ranking with the final reranked ranking and separate retrieval failures from ranking or final-answer failures.
>>>>>>> f07dea0 (Add reranker and update RAG docs)

---

## 19. Git / large files

The repository is configured not to commit the local dataset and large GGUF model files. Before pushing, check:

```bash
git status
```

The `.gitignore` currently covers patterns including:

```text
*.gguf
data_photo_coupler/
data_photo*/
```

If a large model/dataset was already tracked before `.gitignore` was added, remove it from Git's index without deleting the local copy, then commit again.

---

## 20. Minimal command sequence

For a new machine with the dataset and Gemma path already configured:

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

# Optional entity-aware interactive mode
python rag_ans.py --top-k 5
<<<<<<< HEAD
=======

# Optional batch evaluation with dedicated reranker
python rag_loop_v2.py --input questions.jsonl --output rag_model_outputs_v2.jsonl --candidate-top-k 30 --top-k 5 --save-results
>>>>>>> f07dea0 (Add reranker and update RAG docs)
```

Or, after validation:

```bash
python rag.py build --context-radius 1
python rag_ans.py --top-k 5
```
