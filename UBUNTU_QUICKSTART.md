# Ubuntu Quick Start — v6

## 1. Edit only the dataset root

Open:

```bash
nano /home/gai/Desktop/rag_multisource_system_v6/rag_app/config.py
```

Normally the only dataset path you need to change is:

```python
DATA_DIR = Path("/home/gai/Desktop/data_photo_coupler")
```

The following are derived automatically:

```python
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdf"
CRAWLER_TEXT_DIR = DATA_DIR / "text"
DB_PATH = DATA_DIR / "everlight.db"

WORK_DIR = DATA_DIR / "rag_v6"
MD_HTML_DIR = WORK_DIR / "md" / "html"
MD_PDF_DIR = WORK_DIR / "md" / "pdf"
PAGE_IMAGE_DIR = WORK_DIR / "page_images"
INDEX_DIR = WORK_DIR / "index"
```

There is **no custom `HF_CACHE_DIR`**. Hugging Face/Transformers uses its normal default cache behavior.

The Gemma model is outside the dataset, so keep/update this separately if necessary:

```python
GEMMA_MODEL_PATH = Path(
    "/home/gai/Desktop/llm_eval-main/gemma-4-E4B-it-Q4_K_M.gguf"
)
```

## 2. Create the Ubuntu Python environment

```bash
cd /home/gai/Desktop/rag_multisource_system_v6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch build appropriate for your machine when needed.
For `llama-cpp-python`, use the same Ubuntu/CUDA installation method that already works in your `llm_eval-main` environment.

## 3. Verify derived paths

```bash
python rag.py paths
python rag.py inspect
```

## 4. Small test first

HTML:

```bash
python rag.py prepare-html --limit 1
```

PDF — one PDF, first three target pages, previous/target/next context:

```bash
python rag.py prepare-pdf --limit 1 --page-limit 3 --context-radius 1
```

Check the generated Markdown before running everything.

## 5. Chunk, index, retrieve, answer

```bash
python rag.py chunk
python rag.py index
python rag.py search "What is the isolation voltage?"
python rag.py ask "What is the isolation voltage?"
```

## 6. Full build

```bash
python rag.py build --context-radius 1
```
