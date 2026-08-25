# v6 Changes

- Simplified dataset path configuration to one root: `DATA_DIR`.
- `RAW_HTML_DIR`, `RAW_PDF_DIR`, `CRAWLER_TEXT_DIR`, `DB_PATH`, and `RAG_READY_DOCUMENTS_PATH` are now derived from `DATA_DIR`.
- `WORK_DIR` and every generated Markdown/image/chunk/index path are derived from `DATA_DIR`.
- Removed project-controlled `HF_CACHE_DIR`, `HF_HUB_CACHE_DIR`, and `TRANSFORMERS_CACHE_DIR`.
- The project no longer sets Hugging Face cache environment variables; Transformers/Hugging Face uses its normal default cache behavior.
- Removed unused `PROJECT_DIR` and `VENV_PYTHON_PATH` from runtime settings.
- PDF previous/target/next-page context logic remains unchanged.
