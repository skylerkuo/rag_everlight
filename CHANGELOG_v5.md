# v5 — Ubuntu Absolute Path Edition

- Removed `UBUNTU_HOME / ...` path composition.
- Every important project/data/output/model/cache path is declared explicitly as a full `/home/...` path in `rag_app/config.py`.
- Added explicit absolute paths for all generated RAG directories and index files.
- `rag.py paths` now prints the entire path configuration.
- BGE-M3 index filenames use explicit configured absolute paths.
- `run_build_all.sh` and `run_test.sh` use absolute project/Python paths.
- Kept v3/v4 PDF behavior: previous + target + next page context, with target-page-only Markdown output.
