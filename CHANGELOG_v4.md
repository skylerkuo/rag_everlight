# v4 changes

- Ubuntu-oriented fixed-path configuration.
- Removed required `--data-dir` argument.
- Removed Gemma GGUF environment-variable dependency.
- Added one central path block in `rag_app/config.py`.
- Added fixed Hugging Face cache directory in code.
- Added `python rag.py paths` to verify all configured locations.
- Added path validation before expensive model inference.
- Preserved v3 sliding PDF context: previous + target + next, single target-page MD output.
- Added Ubuntu shell runners and quick-start documentation.
