# v3 change: context-aware PDF parsing

- PDF -> MD no longer gives Qwen only one isolated page.
- Default `PDF_CONTEXT_RADIUS=1` supplies previous + target + next page images.
- Images are explicitly labeled so Qwen knows which one is the target.
- The prompt forbids copying neighbor-only factual content into the target MD.
- Output remains exactly one MD per target PDF page.
- Retrieval/QA still attaches only the exact target page image associated with a hit.
- CLI option added: `--context-radius` for `prepare-pdf`, `prepare`, and `build`.
