# Design decisions

## Why heading-aware Markdown chunks

Fixed character windows are easy but they often mix unrelated product features,
applications, tables and navigation text.  Markdown created by Gemma/Qwen already
contains useful structure, so the chunker should exploit it instead of discarding it.

## Why PDF chunks do not cross page boundaries

The online QA path attaches the exact source page image when a PDF chunk is retrieved.
A chunk spanning pages 5 and 6 would create ambiguous visual grounding.  Therefore a
PDF page is a hard parent boundary; smaller chunks may be created inside that page.

## Why HTML stays one MD per source page

This preserves the crawler's 1:1 source identity and source URL.  Chunking happens
later, so retrieval granularity can evolve without rerunning Gemma conversion.

## Why store both content and embedding_text

`content` is the evidence shown to the answer model. `embedding_text` adds the title,
heading path and PDF page number before embedding. This improves retrieval without
polluting the displayed source text.

## Why BGE-M3 hybrid retrieval

Product/technical documents need both semantics and exact lexical matches such as
EL3120, CTR, Vrms, SOP-DC, and part numbers. Dense retrieval helps paraphrases while
sparse weights help exact identifiers. RRF avoids assuming their raw score scales are
comparable. Optional BGE-M3 pair scoring then uses the model's multi-vector capability
on only a small candidate set.

## Why Qwen uses previous + target + next page for PDF parsing

Parsing only one PDF page is brittle for technical documents because headings, tables,
figures and lists often cross page boundaries. The default parser therefore uses a
sliding visual context window with `PDF_CONTEXT_RADIUS=1`.

For target page `p`, Qwen normally receives `[p-1, p, p+1]`, with every image explicitly
labeled. The model is instructed to output only page `p`; neighboring pages are context
only. This keeps the downstream invariant `one PDF page -> one MD -> one page image`
while improving structural continuity.

A wider radius is configurable but is not the default because visual-token cost, VRAM,
and the risk of cross-page content leakage increase with each extra page. Radius 1 is
the best default trade-off for the current technical PDF corpus.

An alternative two-pass design would parse every page independently and then run a
second stitching model over adjacent Markdown outputs. That gives even stricter page
isolation, but doubles pipeline complexity and inference work. The current sliding
3-page method is simpler and gives Qwen the original visual evidence needed to recover
continued tables and layouts.
