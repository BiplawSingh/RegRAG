---
title: RegRAG
emoji: 🏦
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: Question answering over 506 RBI circulars, with verified citations
---

# RegRAG

Question answering over 506 Reserve Bank of India circulars, with citations that
are **checked against the source they claim** rather than merely displayed.

Full write-up, evaluation harness and measured results:
**https://github.com/BiplawSingh/RegRAG**

> **Run it locally:** `python space/sync.py && python space/app.py`
>
> This app is not currently hosted. The frontmatter above is kept so it *can* be
> deployed to a Hugging Face Space unchanged, but a public demo would draw on a
> personal free-tier generation allowance of ~70 questions a day and degrade to
> retrieval-only under any real traffic. The project's claims rest on the measured
> results in the repo, not on a demo.

## What this demo is showing

RBI reissues the same rule once per class of regulated entity, so a KYC amendment
exists as ten near-identical circulars — **74% of the corpus sits in a
near-duplicate family**. Retrieval that ignores which entity a document binds
returns the wrong institution's rules most of the time (measured: 35% precision
unfiltered vs 92% with the filter). Every chunk carries an `applies_to` tag, and
the filter runs *inside* the search rather than over its results.

Retrieval is a hybrid fetch — dense plus BM25, fused with reciprocal-rank fusion —
re-sorted by a cross-encoder that is shown each passage's document subject and
section heading. **Chunk-level retrieval recall is 78% with reranking and 60%
without**; the toggle in the UI runs both so the difference is visible on your own
question rather than only in a table.

| | measured |
|---|---|
| chunk-level retrieval recall @ k=8 | 54/69 (78%) |
| gold document retrieved | 60/69 (87%) |
| answer correct, given the right passage | 95% |

## How it differs from the evaluated pipeline

The dense channel is a numpy dot product over `embeddings.npy` instead of a Qdrant
query, so the Space needs no vector server. Everything else — BM25, the fusion,
the entity mask, the cross-encoder, generation and citation verification — is
imported from the same modules the evaluated pipeline runs. That substitution is
verified rather than assumed: `retrieval_lab.py baseline` scores the numpy path
against the saved Qdrant run and they agree question-for-question.

## Notes

- Generation runs on a **free-tier Groq allowance** that a public demo can exhaust.
  When it does, the app falls back to showing retrieval alone rather than erroring —
  degraded honestly beats down.
- A **verified citation is not proof of correct attribution.** When ten sibling
  circulars carry the same sentence, a quote lifted from one is genuinely a
  substring of the others; the app flags those cases instead of showing a green
  tick that claims more than it proves.
- The last example question is **not answerable from this corpus**. Refusing it is
  the correct behaviour, and one of the things the evaluation measures.
