# RegRAG

A question-answering system over 506 RBI (Reserve Bank of India) circulars. Ask a
question about Indian banking regulation; get an answer with citations that are
automatically verified against the source documents.

**Status: Week 1 of 10 (naive pipeline working end-to-end). Not yet evaluated
against a ground-truth answer key — that's Week 2.**

## Why this corpus is harder than it looks

RBI reissues the same rule once per class of regulated entity — a KYC amendment
exists as ten near-identical circulars, one each for Commercial Banks, Small
Finance Banks, NBFCs, and so on. **74% of the corpus sits in a near-duplicate
family.** Retrieval that ignores which entity a document binds returns the wrong
institution's rules most of the time (measured: 35% precision unfiltered vs. 92%
with the entity filter — see `embed.py`). Every chunk in this pipeline carries an
`applies_to` tag for exactly this reason.

## Pipeline

Each stage is a standalone script, run in order:

```
python rbi_scraper.py harvest     # walk RBI's circular pages -> manifest.db
python rbi_scraper.py text        # cache each circular's full body -> data/html/
python chunker.py                 # split into retrieval-sized chunks -> chunks.jsonl
python embed.py build             # embed every chunk (BGE-small)   -> embeddings.npy
python index_qdrant.py build      # load into Qdrant (needs Docker) -> local vector index
python rag.py ask "..."           # ask a question
python rag.py batch               # run the 20-question test set
python rag.py history             # everything ever asked, all-time
```

### Requirements to run the last three steps

- **Qdrant** running locally: `docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`
- **A Gemini API key** (free tier): `setx GEMINI_API_KEY "..."` — get one at
  <https://aistudio.google.com/apikey>. Check
  <https://aistudio.google.com/rate-limit> for your daily quota per model before
  running `batch` — the free tier meters requests *per day, per model*.

## Current numbers

- **506 circulars**, Feb 2024 – Aug 2026, harvested directly (PDF downloads are
  blocked by a WAF; the corpus is built from the same detail pages' HTML instead —
  see the design note at the top of `rbi_scraper.py`)
- **9,278 chunks**, each carrying its document's number, date, department, and the
  regulated-entity classes it binds
- **Entity filter**: 35% → 92% retrieval precision (measured against 8 hand-written
  queries, `embed.py entity-test`)
- **First end-to-end test**: 17/20 questions answered (3 blocked by daily quota),
  citations 37/37 verified against source text — see `python rag.py history`

## What's not solved yet

- **Amendment directions state deltas, not rules.** ~28% of chunks belong to
  documents that say "paragraph 5.2 shall be substituted" without restating the
  rule. Retrieval can land exactly on the right document and still have nothing
  to answer from. The pipeline correctly declines in this case rather than
  guessing — but it means a fraction of questions are genuinely unanswerable from
  a single retrieved document as things stand. Resolving amendment chains is a
  Week 3 target.
- **Citation verification checks that quotes are real, not that answers are
  complete.** A verified citation can still support a shallow or incomplete
  answer. Faithfulness scoring is a Week 2 addition.
- **No ground-truth eval yet.** The 20-question batch is a smoke test, not a
  scored benchmark. Week 2 builds a 75–100 question answer key written before
  looking at what the pipeline produces.

## Every question is logged, automatically

`rag.py` appends every question — asked one at a time or as part of a batch — to
`data/log.jsonl`, along with the answer, retrieved documents, citation
verification results, and timing. Nothing is overwritten; `python rag.py history`
summarizes it. `data/answers.jsonl` is separate: just the most recent `batch` run,
for a quick look at the latest results.
