# RegRAG

A question-answering system over 506 RBI (Reserve Bank of India) circulars. Ask a
question about Indian banking regulation; get an answer with citations that are
automatically verified against the source documents.

**Status: end-to-end pipeline working and measured against a hand-written answer
key, with each failure attributed to retrieval or to generation.** Retrieval and
citation accuracy have numbers; answer faithfulness does not yet.

## Why this corpus is harder than it looks

RBI reissues the same rule once per class of regulated entity — a KYC amendment
exists as ten near-identical circulars, one each for Commercial Banks, Small
Finance Banks, NBFCs, and so on. **74% of the corpus sits in a near-duplicate
family.** Retrieval that ignores which entity a document binds returns the wrong
institution's rules most of the time (measured: 35% precision unfiltered vs. 92%
with the entity filter — see `embed.py entity-test`). Every chunk in this pipeline
carries an `applies_to` tag for exactly this reason.

This one property shapes every design decision below, and it is also why several
obvious-looking metrics turn out to measure less than they appear to.

## Pipeline

Each stage is a standalone script, run in order:

```
python rbi_scraper.py harvest     # walk RBI's circular pages -> manifest.db
python rbi_scraper.py text        # cache each circular's full body -> data/html/
python chunker.py                 # split into retrieval-sized chunks -> chunks.jsonl
python embed.py build             # embed every chunk (BGE-small)   -> embeddings.npy
python index_qdrant.py build      # load into Qdrant (needs Docker) -> local vector index
python rag.py ask "..."           # ask a question
python rag.py history             # everything ever asked, all-time
```

Evaluation is a separate track, deliberately split by what each metric costs:

```
python qa.py draft                # add question stubs to data/qa_draft.yaml
python qa.py validate             # what's incomplete or malformed
python qa.py export               # -> data/qa.jsonl, the graded set
python eval.py retrieval          # hit-rate@k -- local, free, unlimited
python eval.py retrieval --sweep  # where the recall curve flattens
python eval.py answers            # end-to-end run (costs quota, resumable)
python eval.py answers --oracle   # feed gold chunks, bypass retrieval
python eval.py report             # scorecard from whatever has been run
```

### Requirements to run generation

- **Qdrant** running locally: `docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`
- **A Groq API key** (free tier) — get one at <https://console.groq.com>, then
  `cp .env.example .env` and paste it in.

  Generation needs strict JSON-schema output, so the model must be one that
  supports it (`openai/gpt-oss-120b` or `-20b`). On the free tier the binding
  limit is **tokens per day (200,000)**, not requests per day (1,000): at ~2,900
  tokens per call that is roughly 70 questions a day, and 8,000 tokens/minute
  caps throughput at ~3 calls a minute. `rag.py` paces itself against the
  per-minute ceiling rather than absorbing 429s. Limits are per model, so a
  second model can carry the eval judge on its own allowance.

  One non-obvious failure mode, since it cost two questions in the first full
  run: under strict JSON schema an answer that exceeds the output budget is not
  a *shorter* answer, it is **no answer** — the object never closes, the required
  fields after it are never emitted, and the API rejects the whole reply with a
  400. `gpt-oss` also spends reasoning tokens from that same budget, so a cap set
  near the expected answer length starves the answer entirely and returns an
  empty generation. `GROQ_MAX_OUTPUT_TOKENS` defaults to 4096 for this reason;
  the longest legitimate answer in the ground-truth set needed 2,311.

## The corpus

- **506 circulars**, Feb 2024 – Aug 2026, harvested directly (PDF downloads are
  blocked by a WAF; the corpus is built from the same detail pages' HTML instead —
  see the design note at the top of `rbi_scraper.py`)
- **9,278 chunks**, each carrying its document's number, date, department, and the
  regulated-entity classes it binds

## The answer key

**90 questions written by hand, before any pipeline output was looked at.** That
ordering is the whole point: grade after seeing results and you unconsciously
write a key the system passes.

| Category | n | What it stresses |
|---|---|---|
| `lookup` | 31 | Single fact, single document |
| `entity_scoped` | 25 | Right rule, right institution — the near-duplicate problem |
| `amendment` | 15 | Documents stating deltas, not rules |
| `cross_document` | 10 | Needs evidence from two circulars at once |
| `unanswerable` | 9 | Correct behaviour is to refuse |

68 are answerable, 22 are not. Note that only 9 sit in the `unanswerable`
*category*: the other 13 were written as ordinary questions and turned out to have
no answer in the corpus, discovered only when going to the source to write the
answer down. That makes them refusal tests in disguise — they look perfectly
answerable, and a system that always produces something confident fails them while
passing everything else.

`gold_chunk_ids` records which chunks a question was drafted against. That is
usually also where it is answered, but not always: a question drafted from a
preamble may be answered by the body of the same document. Where the two came
apart the pointer was corrected, and the correction logged in
`data/qa_review.yaml` — see q062, the one instance found so far. Verdicts in that
file are never edited to match model output; a pointer that indexes the wrong text
is a different thing from a judgement about whether an answer was right.

## Measured results

### Retrieval — `python eval.py retrieval`

| | exact gold doc | gold doc or sibling |
|---|---|---|
| entity filter **on** | 52/68 (76%) | 54/68 (79%) |
| entity filter **off** | 50/68 (73%) | 53/68 (77%) |

By category, filter on: `entity_scoped` **19/19**, `amendment` 9/12,
`lookup` 19/29, `cross_document` 5/8 (both documents: 3/8).

Two things worth reading carefully here:

- **The "sibling" column exists because strict scoring is unfair to some
  questions.** A question that names no institution — *"what must an internal
  audit review at minimum?"* — has ten equally correct source documents, and the
  key can only name one. Returning the Co-operative Banks copy instead of the
  Payments Banks copy is not a miss. Both numbers are reported; the key is never
  edited to match what the system returned. See `data/qa_review.yaml`.
- **The entity filter is worth ~3 points here, not the 57 the precision figure
  above suggests.** Those measure different things: precision asks whether the
  *other* seven slots are the right entity's, hit-rate only asks whether the gold
  document appeared at all. Both numbers are real; neither substitutes for the
  other.

When retrieval succeeds it succeeds decisively — **median rank of a hit is 1**.
Failures are absolute rather than ranking problems: sweeping k from 1 to 20 lifts
recall by a modest margin and then flattens, so a reranker over a wider candidate
set has little left to recover. Chunk length shows no correlation with misses
either. (The sweep numbers currently on record predate the latest key revision and
should be re-run before being quoted.)

### Generation — `python eval.py answers`

Complete run, all 90 questions, no errors:

- **Citations verified: 121/129 (93%)** — 0 misattributed, 8 not present in any
  retrieved source.
- **Of those verified, 19% are unfalsifiable.** When ten sibling circulars carry
  the same sentence, a quote lifted from the NBFC copy is genuinely a substring of
  the Payments Banks copy too, so the check returns `ok` and *cannot* detect
  misattribution. One quote was found in 5 of the 8 retrieved documents. This is
  why "0 misattributed" is partly good behaviour and partly a blind spot.
- **Refusal: 10/23 overall** — but the aggregate hides the finding that matters.

#### Refusal splits cleanly, and not in the flattering direction

| Kind of unanswerable question | Refused correctly |
|---|---|
| Explicitly out of scope (`unanswerable` category) | **8/9** |
| Disguised — reads like regulation, drawn from a real circular | **2/14** |

**The system detects *off-topic* reliably and *on-topic but unsupported* almost
never.** Asked the RBI governor's name or what time banks break for lunch, it
declines. Asked something that sounds exactly like a regulatory question, sourced
from a genuine circular that happens not to contain the answer, it answers anyway
in 12 of 14 cases — fluently, with citations that verify.

That second row is the one that matters for a regulatory tool, and it is the
weakest result in the project. It is also invisible to every other metric here: a
confident wrong answer with a real quote passes citation verification.

The failures have named modes, recorded with evidence in `data/qa_review.yaml`:
`circular_restatement` (offering the rule as its own justification),
`definitional_deflection` (answering "what does X contain?" with X's title),
`boilerplate_as_substance`, `metadata_as_substance`, `template_regurgitation`,
`source_restatement`, and exactly one outright `fabrication`. Only one in thirteen
is invention — the rest are evasion that reads as an answer, which is a harder
problem than hallucination and a much quieter one.

### Retrieval or generation? — `python eval.py answers --oracle`

Oracle mode feeds the gold chunks directly and skips retrieval, so comparing the
two runs attributes each failure to a stage instead of guessing.

| | end-to-end | oracle |
|---|---|---|
| Refused, explicit out-of-scope | 8/9 | **9/9** |
| Refused, disguised unanswerable | 2/14 | **6/14** |
| Refused when it *shouldn't* have | 11/67 | **1/67** |
| Citations verified | 121/129 (93%) | 115/120 (95%) |
| — of those, unfalsifiable | 24 (19%) | **2 (1%)** |
| Median latency | 2.0s | 2.1s |

Three conclusions, each attributable:

- **Ten wrongful refusals were retrieval's fault.** Questions the corpus answers,
  declined only because the evidence never surfaced. `report` lists them by id.
  Hit-rate@8 alone never revealed this.
- **Four refusal failures were retrieval's fault too.** q033, q063, q070 and q072
  refuse correctly the moment the near-duplicate distractors are removed.
  Surrounded by ten variants of one rule the model feels informed enough to answer;
  handed a single clean chunk it concedes the chunk does not say.
- **Eight refusal failures are not retrieval's fault.** Given nothing but the exact
  chunk the question was written from, the system still answers. No retrieval work
  will touch these; they need the prompt or the output contract.

The **19% → 1% collapse in unfalsifiable citations** is the cleanest evidence in
the project that the near-duplicate structure — not the checker — is what makes
citation verification weaker than its headline number. Remove the siblings and the
ambiguity disappears entirely.

One caveat on oracle mode: **it is only as good as `gold_chunk_ids`.** Where a
question was authored from a preamble, oracle feeds strictly *weaker* evidence than
retrieval finds by itself, and a failure then indicts the key rather than the model.
q062 is exactly that case — its gold chunks are two document openings, while the
passages that answer it sit in the same document's body and were never listed. It
is the only one of the eight with this problem, and it is documented in
`data/qa_review.yaml` rather than quietly patched.

## What's not solved

- **Refusal on in-scope questions, above all.** A confident wrong answer about
  banking regulation is worse than no answer, and 2/14 is where the system is
  furthest from usable. Out-of-scope rejection already works; the gap is knowing
  when a *plausible* question isn't supported by the text in hand.
- **Amendment directions state deltas, not rules.** ~28% of chunks belong to
  documents that say "paragraph 5.2 shall be substituted" without restating the
  rule. Retrieval can land exactly on the right document and still have nothing to
  answer from.
- **Citation verification checks that quotes are real, not that answers are
  complete, nor that attribution is correct.** See the 16% figure above.
- **Answer faithfulness is unmeasured.** An LLM judge is the next addition; it will
  run on a different Groq model so it draws on a separate token allowance.
- **Deduplication before generation is unbuilt, and now clearly justified.** The
  entity filter cannot separate same-entity boilerplate: after filtering, up to ten
  near-identical copies of one rule still compete for eight slots. That noise costs
  4 refusal failures and nearly all of the citation ambiguity.
- **`gold_chunk_ids` records the chunk a question was authored from, not
  necessarily the chunk that answers it.** For most questions those coincide; for
  ones drafted from a preamble they do not, which distorts oracle-mode results.
  q062 is the known instance.

## Every question is logged, automatically

`rag.py` appends every question — asked one at a time or as part of a batch — to
`data/log.jsonl`, along with the answer, retrieved documents, citation
verification results, and timing. Nothing is overwritten; `python rag.py history`
summarizes it. Evaluation runs write separately to `data/eval/`.

Tracing is optional: set Langfuse keys in `.env` to get a span per pipeline stage,
or leave them blank and the pipeline runs identically with tracing disabled.
