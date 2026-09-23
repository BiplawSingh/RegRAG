# Design decisions: the web demo

Each entry: what was chosen, what was rejected, and why. The retrieval and
generation decisions are in the README, next to the measurements behind them;
this file covers the API and the frontend built on top of that pipeline.

---

## The API

### 1. Citation markers sit in a row under the answer, not inline in the prose
**Chose:** one clickable `[n]` per citation, in a row directly below the answer.
**Rejected:** (a) changing the prompt so the model writes `[n]` into its answer;
(b) guessing which sentence each quote supports and inserting the marker there.
**Why:** the model returns citations as a separate list of quotes, not positions
in the text. None of the 270 answers across the three eval runs contains a
marker. Option (a) changes generation, so the 69% end-to-end and 95% citation
figures would no longer describe the running system without a re-run. Option (b)
puts a marker on a sentence its quote may not support, which is a quiet
fabrication in a project whose point is catching those. Markers under the answer
promise only what is true: "this answer drew on these quotes".

### 2. Scores come from `retrieval.search(with_scores=True)`
**Chose:** an optional flag on `search()` that attaches the numbers it already
computes (dense rank, BM25 rank, RRF score, cross-encoder score) to each chunk.
**Rejected:** (a) rebuilding the fetch, fuse and rerank sequence inside the API to
capture scores; (b) showing ranks only.
**Why:** (a) creates a second copy of the ranking path, which is exactly the drift
`space/sync.py` exists to prevent. (b) would hide the cross-encoder score, and the
reranker is the project's main result. The flag is off by default, and two checks
back up "changes nothing": a unit test (`test_asking_for_scores_does_not_change_the_ranking`)
and a one-off run over 25 real questions against the committed version (25/25 identical top-8s).

### 3. Channel positions are reported as ranks, not raw similarity
**Chose:** `dense_rank` and `bm25_rank` (the position in each channel's top 50, or
null if the chunk was not there).
**Rejected:** raw cosine and BM25 scores.
**Why:** it is the same reason the pipeline uses RRF. The two raw scores live on
unrelated scales, so putting them side by side invites a comparison that means
nothing. Ranks are what fusion actually reads. A `null` is informative too: it
shows a chunk that only one channel found, which is the whole case for having two.

### 4. The dense channel runs in numpy, not Qdrant Cloud
**Chose:** a dot product against `embeddings.npy` (14 MB), as `space/app.py` does.
**Rejected:** Qdrant Cloud's free tier.
**Why:** `retrieval_lab.py baseline` shows the numpy path returns the same results
as Qdrant question for question, so nothing measured changes. Qdrant brings a
second service, a second secret, a network hop per question, and a free cluster
that is suspended when idle: a second cold start stacked on the host's. The
entity filter is the one thing Qdrant was chosen for, and `retrieval.entity_mask`
applies it identically in numpy. At 9,278 vectors, exact search takes milliseconds.

### 5. Verification is reported as `status` + `found_in`, with `verified` derived
**Chose:** `status: ok | wrong_source | not_found`, the list of sources that
contain the quote, and `verified` (= `status == "ok"`) as a convenience.
**Rejected:** a single `verified: bool`.
**Why:** a boolean collapses three different outcomes, and it hides the case the
README calls unfalsifiable: a quote found in several sibling circulars verifies
against all of them, so `ok` cannot confirm which one the model meant. With
`found_in`, the UI can say so instead of showing a green tick that implies more
than it proves.

### 6. Highlight offsets are computed in the API, not added to `verify()`
**Chose:** `api/highlight.py` redoes verify()'s normalisation (collapse
whitespace, lowercase) while keeping a map from each normalised character back to
the original. A match maps straight back to offsets in the original text.
**Rejected:** changing `verify()` to return offsets.
**Why:** `verify()` is what the citation-accuracy metric runs on, and it only needs
yes or no. The mapping is a display concern. A test asserts that the rebuilt
normalisation agrees with `rag.norm`, so a highlight cannot disagree with the
verification badge beside it. Where the highlight goes: the cited source if the
quote verified, the source that really contains it if it was misattributed, and
nowhere if it is in no source.

### 7. The API calls the pipeline's stages, not `rag.run_one()`
**Chose:** `detect_entity`, then `search`, `generate` and `verify`, called
individually.
**Rejected:** `run_one()`.
**Why:** `run_one()` prints to a terminal and appends every question to
`data/log.jsonl`. A public server should not do either. In particular, it should
not keep a permanent record of strangers' questions without saying so.

### 8. Deploying at all, and a rate-limit design shaped by the Groq free tier
**Context:** the README previously said the demo was deliberately not deployed,
because about 70 answers a day of free-tier allowance would be drained by traffic.
**Chose:** deploy, with two kinds of limit:
- **Requests** (5 per IP per minute) cost CPU. Over the limit: HTTP 429 with
  `Retry-After`.
- **Generations** (20 per IP per day; 3 a minute and 60 a day globally) cost the
  Groq allowance. Over any of these, the request still succeeds with
  **retrieval only** (`generation: "rate_limited"`) and a plain-English notice.

**Rejected:** a flat 429 for everything; no global cap.
**Why:** retrieval is the half of the project a visitor can still learn from
(passages, scores, reranking), and it costs nothing from the quota. The global
per-minute cap also keeps `rag._throttle()` from sleeping for up to a minute
inside a request; that pacing suits a batch run but would look like a hang in a
browser. The generation budgets are checked all-or-nothing, so a request refused
by the global cap does not also use up that visitor's daily allowance.

### 9. A hand-written, in-memory limiter
**Chose:** about 60 lines of rolling windows (`api/limits.py`).
**Rejected:** slowapi (a library) and Redis.
**Why:** the server runs one process, so there is exactly one set of counters. A
restart resets them, which errs in the visitor's favour. Redis would add a service
to solve a multi-process problem this deployment does not have. The limiter is
also small enough to explain line by line, and its rules (two budgets,
all-or-nothing) do not fit a decorator-per-route library cleanly.

### 10. The visitor's IP is read from the right of X-Forwarded-For
**Chose:** a `FORWARDED_HOPS` setting. The IP is the entry that many places from
the right, and the header is ignored when the setting is 0.
**Rejected:** trusting the first entry, or ignoring the header.
**Why:** behind a host's proxy every request appears to come from the proxy, so
ignoring the header would give all visitors one shared budget. But anyone can
write anything at the left of that header, and trusting it would let a visitor
reset their limit on every request. Only the entries our own proxies appended can
be believed. A test covers the spoofing case.

### 11. Plain `def` endpoints, one worker, a lock around retrieval
**Chose:** synchronous endpoints (FastAPI runs them in a thread pool), a single
server process, and a lock so one retrieval runs at a time. Generation, which
spends its time waiting on the network, runs outside the lock.
**Rejected:** `async def` endpoints; several workers.
**Why:** everything the endpoint calls blocks: torch inference and a synchronous
Groq client. Inside `async def`, that code would freeze the event loop for every
visitor. The pipeline's module-level caches were written for a single-threaded
CLI, and several worker processes would each load both models (about 1 GB each)
on a free-tier machine. With a 3-a-minute generation cap, one worker is not the
bottleneck.

### 12. Models load at startup, and a failed load is not fatal
**Chose:** load everything in FastAPI's `lifespan` hook. If loading fails, the
server still starts, `/health` reports `"starting"`, and `/ask` returns 503.
**Rejected:** lazy loading on the first request; crashing on failure.
**Why:** lazy loading makes the first visitor wait about 30 seconds, and on a demo
that is usually the person you most wanted to impress. A server that is up and
says what is wrong is easier to diagnose on a hosted platform than one that died
before logging anything.

### 13. Retrieval failure is an error; generation failure is not
**Chose:** 503 if retrieval fails. If generation fails (quota, outage, bad
output), return 200 with `generation: "unavailable"` and the sources.
**Why:** without retrieval there is nothing to show. Without generation, the
passages are still the evidence an answer would have been written from.

### 14. Citations point at sources by number instead of copying the text
**Chose:** `citation.source` and `highlight.source` are indexes into `sources[]`.
**Rejected:** embedding the chunk text in each citation.
**Why:** several citations often share one source (5 of 5 did in the first live
test), and the side panel and the retrieval view show the same passages. There is
one copy, and every view looks it up.

### 15. Fields left out on purpose
- **No k / filter / rerank toggles in the API.** They are in the local Gradio app
  for exploring. On a public endpoint they multiply what can be requested without
  adding to what a first-time visitor needs to see.
- **No `other_entity` flag on sources** (it was in the first draft of the contract). The API
  always runs with the entity filter on, and the filter admits only matching or
  unscoped passages, so the flag would always be false. A field that cannot vary
  is noise in the contract.

### 16. `create_app()` factory, so tests inject fakes
**Chose:** `create_app(pipeline, limiter, origins, forwarded_hops)`. Tests pass a
fake pipeline and a small limiter; the server passes nothing and gets the real
ones, configured from the environment.
**Rejected:** FastAPI's `dependency_overrides`, or patching module globals.
**Why:** the dependencies are visible in one function signature, which is plain
constructor injection (the Spring habit). The fake replaces only `retrieve` and
`generate`, the two stages that need a model or a key, so response building,
highlighting and the fallbacks are tested as real code.
