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

---

## The frontend

### 17. One `status` value for the page, not separate loading, error and data flags
**Chose:** a single piece of state that is exactly one of `idle`, `loading`,
`error` (with a message) or `done` (with a result). In TypeScript this is a
"discriminated union", the same idea as a Java `sealed interface` with one
record per case.
**Rejected:** `isLoading`, `error` and `data` as three independent pieces of state.
**Why:** three independent values allow combinations that should never happen, such
as an error and a stale answer shown together, or a spinner over a finished result.
Every place that sets them has to remember to reset the other two. With one value,
the impossible combinations cannot be written, and the compiler checks that a
result is only read in the `done` state.

### 18. Fetch in the click handler, with no data-fetching library
**Chose:** `askQuestion()` is called directly from the form's submit handler.
**Rejected:** `useEffect` that fetches when the question changes; React Query / SWR.
**Why:** a question is asked because someone pressed a button, not because a value
changed, so the handler is where the request belongs. An effect would fire on every
keystroke, or need extra flags to stop it. React Query earns its place with caching,
background refetching and many endpoints; this app has one POST that should never
be cached or refetched.

### 19. The API address is a build-time setting (`VITE_API_URL`), not a dev proxy
**Chose:** the frontend calls `VITE_API_URL` directly in development and production.
**Rejected:** Vite's dev-server proxy, which hides cross-origin calls locally.
**Why:** in production the frontend (Vercel) and the API (another host) are on
different origins, so CORS must work. Calling the API directly in development means
CORS is exercised on every local run instead of being discovered broken at deploy
time. Anything prefixed `VITE_` is compiled into the public JavaScript, so this
variable holds only a URL and never a key.

### 20. The answer is rendered as plain text, not Markdown
**Chose:** plain text with CSS `whitespace-pre-line`, which keeps the model's line
breaks.
**Rejected:** a Markdown renderer.
**Why:** measured, not assumed: of 19 logged answers, none used bold, bullets or
tables, and one used a numbered list laid out with line breaks, which
`pre-line` already displays correctly. A Markdown library adds a dependency, and
rendering model output as HTML adds a path for injected markup. React escapes
plain text automatically.

### 21. The question text lives in `App`, not inside the form
**Chose:** a "controlled" form. `App` holds the text and passes it down with a
setter, a pattern React calls "lifting state up".
**Why:** the example-question chips (next phase) have to fill the box and ask. If
the form kept its own text, nothing outside it could change that text.

### 22. The Ask button is disabled while a request is in flight
**Chose:** disable the button while loading.
**Rejected:** cancelling the previous request with `AbortController` when a new one
starts.
**Why:** each request costs Groq allowance, so a second request fired while the
first is running is exactly what the rate limits exist to discourage. Disabling the
button makes the out-of-order-response problem impossible instead of handling it.

### 23. A 90-second timeout, and error messages written for visitors
**Chose:** `AbortSignal.timeout(90s)`. Every failure becomes one sentence that says
what happened and what to do, and every error state has a Try again button.
**Why:** a free host that has gone to sleep can take close to a minute to wake up,
so a short timeout would fail exactly the first visitor. A request with no timeout
at all can spin forever. "Failed to fetch" means nothing to a recruiter; "the free
host sleeps when idle, try again shortly" tells them it isn't broken.

### 24. `[n]` means a citation; retrieved chunks are "passage n"
**Chose:** square brackets only for citation markers; passages are named in words.
**Why:** citation [2] often cites passage 1. The first build wrote both as `[n]`, and
the preview for citation [2] read "also appears in sources [2], [3]…", which looks
self-referential. Two numbering systems need two notations.

### 25. Verification is described in one module, with its own colour for "ambiguous"
**Chose:** `verification.ts` turns a citation into a tone, an icon and one plain
sentence. Everything that shows a verification result uses it. A verified quote that
also appears in sibling passages gets its own colour (blue), not green.
**Why:** the marker, the side panel and the retrieval details must never describe the
same result differently. A green tick on a quote that five circulars
share would claim more than the check proved, which is the README's
unfalsifiable-citation finding made visible.

### 26. The source panel slides over the page; it doesn't split the layout
**Chose:** a panel that slides in from the right over a dimmed page, full width on
a phone. Escape, the ✕ or a click outside closes it.
**Rejected:** expanding the passage inline under the answer; a permanent
two-column layout.
**Why:** passages run up to about 2,300 characters, and reading one needs the
room. Inline expansion pushes the answer and the other markers off screen. Two
columns squeeze the answer on a laptop and don't fit on a phone at all. An
overlay gives the passage the whole height on any screen. Because it covers the
page, it behaves like a dialog: focus moves into it, the page behind stops
scrolling, and focus returns to the marker that opened it, so keyboard and
screen-reader users don't get lost.

### 27. The panel shows the passage where the quote actually is
**Chose:** a misattributed quote opens the passage that really contains it,
highlighted, with a note saying which passage was cited. A quote found nowhere
opens the passage it cited, unhighlighted, and prints the quote in the note.
**Why:** a highlight that wasn't in the text would be a small fabrication of its
own. Showing where the quote really is makes the verification result checkable
by eye rather than taken on trust.

### 28. The retrieval details use the browser's `<details>` element
**Chose:** native `<details>`/`<summary>`.
**Rejected:** a `useState` open/closed flag and a custom button.
**Why:** the browser already provides a collapsible section that is
keyboard-accessible and announced correctly by screen readers, with no state to
manage. It opens automatically when there is no answer (quota used up, model
unreachable), because then the passages are the only content. Each new answer
gets a fresh section via React's `key`, so it doesn't inherit the previous
answer's open or closed state.

### 29. The retrieval view states only what it can compute
**Chose:** each passage shows its rank in both searches, its fused score, its
rerank score, and whether one or both searches found it. Step 1 counts how many
classes of institution the passages span.
**Why:** it lets a visitor see the README's findings on their own question: a
passage the reranker took from #18 to #1, passages only one search found, and an
unscoped question pulling in near-identical copies of one rule. The wording
reports counts ("circulars binding 5 different classes") rather than asserting
the copies are identical, because the UI can count classes but can't prove the
text matches.

### 30. The example questions were chosen by running them
**Chose:** four questions, each labelled with the behaviour it shows, all run
against the live API first.
**Rejected:** "What is the RBI repo rate?" as the refusal example, although
`space/app.py` had labelled it "not in the corpus -- should decline" (since
corrected).
**Why:** run live, it was answered ("5.25 per cent") with a verified citation.
The corpus does contain it. A chip promising a refusal and delivering an answer
would undermine the one claim the demo makes, so the refusal example is an
off-topic question ("What time do banks close for lunch?"), which the pipeline
declines, consistent with the README's finding that off-topic refusal works.
After the first answer the chips shrink to one line, so they stay available
without pushing the answer down.

### 31. `useCallback` for the panel's close handler, and nowhere else
**Chose:** `closePanel` is wrapped in `useCallback`.
**Why:** a function defined inside a component is a new object on every render.
The panel re-registers its Escape-key listener whenever its `onClose` prop
changes, so an unwrapped handler would remove and re-add the listener on every
render. `useCallback` keeps the same function between renders. Nothing else is
wrapped: elsewhere a new function per render costs nothing measurable, and
wrapping everything "just in case" is exactly the over-engineering this app avoids.

### 32. The About figures are recomputed, and the 35% → 92% figure is not used
**Chose:** six figures, each checked against the repo's own tools before use
(`judge.py report`, `data/eval/retrieval_k8_filtered*.json`,
`data/eval/answers_e2e_k8.jsonl`, `data/eval/entity_filter_k8.json`). The
component lists its sources in a comment.
**Rejected:** the "entity precision 35% → 92%" headline.
**Why:** it came from 8 hand-picked queries using dense search only, and the
README says it is not comparable to the main measurement. The entity figure used
instead (35% wrong-institution passages even after reranking, across 36
questions) is the one the README defends. Recomputing, not copying, also caught
a trap: the judged-answers file is append-only, so it holds more rows than
questions, and counting rows naively gives 52/73 instead of the correct 48/69.
`judge.py report` removes the duplicates, so its number is the one quoted.

### 33. The weakest result is on the page too
**Chose:** a tile saying only 3 of 12 plausible-but-unanswerable questions are
refused.
**Why:** the README calls this the project's weakest result. Every other tile
is a success, and a page of only good numbers reads as marketing. Reporting the
failure next to them is what makes the successes believable. It's also the most
interesting thing to discuss in an interview.

### 34. Empty and degraded states say what happened in plain words
**Chose:** a message for each case: an answer with no citations ("none of it
could be checked, treat it with caution"), no retrieved passages, generation
switched off or rate-limited, and the model declining to answer.
**Why:** an answer with no citations shows no red markers, and that absence
reads as "all checks passed" when nothing was checked. Each state gets a
sentence rather than an empty box, so a visitor never has to guess whether the
page is broken.

### 35. Mobile: bigger touch targets, and example chips scroll sideways
**Chose:** citation markers are at least 36px tall on phones (26px from the `sm`
breakpoint up). After the first answer, the example chips become one row that
scrolls sideways on phones and wraps on wider screens. Tested at 320px, 390px and
1280px with no sideways scrolling of the page.
**Why:** on a phone the four chips wrapped into four rows and pushed the answer
below the fold. A small marker is fine for a mouse but is easy to miss with a
finger.

### 36. Link previews, no preview image
**Chose:** a title and description in Open Graph meta tags, and a custom favicon.
**Rejected:** an `og:image` for now.
**Why:** the link will be pasted into messages to recruiters, and a preview with a
real title beats "frontend". An `og:image` needs an absolute URL on the deployed
site, so it waits until there is one.

---

## Deployment

### 37. Hugging Face Spaces for the API, Vercel for the frontend
**Chose:** a Docker Space on Hugging Face's free CPU tier (16 GB RAM, 2 vCPU), and
Vercel for the static frontend.
**Rejected:** Render's free tier (512 MB, too small for torch plus two models);
Fly.io (no free allowance for new accounts, though it is the right upgrade if a
warm, always-fast demo becomes worth a few dollars a month); one host for both.
**Why:** memory decided it before anything else was compared. The API needs about
1–1.5 GB, and only the Space offers that for free. The cost is a sleep after
about 48 idle hours and a slow first request after it. The frontend is plain
files, and a CDN serves those better than any Python host. DEPLOY.md has the full
comparison.

### 38. The Space is assembled by the existing sync script, and uploaded rather than pushed
**Chose:** `space/sync.py --target api` copies the pipeline, the API and the data
into `api_space/` beside a hand-written Dockerfile. `hf upload` sends the folder.
**Rejected:** a second, hand-maintained copy of the modules; `git init` inside
`api_space/`.
**Why:** the sync script already exists to stop a demo drifting from the measured
pipeline, and `--check` fails loudly if it has. Extending it keeps one mechanism
for both Spaces. A git repo nested inside this one confuses the outer repo, and
`hf upload` handles the two large data files without it.

### 39. CPU-only PyTorch, with the models built into the image
**Chose:** install torch from PyTorch's CPU index, and download both models
during `docker build`.
**Why:** the default torch wheel bundles CUDA libraries for a GPU the Space
doesn't have, which makes the image several gigabytes larger for nothing. Models
fetched at build time are loaded from disk when the Space wakes, instead of
downloaded while the first visitor waits.

### 40. The proxy setting is measured after deploying, not assumed
**Chose:** an endpoint that is off by default (`DEBUG_FORWARDING=1`). It shows
the caller their own request as the server sees it, so `FORWARDED_HOPS` can be
checked once and the endpoint switched off again.
**Why:** how many proxies Hugging Face puts in front of a Space is not something to
guess. Too few trusted, and every visitor shares one rate-limit budget. Too many,
and anyone can reset theirs with a fake header. The endpoint only ever shows the
caller's own addresses, and it stays off after the check.

### 41. The Space is public
**Why:** a private Space rejects requests that don't carry a Hugging Face login,
so a visitor's browser could not call it. Everything in the Space is already in
the public GitHub repo or is the RBI's own published text. The Groq key is a
Space secret, never a file.
