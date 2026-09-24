"""
RegRAG — Hugging Face Space
---------------------------
A demo of the pipeline described in the repo README, running on free CPU.

WHAT IS DIFFERENT HERE, AND WHAT IS DELIBERATELY NOT
The dense channel is a numpy dot product against `embeddings.npy` rather than a
Qdrant query. Nothing else changes: BM25, the RRF fusion, the entity mask and the
cross-encoder are imported from `retrieval.py`, the same module the pipeline runs,
and generation and citation verification come from `rag.py`. That substitution is
safe because it is measured, not assumed -- `retrieval_lab.py baseline` scores the
numpy path against the saved Qdrant run and they agree question-for-question, on
both the current and the previous configuration.

It also means this Space needs no Docker, no vector server and no cold-start
container: two files (27MB) and two small models.

WHY THE DEMO STILL WORKS WITHOUT AN API KEY
Generation runs on a free-tier Groq allowance that a public demo can exhaust in an
afternoon. When the key is missing or the allowance is gone, the app falls back to
showing retrieval alone rather than an error page. A retrieval-only demo still
shows the thing this project is actually about -- which passages come back, which
institution class they bind, and what reranking did to the ordering -- and a demo
that is honestly degraded beats one that is down.

The rerank toggle is not a setting anyone needs. It is there because the central
result of the project (chunk recall 60% -> 78%) is otherwise a table in a README,
and a reader can flip it and watch the ordering change on their own question.
"""

import os
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rag                      # noqa: E402  -- generation + citation verification
import retrieval as R           # noqa: E402  -- BM25, RRF, entity mask, reranker

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
VECS = Path("data/embeddings.npy")

EXAMPLES = [
    "Who can be an Internal Ombudsman?",
    "What are the KYC requirements for Payments Banks?",
    "Which institutions are covered by the term 'All India Financial Institutions'?",
    "What conduct rules apply to recovery agents engaged by Housing Finance Companies?",
    "When is a loan account treated as out of order?",
    "What is the RBI repo rate?",          # in the corpus: answered, with a verified citation
    "What time do banks close for lunch?",  # not in the corpus -- should decline
]

_STATE = {}


def boot():
    """Load the corpus, the vectors and the query encoder once, at startup.

    Gradio would otherwise pay this on the first request and time it out. The
    cross-encoder and the BM25 index are built lazily inside retrieval.py on
    first use, which is fine -- they cost a couple of seconds, not thirty.
    """
    if "vecs" in _STATE:
        return
    from sentence_transformers import SentenceTransformer

    _STATE["chunks"] = R.corpus()
    _STATE["vecs"] = np.load(VECS)
    _STATE["encoder"] = SentenceTransformer("BAAI/bge-small-en-v1.5")

    import importlib.util

    spec = importlib.util.spec_from_file_location("rbi_scraper", "rbi_scraper.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _STATE["scan"] = mod.scan_entities


def dense_ids(question, entities, use_filter, depth):
    """The dense channel, as a dot product. Qdrant's job, done in numpy."""
    qv = _STATE["encoder"].encode(QUERY_PREFIX + question,
                                  normalize_embeddings=True).astype("float32")
    mask = R.entity_mask(_STATE["chunks"], entities) if use_filter else None
    return R.top_k(_STATE["vecs"] @ qv, mask, depth)


def has_key():
    return bool(os.getenv("GROQ_API_KEY"))


def sources_markdown(chunks, entities):
    rows = []
    for i, c in enumerate(chunks, 1):
        binds = ", ".join(c.get("applies_to") or []) or "_(binds no particular class)_"
        # Flag any passage bound to a class the question did not name. In a corpus
        # that is 74% near-duplicates this is the failure worth seeing, and it is
        # invisible if the sources are just listed.
        off = ""
        if entities and c.get("applies_to") and not (set(entities) & set(c["applies_to"])):
            off = "  ⚠️ **other entity class**"
        rows.append(
            f"**[{i}] {c['number']}** · {c.get('issue_date') or '—'}{off}\n\n"
            f"- binds: {binds}\n"
            f"- document: {c.get('subject') or '—'}\n"
            f"- section: `{c.get('section') or '—'}`\n\n"
            f"> {c['text'][:600].strip()}{'…' if len(c['text']) > 600 else ''}\n"
        )
    return "\n---\n".join(rows)


def citations_markdown(checks):
    if not checks:
        return "_No citations returned._"
    mark = {"ok": "✅ verified", "wrong_source": "⚠️ wrong source", "not_found": "❌ not found"}
    out = []
    for c in checks:
        line = f"{mark[c['status']]} · **[{c['source_id']}]** {c['cited_doc'] or '—'}"
        if c["status"] == "ok" and len(c.get("found_in") or []) > 1:
            # The unfalsifiable case: when ten sibling circulars carry one sentence,
            # "verified" cannot mean the attribution was checked. Say so rather than
            # letting a green tick imply more than it proves.
            line += (f"  \n  _quote also appears in sources {c['found_in']} — "
                     f"verification cannot confirm attribution here_")
        out.append(f"{line}\n\n  > {c['quote'][:220]}")
    return "\n\n".join(out)


def ask(question, k, use_filter, use_rerank):
    question = (question or "").strip()
    if not question:
        return "Ask something.", "", "", ""
    boot()

    t0 = time.time()
    entities = _STATE["scan"](question) or None
    ids = dense_ids(question, entities, use_filter, R.DEPTH)
    chunks = R.search(question, k=int(k), entity=entities, use_filter=use_filter,
                      dense_ids=ids, do_rerank=use_rerank)
    retrieval_s = time.time() - t0

    detected = ", ".join(entities) if entities else "none detected"
    meta = (f"**entity:** {detected} · **filter:** {'on' if use_filter else 'off'} · "
            f"**rerank:** {'on' if use_rerank else 'off'} · **k:** {int(k)} · "
            f"retrieval {retrieval_s:.1f}s")

    if not has_key():
        answer = (
            "### Retrieval only\n\n"
            "No `GROQ_API_KEY` is configured on this Space, so generation is off and "
            "the retrieved passages are shown below as-is.\n\n"
            "This is still the interesting half: the passages are what the retriever "
            "found, the `binds` line is the entity class each one governs, and toggling "
            "**rerank** re-runs the same query through the cross-encoder so you can "
            "watch the ordering change."
        )
        return answer, meta, "_Generation disabled._", sources_markdown(chunks, entities)

    try:
        ans, secs, _usage, _prompt = rag.generate(question, chunks)
    except Exception as e:
        # Almost always the free-tier daily allowance. Say which, rather than
        # surfacing a stack trace to someone who just wanted to try a question.
        msg = str(e)
        daily = "per day" in msg.lower() or "rate" in msg.lower()
        note = ("The free-tier daily token allowance is exhausted — try again tomorrow, "
                "or clone the Space and add your own key."
                if daily else f"`{e.__class__.__name__}: {msg[:200]}`")
        return (f"### Generation unavailable\n\n{note}\n\nRetrieval still ran; "
                f"the passages are below."), meta, "_Generation failed._", \
               sources_markdown(chunks, entities)

    checks = rag.verify(ans, chunks)
    verified = sum(1 for c in checks if c["status"] == "ok")
    head = ("> **The sources do not answer this.** The pipeline set "
            "`insufficient_context`.\n\n" if ans.insufficient_context else "")
    meta += f" · generation {secs:.1f}s · **{verified}/{len(checks)} citations verified**"
    return head + ans.answer, meta, citations_markdown(checks), \
        sources_markdown(chunks, entities)


INTRO = """
# RegRAG — question answering over 506 RBI circulars

Ask about Indian banking regulation. Every answer is grounded in retrieved
circulars, and each quote is **checked against the source it claims** — a citation
that cannot be found is shown as failed rather than quietly displayed.

**Why this corpus is hard:** RBI reissues the same rule once per class of regulated
entity, so a KYC amendment exists as ten near-identical circulars. **74% of the
corpus sits in a near-duplicate family.** Retrieval that ignores which entity a
document binds returns the wrong institution's rules most of the time — so every
chunk carries an `applies_to` tag and the filter runs *inside* the search.

**Try turning off `rerank`** and asking the same question again. Chunk-level
retrieval recall is 78% with it and 60% without; the ordering below is where that
difference lives.
"""


with gr.Blocks(title="RegRAG") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        q = gr.Textbox(label="Question", placeholder="Who can be an Internal Ombudsman?",
                       scale=4, autofocus=True)
        go = gr.Button("Ask", variant="primary", scale=1)
    with gr.Row():
        k = gr.Slider(3, 12, value=8, step=1, label="passages retrieved (k)")
        f = gr.Checkbox(value=True, label="entity filter")
        rr = gr.Checkbox(value=True, label="rerank (cross-encoder)")
    gr.Examples(EXAMPLES, inputs=q, label="Examples — the last one is not in the corpus")

    meta = gr.Markdown()
    answer = gr.Markdown(label="Answer")
    with gr.Accordion("Citation checks", open=True):
        cites = gr.Markdown()
    with gr.Accordion("Retrieved passages", open=False):
        srcs = gr.Markdown()

    gr.Markdown(
        "---\nBuilt from [the RegRAG repo](https://github.com/BiplawSingh/RegRAG). "
        "Retrieval, fusion and reranking are imported from the same modules the "
        "evaluated pipeline runs; only the dense channel differs (numpy instead of "
        "Qdrant), and that substitution is verified question-for-question by "
        "`retrieval_lab.py baseline`."
    )

    for trigger in (go.click, q.submit):
        trigger(ask, inputs=[q, k, f, rr], outputs=[answer, meta, cites, srcs])


# Warm the caches while the Space is still starting up, not on the first visitor's
# click. Cold, the first question pays for two model downloads and a 27MB load --
# about a minute of a blank screen for whoever arrives first, which on a demo is
# usually the person you most wanted to impress.
#
# Guarded, and deliberately not fatal: if the warm-up fails the Space should still
# come up and let `ask` retry and report the error properly, rather than dying at
# import with a stack trace no visitor can see.
try:
    boot()
    R.bm25()          # ~2s to build; do it here rather than mid-question
    R.get_reranker()  # downloads once, then cached in the container
except Exception as exc:                                  # noqa: BLE001
    print(f"warm-up failed ({exc.__class__.__name__}: {exc}); "
          f"falling back to loading on first request")

if __name__ == "__main__":
    # theme goes to launch(), not to Blocks(): Gradio 6 moved it, and passing it to
    # the constructor warns on 6 and would break the layout on a future major.
    demo.launch(theme=gr.themes.Soft())
