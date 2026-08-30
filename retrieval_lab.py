"""
retrieval_lab.py
----------------
An offline bench for retrieval variants, scored on the one metric that matters:
did the chunk that answers the question actually arrive?

    python retrieval_lab.py baseline          # reproduce eval.py's number in numpy
    python retrieval_lab.py compare           # every variant, side by side
    python retrieval_lab.py inspect q017      # why one question misses

WHY THIS EXISTS SEPARATELY FROM eval.py
`eval.py retrieval` measures the pipeline as deployed: Qdrant, the entity filter,
whatever is indexed right now. That is the number to publish, and it is the wrong
number to iterate against -- changing the ranking function means re-indexing 9,278
points before you can see whether the idea was any good, which is a ten-minute
round trip on a question you should be able to answer in ten seconds.

So this file reimplements the *same* scoring against the raw arrays: embeddings.npy
for dense, an in-process BM25 for lexical, and a numpy mask for the entity filter.
`baseline` exists to prove that reimplementation is faithful -- it must reproduce
eval.py's chunk-rate exactly, or nothing measured here transfers. A bench that
quietly diverges from the pipeline is worse than no bench, because every number it
prints is about a system nobody runs.

WHAT IS BEING MEASURED
Chunk recall@k, against `gold_chunk_ids`. Not document hit-rate: the pipeline
consumes eight passages, and a document scores as "retrieved" the moment any of its
chunks comes back, including one that does not answer the question. The gap between
those two figures is 16 points, and the whole of the work this bench supports lives
inside it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The ranking machinery is imported, never reimplemented. A bench that carries its
# own copy of BM25 or its own fusion constant measures a system nobody runs, and
# the divergence is silent -- the numbers still look like numbers. Only the dense
# channel differs here (a numpy dot product instead of a Qdrant round-trip), which
# `baseline` exists to prove is not a difference that matters.
import retrieval as R
from retrieval import BM25, header_text, rrf, top_k  # noqa: F401  -- re-exported

ROOT = Path("data")
CHUNKS = ROOT / "chunks.jsonl"
VECS = ROOT / "embeddings.npy"
QA = ROOT / "qa.jsonl"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_K = 8


# --------------------------------- corpus ------------------------------------
def load():
    for p in (CHUNKS, VECS, QA):
        if not p.exists():
            sys.exit(f"missing {p}")
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    vecs = np.load(VECS)
    if len(vecs) != len(chunks):
        sys.exit(f"embeddings ({len(vecs)}) and chunks ({len(chunks)}) disagree -- rebuild")
    qa = [json.loads(l) for l in open(QA, encoding="utf-8")]
    # Only answerable questions with a gold chunk pointer can be scored here.
    # Unanswerable ones have nothing to find, and scoring them would silently
    # reward retrieving nothing.
    scored = [q for q in qa if q["answerable"] and q.get("gold_chunk_ids")]
    return chunks, vecs, scored


_MODEL = {}


def embed_queries(texts):
    """Encode with the same model and the same query prefix as the pipeline.

    BGE v1.5 is asymmetric: queries carry the instruction prefix, passages do not.
    Dropping the prefix here would make every variant look slightly worse for a
    reason that has nothing to do with the variant.
    """
    if "m" not in _MODEL:
        import os

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer

        _MODEL["m"] = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL["m"].encode([QUERY_PREFIX + t for t in texts],
                              normalize_embeddings=True).astype("float32")


def detect_entities(question):
    """The pipeline's own entity detector, so the filter is identical."""
    if "scan" not in _MODEL:
        import importlib.util

        spec = importlib.util.spec_from_file_location("rbi_scraper", "rbi_scraper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODEL["scan"] = mod.scan_entities
    return _MODEL["scan"](question) or None


entity_mask = R.entity_mask       # same mask the pipeline builds, same OR-empty clause


# ------------------------------- variants ---------------------------------------
# Each variant is a function (ctx, question_index, k) -> list of chunk ids, so a new
# idea is one function and one registry line rather than a fork of the harness.

class Ctx:
    def __init__(self, chunks, vecs, qa, alt_vecs=None):
        self.chunks, self.vecs, self.qa = chunks, vecs, qa
        self.alt_vecs = alt_vecs
        self.qvecs = embed_queries([q["question"] for q in qa])
        self.masks = [entity_mask(chunks, detect_entities(q["question"])) for q in qa]
        self.bm25 = None

    def need_bm25(self):
        if self.bm25 is None:
            print("  building BM25 index ...", flush=True)
            self.bm25 = BM25([c["text"] for c in self.chunks])
        return self.bm25


def v_dense(ctx, i, k):
    return top_k(ctx.vecs @ ctx.qvecs[i], ctx.masks[i], k)


def v_bm25(ctx, i, k):
    return top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], k)


def v_hybrid(ctx, i, k, depth=50):
    """RRF over the dense and lexical rankings, each taken to `depth` first.

    Fusing full-corpus rankings would let a chunk that is 4,000th on both lists
    outrank one that is 9th on a single list, which is not what either channel is
    saying. Truncating to a depth well past k keeps the fusion to candidates at
    least one channel actually likes.
    """
    d = top_k(ctx.vecs @ ctx.qvecs[i], ctx.masks[i], depth)
    l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], depth)
    return rrf([d, l])[:k]


def v_dense_ctx(ctx, i, k):
    """Dense, but over embeddings that include the document/section header.

    See `python retrieval_lab.py build-context-vectors`. Falls back loudly rather
    than silently scoring the baseline twice.
    """
    if ctx.alt_vecs is None:
        raise SystemExit("no data/embeddings_ctx.npy -- run: "
                         "python retrieval_lab.py build-context-vectors")
    return top_k(ctx.alt_vecs @ ctx.qvecs[i], ctx.masks[i], k)


def v_hybrid_ctx(ctx, i, k, depth=50):
    if ctx.alt_vecs is None:
        raise SystemExit("no data/embeddings_ctx.npy -- run: "
                         "python retrieval_lab.py build-context-vectors")
    d = top_k(ctx.alt_vecs @ ctx.qvecs[i], ctx.masks[i], depth)
    l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], depth)
    return rrf([d, l])[:k]


def _neighbours(ctx, ranked, k, top=4):
    """Spend some of the k slots on the passages either side of the best hits.

    Measured motivation, not a hunch: of the 11 questions that retrieve the right
    document and miss the passage, 5 have a gold chunk within two positions of
    something already returned. Those are not retrieval failures in any interesting
    sense -- the search landed on the right paragraph's neighbour, which is what a
    fixed-size chunker does to a rule that spans a boundary.

    The cost is real and has to be stated: every slot given to a neighbour is a
    slot not given to the next-best independent hit, so this trades breadth for
    depth. Whether that is a good trade is exactly what the bench measures.
    """
    by_doc = {}
    for j, c in enumerate(ctx.chunks):
        by_doc.setdefault(c["doc_key"], {})[c["chunk_index"]] = j

    out = []
    for j in ranked[:top]:
        c = ctx.chunks[j]
        sib = by_doc[c["doc_key"]]
        for pos in (c["chunk_index"], c["chunk_index"] - 1, c["chunk_index"] + 1):
            n = sib.get(pos)
            if n is not None and n not in out:
                out.append(n)
    for j in ranked:                       # backfill from the original ranking
        if j not in out:
            out.append(j)
    return out[:k]


def v_dense_nbr(ctx, i, k):
    return _neighbours(ctx, top_k(ctx.vecs @ ctx.qvecs[i], ctx.masks[i], k * 3), k)


def v_hybrid_nbr(ctx, i, k, depth=50):
    d = top_k(ctx.vecs @ ctx.qvecs[i], ctx.masks[i], depth)
    l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], depth)
    return _neighbours(ctx, rrf([d, l]), k)


def v_hybrid_ctx_nbr(ctx, i, k, depth=50):
    if ctx.alt_vecs is None:
        raise SystemExit("no data/embeddings_ctx.npy -- run: "
                         "python retrieval_lab.py build-context-vectors")
    d = top_k(ctx.alt_vecs @ ctx.qvecs[i], ctx.masks[i], depth)
    l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], depth)
    return _neighbours(ctx, rrf([d, l]), k)


def _rerank(ctx, i, candidates, k, with_header=False, model=None):
    """Re-sort a wide candidate set with a cross-encoder, keep the best k.

    The bi-encoder that built the index has to compress a passage into 384
    numbers before it has ever seen the question. A cross-encoder reads both at
    once, which is why it can tell "the paragraph that states the rule" from "a
    paragraph in the document that states the rule" -- the exact distinction the
    chunk-vs-document gap is made of. It is also ~50x more expensive per pair,
    which is why it runs over 50 candidates and not over 9,278.

    The ceiling is fixed by the candidate set, not by the reranker: whatever the
    fetch misses, no re-sorting recovers. Measured at depth 50, dense and BM25
    between them surface the answering chunk for 61 of 69 questions, so that is
    what this can reach at best.
    """
    return R.rerank(ctx.qa[i]["question"], candidates, k,
                    with_header=with_header, model=model)


def _candidates(ctx, i, depth, vecs=None):
    """The pool a reranker re-sorts: dense and lexical, fused, then truncated.

    Fused rather than concatenated so the truncation keeps what both channels
    like. The entity filter has already run inside each channel, so the pool is
    entity-clean before the cross-encoder ever sees it -- reranking is not asked
    to relearn a constraint the metadata already settles.
    """
    v = ctx.vecs if vecs is None else vecs
    d = top_k(v @ ctx.qvecs[i], ctx.masks[i], depth)
    l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), ctx.masks[i], depth)
    return rrf([d, l])[:depth]


def v_rerank(ctx, i, k, depth=50):
    return _rerank(ctx, i, _candidates(ctx, i, depth), k)


def v_rerank_hdr(ctx, i, k, depth=50):
    """As above, but the cross-encoder is shown the document and section heading.

    Worth testing separately from the bi-encoder version, where the same idea lost:
    a cross-encoder can use "which document is this" as evidence about relevance
    to the question, whereas a bi-encoder just gets a blurrier passage vector.
    """
    return _rerank(ctx, i, _candidates(ctx, i, depth), k, with_header=True)


def v_rerank_dense(ctx, i, k, depth=50):
    """Reranking a dense-only pool -- the ablation that says whether the lexical
    channel is earning its place, or whether the reranker would have found the
    same chunks anyway."""
    return _rerank(ctx, i, top_k(ctx.vecs @ ctx.qvecs[i], ctx.masks[i], depth), k)


def v_rerank_soft(ctx, i, k, depth=50, with_header=False):
    """Rerank a pool that is mostly filtered, but not exclusively.

    Motivated by a specific measured failure, q017: "Which institutions are
    covered by the term 'All India Financial Institutions'?" names an entity
    class, so the detector fires and the filter admits only chunks binding
    AIFIs. But the passage that answers it is the *definitions* section of the
    Small Finance Banks Directions, which binds SFBs -- the question mentions
    the class as the thing being defined, not as the class being regulated. The
    filter does not merely fail to help there; it removes the answer from the
    index before the search runs.

    So the pool is built from the filtered channels first and then topped up with
    unfiltered candidates, leaving the cross-encoder to decide. The filtered half
    keeps the near-duplicate noise down; the unfiltered half means a detector
    false-positive costs rank rather than reachability. The wrong-entity column is
    what says whether that trade actually held.
    """
    pool = _candidates(ctx, i, depth)
    if ctx.masks[i] is not None:
        d = top_k(ctx.vecs @ ctx.qvecs[i], None, depth // 2)
        l = top_k(ctx.need_bm25().score(ctx.qa[i]["question"]), None, depth // 2)
        for j in rrf([d, l]):
            if j not in pool:
                pool.append(j)
    return _rerank(ctx, i, pool, k, with_header=with_header)


def v_rerank_soft_hdr(ctx, i, k, depth=50):
    return v_rerank_soft(ctx, i, k, depth, with_header=True)


# The winning variant costs 11 seconds a question on this CPU, against a
# generation stage that answers in two. That is a real regression in the thing a
# user experiences, so the two obvious ways to buy it back get measured rather
# than assumed: rerank fewer candidates, or rerank them with a smaller model.
def v_rerank_hdr_d20(ctx, i, k):
    return _rerank(ctx, i, _candidates(ctx, i, 20), k, with_header=True)


MINI = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def v_rerank_hdr_mini(ctx, i, k, depth=50):
    return _rerank(ctx, i, _candidates(ctx, i, depth), k, with_header=True, model=MINI)


VARIANTS = {
    "dense":          ("dense only (the pipeline as deployed)", v_dense),
    "bm25":           ("lexical only", v_bm25),
    "hybrid":         ("RRF(dense, bm25)", v_hybrid),
    "dense-ctx":      ("dense over context-prefixed embeddings", v_dense_ctx),
    "hybrid-ctx":     ("RRF(dense-ctx, bm25)", v_hybrid_ctx),
    "dense-nbr":      ("dense + neighbour expansion", v_dense_nbr),
    "hybrid-nbr":     ("RRF(dense, bm25) + neighbours", v_hybrid_nbr),
    "hybrid-ctx-nbr": ("RRF(dense-ctx, bm25) + neighbours", v_hybrid_ctx_nbr),
    "rerank":         ("cross-encoder over the hybrid top-50", v_rerank),
    "rerank-hdr":     ("cross-encoder, shown doc + section", v_rerank_hdr),
    "rerank-dense":   ("cross-encoder over a dense-only top-50", v_rerank_dense),
    "rerank-soft":     ("cross-encoder, filtered pool topped up unfiltered", v_rerank_soft),
    "rerank-soft-hdr": ("as rerank-soft, shown doc + section", v_rerank_soft_hdr),
    "rerank-hdr-d20":  ("rerank-hdr over a 20-deep pool", v_rerank_hdr_d20),
    "rerank-hdr-mini": ("rerank-hdr with a 22M-param cross-encoder", v_rerank_hdr_mini),
}


# -------------------------------- scoring ---------------------------------------
def score(ctx, fn, k=DEFAULT_K):
    """Chunk recall@k, plus the document figure and the wrong-entity rate.

    WHY THE THIRD NUMBER IS NOT OPTIONAL
    This corpus is 74% near-duplicates, and the entity filter is what keeps a
    Payments Banks question from being answered out of the NBFC copy of the same
    rule. Any change that loosens retrieval can buy chunk recall by readmitting
    exactly that noise, and against a recall-only bench it would look like
    progress. So every variant also reports how many of the passages it returned
    bind an entity class the question did not ask about -- measured only on
    questions that name a class, since a question naming none cannot have a wrong
    answer to this. Unscoped chunks (the 41 circulars that bind nobody in
    particular) are never counted as wrong: they are legitimately reachable.
    """
    rows = []
    for i, q in enumerate(ctx.qa):
        got = fn(ctx, i, k)
        gold_ids = set(q["gold_chunk_ids"])
        gold_docs = set(q["gold_docs"])
        got_docs = [ctx.chunks[j]["number"] for j in got]
        rank = next((r for r, j in enumerate(got, 1) if j in gold_ids), None)

        want = set(detect_entities(q["question"]) or [])
        wrong = scoped = 0
        if want:
            for j in got:
                binds = set(ctx.chunks[j]["applies_to"] or [])
                if binds:
                    scoped += 1
                    wrong += not (binds & want)

        rows.append({
            "id": q["id"], "category": q["category"],
            "chunk_hit": bool(gold_ids & set(got)),
            "doc_hit": bool(gold_docs & set(got_docs)),
            "chunk_rank": rank, "retrieved": got,
            "wrong_entity": wrong, "entity_scoped_slots": scoped,
        })
    return rows


def summarize(rows, label, k, baseline=None):
    n = len(rows)
    ch = sum(r["chunk_hit"] for r in rows)
    dh = sum(r["doc_hit"] for r in rows)
    wrong = sum(r["wrong_entity"] for r in rows)
    slots = sum(r["entity_scoped_slots"] for r in rows)
    delta = ""
    if baseline is not None:
        b = sum(r["chunk_hit"] for r in baseline)
        delta = f"  {ch - b:+d}" if ch != b else "   ="
    print(f"  {label:<44}{f'{ch}/{n}':>8}{f'({ch * 100 // n}%)':>7}"
          f"{f'{dh}/{n}':>9}{f'{wrong * 100 // max(slots, 1)}%':>8}{delta}")
    return ch


LAB_RESULTS = ROOT / "eval" / "lab_variants.jsonl"


def cmd_run(names, k=DEFAULT_K, force=False):
    """Score variants one at a time, appending each to LAB_RESULTS as it finishes.

    Resumable for the same reason `eval.py answers` is, and a sharper one: the
    cross-encoder holds ~1GB and a run of six variants was killed twice on this
    machine partway through, losing every number it had already computed. Work
    that takes twenty minutes should not be all-or-nothing.
    """
    chunks, vecs, qa = load()
    alt = np.load(ROOT / "embeddings_ctx.npy") if (ROOT / "embeddings_ctx.npy").exists() else None
    LAB_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if LAB_RESULTS.exists() and not force:
        for line in open(LAB_RESULTS, encoding="utf-8"):
            r = json.loads(line)
            if r["k"] == k:
                done.add(r["variant"])

    todo = [n for n in names if n not in done]
    if not todo:
        print(f"all {len(names)} variant(s) already scored at k={k} -- "
              f"see `compare`, or pass --force")
        return

    ctx = Ctx(chunks, vecs, qa, alt_vecs=alt)
    with open(LAB_RESULTS, "a", encoding="utf-8") as f:
        for name in todo:
            desc, fn = VARIANTS[name]
            print(f"scoring {name} ...", flush=True)
            rows = score(ctx, fn, k)
            ch = sum(r["chunk_hit"] for r in rows)
            f.write(json.dumps({"variant": name, "desc": desc, "k": k,
                                "rows": rows}, ensure_ascii=False) + "\n")
            f.flush()                        # survive a kill mid-run
            print(f"  {name}: chunk {ch}/{len(rows)}", flush=True)


def cmd_compare(k=DEFAULT_K, which=None):
    if not LAB_RESULTS.exists():
        sys.exit(f"nothing scored yet -- run: python retrieval_lab.py run {' '.join(VARIANTS)}")
    saved = {}
    for line in open(LAB_RESULTS, encoding="utf-8"):
        r = json.loads(line)
        if r["k"] == k:
            saved[r["variant"]] = r          # a re-run of a variant supersedes the old row

    names = [n for n in (which or VARIANTS) if n in saved]
    missing = [n for n in (which or []) if n not in saved]
    if missing:
        print(f"not scored at k={k}: {', '.join(missing)}  "
              f"(run: python retrieval_lab.py run {' '.join(missing)})\n")
    if not names:
        return {}

    n_q = len(saved[names[0]]["rows"])
    print(f"{n_q} answerable questions with gold chunk pointers | k={k}")
    print(f"\n  {'variant':<44}{'chunk':>8}{'':>7}{'doc':>9}"
          f"{'wrong-ent':>10}  vs dense")
    base_rows = saved["dense"]["rows"] if "dense" in saved else None
    results = {}
    for name in names:
        rows = saved[name]["rows"]
        results[name] = rows
        summarize(rows, f"{name}  --  {saved[name]['desc']}", k,
                  baseline=base_rows if name != "dense" else None)

    # Which questions each variant fixes or breaks relative to dense. The net
    # number hides this: a variant that fixes six and breaks four reads as +2, and
    # the four it broke are a different bug from the ones it never touched.
    if base_rows and len(results) > 1:
        base = {r["id"]: r["chunk_hit"] for r in base_rows}
        for name, rows in results.items():
            if name == "dense":
                continue
            fixed = [r["id"] for r in rows if r["chunk_hit"] and not base[r["id"]]]
            broke = [r["id"] for r in rows if not r["chunk_hit"] and base[r["id"]]]
            if fixed or broke:
                print(f"\n  {name}:")
                if fixed:
                    print(f"    fixed  ({len(fixed)}): {', '.join(fixed)}")
                if broke:
                    print(f"    BROKE  ({len(broke)}): {', '.join(broke)}")
    return results


# The variant that corresponds to what rag.py actually runs: hybrid fetch at
# depth 50, cross-encoder rerank with the header, the small model. Named here so
# that when the pipeline changes again, `baseline` fails loudly until this line is
# updated to match -- which is the entire point of the check.
DEPLOYED = "rerank-hdr-mini"

# What shipped before the reranker, checked against the run preserved from then.
LEGACY = ("dense", "retrieval_k{k}_filtered_prererank.json")


def _agreement(rows, saved_path, label):
    if not saved_path.exists():
        print(f"  no {saved_path.name} to compare against")
        return None
    ref = {r["id"]: r for r in json.loads(saved_path.read_text(encoding="utf-8"))["rows"]}
    ch = sum(r["chunk_hit"] for r in rows)
    ref_ch = sum(1 for r in ref.values() if r.get("chunk_hit"))
    print(f"  lab {ch}/{len(rows)}   eval.py {ref_ch}/{len(ref)}   ({label})")
    disagree = [r["id"] for r in rows
                if r["id"] in ref and ref[r["id"]].get("chunk_hit") is not None
                and r["chunk_hit"] != ref[r["id"]]["chunk_hit"]]
    if disagree:
        print(f"  DISAGREE on {len(disagree)}: {', '.join(disagree)}")
    else:
        print("  agree on every question")
    return not disagree


def cmd_baseline(k=DEFAULT_K):
    """Prove the bench reproduces the deployed pipeline before trusting it.

    eval.py measures through Qdrant; this measures through numpy. They should not
    merely be close -- the search is exact in both, over the same vectors, with the
    same filter, and the ranking code is literally the same module, so any
    disagreement at all means one of them is not doing what it claims. Compared
    against the saved runs rather than numbers typed into a comment, so the check
    cannot rot.

    Two comparisons, because there are two saved runs worth agreeing with: the
    current pipeline, and the dense-only one it replaced. The second is not
    ceremony -- it is what stops the old numbers quoted in the README from
    becoming folklore.
    """
    chunks, vecs, qa = load()
    ctx = Ctx(chunks, vecs, qa)
    ok = []

    print(f"deployed config ({DEPLOYED}) vs data/eval/retrieval_k{k}_filtered.json")
    ok.append(_agreement(score(ctx, VARIANTS[DEPLOYED][1], k),
                         ROOT / "eval" / f"retrieval_k{k}_filtered.json", "current"))

    name, tmpl = LEGACY
    print(f"\npre-reranker config ({name}) vs {tmpl.format(k=k)}")
    ok.append(_agreement(score(ctx, VARIANTS[name][1], k),
                         ROOT / "eval" / tmpl.format(k=k), "before the reranker"))

    if all(o for o in ok if o is not None):
        print("\nthe bench is faithful to the pipeline")
    else:
        print("\nthe bench is NOT measuring the deployed pipeline -- fix before trusting it")


def cmd_inspect(qid, k=DEFAULT_K):
    """One question, every variant, with what actually came back.

    Aggregates say a variant is worth three questions; they never say why. This
    does, and reading a handful of these is what stops a fusion weight from being
    tuned into the noise.
    """
    chunks, vecs, qa = load()
    alt = np.load(ROOT / "embeddings_ctx.npy") if (ROOT / "embeddings_ctx.npy").exists() else None
    q = next((x for x in qa if x["id"] == qid), None)
    if not q:
        sys.exit(f"{qid} is not an answerable question with gold chunks")
    ctx = Ctx(chunks, vecs, [q], alt_vecs=alt)

    print(f"{qid}  [{q['category']}]  {q['question']}")
    print(f"entities detected: {detect_entities(q['question'])}")
    print(f"gold docs   {q['gold_docs']}")
    print(f"gold chunks {q['gold_chunk_ids']}")
    for gid in q["gold_chunk_ids"]:
        g = chunks[gid]
        print(f"  [{gid}] {g['number']}  section: {(g.get('section') or '-')[:56]}")
        print(f"        {g['text'][:220].strip()}")

    for name, (desc, fn) in VARIANTS.items():
        if name.endswith("-ctx") and alt is None:
            continue
        got = fn(ctx, 0, k)
        mark = "HIT " if set(got) & set(q["gold_chunk_ids"]) else "miss"
        print(f"\n{mark} {name}")
        for rank, j in enumerate(got, 1):
            c = chunks[j]
            star = " <-- GOLD" if j in q["gold_chunk_ids"] else ""
            print(f"  {rank:>2}. [{j}] {c['number']:<20} {(c.get('section') or '-')[:44]}{star}")


# ------------------------- context-prefixed embeddings --------------------------
# The same header the cross-encoder is shown, here fed to the bi-encoder instead.
# Testing one idea in both places is the point: it wins in one and loses in the
# other, and it would be easy to conclude "headers help" from either half alone.
context_text = header_text


def cmd_build_ctx():
    chunks, _, _ = load()
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print(f"embedding {len(chunks)} context-prefixed chunks (this takes a few minutes) ...")
    vecs = model.encode([context_text(c) for c in chunks], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=True).astype("float32")
    out = ROOT / "embeddings_ctx.npy"
    np.save(out, vecs)
    print(f"wrote {out}  shape={vecs.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline", help="check the bench reproduces eval.py")
    b.add_argument("-k", type=int, default=DEFAULT_K)
    r_ = sub.add_parser("run", help="score variants, appending each as it finishes")
    r_.add_argument("names", nargs="+", choices=list(VARIANTS))
    r_.add_argument("-k", type=int, default=DEFAULT_K)
    r_.add_argument("--force", action="store_true", help="re-score even if already saved")
    c = sub.add_parser("compare", help="table from whatever `run` has scored")
    c.add_argument("-k", type=int, default=DEFAULT_K)
    c.add_argument("--only", nargs="*", choices=list(VARIANTS), default=None)
    i = sub.add_parser("inspect", help="one question, every variant")
    i.add_argument("qid")
    i.add_argument("-k", type=int, default=DEFAULT_K)
    sub.add_parser("build-context-vectors", help="-> data/embeddings_ctx.npy")

    a = ap.parse_args()
    if a.cmd == "baseline":
        cmd_baseline(a.k)
    elif a.cmd == "run":
        cmd_run(a.names, a.k, a.force)
    elif a.cmd == "compare":
        cmd_compare(a.k, a.only)
    elif a.cmd == "inspect":
        cmd_inspect(a.qid, a.k)
    else:
        cmd_build_ctx()
