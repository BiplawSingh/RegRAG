"""
eval.py
-------
Scores the pipeline against the ground-truth set in data/qa.jsonl.

    python eval.py retrieval              # free, unlimited, no LLM involved
    python eval.py retrieval --no-filter  # the ablation: entity filter off
    python eval.py retrieval --sweep      # hit-rate at k = 1,3,5,8,10,20
    python eval.py answers                # end-to-end run (costs quota, resumable)
    python eval.py answers --oracle       # feed gold chunks, skip retrieval
    python eval.py report                 # scorecard from whatever has been run

Three metrics, and only one of them needs a model:

  retrieval hit-rate@k   did the gold document reach the top k?   local, free
  citation accuracy      does each quote occur where it claimed?  string compare
  answer faithfulness    does the answer match ground truth?      LLM judge

That asymmetry shapes the workflow. Retrieval can be re-measured as often as you
like -- embedding and vector search both run on this machine -- so retrieval
changes get iterated first and cheaply. Only once retrieval is settled is it worth
spending a day's token allowance on generation.

WHY --oracle EXISTS
Comparing the normal run against the oracle run separates two failure modes that
look identical in the output:

    fails normally, passes with oracle  -> retrieval never found the evidence
    fails both                          -> generation mishandled evidence it had
    passes normally, gold doc missed    -> gold_docs is incomplete; something else
                                           in the corpus answers it too

That third case is the one that improves the answer key over time.
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import env  # noqa: F401  -- loads .env before anything reads os.getenv

ROOT = Path("data")
QA = ROOT / "qa.jsonl"
RUNS = ROOT / "eval"

CATEGORIES = ["lookup", "entity_scoped", "cross_document", "amendment", "unanswerable"]


def load_qa():
    if not QA.exists():
        sys.exit("no data/qa.jsonl -- run: python qa.py export")
    return [json.loads(l) for l in open(QA, encoding="utf-8")]


def doc_families():
    """number -> family key, where a "family" is one rule reissued per entity class.

    Reuses qa.family_key so "family" means exactly one thing across the project --
    the same normalisation the 74% near-duplicate figure was measured with.

    WHY THIS METRIC EXISTS
    Strict hit-rate asks "did we return the document the answer key names". For a
    question that names no institution -- "what must an internal audit review at
    minimum?" -- that question is unfair: RBI issues that rule ten times, once per
    entity class, and the key can only name one of them. Returning the
    Co-operative Banks copy instead of the Payments Banks copy is scored as a
    total miss when it is nothing of the kind.

    So family hit-rate is reported *alongside* the strict number, never instead of
    it. Loosening the gold set itself would be the wrong fix: editing the answer
    key to match what the system returned is how a benchmark stops measuring
    anything. Two numbers, both honest, is the correct shape -- the strict one is
    the floor, the family one is the ceiling, and the gap between them is a
    statement about how precisely the questions were worded.
    """
    from qa import family_key

    fam = {}
    with open(ROOT / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            n = c.get("number")
            if n and n not in fam:
                fam[n] = family_key(c)
    return fam


def rag():
    """Import rag.py lazily -- it pulls in torch, which is slow to load."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("rag", "rag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight():
    """Fail fast and legibly when Qdrant isn't up.

    Without this, a stopped container surfaces forty lines of qdrant_client stack
    ending in WinError 10061 -- which reads like a bug in the harness rather than
    "start Docker". Long runs deserve to fail in the first second, not the tenth.
    """
    import urllib.request
    import urllib.error

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        with urllib.request.urlopen(f"{url}/collections/rbi_circulars", timeout=5) as r:
            info = json.loads(r.read())["result"]
    except Exception:
        sys.exit(
            f"Qdrant unreachable at {url}\n"
            "  start Docker Desktop, then:\n"
            "    docker start qdrant\n"
            "  or create it:\n"
            "    docker run -d --name qdrant -p 6333:6333 "
            "-v qdrant_storage:/qdrant/storage qdrant/qdrant"
        )
    if not info.get("points_count"):
        sys.exit("collection 'rbi_circulars' is empty -- run: python index_qdrant.py build")
    return info["points_count"]


# ------------------------------- retrieval ----------------------------------
def score_retrieval(qa, k=8, use_filter=True, quiet=False):
    """Hit-rate@k per question. No LLM, so this is free and repeatable.

    A question counts as a hit when *any* of its gold documents appears in the
    retrieved set. For the ten cross-document questions that is deliberately
    lenient at this stage -- `both_docs` below reports the stricter number
    separately, because needing two documents is a different problem from
    needing one.

    Unanswerable questions are excluded: there is no gold document to find, and
    scoring them here would silently reward retrieving nothing.
    """
    r = rag()
    fam = doc_families()
    rows = []
    scored = [q for q in qa if q["answerable"] and q["gold_docs"]]

    for i, q in enumerate(scored, 1):
        entity = r.detect_entity(q["question"])
        chunks, _ = r.retrieve(q["question"], k=k, entity=entity, use_filter=use_filter)
        got = [c["number"] for c in chunks]
        gold = set(q["gold_docs"])
        found = gold & set(got)
        # rank of the first gold document, 1-indexed; None if absent
        rank = next((j for j, n in enumerate(got, 1) if n in gold), None)
        # A sibling counts only when the family key is non-empty: family_key
        # returns "" for subjects it cannot parse, and treating "" == "" as a
        # match would score every unparseable document against every other one.
        gold_fams = {fam.get(g) for g in q["gold_docs"]} - {None, ""}
        got_fams = {fam.get(n) for n in got} - {None, ""}
        # Chunk-level hit. Document-level hit-rate flatters retrieval, because the
        # pipeline does not consume documents -- it consumes eight chunks. A
        # document scores as retrieved when ANY of its chunks comes back, including
        # one that does not answer the question. Measured after the fact: three
        # questions retrieved the right document (one at rank 1) and still got the
        # wrong answer, because the passage that answers them never arrived.
        got_ids = [c.get("chunk_id") for c in chunks]
        gold_ids = set(q.get("gold_chunk_ids") or [])
        rows.append({
            "id": q["id"], "category": q["category"],
            "gold_docs": q["gold_docs"], "retrieved": got,
            "hit": bool(found), "both_docs": len(found) == len(gold),
            "family_hit": bool(gold_fams & got_fams),
            "chunk_hit": bool(gold_ids & set(got_ids)) if gold_ids else None,
            "retrieved_chunk_ids": got_ids,
            "first_rank": rank, "entity": entity,
        })
        if not quiet and i % 10 == 0:
            print(f"  {i}/{len(scored)}", flush=True)
    return rows


def summarize_retrieval(rows, k, label):
    hits = sum(r["hit"] for r in rows)
    fam_hits = sum(bool(r["hit"] or r.get("family_hit")) for r in rows)
    n = len(rows)
    print(f"\n{label}  (k={k}, {n} answerable questions)")
    print(f"  hit-rate@{k}        {hits}/{n}  ({hits * 100 // max(n, 1)}%)   exact gold document")
    print(f"  family-rate@{k}     {fam_hits}/{n}  ({fam_hits * 100 // max(n, 1)}%)   "
          f"gold document or a sibling binding another entity")
    if fam_hits > hits:
        print(f"                     {fam_hits - hits} question(s) found the right rule "
              f"for the wrong institution class")

    # The stricter, more honest number: did the passage that answers the question
    # actually arrive? Reported under the document figure so the gap between them
    # is visible -- that gap is "right document, wrong passage", which needs
    # different work from "wrong document entirely".
    scored_c = [r for r in rows if r.get("chunk_hit") is not None]
    if scored_c:
        ch = sum(r["chunk_hit"] for r in scored_c)
        m = len(scored_c)
        print(f"  chunk-rate@{k}      {ch}/{m}  ({ch * 100 // m}%)   the answering "
              f"chunk itself, not just its document")
        doc_only = [r for r in scored_c if r["hit"] and not r["chunk_hit"]]
        if doc_only:
            print(f"                     {len(doc_only)} question(s) retrieved the right "
                  f"document but not the passage:")
            for r in doc_only[:8]:
                print(f"                       {r['id']}  {r['category']}"
                      f"  (doc at rank {r['first_rank']})")
            if len(doc_only) > 8:
                print(f"                       ... and {len(doc_only) - 8} more")

    multi = [r for r in rows if len(r["gold_docs"]) > 1]
    if multi:
        both = sum(r["both_docs"] for r in multi)
        print(f"  all gold docs      {both}/{len(multi)}  (cross-document only)")

    ranks = [r["first_rank"] for r in rows if r["first_rank"]]
    if ranks:
        print(f"  rank of first hit  median {int(statistics.median(ranks))}  "
              f"best {min(ranks)}  worst {max(ranks)}")

    print(f"\n  {'category':<18}{'hit':>6}{'of':>5}{'rate':>8}")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["category"]][0] += 1
        by[r["category"]][1] += 1 if r["hit"] else 0
    for cat in CATEGORIES:
        if cat in by:
            tot, hit = by[cat]
            print(f"  {cat:<18}{hit:>6}{tot:>5}{hit * 100 // tot:>7}%")

    # Split rather than sort-and-truncate: a sibling miss is a wording problem in
    # the question, a bare miss is a retrieval problem, and they get worked on
    # differently. Listing them separately also stops the shorter list -- usually
    # the siblings -- from being cut off by the truncation on the longer one.
    misses = [r for r in rows if not r["hit"]]
    bare = [r for r in misses if not r.get("family_hit")]
    sibling = [r for r in misses if r.get("family_hit")]

    if bare:
        print(f"\n  {len(bare)} miss(es) -- nothing from the right family retrieved:")
        for m in bare[:14]:
            print(f"    {m['id']}  {m['category']:<15} wanted {m['gold_docs']}")
        if len(bare) > 14:
            print(f"    ... and {len(bare) - 14} more")
    if sibling:
        print(f"\n  {len(sibling)} near-miss(es) -- right rule, wrong entity class:")
        for m in sibling:
            print(f"    {m['id']}  {m['category']:<15} wanted {m['gold_docs']}")
        print("    (question names no institution, so the key's choice of sibling is arbitrary)")


def cmd_retrieval(k=8, use_filter=True, sweep=False):
    n = preflight()
    qa = load_qa()
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"qdrant: {n:,} points | ground truth: {len(qa)} questions")

    if sweep:
        # The interesting question is not "what is hit-rate at 8" but "where does
        # the curve flatten" -- that is what tells you whether a reranker over a
        # wider candidate set has anything left to recover.
        print("recall as k grows:\n")
        print(f"  {'k':>4}{'doc':>12}{'family':>12}{'chunk':>12}{'median rank':>14}")
        by_k = []
        for kk in (1, 3, 5, 8, 10, 20):
            rows = score_retrieval(qa, k=kk, use_filter=use_filter, quiet=True)
            hits = sum(r["hit"] for r in rows)
            fam = sum(bool(r["hit"] or r.get("family_hit")) for r in rows)
            sc = [r for r in rows if r.get("chunk_hit") is not None]
            ch = sum(r["chunk_hit"] for r in sc)
            ranks = [r["first_rank"] for r in rows if r["first_rank"]]
            med = int(statistics.median(ranks)) if ranks else "-"
            n = len(rows)
            print(f"  {kk:>4}{f'{hits}/{n}':>12}{f'{fam}/{n}':>12}"
                  f"{f'{ch}/{len(sc)}':>12}{med:>14}")
            by_k.append((kk, {"doc": hits, "chunk": ch, "n": len(sc)}))
        # Report the headroom rather than asserting a conclusion about it. A
        # reranker fetches a wide candidate set and re-sorts it down to k, so what
        # it could recover -- at absolute best, with perfect ranking -- is exactly
        # the recall a wider k already reaches. Chunk recall is the number that
        # matters here: the pipeline consumes passages, not documents.
        # Baseline is the k actually in use, not k=1. A reranker replaces the
        # current operating point, so the gain is measured against that -- not
        # against a configuration nobody runs.
        base = dict(by_k).get(k, dict(by_k)[by_k[0][0]])
        k_base = k if k in dict(by_k) else by_k[0][0]
        if len(by_k) > 1:
            k_hi, hi = by_k[-1]
            lo = base
            print(f"\n  reranking headroom, chunk level: {hi['chunk']}/{hi['n']} at "
                  f"k={k_hi} vs {lo['chunk']}/{lo['n']} at the operating k={k_base}")
            print(f"    a perfect reranker over the k={k_hi} set recovers at most "
                  f"{hi['chunk'] - lo['chunk']} question(s); a real one, fewer.")
            print(f"    the other {hi['n'] - hi['chunk']} never surface the right "
                  f"passage at k={k_hi} at all, so no re-sorting reaches them.")
        return

    label = "entity filter ON" if use_filter else "entity filter OFF (ablation)"
    rows = score_retrieval(qa, k=k, use_filter=use_filter)
    summarize_retrieval(rows, k, label)

    out = RUNS / f"retrieval_k{k}_{'filtered' if use_filter else 'nofilter'}.json"
    out.write_text(json.dumps({
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k, "filtered": use_filter, "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


# -------------------------------- answers -----------------------------------
def answers_path(oracle, k):
    return RUNS / f"answers_{'oracle' if oracle else 'e2e'}_k{k}.jsonl"


REVIEW = ROOT / "qa_review.yaml"


def load_review():
    """Hand-review verdicts on the unanswerable questions, or {} if none.

    Kept out of the answer key on purpose. The key was written before any model
    output existed, and editing it to match what the system produced is how a
    benchmark decays into a tautology. So the review is an overlay: the strict
    score and the reviewed score both get printed, and a reader can see the size
    of the disagreement rather than having to trust one number.

    'disputed' deliberately counts as a failure. An unresolved question should
    never flatter the score -- otherwise the cheapest way to improve a result is
    to leave things unresolved.
    """
    if not REVIEW.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(REVIEW.read_text(encoding="utf-8")) or {}
    return {v["id"]: v.get("verdict") for v in data.get("verdicts") or []}


def cmd_answers(k=8, use_filter=True, oracle=False, limit=None):
    """Generate an answer per question. Resumable, because the daily token
    allowance will not cover a full pass in one sitting and a run that cannot
    checkpoint is a run that never finishes."""
    if not oracle:
        preflight()          # oracle mode bypasses retrieval, so Qdrant isn't needed
    qa = load_qa()
    r = rag()
    RUNS.mkdir(parents=True, exist_ok=True)
    path = answers_path(oracle, k)

    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            done.add(json.loads(line)["id"])
        print(f"resuming: {len(done)} already answered in {path.name}")

    todo = [q for q in qa if q["id"] not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("nothing left to answer")
        return

    chunks_by_id = None
    if oracle:
        chunks_by_id = [json.loads(l) for l in open(ROOT / "chunks.jsonl", encoding="utf-8")]

    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            print(f"[{i:>3}/{len(todo)}] {q['id']}  {q['question'][:56]}", flush=True)
            try:
                if oracle:
                    ids = q["gold_chunk_ids"] or []
                    chunks = [chunks_by_id[j] for j in ids]
                    if not chunks:               # unanswerable: nothing to feed
                        chunks = []
                    entity = None
                else:
                    entity = r.detect_entity(q["question"])
                    chunks, _ = r.retrieve(q["question"], k=k, entity=entity,
                                           use_filter=use_filter)
                ans, secs, usage, _ = r.generate(q["question"], chunks)
                checks = r.verify(ans, chunks)
                rec = {
                    "id": q["id"], "category": q["category"],
                    "answerable": q["answerable"], "gold_docs": q["gold_docs"],
                    "question": q["question"], "expected": q["answer"],
                    "predicted": ans.answer,
                    "said_insufficient": ans.insufficient_context,
                    "retrieved": [c["number"] for c in chunks],
                    "citations": checks,
                    "verified": sum(1 for c in checks if c["status"] == "ok"),
                    "n_citations": len(checks),
                    "seconds": round(secs, 2), "usage": usage,
                    "oracle": oracle, "k": k, "filtered": use_filter,
                    "model": r.GEN_MODEL,
                }
            except Exception as e:
                msg = f"{e.__class__.__name__}: {e}"
                if "RateLimitError" in msg and ("per day" in msg.lower() or "TPD" in msg):
                    print(f"\nDAILY ALLOWANCE EXHAUSTED at {i}/{len(todo)}. "
                          f"Re-run the same command tomorrow -- it resumes.")
                    break
                print(f"       FAILED {msg[:130]}")
                rec = {"id": q["id"], "error": msg}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()                            # survive a kill mid-run

    print(f"\nwrote {path}")


# --------------------------------- report -----------------------------------
def cmd_report(k=8):
    qa = {q["id"]: q for q in load_qa()}
    print(f"ground truth: {len(qa)} questions\n")

    for oracle in (False, True):
        path = answers_path(oracle, k)
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        ok = [r for r in rows if "error" not in r]
        if not ok:
            continue

        label = "ORACLE (gold chunks fed directly)" if oracle else "END-TO-END"
        print(f"=== {label} ===  {len(ok)}/{len(qa)} answered")

        cites = sum(r["n_citations"] for r in ok)
        good = sum(r["verified"] for r in ok)
        print(f"  citations verified   {good}/{cites}"
              f"  ({good * 100 // max(cites, 1)}%)")
        bad = [c for r in ok for c in r["citations"] if c["status"] != "ok"]
        wrong = sum(1 for c in bad if c["status"] == "wrong_source")
        absent = sum(1 for c in bad if c["status"] == "not_found")
        print(f"    misattributed      {wrong}")
        print(f"    not in any source  {absent}")

        # How much of that verified figure is actually load-bearing. When ten
        # sibling circulars carry the same sentence, a quote lifted from the NBFC
        # copy is genuinely a substring of the Payments Banks copy too -- so the
        # check returns ok and CANNOT detect misattribution. Those passes are
        # unfalsifiable, and reporting the verified rate without them overstates
        # what the check proves. This is why misattributed reads 0: partly good
        # behaviour, partly a blind spot.
        verified = [c for r in ok for c in r["citations"] if c["status"] == "ok"]
        ambiguous = [c for c in verified if len(c.get("found_in") or []) > 1]
        if verified:
            print(f"    of which unfalsifiable {len(ambiguous)}/{len(verified)}"
                  f"  ({len(ambiguous) * 100 // len(verified)}%)"
                  f" -- quote present in >1 retrieved doc")
            if ambiguous:
                worst = max(len(c.get("found_in") or []) for c in ambiguous)
                print(f"      worst: one quote found in {worst} retrieved documents")

        # Refusal behaviour: the metric no citation check can capture, because a
        # confident wrong answer with a real quote passes every other test.
        ans_q = [r for r in ok if r["answerable"]]
        unans_q = [r for r in ok if not r["answerable"]]
        if unans_q:
            correct = sum(1 for r in unans_q if r["said_insufficient"])
            print(f"  refused when it should  {correct}/{len(unans_q)}"
                  f"   (strict: key as written)")
            # Reviewed: a question whose answer exists elsewhere in the corpus was
            # never unanswerable, so answering it is not a failure. See qa_review.yaml.
            verdicts = load_review()
            if verdicts:
                seen = [r for r in unans_q if r["id"] in verdicts]
                # Count what is genuinely un-reviewed before dropping superseded
                # rows, or a rewritten question gets reported as an outstanding
                # review it no longer needs.
                missing = len(unans_q) - len(seen)
                # 'superseded' verdicts judged a question that has since been
                # rewritten, so they describe text no longer in the key. Drop them
                # from both numerator and denominator rather than guessing.
                seen = [r for r in seen if verdicts[r["id"]] != "superseded"]
                if seen:
                    good = sum(1 for r in seen
                               if verdicts[r["id"]] in ("correct_refusal", "correct_answer"))
                    disputed = sum(1 for r in seen if verdicts[r["id"]] == "disputed")
                    extra = f", {disputed} disputed counted against" if disputed else ""
                    print(f"  acceptable on review    {good}/{len(seen)}"
                          f"   (hand-reviewed{extra})")
                if missing:
                    print(f"    {missing} answered but not yet reviewed")
        if ans_q:
            wrongly = sum(1 for r in ans_q if r["said_insufficient"])
            print(f"  refused when it shouldn't {wrongly}/{len(ans_q)}")

        secs = [r["seconds"] for r in ok]
        toks = [r["usage"].get("input", 0) + r["usage"].get("output", 0)
                for r in ok if r.get("usage")]
        print(f"  median latency       {statistics.median(secs):.1f}s")
        if toks:
            print(f"  tokens per call      median {int(statistics.median(toks))}"
                  f"  total {sum(toks):,}")
        print()

    e2e, orc = answers_path(False, k), answers_path(True, k)
    if e2e.exists() and orc.exists():
        a = {json.loads(l)["id"]: json.loads(l) for l in open(e2e, encoding="utf-8")}
        b = {json.loads(l)["id"]: json.loads(l) for l in open(orc, encoding="utf-8")}
        shared = [i for i in a if i in b and "error" not in a[i] and "error" not in b[i]]
        if shared:
            only_oracle = [i for i in shared
                           if a[i]["said_insufficient"] and not b[i]["said_insufficient"]]
            print("=== retrieval vs generation ===")
            print(f"  answered only when gold chunks were supplied: {len(only_oracle)}")
            if only_oracle:
                print("    (these are retrieval failures, not generation failures)")
                for i in only_oracle[:10]:
                    print(f"      {i}  {a[i]['question'][:60]}")

    for name, path in (("retrieval, filtered", RUNS / f"retrieval_k{k}_filtered.json"),
                       ("retrieval, no filter", RUNS / f"retrieval_k{k}_nofilter.json")):
        if path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            rows = d["rows"]
            n = max(len(rows), 1)
            hits = sum(r["hit"] for r in rows)
            line = f"\n{name}:  exact {hits}/{len(rows)} ({hits * 100 // n}%)"
            # A run saved before family scoring existed has no family_hit key. Say
            # so rather than printing family == exact, which reads as a measured
            # result showing no sibling misses -- the opposite of what it means.
            if any("family_hit" in r for r in rows):
                fam = sum(bool(r["hit"] or r.get("family_hit")) for r in rows)
                line += f"   family {fam}/{len(rows)} ({fam * 100 // n}%)"
            else:
                line += "   family not scored (re-run to measure)"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r_ = sub.add_parser("retrieval", help="hit-rate@k -- no LLM, free to re-run")
    r_.add_argument("-k", type=int, default=8)
    r_.add_argument("--no-filter", action="store_true", help="ablation: entity filter off")
    r_.add_argument("--sweep", action="store_true", help="hit-rate across several k")

    a_ = sub.add_parser("answers", help="generate answers (costs quota, resumable)")
    a_.add_argument("-k", type=int, default=8)
    a_.add_argument("--no-filter", action="store_true")
    a_.add_argument("--oracle", action="store_true",
                    help="feed gold chunks directly, bypassing retrieval")
    a_.add_argument("--limit", type=int, default=None, help="stop after N questions")

    p_ = sub.add_parser("report", help="scorecard from whatever has been run")
    p_.add_argument("-k", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "retrieval":
        cmd_retrieval(k=args.k, use_filter=not args.no_filter, sweep=args.sweep)
    elif args.cmd == "answers":
        cmd_answers(k=args.k, use_filter=not args.no_filter,
                    oracle=args.oracle, limit=args.limit)
    else:
        cmd_report(k=args.k)
