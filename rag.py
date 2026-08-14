"""
rag.py
------
The naive end-to-end pipeline: retrieve -> generate -> verify.

    python rag.py models                          # what your key can actually call
    python rag.py ask "What KYC rules apply to Payments Banks?"
    python rag.py ask "..." --entity "Payments Banks" --no-filter
    python rag.py batch                           # the 20-question run -> data/answers.jsonl
    python rag.py history                         # everything ever asked -> data/log.jsonl

Every question asked through `ask` or `batch` is appended to data/log.jsonl automatically
-- nothing is ever lost, unlike data/answers.jsonl which is just the latest batch snapshot
and gets overwritten each run.

Generation runs on Gemini's free tier. That costs one capability worth naming:
this pipeline gets no native citation offsets, so citations are reconstructed and
then *checked*. The model returns a verbatim `quote` per claim, and verify_quote()
confirms that span actually occurs in the chunk it was attributed to.

That check is the point. A model can emit a citation that looks right and quotes
text no source contains -- prose-parsed citations hide it, and a metric built on
"the model said [3]" scores a fabrication as a hit. An unverified quote here is a
visible, countable failure, which is what Week 2's citation-accuracy number needs.
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path("data")
OUT = ROOT / "answers.jsonl"      # latest batch run only -- overwritten every `batch`
LOG = ROOT / "log.jsonl"          # every question ever asked -- append-only, never overwritten

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Flash-Lite, not Flash: the free tier meters requests per day *per model*, and the
# first batch run exhausted gemini-3.6-flash's 20/day allowance in a single pass
# (17 answers + 3 failures -- failed calls still consume quota). Week 2 needs
# 75-100 questions across several pipeline versions, so daily allowance is the
# binding constraint, not per-request quality.
#
# Per-model RPD is not in the API (models.list() returns capabilities, never quota)
# -- it's a browser-authenticated AI Studio page, so it can't be checked here. Read
# yours at https://aistudio.google.com/rate-limit and set GEMINI_MODEL if a
# different model wins. `python rag.py models` lists everything your key can call.
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

SYSTEM = """You answer questions about Indian banking regulation using ONLY the \
numbered SOURCES provided. These are Reserve Bank of India circulars.

Rules:
- Use only what the sources state. Never use outside knowledge about RBI regulation.
- RBI issues near-identical circulars for different classes of institution. A rule \
binding Small Finance Banks does NOT bind Payments Banks. Check each source's \
"binds" line before relying on it, and never present a rule for one class as \
applying to another.
- Every factual claim needs a citation whose `quote` is copied VERBATIM from the \
source -- character for character, not paraphrased or reflowed.
- If the sources do not answer the question, set insufficient_context to true and \
say what is missing. Do not guess."""


class Citation(BaseModel):
    source_id: int = Field(description="The [n] number of the source being cited.")
    quote: str = Field(description="Text copied verbatim from that source.")


class Answer(BaseModel):
    answer: str = Field(description="The answer, grounded only in the sources.")
    citations: list[Citation] = Field(description="One per factual claim.")
    insufficient_context: bool = Field(
        description="True if the sources do not contain the answer."
    )


# ------------------------------- retrieval ----------------------------------
_MODULES = {}


def _load(name, path):
    """Load a sibling script once and keep it -- these carry heavy imports."""
    if name not in _MODULES:
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def detect_entity(question: str):
    """Every entity class the question names, using the corpus's own vocabulary.

    Returns a list, not one class: a comparison question names two, and keeping only
    the first would filter away half of what it is asking about.
    """
    return _load("rbi_scraper", "rbi_scraper.py").scan_entities(question) or None


_RETRIEVER = {}


def retrieve(question, k=8, entity=None, use_filter=True):
    iq = _load("index_qdrant", "index_qdrant.py")
    if not _RETRIEVER:                       # client + embedding model are expensive
        client, where = iq.connect(QDRANT_URL)
        _RETRIEVER.update(client=client, where=where, model=iq.get_model())
    points = iq.query(_RETRIEVER["client"], _RETRIEVER["model"], question, k=k,
                      entity=entity if use_filter else None)
    return [p.payload for p in points], _RETRIEVER["where"]


def format_sources(chunks):
    """Numbered blocks. `binds` is shown because it is what the model must check."""
    out = []
    for i, c in enumerate(chunks, 1):
        binds = ", ".join(c.get("applies_to") or []) or "(not entity-scoped)"
        out.append(
            f"[{i}] {c['number']}  ({c.get('issue_date')})\n"
            f"binds: {binds}\n"
            f"document: {c.get('subject')}\n"
            f"section: {c.get('section') or '-'}\n"
            f"---\n{c['text']}"
        )
    return "\n\n".join(out)


# ------------------------------- generation ---------------------------------
_GENAI = {}


def genai_client():
    """One long-lived client.

    Must be held in a module-level reference, not built inline. A temporary
    `genai.Client()` is freed as soon as the expression that made it finishes, and
    its finalizer closes the underlying httpx client -- so a lazily-issued request
    (anything paginated, like models.list) then fails with "client has been closed",
    which reads like a network or auth fault rather than an object-lifetime one.
    """
    if "client" not in _GENAI:
        from google import genai

        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            sys.exit("set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)")
        _GENAI["client"] = genai.Client()
    return _GENAI["client"]


def generate(question, chunks, attempts=4):
    """Generate one answer, retrying only what is worth retrying.

    503 UNAVAILABLE is transient server load and succeeds on a retry -- but a failed
    call still consumes free-tier quota, so letting it through costs a request and
    returns nothing. 429 RESOURCE_EXHAUSTED is the opposite: the daily allowance is
    gone and every further attempt burns nothing but wall-clock, so it is raised
    immediately rather than retried. The server's own `retryDelay` is honoured when
    present, since it knows more than a fixed backoff does.
    """
    from google.genai import types

    client = genai_client()
    prompt = f"SOURCES\n\n{format_sources(chunks)}\n\nQUESTION\n{question}"
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_schema=Answer,
    )

    t0 = time.time()
    for attempt in range(attempts):
        try:
            resp = client.models.generate_content(
                model=GEN_MODEL, contents=prompt, config=cfg
            )
            return Answer.model_validate_json(resp.text), time.time() - t0
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                raise                                    # quota gone; retrying wastes time
            transient = any(s in msg for s in ("UNAVAILABLE", "503", "500", "INTERNAL"))
            if not transient or attempt == attempts - 1:
                raise
            m = re.search(r"'retryDelay':\s*'(\d+)s'", msg)
            wait = int(m.group(1)) if m else 2 ** attempt
            print(f"        {('503/transient')} retry {attempt + 1}/{attempts - 1} in {wait}s",
                  flush=True)
            time.sleep(wait)


# ------------------------------ verification --------------------------------
def norm(s: str) -> str:
    """Whitespace-insensitive compare: a reflowed quote is a match, not a miss."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def verify(ans: Answer, chunks):
    """Check each quote occurs in the chunk it was attributed to.

    Three outcomes, and they mean different things:
      ok           -- quote found in the cited source
      wrong_source -- quote exists in the corpus, but not in the source cited
      not_found    -- quote appears in no retrieved source (fabricated or paraphrased)
    """
    results = []
    normed = [norm(c["text"]) for c in chunks]
    for cit in ans.citations:
        q = norm(cit.quote)
        idx = cit.source_id - 1
        in_cited = 0 <= idx < len(chunks) and q and q in normed[idx]
        elsewhere = [i + 1 for i, t in enumerate(normed) if q and q in t]
        results.append({
            "source_id": cit.source_id,
            "quote": cit.quote,
            "status": "ok" if in_cited else ("wrong_source" if elsewhere else "not_found"),
            "found_in": elsewhere,
            "cited_doc": chunks[idx]["number"] if 0 <= idx < len(chunks) else None,
        })
    return results


# --------------------------------- logging -----------------------------------
def log_record(rec: dict):
    """Append one record to the permanent history file.

    Every question that goes through run_one() lands here, whether asked one at a
    time or as part of a batch. `data/answers.jsonl` is only the most recent batch
    snapshot and gets overwritten on every run -- LOG is the durable one, so nothing
    that's been asked is ever lost.
    """
    ROOT.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_history(tail=10):
    """Summarize everything ever logged, then show the most recent questions."""
    if not LOG.exists():
        print(f"no history yet -- {LOG} doesn't exist. Ask a question first.")
        return

    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")]
    ok = [r for r in rows if "error" not in r]
    print(f"total questions logged   {len(rows)}")
    if rows:
        print(f"date range               {rows[0]['timestamp'][:10]} -> {rows[-1]['timestamp'][:10]}")
    if ok:
        cites = sum(r["n_citations"] for r in ok)
        good = sum(r["verified"] for r in ok)
        print(f"answered                 {len(ok)}  "
              f"({sum(1 for r in ok if r['insufficient_context'])} said insufficient context)")
        print(f"citations verified       {good}/{cites}"
              f"  ({good * 100 // max(cites, 1)}%)")
        by_model = {}
        for r in ok:
            by_model[r["model"]] = by_model.get(r["model"], 0) + 1
        for m, n in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"  {n:>3} via {m}")

    if tail:
        print(f"\nlast {min(tail, len(rows))} questions:")
        for r in rows[-tail:]:
            when = r.get("timestamp", "?")[:16].replace("T", " ")
            if "error" in r:
                print(f"  {when}  ERROR  {r['question'][:60]}")
            else:
                mark = "?" if r["insufficient_context"] else " "
                print(f"  {when} {mark} {r['verified']}/{r['n_citations']} cited"
                      f"  {r['question'][:56]}")


# --------------------------------- commands ---------------------------------
def cmd_models(all_models=False):
    """List models this key can call.

    Note what this does NOT show: rate limits. The API exposes capabilities only --
    per-model daily/minute quotas live in AI Studio (https://aistudio.google.com/rate-limit)
    and are the number that actually decides whether a model can carry Week 2.
    """
    client = genai_client()          # held for the whole iteration; see genai_client()
    rows = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not all_models and actions and "generateContent" not in actions:
            continue
        rows.append((
            (m.name or "").replace("models/", ""),
            m.input_token_limit or 0,
            m.output_token_limit or 0,
            "yes" if getattr(m, "thinking", False) else "-",
            m.display_name or "",
        ))

    rows.sort()
    print(f"{'model id':<38}{'in':>10}{'out':>9}{'think':>7}  name")
    print("-" * 92)
    for name, tin, tout, think, disp in rows:
        mark = " *" if name == GEN_MODEL else "  "
        print(f"{mark}{name:<36}{tin:>10,}{tout:>9,}{think:>7}  {disp[:30]}")
    print(f"\n{len(rows)} callable  (* = current GEN_MODEL)")
    print("quotas are not in the API -- see https://aistudio.google.com/rate-limit")


def run_one(question, k=8, entity=None, use_filter=True, quiet=False,
            run_id=None, source="ask"):
    entity = entity or detect_entity(question)
    chunks, where = retrieve(question, k=k, entity=entity, use_filter=use_filter)
    ans, secs = generate(question, chunks)
    checks = verify(ans, chunks)

    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id or uuid.uuid4().hex[:8],
        "source": source,               # "ask" (one-off) or "batch"
        "question": question,
        "entity": entity,
        "filtered": bool(entity) and use_filter,
        "retrieved": [c["number"] for c in chunks],
        "answer": ans.answer,
        "insufficient_context": ans.insufficient_context,
        "citations": checks,
        "verified": sum(1 for c in checks if c["status"] == "ok"),
        "n_citations": len(checks),
        "seconds": round(secs, 2),
        "model": GEN_MODEL,
    }
    log_record(rec)          # every question, logged automatically -- see log_record()
    if not quiet:
        print(f"\nQ  {question}")
        shown = ", ".join(entity) if entity else "none detected"
        print(f"   entity={shown}  filter={'on' if rec['filtered'] else 'off'}"
              f"  k={k}  {secs:.1f}s  [{where}]")
        if ans.insufficient_context:
            print("   INSUFFICIENT CONTEXT")
        print(f"\n{ans.answer}\n")
        for c in checks:
            mark = {"ok": "OK  ", "wrong_source": "SRC ", "not_found": "MISS"}[c["status"]]
            print(f"   {mark} [{c['source_id']}] {c['cited_doc']}")
            print(f"        \"{c['quote'][:96]}\"")
            if c["status"] == "wrong_source":
                print(f"        (actually in source {c['found_in']})")
        print(f"\n   verified {rec['verified']}/{rec['n_citations']} citations")
    return rec


# Twenty questions spanning the shapes the eval will need: single-document lookups,
# entity-scoped questions that sit inside a near-duplicate family, cross-document
# comparisons, in-force/amendment questions, and four with no answer in the corpus
# (an honest pipeline must say so rather than invent one).
QUESTIONS = [
    "What are the KYC requirements for Payments Banks?",
    "What conduct rules apply to recovery agents engaged by Housing Finance Companies?",
    "What are the fraud risk management directions for Urban Co-operative Banks?",
    "How must Small Finance Banks present financial statement disclosures?",
    "What capital adequacy requirements apply to Regional Rural Banks?",
    "What are the asset classification norms for NBFCs?",
    "What supervisory returns must Rural Co-operative Banks file, and by when?",
    "What responsible business conduct rules apply to Local Area Banks?",
    "What are the provisioning floors for secured retail loans?",
    "When is a loan account treated as out of order?",
    "What is the definition of a Non-Performing Asset for commercial banks?",
    "What rules govern Special Rupee Vostro Accounts?",
    "How should banks handle soiled and mutilated banknotes?",
    "What are the reporting obligations for Fake Indian Currency Notes?",
    "Do the recovery agent rules for NBFCs differ from those for Housing Finance Companies?",
    "Which entity classes received the Responsible Business Conduct amendment directions?",
    "What changed in the third amendment to the Housing Finance Companies directions?",
    "What is the current expected credit loss provisioning framework for commercial banks?",
    "What is the RBI repo rate?",
    "What are the GST filing deadlines for banks?",
]


def cmd_batch(k=8, use_filter=True):
    ROOT.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex[:8]     # shared by every question in this batch, so
    rows = []                        # `history` can group them back together
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:>2}/{len(QUESTIONS)}] {q[:64]}", flush=True)
        try:
            rows.append(run_one(q, k=k, use_filter=use_filter, quiet=True,
                                 run_id=run_id, source="batch"))
        except Exception as e:
            msg = f"{e.__class__.__name__}: {e}"
            err = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "run_id": run_id, "source": "batch", "question": q, "error": msg}
            rows.append(err)
            log_record(err)           # failures get recorded too, not just successes
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                # Daily quota is gone. Every remaining question would fail the same
                # way, so stop and keep the partial run rather than logging 6 more
                # identical errors and pretending the batch completed.
                print(f"        QUOTA EXHAUSTED -- stopping at {i}/{len(QUESTIONS)}")
                break
            print(f"        FAILED {msg}")
        time.sleep(1.0)                             # free-tier rate limits

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in rows if "error" not in r]
    cites = sum(r["n_citations"] for r in ok)
    good = sum(r["verified"] for r in ok)
    bad = [c for r in ok for c in r["citations"] if c["status"] != "ok"]
    print(f"\nanswered            {len(ok)}/{len(rows)}")
    # Don't label these "expected" -- which questions this corpus can answer is a
    # measured fact, not a constant. Measured on the first run: an insufficient_context
    # here usually means retrieval found the right *topic* but landed on an Amendment
    # Direction, which states a delta ("paragraph 5.2 shall be substituted") rather
    # than the rule. Declining is correct there; the corpus is the problem.
    print(f"said insufficient   {sum(1 for r in ok if r['insufficient_context'])}"
          f"   (verify each against the corpus before scoring it)")
    print(f"citations verified  {good}/{cites}"
          f"  ({good * 100 // max(cites, 1)}%)")
    print(f"  wrong source      {sum(1 for c in bad if c['status'] == 'wrong_source')}")
    print(f"  not in any source {sum(1 for c in bad if c['status'] == 'not_found')}")
    print(f"median latency      {sorted(r['seconds'] for r in ok)[len(ok) // 2]:.1f}s"
          if ok else "")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("models")
    mp.add_argument("--all", action="store_true",
                    help="include embedding/other models, not just generateContent")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--entity", default=None, help="override auto-detection")
    a.add_argument("-k", type=int, default=8)
    a.add_argument("--no-filter", action="store_true", help="ablation: skip the entity filter")
    b = sub.add_parser("batch")
    b.add_argument("-k", type=int, default=8)
    b.add_argument("--no-filter", action="store_true")
    h = sub.add_parser("history", help="everything ever logged -- data/log.jsonl")
    h.add_argument("--tail", type=int, default=10, help="how many recent questions to show")
    args = ap.parse_args()

    if args.cmd == "models":
        cmd_models(all_models=args.all)
    elif args.cmd == "ask":
        run_one(args.question, k=args.k, entity=args.entity, use_filter=not args.no_filter)
    elif args.cmd == "batch":
        cmd_batch(k=args.k, use_filter=not args.no_filter)
    else:
        cmd_history(tail=args.tail)
