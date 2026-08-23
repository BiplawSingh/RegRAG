"""
qa.py
-----
Authoring tool for the ground-truth question set.

    python qa.py draft              # add stratified stubs -> data/qa_draft.yaml
    python qa.py draft --n 30       # add 30 more
    python qa.py stats              # how far along, and coverage by category
    python qa.py validate           # what's incomplete or malformed
    python qa.py export             # -> data/qa.jsonl for the eval harness

The workflow is: draft stubs, fill them in your editor (VS Code handles YAML far
better than a terminal prompt does for 90 multi-sentence answers), validate,
export.

WHAT THIS TOOL DOES NOT DO: write questions or answers. It picks *which* chunks
you look at and does the clerical work around them. The judgement stays yours,
because an eval whose answers came from a model is an eval that measures
agreement with that model rather than correctness -- and you can't defend the
number when someone asks how you know it's right.

Stubs are stratified across the failure modes this corpus is already known to
have, because 90 deliberately-chosen questions say far more than 300 random ones:

  lookup          plain factual answer, stated in one chunk of a base direction
  entity_scoped   from a near-duplicate family, where a near-identical circular
                  exists for other institution types -- the KYC-for-Payments-Banks
                  shape, and the case the entity filter exists to handle
  cross_document  needs two documents; two variants of one family are supplied
  amendment       from an Amendment Direction. These state deltas ("paragraph 5.2
                  shall be substituted") and may not contain the rule at all --
                  mark answerable: false when the chunk genuinely doesn't answer it
  unanswerable    no chunk supplied. Write something this corpus cannot answer, to
                  check the pipeline refuses instead of inventing
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path("data")
CHUNKS = ROOT / "chunks.jsonl"
DRAFT = ROOT / "qa_draft.yaml"
EXPORT = ROOT / "qa.jsonl"

# Target mix for a ~90-question set. Weighted toward the cases that discriminate:
# a lookup question passes on almost any pipeline, an entity_scoped one doesn't.
TARGET = {
    "lookup": 30,
    "entity_scoped": 25,
    "cross_document": 10,
    "amendment": 15,
    "unanswerable": 10,
}

AMEND_RE = re.compile(r"amendment directions", re.I)
# A chunk starting lower-case is a mid-sentence fragment -- readable in context,
# useless as a standalone question source.
STARTS_CLEAN_RE = re.compile(r"^[\"'(\[]?[A-Z0-9]")

# Readability, and why it's scored rather than ignored.
#
# You cannot write ground truth for a rule you can't read. A chunk about Stage 1
# ECL provisioning floors on DCCO-deferred project finance is a perfectly valid
# regulatory rule and a terrible question source for anyone who isn't already a
# credit-risk specialist -- you'd be guessing at correctness, which defeats the
# entire point of a hand-written answer key.
#
# Plenty of this corpus is plain-language: recovery agent conduct, counterfeit
# notes, complaints, KYC documents, lockers, ATM rules. Those make questions a
# non-specialist can verify, and they are no less real as regulation.
JARGON_RE = re.compile(
    r"provision|capital adequacy|risk weight|Basel|\bECL\b|exposure|securitis|"
    r"derivative|liquidity coverage|tier [12]|CRAR|prudential floor|Stage [123]|"
    r"amortis|hedge accounting|risk-weighted|counterparty", re.I)
PLAIN_RE = re.compile(
    r"complaint|grievance|ombudsman|banknote|counterfeit|soiled|mutilated|locker|"
    r"nomination|recovery agent|working hours|\bATM\b|passbook|cheque|\bKYC\b|"
    r"identity|address proof|harass|display|website|call centre|telephone|"
    r"customer service|borrower|guarantor|training|signage|holiday", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")


def readability(c) -> int:
    """Higher = easier for a non-specialist to turn into a verifiable question."""
    t = c["text"]
    return (
        2 * len(set(m.lower() for m in PLAIN_RE.findall(t)))
        - 3 * len(JARGON_RE.findall(t))
        - len(ACRONYM_RE.findall(t))
    )


def prefer_readable(pool, chunks, rng, n, plain=True):
    """Sample n from pool, optionally biased toward plain-language chunks.

    With plain=False you get the corpus as it really is -- including the dense
    prudential material. That is the better eval (it covers what the system will
    actually be asked) as long as you can verify the answers; pair it with
    `python explain.py` for plain-English translations of each chunk.

    Sorting and taking the top n outright would return near-identical chunks --
    this corpus reissues the same rule per entity, so the top of any ranking is
    ten copies of one thing. Sampling from a window avoids that.
    """
    if not pool:
        return []
    if not plain:
        return rng.sample(pool, min(n, len(pool)))
    ranked = sorted(pool, key=lambda i: readability(chunks[i]), reverse=True)
    window = ranked[:max(n * 4, min(len(ranked), 40))]
    return rng.sample(window, min(n, len(window)))


def load_chunks():
    if not CHUNKS.exists():
        sys.exit("no data/chunks.jsonl -- run: python chunker.py")
    return [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]


def usable(c) -> bool:
    """Is this chunk a fair thing to ask a question about?"""
    t = c["text"].lstrip()
    return (
        c["n_chars"] >= 400
        and not t.startswith("|")            # a markdown table, not prose
        and STARTS_CLEAN_RE.match(t)         # not a mid-sentence fragment
        and c.get("subject")
    )


def family_key(c) -> str:
    """Group near-identical circulars that differ only by which entity they bind.

    Same normalisation used to measure the 74% near-duplicate figure: strip the
    entity out of the subject, keep the topic.
    """
    subj = c.get("subject") or ""
    m = re.search(r"Reserve Bank of India\s*[(\[](.+?)[)\]]", subj, re.I)
    if not m:
        return ""
    inner = m.group(1)
    parts = re.split(r"\s+[-–—:]\s+|\s+[–—]\s*|,\s", inner)
    topic = " ".join(parts[1:]).strip()
    tail = subj[m.end():].strip()
    return re.sub(r"\W+", " ", (topic + " " + tail)).strip().lower()


def pick(chunks, used_ids, counts, rng, plain=True):
    """Choose chunk indices per category, avoiding anything already drafted."""
    idx = {i: c for i, c in enumerate(chunks) if usable(c) and i not in used_ids}

    families = defaultdict(list)
    for i, c in idx.items():
        k = family_key(c)
        if k:
            families[k].append(i)
    multi = {k: v for k, v in families.items()
             if len({chunks[i]["doc_key"] for i in v}) > 1}

    amendments = [i for i, c in idx.items() if AMEND_RE.search(c.get("subject") or "")]
    base = [i for i, c in idx.items() if not AMEND_RE.search(c.get("subject") or "")]
    in_family = [i for k, v in multi.items() for i in v]

    out = defaultdict(list)

    need = TARGET["lookup"] - counts.get("lookup", 0)
    if need > 0:
        out["lookup"] = prefer_readable(base, chunks, rng, need, plain)

    taken = set(out["lookup"])
    need = TARGET["entity_scoped"] - counts.get("entity_scoped", 0)
    pool = [i for i in in_family if i not in taken and chunks[i].get("applies_to")]
    if need > 0 and pool:
        out["entity_scoped"] = prefer_readable(pool, chunks, rng, need, plain)

    taken |= set(out["entity_scoped"])
    need = TARGET["amendment"] - counts.get("amendment", 0)
    pool = [i for i in amendments if i not in taken]
    if need > 0 and pool:
        out["amendment"] = prefer_readable(pool, chunks, rng, need, plain)

    # cross_document: two chunks from *different* documents in one family, so the
    # question genuinely requires both
    taken |= set(out["amendment"])
    need = TARGET["cross_document"] - counts.get("cross_document", 0)
    pairs = []
    for k, members in multi.items():
        by_doc = defaultdict(list)
        for i in members:
            if i not in taken:
                by_doc[chunks[i]["doc_key"]].append(i)
        if len(by_doc) >= 2:
            docs = rng.sample(sorted(by_doc), 2)
            pairs.append((by_doc[docs[0]][0], by_doc[docs[1]][0]))
    if need > 0 and pairs:
        if plain:
            pairs.sort(key=lambda p: readability(chunks[p[0]]) + readability(chunks[p[1]]),
                       reverse=True)
            pairs = pairs[:max(need * 4, 20)]
        out["cross_document"] = rng.sample(pairs, min(need, len(pairs)))

    out["unanswerable"] = [None] * max(0, TARGET["unanswerable"]
                                       - counts.get("unanswerable", 0))
    return out


def block(text, indent="    "):
    """Emit text as a YAML literal block, safe against odd whitespace."""
    lines = [l.rstrip().replace("\t", "  ") for l in text.splitlines()]
    return "\n".join(indent + l if l else "" for l in lines)


def stub(qid, category, chunks, idxs):
    """One YAML entry. Metadata pre-filled; question and answer left for you."""
    answerable = "false" if category == "unanswerable" else "true"
    head = [
        f"- id: {qid}",
        f"  category: {category}",
        f"  answerable: {answerable}",
        "  question: \"\"",
        "  answer: |",
        "    ",
        "  notes: \"\"",
    ]
    if not idxs:
        head.insert(3, "  # No source chunk. Write something this corpus cannot answer")
        head.insert(4, "  # (e.g. a topic RBI doesn't cover in these circulars).")
        head.append("  gold_chunk_ids: []")
        head.append("  gold_docs: []")
        return "\n".join(head)

    docs, sources = [], []
    for i in idxs:
        c = chunks[i]
        docs.append(c["number"])
        binds = ", ".join(c.get("applies_to") or []) or "(not entity-scoped)"
        sources.append(
            f"  # --- source chunk {i} ---\n"
            f"  #   {c['number']}   {c.get('issue_date')}   {c.get('department')}\n"
            f"  #   binds:    {binds}\n"
            f"  #   document: {(c.get('subject') or '')[:90]}\n"
            f"  #   section:  {(c.get('section') or '-')[:90]}"
        )
    head.append(f"  gold_chunk_ids: {list(idxs)}")
    head.append(f"  gold_docs: {json.dumps(docs)}")
    head.append("  source_text: |")
    joined = "\n\n----- next document -----\n\n".join(chunks[i]["text"] for i in idxs)
    head.append(block(joined))
    return "\n".join(sources) + "\n" + "\n".join(head)


def read_draft():
    if not DRAFT.exists():
        return []
    data = yaml.safe_load(DRAFT.read_text(encoding="utf-8")) or []
    return data if isinstance(data, list) else []


def cmd_draft(n=None, seed=0, plain=True):
    import random

    chunks = load_chunks()
    existing = read_draft()
    used = {i for e in existing for i in (e.get("gold_chunk_ids") or [])}
    counts = defaultdict(int)
    for e in existing:
        counts[e.get("category", "?")] += 1

    if n is not None:                    # scale the whole target mix down to n
        total = sum(TARGET.values())
        for k in TARGET:
            TARGET[k] = max(1, round(TARGET[k] * n / total))

    picks = pick(chunks, used, counts, random.Random(seed), plain)
    made = sum(len(v) for v in picks.values())
    if not made:
        print("nothing to add -- targets already met. `python qa.py stats`")
        return

    next_id = len(existing) + 1
    parts = []
    for cat in ("lookup", "entity_scoped", "cross_document", "amendment", "unanswerable"):
        for item in picks.get(cat, []):
            idxs = [] if item is None else (list(item) if isinstance(item, tuple) else [item])
            parts.append(stub(f"q{next_id:03d}", cat, chunks, idxs))
            next_id += 1

    header = "" if existing else (
        "# Ground-truth question set. Fill in `question` and `answer` for each entry.\n"
        "#\n"
        "# The answer is what a correct response looks like, in your own words, based\n"
        "# ONLY on the source text shown. If the chunk doesn't actually answer the\n"
        "# question you had in mind, either change the question or set answerable: false.\n"
        "#\n"
        "# Leave `source_text` and `gold_chunk_ids` alone -- the harness uses them to\n"
        "# score whether retrieval found the right documents.\n"
        "#\n"
        "#   python qa.py stats      progress\n"
        "#   python qa.py validate   what's incomplete\n"
        "#   python qa.py export     -> data/qa.jsonl\n\n"
    )
    with open(DRAFT, "a", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write("\n\n".join(parts) + "\n\n")

    print(f"added {made} stubs -> {DRAFT}")
    for cat in TARGET:
        got = counts[cat] + len(picks.get(cat, []))
        print(f"  {cat:<16} {got}/{TARGET[cat]}")
    print(f"\nopen {DRAFT} and fill in question + answer for each.")


def _filled(e):
    """Is this entry finished?

    An answerable entry needs a written answer. An unanswerable one needs a stated
    reason, and that reason is equally at home in `notes` -- the point is that the
    judgement is recorded somewhere a reader can check, not which field holds it.
    Without one, a deliberate refusal case is indistinguishable from an entry
    someone simply didn't finish.
    """
    if not (e.get("question") or "").strip():
        return False
    if e.get("answerable") is False:
        return bool((e.get("answer") or "").strip() or (e.get("notes") or "").strip())
    return bool((e.get("answer") or "").strip())


def cmd_stats():
    entries = read_draft()
    if not entries:
        print(f"no {DRAFT} yet -- run: python qa.py draft")
        return
    done = [e for e in entries if _filled(e)]
    print(f"drafted   {len(entries)}")
    print(f"completed {len(done)}  ({len(done) * 100 // max(len(entries), 1)}%)")
    print()
    by = defaultdict(lambda: [0, 0])
    for e in entries:
        c = e.get("category", "?")
        by[c][0] += 1
        by[c][1] += 1 if _filled(e) else 0
    print(f"{'category':<18}{'done':>6}{'drafted':>9}{'target':>8}")
    for cat in list(TARGET) + [c for c in by if c not in TARGET]:
        n, d = by.get(cat, [0, 0])
        print(f"{cat:<18}{d:>6}{n:>9}{TARGET.get(cat, '-'):>8}")


def cmd_validate():
    entries = read_draft()
    if not entries:
        print(f"no {DRAFT} yet -- run: python qa.py draft")
        return
    problems, ids = [], set()
    for e in entries:
        qid = e.get("id", "?")
        if qid in ids:
            problems.append(f"{qid}: duplicate id")
        ids.add(qid)
        q = (e.get("question") or "").strip()
        a = (e.get("answer") or "").strip()
        n = (e.get("notes") or "").strip()
        if not q:
            problems.append(f"{qid}: no question")
        elif len(q) < 12:
            problems.append(f"{qid}: question looks too short -- {q!r}")
        if e.get("answerable") is False:
            # An unanswerable entry needs its reason recorded somewhere checkable --
            # either field will do. Without one there's no way to tell a deliberate
            # refusal case from an entry nobody finished.
            if not (a or n):
                problems.append(f"{qid}: answerable=false but no reason in answer or notes")
        else:
            if not a:
                problems.append(f"{qid}: no answer")
            if not e.get("gold_docs"):
                problems.append(f"{qid}: answerable but no gold_docs")

    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:40]:
            print("  " + p)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
    else:
        print(f"all {len(entries)} entries valid")


def cmd_export():
    entries = read_draft()
    ready = [e for e in entries if _filled(e)]
    if not ready:
        print("nothing complete to export yet")
        return
    ROOT.mkdir(exist_ok=True)
    with open(EXPORT, "w", encoding="utf-8") as f:
        for e in ready:
            f.write(json.dumps({
                "id": e.get("id"),
                "category": e.get("category"),
                "question": (e.get("question") or "").strip(),
                "answer": (e.get("answer") or "").strip(),
                "answerable": e.get("answerable", True),
                "gold_docs": e.get("gold_docs") or [],
                "gold_chunk_ids": e.get("gold_chunk_ids") or [],
                "notes": (e.get("notes") or "").strip(),
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
    skipped = len(entries) - len(ready)
    print(f"exported {len(ready)} -> {EXPORT}" + (f"  ({skipped} still incomplete)" if skipped else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft", help="add stratified stubs to the draft file")
    d.add_argument("--n", type=int, default=None, help="roughly how many to add")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-plain", action="store_true",
                   help="don't bias toward plain-language chunks; sample the corpus as-is")
    sub.add_parser("stats", help="progress and category coverage")
    sub.add_parser("validate", help="report incomplete or malformed entries")
    sub.add_parser("export", help="write data/qa.jsonl for the eval harness")
    args = ap.parse_args()

    {"draft": lambda: cmd_draft(args.n, args.seed, not args.no_plain),
     "stats": cmd_stats,
     "validate": cmd_validate,
     "export": cmd_export}[args.cmd]()
