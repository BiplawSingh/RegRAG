"""
retrieval.py
------------
Hybrid candidate generation and cross-encoder reranking -- the ranking half of
the pipeline, factored out so that exactly one implementation exists.

    python retrieval.py ask "Who can be an Internal Ombudsman?"
    python retrieval.py ask "..." --no-rerank        # the ablation
    python retrieval.py timing                       # what reranking costs

WHY THIS FILE EXISTS
`rag.retrieve` used to be a dense top-k out of Qdrant and nothing else. Measured
on the 69 answerable questions, that found the chunk that actually answers the
question 42 times. The same questions, same filter, same embeddings, with a
lexical channel added and a cross-encoder re-sorting the pool: 54. The pipeline
was leaving a quarter of its own recall in the candidate set, unranked.

Three stages, and each is here for a measured reason:

  1. TWO CHANNELS. Dense retrieval represents a 250-word passage as 384 numbers
     before it has seen the question, which is fine for topical similarity and
     poor for exact strings -- and this corpus turns on exact strings ("All India
     Financial Institutions", "out of order", section 45-IA). BM25 has the
     opposite bias. Measured at depth 50, dense alone surfaces the answering
     chunk for 59 of 69 questions and BM25 for 50, but between them 61: the
     channels miss different questions, which is the only reason fusing them is
     worth anything.

  2. RRF, NOT A WEIGHTED SUM. Cosine sits in [-1, 1] and clusters near 0.7 here;
     BM25 is unbounded and scales with query length. Any weighted sum of the raw
     scores is really one channel plus noise, with a weight fitted to whatever
     normalisation happened to be in use. Reciprocal-rank fusion reads only the
     ordering, so there is no scale to get wrong.

  3. A CROSS-ENCODER, OVER 50 CANDIDATES AND NOT 9,278. It reads question and
     passage together, which is what lets it tell the paragraph that states a
     rule from another paragraph of the document that states the rule -- and
     that distinction is precisely what the chunk-vs-document gap was made of.
     It is also ~50x the cost per pair, so it re-sorts a pool rather than
     searching one.

The reranker is shown the document subject and section heading above the passage
text. That detail is worth two-thirds of the gain and it is not obvious: the same
headers, prepended to the *passage embeddings*, made dense retrieval slightly
worse. A bi-encoder given a header produces a blurrier passage vector; a
cross-encoder given a header gets evidence it can weigh against the question.
Same information, opposite sign, because the two models use it differently.
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("data")
CHUNKS = ROOT / "chunks.jsonl"

# How many candidates each channel contributes before reranking. 50 was measured,
# not picked: chunk recall in the fused pool is 54/69 at depth 20, 61/69 at 50,
# and 63/69 at 100 -- and the cross-encoder cost is linear in this number, so
# going past 50 buys two questions for double the latency.
DEPTH = int(os.getenv("RETRIEVAL_DEPTH", "50"))

# Reranking is on by default and can be switched off for the ablation. Kept as an
# env toggle rather than a code edit so `eval.py retrieval` can measure both
# without the two runs differing in anything else.
RERANK = os.getenv("RETRIEVAL_RERANK", "1") not in ("0", "false", "no")
# The small one, on measurement rather than on principle. bge-reranker-base (278M)
# and ms-marco-MiniLM-L-6-v2 (22M) both score 54/69 here -- identical chunk recall,
# same questions fixed -- and the small model is an order of magnitude faster on
# CPU. Reranking 50 candidates is the pipeline's slowest stage either way, so the
# larger model would have to earn its latency with recall, and it does not.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# --------------------------------- corpus -------------------------------------
_CACHE = {}


def corpus():
    """chunks.jsonl, in row order -- which is also the chunk_id order.

    index_qdrant.build assigns point ids by row index, so a chunk's position in
    this list IS its id everywhere else in the project: in Qdrant, in
    gold_chunk_ids, in the eval rows. Loading the file here rather than trusting
    the Qdrant payload also means both channels return identically shaped records,
    instead of one shape from the vector store and another from disk.
    """
    if "chunks" not in _CACHE:
        if not CHUNKS.exists():
            sys.exit("no data/chunks.jsonl -- run: python chunker.py")
        _CACHE["chunks"] = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    return _CACHE["chunks"]


def header_text(c):
    """What the cross-encoder is shown: document subject, section path, passage.

    A paragraph reading "the period shall not exceed six months from the date of
    seizure" is, on its own, about nothing in particular. The subject line and the
    section path are what make it about the sale of seized property -- and both
    were already in the payload, already shown to the generator, and never shown
    to anything that ranks.
    """
    head = c.get("subject") or ""
    sec = c.get("section") or ""
    return f"{head}\n{sec}\n{c['text']}".strip() if (head or sec) else c["text"]


def entity_mask(chunks, entities):
    """`applies_to contains any of <entities> OR applies_to is empty`, as a mask.

    Mirrors index_qdrant.entity_filter exactly, including the OR-empty clause --
    41 circulars bind nobody in particular (FEMA notices, district
    reassignments) and strict containment would make them unreachable for every
    filtered query.
    """
    if not entities:
        return None
    want = set(entities)
    return np.array([(not c["applies_to"]) or bool(want & set(c["applies_to"]))
                     for c in chunks])


# ---------------------------------- BM25 ---------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a scipy sparse term-document matrix.

    Written out rather than pulled in as a dependency: it is forty lines, the
    corpus is 9,278 chunks, and the index builds in about two seconds -- cheaper
    than persisting it and then having to keep the persisted copy in step with
    chunks.jsonl.
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        from scipy.sparse import csr_matrix

        self.k1, self.b = k1, b
        vocab = {}
        indptr, indices, data, lengths = [0], [], [], []
        for d in docs:
            counts = Counter(tokenize(d))
            lengths.append(sum(counts.values()) or 1)
            for term, n in counts.items():
                indices.append(vocab.setdefault(term, len(vocab)))
                data.append(n)
            indptr.append(len(indices))
        self.vocab = vocab
        self.lengths = np.array(lengths, dtype="float32")
        self.avgdl = float(self.lengths.mean())
        tf = csr_matrix((np.array(data, dtype="float32"), indices, indptr),
                        shape=(len(docs), len(vocab)))
        df = np.diff(tf.tocsc().indptr).astype("float32")
        n = len(docs)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype("float32")
        self.tf = tf.tocsc()
        self.norm = (k1 * (1 - b + b * self.lengths / self.avgdl)).astype("float32")

    def score(self, query):
        out = np.zeros(self.tf.shape[0], dtype="float32")
        for term in tokenize(query):
            j = self.vocab.get(term)
            if j is None:                    # unknown terms contribute nothing
                continue
            col = self.tf[:, j]
            rows, freqs = col.indices, col.data
            out[rows] += self.idf[j] * (freqs * (self.k1 + 1)) / (freqs + self.norm[rows])
        return out


def bm25():
    if "bm25" not in _CACHE:
        _CACHE["bm25"] = BM25([c["text"] for c in corpus()])
    return _CACHE["bm25"]


# --------------------------------- ranking --------------------------------------
def top_k(scores, mask, k):
    s = scores if mask is None else np.where(mask, scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-s[idx])]]


def rrf(rankings, k_const=60):
    """Reciprocal-rank fusion: sum of 1/(60 + rank) across the input orderings."""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            fused[idx] += 1.0 / (k_const + rank)
    return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]


def get_reranker(model=None):
    name = model or RERANK_MODEL
    key = f"rr:{name}"
    if key not in _CACHE:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import CrossEncoder

        _CACHE[key] = CrossEncoder(name)
    return _CACHE[key]


def rerank(question, candidates, k, with_header=True, model=None):
    """Re-sort candidate chunk ids with the cross-encoder, keep the best k.

    The ceiling here is fixed by the candidate set, not by the model: whatever the
    fetch missed, no re-sorting recovers. That is why DEPTH is a measured number
    and not a default someone liked the look of.
    """
    if not candidates:
        return []
    chunks = corpus()
    pairs = [(question, header_text(chunks[j]) if with_header else chunks[j]["text"])
             for j in candidates]
    scores = get_reranker(model).predict(pairs, batch_size=32, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))
    return [candidates[j] for j in order[:k]]


def fuse(dense_ids, lexical_ids, depth=DEPTH):
    """The candidate pool: both channels' rankings, fused, truncated to depth.

    Truncating each channel before fusing matters. Over full-corpus rankings a
    chunk sitting 4,000th on both lists would outrank one sitting 9th on a single
    list, which is not what either channel is saying.
    """
    return rrf([dense_ids[:depth], lexical_ids[:depth]])[:depth]


# --------------------------------- search ----------------------------------------
def lexical_ids(question, entities, use_filter=True, depth=DEPTH):
    mask = entity_mask(corpus(), entities) if use_filter else None
    return top_k(bm25().score(question), mask, depth)


def search(question, k=8, entity=None, use_filter=True, dense_ids=None,
           do_rerank=None, depth=DEPTH):
    """Retrieve k chunks: hybrid fetch, then cross-encoder rerank.

    `dense_ids` lets the caller supply the dense ranking it already has -- the
    pipeline gets it from Qdrant, the offline bench from a numpy dot product.
    Everything after that point is shared, which is the reason the bench's numbers
    transfer to the deployed system instead of merely resembling them.
    """
    chunks = corpus()
    do_rerank = RERANK if do_rerank is None else do_rerank

    if dense_ids is None:
        dense_ids = _dense_from_qdrant(question, entity if use_filter else None, depth)
    lex = lexical_ids(question, entity, use_filter=use_filter, depth=depth)

    if not do_rerank:
        # The ablation path. Fusing without reranking is worth +1 question over
        # dense alone, which is the honest measure of how much of the gain belongs
        # to the second channel rather than to the cross-encoder: almost none of
        # it. BM25's value here is in what it puts in the POOL, not where it ranks.
        pool = fuse(dense_ids, lex, depth)
        ids = pool[:k]
    else:
        ids = rerank(question, fuse(dense_ids, lex, depth), k)

    out = []
    for j in ids:
        c = dict(chunks[j])
        c["chunk_id"] = j
        out.append(c)
    return out


def _dense_from_qdrant(question, entity, depth):
    import importlib.util

    if "iq" not in _CACHE:
        spec = importlib.util.spec_from_file_location("index_qdrant", "index_qdrant.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CACHE["iq"] = mod
        _CACHE["client"], _CACHE["where"] = mod.connect(os.getenv("QDRANT_URL",
                                                                  "http://localhost:6333"))
        _CACHE["model"] = mod.get_model()
    points = _CACHE["iq"].query(_CACHE["client"], _CACHE["model"], question,
                                k=depth, entity=entity)
    return [int(p.id) for p in points]


def where():
    return _CACHE.get("where", "not connected")


# ----------------------------------- cli -------------------------------------------
if __name__ == "__main__":
    import argparse

    import env  # noqa: F401

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("-k", type=int, default=8)
    a.add_argument("--entity", nargs="*", default=None)
    a.add_argument("--no-rerank", action="store_true")
    a.add_argument("--no-filter", action="store_true")
    t = sub.add_parser("timing", help="what each stage costs, per question")
    t.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    def detect(q):
        import importlib.util

        spec = importlib.util.spec_from_file_location("rbi_scraper", "rbi_scraper.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.scan_entities(q) or None

    if args.cmd == "ask":
        ents = args.entity or detect(args.question)
        t0 = time.time()
        hits = search(args.question, k=args.k, entity=ents,
                      use_filter=not args.no_filter,
                      do_rerank=not args.no_rerank)
        print(f'QUERY: "{args.question}"')
        print(f"entities: {ents}   rerank: {not args.no_rerank}   "
              f"{time.time() - t0:.1f}s   [{where()}]\n")
        for rank, c in enumerate(hits, 1):
            print(f"{rank:>2}. [{c['chunk_id']}] {c['number']}  "
                  f"[{', '.join(c['applies_to']) or 'unscoped'}]")
            print(f"    {(c.get('section') or '-')[:70]}")
            print(f"    {c['text'][:140].strip()}\n")
    else:
        qs = [json.loads(l) for l in open(ROOT / "qa.jsonl", encoding="utf-8")][:args.n]
        # Warm every cache first. Timing the first call would measure a model
        # download and a BM25 build, neither of which happens per question.
        search(qs[0]["question"], k=8, entity=detect(qs[0]["question"]))
        print(f"{'question':<44}{'fetch':>9}{'rerank':>9}{'total':>9}")
        for q in qs:
            ents = detect(q["question"])
            t0 = time.time()
            d = _dense_from_qdrant(q["question"], ents, DEPTH)
            lex = lexical_ids(q["question"], ents)
            t1 = time.time()
            rerank(q["question"], fuse(d, lex), 8)
            t2 = time.time()
            print(f"{q['question'][:42]:<44}{t1 - t0:>8.2f}s{t2 - t1:>8.2f}s{t2 - t0:>8.2f}s")
