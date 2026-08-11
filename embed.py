"""
embed.py
--------
Embed the chunk corpus with BGE and probe nearest-neighbour quality.

    python embed.py build          # -> data/embeddings.npy  (~9 min, CPU)
    python embed.py probe          # random chunks + their nearest neighbours
    python embed.py entity-test    # the experiment that matters (see below)

`entity-test` measures the failure this corpus is built around. RBI reissues the same
Directions once per regulated-entity class, so ~74% of documents sit in a near-duplicate
family. The hypothesis from corpus analysis was that embeddings cannot separate those
variants -- the discriminating tokens are a rounding error in a 400-word passage -- so an
unfiltered top-k fills with rules that bind someone else. This measures whether that is
actually true, and what the `applies_to` pre-filter recovers.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path("data")
CHUNKS = ROOT / "chunks.jsonl"
VECS = ROOT / "embeddings.npy"

MODEL = "BAAI/bge-small-en-v1.5"
# BGE v1.5 asks for an instruction prefix on queries only; passages go in bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks():
    if not CHUNKS.exists():
        sys.exit("no data/chunks.jsonl -- run: python chunker.py")
    return [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)


def build():
    chunks = load_chunks()
    model = get_model()
    print(f"embedding {len(chunks)} chunks with {MODEL} ...")
    vecs = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        normalize_embeddings=True,      # cosine == dot product downstream
        show_progress_bar=True,
    ).astype("float32")
    np.save(VECS, vecs)
    print(f"wrote {VECS}  shape={vecs.shape}")


def load_vecs(chunks):
    if not VECS.exists():
        sys.exit("no data/embeddings.npy -- run: python embed.py build")
    v = np.load(VECS)
    if len(v) != len(chunks):
        sys.exit(f"embeddings ({len(v)}) and chunks ({len(chunks)}) disagree -- rebuild")
    return v


def search(vecs, qvec, k, mask=None):
    """Cosine similarity over normalised vectors. `mask` is the pre-filter.

    Returns at most k rows, and fewer when the filter leaves fewer candidates --
    a narrow filter must yield a short result list, never k rows padded with
    disqualified ones. Callers should use len() of the result, not k.
    """
    sims = vecs @ qvec
    if mask is not None:
        eligible = int(mask.sum())
        if eligible == 0:
            return np.empty(0, dtype=np.intp), sims
        k = min(k, eligible)           # never ask for more than the filter allows
        sims = np.where(mask, sims, -np.inf)
    k = min(k, len(sims))
    if k <= 0:
        return np.empty(0, dtype=np.intp), sims
    idx = np.argpartition(-sims, k - 1)[:k]
    return idx[np.argsort(-sims[idx])], sims


def probe(n=6, k=5):
    """Sanity check: does a chunk's nearest neighbour look related to it?"""
    chunks = load_chunks()
    vecs = load_vecs(chunks)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(chunks), n, replace=False):
        c = chunks[i]
        idx, sims = search(vecs, vecs[i], k + 1)
        print("=" * 78)
        print(f"SEED  {c['number']}  [{', '.join(c['applies_to']) or 'no entity'}]")
        print(f"      {c['text'][:110].strip()}")
        for j in idx:
            if j == i:
                continue
            n_ = chunks[j]
            same = "same-doc " if n_["doc_key"] == c["doc_key"] else ""
            print(f"  {sims[j]:.3f} {same}{n_['number']:<22} "
                  f"[{', '.join(n_['applies_to'])[:34] or 'no entity'}]")
            print(f"        {n_['text'][:96].strip()}")


# The queries are entity-scoped on purpose: each names a topic that exists as a
# near-duplicate family across many entity classes, plus the one class being asked about.
ENTITY_QUERIES = [
    ("What are the KYC requirements for Payments Banks?", "Payments Banks"),
    ("Fraud risk management directions for Urban Co-operative Banks", "Urban Co-operative Banks"),
    ("Income recognition and asset classification norms for NBFCs", "Non-Banking Financial Companies"),
    ("Recovery agent conduct rules for Small Finance Banks", "Small Finance Banks"),
    ("Capital adequacy requirements for Regional Rural Banks", "Regional Rural Banks"),
    ("Responsible business conduct rules for Local Area Banks", "Local Area Banks"),
    ("Financial statement disclosure requirements for Commercial Banks", "Commercial Banks"),
    ("Supervisory return filing obligations for Rural Co-operative Banks", "Rural Co-operative Banks"),
]


def entity_test(k=10):
    chunks = load_chunks()
    vecs = load_vecs(chunks)
    model = get_model()
    applies = [set(c["applies_to"]) for c in chunks]

    print(f"top-{k} entity purity: how many retrieved chunks bind the entity asked about\n")
    head = f"{'query topic':<34}{'unfiltered':>12}{'pre-filtered':>14}"
    print(head)
    print("-" * len(head))

    raw_tot = filt_tot = raw_n = filt_n = 0
    empties = np.array([not a for a in applies])
    for q, entity in ENTITY_QUERIES:
        qv = model.encode(QUERY_PREFIX + q, normalize_embeddings=True).astype("float32")

        idx, _ = search(vecs, qv, k)
        raw_hits = sum(1 for i in idx if entity in applies[i])

        # the filter that must run *during* search: bound to this entity, or unscoped
        mask = np.array([entity in a for a in applies]) | empties
        fidx, _ = search(vecs, qv, k, mask=mask)
        filt_hits = sum(1 for i in fidx if entity in applies[i])

        # denominators come from what search actually returned, since a narrow
        # filter legitimately yields fewer than k rows
        raw_tot += raw_hits;  raw_n += len(idx)
        filt_tot += filt_hits; filt_n += len(fidx)
        print(f"{entity[:32]:<34}{raw_hits:>7}/{len(idx):<4}{filt_hits:>9}/{len(fidx):<4}")

    print("-" * len(head))
    print(f"{'TOTAL':<34}{raw_tot:>7}/{raw_n:<4}{filt_tot:>9}/{filt_n:<4}")
    print(f"{'precision':<34}{raw_tot/max(raw_n,1):>11.0%}{filt_tot/max(filt_n,1):>13.0%}")


def worked_example(k=10):
    """One query, shown in full -- the numbers above are easy to disbelieve."""
    chunks = load_chunks()
    vecs = load_vecs(chunks)
    model = get_model()
    q, entity = ENTITY_QUERIES[0]
    qv = model.encode(QUERY_PREFIX + q, normalize_embeddings=True).astype("float32")
    idx, sims = search(vecs, qv, k)
    print(f'QUERY: "{q}"')
    print(f'asking about: {entity}\n')
    print(f"{'#':<4}{'sim':<8}{'binds':<34}{'document':<22}")
    print("-" * 74)
    for rank, i in enumerate(idx, 1):
        c = chunks[i]
        ok = "OK " if entity in c["applies_to"] else "-- "
        ents = ", ".join(c["applies_to"])[:31] or "(unscoped)"
        print(f"{ok}{rank:<2}{sims[i]:<8.3f}{ents:<34}{c['number']:<22}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "probe", "entity-test", "example"])
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    {"build": build, "probe": probe,
     "entity-test": lambda: entity_test(a.k),
     "example": lambda: worked_example(a.k)}[a.cmd]()
