"""
index_qdrant.py
---------------
Index the embedded corpus into Qdrant and verify the entity pre-filter survives
the move out of NumPy.

    python index_qdrant.py build          # create collection + upsert 9,278 points
    python index_qdrant.py test           # re-run the 8-query entity benchmark
    python index_qdrant.py ask "..." --entity "Payments Banks"

    # against a real server instead of embedded mode:
    docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
    python index_qdrant.py build --url http://localhost:6333

Why a vector store at all, when brute-force NumPy already searches 9,278 vectors in
milliseconds: it is not for speed. This corpus is ~74% near-duplicate documents (the
same Directions reissued per regulated-entity class), so retrieval quality depends on
a metadata filter running *inside* the search rather than after it. That is the thing
a vector store provides and a dot product does not. `test` exists to prove the ported
filter reproduces the NumPy numbers exactly (35% -> 92%), not approximately.
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
from qdrant_client import QdrantClient, models

import env  # noqa: F401  -- loads .env before QDRANT_URL is read in the CLI below

ROOT = Path("data")
CHUNKS = ROOT / "chunks.jsonl"
VECS = ROOT / "embeddings.npy"
LOCAL_PATH = ROOT / "qdrant"          # embedded mode; no server, no Docker

COLLECTION = "rbi_circulars"
DIM = 384
BATCH = 256

# Payload fields worth an index. `applies_to` is the one the corpus depends on;
# the others are cheap and make later filtering (by department, by date) possible.
INDEXED_FIELDS = {
    "applies_to": models.PayloadSchemaType.KEYWORD,
    "department": models.PayloadSchemaType.KEYWORD,
    "issue_date": models.PayloadSchemaType.KEYWORD,
    "doc_key":    models.PayloadSchemaType.KEYWORD,
}


def connect(url=None):
    """A real server when given a URL, otherwise Qdrant's embedded mode.

    Same client API either way, so nothing downstream changes. Embedded mode does
    exact search rather than HNSW -- at 9,278 vectors that is not a meaningful
    difference, and it means this runs without Docker.

    The explicit timeout matters: the client defaults to 5s, and an upsert batch
    carrying full chunk text plus `wait=True` (which blocks until the server has
    actually indexed it) routinely exceeds that. The failure looks like a dead
    server -- it isn't; it is a client giving up early.
    """
    if url:
        return QdrantClient(url=url, timeout=120), f"server {url}"
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(LOCAL_PATH)), f"embedded ({LOCAL_PATH})"


def load_corpus():
    if not CHUNKS.exists() or not VECS.exists():
        sys.exit("missing corpus -- run: python chunker.py && python embed.py build")
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    vecs = np.load(VECS)
    if len(vecs) != len(chunks):
        sys.exit(f"embeddings ({len(vecs)}) and chunks ({len(chunks)}) disagree -- rebuild")
    return chunks, vecs


def entity_filter(entity):
    """`applies_to contains any of <entity> OR applies_to is empty`.

    The OR-empty clause is not optional. 41 circulars are legitimately not
    entity-scoped (FEMA notices, district reassignments); strict containment would
    make them unreachable for every filtered query -- a silent recall hole, not a
    visible error. `should` is Qdrant's OR, satisfied by at least one condition.

    `entity` may be a list: a question comparing two classes ("do the NBFC rules
    differ from the HFC rules?") must see both, and filtering to whichever one was
    detected first would make the comparison unanswerable while looking like a
    normal filtered query.
    """
    entities = [entity] if isinstance(entity, str) else list(entity)
    return models.Filter(should=[
        models.FieldCondition(key="applies_to", match=models.MatchAny(any=entities)),
        models.IsEmptyCondition(is_empty=models.PayloadField(key="applies_to")),
    ])


def build(url=None, recreate=True):
    chunks, vecs = load_corpus()
    client, where = connect(url)
    print(f"connected: {where}")

    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        )
        print(f"created collection {COLLECTION!r} (dim={DIM}, cosine)")

    for field, schema in INDEXED_FIELDS.items():
        client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
    print(f"payload indexes: {', '.join(INDEXED_FIELDS)}")

    for start in range(0, len(chunks), BATCH):
        batch = chunks[start:start + BATCH]
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=start + i,                       # row order == chunks.jsonl order
                    vector=vecs[start + i].tolist(),
                    payload={
                        "doc_key":    c["doc_key"],
                        "number":     c["number"],
                        "ref_code":   c["ref_code"],
                        "issue_date": c["issue_date"],
                        "department": c["department"],
                        "applies_to": c["applies_to"],   # list -> Qdrant keyword array
                        "subject":    c["subject"],
                        "section":    c["section"],
                        "source_url": c["source_url"],
                        "chunk_index": c["chunk_index"],
                        "text":       c["text"],
                    },
                )
                for i, c in enumerate(batch)
            ],
            wait=True,
        )
        print(f"  upserted {min(start + BATCH, len(chunks))}/{len(chunks)}", end="\r")

    count = client.count(COLLECTION, exact=True).count
    print(f"\nindexed {count} points into {COLLECTION!r}")
    if count != len(chunks):
        sys.exit(f"count mismatch: expected {len(chunks)}, got {count}")


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Same eight queries as the NumPy benchmark, so the numbers are directly comparable.
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


def query(client, model, text, k=10, entity=None):
    qv = model.encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
    res = client.query_points(
        COLLECTION,
        query=qv,
        limit=k,
        query_filter=entity_filter(entity) if entity else None,
        with_payload=True,
    )
    return res.points


def test(url=None, k=10):
    client, where = connect(url)
    model = get_model()
    print(f"connected: {where}\n")
    print(f"top-{k} entity precision, filter applied inside Qdrant\n")
    head = f"{'query topic':<34}{'unfiltered':>12}{'pre-filtered':>14}"
    print(head)
    print("-" * len(head))

    raw_tot = filt_tot = raw_n = filt_n = 0
    for q, entity in ENTITY_QUERIES:
        raw = query(client, model, q, k)
        filt = query(client, model, q, k, entity=entity)
        rh = sum(1 for p in raw if entity in (p.payload["applies_to"] or []))
        fh = sum(1 for p in filt if entity in (p.payload["applies_to"] or []))
        raw_tot += rh; raw_n += len(raw)
        filt_tot += fh; filt_n += len(filt)
        print(f"{entity[:32]:<34}{rh:>7}/{len(raw):<4}{fh:>9}/{len(filt):<4}")

    print("-" * len(head))
    print(f"{'TOTAL':<34}{raw_tot:>7}/{raw_n:<4}{filt_tot:>9}/{filt_n:<4}")
    print(f"{'precision':<34}{raw_tot/max(raw_n,1):>11.0%}{filt_tot/max(filt_n,1):>13.0%}")
    print("\nNumPy baseline was 28/80 (35%) -> 74/80 (92%). These must match.")


def ask(text, url=None, k=5, entity=None):
    client, where = connect(url)
    model = get_model()
    pts = query(client, model, text, k, entity=entity)
    print(f'QUERY: "{text}"')
    print(f"filter: {entity or '(none)'}   [{where}]\n")
    for rank, p in enumerate(pts, 1):
        pay = p.payload
        ents = ", ".join(pay["applies_to"] or []) or "(unscoped)"
        print(f"{rank}. {p.score:.3f}  {pay['number']}  [{ents[:44]}]")
        if pay.get("section"):
            print(f"   section: {pay['section'][:70]}")
        print(f"   {pay['text'][:150].strip()}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "test", "ask"])
    ap.add_argument("text", nargs="?", help="question, for `ask`")
    ap.add_argument("--url", default=os.getenv("QDRANT_URL"),
                    help="Qdrant server URL; omit for embedded mode")
    ap.add_argument("--entity", help="restrict to a regulated-entity class")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()

    if a.cmd == "build":
        build(a.url)
    elif a.cmd == "test":
        test(a.url, a.k)
    else:
        if not a.text:
            sys.exit('ask needs a question: python index_qdrant.py ask "..."')
        ask(a.text, a.url, a.k, a.entity)
