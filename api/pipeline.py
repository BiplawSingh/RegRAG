"""
The pipeline, called the way a web server needs it called.

Nothing here retrieves, ranks, generates or verifies -- those are imported from
retrieval.py and rag.py, the same modules the evaluation measured. This file only
does what a server needs and a CLI does not:

  - loads every model once at startup, not on the first visitor's request
  - runs the dense channel as a numpy dot product instead of a Qdrant query
    (the substitution `retrieval_lab.py baseline` verifies question-for-question)
  - never lets a missing key or an exhausted quota take retrieval down with it
  - turns the pipeline's dicts into the typed contract in schemas.py

It deliberately does not call rag.run_one(): that prints to a terminal and
appends to data/log.jsonl, neither of which a public endpoint should do.
"""

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

import rag
import retrieval as R
from api.highlight import locate
from api.schemas import (AskResponse, Circular, Citation, Highlight, RetrievalInfo,
                         Scores, Source, Timings)

log = logging.getLogger("regrag.api")

# rag.py and retrieval.py resolve data/ relative to the working directory.
ROOT = Path(__file__).resolve().parent.parent
VECS = Path("data/embeddings.npy")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Pipeline:
    def __init__(self, k=8):
        self.k = k
        self.ready = False
        self.chunks = []
        # Embedding and reranking are CPU-bound and share module-level caches
        # that were written for a single-threaded CLI. One at a time is both the
        # safe choice and, on a 2-vCPU host, not measurably slower.
        self._retrieval_lock = threading.Lock()

    # ------------------------------------------------------------- startup --
    def warm(self):
        """Load everything a question needs: ~30s once, instead of on a request."""
        from sentence_transformers import SentenceTransformer

        os.chdir(ROOT)
        self.chunks = R.corpus()
        self.vecs = np.load(VECS)
        if len(self.vecs) != len(self.chunks):
            raise RuntimeError(f"embeddings ({len(self.vecs)}) and chunks "
                               f"({len(self.chunks)}) disagree -- rebuild")
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        R.bm25()
        R.get_reranker()
        rag.detect_entity("warm-up")          # loads the entity detector module
        self.ready = True

    def generation_enabled(self):
        return bool(os.getenv("GROQ_API_KEY"))

    # -------------------------------------------------------------- stages --
    def retrieve(self, question):
        """(chunks with scores, detected entities). Entity filter always on."""
        with self._retrieval_lock:
            entities = rag.detect_entity(question) or []
            qv = self.encoder.encode(QUERY_PREFIX + question,
                                     normalize_embeddings=True).astype("float32")
            mask = R.entity_mask(self.chunks, entities)
            dense = R.top_k(self.vecs @ qv, mask, R.DEPTH)
            chunks = R.search(question, k=self.k, entity=entities or None,
                              dense_ids=dense, with_scores=True)
        return chunks, entities

    def generate(self, question, chunks):
        """(rag.Answer, seconds). Raises whatever rag.generate raises."""
        ans, secs, _usage, _prompt = rag.generate(question, chunks)
        return ans, secs

    # ---------------------------------------------------------------- ask --
    def ask(self, question, generation_allowed):
        t0 = time.time()
        chunks, entities = self.retrieve(question)
        retrieval_ms = int((time.time() - t0) * 1000)

        ans, gen_ms, status, notice = None, None, "ok", None
        if not self.generation_enabled():
            status = "unavailable"
            notice = ("Answer generation is switched off on this server, so only the "
                      "retrieved passages are shown.")
        elif not generation_allowed:
            status = "rate_limited"
            notice = ("This demo runs on a free daily model allowance, and it is used up "
                      "for now. Retrieval still ran: these are the passages an answer "
                      "would have been written from.")
        else:
            try:
                ans, secs = self.generate(question, chunks)
                gen_ms = int(secs * 1000)
            except Exception:                              # noqa: BLE001
                # Almost always the provider's quota or an outage. Retrieval has
                # already succeeded, and it is worth showing on its own.
                log.exception("generation failed")
                status = "unavailable"
                notice = ("The language model could not be reached just now. Retrieval "
                          "still ran: these are the passages an answer would have "
                          "been written from.")

        checks = rag.verify(ans, chunks) if ans else []
        return AskResponse(
            question=question,
            answer=ans.answer if ans else None,
            insufficient_context=bool(ans and ans.insufficient_context),
            generation="ok" if ans else status,
            notice=notice,
            citations=build_citations(checks, chunks),
            sources=build_sources(chunks),
            retrieval=RetrievalInfo(entities=entities, filter_applied=bool(entities),
                                    reranked=R.RERANK, candidates=R.DEPTH, k=self.k),
            timings_ms=Timings(retrieval=retrieval_ms, generation=gen_ms),
            model=rag.GEN_MODEL if ans else None,
        )


def build_sources(chunks):
    return [
        Source(
            rank=i,
            chunk_id=c["chunk_id"],
            text=c["text"],
            section=c.get("section"),
            circular=Circular(
                number=c["number"],
                ref_code=c.get("ref_code"),
                title=c.get("subject"),
                issue_date=c.get("issue_date"),
                department=c.get("department"),
                applies_to=c.get("applies_to") or [],
                source_url=c.get("source_url"),
            ),
            scores=Scores(**c["scores"]),
        )
        for i, c in enumerate(chunks, 1)
    ]


def build_citations(checks, chunks):
    """verify()'s results, plus where to highlight each quote.

    The highlight goes where the quote actually IS: the cited source when it
    verified, the first source containing it when it was misattributed, nowhere
    when it is in no source at all.
    """
    out = []
    for n, c in enumerate(checks, 1):
        where = c["source_id"] if c["status"] == "ok" else \
            (c["found_in"][0] if c["found_in"] else None)
        span = locate(c["quote"], chunks[where - 1]["text"]) if where else None
        out.append(Citation(
            id=n,
            source=c["source_id"],
            quote=c["quote"],
            status=c["status"],
            verified=c["status"] == "ok",
            found_in=c["found_in"],
            highlight=Highlight(source=where, start=span[0], end=span[1]) if span else None,
        ))
    return out
