"""
The API contract, as Pydantic models. `frontend/src/types.ts` mirrors these by
hand; if a field changes here it changes there in the same commit.

Citation verification is reported as `status` + `found_in`, not a bare boolean.
A boolean would erase the distinction the README spends a section on: a quote that
occurs in five sibling circulars verifies as `ok` against all of them, so `ok`
cannot confirm attribution there. `verified` is kept as a convenience for the
common check, but it is derived from `status` and never the only thing sent.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    # Stripped before the length check, so "   " is rejected as empty rather than
    # accepted as a three-character question.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=MIN_QUESTION_CHARS, max_length=MAX_QUESTION_CHARS)


class Circular(BaseModel):
    number: str
    ref_code: str | None
    title: str | None
    issue_date: str | None
    department: str | None
    applies_to: list[str]
    source_url: str | None


class Scores(BaseModel):
    """Each ranking stage's view of one passage. See retrieval.search(with_scores)."""
    dense_rank: int | None       # position in the dense top-50; None if absent
    bm25_rank: int | None        # position in the BM25 top-50; None if absent
    rrf: float                   # fused score -- what built the candidate pool
    rerank: float | None         # cross-encoder score -- what ordered the final k


class Source(BaseModel):
    rank: int                    # 1-based; what citations' `source` refers to
    chunk_id: int
    text: str
    section: str | None
    circular: Circular
    scores: Scores


class Highlight(BaseModel):
    source: int                  # which Source the span is in
    start: int                   # character offsets into that Source's `text`
    end: int


class Citation(BaseModel):
    id: int                      # 1-based marker number shown in the UI
    source: int                  # the Source the model attributed the quote to
    quote: str
    status: Literal["ok", "wrong_source", "not_found"]
    verified: bool               # status == "ok"
    found_in: list[int]          # every Source containing the quote
    highlight: Highlight | None  # None when the quote is in no retrieved Source


class RetrievalInfo(BaseModel):
    entities: list[str]
    filter_applied: bool
    reranked: bool
    candidates: int              # pool size per channel before reranking
    k: int


class Timings(BaseModel):
    retrieval: int
    generation: int | None


class AskResponse(BaseModel):
    question: str
    answer: str | None           # None whenever generation did not run
    insufficient_context: bool
    generation: Literal["ok", "rate_limited", "unavailable"]
    notice: str | None           # why generation did not run, in plain words
    citations: list[Citation]
    sources: list[Source]
    retrieval: RetrievalInfo
    timings_ms: Timings
    model: str | None


class Health(BaseModel):
    status: Literal["ok", "starting"]
    chunks: int
    models_loaded: bool
    generation_enabled: bool
