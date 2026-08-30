"""
Tests for the ranking machinery in retrieval.py.

WHAT IS WORTH TESTING HERE, AND WHAT IS NOT
Retrieval quality is measured, not asserted: whether the reranker is worth twelve
questions is a question for `retrieval_lab.py`, against the answer key, and no
unit test can or should have an opinion about it. What belongs here is the layer
underneath -- the parts with a right answer that does not depend on the corpus,
the model, or anything downloaded.

The entity mask gets the most attention on purpose. Its OR-empty clause is the
single most load-bearing line in the retrieval path: drop it and 41 circulars
that bind no particular institution become unreachable for every filtered query,
which is a silent recall hole rather than a visible error. Exactly the kind of
regression that survives code review and shows up as an unexplained eval drop
three weeks later.
"""

import numpy as np
import pytest

import retrieval as R


# ---------------------------------- BM25 --------------------------------------
DOCS = [
    "The bank shall report counterfeit notes to the police station.",
    "All India Financial Institutions include EXIM Bank and NABARD.",
    "The bank shall maintain a record of soiled and mutilated banknotes.",
    "Nothing in this paragraph applies to Payments Banks.",
]


@pytest.fixture(scope="module")
def bm25():
    return R.BM25(DOCS)


def test_bm25_ranks_the_document_containing_a_rare_term(bm25):
    """The whole reason a lexical channel is here: rare strings are strong signals."""
    scores = bm25.score("NABARD")
    assert int(np.argmax(scores)) == 1
    assert scores[1] > 0


def test_bm25_ignores_terms_absent_from_the_vocabulary(bm25):
    """An unknown term must contribute nothing, not raise and not match everything."""
    assert bm25.score("cryptocurrency").sum() == 0.0


def test_bm25_scores_every_document(bm25):
    assert bm25.score("bank").shape == (len(DOCS),)


def test_bm25_common_term_does_not_beat_a_rare_one(bm25):
    """idf has to actually be doing something: 'bank' is in three of four docs."""
    common = bm25.score("bank").max()
    rare = bm25.score("NABARD").max()
    assert rare > common


# ------------------------------- entity mask -----------------------------------
CHUNKS = [
    {"applies_to": ["Payments Banks"]},
    {"applies_to": ["Small Finance Banks"]},
    {"applies_to": []},                                   # FEMA notice, binds nobody
    {"applies_to": ["Payments Banks", "Small Finance Banks"]},
]


def test_entity_mask_admits_the_asked_class_and_the_unscoped():
    mask = R.entity_mask(CHUNKS, ["Payments Banks"])
    assert list(mask) == [True, False, True, True]


def test_entity_mask_or_empty_clause_is_present():
    """The 41 non-entity-scoped circulars must stay reachable under every filter.

    Asserted separately from the test above, and with a class that matches nothing,
    so that a change making the mask strict fails here with an obvious message
    rather than looking like an ordinary off-by-one in a list comparison.
    """
    mask = R.entity_mask(CHUNKS, ["Urban Co-operative Banks"])
    assert mask[2], "unscoped chunks must survive the entity filter"
    assert not mask[0] and not mask[1]


def test_entity_mask_accepts_several_classes():
    """A comparison question names two, and keeping only the first would filter
    away half of what it is asking about."""
    mask = R.entity_mask(CHUNKS, ["Payments Banks", "Small Finance Banks"])
    assert all(mask)


def test_entity_mask_is_none_when_no_entity_was_detected():
    assert R.entity_mask(CHUNKS, None) is None
    assert R.entity_mask(CHUNKS, []) is None


# --------------------------------- top_k ----------------------------------------
def test_top_k_returns_indices_best_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7], dtype="float32")
    assert R.top_k(scores, None, 3) == [1, 3, 2]


def test_top_k_respects_the_mask():
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype="float32")
    mask = np.array([False, True, False, True])
    assert R.top_k(scores, mask, 4) == [1, 3]


def test_top_k_returns_fewer_than_k_when_the_filter_is_narrow():
    """A narrow filter must yield a short list, never k rows padded with
    disqualified ones -- callers use len(), and padding would be silent."""
    scores = np.array([0.9, 0.8, 0.7], dtype="float32")
    mask = np.array([True, False, False])
    assert R.top_k(scores, mask, 8) == [0]


def test_top_k_of_an_empty_filter_is_empty():
    scores = np.array([0.9, 0.8], dtype="float32")
    assert R.top_k(scores, np.array([False, False]), 5) == []


# ----------------------------------- RRF -----------------------------------------
def test_rrf_rewards_agreement_between_channels():
    """A chunk both channels like beats one that only tops a single list."""
    dense = [10, 20, 30]
    lexical = [40, 20, 50]
    assert R.rrf([dense, lexical])[0] == 20


def test_rrf_reads_only_rank_not_score():
    """The reason RRF was chosen over a weighted sum: cosine and BM25 live on
    different scales, so anything reading magnitudes is fitted to a normalisation
    rather than to the data. Identical orderings must fuse identically."""
    a = R.rrf([[1, 2, 3], [3, 2, 1]])
    b = R.rrf([[1, 2, 3], [3, 2, 1]])
    assert a == b
    assert set(a) == {1, 2, 3}


def test_rrf_preserves_a_single_ranking():
    assert R.rrf([[7, 8, 9]]) == [7, 8, 9]


def test_fuse_truncates_each_channel_before_fusing():
    """Fusing full-corpus rankings would let a chunk sitting 4,000th on both lists
    outrank one sitting 9th on a single list, which is not what either channel is
    saying."""
    dense = list(range(100))
    lexical = list(range(100, 200))
    out = R.fuse(dense, lexical, depth=10)
    assert len(out) == 10
    assert set(out) <= set(dense[:10]) | set(lexical[:10])


# -------------------------------- header text -------------------------------------
def test_header_text_prepends_subject_and_section():
    c = {"subject": "Fraud Risk Management", "section": "Chapter II > A. Reporting",
         "text": "The bank shall report."}
    out = R.header_text(c)
    assert out.startswith("Fraud Risk Management")
    assert "Chapter II > A. Reporting" in out
    assert out.endswith("The bank shall report.")


def test_header_text_falls_back_to_bare_text():
    """A chunk with no subject and no section must not gain leading blank lines --
    those are tokens the cross-encoder pays for and learns nothing from."""
    c = {"subject": None, "section": None, "text": "The bank shall report."}
    assert R.header_text(c) == "The bank shall report."
