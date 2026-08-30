"""
Tests for citation verification (rag.py) and answer-key validation (qa.py).

Both are checks the project's honesty rests on, and both are pure functions over
strings -- no model, no corpus, no network. If citation verification is wrong,
every "93% verified" figure in the README is wrong with it, and nothing else in
the pipeline would notice.
"""

import rag
from rag import Answer, Citation
import qa


CHUNKS = [
    {"number": "RBI/2025-26/100", "text": "The bank shall report counterfeit notes."},
    {"number": "RBI/2025-26/101", "text": "Payments Banks shall maintain a record."},
    {"number": "RBI/2025-26/102", "text": "The bank shall report counterfeit notes."},
]


def _answer(*citations):
    return Answer(answer="x", citations=list(citations), insufficient_context=False)


# ------------------------------ verification ------------------------------------
def test_quote_found_in_the_cited_source_is_ok():
    checks = rag.verify(_answer(Citation(source_id=1, quote="report counterfeit notes")),
                        CHUNKS)
    assert checks[0]["status"] == "ok"
    assert checks[0]["cited_doc"] == "RBI/2025-26/100"


def test_quote_in_the_corpus_but_not_the_cited_source_is_wrong_source():
    """Misattribution -- the failure mode that a prose-parsed citation hides."""
    checks = rag.verify(_answer(Citation(source_id=2, quote="report counterfeit notes")),
                        CHUNKS)
    assert checks[0]["status"] == "wrong_source"
    assert checks[0]["found_in"] == [1, 3]


def test_quote_in_no_source_is_not_found():
    """Fabrication. Exactly one answer in the first full run hit this."""
    checks = rag.verify(_answer(Citation(source_id=1, quote="an ECL staging framework")),
                        CHUNKS)
    assert checks[0]["status"] == "not_found"
    assert checks[0]["found_in"] == []


def test_reflowed_whitespace_still_matches():
    """A model that re-wraps a quote has not fabricated it, and scoring that as a
    miss would understate citation accuracy for a formatting reason."""
    checks = rag.verify(
        _answer(Citation(source_id=1, quote="report\n   counterfeit    notes")), CHUNKS)
    assert checks[0]["status"] == "ok"


def test_case_differences_still_match():
    checks = rag.verify(_answer(Citation(source_id=1, quote="REPORT COUNTERFEIT NOTES")),
                        CHUNKS)
    assert checks[0]["status"] == "ok"


def test_a_quote_present_in_several_sources_records_all_of_them():
    """This is what makes the 'unfalsifiable' count computable. When ten sibling
    circulars carry one sentence, `ok` cannot mean the attribution was checked --
    and found_in with more than one entry is the only evidence of that."""
    checks = rag.verify(_answer(Citation(source_id=1, quote="report counterfeit notes")),
                        CHUNKS)
    assert len(checks[0]["found_in"]) > 1


def test_out_of_range_source_id_does_not_raise():
    """The model picks source_id itself, so it can pick one that does not exist.
    That must be a countable failure, not a crash after a paid call."""
    checks = rag.verify(_answer(Citation(source_id=99, quote="report counterfeit notes")),
                        CHUNKS)
    assert checks[0]["status"] == "wrong_source"
    assert checks[0]["cited_doc"] is None


def test_empty_quote_is_never_ok():
    """An empty string is a substring of everything. Without the guard in verify()
    a model could earn a verified citation by quoting nothing at all."""
    checks = rag.verify(_answer(Citation(source_id=1, quote="")), CHUNKS)
    assert checks[0]["status"] != "ok"


def test_no_citations_verifies_nothing():
    assert rag.verify(_answer(), CHUNKS) == []


# --------------------------- answer-key validation --------------------------------
def test_refusal_text_in_an_answer_field_is_caught():
    """The q008 gap: an entry sat at answerable=true with an answer reading
    "Not answerable from this source", and validation passed because the only
    test was that the field was non-empty. It then exported into the graded set
    as an answerable question whose gold answer was a refusal."""
    assert qa._is_refusal_text("Not answerable from this source")


def test_other_refusal_shapes_are_caught():
    for text in ("N/A", "TBD", "not in the corpus",
                 "The source does not state a rationale.",
                 "cannot be answered from this chunk", "?"):
        assert qa._is_refusal_text(text), text


def test_a_real_answer_that_mentions_absence_is_not_a_refusal():
    """Deliberately narrow. A real answer may well report that the circular is
    silent on something -- that is a finding, and flagging it would train the
    author to phrase around the validator, which is worse than no validator."""
    real = ("The circular does not specify a deadline, but requires the return to "
            "be filed on the CIMS portal.")
    assert not qa._is_refusal_text(real)


def test_ordinary_answers_pass():
    for text in ("Sections 29A, 30A and 32 of the National Housing Bank Act, 1987.",
                 "Six months from the date of seizure.",
                 "An Internal Ombudsman must be a retired or serving officer."):
        assert not qa._is_refusal_text(text), text
