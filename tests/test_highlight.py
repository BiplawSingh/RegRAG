"""
Tests for api/highlight.py: offsets into the original text, for quotes matched
the way rag.verify() matches them.

The property that matters is agreement with verify(): every quote verify() calls
present should get a span, and the span should be the quote. A highlight that
disagrees with the verification badge next to it is worse than no highlight.
"""

import pytest

import rag
from api.highlight import _normalised_with_map, locate


def span(quote, text):
    s = locate(quote, text)
    return text[s[0]:s[1]] if s else None


def test_exact_quote():
    assert span("serving officer", "a retired or serving officer.") == "serving officer"


def test_reflowed_quote_maps_back_to_the_original_whitespace():
    text = "shall   be\n\treported to  the police"
    assert span("be reported to the", text) == "be\n\treported to  the"


def test_case_is_ignored_but_the_original_case_is_returned():
    assert span("GENERAL MANAGER", "a General Manager in") == "General Manager"


def test_unicode_whitespace_rbi_text_is_full_of():
    text = "non‑banking finance, regulation"      # nb-hyphen, narrow nbsp
    assert span("finance, regulation", text) == "finance, regulation"


def test_absent_quote_has_no_span():
    assert locate("a law degree", "a retired or serving officer") is None


def test_empty_quote_has_no_span():
    # same guard as verify(): "" is a substring of everything
    assert locate("   ", "anything") is None


@pytest.mark.parametrize("text", [
    "  leading and trailing  ",
    "A\n\nB\tC",
    "ALL CAPS Ω and İstanbul",
    "",
])
def test_normalisation_agrees_with_rag_norm(text):
    s, where = _normalised_with_map(text)
    assert s.strip() == rag.norm(text)
    assert len(where) == len(s)
