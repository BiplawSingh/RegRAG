"""
Tests for the HTTP layer: the contract, the limits, and the fallbacks.

No corpus, no model, no API key -- same rule as the rest of tests/. The fake
pipeline replaces only the two stages that need those (retrieve, generate);
everything that turns their output into a response -- verification, highlight
offsets, the generation fallbacks -- is the real code.
"""

import pytest
from fastapi.testclient import TestClient

import rag
from api.limits import RateLimiter, Window
from api.main import create_app
from api.pipeline import Pipeline

PASSAGE = ("3. The IO shall either be a retired or serving officer,\n"
           "in the rank equivalent to a General Manager.")
SIBLING = "Unrelated text. The IO shall not be over 70 years of age."


def chunk(i, text, applies_to=("Commercial Banks",)):
    return {"chunk_id": i, "text": text, "number": f"RBI/2026-27/{i}",
            "ref_code": "CEPD.X", "subject": "Internal Ombudsman Directions",
            "issue_date": "2026-01-01", "department": "CEPD",
            "applies_to": list(applies_to), "section": "3. Eligibility",
            "source_url": f"https://rbi.org.in/{i}",
            "scores": {"dense_rank": i, "bm25_rank": None, "rrf": 0.016, "rerank": 4.2}}


class FakePipeline(Pipeline):
    def __init__(self, answer=None, key=True, fail=False):
        super().__init__()
        self.answer, self.key, self.fail = answer, key, fail
        self.generated = 0

    def warm(self):
        self.chunks = [None] * 3
        self.ready = True

    def generation_enabled(self):
        return self.key

    def retrieve(self, question):
        return [chunk(10, PASSAGE), chunk(11, SIBLING)], ["Commercial Banks"]

    def generate(self, question, chunks):
        self.generated += 1
        if self.fail:
            raise RuntimeError("quota")
        return self.answer, 1.25


ANSWER = rag.Answer(
    answer="A retired or serving officer of GM rank, under 70.",
    insufficient_context=False,
    citations=[
        # reflowed and recased: must still verify, and still highlight
        rag.Citation(source_id=1, quote="in the rank   equivalent to a GENERAL manager."),
        # attributed to [1], actually in [2]
        rag.Citation(source_id=1, quote="The IO shall not be over 70 years of age."),
        # in nothing
        rag.Citation(source_id=2, quote="must hold a law degree"),
    ],
)


def client(pipeline=None, limiter=None, **kw):
    app = create_app(pipeline or FakePipeline(ANSWER),
                     limiter or RateLimiter(ip_per_minute=100), **kw)
    return TestClient(app)


@pytest.fixture
def c():
    with client() as tc:           # `with` runs startup, i.e. warm()
        yield tc


# ---------------------------------------------------------------- contract --
def test_health_reports_ready_after_startup(c):
    body = c.get("/health").json()
    assert body == {"status": "ok", "chunks": 3, "models_loaded": True,
                    "generation_enabled": True}


def test_ask_returns_answer_sources_and_retrieval_info(c):
    body = c.post("/ask", json={"question": "Who can be an IO?"}).json()
    assert body["generation"] == "ok"
    assert body["answer"].startswith("A retired")
    assert [s["rank"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["circular"]["title"] == "Internal Ombudsman Directions"
    assert body["sources"][0]["scores"]["rerank"] == 4.2
    assert body["retrieval"]["entities"] == ["Commercial Banks"]
    assert body["timings_ms"]["generation"] == 1250


def test_verified_quote_highlights_the_original_text(c):
    cit = c.post("/ask", json={"question": "Who can be an IO?"}).json()["citations"][0]
    assert cit["status"] == "ok" and cit["verified"]
    h = cit["highlight"]
    assert h["source"] == 1
    # offsets index the ORIGINAL passage, newline and capitals included
    assert PASSAGE[h["start"]:h["end"]] == "in the rank equivalent to a General Manager."


def test_misattributed_quote_highlights_where_it_really_is(c):
    cit = c.post("/ask", json={"question": "Who can be an IO?"}).json()["citations"][1]
    assert cit["status"] == "wrong_source" and not cit["verified"]
    assert cit["source"] == 1 and cit["found_in"] == [2]
    h = cit["highlight"]
    assert h["source"] == 2
    assert SIBLING[h["start"]:h["end"]] == "The IO shall not be over 70 years of age."


def test_unfound_quote_has_no_highlight(c):
    cit = c.post("/ask", json={"question": "Who can be an IO?"}).json()["citations"][2]
    assert cit["status"] == "not_found" and cit["highlight"] is None


# -------------------------------------------------------------- validation --
@pytest.mark.parametrize("payload", [
    {"question": "hi"},
    {"question": "   "},
    {"question": "x" * 501},
    {},
    {"question": 42},
])
def test_bad_questions_are_rejected_before_any_work(payload):
    fake = FakePipeline(ANSWER)
    with client(fake) as tc:
        assert tc.post("/ask", json=payload).status_code == 422
    assert fake.generated == 0


def test_question_is_stripped(c):
    body = c.post("/ask", json={"question": "  Who can be an IO?  "}).json()
    assert body["question"] == "Who can be an IO?"


# ---------------------------------------------------------------- fallbacks --
def test_without_a_key_retrieval_still_answers():
    fake = FakePipeline(ANSWER, key=False)
    with client(fake) as tc:
        body = tc.post("/ask", json={"question": "Who can be an IO?"}).json()
    assert body["generation"] == "unavailable"
    assert body["answer"] is None and body["citations"] == []
    assert len(body["sources"]) == 2 and body["notice"]
    assert fake.generated == 0


def test_a_generation_failure_still_returns_the_sources():
    with client(FakePipeline(ANSWER, fail=True)) as tc:
        r = tc.post("/ask", json={"question": "Who can be an IO?"})
    assert r.status_code == 200
    assert r.json()["generation"] == "unavailable" and len(r.json()["sources"]) == 2


def test_not_ready_is_503_not_a_crash():
    class Cold(FakePipeline):
        def warm(self):
            raise RuntimeError("model download failed")

    with client(Cold(ANSWER)) as tc:
        assert tc.get("/health").json()["status"] == "starting"
        assert tc.post("/ask", json={"question": "Who can be an IO?"}).status_code == 503


# ------------------------------------------------------------------ limits --
def test_too_many_requests_from_one_address_is_429():
    with client(limiter=RateLimiter(ip_per_minute=2)) as tc:
        codes = [tc.post("/ask", json={"question": "Who can be an IO?"}).status_code
                 for _ in range(3)]
        r = tc.post("/ask", json={"question": "Who can be an IO?"})
    assert codes == [200, 200, 429]
    assert r.json()["retry_after"] > 0 and int(r.headers["Retry-After"]) > 0


def test_exhausted_generation_budget_degrades_to_retrieval_only():
    fake = FakePipeline(ANSWER)
    with client(fake, RateLimiter(ip_per_minute=100, gen_per_day=1)) as tc:
        first = tc.post("/ask", json={"question": "Who can be an IO?"}).json()
        second = tc.post("/ask", json={"question": "Who can be an IO?"}).json()
    assert first["generation"] == "ok"
    assert second["generation"] == "rate_limited" and len(second["sources"]) == 2
    assert fake.generated == 1


def test_forwarded_addresses_are_limited_separately_behind_a_proxy():
    with client(limiter=RateLimiter(ip_per_minute=1), forwarded_hops=1) as tc:
        a = tc.post("/ask", json={"question": "Who can be an IO?"},
                    headers={"X-Forwarded-For": "1.1.1.1"})
        b = tc.post("/ask", json={"question": "Who can be an IO?"},
                    headers={"X-Forwarded-For": "2.2.2.2"})
        # a spoofed left-hand entry does not buy a fresh budget
        a2 = tc.post("/ask", json={"question": "Who can be an IO?"},
                     headers={"X-Forwarded-For": "9.9.9.9, 1.1.1.1"})
    assert (a.status_code, b.status_code, a2.status_code) == (200, 200, 429)


def test_global_cap_refusal_does_not_spend_the_visitors_own_budget():
    t = [0.0]
    lim = RateLimiter(ip_gen_per_day=2, gen_per_minute=1, clock=lambda: t[0])
    assert lim.allow_generation("a")
    assert not lim.allow_generation("a")      # global per-minute cap
    t[0] = 61
    assert lim.allow_generation("a")          # still has its 2nd daily one


def test_window_rolls_over():
    w = Window(limit=1, seconds=60)
    w.record(0)
    assert not w.has_room(59.9) and w.retry_after(30) == 30
    assert w.has_room(60)


# -------------------------------------------------------------------- CORS --
def test_cors_allows_only_the_configured_origin():
    with client(origins=["https://regrag.vercel.app"]) as tc:
        ok = tc.options("/ask", headers={"Origin": "https://regrag.vercel.app",
                                         "Access-Control-Request-Method": "POST"})
        bad = tc.options("/ask", headers={"Origin": "https://evil.example",
                                          "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "https://regrag.vercel.app"
    assert "access-control-allow-origin" not in bad.headers
