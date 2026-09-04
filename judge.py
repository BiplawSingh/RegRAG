"""
judge.py
--------
Scores answer quality against the hand-written answer key, using a second model.

    python judge.py calibrate         # DO THIS FIRST -- is the judge any good?
    python judge.py run               # grade the end-to-end run
    python judge.py run --oracle      # grade the oracle run
    python judge.py report            # verdict breakdown, both runs

WHY A JUDGE AT ALL
Two of the three metrics in eval.py are deterministic: did the gold document get
retrieved, and does each quote occur in the document it was credited to. Neither
says whether the ANSWER is any good. A verified citation supporting a shallow,
evasive or half-complete answer passes both -- and this corpus produces exactly
that failure, repeatedly. Somebody has to read the answer against the key. A
second model reads 68 of them in an hour; a person does not.

SCOPE: THE 68 ANSWERABLE QUESTIONS
The 22 unanswerable ones are already scored exactly, by whether the pipeline set
insufficient_context. Handing those to a judge would replace a deterministic
check with a noisy one -- strictly worse. The judge grades only questions the key
says have an answer.

WHY qwen AND NOT gpt-oss-20b
Model judges show self-preference: they score outputs from their own family more
generously. gpt-oss-20b shares its lineage with the gpt-oss-120b that wrote these
answers, so it is the wrong marker for exactly the reason a student is the wrong
marker for their own exam. Qwen is an unrelated vendor. It also draws a separate
free-tier allowance, so judging costs nothing from the generation budget.

WHY calibrate EXISTS, AND WHY IT RUNS FIRST
An unvalidated judge does not measure quality -- it launders an opinion into a
number that looks like a measurement, which is worse than having no number,
because people act on it. Model judges are known to be lenient and to reward
length and confidence over correctness. So before any verdict here is quoted,
`calibrate` checks the judge against cases whose right answer is already known:

  positive control   the predicted answer IS the key's answer, verbatim
                     -> a judge that cannot mark this correct is broken
  negative control   the predicted answer is ANOTHER question's answer
                     -> a judge that marks this correct is rubber-stamping

Both controls are free of human labelling and cannot be gamed by the judge. A
judge that does not pass both is not used.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import env  # noqa: F401  -- loads .env before anything reads os.getenv

ROOT = Path("data")
RUNS = ROOT / "eval"
QA = ROOT / "qa.jsonl"

# An unrelated vendor, deliberately. See the module docstring.
JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.6-27b")
JUDGE_TPM = int(os.getenv("GROQ_JUDGE_TPM_LIMIT", "8000"))
# Two separate ceilings squeeze this number from both sides, and getting it wrong
# fails in two different ways -- both of which happened here before it was right.
#
# FROM ABOVE: Groq enforces OUTPUT tokens per minute separately from total TPM,
# and on this model OTPM is 1,000. A request *asking* for 2,048 is rejected before
# it runs, however short the reply would have been. The old 2048 default failed 17
# of the first 31 judgements with "Request too large", which reads like an
# input-size problem and is not one.
#
# FROM BELOW: qwen3.6 is a reasoning model and spends thinking tokens from this
# same budget. Dropping the cap to 512 traded the first error for the second --
# 400 json_validate_failed with an EMPTY failed_generation, meaning the budget was
# consumed before a single character of the verdict was emitted. Under a strict
# schema that is not a shorter answer, it is no answer. rag.py carries the same
# warning about gpt-oss; the lesson did not transfer until it was re-learned.
#
# So: turn the thinking off rather than trying to buy room for it. Reasoning adds
# nothing to a three-way classification against a reference answer, and with
# `reasoning_effort="none"` the whole budget belongs to the verdict. Measured
# across 67 judgements, the object itself is small -- median 58 tokens, p90 112,
# longest 165 -- so 900 is 5x the longest ever seen and still under OTPM.
#
# Changing this changes the grader, so `calibrate` must be re-run afterwards: a
# calibration measured with reasoning on does not describe a judge running with it
# off. Groq accepts only "none" or "default" here.
JUDGE_MAX_OUT = int(os.getenv("GROQ_JUDGE_MAX_OUTPUT_TOKENS", "900"))
JUDGE_REASONING = os.getenv("GROQ_JUDGE_REASONING_EFFORT", "none")

VERDICTS = ["correct", "partially_correct", "incorrect", "unsupported_refusal"]

SYSTEM = """You grade answers about Indian banking regulation against a reference \
answer written by a human expert. The reference is correct by definition. Your job \
is only to judge whether the candidate answer says what the reference says.

Verdicts:
- correct: states the substance of the reference. Different wording is fine. \
Extra correct detail is fine.
- partially_correct: states some of the reference but omits something material, \
or hedges the key fact into uselessness.
- incorrect: contradicts the reference, or answers a different question than the \
one asked.
- unsupported_refusal: declines to answer, or claims the sources are insufficient, \
when the reference contains an answer.

These do NOT make an answer correct, and you must mark them incorrect:
- Restating the question as though it were the answer. Asked WHY a rule exists, \
replying that the rule is required is not a reason.
- Answering "what does X contain / what is X about" by giving X's title, number \
or date instead of its content.
- Quoting the boilerplate that grants an authority to issue a rule, as though it \
were the rule.
- Generic regulatory filler: "in accordance with applicable guidelines and best \
practices" and similar, with no specific content.

Length, fluency and confidence are not quality. A long, well-written answer that \
misses the reference's point is incorrect. A one-line answer that states it is \
correct.

You must quote, in `evidence`, the exact span of the REFERENCE answer that the \
candidate missed or contradicted. If the verdict is `correct`, leave it empty."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "evidence", "why"],
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "evidence": {
            "type": "string",
            "description": "Verbatim span of the REFERENCE the candidate missed "
                           "or contradicted. Empty when verdict is correct.",
        },
        "why": {"type": "string", "description": "One sentence. No hedging."},
    },
}

_WINDOW = []


def _throttle(est):
    """Pace against the judge model's own TPM ceiling.

    Deliberately not shared with rag.py's window: limits are per model, so folding
    both into one budget would throttle each against traffic it never sent.
    """
    now = time.time()
    _WINDOW[:] = [(t, n) for t, n in _WINDOW if now - t < 60]
    used = sum(n for _, n in _WINDOW)
    if used + est > JUDGE_TPM and _WINDOW:
        wait = 60 - (now - _WINDOW[0][0]) + 0.5
        if wait > 0:
            print(f"        pacing {wait:.0f}s (TPM {used}/{JUDGE_TPM})", flush=True)
            time.sleep(wait)
            _WINDOW.clear()


_CLIENT = {}


def client():
    if "c" not in _CLIENT:
        from groq import Groq

        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("no GROQ_API_KEY -- see .env.example")
        _CLIENT["c"] = Groq(api_key=key)
    return _CLIENT["c"]


def judge_one(question, reference, candidate, said_insufficient, attempts=4):
    """One verdict. Raises on anything it cannot classify -- callers record that."""
    from groq import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

    if said_insufficient:
        # Make the refusal explicit rather than passing an empty-looking answer and
        # hoping the judge infers it. The candidate text in a refusal is usually a
        # note about what is missing, which reads like a partial answer otherwise.
        candidate = f"[DECLINED TO ANSWER] {candidate}"

    user = (f"QUESTION\n{question}\n\n"
            f"REFERENCE ANSWER (correct by definition)\n{reference}\n\n"
            f"CANDIDATE ANSWER\n{candidate}")

    for attempt in range(attempts):
        _throttle(len(user) // 4 + 600)
        try:
            r = client().chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                max_completion_tokens=JUDGE_MAX_OUT,
                reasoning_effort=JUDGE_REASONING,
                temperature=0,          # a grader that varies run to run is not a grader
                response_format={"type": "json_schema", "json_schema": {
                    "name": "verdict", "strict": True, "schema": SCHEMA}},
            )
            u = r.usage
            _WINDOW.append((time.time(), u.total_tokens if u else 0))
            return json.loads(r.choices[0].message.content or "{}")

        except RateLimitError as e:
            if "per day" in str(e).lower() or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 5)
        except BadRequestError as e:
            if "json_validate_failed" not in str(e) or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
        except (InternalServerError, APIConnectionError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def load_qa():
    if not QA.exists():
        sys.exit("no data/qa.jsonl -- run: python qa.py export")
    return {q["id"]: q for q in (json.loads(l) for l in open(QA, encoding="utf-8"))}


def load_run(oracle, k=8):
    p = RUNS / f"answers_{'oracle' if oracle else 'e2e'}_k{k}.jsonl"
    if not p.exists():
        sys.exit(f"no {p} -- run: python eval.py answers"
                 f"{' --oracle' if oracle else ''}")
    return p, [json.loads(l) for l in open(p, encoding="utf-8")]


# ------------------------------- calibration --------------------------------
def cmd_calibrate(n=8, seed=0):
    """Check the judge against cases whose correct verdict is already known.

    Neither control needs a human label, and neither can be satisfied by a judge
    that is simply agreeable:

      positive  candidate := the reference itself. Anything but `correct` means
                the judge cannot recognise a right answer, and every verdict it
                produces is noise.
      negative  candidate := a DIFFERENT question's reference. It is fluent,
                on-topic, regulatory in tone, and answers the wrong question.
                Marking it correct means the judge is rewarding plausibility,
                which is the exact failure that makes model judges worthless.

    A judge that fails either control is not used, and its numbers are not quoted.
    """
    qa = load_qa()
    answerable = [q for q in qa.values() if q["answerable"] and (q.get("answer") or "").strip()]
    rng = random.Random(seed)
    picks = rng.sample(answerable, min(n, len(answerable)))

    print(f"judge: {JUDGE_MODEL}")
    print(f"calibrating on {len(picks)} positive + {len(picks)} negative controls\n")

    rows = []
    for i, q in enumerate(picks, 1):
        # positive: the reference graded against itself
        v = judge_one(q["question"], q["answer"], q["answer"], False)
        ok = v["verdict"] == "correct"
        rows.append(("positive", q["id"], v["verdict"], ok))
        print(f"  [{i:>2}/{len(picks)}] {q['id']} positive -> {v['verdict']:<18}"
              f"{'ok' if ok else 'FAIL'}", flush=True)

        # negative: another question's reference, which is fluent and wrong
        other = rng.choice([o for o in answerable if o["id"] != q["id"]])
        v = judge_one(q["question"], q["answer"], other["answer"], False)
        ok = v["verdict"] in ("incorrect", "partially_correct")
        strict_ok = v["verdict"] == "incorrect"
        rows.append(("negative", q["id"], v["verdict"], strict_ok))
        print(f"  [{i:>2}/{len(picks)}] {q['id']} negative -> {v['verdict']:<18}"
              f"{'ok' if strict_ok else ('lenient' if ok else 'FAIL')}", flush=True)

    pos = [r for r in rows if r[0] == "positive"]
    neg = [r for r in rows if r[0] == "negative"]
    p_ok = sum(r[3] for r in pos)
    n_ok = sum(r[3] for r in neg)

    print(f"\n  positive controls  {p_ok}/{len(pos)} marked correct")
    print(f"  negative controls  {n_ok}/{len(neg)} marked incorrect")

    out = RUNS / "judge_calibration.json"
    out.write_text(json.dumps({
        "model": JUDGE_MODEL, "seed": seed,
        "positive_pass": p_ok, "positive_n": len(pos),
        "negative_pass": n_ok, "negative_n": len(neg),
        "rows": [{"kind": k, "id": i, "verdict": v, "pass": bool(o)} for k, i, v, o in rows],
    }, indent=1), encoding="utf-8")

    verdict = "USABLE" if p_ok == len(pos) and n_ok == len(neg) else "NOT USABLE"
    print(f"\n  {verdict}")
    if verdict == "NOT USABLE":
        print("  Do not quote this judge's scores. Fix the prompt or change models,")
        print("  then calibrate again -- a judge that cannot separate a right answer")
        print("  from a fluent wrong one is measuring fluency, not correctness.")
    print(f"\nwrote {out}")


# ---------------------------------- run -------------------------------------
def cmd_run(oracle=False, k=8, limit=None):
    """Grade one answer run. Resumable, same as eval.py answers."""
    cal = RUNS / "judge_calibration.json"
    if not cal.exists():
        sys.exit("no calibration on record -- run: python judge.py calibrate\n"
                 "  (a judge nobody checked is not a measurement)")

    qa = load_qa()
    src, rows = load_run(oracle, k)
    out = RUNS / f"judged_{'oracle' if oracle else 'e2e'}_k{k}.jsonl"

    # Resume past what SUCCEEDED, not past what was attempted. Counting a failed
    # row as done means one transient 429 permanently marks a question unjudged:
    # the re-run skips it, the tally silently shrinks, and nothing reports that it
    # happened. Observed for real -- a rate-limit misconfiguration failed 17 of the
    # first 31 rows, and every one of them would have been skipped forever.
    done, failed = set(), 0
    if out.exists():
        for line in open(out, encoding="utf-8"):
            row = json.loads(line)
            if "error" in row:
                failed += 1
            else:
                done.add(row["id"])
        print(f"resuming: {len(done)} already judged"
              + (f", retrying {failed} that failed" if failed else ""))

    # Only answerable questions. The rest are scored exactly by said_insufficient.
    todo = [r for r in rows
            if "error" not in r and r["id"] not in done
            and qa.get(r["id"], {}).get("answerable")
            and (qa[r["id"]].get("answer") or "").strip()]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("nothing left to judge")
        return

    print(f"judge: {JUDGE_MODEL}  |  {len(todo)} answerable questions from {src.name}\n")
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            q = qa[r["id"]]
            print(f"[{i:>3}/{len(todo)}] {r['id']}  {q['question'][:52]}", flush=True)
            try:
                v = judge_one(q["question"], q["answer"], r["predicted"],
                              r.get("said_insufficient", False))
                rec = {"id": r["id"], "category": r["category"], "oracle": oracle,
                       "verdict": v["verdict"], "evidence": v.get("evidence", ""),
                       "why": v.get("why", ""), "judge_model": JUDGE_MODEL}
            except Exception as e:
                msg = f"{e.__class__.__name__}: {e}"
                if "per day" in msg.lower():
                    print(f"\nDAILY ALLOWANCE EXHAUSTED at {i}/{len(todo)}. "
                          f"Re-run tomorrow -- it resumes.")
                    break
                print(f"       FAILED {msg[:120]}")
                rec = {"id": r["id"], "error": msg}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"\nwrote {out}")


# -------------------------------- report ------------------------------------
def cmd_report(k=8):
    cal = RUNS / "judge_calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text(encoding="utf-8"))
        status = ("USABLE" if c["positive_pass"] == c["positive_n"]
                  and c["negative_pass"] == c["negative_n"] else "NOT USABLE")
        print(f"judge {c['model']}: positives {c['positive_pass']}/{c['positive_n']}, "
              f"negatives {c['negative_pass']}/{c['negative_n']}  -> {status}\n")

    for oracle in (False, True):
        p = RUNS / f"judged_{'oracle' if oracle else 'e2e'}_k{k}.jsonl"
        if not p.exists():
            continue
        # One verdict per question, keeping the most recent. The file is append-only
        # and resumable, so a question can legitimately appear more than once -- a
        # row that errored and was later retried, or, as happened here, two runs
        # appending concurrently because a kill did not take. Counting rows instead
        # of questions then reports more questions judged than the key contains
        # (73 of 69), and quietly inflates every category tally with it. A scoreboard
        # that can print a denominator larger than the answer key is not one to
        # trust, so the de-duplication lives here rather than in a cleanup script.
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        by_id = {r["id"]: r for r in rows if "error" not in r}
        ok = list(by_id.values())
        if not ok:
            continue
        dupes = sum(1 for r in rows if "error" not in r) - len(ok)
        label = "ORACLE" if oracle else "END-TO-END"
        counts = Counter(r["verdict"] for r in ok)
        print(f"=== {label} ===  {len(ok)} answerable questions judged"
              + (f"  ({dupes} duplicate row(s) collapsed)" if dupes else ""))
        for v in VERDICTS:
            if counts[v]:
                print(f"  {v:<22}{counts[v]:>4}  ({counts[v] * 100 // len(ok)}%)")
        good = counts["correct"]
        print(f"  {'-' * 30}")
        print(f"  fully correct         {good:>4}/{len(ok)}"
              f"  ({good * 100 // len(ok)}%)")

        by = {}
        for r in ok:
            by.setdefault(r["category"], Counter())[r["verdict"]] += 1
        print(f"\n  {'category':<18}{'correct':>9}{'of':>5}")
        for cat, cnt in sorted(by.items()):
            tot = sum(cnt.values())
            print(f"  {cat:<18}{cnt['correct']:>9}{tot:>5}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c_ = sub.add_parser("calibrate", help="validate the judge before trusting it")
    c_.add_argument("-n", type=int, default=8, help="controls of each kind")
    c_.add_argument("--seed", type=int, default=0)

    r_ = sub.add_parser("run", help="grade an answer run")
    r_.add_argument("--oracle", action="store_true")
    r_.add_argument("-k", type=int, default=8)
    r_.add_argument("--limit", type=int, default=None)

    p_ = sub.add_parser("report", help="verdict breakdown")
    p_.add_argument("-k", type=int, default=8)

    a = ap.parse_args()
    if a.cmd == "calibrate":
        cmd_calibrate(n=a.n, seed=a.seed)
    elif a.cmd == "run":
        cmd_run(oracle=a.oracle, k=a.k, limit=a.limit)
    else:
        cmd_report(k=a.k)
