"""Put the repo root on sys.path so `tests/` can import the modules it tests.

WHY THIS FILE EXISTS
The scripts here are top-level modules (`retrieval.py`, `rag.py`, `qa.py`), not an
installed package, so importing them depends on the repo root being on sys.path --
and whether it is depends on how pytest was started:

    python -m pytest tests/     works: `python -m` prepends the working directory
    pytest tests/               fails: the console script prepends nothing

Both are reasonable things to type, and the second is what CI ran, which is how a
suite that passed on every local run collected zero tests on the first push. The
fix belongs here rather than in the CI command, so the failure cannot come back
via a different invocation -- someone cloning the repo and typing `pytest` should
not have to know this.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
