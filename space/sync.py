"""
Assemble the Hugging Face Space from the source of truth.

    python space/sync.py            # copy modules + data into space/
    python space/sync.py --check    # report drift without copying

WHY A SCRIPT AND NOT A COMMITTED COPY
A Space is its own git repo, so the obvious move is to paste `retrieval.py` and
`rag.py` next to `app.py` and be done. That is how a demo starts quietly lying:
the pipeline gets a fix, the Space keeps the old ranking, and the thing the public
can actually click stops being the thing the README measured. Nobody notices,
because both still work.

So the copies are generated, never hand-edited, and `--check` fails loudly when
they have drifted. The copied files are gitignored in this repo -- the originals
are the only version under review here, and the Space repo gets the assembled
directory.

`app.py` is the one file in space/ that is genuinely written here, because it is
the only part that has no counterpart in the pipeline.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"

# Every module app.py imports, directly or transitively.
#   retrieval.py   BM25, RRF, entity mask, the reranker -- the whole ranking path
#   rag.py         generate() and verify(), so citations are checked identically
#   rbi_scraper.py scan_entities(), the entity detector the filter depends on
#   env.py         loads .env; stdlib only
#   tracing.py     rag.py imports it; no-ops without LANGFUSE_* keys
MODULES = ["retrieval.py", "rag.py", "rbi_scraper.py", "env.py", "tracing.py"]

# 27MB total. Both are gitignored in this repo (derived artifacts) and both must
# be committed to the Space repo, which is what makes it run without Qdrant.
DATA = ["data/chunks.jsonl", "data/embeddings.npy"]


def pairs():
    for rel in MODULES + DATA:
        yield ROOT / rel, SPACE / rel


def cmd_check():
    drift, missing = [], []
    for src, dst in pairs():
        if not src.exists():
            sys.exit(f"missing source: {src.relative_to(ROOT)} -- nothing to sync from")
        if not dst.exists():
            missing.append(dst.relative_to(SPACE))
        elif not filecmp.cmp(src, dst, shallow=False):
            drift.append(dst.relative_to(SPACE))

    for label, items in (("never copied", missing), ("DRIFTED from the original", drift)):
        for i in items:
            print(f"  {i}  -- {label}")
    if missing or drift:
        print(f"\n{len(missing) + len(drift)} file(s) out of date -- run: python space/sync.py")
        return 1
    print(f"all {len(MODULES) + len(DATA)} file(s) match the source of truth")
    return 0


def cmd_sync():
    for src, dst in pairs():
        if not src.exists():
            sys.exit(f"missing source: {src.relative_to(ROOT)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {dst.relative_to(SPACE)}  ({dst.stat().st_size:,} bytes)")
    print(f"\nsynced {len(MODULES) + len(DATA)} file(s) into {SPACE.name}/")
    print("deploy:  cd space && git init && git remote add origin "
          "https://huggingface.co/spaces/<user>/RegRAG && git add -A && git push")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report drift instead of copying; exits 1 if out of date")
    sys.exit(cmd_check() if ap.parse_args().check else cmd_sync())
