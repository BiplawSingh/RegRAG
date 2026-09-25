"""
Assemble a deployable directory from the source of truth.

    python space/sync.py                     # the Gradio demo  -> space/
    python space/sync.py --target api        # the web API      -> api_space/ (Cloud Run)
    python space/sync.py --check [--target api]   # report drift without copying

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

In each target, only the files that have no counterpart in the pipeline are
written by hand: `app.py` for the Gradio demo; the Dockerfile, .gcloudignore,
README and requirements for the API.
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Every pipeline module the app imports, directly or transitively.
#   retrieval.py   BM25, RRF, entity mask, the reranker -- the whole ranking path
#   rag.py         generate() and verify(), so citations are checked identically
#   rbi_scraper.py scan_entities(), the entity detector the filter depends on
#   env.py         loads .env; stdlib only
#   tracing.py     rag.py imports it; no-ops without LANGFUSE_* keys
PIPELINE = ["retrieval.py", "rag.py", "rbi_scraper.py", "env.py", "tracing.py"]

# 27MB total. Both are gitignored in this repo (derived artifacts) and both must
# be committed to the Space repo, which is what makes it run without Qdrant.
DATA = ["data/chunks.jsonl", "data/embeddings.npy"]

TARGETS = {
    "gradio": (ROOT / "space", PIPELINE + DATA),
    "api": (ROOT / "api_space",
            PIPELINE + DATA + ["api/__init__.py", "api/main.py", "api/pipeline.py",
                               "api/schemas.py", "api/limits.py", "api/highlight.py"]),
}


def pairs(target):
    dest, files = TARGETS[target]
    for rel in files:
        yield ROOT / rel, dest / rel


def cmd_check(target):
    dest, files = TARGETS[target]
    drift, missing = [], []
    for src, dst in pairs(target):
        if not src.exists():
            sys.exit(f"missing source: {src.relative_to(ROOT)} -- nothing to sync from")
        if not dst.exists():
            missing.append(dst.relative_to(dest))
        elif not filecmp.cmp(src, dst, shallow=False):
            drift.append(dst.relative_to(dest))

    for label, items in (("never copied", missing), ("DRIFTED from the original", drift)):
        for i in items:
            print(f"  {i}  -- {label}")
    if missing or drift:
        flag = "" if target == "gradio" else f" --target {target}"
        print(f"\n{len(missing) + len(drift)} file(s) out of date -- run: python space/sync.py{flag}")
        return 1
    print(f"all {len(files)} file(s) match the source of truth")
    return 0


def cmd_sync(target):
    dest, files = TARGETS[target]
    for src, dst in pairs(target):
        if not src.exists():
            sys.exit(f"missing source: {src.relative_to(ROOT)}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {dst.relative_to(dest)}  ({dst.stat().st_size:,} bytes)")
    print(f"\nsynced {len(files)} file(s) into {dest.name}/")
    print("deploy: see DEPLOY.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS), default="gradio",
                    help="which Space to assemble (default: gradio)")
    ap.add_argument("--check", action="store_true",
                    help="report drift instead of copying; exits 1 if out of date")
    a = ap.parse_args()
    sys.exit(cmd_check(a.target) if a.check else cmd_sync(a.target))
