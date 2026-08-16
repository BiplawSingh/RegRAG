"""
env.py
------
Loads .env once, for every script in this project.

Import it before anything that reads an environment variable:

    import env  # noqa: F401  -- must precede os.getenv() at module level

That import-first ordering matters. `rag.py` resolves GEN_MODEL and QDRANT_URL at
*import* time, so a loader that ran later would read defaults and silently ignore
your .env.

Real values live in .env (git-ignored). .env.example is the committed template --
copy it, fill it in, and never commit the copy:

    cp .env.example .env

Anything already set in the real environment wins over .env (`override=False`).
That's deliberate: a value exported in your shell, or injected by CI, or set via
`setx`, should beat a checked-out file. It also means a leftover placeholder in
.env can't clobber a key you actually set.
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).with_name(".env")


def load():
    """Load .env if present. Missing file or missing package is not an error."""
    if not _ENV_FILE.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:                      # pip install python-dotenv
        return _load_manually()
    return load_dotenv(_ENV_FILE, override=False)


def _load_manually():
    """Minimal fallback parser, so a missing python-dotenv doesn't break anything.

    Handles `KEY=value`, comments, blank lines, and surrounding quotes. Not a full
    dotenv implementation -- no multi-line values, no variable interpolation.
    """
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)    # setdefault == override=False
    return True


def summary() -> str:
    """One line per variable, showing whether it's set -- never the value itself."""
    names = [
        "GEMINI_API_KEY", "GEMINI_MODEL", "QDRANT_URL",
        "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
        "SCRAPER_CONTACT",
    ]
    secret = {"GEMINI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"}
    lines = [f"env: {'.env loaded' if _ENV_FILE.exists() else 'no .env file'}"]
    for n in names:
        v = os.getenv(n)
        if not v:
            shown = "-"
        elif n in secret:
            shown = f"set ({len(v)} chars)"     # never print a key
        else:
            shown = v
        lines.append(f"  {n:<22} {shown}")
    return "\n".join(lines)


load()
