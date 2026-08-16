"""
tracing.py
----------
Langfuse tracing, wrapped so the pipeline never depends on it working.

    setx LANGFUSE_PUBLIC_KEY "pk-lf-..."
    setx LANGFUSE_SECRET_KEY "sk-lf-..."
    # optional, defaults to Langfuse Cloud:
    setx LANGFUSE_BASE_URL   "https://cloud.langfuse.com"

Everything here degrades to a no-op when the keys are absent, the package isn't
installed, or the service is unreachable. That is deliberate: observability
failing should never take the thing it observes down with it. Run without keys
and the pipeline behaves exactly as before, minus the traces.

Why this alongside data/log.jsonl rather than instead of it -- the two answer
different questions:

    log.jsonl   durable local record. Greppable, offline, survives forever,
                and is the thing an eval script reads.
    Langfuse    a UI over *nested* timings -- embed vs. vector-search vs. LLM
                as separate spans under one trace -- plus token/cost capture
                and the dataset features Week 2's eval needs.

log.jsonl gives one `seconds` number per question. A trace tells you which of
the three stages spent it.
"""

import os

_STATE = {}


class _NoOp:
    """Stand-in observation for when tracing is off. Swallows every call."""

    def update(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client():
    """Resolve the client once. Any failure disables tracing rather than raising."""
    if "client" in _STATE:
        return _STATE["client"]

    _STATE["client"] = None                      # cached failure; don't retry per call
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client

        client = get_client()
        if not client.auth_check():              # bad keys: fail here, not mid-run
            print("langfuse: auth check failed -- tracing disabled")
            return None
        _STATE["client"] = client
    except Exception as e:                       # not installed, network down, API change
        print(f"langfuse: disabled ({e.__class__.__name__})")
    return _STATE["client"]


def enabled() -> bool:
    return _client() is not None


def observe(name, as_type="span", **kwargs):
    """Start an observation, or a no-op if tracing is unavailable.

    Use as a context manager. `as_type="generation"` marks an LLM call, which is
    what makes Langfuse attribute tokens and cost to it rather than treating it
    as a plain timing span.
    """
    client = _client()
    if client is None:
        return _NoOp()
    try:
        return client.start_as_current_observation(as_type=as_type, name=name, **kwargs)
    except Exception:
        return _NoOp()


def flush():
    """Send buffered events. Required before a short-lived process exits."""
    client = _client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


def status() -> str:
    if enabled():
        host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        return f"langfuse: on ({host})"
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return "langfuse: off (no LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)"
    return "langfuse: off"
