"""
Where a verified quote sits in the original passage, as character offsets.

rag.verify() decides *whether* a quote is in a passage by comparing normalised
text -- whitespace runs collapsed, lowercased -- so a quote the model reflowed
still counts. That comparison never needs a position, so it never produces one.
The UI does need one, and the offsets must be into the ORIGINAL text, because
that is what it displays.

So this rebuilds the same normalisation while remembering, for every normalised
character, which original character it came from. A match in normalised space
then maps straight back. verify() is left untouched; this only answers a
question it had no reason to ask.
"""

import re

from rag import norm

_WS = re.compile(r"\s+")


def _normalised_with_map(text):
    """rag.norm(text) without the final strip, plus an index back into `text`.

    The strip is safe to skip: norm(quote) is itself stripped and non-empty, so it
    can never match the single leading or trailing space that strip would remove.
    """
    out, where = [], []

    def emit(i):
        for ch in text[i].lower():       # lower() can expand one char to two
            out.append(ch)
            where.append(i)

    pos = 0
    for m in _WS.finditer(text):
        for i in range(pos, m.start()):
            emit(i)
        out.append(" ")
        where.append(m.start())
        pos = m.end()
    for i in range(pos, len(text)):
        emit(i)
    return "".join(out), where


def locate(quote, text):
    """(start, end) of `quote` in `text`, matched as verify() matches; else None.

    None is a real answer, not an error: a `not_found` citation has no span, and
    in rare Unicode cases (context-dependent lowercasing) the per-character
    rebuild can disagree with norm() -- then the citation keeps its verification
    status and simply shows no highlight, rather than a wrong one.
    """
    q = norm(quote)
    if not q:
        return None
    s, where = _normalised_with_map(text)
    i = s.find(q)
    if i < 0:
        return None
    return where[i], where[i + len(q) - 1] + 1
