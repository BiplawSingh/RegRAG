"""
chunker.py
----------
Turn cached RBI circular bodies into retrieval-ready chunks with metadata.

    python rbi_scraper.py text      # cache bodies first (network)
    python chunker.py               # -> data/chunks.jsonl  (offline)
    python chunker.py --stats       # corpus shape, no rewrite

Every chunk carries `applies_to` — the regulated-entity classes the circular binds.
That is not decoration. ~74% of this corpus lives in near-duplicate families (the same
Directions reissued per entity class: KYC exists 10 times, once each for Commercial
Banks, SFBs, UCBs, NBFCs, ...). Their text differs by one noun phrase, so embeddings
cannot separate them and a top-k retrieve returns one chunk from each variant. The
filter has to run *during* the vector search, which means the field must live on the
chunk, not on a document row you join against afterwards.

Query side, the filter must be `applies_to contains <entity> OR applies_to is empty` —
strict containment silently drops the ~8% of circulars that aren't entity-scoped
(FEMA notices, district reassignments) and would make them unreachable for every query.
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

ROOT = Path("data")
DB_PATH = ROOT / "manifest.db"
OUT_PATH = ROOT / "chunks.jsonl"

TARGET_CHARS = 1100      # ~250-300 tokens; comfortable for most embedding models
MAX_CHARS = 1800         # hard ceiling before a forced split
OVERLAP_CHARS = 150      # carried tail, so a sentence split across chunks stays findable
MIN_CHARS = 80           # below this a chunk is noise (stray footnote, page artifact)

# ------------------------- structure recognition ----------------------------
# RBI Directions nest as: Chapter -> lettered section -> numbered para -> (i)/(a) items.
# Nothing is marked up as a heading in the HTML, so these are recognised by shape.
HEADING_RES = [
    (0, re.compile(r"^(Chapter|Part)\s+[IVXLC\d]+\b", re.I)),
    (0, re.compile(r"^(Annex(ure)?|Schedule|Appendix)\b[\s\-–:]*[IVXLC\d]*", re.I)),
    (0, re.compile(r"^(Introduction|Preamble|Table of Contents)\s*$", re.I)),
    (1, re.compile(r"^[A-Z]\.\s+[A-Z]")),          # "A. Short title and commencement"
    (1, re.compile(r"^\d{1,3}\.\s+[A-Z][a-z]")),   # "3. Short Title and Commencement"
]
# a heading is a *short* line; long paragraphs that happen to start "1. The bank shall"
# are body text, not structure
HEADING_MAX_CHARS = 120

# numbered body paragraphs — used as clean split points, not as headings
PARA_NUM_RE = re.compile(r"^(\(?\d{1,3}[A-Z]?\)?[.)]|\([ivxlc]+\)|\([a-z]\))\s")

# footnote/reference debris that pollutes chunk text
FOOTNOTE_MARK_RE = re.compile(r"\[\s*\d+\s*\]|\s*\d+\s*$")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def is_heading(text: str):
    """Return heading level (0=chapter, 1=section) or None."""
    if len(text) > HEADING_MAX_CHARS:
        return None
    for level, rx in HEADING_RES:
        if rx.match(text):
            return level
    return None


def table_to_markdown(tbl) -> str:
    """Tables survive as markdown pipes — an LLM reads them, and they stay one unit."""
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [clean(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join([" --- "] * width) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _table_width(tbl) -> int:
    rows = tbl.find_all("tr")
    return max((len(r.find_all(["td", "th"])) for r in rows), default=0)


def _is_data_table(tbl) -> bool:
    """RBI uses <table> for page layout as much as for data. Real tables have columns."""
    return _table_width(tbl) >= 2


def _is_toc(tbl) -> bool:
    """The rendered Table of Contents — navigation, not content. Never a useful chunk."""
    head = clean(tbl.get_text(" ", strip=True))[:40].lower()
    if head.startswith("table of contents"):
        return True
    rows = tbl.find_all("tr")
    linky = sum(1 for r in rows if r.find("a"))
    return _table_width(tbl) == 1 and len(rows) >= 8 and linky >= len(rows) * 0.8


def split_table_markdown(md: str, max_chars: int):
    """Break an oversized table into row groups, repeating the header in each part.

    A 24k-char table exceeds every embedding model's context and would retrieve as one
    undifferentiated blob; the header repeat keeps each part readable on its own.
    """
    lines = md.split("\n")
    if len(lines) < 3 or len(md) <= max_chars:
        return [md]
    header, sep, rows = lines[0], lines[1], lines[2:]
    budget = max_chars - len(header) - len(sep) - 2
    parts, cur, cur_len = [], [], 0
    for row in rows:
        if cur and cur_len + len(row) > budget:
            parts.append("\n".join([header, sep] + cur))
            cur, cur_len = [], 0
        cur.append(row)
        cur_len += len(row) + 1
    if cur:
        parts.append("\n".join([header, sep] + cur))
    return parts


def iter_blocks(body_html: str):
    """Walk the body in document order, yielding ('text'|'table', string).

    The cached file is itself the <table class="td"> wrapper, and RBI nests layout
    tables inside it, so 'is this element a table?' is not the right question — only
    genuine multi-column tables are emitted as table blocks. Layout tables are
    descended into so their paragraphs flow normally; a data table's inner <p> tags
    are consumed by the table so cell text is never emitted twice.
    """
    soup = BeautifulSoup(body_html, "html.parser")
    for sup in soup.find_all("sup"):          # footnote markers, not content
        sup.decompose()

    root = soup.find("table", class_="td") or soup
    tables = [t for t in root.find_all("table")]

    toc_ids, data_ids, consumed = set(), set(), set()
    for t in tables:
        if _is_toc(t):
            toc_ids.add(id(t))
            consumed.update(id(d) for d in t.find_all(["p", "table"]))
        elif _is_data_table(t):
            data_ids.add(id(t))
            consumed.update(id(d) for d in t.find_all(["p", "table"]))

    for el in root.find_all(["p", "table"]):
        if id(el) in consumed:
            continue
        if el.name == "table":
            if id(el) in data_ids:
                md = table_to_markdown(el)
                if md:
                    yield "table", md
            continue                          # layout/ToC table -> nothing of its own
        t = clean(el.get_text(" ", strip=True))
        if t:
            yield "text", t


# ------------------------------- chunking -----------------------------------
def chunk_document(body_html: str, meta: dict):
    """Structure-aware split. Tables stay whole; sections never bleed into each other."""
    chunks = []
    section_path = []          # e.g. ["Chapter II: Classification", "A. Applicability"]
    buf, buf_len = [], 0

    def flush(carry_overlap=True):
        nonlocal buf, buf_len
        text = "\n".join(buf).strip()
        if len(text) >= MIN_CHARS:
            chunks.append({
                **meta,
                "chunk_index": len(chunks),
                "section_path": list(section_path),
                "section": " > ".join(section_path) if section_path else None,
                "text": text,
                "n_chars": len(text),
            })
        # carry a tail so a fact split across the boundary is still retrievable
        if carry_overlap and text and len(text) > OVERLAP_CHARS:
            tail = text[-OVERLAP_CHARS:]
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            buf, buf_len = [tail], len(tail)
        else:
            buf, buf_len = [], 0

    for kind, text in iter_blocks(body_html):
        if kind == "table":
            flush(carry_overlap=False)        # a table is its own unit
            for part in split_table_markdown(text, MAX_CHARS):
                buf, buf_len = [part], len(part)
                flush(carry_overlap=False)
            continue

        level = is_heading(text)
        if level is not None:
            flush(carry_overlap=False)        # don't let one section trail into the next
            del section_path[level:]
            section_path.append(text)
            continue

        # oversized single paragraph -> split on sentence boundaries
        if len(text) > MAX_CHARS:
            flush()
            for piece in re.split(r"(?<=[.;])\s+(?=[A-Z(])", text):
                if buf_len + len(piece) > TARGET_CHARS and buf_len:
                    flush()
                buf.append(piece)
                buf_len += len(piece) + 1
            continue

        if buf_len + len(text) > TARGET_CHARS and buf_len:
            flush()
        buf.append(text)
        buf_len += len(text) + 1

    flush(carry_overlap=False)
    return chunks


# --------------------------------- driver -----------------------------------
def load_docs():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con.execute(
        "SELECT key, rbi_id, number, ref_code, issue_date, department, applies_to, "
        "       meant_for, subject, source_url, body_path "
        "FROM docs WHERE body_path IS NOT NULL ORDER BY issue_date DESC"
    ).fetchall()


def build(stats_only=False):
    docs = load_docs()
    if not docs:
        print("no cached bodies — run: python rbi_scraper.py text")
        return

    all_chunks, per_doc, entity_chunks = [], [], Counter()
    no_entity = 0
    for d in docs:
        path = Path(d["body_path"])
        if not path.exists():
            continue
        applies_to = json.loads(d["applies_to"] or "[]")
        meta = {
            "doc_key": d["key"],
            "rbi_id": d["rbi_id"],
            "number": d["number"],
            "ref_code": d["ref_code"],
            "issue_date": d["issue_date"],
            "department": d["department"],
            "applies_to": applies_to,          # <- the filter field, on every chunk
            "meant_for": d["meant_for"],
            "subject": d["subject"],
            "source_url": d["source_url"],
        }
        cs = chunk_document(path.read_text(encoding="utf-8"), meta)
        all_chunks.extend(cs)
        per_doc.append(len(cs))
        if applies_to:
            for e in applies_to:
                entity_chunks[e] += len(cs)
        else:
            no_entity += len(cs)

    if not stats_only:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for c in all_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [c["n_chars"] for c in all_chunks]
    per_doc.sort()
    print(f"documents          {len(per_doc)}")
    print(f"chunks             {len(all_chunks)}")
    print(f"chunks/doc         min={per_doc[0]} median={per_doc[len(per_doc)//2]} max={per_doc[-1]}")
    print(f"chunk chars        min={min(sizes)} mean={sum(sizes)//len(sizes)} max={max(sizes)}")
    print(f"with section path  {sum(1 for c in all_chunks if c['section'])}"
          f" ({sum(1 for c in all_chunks if c['section'])*100//len(all_chunks)}%)")
    print(f"tables preserved   {sum(1 for c in all_chunks if c['text'].startswith('|'))}")
    print()
    print("chunks per entity class (a chunk counts once per class it binds):")
    for e, n in entity_chunks.most_common():
        print(f"   {n:>6}  {e}")
    print(f"   {no_entity:>6}  (no entity class — must stay reachable via OR-empty filter)")
    if not stats_only:
        print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="report only, don't rewrite output")
    build(stats_only=ap.parse_args().stats)
