"""
rbi_scraper.py
--------------
Bulk-harvest RBI circulars + master directions, then organize by date/category.

Pipeline (run phases in order, each is resumable):
    python rbi_scraper.py harvest      # walk ?Id= pages -> SQLite manifest
    python rbi_scraper.py directions   # scrape Master Directions listing -> manifest
    python rbi_scraper.py download     # fetch PDFs -> data/circulars/YYYY/MM/
    python rbi_scraper.py organize     # build category symlink views + export CSV

Deps:  pip install httpx beautifulsoup4
       (no browser needed — this route avoids RBI's ASP.NET __doPostBack filters
        by enumerating the sequential ?Id= detail pages directly.)

Be polite: this hits a government server. Keep CONCURRENCY low, keep the delay,
put a real contact address in USER_AGENT, and check rbi.org.in/robots.txt +
the site's terms before running a big sweep. The content is public regulatory
data, but courtesy still applies.
"""

import asyncio
import csv
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# ----------------------------- config ---------------------------------------
BASE = "https://www.rbi.org.in/scripts"
CIRCULAR_DETAIL = BASE + "/BS_CircularIndexDisplay.aspx?Id={id}"
MASTER_DIRECTIONS = BASE + "/BS_ViewMasterDirections.aspx"

START_ID = 13673          # ~ current max; bump this before a fresh run
SCAN_BACK = 700           # how many IDs to walk downward (gives headroom for gaps)
TARGET_DOCS = 500         # stop early once we've banked this many valid circulars
STOP_BEFORE = None        # e.g. datetime(2022, 1, 1) to cut off old docs; None = no cutoff

CONCURRENCY = 3           # be gentle
REQUEST_DELAY = 0.4       # seconds between requests per worker

# Scraping courtesy wants a reachable contact in the UA, but it must never be committed.
# Set it in the environment:  export SCRAPER_CONTACT="you@example.com"
_CONTACT = os.getenv("SCRAPER_CONTACT", "").strip()
USER_AGENT = f"rbi-research-scraper/1.0 ({_CONTACT})" if _CONTACT else "rbi-research-scraper/1.0"

ROOT = Path("data")
PDF_ROOT = ROOT / "circulars"
HTML_ROOT = ROOT / "html"        # cached document bodies (the real corpus source)
CATEGORY_ROOT = ROOT / "by_category"
DB_PATH = ROOT / "manifest.db"

HEADERS = {"User-Agent": USER_AGENT}

# ----------------------------- storage --------------------------------------
def db():
    ROOT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            key          TEXT PRIMARY KEY,   -- doc_type:id  (stable dedupe key)
            doc_type     TEXT,               -- 'circular' | 'master_direction'
            rbi_id       INTEGER,
            number       TEXT,
            issue_date   TEXT,               -- ISO YYYY-MM-DD when parseable
            raw_date     TEXT,
            department   TEXT,               -- code prefix of the ref number (DOR, DoS, DCM, ...)
            ref_code     TEXT,               -- full departmental ref, e.g. DOR.MCS.REC.No.201/...
            meant_for    TEXT,               -- addressee line ("All Banks"), when the circular has one
            applies_to   TEXT,               -- JSON list of regulated-entity classes bound
            subject      TEXT,
            source_url   TEXT,
            pdf_url      TEXT,
            local_path   TEXT,
            fetched_at   TEXT
        )
    """)
    # migrate manifests created before these columns existed
    have = {r[1] for r in con.execute("PRAGMA table_info(docs)")}
    for col in ("ref_code", "applies_to", "body_path"):
        if col not in have:
            con.execute(f"ALTER TABLE docs ADD COLUMN {col} TEXT")
    con.commit()
    return con


def upsert(con, row: dict):
    cols = ",".join(row)
    ph = ",".join("?" for _ in row)
    updates = ",".join(f"{c}=excluded.{c}" for c in row if c != "key")
    con.execute(
        f"INSERT INTO docs ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(key) DO UPDATE SET {updates}",
        list(row.values()),
    )
    con.commit()

# ----------------------------- parsing --------------------------------------
# RBI stamps circulars three ways, all of which are valid:
#   RBI/2026-2027/231        RBI/2026-27/230        RBI/DOR/2026-27/213
NUM_RE = re.compile(r"RBI/(?:[A-Za-z]+/)?\d{4}-\d{2,4}/\d+")
# detail pages spell dates out ("August 6, 2026"); dotted form kept as a fallback
LONG_DATE_RE = re.compile(r"\b([A-Z][a-z]{2,8})\s+(\d{1,2}),\s*(\d{4})\b")
DOT_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")

# the circular's own file always lives under /rdocs/Notification/; /rdocs/content/pdfs/
# is a mix of page furniture (Utkarsh, Accessibility) and per-circular annexes
DOC_PDF_RE = re.compile(r"rbidocs\.rbi\.org\.in/rdocs/Notification/", re.I)
CHROME_PDF_RE = re.compile(r"/rdocs/content/pdfs/(Utkarsh|Accessibility)", re.I)

# paragraphs that sit between the date and the real subject line
SKIP_P_RE = re.compile(r"^(previous versions|madam|dear\s|sir\b|madam/|dear sir)", re.I)
# the addressee block. Seen as "The Chairman/...", "The Chairpersons/ CEOs of...",
# "To, All Authorised Persons", "All Scheduled Commercial Banks", "Madam/Sir" variants.
ADDRESSEE_RE = re.compile(
    r"^(to[,\s]|the\s+(chair(man|person)s?|chief|managing|principal|md\b)|all\s+\S|"
    r".*\b(chairperson|chairman|managing director|chief executive)\b)", re.I
)


def iso_date(text: str):
    """Return (ISO date, raw string) for either RBI date style."""
    m = LONG_DATE_RE.search(text or "")
    if m:
        mon, d, y = m.groups()
        raw = m.group(0)
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{mon} {d} {y}", fmt).strftime("%Y-%m-%d"), raw
            except ValueError:
                continue
        return None, raw

    m = DOT_DATE_RE.search(text or "")
    if not m:
        return None, None
    d, mth, y = m.groups()
    raw = m.group(0)
    try:
        return datetime(int(y), int(mth), int(d)).strftime("%Y-%m-%d"), raw
    except ValueError:
        return None, raw


# The regulated-entity classes RBI writes circulars against. RBI issues near-identical
# parallel circulars per class (the same amendment for SFBs, UCBs, NBFCs, ...), so this
# is the metadata filter that keeps retrieval from confusing one class's rule for another.
# Order matters: more specific patterns first, since scanning stops at neither.
ENTITY_PATTERNS = [
    (r"small finance banks?|\bSFBs?\b", "Small Finance Banks"),
    (r"payments? banks?\b", "Payments Banks"),
    (r"local area banks?|\bLABs?\b", "Local Area Banks"),
    (r"regional rural banks?|\bRRBs?\b", "Regional Rural Banks"),
    (r"(primary\s*\(urban\)|urban)\s+co-?operative banks?|\bUCBs?\b", "Urban Co-operative Banks"),
    (r"(rural|state|district central)\s+co-?operative banks?|\bStCBs?\b|\bDCCBs?\b",
     "Rural Co-operative Banks"),
    (r"scheduled commercial banks?|commercial banks?", "Commercial Banks"),
    (r"housing finance (companies|company)|\bHFCs?\b", "Housing Finance Companies"),
    (r"non-?banking financial (companies|company)|\bNBFCs?\b",
     "Non-Banking Financial Companies"),
    (r"asset reconstruction (companies|company)|\bARCs?\b", "Asset Reconstruction Companies"),
    (r"credit information (companies|company)|\bCICs?\b", "Credit Information Companies"),
    (r"all india financial institutions?|\bAIFIs?\b", "All India Financial Institutions"),
    (r"standalone primary dealers?|\bSPDs?\b", "Standalone Primary Dealers"),
    (r"authorised dealer category[-\s]*I\b|\bAD category[-\s]*I\b",
     "Authorised Dealers (Category-I)"),
    (r"authorised persons?", "Authorised Persons"),
    (r"payment system (operators|providers)|\bPSOs?\b", "Payment System Operators"),
    (r"foreign banks?", "Foreign Banks"),
    (r"agency banks?", "Agency Banks"),
    (r"primary dealers?", "Primary Dealers"),
]
# addressed to the whole banking system rather than a named class; only used when no
# specific class matched, otherwise it would swallow every multi-class circular
GENERIC_ALL_BANKS_RE = re.compile(r"\ball\s+banks?\b|all scheduled banks?", re.I)
ENTITY_RES = [(re.compile(p, re.I), name) for p, name in ENTITY_PATTERNS]

# "Reserve Bank of India (Small Finance Banks - Financial Statements ...)" — the class
# is the head of the parenthetical, before the spaced dash that starts the topic.
TITLE_ENTITY_RE = re.compile(r"Reserve Bank of India\s*[(\[](.+?)[)\]]", re.I)
TOPIC_SPLIT_RE = re.compile(r"\s+[-–—:]\s+|\s+[–—]\s*|,\s")


def scan_entities(text: str):
    """Every regulated-entity class named in `text`, canonicalised, in stable order."""
    out = []
    for rx, name in ENTITY_RES:
        if name not in out and rx.search(text or ""):
            out.append(name)
    return out


def applies_to_of(subject: str, meant_for: str):
    """Who a circular binds. Multi-valued: one circular can bind several classes.

    Two independent sources, because RBI changed house style mid-corpus:
      - new "Directions" title  -> Reserve Bank of India (Payments Banks - ...)
      - older circulars         -> the addressee block, e.g. "All Authorised Persons"
    """
    found = []
    m = TITLE_ENTITY_RE.search(subject or "")
    if m:
        # scan only the class head, not the topic, so "...excluding NBFCs" can't leak in
        found += scan_entities(TOPIC_SPLIT_RE.split(m.group(1))[0])
    if meant_for:
        found += scan_entities(meant_for)
    seen = []
    for e in found:
        if e not in seen:
            seen.append(e)
    if not seen and GENERIC_ALL_BANKS_RE.search(meant_for or ""):
        seen = ["All Banks"]
    return seen


def department_of(number: str, ref_code: str):
    """Issuing department code, from the circular number or the departmental ref.

    Shapes seen in the wild:
        RBI/DOR/2026-27/213     -> DOR          (number carries it outright)
        DoS.CO.PPG.47/...       -> DOS          (leading token; casing varies)
        CO.DGBA.GBD.No.S228/... -> DGBA         ("CO" = Central Office, not a dept)
        RBI/CO.DPSS.POLC.../... -> DPSS         (ref repeats the RBI/ prefix)
        DCM(Plg) No.S737/...    -> DCM          (no separator before the sub-unit)
        REF.No.MPD.BC.401/...   -> MPD          ("REF.No." is just "reference no.")
        A.P. (DIR Series) ...   -> AP-DIR       (series label; page names no dept)
    """
    m = re.match(r"RBI/([A-Za-z]+)/", number or "")
    if m:
        return m.group(1).upper()

    ref = (ref_code or "").strip()
    if not ref:
        return None
    if re.match(r"A\.\s*P\.", ref, re.I):
        return "AP-DIR"
    for token in re.findall(r"[A-Za-z]+", ref):
        tok = token.upper()
        if len(tok) > 1 and tok not in ("CO", "RBI", "NO", "REF"):
            return tok
    return None


def pick_pdf(soup):
    """The circular's PDF, not the site-chrome PDFs every page carries."""
    fallback = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not (href.lower().endswith(".pdf") or "rbidocs.rbi.org.in" in href.lower()):
            continue
        full = href if href.startswith("http") else "https://www.rbi.org.in" + href
        if DOC_PDF_RE.search(full):
            return full
        if fallback is None and not CHROME_PDF_RE.search(full):
            fallback = full
    return fallback


def parse_circular(rbi_id: int, html: str):
    """Tolerant extraction. Returns dict or None if the page isn't a circular."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("table", class_="td")   # the circular text lives in this table
    if body is None:
        return None

    paras = [t for t in (p.get_text(" ", strip=True) for p in body.find_all("p")) if t]
    head = next((i for i, t in enumerate(paras) if NUM_RE.search(t)), None)
    if head is None:
        return None  # not a circular detail page (gap / other content type)

    num = NUM_RE.search(paras[head])
    # the departmental ref trails the number on the same line
    ref_code = paras[head][num.end():].strip() or None
    dept = department_of(num.group(0), ref_code)

    # date sits in one of the next couple of paragraphs
    iso = raw = None
    date_i = head
    for i in range(head + 1, min(head + 4, len(paras))):
        iso, raw = iso_date(paras[i])
        if raw:
            date_i = i
            break

    # subject is the first real line after the date, skipping "Previous Versions",
    # the addressee block and the salutation
    subject, addressee = None, []
    for t in paras[date_i + 1:date_i + 8]:
        if ADDRESSEE_RE.match(t):
            addressee.append(t)  # often split across lines ("To," / "All Authorised Persons")
            continue
        if SKIP_P_RE.match(t) or len(t) < 12:
            continue
        subject = t[:400]
        break
    meant_for = " ".join(addressee)[:300] or None

    return {
        "key": f"circular:{rbi_id}",
        "doc_type": "circular",
        "rbi_id": rbi_id,
        "number": num.group(0),
        "issue_date": iso,
        "raw_date": raw,
        "department": dept,
        "ref_code": ref_code,
        "meant_for": meant_for,
        "applies_to": json.dumps(applies_to_of(subject, meant_for)),
        "subject": subject,
        "source_url": CIRCULAR_DETAIL.format(id=rbi_id),
        "pdf_url": pick_pdf(soup),
        "local_path": None,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

# ----------------------------- phase 1: harvest -----------------------------
async def harvest():
    con = db()
    ids = list(range(START_ID, START_ID - SCAN_BACK, -1))
    sem = asyncio.Semaphore(CONCURRENCY)
    banked = 0

    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        async def one(rbi_id):
            nonlocal banked
            if banked >= TARGET_DOCS:
                return
            async with sem:
                await asyncio.sleep(REQUEST_DELAY)
                try:
                    r = await client.get(CIRCULAR_DETAIL.format(id=rbi_id))
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"  id={rbi_id} skip ({e.__class__.__name__})")
                    return
            row = parse_circular(rbi_id, r.text)
            if not row:
                return
            if STOP_BEFORE and row["issue_date"] and \
               datetime.fromisoformat(row["issue_date"]) < STOP_BEFORE:
                return
            upsert(con, row)
            banked += 1
            print(f"  [{banked:>3}] {row['number']}  {row['issue_date']}  {row['subject'][:60]}")

        # process in ID order, chunked so TARGET_DOCS can short-circuit
        for i in range(0, len(ids), CONCURRENCY * 4):
            if banked >= TARGET_DOCS:
                break
            await asyncio.gather(*(one(x) for x in ids[i:i + CONCURRENCY * 4]))

    print(f"harvest done: {banked} circulars in manifest")

# ----------------------------- phase 2: master directions -------------------
async def directions():
    con = db()
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        r = await client.get(MASTER_DIRECTIONS)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    n = 0
    for a in soup.find_all("a", href=True):
        subject = a.get_text(" ", strip=True)
        if len(subject) < 12:  # skip nav chrome
            continue
        href = a["href"]
        detail = href if href.startswith("http") else BASE + "/" + href.lstrip("/")
        # the row text around the link usually carries the date
        row_text = a.find_parent(["tr", "li", "p"])
        iso, raw = iso_date(row_text.get_text(" ", strip=True) if row_text else subject)
        if "MasterDirection" not in href and "MasterDirection" not in subject.replace(" ", ""):
            # keep it loose: MDs link out via various script names — filter by page section instead
            if "Master Direction" not in subject:
                continue
        key = f"md:{abs(hash(detail)) % 10**9}"
        upsert(con, {
            "key": key, "doc_type": "master_direction", "rbi_id": None,
            "number": None, "issue_date": iso, "raw_date": raw,
            "department": None, "ref_code": None,
            "meant_for": "Master Direction",
            "applies_to": json.dumps(scan_entities(subject)),
            "subject": subject,
            "source_url": detail, "pdf_url": None, "local_path": None,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        n += 1
    print(f"directions done: {n} master directions in manifest "
          f"(pdf_url resolved lazily at download time)")

# ----------------------------- phase 3: download ----------------------------
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120].strip("_")


async def resolve_pdf(client, source_url):
    """Follow a detail page and pull its first PDF link (for MDs / missing pdf_url)."""
    try:
        r = await client.get(source_url)
        r.raise_for_status()
    except httpx.HTTPError:
        return None
    return pick_pdf(BeautifulSoup(r.text, "html.parser"))


async def download():
    con = db()
    rows = con.execute(
        "SELECT key, number, issue_date, meant_for, subject, source_url, pdf_url "
        "FROM docs WHERE local_path IS NULL"
    ).fetchall()
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        async def one(row):
            key, number, iso, meant_for, subject, source_url, pdf_url = row
            async with sem:
                await asyncio.sleep(REQUEST_DELAY)
                if not pdf_url:
                    pdf_url = await resolve_pdf(client, source_url)
                if not pdf_url:
                    print(f"  no pdf for {number or subject[:40]}")
                    return
                try:
                    r = await client.get(pdf_url)
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"  fail {pdf_url} ({e.__class__.__name__})")
                    return
                # rbidocs sits behind a WAF that answers with a 200 + HTML challenge
                # page. Without this check those get written out as .pdf files and the
                # whole corpus silently becomes 500 copies of a captcha.
                if not r.content.startswith(b"%PDF-"):
                    print(f"  BLOCKED (not a PDF, got {r.headers.get('content-type','?')}) "
                          f"{number or (subject or '')[:40]}")
                    return

            # organize primary copy by date: YYYY/MM/
            y, m = (iso[:4], iso[5:7]) if iso else ("undated", "00")
            folder = PDF_ROOT / y / m
            folder.mkdir(parents=True, exist_ok=True)
            fname = safe_name(number or subject or key) + ".pdf"
            path = folder / fname
            path.write_bytes(r.content)
            con.execute("UPDATE docs SET pdf_url=?, local_path=? WHERE key=?",
                        (pdf_url, str(path), key))
            con.commit()
            print(f"  saved {path}")

        await asyncio.gather(*(one(r) for r in rows))
    print("download done")

# ----------------------------- phase 3b: body text --------------------------
async def fetch_text():
    """Cache each circular's body HTML locally.

    The detail pages carry the full text of the circular — including tables as real
    <table> markup — so they, not the PDFs, are the corpus source. (The PDF host sits
    behind a WAF that answers automated requests with a challenge page; see download().)
    Caching to disk keeps chunking iteration offline instead of re-hitting the site.
    """
    con = db()
    HTML_ROOT.mkdir(parents=True, exist_ok=True)
    rows = con.execute(
        "SELECT key, source_url FROM docs WHERE doc_type='circular' AND body_path IS NULL"
    ).fetchall()
    sem = asyncio.Semaphore(CONCURRENCY)
    saved = skipped = 0

    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:
        async def one(key, source_url):
            nonlocal saved, skipped
            path = HTML_ROOT / (safe_name(key) + ".html")
            if not path.exists():
                async with sem:
                    await asyncio.sleep(REQUEST_DELAY)
                    try:
                        r = await client.get(source_url)
                        r.raise_for_status()
                    except httpx.HTTPError as e:
                        print(f"  fail {key} ({e.__class__.__name__})")
                        return
                body = BeautifulSoup(r.text, "html.parser").find("table", class_="td")
                if body is None:
                    skipped += 1
                    return
                path.write_text(str(body), encoding="utf-8")
                saved += 1
            con.execute("UPDATE docs SET body_path=? WHERE key=?", (str(path), key))
            con.commit()

        await asyncio.gather(*(one(k, u) for k, u in rows))
    print(f"text done: {saved} bodies cached in {HTML_ROOT} ({skipped} had no body table)")

# ----------------------------- phase 4: organize ----------------------------
def organize():
    """Category views as symlinks (no file duplication) + a flat CSV index."""
    con = db()
    CATEGORY_ROOT.mkdir(parents=True, exist_ok=True)

    rows = con.execute(
        "SELECT applies_to, department, local_path "
        "FROM docs WHERE local_path IS NOT NULL"
    ).fetchall()

    for applies_to, dept, local_path in rows:
        # a circular can bind several entity classes, so it appears under each
        cats = json.loads(applies_to or "[]") or []
        for cat in [f"entity/{c}" for c in cats] + ([f"department/{dept}"] if dept else []):
            cdir = CATEGORY_ROOT / Path(cat).parent / safe_name(Path(cat).name)
            cdir.mkdir(parents=True, exist_ok=True)
            link = cdir / Path(local_path).name
            if not link.exists():
                try:
                    os.symlink(Path(local_path).resolve(), link)
                except OSError:
                    pass  # symlinks unsupported (e.g. some Windows setups) -> skip

    with open(ROOT / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "issue_date", "department", "applies_to",
                    "meant_for", "subject", "path"])
        for r in con.execute(
            "SELECT number, issue_date, department, applies_to, meant_for, subject, local_path "
            "FROM docs ORDER BY issue_date DESC"
        ):
            r = list(r)
            r[3] = "; ".join(json.loads(r[3] or "[]"))  # flatten for spreadsheet use
            w.writerow(r)
    print(f"organize done: category views in {CATEGORY_ROOT}, index at {ROOT/'index.csv'}")

# ----------------------------- cli ------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "harvest"
    if cmd == "harvest":
        asyncio.run(harvest())
    elif cmd == "directions":
        asyncio.run(directions())
    elif cmd == "download":
        asyncio.run(download())
    elif cmd == "text":
        asyncio.run(fetch_text())
    elif cmd == "organize":
        organize()
    else:
        print(__doc__)
