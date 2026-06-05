"""Shared scraping pipeline: fetch -> extract -> dedup, with full provenance.

Used by both corpora. Politeness is enforced (1 req/sec per host, robots.txt
respected). Content is extracted with trafilatura (HTML) or pypdf (PDF), deduped
first by canonical URL and then by content SHA-256, and stub-filtered. Every
attempted URL produces a provenance row recording the outcome.

The pipeline is idempotent and resumable: each fetched URL is cached on disk by a
hash of its canonical form, so re-runs are free and a crashed run resumes from the
cache. ``trafilatura`` and ``pypdf`` are imported lazily inside the extractors so
this module (and its pure-function tests) import without those deps installed.

Network execution is deferred — wired here, run by the corpus orchestrators.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import polars as pl

from entanglement.config import stub_min_chars
from entanglement.normalize import canonicalize_url

DOCUMENTS_COLUMNS = ["content_hash", "url", "raw_url", "extractor", "n_chars", "text"]
PROVENANCE_COLUMNS = [
    "url", "raw_url", "fetch_ts", "http_status",
    "content_hash", "extractor", "n_chars", "success", "reason",
]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no network)
# --------------------------------------------------------------------------- #

def content_hash(text: str) -> str:
    """SHA-256 hex digest of extracted text (the document dedup key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_stub(text: str, min_chars: int | None = None) -> bool:
    """True if the extracted text is too short to be a real document."""
    threshold = stub_min_chars() if min_chars is None else min_chars
    return len(text) < threshold


def classify_content_type(content_type: str | None, url: str) -> str:
    """Return "pdf" | "html" | "other" from a Content-Type header (URL fallback)."""
    ct = (content_type or "").lower()
    if "application/pdf" in ct or urlsplit(url).path.lower().endswith(".pdf"):
        return "pdf"
    if "html" in ct or "xml" in ct or ct == "":
        return "html"
    if ct.startswith("text/"):
        return "html"
    return "other"


def dedup_urls(urls: list[str]) -> list[str]:
    """Canonicalize and de-duplicate URLs, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        canon = canonicalize_url(raw)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def extract_html(raw: bytes | str) -> str | None:
    """Extract main text from HTML via trafilatura; None if nothing usable."""
    import trafilatura

    html = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    return trafilatura.extract(html, include_comments=False, include_tables=True)


def extract_pdf(raw: bytes) -> str | None:
    """Extract text from a PDF via pypdf; None on failure."""
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception:
        return None
    text = "\n".join(pages).strip()
    return text or None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Politeness
# --------------------------------------------------------------------------- #

class RateLimiter:
    """Enforce a minimum interval between requests to the same host."""

    def __init__(self, per_host_interval: float = 1.0, *, clock=time.monotonic, sleep=time.sleep):
        self.interval = per_host_interval
        self._clock = clock
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def wait(self, host: str) -> None:
        now = self._clock()
        last = self._last.get(host)
        if last is not None:
            elapsed = now - last
            if elapsed < self.interval:
                self._sleep(self.interval - elapsed)
        self._last[host] = self._clock()


class RobotsCache:
    """Cache robots.txt per host and answer can_fetch. Fail-open on fetch error."""

    def __init__(self, user_agent: str, *, fetcher=None):
        self.user_agent = user_agent
        self._fetcher = fetcher  # callable(robots_url) -> text | None; None => urllib
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}

    def _parser_for(self, scheme: str, host: str):
        key = f"{scheme}://{host}"
        if key in self._parsers:
            return self._parsers[key]
        rp = robotparser.RobotFileParser()
        robots_url = f"{key}/robots.txt"
        try:
            if self._fetcher is not None:
                text = self._fetcher(robots_url)
                if text is None:
                    raise OSError("no robots")
                rp.parse(text.splitlines())
            else:
                rp.set_url(robots_url)
                rp.read()
        except Exception:
            rp = None  # fail-open: treat as allowed
        self._parsers[key] = rp
        return rp

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        rp = self._parser_for(parts.scheme or "https", parts.netloc.lower())
        return True if rp is None else rp.can_fetch(self.user_agent, url)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

@dataclass
class FetchResult:
    url: str            # canonical
    raw_url: str
    fetch_ts: str
    http_status: int | None
    text: str | None
    extractor: str | None
    success: bool
    reason: str         # "" on success; else robots_disallow|dead|non_text|stub|error:<msg>


def _cache_path(cache_dir: Path, canon_url: str) -> Path:
    return cache_dir / f"{hashlib.sha256(canon_url.encode()).hexdigest()}.json"


def fetch_one(
    raw_url: str,
    *,
    client,
    limiter: RateLimiter,
    robots: RobotsCache,
    cache_dir: Path | None = None,
) -> FetchResult:
    """Fetch, extract, and classify a single URL. Cached + resumable."""
    canon = canonicalize_url(raw_url)
    if cache_dir is not None:
        cpath = _cache_path(cache_dir, canon)
        if cpath.exists():
            return FetchResult(**json.loads(cpath.read_text()))

    def finish(result: FetchResult) -> FetchResult:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _cache_path(cache_dir, canon).write_text(json.dumps(asdict(result)))
        return result

    if not robots.allowed(canon):
        return finish(FetchResult(canon, raw_url, _now_iso(), None, None, None, False,
                                  "robots_disallow"))

    host = urlsplit(canon).netloc.lower()
    limiter.wait(host)
    try:
        resp = client.get(canon, follow_redirects=True)
        status = resp.status_code
        if status >= 400:
            return finish(FetchResult(canon, raw_url, _now_iso(), status, None, None, False,
                                      "dead"))
        kind = classify_content_type(resp.headers.get("content-type"), canon)
        if kind == "other":
            return finish(FetchResult(canon, raw_url, _now_iso(), status, None, None, False,
                                      "non_text"))
        text = extract_pdf(resp.content) if kind == "pdf" else extract_html(resp.content)
        extractor = "pypdf" if kind == "pdf" else "trafilatura"
        if not text:
            return finish(FetchResult(canon, raw_url, _now_iso(), status, None, extractor, False,
                                      "non_text"))
        if is_stub(text):
            return finish(FetchResult(canon, raw_url, _now_iso(), status, None, extractor, False,
                                      "stub"))
        return finish(FetchResult(canon, raw_url, _now_iso(), status, text, extractor, True, ""))
    except Exception as exc:  # network / parse errors
        return finish(FetchResult(canon, raw_url, _now_iso(), None, None, None, False,
                                  f"error:{type(exc).__name__}"))


def results_to_frames(results: list[FetchResult]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split fetch results into (documents deduped by content_hash, provenance)."""
    prov_rows: list[dict] = []
    doc_rows: list[dict] = []
    seen_hashes: set[str] = set()
    for r in results:
        h = content_hash(r.text) if r.success and r.text else None
        prov_rows.append({
            "url": r.url, "raw_url": r.raw_url, "fetch_ts": r.fetch_ts,
            "http_status": r.http_status, "content_hash": h, "extractor": r.extractor,
            "n_chars": len(r.text) if r.text else None, "success": r.success, "reason": r.reason,
        })
        if h and h not in seen_hashes:
            seen_hashes.add(h)
            doc_rows.append({
                "content_hash": h, "url": r.url, "raw_url": r.raw_url,
                "extractor": r.extractor, "n_chars": len(r.text), "text": r.text,
            })
    documents = pl.DataFrame(doc_rows, schema=DOCUMENTS_COLUMNS)
    provenance = pl.DataFrame(prov_rows, schema=PROVENANCE_COLUMNS)
    return documents, provenance


def scrape(
    raw_urls: list[str],
    *,
    cache_dir: Path,
    user_agent: str,
    per_host_interval: float = 1.0,
    client=None,
    progress_every: int = 25,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fetch all URLs (deduped by canonical form) and return (documents, provenance).

    Emits a progress bar to stderr every ``progress_every`` URLs (0 to disable) —
    count/total/percent + ok/fail tally + elapsed + ETA. Cache hits fly through, so the
    bar advances fast over already-scraped URLs on a resumed run.
    """
    import sys
    import time

    import httpx

    urls = dedup_urls(raw_urls)
    total = len(urls)
    own_client = client is None
    if own_client:
        client = httpx.Client(headers={"user-agent": user_agent}, timeout=30.0)
    limiter = RateLimiter(per_host_interval)

    def _robots_fetch(robots_url: str) -> str | None:
        # bounded robots fetch via httpx (urllib.robotparser.read() has NO timeout and
        # will hang the whole sequential scrape on an unresponsive host). None => fail-open.
        try:
            resp = client.get(robots_url, follow_redirects=True, timeout=10.0)
            return resp.text if resp.status_code < 400 else ""
        except Exception:
            return None

    robots = RobotsCache(user_agent, fetcher=_robots_fetch)
    results: list[FetchResult] = []
    ok = 0
    t0 = time.monotonic()
    try:
        for i, u in enumerate(urls, 1):
            r = fetch_one(u, client=client, limiter=limiter, robots=robots, cache_dir=cache_dir)
            results.append(r)
            ok += int(r.success)
            if progress_every and (i % progress_every == 0 or i == total):
                el = time.monotonic() - t0
                eta = (total - i) * el / i if i else 0.0
                pct = 100 * i / total if total else 100.0
                bar = "#" * int(pct / 2.5) + "-" * (40 - int(pct / 2.5))
                print(f"[scrape {i}/{total} {pct:4.1f}%] [{bar}] ok={ok} fail={i - ok} "
                      f"{el / 60:.1f}m ETA~{eta / 60:.0f}m", file=sys.stderr, flush=True)
    finally:
        if own_client:
            client.close()
    return results_to_frames(results)
