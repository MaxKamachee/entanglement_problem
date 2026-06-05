"""Build the substrate corpus — the "dual" bucket of the offense/dual/defense Venn.

Substrate = operational prerequisite knowledge (networking, OS, architecture, web,
crypto, recon) that both offensive and defensive practitioners depend on. The
thesis this corpus enables is that the shared substrate is the locus of
offense/defense entanglement (dual-use *by application*).

`configs/substrate_sources.yaml` is the source of truth for composition; handlers
here are generic per source `type`. Each source unit produces chunks (structural-
boundary chunking, never character windows), appended to
`data/substrate_corpus.parquet` with cross-source content-hash dedup, plus a
`data/build_reports/<id>.yaml` build report.

This module is built incrementally, one handler type per source tier. Handler
types implemented so far: rfc_text.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from entanglement.config import scrape_user_agent
from entanglement.scrape import content_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCES_YAML = ROOT / "configs" / "substrate_sources.yaml"
CORPUS_PATH = ROOT / "data" / "substrate_corpus.parquet"
REPORTS_DIR = ROOT / "data" / "build_reports"
SUBSTRATE_CACHE = ROOT / "data" / "substrate_cache"

TOPICS = {"networking", "os_internals", "architecture", "web", "crypto", "recon"}

SUBSTRATE_COLUMNS = [
    "chunk_id",             # sha256 of text — also the dedup key
    "source_ids",           # list[str]: source units this chunk came from (cross-source dedup tags both)
    "source_name",
    "topic",
    "version_pin",
    "text",
    "n_chars",
    "chunk_position",       # structural locator (e.g. RFC section number)
    "extractor",
    "extraction_warnings",  # list[str]
    "license",
]

# polars schema so empty frames and list columns type correctly
_SCHEMA = {
    "chunk_id": pl.String, "source_ids": pl.List(pl.String), "source_name": pl.String,
    "topic": pl.String, "version_pin": pl.String, "text": pl.String, "n_chars": pl.Int64,
    "chunk_position": pl.String, "extractor": pl.String,
    "extraction_warnings": pl.List(pl.String), "license": pl.String,
}

MIN_CHUNK_CHARS = 80  # drop section fragments / stray headers below this


# --------------------------------------------------------------------------- #
# Density / quality
# --------------------------------------------------------------------------- #

def alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are alphabetic."""
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return 0.0
    return sum(c.isalpha() for c in non_ws) / len(non_ws)


def density_decision(text: str, policy: str) -> tuple[bool, list[str]]:
    """Return (drop, warnings) per the source's density policy.

    flag_only: never drop on density (legitimately symbol-dense sources); warn if low.
    standard : >0.7 clean; 0.5-0.7 keep+warn; <0.5 drop.
    """
    ratio = alpha_ratio(text)
    if ratio >= 0.7:
        return False, []
    warn = [f"low_text_density:{ratio:.2f}"]
    if policy == "flag_only":
        return False, warn
    if ratio < 0.5:
        return True, warn
    return False, warn


# --------------------------------------------------------------------------- #
# RFC handler
# --------------------------------------------------------------------------- #

_FOOTER_RE = re.compile(r"\[Page \d+\]\s*$")
_RUNHEADER_RE = re.compile(r"^RFC \d+\s")
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_APPENDIX_RE = re.compile(r"^(Appendix [A-Z][0-9]*)\.?\s+(\S.*)$", re.IGNORECASE)
_TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")


def strip_rfc_artifacts(raw: str) -> str:
    """Remove page breaks, ``[Page N]`` footers, and running headers from an RFC .txt."""
    raw = raw.replace("\r\n", "\n").replace("\f", "\n")
    kept = [
        line for line in raw.split("\n")
        if not _FOOTER_RE.search(line) and not _RUNHEADER_RE.match(line)
    ]
    return "\n".join(kept)


def _is_toc(text: str) -> bool:
    """True if the chunk is dominated by table-of-contents dot leaders."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    leaders = sum(1 for ln in lines if _TOC_LEADER_RE.search(ln))
    return leaders / len(lines) > 0.5


def split_rfc_sections(text: str) -> list[tuple[str, str]]:
    """Split cleaned RFC text into (section_position, section_text) at section headers."""
    sections: list[tuple[str, list[str]]] = []
    pos, cur = "front", []
    for line in text.split("\n"):
        m = _SECTION_RE.match(line) or _APPENDIX_RE.match(line)
        if m:
            sections.append((pos, cur))
            pos, cur = m.group(1), [line]
        else:
            cur.append(line)
    sections.append((pos, cur))
    return [(p, "\n".join(ls).strip()) for p, ls in sections]


def _emit_chunks(
    sections: list[tuple[str, str]],
    *,
    source_id: str,
    source_name: str,
    topic: str,
    version_pin: str,
    extractor: str,
    license_: str,
    policy: str,
    dropped: dict[str, int],
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[dict]:
    """Turn (position, text) sections into chunk rows, applying the drop/flag policy.

    Shared by every handler so chunk semantics (too-short, ToC, density, schema) are
    identical across source types. ``dropped`` is mutated with per-reason counts.
    ``min_chars`` is overridable per source (PDFs carry more short-fragment noise —
    front-matter boilerplate, figure captions, ToC lines — than plain-text RFCs).
    """
    if topic not in TOPICS:
        raise ValueError(f"{source_id}: topic '{topic}' not in {TOPICS}")
    rows: list[dict] = []
    for position, sec_text in sections:
        if len(sec_text) < min_chars:
            dropped["too_short"] = dropped.get("too_short", 0) + 1
            continue
        if _is_toc(sec_text):
            dropped["table_of_contents"] = dropped.get("table_of_contents", 0) + 1
            continue
        drop, warnings = density_decision(sec_text, policy)
        if drop:
            dropped["low_text_density"] = dropped.get("low_text_density", 0) + 1
            continue
        rows.append({
            "chunk_id": content_hash(sec_text),
            "source_ids": [source_id],
            "source_name": source_name,
            "topic": topic,
            "version_pin": version_pin,
            "text": sec_text,
            "n_chars": len(sec_text),
            "chunk_position": position,
            "extractor": extractor,
            "extraction_warnings": warnings,
            "license": license_,
        })
    return rows


def _build_report(df: pl.DataFrame, *, source_id: str, version_pin: str, license_: str,
                  dropped: dict[str, int]) -> dict:
    df2, intra = _dedup_within(df)
    if intra:
        dropped["intra_source_duplicate"] = intra
    report = {
        "source_id": source_id,
        "source_version_pin": version_pin,
        "chunks_produced": df2.height,
        "chunks_dropped": sum(dropped.values()),
        "chunks_dropped_reasons": dropped,
        "chars_total": int(df2["n_chars"].sum()) if df2.height else 0,
        "quality_flags": sorted(
            {w.split(":")[0] for ws in df2["extraction_warnings"].to_list() for w in ws}
        ),
        "license_note": license_,
        "extraction_warnings": [],
    }
    return df2, report


def build_rfc_source(entry: dict, *, fetcher) -> tuple[pl.DataFrame, dict]:
    """Build chunks for the RFC source unit. ``fetcher(number) -> raw_text`` (injectable)."""
    sid, license_ = entry["id"], entry["license"]
    policy = entry.get("density_policy", "standard")
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for rfc in entry["rfcs"]:
        number, topic = rfc["number"], rfc["topic"]
        cleaned = strip_rfc_artifacts(fetcher(number))
        rows += _emit_chunks(
            split_rfc_sections(cleaned),
            source_id=sid, source_name=f"RFC {number}", topic=topic,
            version_pin=f"RFC{number}", extractor="rfc_text",
            license_=license_, policy=policy, dropped=dropped,
        )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=entry["version_pin"],
                         license_=license_, dropped=dropped)


# --------------------------------------------------------------------------- #
# pdf_single handler — one PDF, document-section chunking
# --------------------------------------------------------------------------- #

_PDF_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z].*)$")
_PDF_APPENDIX_RE = re.compile(r"^(Appendix\s+[A-Z][0-9]*)\.?\s*(\S.*)$", re.IGNORECASE)
_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")


def split_doc_sections(text: str) -> list[tuple[str, str]]:
    """Split extracted PDF text into (section_position, text) at numbered/appendix headers.

    Strips bare page-number lines and form-feeds (the main PDF-extraction noise).
    """
    lines = [ln for ln in text.replace("\f", "\n").split("\n") if not _PAGENUM_RE.match(ln)]
    sections: list[tuple[str, list[str]]] = []
    pos, cur = "front", []
    for line in lines:
        m = _PDF_SECTION_RE.match(line) or _PDF_APPENDIX_RE.match(line)
        if m:
            sections.append((pos, cur))
            pos, cur = m.group(1), [line]
        else:
            cur.append(line)
    sections.append((pos, cur))
    return [(p, "\n".join(ls).strip()) for p, ls in sections]


def build_pdf_single_source(entry: dict, *, extracted_text: str, version_pin: str
                            ) -> tuple[pl.DataFrame, dict]:
    """Build chunks from a single already-extracted PDF (pure; network/extract live in dispatch)."""
    sid, license_ = entry["id"], entry["license"]
    rows = _emit_chunks(
        split_doc_sections(extracted_text),
        source_id=sid, source_name=entry["source_name"], topic=entry["topic"],
        version_pin=version_pin, extractor="pdf_single",
        license_=license_, policy=entry.get("density_policy", "standard"),
        dropped=(dropped := {}),
        min_chars=entry.get("min_chunk_chars", MIN_CHUNK_CHARS),
    )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=version_pin,
                         license_=license_, dropped=dropped)


def build_pdf_chapter_collection(entry: dict, *, chapters: list[tuple[str, str, str]]
                                 ) -> tuple[pl.DataFrame, dict]:
    """Build chunks from a collection of chapter PDFs (pure; download/extract live in dispatch).

    ``chapters`` is a list of (stem, extracted_text, version_pin). Each chapter is
    document-section split like ``pdf_single``; chunks are tagged with a per-chapter
    source_name and version_pin so provenance is per chapter.
    """
    sid, license_ = entry["id"], entry["license"]
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for stem, text, vpin in chapters:
        rows += _emit_chunks(
            split_doc_sections(text),
            source_id=sid, source_name=f"{entry['source_name']}: {stem}", topic=entry["topic"],
            version_pin=vpin, extractor="pdf_chapter_collection",
            license_=license_, policy=entry.get("density_policy", "standard"),
            dropped=dropped, min_chars=entry.get("min_chunk_chars", MIN_CHUNK_CHARS),
        )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=entry["version_pin"],
                         license_=license_, dropped=dropped)


def build_pdf_chapter_extract(entry: dict, *, ranges: list[tuple[str, str, str]]
                              ) -> tuple[pl.DataFrame, dict]:
    """Build chunks from page-range extracts of one large PDF (pure; download/slice in dispatch).

    ``ranges`` is a list of (label, extracted_text, version_pin) — one per pinned page range
    (e.g. a conceptual chapter), document-section split within each.
    """
    sid, license_ = entry["id"], entry["license"]
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for label, text, vpin in ranges:
        rows += _emit_chunks(
            split_doc_sections(text),
            source_id=sid, source_name=f"{entry['source_name']}: {label}", topic=entry["topic"],
            version_pin=vpin, extractor="pdf_chapter_extract",
            license_=license_, policy=entry.get("density_policy", "standard"),
            dropped=dropped, min_chars=entry.get("min_chunk_chars", MIN_CHUNK_CHARS),
        )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=entry["version_pin"],
                         license_=license_, dropped=dropped)


def extract_pdf_pages(path: Path, start: int, end: int) -> str:
    """Extract text from a 1-indexed inclusive page range [start, end] of a PDF."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = reader.pages[start - 1:end]
    return "\n".join((p.extract_text() or "") for p in pages)


def download_pdf(url: str, dest: Path, *, client) -> tuple[Path, str]:
    """Download a PDF to ``dest``; return (path, sha256-of-bytes)."""
    resp = client.get(url, follow_redirects=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest, hashlib.sha256(resp.content).hexdigest()


def extract_pdf_text(path: Path) -> str:
    """Extract text from every page of a PDF via pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# --------------------------------------------------------------------------- #
# github_markdown handler — clone a repo subtree, one chunk per markdown file
# --------------------------------------------------------------------------- #

_MD_MACRO_RE = re.compile(r"\{\{[^}]*\}\}")  # MDN KumaScript macros, e.g. {{cssxref("x")}}


def strip_markdown_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block (``---`` ... ``---``) if present."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def clean_markdown(text: str) -> str:
    """Strip MDN frontmatter and KumaScript macros, leaving readable prose."""
    return _MD_MACRO_RE.sub("", strip_markdown_frontmatter(text)).strip()


def build_github_markdown_source(entry: dict, *, walk_dir: Path, version_pin: str
                                 ) -> tuple[pl.DataFrame, dict]:
    """One chunk per ``*.md`` file under ``walk_dir`` (pure; clone lives in dispatch)."""
    sid, license_ = entry["id"], entry["license"]
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for path in sorted(walk_dir.rglob("*.md")):
        cleaned = clean_markdown(path.read_text(encoding="utf-8", errors="ignore"))
        rows += _emit_chunks(
            [(str(path.relative_to(walk_dir)), cleaned)],
            source_id=sid, source_name=entry["source_name"], topic=entry["topic"],
            version_pin=version_pin, extractor="github_markdown",
            license_=license_, policy=entry.get("density_policy", "standard"),
            dropped=dropped, min_chars=entry.get("min_chunk_chars", MIN_CHUNK_CHARS),
        )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=version_pin,
                         license_=license_, dropped=dropped)


# --------------------------------------------------------------------------- #
# html_book_chapters handler — index page lists chapter URLs; one chunk per page
# --------------------------------------------------------------------------- #

_HREF_RE = re.compile(r"""href=["']([^"'#:]+\.html)["']""")  # relative .html only (": " excludes abs URLs)


def discover_chapter_stems(index_html: str, deny_prefixes: list[str]) -> list[str]:
    """Extract relative chapter stems from an index/ToC page, minus denied prefixes."""
    seen: set[str] = set()
    out: list[str] = []
    for href in _HREF_RE.findall(index_html):
        stem = href.split("/")[-1][:-5]  # drop ".html"
        if stem in seen:
            continue
        seen.add(stem)
        if any(stem == d or stem.startswith(d) for d in deny_prefixes):
            continue
        out.append(stem)
    return out


def build_html_book_chapters(entry: dict, *, pages: list[tuple[str, str, str]]
                             ) -> tuple[pl.DataFrame, dict]:
    """One chunk per book page (pure; fetch/extract live in dispatch).

    ``pages`` is a list of (stem, extracted_text, version_pin).
    """
    sid, license_ = entry["id"], entry["license"]
    rows: list[dict] = []
    dropped: dict[str, int] = {}
    for stem, text, vpin in pages:
        rows += _emit_chunks(
            [(stem, text)],
            source_id=sid, source_name=f"{entry['source_name']}: {stem}", topic=entry["topic"],
            version_pin=vpin, extractor="html_book_chapters",
            license_=license_, policy=entry.get("density_policy", "standard"),
            dropped=dropped, min_chars=entry.get("min_chunk_chars", MIN_CHUNK_CHARS),
        )
    df = pl.DataFrame(rows, schema=_SCHEMA)
    return _build_report(df, source_id=sid, version_pin=entry["version_pin"],
                         license_=license_, dropped=dropped)


def sparse_clone(repo_url: str, subdir: str, dest: Path) -> str:
    """Shallow, blobless, sparse clone of just ``subdir``; return the pinned HEAD SHA."""
    import subprocess

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(dest)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "sparse-checkout", "set", subdir],
            check=True, capture_output=True,
        )
    out = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------- #
# Generic corpus assembly
# --------------------------------------------------------------------------- #

def _dedup_within(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop intra-source duplicate chunks (same content hash), keep first."""
    if df.is_empty():
        return df, 0
    deduped = df.unique(subset=["chunk_id"], keep="first", maintain_order=True)
    return deduped, df.height - deduped.height


def append_to_corpus(new_chunks: pl.DataFrame, path: Path = CORPUS_PATH) -> int:
    """Append chunks with cross-source content-hash dedup. Returns # cross-source dups merged.

    A chunk already present (by ``chunk_id``) from another source has the new
    ``source_id`` merged into its ``source_ids`` list rather than added as a row.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        new_chunks.write_parquet(path)
        return 0

    existing = pl.read_parquet(path)
    by_id = {r["chunk_id"]: r for r in existing.iter_rows(named=True)}
    merged = 0
    additions: list[dict] = []
    for r in new_chunks.iter_rows(named=True):
        cid = r["chunk_id"]
        if cid in by_id:
            sids = by_id[cid]["source_ids"]
            for s in r["source_ids"]:
                if s not in sids:
                    sids.append(s)
                    merged += 1
        else:
            by_id[cid] = r
            additions.append(r)
    out = pl.DataFrame(list(by_id.values()), schema=_SCHEMA)
    out.write_parquet(path)
    return merged


def write_build_report(report: dict, *, build_dir: Path = REPORTS_DIR) -> Path:
    """Write data/build_reports/<source_id>.yaml with the required fields + timestamp."""
    build_dir.mkdir(parents=True, exist_ok=True)
    report = {**report, "build_timestamp": datetime.now(timezone.utc).isoformat()}
    path = build_dir / f"{report['source_id']}.yaml"
    path.write_text(yaml.safe_dump(report, sort_keys=False))
    return path


# --------------------------------------------------------------------------- #
# Network fetch + dispatch
# --------------------------------------------------------------------------- #

def fetch_rfc(number: int, *, client, base_url: str) -> str:
    """Fetch an RFC .txt politely (one host, sequential)."""
    resp = client.get(base_url.format(number=number), follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def load_sources(path: Path = SOURCES_YAML) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text())
    return {s["id"]: s for s in data["sources"]}


_HANDLERS = {"rfc_text", "pdf_single", "github_markdown", "pdf_chapter_collection",
             "html_book_chapters", "pdf_chapter_extract"}


def build_source(source_id: str, *, client=None) -> tuple[pl.DataFrame, dict]:
    """Dispatch to the handler for ``source_id`` and return (chunks, report)."""
    import httpx

    entry = load_sources()[source_id]
    stype = entry["type"]
    if stype not in _HANDLERS:
        raise NotImplementedError(f"handler '{stype}' not implemented yet")

    own = client is None
    if own:
        client = httpx.Client(headers={"user-agent": scrape_user_agent()}, timeout=60.0)
    try:
        if stype == "rfc_text":
            base_url = entry["base_url"]

            def fetcher(number: int) -> str:
                text = fetch_rfc(number, client=client, base_url=base_url)
                time.sleep(1.0)  # polite: 1 req/s to rfc-editor.org
                return text

            return build_rfc_source(entry, fetcher=fetcher)

        if stype == "pdf_single":
            dest = SUBSTRATE_CACHE / f"{source_id}.pdf"
            _, sha = download_pdf(entry["url"], dest, client=client)
            text = extract_pdf_text(dest)
            return build_pdf_single_source(
                entry, extracted_text=text, version_pin=f"download {_today()}; sha256:{sha}"
            )

        if stype == "github_markdown":
            repo_dir = SUBSTRATE_CACHE / source_id
            sha = sparse_clone(entry["repo"], entry["subdir"], repo_dir)
            return build_github_markdown_source(
                entry, walk_dir=repo_dir / entry["subdir"],
                version_pin=f"git {entry['repo']}@{sha}",
            )

        if stype == "pdf_chapter_collection":
            cache = SUBSTRATE_CACHE / source_id
            chapters = []
            for stem in entry["chapters"]:
                dest = cache / f"{stem}.pdf"
                _, sha = download_pdf(entry["base_url"].format(stem=stem), dest, client=client)
                chapters.append((stem, extract_pdf_text(dest), f"{stem}; sha256:{sha}"))
                time.sleep(1.0)  # polite: 1 req/s
            return build_pdf_chapter_collection(entry, chapters=chapters)

        if stype == "html_book_chapters":
            from entanglement.scrape import extract_html

            index_html = client.get(entry["index_url"], follow_redirects=True).text
            stems = discover_chapter_stems(index_html, entry.get("deny_prefixes", []))
            pages = []
            for stem in stems:
                url = entry["base_url"].format(stem=stem)
                time.sleep(1.0)  # polite: 1 req/s
                resp = client.get(url, follow_redirects=True)
                if resp.status_code >= 400:
                    continue
                text = extract_html(resp.content)
                if not text:
                    continue
                sha = hashlib.sha256(resp.content).hexdigest()
                pages.append((stem, text, f"{url}; sha256:{sha}"))
            return build_html_book_chapters(entry, pages=pages)

        if stype == "pdf_chapter_extract":
            dest = SUBSTRATE_CACHE / f"{source_id}.pdf"
            _, sha = download_pdf(entry["url"], dest, client=client)
            ranges = []
            for r in entry["page_ranges"]:
                label, start, end = r["label"], int(r["start"]), int(r["end"])
                text = extract_pdf_pages(dest, start, end)
                ranges.append((label, text, f"{label}; pages {start}-{end}; sha256:{sha}"))
            return build_pdf_chapter_extract(entry, ranges=ranges)
        raise NotImplementedError(stype)  # unreachable
    finally:
        if own:
            client.close()


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def source_location(entry: dict) -> str:
    """Human-readable exact origin of a source, for the build report (the 'where')."""
    if entry.get("url"):
        return entry["url"]
    if entry.get("repo"):
        return f"git {entry['repo']} :: {entry.get('subdir', '')}"
    if entry.get("base_url"):
        loc = entry["base_url"]
        if entry.get("index_url"):
            return f"{entry['index_url']} (index) -> {loc}"
        return loc
    return entry.get("index_url", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a substrate corpus source unit.")
    parser.add_argument("source_id", help="source id from configs/substrate_sources.yaml")
    args = parser.parse_args()

    entry = load_sources()[args.source_id]
    chunks, report = build_source(args.source_id)
    report["source_location"] = source_location(entry)
    report["handler"] = entry["type"]
    merged = append_to_corpus(chunks)
    if merged:
        report["chunks_dropped_reasons"]["cross_source_duplicate"] = merged
        report["chunks_dropped"] += merged
        report["chunks_produced"] -= merged
    report_path = write_build_report(report)

    print(f"source: {report['source_id']}")
    print(f"chunks produced: {report['chunks_produced']}  dropped: {report['chunks_dropped']}")
    print(f"dropped reasons: {report['chunks_dropped_reasons']}")
    print(f"chars total:     {report['chars_total']:,}")
    print(f"quality flags:   {report['quality_flags']}")
    print(f"report:          {report_path.relative_to(ROOT)}")
    total = pl.read_parquet(CORPUS_PATH)
    print(f"corpus now:      {total.height} chunks across sources "
          f"{sorted({s for ss in total['source_ids'].to_list() for s in ss})}")


if __name__ == "__main__":
    main()
