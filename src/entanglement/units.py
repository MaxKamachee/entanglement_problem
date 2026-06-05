"""Normalized analysis units — a comparable-granularity, format-stripped view of a corpus.

The structural chunks in ``*_corpus.parquet`` are the provenance record (one chunk =
one RFC section / PDF section / markdown file). For the separability experiments
(does embedding/gradient geometry predict the unlearning tax?), feeding those chunks
directly is confounded two ways: chunk size spans ~400x across sources (a length
signal that dominates embeddings), and each source carries a distinct surface format
(RFC column-wrapping, pypdf kerning, markdown/HTML) that a model can separate on
*instead of* content.

This module derives an **analysis-units** table that fixes both, without touching the
provenance corpus:

* format-strip: remove markdown/HTML markup, de-wrap hard-wrapped lines into flowing
  prose, collapse whitespace — so geometry reflects content, not source format;
* re-window: re-segment each chunk at paragraph/sentence boundaries to a consistent
  target size (splitting oversized chunks, e.g. the 40k-char RFC appendices), so the
  buckets are embedded at comparable granularity;
* drop boilerplate residue (RFC front matter, NIST process/changelog sections).

Each unit keeps ``parent_id`` (the source chunk's ``chunk_id``) so analysis results
trace back to provenance. ``build_units`` is generic so the same layer applies to the
offense/defense corpora later.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from entanglement.scrape import content_hash

ROOT = Path(__file__).resolve().parents[2]

TARGET_CHARS = 3000   # ~750 tokens — a comparable embedding-unit size
HARD_MAX = 4000       # ~1000 tokens — never exceed (most embedders cap ~512-8k tok)
MIN_TAIL = 1200       # trailing fragment shorter than this is merged into the previous unit

UNIT_COLUMNS = [
    "unit_id",        # sha256 of normalized unit text (dedup key)
    "parent_id",      # source chunk_id (provenance link)
    "source_ids",
    "source_name",
    "topic",
    "version_pin",
    "unit_position",  # "<chunk_position>#<window-index>"
    "n_chars",
    "text",           # normalized
    "license",
]

_UNIT_SCHEMA = {
    "unit_id": pl.String, "parent_id": pl.String, "source_ids": pl.List(pl.String),
    "source_name": pl.String, "topic": pl.String, "version_pin": pl.String,
    "unit_position": pl.String, "n_chars": pl.Int64, "text": pl.String, "license": pl.String,
}


# --------------------------------------------------------------------------- #
# Format stripping
# --------------------------------------------------------------------------- #

_HTML_TAG = re.compile(r"<[^>]+>")
_MD_FENCE = re.compile(r"^\s*```.*$", re.MULTILINE)        # code-fence markers (keep code body)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")            # keep link text, drop target
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_LIST = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)   # bullet markers
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_EMPH = re.compile(r"(\*\*|__|`)")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINL = re.compile(r"\n{3,}")


def strip_markup(text: str) -> str:
    """Remove HTML tags and markdown syntax, keeping the underlying text/code body."""
    text = _HTML_TAG.sub("", text)
    text = _MD_FENCE.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_EMPH.sub("", text)
    return text


def dewrap(text: str) -> str:
    """Join hard-wrapped lines within a paragraph into flowing prose (paragraph breaks kept).

    Removes the fixed-column wrapping fingerprint of RFC/PDF text. ASCII diagrams get
    flattened, but they are a small, already-low-value fraction.
    """
    paras = _PARA_SPLIT.split(text)
    joined = [
        " ".join(ln.strip() for ln in p.splitlines() if ln.strip())
        for p in paras
    ]
    return "\n\n".join(p for p in joined if p)


# Typographic ligatures that survive PDF text extraction (e.g. "deﬁned" -> "defined").
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "ft", "ﬆ": "st",
})


def clean_text(text: str) -> str:
    """Full normalization: repair ligatures, strip markup, de-wrap, collapse whitespace."""
    text = text.translate(_LIGATURES)
    text = strip_markup(text)
    text = dewrap(text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Re-segmentation
# --------------------------------------------------------------------------- #

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _wordsplit(text: str, hard_max: int) -> list[str]:
    """Last-resort split of a punctuation-free blob (grammars, hex/ASCII tables) at word
    boundaries so no unit ever exceeds hard_max."""
    out = []
    while len(text) > hard_max:
        cut = text.rfind(" ", 0, hard_max)
        cut = cut if cut > 0 else hard_max
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def _hardsplit(paragraph: str, target: int, hard_max: int) -> list[str]:
    """Split an over-long paragraph at sentence boundaries into ~target-sized pieces,
    then word-split any piece that still exceeds hard_max (no sentence delimiters)."""
    pieces, buf = [], ""
    for sent in _SENT_SPLIT.split(paragraph):
        if buf and len(buf) + len(sent) > target:
            pieces.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        pieces.append(buf.strip())
    out = []
    for p in pieces:
        out.extend(_wordsplit(p, hard_max) if len(p) > hard_max else [p])
    return out


def resegment(text: str, *, target: int = TARGET_CHARS, hard_max: int = HARD_MAX,
              min_tail: int = MIN_TAIL) -> list[str]:
    """Re-window normalized text to ~target-sized units at paragraph/sentence boundaries."""
    units: list[str] = []
    buf = ""
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > hard_max:
            if buf:
                units.append(buf)
                buf = ""
            units.extend(_hardsplit(para, target, hard_max))
            continue
        if buf and len(buf) + len(para) + 2 > hard_max:
            units.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
        if len(buf) >= target:
            units.append(buf)
            buf = ""
    if buf:
        # merge a short tail into the previous unit only if it stays under the cap
        if units and len(buf) < min_tail and len(units[-1]) + len(buf) + 2 <= hard_max:
            units[-1] = f"{units[-1]}\n\n{buf}"
        else:
            units.append(buf)
    return [u.strip() for u in units if u.strip()]


# --------------------------------------------------------------------------- #
# Boilerplate residue
# --------------------------------------------------------------------------- #

# NIST FIPS announcement / process items (start-anchored: chunk begins with "N. Title.")
_NIST_PROCESS = re.compile(
    r"^\s*\W?\s*(\d+\.\s*)?("
    r"explanation\.|applicability\.|implementations\.|export control\.|qualifications\.|"
    r"how to cite|maintenance agency|patents\.|name of standard|category of standard|"
    r"waiver procedure|implementation schedule|specifications\.|inquiries and comments|"
    r"effective date|approving authority"
    r")",
    re.IGNORECASE,
)
# IETF RFC front-matter markers (appear near the very start of the boilerplate block)
_RFC_FRONTMATTER = re.compile(
    r"(internet engineering task force|network working group|status of this memo|"
    r"copyright notice|request for comments:|category: (standards|informational|experimental))",
    re.IGNORECASE,
)


def is_boilerplate(text: str, position: str) -> bool:
    """True for low-value process/front-matter residue (RFC IETF header, NIST announcement/changelog).

    Content-based, not position-based: a chunk's ``position`` of "front" means different
    things per source (IETF boilerplate for an RFC, but a valuable chapter intro for an
    OS textbook), so we match on the actual boilerplate text instead.
    """
    head = text.lstrip("﻿").lstrip()
    return bool(_NIST_PROCESS.match(head)) or bool(_RFC_FRONTMATTER.search(head[:300]))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build_units(
    corpus: pl.DataFrame,
    *,
    carry: tuple[str, ...] = ("source_ids", "source_name", "topic", "version_pin", "license"),
    target: int = TARGET_CHARS,
    hard_max: int = HARD_MAX,
) -> tuple[pl.DataFrame, dict]:
    """Derive normalized analysis units from a provenance corpus. Returns (units, stats)."""
    rows: list[dict] = []
    dropped = {"boilerplate": 0}
    for r in corpus.iter_rows(named=True):
        position = r.get("chunk_position", "")
        if is_boilerplate(r["text"], position):
            dropped["boilerplate"] += 1
            continue
        cleaned = clean_text(r["text"])
        for i, unit_text in enumerate(resegment(cleaned, target=target, hard_max=hard_max)):
            row = {c: r[c] for c in carry}
            row.update({
                "unit_id": content_hash(unit_text),
                "parent_id": r["chunk_id"],
                "unit_position": f"{position}#{i}",
                "n_chars": len(unit_text),
                "text": unit_text,
            })
            rows.append(row)
    units = pl.DataFrame(rows, schema=_UNIT_SCHEMA)
    before = units.height
    units = units.unique(subset=["unit_id"], keep="first", maintain_order=True)
    stats = {
        "source_chunks": corpus.height,
        "units": units.height,
        "dropped_boilerplate": dropped["boilerplate"],
        "dropped_duplicate_units": before - units.height,
        "median_chars": int(units["n_chars"].median()) if units.height else 0,
        "max_chars": int(units["n_chars"].max()) if units.height else 0,
    }
    return units, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized analysis units from a corpus.")
    parser.add_argument("--corpus", default=str(ROOT / "data" / "substrate_corpus.parquet"))
    parser.add_argument("--out", default=str(ROOT / "data" / "substrate_units.parquet"))
    args = parser.parse_args()

    corpus = pl.read_parquet(args.corpus)
    units, stats = build_units(corpus)
    units.write_parquet(args.out)
    print(f"source chunks:        {stats['source_chunks']}")
    print(f"units produced:       {stats['units']}")
    print(f"dropped boilerplate:  {stats['dropped_boilerplate']}")
    print(f"dropped dup units:    {stats['dropped_duplicate_units']}")
    print(f"unit chars median/max: {stats['median_chars']} / {stats['max_chars']}")
    print("by source:")
    for row in units.group_by("source_name").agg(
        pl.len().alias("units"), pl.col("n_chars").median().alias("median")
    ).sort("source_name").iter_rows(named=True):
        print(f"   {row['source_name']:34s} {row['units']:4d} units  median {int(row['median'])}c")


if __name__ == "__main__":
    main()
