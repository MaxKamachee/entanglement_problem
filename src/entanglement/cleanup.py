"""Reproducible post-prune cleanup filters for the analysis-units corpus.

Applied (config-gated, same pattern as the framework-metadata prune) in
``analysis_units.main`` after the prune, before the parquet is written, so the cleaned corpus is
regenerable from source. Two bucket-specific passes, motivated by `reports/corpus_quality_audit.md`:

**Offense** — (a) drop `tiny` units (IOC / hash-dump fragments below a char floor); (b) MinHash
near-duplicate collapse at Jaccard ≥0.85 (the same threat report cited by many techniques, keep one
representative per cluster).

**Dual** — (a) drop units that are >50% non-prose (raw debug/traceroute dumps, base64 image blobs);
(b) for short fragments below a char floor, drop truncation artifacts / exercise prompts, keep
coherent prose.

All thresholds live in ``configs/corpus.yaml``. Pure functions take explicit params so tests stay
offline.
"""

from __future__ import annotations

import re

import polars as pl

from entanglement.corpus_audit import _union_find, minhash_signatures, near_duplicate_pairs

_WORD = re.compile(r"[A-Za-z]+")
_EXERCISE = re.compile(r"^\s*(exercise|problem|question|solution|proof|example)\b[\s.:0-9]",
                       re.IGNORECASE)
_ENDS_OK = re.compile(r"[.!?:)\]\"']\s*$")          # terminal punctuation = likely complete


def nonprose_ratio(text: str) -> float:
    """1 - (alphabetic-word chars / total chars). High for numeric/debug dumps."""
    n = len(text)
    if not n:
        return 1.0
    word_chars = sum(len(w) for w in _WORD.findall(text))
    return 1.0 - word_chars / n


def longtoken_ratio(text: str) -> float:
    """Fraction of chars sitting in whitespace tokens longer than 30 chars (base64 / blobs)."""
    n = len(text)
    if not n:
        return 0.0
    return sum(len(t) for t in text.split() if len(t) > 30) / n


def is_nonprose(text: str, max_nonprose: float, max_longtoken: float) -> bool:
    """A document dominated by non-prose content (debug dump or base64 blob)."""
    return nonprose_ratio(text) > max_nonprose or longtoken_ratio(text) > max_longtoken


def is_bad_short_fragment(text: str) -> bool:
    """For short docs: True if a truncation artifact or an exercise prompt (drop), False if coherent."""
    t = text.strip()
    if _EXERCISE.match(t):
        return True
    return not _ENDS_OK.search(t)        # no terminal punctuation -> likely mid-sentence cut


def _near_dup_drop_ids(sub: pl.DataFrame, jaccard: float) -> set[str]:
    """unit_ids to drop in a near-dup collapse: keep the first member of each cluster."""
    if sub.height < 2:
        return set()
    sigs = minhash_signatures(sub["text"].to_list())
    pairs = near_duplicate_pairs(sigs, threshold=jaccard)
    comp = _union_find(sub.height, pairs)
    ids = sub["unit_id"].to_list()
    keep_first: dict[int, int] = {}
    drop: set[str] = set()
    for i in range(sub.height):
        root = comp[i]
        if root in keep_first:
            drop.add(ids[i])
        else:
            keep_first[root] = i
    return drop


def apply_cleanup(
    units: pl.DataFrame,
    *,
    offense_tiny_chars: int = 250,
    offense_neardup_jaccard: float = 0.85,
    dual_max_nonprose_ratio: float = 0.5,
    dual_max_longtoken_ratio: float = 0.3,
    dual_short_fragment_chars: int = 500,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (cleaned_units, drop_log[unit_id, bucket, reason]). Deterministic."""
    drop: dict[str, str] = {}        # unit_id -> reason (first reason wins)

    def mark(uid, reason):
        drop.setdefault(uid, reason)

    for r in units.iter_rows(named=True):
        uid, bucket, text, nchars = r["unit_id"], r["bucket"], r["text"], r["n_chars"]
        if bucket == "offense" and nchars < offense_tiny_chars:
            mark(uid, "offense_tiny")
        elif bucket == "dual":
            if is_nonprose(text, dual_max_nonprose_ratio, dual_max_longtoken_ratio):
                mark(uid, "dual_nonprose")
            elif nchars < dual_short_fragment_chars and is_bad_short_fragment(text):
                mark(uid, "dual_bad_short_fragment")

    # offense near-dup collapse on what survives the tiny drop
    offense_alive = units.filter(
        (pl.col("bucket") == "offense") & ~pl.col("unit_id").is_in(list(drop))
    )
    for uid in _near_dup_drop_ids(offense_alive, offense_neardup_jaccard):
        mark(uid, "offense_neardup")

    cleaned = units.filter(~pl.col("unit_id").is_in(list(drop)))
    log = pl.DataFrame(
        {"unit_id": list(drop), "reason": list(drop.values())},
        schema={"unit_id": pl.String, "reason": pl.String},
    ).join(units.select("unit_id", "bucket"), on="unit_id", how="left").select(
        "unit_id", "bucket", "reason"
    )
    return cleaned, log
