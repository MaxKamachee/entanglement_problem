"""Corpus integrity invariants for the frozen analysis-units artifact.

`validate_corpus` is pure (takes a DataFrame, returns a list of violation strings; empty = clean).
The freeze step calls it and refuses to stamp v1 on any violation, so the frozen corpus can't silently
drift (a missing bucket, an empty doc, a duplicate id, an out-of-range length, a content-hash mismatch).
"""

from __future__ import annotations

import polars as pl

from entanglement.units import content_hash

EXPECTED_BUCKETS = {"offense", "dual", "defense"}
EXPECTED_LAYERS = {"procedure", "offense_external", "prose", "defense_external",
                   "external_supplement", "substrate"}
HARD_MAX_CHARS = 4100          # resegment hard_max (4000) + small tolerance
SCHEMA = ["unit_id", "bucket", "layer", "topic", "n_chars", "text"]


def validate_corpus(df: pl.DataFrame) -> list[str]:
    v: list[str] = []
    if df.columns != SCHEMA:
        v.append(f"schema mismatch: {df.columns} != {SCHEMA}")
        return v                       # downstream checks assume the schema
    buckets = set(df["bucket"].unique().to_list())
    if buckets != EXPECTED_BUCKETS:
        v.append(f"buckets {buckets} != {EXPECTED_BUCKETS}")
    bad_layers = set(df["layer"].unique().to_list()) - EXPECTED_LAYERS
    if bad_layers:
        v.append(f"unexpected layers: {bad_layers}")
    n_empty = df.filter((pl.col("text").is_null()) | (pl.col("text").str.len_chars() == 0)).height
    if n_empty:
        v.append(f"{n_empty} units with null/empty text")
    n_dup = df.height - df["unit_id"].n_unique()
    if n_dup:
        v.append(f"{n_dup} duplicate unit_id values")
    n_oob = df.filter((pl.col("n_chars") < 1) | (pl.col("n_chars") > HARD_MAX_CHARS)).height
    if n_oob:
        v.append(f"{n_oob} units with n_chars outside [1, {HARD_MAX_CHARS}]")
    n_len_mismatch = df.filter(pl.col("n_chars") != pl.col("text").str.len_chars()).height
    if n_len_mismatch:
        v.append(f"{n_len_mismatch} units where n_chars != len(text)")
    # content-hash integrity: unit_id must equal content_hash(text)
    bad_hash = sum(1 for r in df.iter_rows(named=True) if r["unit_id"] != content_hash(r["text"]))
    if bad_hash:
        v.append(f"{bad_hash} units where unit_id != content_hash(text)")
    return v
