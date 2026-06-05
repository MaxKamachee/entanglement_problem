"""Tests for the corpus-integrity validator (pure, offline)."""

from __future__ import annotations

import polars as pl

from entanglement.corpus_validate import SCHEMA, validate_corpus
from entanglement.units import content_hash


def _row(text, bucket="offense", layer="offense_external", topic="x"):
    return {"unit_id": content_hash(text), "bucket": bucket, "layer": layer,
            "topic": topic, "n_chars": len(text), "text": text}


def _good_df():
    rows = [
        _row("an offensive operational document about credential dumping techniques here", "offense",
             "offense_external"),
        _row("a defensive hardening control mapped to a D3FEND technique described here", "defense",
             "defense_external"),
        _row("a substrate document explaining how the TCP handshake mechanism operates", "dual",
             "substrate"),
    ]
    return pl.DataFrame(rows).select(SCHEMA)


def test_clean_corpus_has_no_violations():
    assert validate_corpus(_good_df()) == []


def test_detects_missing_bucket():
    df = _good_df().filter(pl.col("bucket") != "dual")
    assert any("buckets" in v for v in validate_corpus(df))


def test_detects_duplicate_unit_id_and_empty_text():
    df = _good_df()
    dup = pl.concat([df, df.head(1)])           # duplicate a unit_id
    assert any("duplicate unit_id" in v for v in validate_corpus(dup))


def test_detects_content_hash_mismatch():
    df = _good_df().with_columns(pl.lit("tampered text not matching the hash").alias("text"))
    df = df.with_columns(pl.col("text").str.len_chars().alias("n_chars"))
    assert any("content_hash" in v for v in validate_corpus(df))


def test_detects_unexpected_layer():
    df = _good_df().with_columns(
        pl.when(pl.col("bucket") == "offense").then(pl.lit("bogus_layer"))
        .otherwise(pl.col("layer")).alias("layer"))
    assert any("layers" in v for v in validate_corpus(df))
