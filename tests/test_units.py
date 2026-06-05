"""Tests for the normalized analysis-units layer."""

from __future__ import annotations

import polars as pl

from entanglement.substrate import _SCHEMA
from entanglement.units import (
    HARD_MAX,
    UNIT_COLUMNS,
    build_units,
    clean_text,
    content_hash,
    dewrap,
    is_boilerplate,
    resegment,
    strip_markup,
)


# --- format stripping ---

def test_strip_markup_links_html_fences():
    raw = "See [the spec](https://x/y) and `code` and <b>bold</b>.\n```js\nvar x=1;\n```"
    out = strip_markup(raw)
    assert "the spec" in out and "https://x/y" not in out   # link text kept, target dropped
    assert "<b>" not in out and "bold" in out               # html stripped, text kept
    assert "var x=1;" in out and "```" not in out           # code body kept, fence gone


def test_dewrap_joins_wrapped_lines():
    wrapped = "This is a hard-wrapped\nRFC paragraph that\nspans lines.\n\nSecond paragraph."
    out = dewrap(wrapped)
    assert out == "This is a hard-wrapped RFC paragraph that spans lines.\n\nSecond paragraph."


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc") == "a b c"


def test_clean_text_repairs_ligatures():
    assert clean_text("the cipher is deﬁned and veriﬁed eﬃciently") == \
        "the cipher is defined and verified efficiently"


# --- re-segmentation ---

def test_resegment_short_text_single_unit():
    assert len(resegment("one short paragraph.", target=3000, hard_max=4000)) == 1


def test_resegment_splits_oversized_and_respects_max():
    big = "\n\n".join(f"Paragraph number {i}. " * 40 for i in range(20))  # ~ tens of k chars
    units = resegment(big, target=3000, hard_max=4000)
    assert len(units) > 1
    assert all(len(u) <= HARD_MAX for u in units)


def test_resegment_hardsplits_single_huge_paragraph():
    huge = "This is a sentence. " * 600   # one ~12k-char paragraph, no blank lines
    units = resegment(huge, target=3000, hard_max=4000)
    assert len(units) >= 3
    assert all(len(u) <= HARD_MAX for u in units)


def test_resegment_caps_punctuation_free_blob():
    blob = "GRAMMAR " + "token=value;rule|alt;" * 2000   # no sentence delimiters at all
    units = resegment(blob, target=3000, hard_max=4000)
    assert all(len(u) <= HARD_MAX for u in units)         # word-split fallback guarantees the cap


# --- boilerplate ---

def test_is_boilerplate():
    # content-based, not position-based
    assert is_boilerplate("15. How to Cite This Publication. NIST has...", "15") is True
    assert is_boilerplate("3. Explanation. The AES specifies...", "3") is True
    assert is_boilerplate("﻿\n\nInternet Engineering Task Force (IETF)\nRequest for...", "front") is True
    # an OS-textbook chapter intro is at position "front" but must be KEPT
    assert is_boilerplate("Concurrency: An Introduction\nThus far we have seen the "
                          "development of...", "front") is False
    assert is_boilerplate("The same-origin policy restricts...", "2.1") is False


# --- build_units ---

def _chunk(cid, text, position, topic="crypto"):
    return {
        "chunk_id": cid, "source_ids": ["s1"], "source_name": "SRC", "topic": topic,
        "version_pin": "v1", "text": text, "n_chars": len(text), "chunk_position": position,
        "extractor": "rfc_text", "extraction_warnings": [], "license": "L",
    }


def test_build_units_schema_parent_link_and_dropping():
    big_body = "\n\n".join(f"Paragraph {i}. " * 50 for i in range(15))
    corpus = pl.DataFrame([
        _chunk("h_big", big_body, "5"),                       # -> multiple units
        _chunk("h_small", "A short normal section body that stays one unit.", "6"),
        _chunk("h_front", "Internet Engineering Task Force (IETF)\nRequest for Comments: 9999\n"
                          "Category: Standards Track\nStatus of This Memo", "front"),  # dropped
    ], schema=_SCHEMA)

    units, stats = build_units(corpus, target=3000, hard_max=4000)
    assert units.columns == UNIT_COLUMNS
    assert stats["dropped_boilerplate"] == 1
    # big chunk produced >1 unit, all linked to parent
    big_units = units.filter(pl.col("parent_id") == "h_big")
    assert big_units.height > 1
    assert all(p.startswith("5#") for p in big_units["unit_position"].to_list())
    # unit_id is content hash of normalized text
    row = units.row(0, named=True)
    assert row["unit_id"] == content_hash(row["text"])
    # carried provenance fields present
    assert row["source_name"] == "SRC" and row["version_pin"] == "v1" and row["topic"] == "crypto"
    # no front-matter unit survived
    assert "h_front" not in units["parent_id"].to_list()


def test_build_units_dedups_identical_units():
    corpus = pl.DataFrame([
        _chunk("h1", "Identical body text repeated across two chunks here.", "1"),
        _chunk("h2", "Identical body text repeated across two chunks here.", "2"),
    ], schema=_SCHEMA)
    units, stats = build_units(corpus)
    assert units.height == 1
    assert stats["dropped_duplicate_units"] == 1
