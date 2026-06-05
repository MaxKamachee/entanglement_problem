"""Tests for the substrate corpus build — RFC handler + generic machinery. Offline."""

from __future__ import annotations

import polars as pl
import pytest

from entanglement.substrate import (
    SUBSTRATE_COLUMNS,
    _dedup_within,
    _is_toc,
    alpha_ratio,
    append_to_corpus,
    build_github_markdown_source,
    build_html_book_chapters,
    build_pdf_chapter_collection,
    build_pdf_chapter_extract,
    build_pdf_single_source,
    build_rfc_source,
    clean_markdown,
    content_hash,
    density_decision,
    discover_chapter_stems,
    split_doc_sections,
    split_rfc_sections,
    strip_rfc_artifacts,
    write_build_report,
)

# Synthetic mini-RFC: running headers, a [Page N] footer, a ToC, two real sections.
SYNTH_RFC = """RFC 9999  Example Protocol  June 2026

Table of Contents

   1. Introduction ....................... 1
   2. Protocol ........................... 2

1.  Introduction

   This document describes an example protocol used to validate the
   substrate RFC extraction handler. It contains enough ordinary prose
   to clear the minimum chunk size and the alphabetic density threshold.

RFC 9999  Example Protocol  June 2026                            [Page 1]

2.  Protocol

   The protocol exchanges messages between peers. This section also has
   plenty of normal alphabetic prose so the density check keeps it clean
   without raising any warning flags whatsoever in the build report.
"""


def _entry(policy="flag_only"):
    return {
        "id": "rfcs_test",
        "license": "TEST-LICENSE",
        "density_policy": policy,
        "version_pin": "RFCs: 9999",
        "rfcs": [{"number": 9999, "topic": "networking"}],
    }


@pytest.fixture
def built():
    return build_rfc_source(_entry(), fetcher=lambda _n: SYNTH_RFC)


# --- artifact stripping + section splitting ---

def test_strip_removes_footer_and_running_header():
    cleaned = strip_rfc_artifacts(SYNTH_RFC)
    assert "[Page 1]" not in cleaned
    assert "RFC 9999  Example Protocol" not in cleaned


def test_split_finds_only_col0_sections():
    cleaned = strip_rfc_artifacts(SYNTH_RFC)
    positions = [p for p, _ in split_rfc_sections(cleaned)]
    # indented ToC entries do NOT split; the two col-0 sections do
    assert "1" in positions and "2" in positions


def test_toc_detection():
    assert _is_toc("1. Intro ........ 1\n2. Body ........ 2") is True
    assert _is_toc("This is ordinary prose with no dot leaders at all.") is False


# --- full build ---

def test_build_schema_and_chunks(built):
    df, _ = built
    assert df.columns == SUBSTRATE_COLUMNS
    assert df.height == 2                       # ToC + front dropped, 2 real sections kept
    assert set(df["chunk_position"].to_list()) == {"1", "2"}


def test_build_tags_and_chunk_id_stable(built):
    df, _ = built
    row = df.filter(pl.col("chunk_position") == "1").row(0, named=True)
    assert row["topic"] == "networking"
    assert row["version_pin"] == "RFC9999"
    assert row["license"] == "TEST-LICENSE"
    assert row["source_ids"] == ["rfcs_test"]
    assert row["extractor"] == "rfc_text"
    assert row["chunk_id"] == content_hash(row["text"])   # chunk_id IS the content hash


def test_build_report_required_fields(built):
    _, report = built
    for f in ("source_id", "source_version_pin", "chunks_produced", "chunks_dropped",
              "chunks_dropped_reasons", "chars_total", "quality_flags", "license_note"):
        assert f in report, f
    assert report["chunks_dropped_reasons"].get("table_of_contents", 0) >= 1


def test_topic_validation_rejects_bad_topic():
    bad = _entry()
    bad["rfcs"] = [{"number": 9999, "topic": "not_a_topic"}]
    with pytest.raises(ValueError):
        build_rfc_source(bad, fetcher=lambda _n: SYNTH_RFC)


# --- density policy ---

@pytest.mark.parametrize("policy,expect_drop", [("standard", True), ("flag_only", False)])
def test_density_policy(policy, expect_drop):
    symbol_dense = "{[()]}=;:<>0123456789 ...... 42"   # very low alpha ratio
    drop, warnings = density_decision(symbol_dense, policy)
    assert drop is expect_drop
    assert warnings and warnings[0].startswith("low_text_density")


def test_alpha_ratio_extremes():
    assert alpha_ratio("abcdef") == 1.0
    assert alpha_ratio("123456") == 0.0
    assert alpha_ratio("") == 0.0


# --- dedup + corpus assembly ---

def test_dedup_within_drops_identical():
    df = pl.DataFrame(
        {"chunk_id": ["h1", "h1", "h2"]},
        schema={"chunk_id": pl.String},
    )
    out, dropped = _dedup_within(df.with_columns(pl.lit(0).alias("x")).select("chunk_id"))
    assert out.height == 2 and dropped == 1


def test_append_cross_source_merges_source_ids(tmp_path):
    path = tmp_path / "substrate.parquet"

    def row(cid, sids, text="body text long enough to be a chunk"):
        return {
            "chunk_id": cid, "source_ids": sids, "source_name": "X", "topic": "crypto",
            "version_pin": "v", "text": text, "n_chars": len(text), "chunk_position": "1",
            "extractor": "rfc_text", "extraction_warnings": [], "license": "L",
        }

    from entanglement.substrate import _SCHEMA
    a = pl.DataFrame([row("h1", ["src_a"])], schema=_SCHEMA)
    assert append_to_corpus(a, path) == 0                       # first write
    b = pl.DataFrame([row("h1", ["src_b"]), row("h2", ["src_b"])], schema=_SCHEMA)
    merged = append_to_corpus(b, path)                          # h1 dup-merged, h2 new
    assert merged == 1
    out = pl.read_parquet(path)
    assert out.height == 2
    h1 = out.filter(pl.col("chunk_id") == "h1").row(0, named=True)
    assert sorted(h1["source_ids"]) == ["src_a", "src_b"]       # both sources tagged


# --- pdf_single handler ---

SYNTH_PDF = """Advanced Example Standard

1. Introduction

   This standard specifies an example transform used to validate the
   pdf_single substrate handler with enough ordinary prose to clear the
   minimum chunk size and the density threshold comfortably.

2

2. Specification

   The transform maps input blocks to output blocks. This section also
   carries sufficient alphabetic prose for the density check to keep it
   clean and unflagged in the resulting build report output.

Appendix A. Test Vectors

   The following describe worked examples in clear descriptive prose so
   that the appendix is retained as a normal chunk rather than dropped
   for being too short or too sparse to be useful substrate content.
"""


def _pdf_entry():
    return {
        "id": "fips_test",
        "type": "pdf_single",
        "source_name": "FIPS TEST",
        "license": "PUBLIC-DOMAIN",
        "density_policy": "flag_only",
        "topic": "crypto",
    }


def test_pdf_single_build_and_pagenum_stripping():
    df, report = build_pdf_single_source(
        _pdf_entry(), extracted_text=SYNTH_PDF, version_pin="sha256:deadbeef"
    )
    assert df.columns == SUBSTRATE_COLUMNS
    positions = set(df["chunk_position"].to_list())
    assert {"1", "2", "Appendix A"} <= positions  # numbered sections + appendix detected
    row = df.filter(pl.col("chunk_position") == "1").row(0, named=True)
    assert row["extractor"] == "pdf_single"
    assert row["version_pin"] == "sha256:deadbeef"
    assert row["topic"] == "crypto" and row["license"] == "PUBLIC-DOMAIN"
    # the bare "2" page-number line must not have split a section or leaked in
    assert "\n2\n" not in "\n".join(df["text"].to_list())


def test_split_doc_sections_strips_pagenumbers():
    secs = dict(split_doc_sections("1. A\n   body\n42\n2. B\n   more body"))
    assert "42" not in secs.get("1", "")


# --- pdf_chapter_collection handler ---

def test_pdf_chapter_collection_per_chapter_provenance():
    entry = {
        "id": "ostep_test", "type": "pdf_chapter_collection",
        "source_name": "OSTEP", "license": "POINTER-ONLY", "topic": "os_internals",
        "density_policy": "flag_only", "version_pin": "per-chapter sha",
    }
    chap_text = ("26.1 Why Use Threads?\n\nThreads let a program run multiple points of "
                 "execution sharing one address space, which is central to OS concurrency "
                 "mechanisms and the bugs that arise from them.\n\n"
                 "26.2 An Example\n\nConsider two threads incrementing a shared counter "
                 "without synchronization, producing a classic data race described at length.")
    df, report = build_pdf_chapter_collection(entry, chapters=[
        ("cpu-intro", chap_text, "cpu-intro; sha256:aaa"),
        ("threads-intro", chap_text.replace("26", "27"), "threads-intro; sha256:bbb"),
    ])
    assert df.columns == SUBSTRATE_COLUMNS
    assert df["extractor"].unique().to_list() == ["pdf_chapter_collection"]
    # per-chapter source_name + version_pin
    names = set(df["source_name"].to_list())
    assert names == {"OSTEP: cpu-intro", "OSTEP: threads-intro"}
    cpu = df.filter(pl.col("source_name") == "OSTEP: cpu-intro").row(0, named=True)
    assert cpu["version_pin"] == "cpu-intro; sha256:aaa" and cpu["topic"] == "os_internals"


# --- html_book_chapters handler ---

def test_discover_chapter_stems_denylist_and_relative_only():
    html = ('<a href="host-discovery.html">x</a>'
            '<a href="scan-methods-udp-scan.html">x</a>'
            '<a href="install-linux.html">x</a>'           # denied: install
            '<a href="zenmap-results.html">x</a>'          # denied: zenmap
            '<a href="https://nmap.org/book/man.html">x</a>'  # absolute -> excluded by regex
            '<a href="host-discovery.html">dup</a>')        # dedup
    stems = discover_chapter_stems(html, deny_prefixes=["install", "zenmap"])
    assert stems == ["host-discovery", "scan-methods-udp-scan"]


def test_build_html_book_chapters_per_page():
    entry = {
        "id": "nmap_test", "type": "html_book_chapters", "source_name": "Nmap book",
        "license": "POINTER-ONLY", "topic": "recon", "density_policy": "flag_only",
        "version_pin": "per-page sha",
    }
    body = ("Host discovery determines which targets on a network are online before "
            "port scanning, using ARP, ICMP, TCP and UDP probes that both attackers "
            "and defenders rely on to map a network's live hosts efficiently.")
    df, report = build_html_book_chapters(entry, pages=[
        ("host-discovery", body, "https://nmap.org/book/host-discovery.html; sha256:aaa"),
    ])
    assert df.columns == SUBSTRATE_COLUMNS and df.height == 1
    row = df.row(0, named=True)
    assert row["extractor"] == "html_book_chapters"
    assert row["source_name"] == "Nmap book: host-discovery"
    assert row["topic"] == "recon" and row["chunk_position"] == "host-discovery"


# --- pdf_chapter_extract handler ---

def test_pdf_chapter_extract_labels_and_pins():
    entry = {
        "id": "intel_test", "type": "pdf_chapter_extract", "source_name": "Intel SDM Vol 3",
        "license": "POINTER-ONLY", "topic": "architecture", "density_policy": "flag_only",
        "version_pin": "rev 999 + sha", "min_chunk_chars": 250,
    }
    ch = ("2.1 System Flags and Fields in the EFLAGS Register\n\nThe system flags and the IOPL "
          "field of the EFLAGS register control task and mode switching, interrupt handling, "
          "instruction tracing, and access rights, which are central to the protection "
          "mechanisms of the processor. These flags are described in detail across the system "
          "programming model and are foundational to both privilege-escalation exploitation and "
          "to runtime defenses such as control-flow integrity and supervisor-mode protections.")
    df, report = build_pdf_chapter_extract(entry, ranges=[
        ("ch2-protection", ch, "ch2-protection; pages 50-90; sha256:abc"),
    ])
    assert df.columns == SUBSTRATE_COLUMNS
    row = df.row(0, named=True)
    assert row["extractor"] == "pdf_chapter_extract"
    assert row["source_name"] == "Intel SDM Vol 3: ch2-protection"
    assert row["version_pin"] == "ch2-protection; pages 50-90; sha256:abc"
    assert row["topic"] == "architecture"


# --- github_markdown handler ---

def test_clean_markdown_strips_frontmatter_and_macros():
    raw = ("---\ntitle: Same-origin policy\nslug: Web/Security/Same-origin_policy\n---\n"
           "The {{Glossary(\"same-origin policy\")}} restricts how a document loaded "
           "from one {{Glossary(\"origin\")}} can interact with another.")
    cleaned = clean_markdown(raw)
    assert "title: Same-origin" not in cleaned       # frontmatter gone
    assert "{{" not in cleaned and "Glossary" not in cleaned  # macros gone
    assert cleaned.startswith("The")


def test_build_github_markdown_per_file(tmp_path):
    (tmp_path / "same-origin").mkdir()
    long_body = "The same-origin policy is a critical security mechanism. " * 6
    (tmp_path / "same-origin" / "index.md").write_text(
        f"---\ntitle: X\n---\n{long_body}", encoding="utf-8"
    )
    (tmp_path / "tiny.md").write_text("---\ntitle: Y\n---\nshort", encoding="utf-8")  # dropped: too_short

    entry = {
        "id": "mdn_test", "type": "github_markdown", "source_name": "MDN TEST",
        "license": "CC-BY-SA 2.5", "topic": "web", "density_policy": "standard",
    }
    df, report = build_github_markdown_source(entry, walk_dir=tmp_path, version_pin="git repo@abc123")
    assert df.columns == SUBSTRATE_COLUMNS
    assert df.height == 1                              # one .md kept, tiny dropped
    row = df.row(0, named=True)
    assert row["extractor"] == "github_markdown"
    assert row["chunk_position"] == "same-origin/index.md"   # path relative to walk_dir
    assert row["topic"] == "web" and row["version_pin"] == "git repo@abc123"
    assert report["chunks_dropped_reasons"].get("too_short") == 1


def test_write_build_report_roundtrip(tmp_path, built):
    _, report = built
    p = write_build_report(report, build_dir=tmp_path)
    assert p.exists()
    import yaml
    loaded = yaml.safe_load(p.read_text())
    assert loaded["source_id"] == "rfcs_test"
    assert "build_timestamp" in loaded
