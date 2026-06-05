"""Tests for defensive prose tagging, bucket review, and VENDOR classification."""

from __future__ import annotations

import polars as pl
import pytest

from entanglement.corpus_defensive import (
    DEFENSE_DOC_COLUMNS,
    DEFENSE_PROSE_COLUMNS,
    bucket_summary,
    sample_other,
    select_defensive_urls,
    tag_defensive_documents,
    tag_prose_with_tactics,
)
from entanglement.d3fend import classify_source

VENDORS = ["microsoft.com", "crowdstrike.com", "unit42"]


# --- classify_source: VENDOR added, five existing buckets unchanged ---

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://patents.google.com/patent/US123", "PATENT"),
        ("https://car.mitre.org/analytics/CAR-2020", "MITRE_CAR"),
        ("https://csrc.nist.gov/pubs/sp/800/53", "NIST"),
        ("https://www.rfc-editor.org/rfc/rfc8446", "RFC"),
        ("https://arxiv.org/abs/2001.00001", "ACADEMIC"),
        ("https://doi.org/10.1145/12345", "ACADEMIC"),
        ("https://www.microsoft.com/security/blog/x", "VENDOR"),
        ("https://unit42.paloaltonetworks.com/report", "VENDOR"),
        ("https://some-random-blog.example/post", "OTHER"),
        # mis-bucketed patents now caught by PATENT
        ("https://patentimages.storage.googleapis.com/23/US11637861.pdf", "PATENT"),
        ("https://www.patentguru.com/en/search/US123", "PATENT"),
        # ATT&CK crosswalk content excluded via its own bucket
        ("https://attack.mitre.org/mitigations/M0800", "ATTACK_XREF"),
        # CWE is NOT attack.mitre.org and stays OTHER (kept under curation)
        ("https://cwe.mitre.org/data/definitions/476.html", "OTHER"),
    ],
)
def test_classify_source_buckets(url, expected):
    assert classify_source(url, vendor_hosts=VENDORS) == expected


def test_vendor_does_not_steal_academic_or_standards():
    # A vendor substring must not override an earlier-matched bucket.
    assert classify_source("https://csrc.nist.gov/x", vendor_hosts=["nist"]) == "NIST"


# --- prose tagging ---

@pytest.fixture
def prose_inputs():
    prose = pl.DataFrame(
        {
            "subject_id": ["D3-NTA", "D3-ALLM", "REF-123"],
            "label": ["Network Traffic Analysis", "Active Logical Link Mapping", "Some Ref"],
            "kind": ["definition", "definition", "kb_abstract"],
            "text": ["analyze traffic ...", "map links ...", "abstract ..."],
            "n_chars": [19, 13, 12],
        }
    )
    tactic_map = pl.DataFrame(
        {"d3fend_id": ["D3-NTA", "D3-ALLM"], "tactic": ["Detect", "Model"]}
    )
    hierarchy = pl.DataFrame(
        {
            "d3fend_id": ["D3-NTA", "D3-ALLM"],
            "parent_id": ["D3-NTA", "D3-LLM"],
        }
    )
    return prose, tactic_map, hierarchy


def test_tag_prose_schema_and_join(prose_inputs):
    tagged = tag_prose_with_tactics(*prose_inputs)
    assert tagged.columns == DEFENSE_PROSE_COLUMNS
    by_id = {r["tech_id"]: r for r in tagged.iter_rows(named=True)}
    assert by_id["D3-NTA"]["tactic"] == "Detect"
    assert by_id["D3-ALLM"]["parent_id"] == "D3-LLM"
    # kb_abstract row has no clean technique mapping -> null tactic/parent
    assert by_id["REF-123"]["tactic"] is None
    assert by_id["REF-123"]["parent_id"] is None


# --- bucket review + selection ---

@pytest.fixture
def refs():
    return pl.DataFrame(
        {
            "url": ["u1", "u2", "u3", "u4", "u5"],
            "title": ["t1", "t2", "t3", "t4", "t5"],
            "concept_label": ["c1", "c2", "c3", "c4", "c5"],
            "bucket": ["NIST", "PATENT", "OTHER", "VENDOR", "MITRE_CAR"],
        }
    )


def test_select_defensive_urls_keeps_config_set(refs):
    kept = select_defensive_urls(
        refs, keep_buckets=frozenset({"NIST", "VENDOR"}), curated_other=False
    )
    assert set(kept["bucket"].to_list()) == {"NIST", "VENDOR"}
    assert "PATENT" not in kept["bucket"].to_list()
    assert "MITRE_CAR" not in kept["bucket"].to_list()


def test_select_defensive_urls_curated_other():
    refs = pl.DataFrame({
        "url": [
            "https://csrc.nist.gov/x",                          # named keep
            "https://trustedcomputinggroup.org/tpm-spec",       # substantive OTHER -> kept
            "https://en.wikipedia.org/wiki/Motion_detector",    # denylisted OTHER -> dropped
            "https://patents.google.com/patent/US1",            # PATENT -> excluded
            "https://attack.mitre.org/mitigations/M0800",       # ATTACK_XREF -> excluded
        ],
        "title": ["a", "b", "c", "d", "e"],
        "concept_label": ["1", "2", "3", "4", "5"],
        "bucket": ["NIST", "OTHER", "OTHER", "PATENT", "ATTACK_XREF"],
    })
    kept = set(
        select_defensive_urls(
            refs, keep_buckets=frozenset({"NIST"}),
            curated_other=True, other_drop_hosts=["wikipedia.org"],
        )["url"].to_list()
    )
    assert "https://csrc.nist.gov/x" in kept
    assert "https://trustedcomputinggroup.org/tpm-spec" in kept     # OTHER substantive kept
    assert "https://en.wikipedia.org/wiki/Motion_detector" not in kept  # denylisted
    assert "https://patents.google.com/patent/US1" not in kept      # PATENT excluded
    assert "https://attack.mitre.org/mitigations/M0800" not in kept  # ATTACK_XREF excluded


def test_tag_defensive_documents_aggregates_concept_tactic_bucket():
    documents = pl.DataFrame({
        "content_hash": ["h1"], "url": ["https://a/x"], "raw_url": ["https://a/x"],
        "extractor": ["trafilatura"], "n_chars": [5000], "text": ["body"],
    })
    provenance = pl.DataFrame({
        "url": ["https://a/x", "https://b/y"], "content_hash": ["h1", "h1"],
    })
    refs = pl.DataFrame({
        "url": ["https://a/x", "https://b/y"],
        "concept_id": ["D3-NTA", "D3-FA"],
        "bucket": ["NIST", "OTHER"],
    })
    prose_tagged = pl.DataFrame({
        "tech_id": ["D3-NTA", "D3-FA"],
        "tactic": ["Detect", "Harden"],
        "parent_id": ["D3-NTA", "D3-DC"],
    })
    row = tag_defensive_documents(documents, provenance, refs, prose_tagged).row(0, named=True)
    assert sorted(row["d3fend_ids"]) == ["D3-FA", "D3-NTA"]
    assert sorted(row["tactics"]) == ["Detect", "Harden"]
    assert sorted(row["buckets"]) == ["NIST", "OTHER"]


def test_defense_doc_columns_constant():
    assert DEFENSE_DOC_COLUMNS[0] == "content_hash" and "buckets" in DEFENSE_DOC_COLUMNS


def test_sample_other_only_other(refs):
    s = sample_other(refs, n=10)
    assert s.height == 1
    assert s["url"].to_list() == ["u3"]


def test_bucket_summary_counts(refs):
    counts = dict(zip(*bucket_summary(refs)[["bucket", "len"]].to_dict(as_series=False).values()))
    assert counts["NIST"] == 1 and counts["OTHER"] == 1
