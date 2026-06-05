"""Tests for Stage B in-hand corpus assembly + provenance integrity."""

from __future__ import annotations

import polars as pl
import pytest

from entanglement.retrieve import CORPUS_COLUMNS, assemble_inhand


@pytest.fixture
def tiny_inputs():
    techniques = pl.DataFrame(
        {
            "tech_id": ["T1003", "T1595", "T9999"],
            "name": ["OS Credential Dumping", "Active Scanning", "No Desc"],
            "description": ["dump creds from lsass ...", "scan the target ...", ""],
        }
    )
    d3fend_prose = pl.DataFrame(
        {
            "subject_id": ["D3-NTA", "REF-X"],
            "label": ["Network Traffic Analysis", "Some Reference"],
            "kind": ["definition", "kb_abstract"],
            "text": ["analyze network traffic to detect ...", "abstract text ..."],
            "n_chars": [37, 17],
        }
    )
    partition = pl.DataFrame(
        {
            "topic_id": ["T1003", "T1595", "T9999", "D3-NTA"],
            "topic_name": ["OS Credential Dumping", "Active Scanning", "No Desc", "NTA"],
            "framework": ["ATTACK", "ATTACK", "ATTACK", "D3FEND"],
            "region": ["dual-use", "dual-use", "offensive", "defensive"],
            "subcap": [
                "credential_access",
                "reconnaissance",
                "impact",
                "detection",
            ],
        }
    )
    return techniques, d3fend_prose, partition


def test_assemble_inhand_schema_and_provenance(tiny_inputs):
    techniques, d3fend_prose, partition = tiny_inputs
    corpus = assemble_inhand(techniques, d3fend_prose, partition, "2026-06-03")

    # exact schema, in order
    assert corpus.columns == CORPUS_COLUMNS
    # empty-description technique (T9999) dropped; 2 ATT&CK + 1 D3FEND def = 3
    assert corpus.height == 3
    # provenance label method on every doc
    assert (corpus["label_method"] == "framework_provenance").all()
    # fetch date stamped
    assert (corpus["fetch_date"] == "2026-06-03").all()
    # n_chars matches text length
    assert (corpus["n_chars"] == corpus["text"].str.len_chars()).all()


def test_region_inherited_from_topic(tiny_inputs):
    techniques, d3fend_prose, partition = tiny_inputs
    corpus = assemble_inhand(techniques, d3fend_prose, partition, "2026-06-03")
    by_topic = dict(zip(corpus["topic_id"], corpus["region"]))
    assert by_topic["T1003"] == "dual-use"  # framework provenance carried through
    assert by_topic["D3-NTA"] == "defensive"


def test_doc_ids_unique_and_readable(tiny_inputs):
    techniques, d3fend_prose, partition = tiny_inputs
    corpus = assemble_inhand(techniques, d3fend_prose, partition, "2026-06-03")
    assert corpus["doc_id"].n_unique() == corpus.height
    assert "ATTACK_DESC:T1003" in corpus["doc_id"].to_list()


def test_kb_abstract_not_in_definitions_join(tiny_inputs):
    # kb_abstract rows must not leak into the corpus via the definition path
    techniques, d3fend_prose, partition = tiny_inputs
    corpus = assemble_inhand(techniques, d3fend_prose, partition, "2026-06-03")
    assert "REF-X" not in corpus["topic_id"].to_list()
