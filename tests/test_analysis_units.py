"""Tests for the unified analysis-units builder."""

from __future__ import annotations

import polars as pl

from entanglement.analysis_units import ANALYSIS_UNIT_COLUMNS, build_analysis_units


def _write(path, **cols):
    pl.DataFrame(cols).write_parquet(path)


def test_build_analysis_units_labels_and_normalizes(tmp_path):
    # distinct text per source (identical text would dedup away)
    _write(tmp_path / "attack_procedures.parquet",
           text=["Offensive procedure: dump credentials from process memory. " * 20],
           tactics=[["execution"]])
    _write(tmp_path / "d3fend_prose_tagged.parquet",
           text=["Defensive technique: analyze network traffic to detect anomalies. " * 20],
           tactic=["Detect"])
    _write(tmp_path / "substrate_units.parquet",
           text=["Substrate: a block cipher maps fixed-size blocks under a key. " * 20],
           topic=["crypto"])

    units = build_analysis_units(tmp_path, prune_framework_metadata=False)
    assert units.columns == ANALYSIS_UNIT_COLUMNS
    assert set(units["bucket"].to_list()) == {"offense", "defense", "dual"}
    by_bucket = {r["bucket"]: r for r in units.iter_rows(named=True)}
    assert by_bucket["offense"]["layer"] == "procedure"
    assert by_bucket["offense"]["topic"] == "execution"     # first of tactics list
    assert by_bucket["defense"]["topic"] == "Detect"        # scalar tactic
    assert by_bucket["dual"]["topic"] == "crypto"
    # unit_id is content hash; text normalized (markup-free, de-wrapped)
    from entanglement.units import content_hash
    assert by_bucket["dual"]["unit_id"] == content_hash(by_bucket["dual"]["text"])


def test_build_analysis_units_dedups_identical_text(tmp_path):
    same = "Identical normalized body shared across two buckets here, long enough to keep. " * 5
    _write(tmp_path / "attack_procedures.parquet", text=[same], tactics=[[]])
    _write(tmp_path / "substrate_units.parquet", text=[same], topic=["os_internals"])
    units = build_analysis_units(tmp_path, prune_framework_metadata=False)
    assert units.height == 1   # identical content collapses to one unit


def test_prune_framework_metadata_drops_procedure_and_prose(tmp_path):
    # one unit per source; with the prune ON, offense/procedure and defense/prose must vanish
    _write(tmp_path / "attack_procedures.parquet",
           text=["Procedure: APT used a tool to dump credentials. " * 10], tactics=[["execution"]])
    _write(tmp_path / "offensive_documents.parquet",
           text=["External offensive analysis of an intrusion campaign in depth. " * 10],
           tactics=[["command-and-control"]])
    _write(tmp_path / "d3fend_prose_tagged.parquet",
           text=["D3FEND definition: a defensive technique catalog entry. " * 10], tactic=["Detect"])
    _write(tmp_path / "defensive_documents.parquet",
           text=["External defensive hardening standard with mechanism detail. " * 10],
           tactics=[["Harden"]])
    _write(tmp_path / "substrate_units.parquet",
           text=["Substrate: a block cipher maps fixed-size blocks under a key. " * 10], topic=["crypto"])

    pruned = build_analysis_units(tmp_path)   # default prune_framework_metadata=True
    pairs = {(r["bucket"], r["layer"]) for r in pruned.iter_rows(named=True)}
    assert ("offense", "procedure") not in pairs
    assert ("defense", "prose") not in pairs
    # external + substrate layers survive, symmetric external-only design
    assert pairs == {("offense", "offense_external"), ("defense", "defense_external"),
                     ("dual", "substrate")}

    full = build_analysis_units(tmp_path, prune_framework_metadata=False)
    full_pairs = {(r["bucket"], r["layer"]) for r in full.iter_rows(named=True)}
    assert ("offense", "procedure") in full_pairs and ("defense", "prose") in full_pairs


def test_missing_sources_skipped(tmp_path):
    _write(tmp_path / "substrate_units.parquet",
           text=["A unit of substrate text long enough to survive resegmentation easily."],
           topic=["web"])
    units = build_analysis_units(tmp_path)   # other source files absent -> skipped, no error
    assert units.height >= 1 and set(units["bucket"].to_list()) == {"dual"}
