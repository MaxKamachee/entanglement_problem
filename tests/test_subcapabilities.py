"""Tests for Stage A: subcapability assignment + taxonomy invariants."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from entanglement.subcapabilities import (
    assign_subcap,
    build_stage_a,
    defensive_tactic_map,
    load_taxonomy,
    offensive_tactic_map,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tax() -> dict:
    return load_taxonomy(ROOT / "configs" / "subcapabilities.yaml")


def test_offensive_map_partitions_all_14_tactics(tax: dict) -> None:
    tac_map = offensive_tactic_map(tax)
    assert set(tac_map) == set(tax["attack_tactic_priority"])
    assert len(tax["attack_tactic_priority"]) == 14


def test_every_subcap_maps_at_least_one_tactic(tax: dict) -> None:
    for subcap, body in tax["offensive"].items():
        assert body["attack_tactics"], f"{subcap} maps no tactics"
    for subcap, body in tax["defensive"].items():
        assert body["d3fend_tactics"], f"{subcap} maps no tactics"


def test_assign_earliest_tactic_wins(tax: dict) -> None:
    tac_map = offensive_tactic_map(tax)
    pri = tax["attack_tactic_priority"]
    # persistence (idx4) beats defense-evasion (idx6)
    assert assign_subcap(["defense-evasion", "persistence"], tac_map, pri) == (
        "persistence_escalation"
    )
    # single tactic
    assert assign_subcap(["impact"], tac_map, pri) == "impact"
    # no usable tactic -> None (orphan)
    assert assign_subcap([], tac_map, pri) is None
    assert assign_subcap(["not-a-tactic"], tac_map, pri) is None


def test_defensive_map_covers_seven_d3fend_tactics(tax: dict) -> None:
    def_map = defensive_tactic_map(tax)
    assert {"Model", "Harden", "Detect", "Isolate", "Deceive", "Evict", "Restore"} == set(
        def_map
    )


def test_stage_a_no_orphans_and_recon_is_dual(tax: dict) -> None:
    techniques = pl.read_parquet(ROOT / "data" / "attack_techniques.parquet")
    edges = pl.read_parquet(ROOT / "data" / "mappings_edges.parquet")
    technique_subcap, topic_partition, summary = build_stage_a(
        tax, techniques, edges, ROOT / "inputs" / "d3fend.csv"
    )
    # every active technique assigned exactly one subcap
    assert summary["n_orphans"] == 0, summary["orphan_ids"][:10]
    assert technique_subcap["subcap"].null_count() == 0
    # reconnaissance subcap (graph-blind) must be dual-use by construction
    recon = technique_subcap.filter(pl.col("subcap") == "reconnaissance")
    assert recon.height > 0
    assert (recon["region"] == "dual-use").all()
    # partition has all three regions
    assert {"offensive", "defensive", "dual-use"} <= set(summary["region_counts"])
