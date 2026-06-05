"""Tests for the GitHub blue-team supplement helpers (pure, offline — no network)."""

from __future__ import annotations

from entanglement.supplement_github import d3fend_tactic, passes_filter


def test_d3fend_tactic_detect():
    txt = ("This Splunk SIEM detection identifies lateral movement; threat hunting queries and YARA "
           "rules help analysts detect and investigate malicious activity in the logs.")
    assert d3fend_tactic(txt) == "Detect"


def test_d3fend_tactic_restore_and_model():
    assert d3fend_tactic("Data recovery: restore deleted files and rebuild from backup after an incident.") == "Restore"
    assert d3fend_tactic("Network discovery and asset inventory using nmap to enumerate hosts and map the network.") == "Model"


def test_d3fend_tactic_fallback():
    assert d3fend_tactic("A general note about team communication and security awareness culture.") == "general_defense"


def test_passes_filter():
    ops = ("Detection guidance for incident response: monitor logs, hunt for malware, analyze "
           "forensic artifacts and alert on threats. " * 10)
    assert passes_filter("Methodology/Threat Hunting/x.md", ops) is True
    assert passes_filter("LICENSE.md", ops) is False              # skipped path
    assert passes_filter("notes.md", "too short") is False        # below floor
    assert passes_filter("notes.md", "x" * 800) is False          # long but no defensive terms
