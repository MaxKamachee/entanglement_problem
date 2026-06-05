"""Tests for procedure-example extraction from STIX `uses` relationships.

No network / no real bundle: a FakeAttackData mimics the slice of the
MitreAttackData interface that procedures.py uses.
"""

from __future__ import annotations

import pytest

from entanglement import procedures
from entanglement.procedures import (
    PROCEDURE_COLUMNS,
    _proc_id,
    build_procedures,
)


def _tech(stix_id: str, tid: str, phases: list[str]) -> dict:
    return {
        "id": stix_id,
        "external_references": [{"source_name": "mitre-attack", "external_id": tid}],
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": p} for p in phases
        ],
    }


def _uses(src: str, tgt: str, desc: str, **extra) -> dict:
    return {
        "relationship_type": "uses",
        "source_ref": src,
        "target_ref": tgt,
        "description": desc,
        **extra,
    }


class FakeAttackData:
    def __init__(self, techniques, relationships, sources):
        self._techniques = techniques
        self._relationships = relationships
        self._sources = sources

    def get_techniques(self, remove_revoked_deprecated=True):
        return self._techniques

    def get_objects_by_type(self, kind):
        return self._relationships if kind == "relationship" else []

    def get_object_by_stix_id(self, sid):
        return self._sources.get(sid)


@pytest.fixture
def fake(monkeypatch):
    techniques = [
        _tech("attack-pattern--ap1", "T1003", ["credential-access"]),
        _tech("attack-pattern--ap2", "T1059.001", ["execution"]),
    ]
    relationships = [
        _uses("malware--m1", "attack-pattern--ap1", "dumps lsass memory"),
        _uses("tool--t1", "attack-pattern--ap2", "runs an encoded powershell command"),
        _uses("intrusion-set--i1", "attack-pattern--ap1", "   "),  # empty desc -> dropped
        _uses("malware--m1", "attack-pattern--unknown", "targets a missing tech"),  # not indexed
        _uses("malware--m1", "attack-pattern--ap1", "revoked rel", revoked=True),  # revoked
        {  # non-uses relationship -> ignored
            "relationship_type": "mitigates",
            "source_ref": "course-of-action--c1",
            "target_ref": "attack-pattern--ap1",
            "description": "not a procedure",
        },
    ]
    sources = {
        "malware--m1": {"type": "malware", "name": "Explosive"},
        "tool--t1": {"type": "tool", "name": "PowerShell"},
        "intrusion-set--i1": {"type": "intrusion-set", "name": "APT-X"},
    }

    def fake_ctor(_path):
        return FakeAttackData(techniques, relationships, sources)

    monkeypatch.setattr(procedures, "MitreAttackData", fake_ctor)
    return build_procedures("ignored.json", "enterprise")


def test_schema_exact(fake):
    assert fake.columns == PROCEDURE_COLUMNS


def test_only_valid_procedures_kept(fake):
    # ap1+m1 and ap2+t1 survive; empty-desc, unknown-target, revoked, non-uses dropped
    assert fake.height == 2
    assert set(fake["tech_id"].to_list()) == {"T1003", "T1059.001"}


def test_parent_and_tactics(fake):
    by_tid = {r["tech_id"]: r for r in fake.iter_rows(named=True)}
    assert by_tid["T1059.001"]["parent_id"] == "T1059"
    assert by_tid["T1003"]["parent_id"] == "T1003"
    assert by_tid["T1003"]["tactics"] == ["credential-access"]
    assert by_tid["T1059.001"]["tactics"] == ["execution"]


def test_source_resolution_and_domain(fake):
    by_tid = {r["tech_id"]: r for r in fake.iter_rows(named=True)}
    assert by_tid["T1003"]["source_type"] == "malware"
    assert by_tid["T1003"]["source_name"] == "Explosive"
    assert (fake["domain"] == "enterprise").all()
    assert (fake["n_chars"] == fake["text"].str.len_chars()).all()


def test_proc_id_stable_and_unique(fake):
    assert fake["proc_id"].n_unique() == fake.height
    assert _proc_id("malware--m1", "attack-pattern--ap1") == _proc_id(
        "malware--m1", "attack-pattern--ap1"
    )
    assert _proc_id("a", "b") != _proc_id("b", "a")


def test_empty_bundle_keeps_schema(monkeypatch):
    monkeypatch.setattr(procedures, "MitreAttackData", lambda _p: FakeAttackData([], [], {}))
    out = build_procedures("ignored.json", "enterprise")
    assert out.columns == PROCEDURE_COLUMNS
    assert out.height == 0
