"""Tests for offensive ontology object-type URL selection + document tagging (offline)."""

from __future__ import annotations

import polars as pl

from entanglement import attack
from entanglement.attack import OFFENSIVE_REF_COLUMNS, build_offensive_refs
from entanglement.corpus_offensive import collect_offensive_urls, tag_documents


# --- build_offensive_refs (multi-object-type extraction, fake bundle) ---

def _ref(url):
    return {"source_name": "x", "url": url}


def _obj(stix_id, refs, name="", extra=None):
    o = {"id": stix_id, "name": name, "external_references": [_ref(u) for u in refs]}
    if extra:
        o.update(extra)
    return o


class FakeAttackData:
    def __init__(self, by_type, techniques):
        self._by_type = by_type
        self._techs = techniques

    def get_techniques(self, remove_revoked_deprecated=True):
        return self._techs

    def get_objects_by_type(self, t):
        return self._by_type.get(t, [])


def test_build_offensive_refs_tags_citing_types(monkeypatch):
    tech = {
        "id": "attack-pattern--ap1", "name": "PowerShell",
        "external_references": [
            {"source_name": "mitre-attack", "external_id": "T1059", "url": "https://attack/T1059"},
            _ref("https://blog.example/powershell-attack"),
        ],
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "execution"}],
    }
    by_type = {
        "attack-pattern": [tech],
        "malware": [_obj("malware--m1", ["https://vendor.example/malware-report"], "Emotet")],
        "intrusion-set": [_obj("intrusion-set--g1", ["https://news.example/apt-attribution"], "APT1")],
        "campaign": [_obj("campaign--c1", ["https://news.example/campaign"], "Op X")],
        "relationship": [{
            "id": "relationship--r1", "relationship_type": "uses",
            "target_ref": "attack-pattern--ap1",
            "external_references": [_ref("https://report.example/procedure-use")],
        }],
    }
    monkeypatch.setattr(attack, "MitreAttackData", lambda _p: FakeAttackData(by_type, [tech]))
    refs = build_offensive_refs("ignored.json", "enterprise")

    assert refs.columns == OFFENSIVE_REF_COLUMNS
    # mitre-attack self-ref excluded; everything else present
    by_url = {r["url"]: r for r in refs.iter_rows(named=True)}
    assert "https://attack/T1059" not in by_url               # self-source dropped
    assert by_url["https://blog.example/powershell-attack"]["citing_type"] == "technique"
    assert by_url["https://blog.example/powershell-attack"]["tech_id"] == "T1059"
    assert by_url["https://vendor.example/malware-report"]["citing_type"] == "software"
    assert by_url["https://news.example/apt-attribution"]["citing_type"] == "group"
    assert by_url["https://news.example/campaign"]["citing_type"] == "campaign"
    assert by_url["https://report.example/procedure-use"]["citing_type"] == "procedure"
    assert by_url["https://report.example/procedure-use"]["tech_id"] == "T1059"


# --- collect_offensive_urls: keep technique/software/procedure, drop group/campaign-only ---

def test_collect_offensive_urls_object_type_filter():
    refs = pl.DataFrame({
        "url": ["u_tech", "u_sw", "u_proc", "u_group", "u_mixed"],
        "raw_url": ["u_tech", "u_sw", "u_proc", "u_group", "u_mixed"],
        "citing_type": ["technique", "software", "procedure", "group", "campaign"],
        "citing_id": ["1", "2", "3", "4", "5"],
        "citing_name": ["", "", "", "", ""],
        "tech_id": ["T1", None, "T2", None, None],
        "tactics": [["execution"], [], ["execution"], [], []],
        "domain": ["enterprise"] * 5,
    })
    # u_mixed is cited by BOTH campaign and technique -> kept
    refs = pl.concat([refs, refs.filter(pl.col("url") == "u_mixed").with_columns(
        pl.lit("technique").alias("citing_type"), pl.lit("T9").alias("tech_id"))])

    kept = set(collect_offensive_urls(refs)["url"].to_list())
    assert {"u_tech", "u_sw", "u_proc"} <= kept     # operational types kept
    assert "u_mixed" in kept                         # group/campaign + technique -> kept
    assert "u_group" not in kept                     # group-only -> dropped


def test_tag_documents_aggregates_types_and_techniques():
    documents = pl.DataFrame({
        "content_hash": ["h1"], "url": ["https://a/x"], "raw_url": ["https://a/x"],
        "extractor": ["trafilatura"], "n_chars": [5000], "text": ["body"],
    })
    provenance = pl.DataFrame({
        "url": ["https://a/x", "https://b/y"], "content_hash": ["h1", "h1"],
    })
    url_tags = pl.DataFrame({
        "url": ["https://a/x", "https://b/y"],
        "tech_ids": [["T1003"], []],
        "citing_types": [["technique"], ["software"]],
    })
    techniques = pl.DataFrame({
        "tech_id": ["T1003"], "parent_id": ["T1003"], "tactics": [["credential-access"]],
    })
    row = tag_documents(documents, provenance, url_tags, techniques).row(0, named=True)
    assert row["tech_ids"] == ["T1003"]
    assert row["parent_ids"] == ["T1003"]
    assert row["tactics"] == ["credential-access"]
    assert sorted(row["citing_types"]) == ["software", "technique"]   # union across both URLs
    assert row["domain"] == "enterprise"
