"""Extract ATT&CK procedure examples (the offensive in-hand layer).

A "procedure example" is the ``description`` on a STIX ``uses`` relationship that
points at a technique: it records how a specific malware family, tool,
intrusion-set, or campaign has been observed performing that technique. These are
in-hand offensive content — concrete, attributed, already inside the framework —
so they form the offensive baseline corpus *before* any external scraping.

Enterprise-only this round (D3FEND maps only to enterprise ATT&CK, so ICS
procedures would be entanglement-orphans). The module is domain-aware
(``build_all_procedures`` takes a ``{domain: bundle_path}`` map) so an ICS bundle
can be dropped in later with no code change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
from mitreattack.stix20 import MitreAttackData

from entanglement.attack import _attack_id
from entanglement.normalize import parent_tech_id

ATTACK_USES = "uses"

PROCEDURE_COLUMNS = [
    "proc_id",      # stable hash of (source_ref, target_ref)
    "tech_id",      # supporting technique (normalized T-id)
    "parent_id",    # parent technique (self for top-level)
    "tactics",      # list[str], the technique's ATT&CK kill-chain phases
    "domain",       # "enterprise" (schema retained for future "ics")
    "source_ref",   # STIX id of the malware/tool/intrusion-set/campaign
    "source_type",  # STIX type of the source object
    "source_name",  # display name of the source object
    "text",         # the procedure-example description
    "n_chars",
]


def _proc_id(source_ref: str, target_ref: str) -> str:
    """Stable, idempotent id for a (source, technique) procedure pair."""
    return hashlib.sha256(f"{source_ref}|{target_ref}".encode()).hexdigest()[:16]


def _technique_index(data: MitreAttackData) -> dict[str, tuple[str, list[str]]]:
    """Map active attack-pattern STIX id -> (T-id, kill-chain tactics)."""
    index: dict[str, tuple[str, list[str]]] = {}
    for tech in data.get_techniques(remove_revoked_deprecated=True):
        tid = _attack_id(tech)
        if tid is None:
            continue
        tactics = [
            ph["phase_name"]
            for ph in tech.get("kill_chain_phases", [])
            if ph.get("kill_chain_name") == "mitre-attack"
        ]
        index[tech["id"]] = (tid, tactics)
    return index


def build_procedures(bundle_path: str | Path, domain: str) -> pl.DataFrame:
    """Return one row per procedure example in ``bundle_path`` for ``domain``.

    Keeps only ``uses`` relationships whose target is an active (non-revoked,
    non-deprecated) technique and that carry a non-empty description.
    """
    data = MitreAttackData(str(bundle_path))
    tech_index = _technique_index(data)

    rows: list[dict] = []
    for rel in data.get_objects_by_type("relationship"):
        if rel.get("relationship_type") != ATTACK_USES:
            continue
        if rel.get("revoked") or rel.get("x_mitre_deprecated"):
            continue
        target = tech_index.get(rel.get("target_ref"))
        if target is None:
            continue  # target not an active attack-pattern
        text = (rel.get("description") or "").strip()
        if not text:
            continue
        tid, tactics = target
        src = data.get_object_by_stix_id(rel["source_ref"])
        rows.append(
            {
                "proc_id": _proc_id(rel["source_ref"], rel["target_ref"]),
                "tech_id": tid,
                "parent_id": parent_tech_id(tid),
                "tactics": tactics,
                "domain": domain,
                "source_ref": rel["source_ref"],
                "source_type": src.get("type", "") if src else "",
                "source_name": src.get("name", "") if src else "",
                "text": text,
                "n_chars": len(text),
            }
        )

    return pl.DataFrame(rows, schema=PROCEDURE_COLUMNS).unique(
        subset=["proc_id"], keep="first"
    )


def build_all_procedures(bundles: dict[str, str | Path]) -> pl.DataFrame:
    """Concatenate procedures across domains. This round: {"enterprise": path}."""
    frames = [build_procedures(path, domain) for domain, path in bundles.items()]
    return pl.concat(frames, how="vertical")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data"
    out.mkdir(exist_ok=True)

    bundles = {"enterprise": root / "inputs" / "enterprise-attack-18.1.json"}
    procs = build_all_procedures(bundles)
    procs.write_parquet(out / "attack_procedures.parquet")

    print(f"procedure examples: {procs.height}")
    print(f"distinct techniques: {procs['tech_id'].n_unique()}")
    print("by domain:")
    for row in procs.group_by("domain").len().sort("domain").iter_rows(named=True):
        print(f"   {row['domain']:12s} {row['len']}")
    print("by source type:")
    for row in (
        procs.group_by("source_type").len().sort("len", descending=True).iter_rows(named=True)
    ):
        print(f"   {row['source_type']:14s} {row['len']}")
    print(f"total chars: {procs['n_chars'].sum():,}")


if __name__ == "__main__":
    main()
