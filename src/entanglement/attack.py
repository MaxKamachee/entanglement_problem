"""Parse the ATT&CK STIX bundle into normalized technique + reference tables.

The offensive side of the Venn. Techniques carry the taxonomy (parent/sub,
tactics, platforms); their ``external_references`` carry the citation URLs that
become the offensive corpus and one half of the dual-use URL intersection.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from mitreattack.stix20 import MitreAttackData

from entanglement.normalize import canonicalize_url, normalize_tech_id, parent_tech_id

# Self-referential source on every technique (carries the T-id, not a doc URL).
_SELF_SOURCE = "mitre-attack"


def _attack_id(technique) -> str | None:
    """Pull the canonical T-id from a technique's mitre-attack external_ref."""
    for ref in technique.get("external_references", []):
        if ref.get("source_name") == _SELF_SOURCE and "external_id" in ref:
            return normalize_tech_id(ref["external_id"])
    return None


def build_attack_tables(
    bundle_path: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (techniques, technique_refs) as polars DataFrames.

    Drops revoked and deprecated techniques. ``technique_refs`` holds one row
    per (technique, external URL), with both the canonicalized and raw URL.
    """
    data = MitreAttackData(str(bundle_path))
    techniques = data.get_techniques(remove_revoked_deprecated=True)

    # Map STIX id -> T-id so we can resolve sub-technique parents.
    stixid_to_tid: dict[str, str] = {}
    for tech in techniques:
        tid = _attack_id(tech)
        if tid:
            stixid_to_tid[tech["id"]] = tid

    tech_rows: list[dict] = []
    ref_rows: list[dict] = []

    for tech in techniques:
        tid = _attack_id(tech)
        if tid is None:
            continue
        is_sub = bool(tech.get("x_mitre_is_subtechnique", False))
        parent = parent_tech_id(tid) if is_sub else tid
        tactics = [
            ph["phase_name"]
            for ph in tech.get("kill_chain_phases", [])
            if ph.get("kill_chain_name") == "mitre-attack"
        ]
        tech_rows.append(
            {
                "tech_id": tid,
                "name": tech.get("name", ""),
                "is_subtechnique": is_sub,
                "parent_id": parent,
                "tactics": tactics,
                "platforms": list(tech.get("x_mitre_platforms", [])),
                "description": tech.get("description", "") or "",
            }
        )
        for ref in tech.get("external_references", []):
            url = ref.get("url")
            if not url or ref.get("source_name") == _SELF_SOURCE:
                continue
            ref_rows.append(
                {
                    "tech_id": tid,
                    "url": canonicalize_url(url),
                    "raw_url": url,
                    "source_name": ref.get("source_name", ""),
                }
            )

    techniques_df = pl.DataFrame(tech_rows)
    refs_df = pl.DataFrame(ref_rows).unique(subset=["tech_id", "url"])
    return techniques_df, refs_df


# STIX object type -> our offensive citing-object category. course-of-action (mitigations)
# is intentionally absent (excluded by the no-crosswalks constraint).
_OFFENSIVE_CATEGORY = {
    "attack-pattern": "technique",
    "malware": "software",
    "tool": "software",
    "intrusion-set": "group",
    "campaign": "campaign",
}

OFFENSIVE_REF_COLUMNS = [
    "url", "raw_url", "citing_type", "citing_id", "citing_name", "tech_id", "tactics", "domain",
]


def build_offensive_refs(bundle_path: str | Path, domain: str = "enterprise") -> pl.DataFrame:
    """One row per (external-ref URL, citing ATT&CK object), tagged with citing-object type.

    Walks every citing object type (techniques, software, groups, campaigns) plus the
    procedure-bearing ``uses`` relationships, so downstream selection can keep refs cited
    by operational/methodological types and drop attribution-only (group/campaign) refs.
    Excludes the ``mitre-attack`` self-source and ``course-of-action`` mitigations.
    """
    data = MitreAttackData(bundle_path if isinstance(bundle_path, str) else str(bundle_path))

    # active attack-pattern stix-id -> (tech_id, tactics), for technique + procedure tagging
    ap_index: dict[str, tuple[str, list[str]]] = {}
    for t in data.get_techniques(remove_revoked_deprecated=True):
        tid = _attack_id(t)
        if tid:
            ap_index[t["id"]] = (
                tid,
                [ph["phase_name"] for ph in t.get("kill_chain_phases", [])
                 if ph.get("kill_chain_name") == "mitre-attack"],
            )

    rows: list[dict] = []

    def emit(refs, citing_type, citing_id, citing_name, tech_id, tactics):
        for ref in refs or []:
            url = ref.get("url")
            if not url or ref.get("source_name") == _SELF_SOURCE:
                continue
            rows.append({
                "url": canonicalize_url(url), "raw_url": url,
                "citing_type": citing_type, "citing_id": citing_id, "citing_name": citing_name,
                "tech_id": tech_id, "tactics": tactics, "domain": domain,
            })

    for stix_type, category in _OFFENSIVE_CATEGORY.items():
        for obj in data.get_objects_by_type(stix_type):
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue
            tid, tactics = ap_index.get(obj["id"], (None, [])) if category == "technique" else (None, [])
            emit(obj.get("external_references"), category, _attack_id(obj) or obj["id"],
                 obj.get("name", ""), tid, tactics)

    for rel in data.get_objects_by_type("relationship"):
        if rel.get("relationship_type") != "uses" or rel.get("revoked") or rel.get("x_mitre_deprecated"):
            continue
        tid, tactics = ap_index.get(rel.get("target_ref"), (None, []))
        emit(rel.get("external_references"), "procedure", rel["id"], "", tid, tactics)

    return pl.DataFrame(rows, schema={
        "url": pl.String, "raw_url": pl.String, "citing_type": pl.String,
        "citing_id": pl.String, "citing_name": pl.String, "tech_id": pl.String,
        "tactics": pl.List(pl.String), "domain": pl.String,
    })


def main() -> None:
    """Build the tables, write parquet to data/, print a summary."""
    root = Path(__file__).resolve().parents[2]
    bundle = root / "inputs" / "enterprise-attack-18.1.json"
    out = root / "data"
    out.mkdir(exist_ok=True)

    techniques_df, refs_df = build_attack_tables(bundle)
    techniques_df.write_parquet(out / "attack_techniques.parquet")
    refs_df.write_parquet(out / "attack_refs.parquet")
    off_refs = build_offensive_refs(bundle)
    off_refs.write_parquet(out / "offensive_refs.parquet")

    n_sub = int(techniques_df["is_subtechnique"].sum())
    print(f"techniques:        {techniques_df.height} "
          f"({n_sub} sub, {techniques_df.height - n_sub} top-level)")
    print(f"reference rows:    {refs_df.height}")
    print(f"distinct ref URLs: {refs_df['url'].n_unique()}")
    print(f"offensive refs:    {off_refs.height} rows, "
          f"{off_refs['url'].n_unique()} distinct URLs across types "
          f"{sorted(off_refs['citing_type'].unique().to_list())}")


if __name__ == "__main__":
    main()
