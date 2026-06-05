"""Parse the 14k-row D3FEND->ATT&CK mappings CSV into the offense/defense edge table.

This is the bridge: each row links a defensive (D3FEND) technique to an
offensive (ATT&CK) technique, mediated by a shared digital artifact and
annotated with verb-chain relations on both sides. It's what lets us measure
artifact-level dual-use (how contested an ATT&CK technique's artifacts are).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from entanglement.normalize import parent_tech_id

# Columns we keep from the wide source CSV (labels + the offensive T-id).
_KEEP = [
    "def_tech_label",
    "def_artifact_label",
    "def_artifact_rel_label",
    "off_artifact_label",
    "off_artifact_rel_label",
    "off_tech_label",
    "off_tech_id",
    "off_tech_parent_label",
]


def build_edges(csv_path: str | Path) -> pl.DataFrame:
    """Return the normalized offense/defense edge table."""
    df = pl.read_csv(csv_path, columns=_KEEP, infer_schema_length=10000)
    # Normalize the offensive technique id + derive its parent, using native
    # string expressions (faster than map_elements; same result as the
    # normalize.py helpers, which the unit tests pin).
    df = df.with_columns(
        pl.col("off_tech_id").str.strip_chars().str.to_uppercase().alias("off_tech_id"),
    ).with_columns(
        pl.col("off_tech_id").str.split(".").list.first().alias("off_tech_parent_id"),
    )
    return df


def coverage_report(edges: pl.DataFrame, attack_techniques: pl.DataFrame) -> dict:
    """Check off_tech_id resolution against the parsed ATT&CK technique set."""
    known = set(attack_techniques["tech_id"].to_list())
    mapped_ids = set(edges["off_tech_id"].drop_nulls().to_list())
    mapped_parents = set(edges["off_tech_parent_id"].drop_nulls().to_list())

    resolved = mapped_ids & known
    # an id "resolves" if itself OR its parent is a known enterprise technique
    unresolved = {
        t for t in mapped_ids
        if t not in known and parent_tech_id(t) not in known
    }
    return {
        "rows": edges.height,
        "distinct_off_tech_ids": len(mapped_ids),
        "resolved_exact": len(resolved),
        "unresolved": sorted(unresolved),
        "distinct_def_techs": edges["def_tech_label"].n_unique(),
        "mapped_parents_in_attack": len(mapped_parents & known),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data"
    out.mkdir(exist_ok=True)

    edges = build_edges(root / "inputs" / "d3fend-full-mappings.csv")
    edges.write_parquet(out / "mappings_edges.parquet")

    attack = pl.read_parquet(out / "attack_techniques.parquet")
    rep = coverage_report(edges, attack)

    print(f"edge rows:              {rep['rows']:,}")
    print(f"distinct D3FEND techs:  {rep['distinct_def_techs']}")
    print(f"distinct ATT&CK ids:    {rep['distinct_off_tech_ids']}")
    print(f"  resolved (exact):     {rep['resolved_exact']}")
    print(f"  resolved via parent:  {rep['mapped_parents_in_attack']} parents in ATT&CK")
    print(f"  UNRESOLVED:           {len(rep['unresolved'])}")
    if rep["unresolved"]:
        print("  unresolved ids (first 30):", rep["unresolved"][:30])


if __name__ == "__main__":
    main()
