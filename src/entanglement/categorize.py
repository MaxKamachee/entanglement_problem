"""Categorize reference URLs into the three-way Venn: Offensive / Defensive / Dual-Use.

Two dual-use signals, combined (tie-break: dual wins):

1. Document-level  — the same URL is cited by both ATT&CK and D3FEND.
2. Artifact-level  — the URL is cited by an ATT&CK technique whose
   "contestedness" (number of distinct D3FEND counterparts it maps to via the
   edge table) is at/above the 75th percentile. These techniques sit on
   heavily-contested artifacts, so their documents behave dual-use even when
   no single document is cited by both sides.

This module reports the empirical overlap; it does NOT yet do content-level
validation of dual-use (a later step).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

DUAL_PERCENTILE = 0.75


def technique_contestedness(edges: pl.DataFrame) -> pl.DataFrame:
    """Distinct D3FEND counterparts per ATT&CK technique (exact id ∪ parent rollup)."""
    by_id = edges.group_by("off_tech_id").agg(
        pl.col("def_tech_label").n_unique().alias("contest_id")
    )
    by_parent = edges.group_by("off_tech_parent_id").agg(
        pl.col("def_tech_label").n_unique().alias("contest_parent")
    )
    return by_id, by_parent


def categorize(
    attack_refs: pl.DataFrame,
    d3fend_refs: pl.DataFrame,
    edges: pl.DataFrame,
) -> tuple[pl.DataFrame, dict]:
    """Return (per-URL category table, summary dict)."""
    by_id, by_parent = technique_contestedness(edges)

    # contestedness per ATT&CK technique = max(own, parent rollup)
    tech_contest = (
        attack_refs.select("tech_id")
        .unique()
        .with_columns(
            pl.col("tech_id").str.split(".").list.first().alias("parent_id")
        )
        .join(by_id, left_on="tech_id", right_on="off_tech_id", how="left")
        .join(by_parent, left_on="parent_id", right_on="off_tech_parent_id", how="left")
        .with_columns(
            pl.max_horizontal(
                pl.col("contest_id").fill_null(0),
                pl.col("contest_parent").fill_null(0),
            ).alias("contest")
        )
        .select("tech_id", "contest")
    )

    # 75th-percentile threshold over techniques that have ANY mapping (>0).
    mapped = tech_contest.filter(pl.col("contest") > 0)["contest"]
    threshold = float(mapped.quantile(DUAL_PERCENTILE)) if mapped.len() else float("inf")

    # max contestedness per offensive URL (over the techniques citing it)
    url_contest = (
        attack_refs.join(tech_contest, on="tech_id", how="left")
        .group_by("url")
        .agg(pl.col("contest").max().alias("max_contest"))
    )

    attack_urls = set(attack_refs["url"].unique().to_list())
    d3fend_urls = set(d3fend_refs["url"].unique().to_list())
    all_urls = sorted(attack_urls | d3fend_urls)

    df = pl.DataFrame({"url": all_urls}).with_columns(
        pl.col("url").is_in(list(attack_urls)).alias("in_attack"),
        pl.col("url").is_in(list(d3fend_urls)).alias("in_d3fend"),
    ).join(url_contest, on="url", how="left").with_columns(
        pl.col("max_contest").fill_null(0)
    )

    df = df.with_columns(
        pl.when(pl.col("in_attack") & pl.col("in_d3fend"))
        .then(pl.lit("Dual-Use"))
        .when(pl.col("in_attack") & (pl.col("max_contest") >= threshold))
        .then(pl.lit("Dual-Use"))
        .when(pl.col("in_attack"))
        .then(pl.lit("Offensive"))
        .otherwise(pl.lit("Defensive"))
        .alias("category")
    ).with_columns(
        # provenance of the dual-use label, for spot-checking
        pl.when(pl.col("in_attack") & pl.col("in_d3fend"))
        .then(pl.lit("shared_citation"))
        .when(pl.col("in_attack") & (pl.col("max_contest") >= threshold))
        .then(pl.lit("artifact_lift"))
        .otherwise(pl.lit(""))
        .alias("dual_reason")
    )

    summary = {
        "attack_urls": len(attack_urls),
        "d3fend_urls": len(d3fend_urls),
        "shared_citation": len(attack_urls & d3fend_urls),
        "contest_threshold_p75": threshold,
        "category_counts": dict(
            df.group_by("category").len().sort("category").iter_rows()
        ),
        "dual_reason_counts": dict(
            df.filter(pl.col("category") == "Dual-Use")
            .group_by("dual_reason").len().iter_rows()
        ),
    }
    return df, summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data"

    attack_refs = pl.read_parquet(out / "attack_refs.parquet")
    d3fend_refs = pl.read_parquet(out / "d3fend_refs.parquet")
    edges = pl.read_parquet(out / "mappings_edges.parquet")

    df, summary = categorize(attack_refs, d3fend_refs, edges)
    df.write_parquet(out / "url_categories.parquet")

    print(f"ATT&CK URLs:            {summary['attack_urls']}")
    print(f"D3FEND URLs:            {summary['d3fend_urls']}")
    print(f"shared-citation (∩):    {summary['shared_citation']}")
    print(f"contestedness p75:      {summary['contest_threshold_p75']:.1f} "
          f"D3FEND counterparts")
    print("\nVenn (per URL):")
    for cat, n in sorted(summary["category_counts"].items()):
        print(f"   {cat:10s} {n}")
    print("\ndual-use provenance:")
    for reason, n in summary["dual_reason_counts"].items():
        print(f"   {reason:16s} {n}")


if __name__ == "__main__":
    main()
