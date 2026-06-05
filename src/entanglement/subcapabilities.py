"""Stage A: subcapability taxonomy, entanglement profile, topic partition.

Reads the locked taxonomy (configs/subcapabilities.yaml) and the parsed ATT&CK /
mappings tables, then:

1. assigns every ATT&CK technique to exactly one offensive subcapability
   (earliest kill-chain tactic wins for multi-tactic techniques);
2. measures per-technique contestedness (distinct D3FEND counterparts via the
   14k mappings, exact id or parent rollup) and rolls it up per subcapability;
3. emits the framework-grounded topic partition (offensive / defensive /
   dual-use) that drives Stage B retrieval.

No external dependencies; pure computation over data/ + inputs/.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_taxonomy(path: Path) -> dict:
    """Load and lightly validate the subcapability taxonomy YAML."""
    with path.open() as f:
        tax = yaml.safe_load(f)
    for key in ("offensive", "defensive", "attack_tactic_priority"):
        if key not in tax:
            raise ValueError(f"taxonomy missing required key: {key}")
    return tax


def offensive_tactic_map(tax: dict) -> dict[str, str]:
    """tactic -> offensive subcapability."""
    out: dict[str, str] = {}
    for subcap, body in tax["offensive"].items():
        for tactic in body["attack_tactics"]:
            out[tactic] = subcap
    return out


def defensive_tactic_map(tax: dict) -> dict[str, str]:
    """D3FEND tactic -> defensive subcapability."""
    out: dict[str, str] = {}
    for subcap, body in tax["defensive"].items():
        for tactic in body["d3fend_tactics"]:
            out[tactic] = subcap
    return out


def assign_subcap(
    tactics: list[str], tactic_to_subcap: dict[str, str], priority: list[str]
) -> str | None:
    """Earliest-kill-chain-tactic-wins assignment to one offensive subcap."""
    ranked = sorted(
        (t for t in tactics if t in priority and t in tactic_to_subcap),
        key=priority.index,
    )
    return tactic_to_subcap[ranked[0]] if ranked else None


def build_contestedness(edges: pl.DataFrame) -> pl.DataFrame:
    """Per ATT&CK technique: distinct D3FEND counterparts (exact id ∪ parent)."""
    by_id = edges.group_by("off_tech_id").agg(
        pl.col("def_tech_label").n_unique().alias("contest_id")
    )
    by_parent = edges.group_by("off_tech_parent_id").agg(
        pl.col("def_tech_label").n_unique().alias("contest_parent")
    )
    return by_id, by_parent


def build_stage_a(
    tax: dict,
    techniques: pl.DataFrame,
    edges: pl.DataFrame,
    d3fend_csv: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Return (technique_subcap, topic_partition, summary)."""
    tac_map = offensive_tactic_map(tax)
    priority = tax["attack_tactic_priority"]
    graph_blind = set(tax.get("graph_blind_subcaps", []))
    pct = float(tax.get("dual_use_percentile", 0.75))

    # 1. assign subcap to every technique
    assigned = techniques.with_columns(
        pl.col("tactics")
        .map_elements(
            lambda ts: assign_subcap(list(ts), tac_map, priority),
            return_dtype=pl.String,
        )
        .alias("subcap")
    )
    orphans = assigned.filter(pl.col("subcap").is_null())

    # 2. contestedness (exact ∪ parent rollup)
    by_id, by_parent = build_contestedness(edges)
    assigned = (
        assigned.join(by_id, left_on="tech_id", right_on="off_tech_id", how="left")
        .join(by_parent, left_on="parent_id", right_on="off_tech_parent_id", how="left")
        .with_columns(
            pl.max_horizontal(
                pl.col("contest_id").fill_null(0),
                pl.col("contest_parent").fill_null(0),
            ).alias("contest")
        )
        .drop("contest_id", "contest_parent")
    )

    # 3. dual-use threshold over mapped techniques (contest > 0)
    mapped = assigned.filter(pl.col("contest") > 0)["contest"]
    threshold = float(mapped.quantile(pct)) if mapped.len() else float("inf")

    assigned = assigned.with_columns(
        pl.when(pl.col("subcap").is_in(list(graph_blind)))
        .then(pl.lit("dual-use"))
        .when(pl.col("contest") >= threshold)
        .then(pl.lit("dual-use"))
        .otherwise(pl.lit("offensive"))
        .alias("region"),
        pl.col("subcap").is_in(list(graph_blind)).alias("graph_blind"),
    )

    technique_subcap = assigned.select(
        "tech_id", "name", "subcap", "region", "contest", "graph_blind"
    )

    # 4. topic partition: ATT&CK techniques (offensive/dual) + D3FEND (defensive)
    off_topics = assigned.select(
        pl.col("tech_id").alias("topic_id"),
        pl.col("name").alias("topic_name"),
        pl.lit("ATTACK").alias("framework"),
        pl.col("region"),
        pl.col("subcap"),
        pl.col("contest").cast(pl.Int64),
        pl.col("graph_blind"),
    )

    def_map = defensive_tactic_map(tax)
    d3f = pl.read_csv(d3fend_csv)
    # The technique name is sparse across three hierarchy columns; coalesce.
    name_expr = pl.coalesce(
        pl.col("D3FEND Technique"),
        pl.col("D3FEND Technique Level 0"),
        pl.col("D3FEND Technique Level 1"),
    )
    def_topics = (
        d3f.filter(pl.col("ID").is_not_null() & (pl.col("ID").str.len_chars() > 0))
        .select(
            pl.col("ID").alias("topic_id"),
            name_expr.alias("topic_name"),
            pl.lit("D3FEND").alias("framework"),
            pl.lit("defensive").alias("region"),
            pl.col("D3FEND Tactic")
            .map_elements(lambda t: def_map.get(t, "unmapped"), return_dtype=pl.String)
            .alias("subcap"),
            pl.lit(0).cast(pl.Int64).alias("contest"),
            pl.lit(False).alias("graph_blind"),
        )
    )
    topic_partition = pl.concat([off_topics, def_topics], how="vertical")

    summary = {
        "n_techniques": techniques.height,
        "n_orphans": orphans.height,
        "orphan_ids": orphans["tech_id"].to_list(),
        "dual_threshold": threshold,
        "region_counts": dict(
            topic_partition.group_by("region").len().sort("region").iter_rows()
        ),
        "n_def_topics": def_topics.height,
    }
    return technique_subcap, topic_partition, summary


def _profile(technique_subcap: pl.DataFrame) -> pl.DataFrame:
    return (
        technique_subcap.group_by("subcap")
        .agg(
            pl.len().alias("n_tech"),
            (pl.col("contest") > 0).sum().alias("n_mapped"),
            pl.col("contest").mean().round(2).alias("mean_contest"),
            pl.col("contest").max().alias("max_contest"),
            pl.col("graph_blind").first().alias("graph_blind"),
            (pl.col("region") == "dual-use").sum().alias("n_dual_topics"),
        )
        .with_columns(
            (pl.col("n_mapped") / pl.col("n_tech")).round(2).alias("mapped_frac")
        )
        .sort("mean_contest", descending=True)
    )


def write_report(profile: pl.DataFrame, summary: dict, out: Path) -> None:
    lines = [
        "# Entanglement profile (Stage A)",
        "",
        "Per offensive subcapability: how entangled its ATT&CK techniques are "
        "with D3FEND defenses, measured as distinct D3FEND counterparts per "
        "technique via the 14,003-row mappings (exact id ∪ parent rollup).",
        "",
        f"- Techniques assigned: **{summary['n_techniques']}** "
        f"(orphans: {summary['n_orphans']})",
        f"- Dual-use contestedness threshold (p75 of mapped): "
        f"**{summary['dual_threshold']:.1f}** D3FEND counterparts",
        "- Topic partition regions: "
        + ", ".join(f"{k}={v}" for k, v in summary["region_counts"].items()),
        "",
        "| subcapability | n_tech | mapped_frac | mean_contest | max | dual_topics | graph_blind |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in profile.iter_rows(named=True):
        lines.append(
            f"| {r['subcap']} | {r['n_tech']} | {r['mapped_frac']:.2f} | "
            f"{r['mean_contest']:.2f} | {r['max_contest']} | {r['n_dual_topics']} | "
            f"{'YES' if r['graph_blind'] else ''} |"
        )
    lines += [
        "",
        "## Key finding: the reconnaissance paradox",
        "",
        "`reconnaissance` (ATT&CK Reconnaissance + Resource-Development) has "
        "**zero D3FEND mappings** — the graph is structurally blind to "
        "pre-compromise activity — yet it is the canonical dual-use capability "
        "(scanning/enumeration/OSINT shared by red and blue teams). It is "
        "therefore flagged dual-use *by construction*, not by the graph. This "
        "is the central evidence that graph-contestedness alone cannot define "
        "dual-use.",
        "",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    tax = load_taxonomy(ROOT / "configs" / "subcapabilities.yaml")
    techniques = pl.read_parquet(ROOT / "data" / "attack_techniques.parquet")
    edges = pl.read_parquet(ROOT / "data" / "mappings_edges.parquet")

    technique_subcap, topic_partition, summary = build_stage_a(
        tax, techniques, edges, ROOT / "inputs" / "d3fend.csv"
    )

    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    technique_subcap.write_parquet(ROOT / "data" / "technique_subcap.parquet")
    topic_partition.write_parquet(ROOT / "data" / "topic_partition.parquet")

    profile = _profile(technique_subcap)
    write_report(profile, summary, ROOT / "reports" / "entanglement_profile.md")

    print(f"techniques assigned: {summary['n_techniques']} "
          f"(orphans: {summary['n_orphans']})")
    if summary["orphan_ids"]:
        print("  orphan ids:", summary["orphan_ids"][:20])
    print(f"dual-use threshold (p75): {summary['dual_threshold']:.1f}")
    print(f"region counts: {summary['region_counts']}")
    print(f"defensive topics (D3FEND): {summary['n_def_topics']}")
    print("\nprofile:")
    print(profile)


if __name__ == "__main__":
    main()
