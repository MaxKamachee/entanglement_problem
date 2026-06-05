"""Stage B (part 1): assemble the in-hand corpus with framework-provenance labels.

Pulls together the authoritative text we already have — ATT&CK technique
descriptions (offensive / dual-use topics) and D3FEND definitions (defensive /
dual-use topics) — into one provenance-labeled corpus table. Every doc inherits
its region from its topic's framework provenance (Stage A's partition); no LLM.

External topic-driven retrieval (arXiv / NIST / etc.) augments this in a later
step. The <2000-char stub filter is intentionally NOT applied here — it targets
scraped web stubs, not curated MITRE prose.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

# Corpus document schema (column order) shared by every source bucket.
CORPUS_COLUMNS = [
    "doc_id",
    "topic_id",
    "topic_name",
    "framework",
    "region",
    "subcap",
    "source_bucket",
    "source_ref",
    "fetch_date",
    "label_method",
    "n_chars",
    "text",
]


def assemble_inhand(
    techniques: pl.DataFrame,
    d3fend_prose: pl.DataFrame,
    partition: pl.DataFrame,
    today: str,
) -> pl.DataFrame:
    """Build the in-hand corpus: ATT&CK descriptions + D3FEND definitions."""
    attack_topics = partition.filter(pl.col("framework") == "ATTACK").select(
        "topic_id", "region", "subcap"
    )
    d3fend_topics = partition.filter(pl.col("framework") == "D3FEND").select(
        "topic_id", "region", "subcap"
    )

    # --- ATT&CK technique descriptions (offensive / dual-use) ---
    att = (
        techniques.join(
            attack_topics, left_on="tech_id", right_on="topic_id", how="inner"
        )
        .filter(pl.col("description").str.len_chars() > 0)
        .select(
            pl.col("tech_id").alias("topic_id"),
            pl.col("name").alias("topic_name"),
            pl.lit("ATTACK").alias("framework"),
            pl.col("region"),
            pl.col("subcap"),
            pl.lit("ATTACK_DESC").alias("source_bucket"),
            (
                "https://attack.mitre.org/techniques/"
                + pl.col("tech_id").str.replace_all(".", "/", literal=True)
            ).alias("source_ref"),
            pl.col("description").alias("text"),
        )
    )

    # --- D3FEND definitions (defensive / dual-use) ---
    defs = (
        d3fend_prose.filter(pl.col("kind") == "definition")
        .join(d3fend_topics, left_on="subject_id", right_on="topic_id", how="inner")
        .select(
            pl.col("subject_id").alias("topic_id"),
            pl.col("label").alias("topic_name"),
            pl.lit("D3FEND").alias("framework"),
            pl.col("region"),
            pl.col("subcap"),
            pl.lit("D3FEND_DEF").alias("source_bucket"),
            ("https://d3fend.mitre.org/technique/d3f:" + pl.col("label").str.replace_all(" ", ""))
            .alias("source_ref"),
            pl.col("text"),
        )
    )

    corpus = pl.concat([att, defs], how="vertical").with_columns(
        # readable, unique-per-(bucket,topic) doc id, e.g. "ATTACK_DESC:T1003"
        (pl.col("source_bucket") + ":" + pl.col("topic_id")).alias("doc_id"),
        pl.lit(today).alias("fetch_date"),
        pl.lit("framework_provenance").alias("label_method"),
        pl.col("text").str.len_chars().alias("n_chars"),
    )
    return corpus.select(CORPUS_COLUMNS)


def _report(corpus: pl.DataFrame, d3fend_prose: pl.DataFrame) -> str:
    by_region = corpus.group_by("region").agg(
        pl.len().alias("docs"), pl.col("n_chars").sum().alias("chars")
    ).sort("region")
    by_bucket = corpus.group_by("source_bucket").agg(
        pl.len().alias("docs"), pl.col("n_chars").sum().alias("chars")
    ).sort("source_bucket")
    kb_chars = int(
        d3fend_prose.filter(pl.col("kind") == "kb_abstract")["n_chars"].sum()
    )
    lines = [
        f"in-hand corpus docs: {corpus.height}",
        f"total chars:         {corpus['n_chars'].sum():,}",
        "",
        "by region:",
    ]
    for r in by_region.iter_rows(named=True):
        lines.append(f"   {r['region']:10s} {r['docs']:5d} docs  {r['chars']:>10,} chars")
    lines.append("by source bucket:")
    for r in by_bucket.iter_rows(named=True):
        lines.append(f"   {r['source_bucket']:12s} {r['docs']:5d} docs  {r['chars']:>10,} chars")
    lines.append(
        f"\nNOTE: {kb_chars:,} more chars of D3FEND kb-abstract prose exist "
        "in-hand, not yet topic-mapped (attach via kb-reference-of later)."
    )
    return "\n".join(lines)


def main() -> None:
    data = ROOT / "data"
    techniques = pl.read_parquet(data / "attack_techniques.parquet")
    d3fend_prose = pl.read_parquet(data / "d3fend_prose.parquet")
    partition = pl.read_parquet(data / "topic_partition.parquet")
    today = _dt.date.today().isoformat()

    corpus = assemble_inhand(techniques, d3fend_prose, partition, today)
    corpus.write_parquet(data / "corpus_inhand.parquet")
    print(_report(corpus, d3fend_prose))


if __name__ == "__main__":
    main()
