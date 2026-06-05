"""Defensive corpus: in-hand prose tagging + external-reference selection.

In-hand layer: D3FEND's own prose (definitions + kb-abstracts, already parsed by
``d3fend.py``) tagged with D3FEND tactic (from inputs/d3fend.csv) and parent
technique (from the OWL subClassOf hierarchy). This is the defensive baseline,
no network required.

External layer (selection only this round — scraping is deferred): the
kb-reference URLs filtered to the kept source buckets. PATENT and MITRE_CAR are
always dropped; OTHER is *not* dropped yet — ``main()`` emits a per-bucket count
table and a 50-row OTHER sample to reports/ so the keep/drop decision on OTHER is
made from data before any scrape.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

from entanglement import scrape
from entanglement.config import (
    defense_keep_buckets,
    defense_keep_other_curated,
    defense_other_drop_hosts,
    scrape_user_agent,
)

ROOT = Path(__file__).resolve().parents[2]

DEFENSE_PROSE_COLUMNS = [
    "tech_id",    # D3FEND id (definitions); ref-local-name for kb_abstracts
    "label",
    "kind",       # "definition" | "kb_abstract"
    "tactic",     # Model/Harden/Detect/Isolate/Deceive/Evict/Restore (null if unmapped)
    "parent_id",  # parent D3FEND technique (self for top-level; null if no concept)
    "text",
    "n_chars",
]

DEFENSE_DOC_COLUMNS = [
    "content_hash", "url", "raw_url", "extractor", "n_chars",
    "d3fend_ids", "tactics", "parent_ids", "buckets", "text",
]


def load_tactic_map(d3fend_csv: str | Path) -> pl.DataFrame:
    """Return [d3fend_id, tactic] from inputs/d3fend.csv (one row per technique)."""
    return (
        pl.read_csv(d3fend_csv)
        .filter(pl.col("ID").is_not_null() & (pl.col("ID").str.len_chars() > 0))
        .select(
            pl.col("ID").alias("d3fend_id"),
            pl.col("D3FEND Tactic").alias("tactic"),
        )
        .unique(subset=["d3fend_id"], keep="first")
    )


def tag_prose_with_tactics(
    prose: pl.DataFrame,
    tactic_map: pl.DataFrame,
    hierarchy: pl.DataFrame,
) -> pl.DataFrame:
    """Attach D3FEND tactic + parent technique to each prose row.

    Definition rows key on a real D3FEND id and map cleanly; kb_abstract rows are
    reference-keyed and generally have no tactic/parent, which are left null.
    """
    parents = hierarchy.select("d3fend_id", "parent_id")
    return (
        prose.rename({"subject_id": "tech_id"})
        .join(tactic_map, left_on="tech_id", right_on="d3fend_id", how="left")
        .join(parents, left_on="tech_id", right_on="d3fend_id", how="left")
        .select(DEFENSE_PROSE_COLUMNS)
    )


def bucket_summary(d3fend_refs: pl.DataFrame) -> pl.DataFrame:
    """Per-bucket reference counts, descending."""
    return d3fend_refs.group_by("bucket").len().sort("len", descending=True)


def sample_other(d3fend_refs: pl.DataFrame, n: int = 50) -> pl.DataFrame:
    """Up to ``n`` OTHER-bucket references for human review."""
    return d3fend_refs.filter(pl.col("bucket") == "OTHER").select(
        "url", "title", "concept_label"
    ).head(n)


def select_defensive_urls(
    d3fend_refs: pl.DataFrame,
    keep_buckets: frozenset[str] | None = None,
    *,
    curated_other: bool | None = None,
    other_drop_hosts: list[str] | None = None,
) -> pl.DataFrame:
    """Select defensive external references to scrape.

    Keep refs whose bucket is in the configured keep set, PLUS (when curated-OTHER is on)
    OTHER refs whose host is not in the noise denylist. PATENT/MITRE_CAR/ATTACK_XREF are
    excluded by virtue of not being in the keep set. ``curated_other``/``other_drop_hosts``
    default to config but can be passed explicitly (tests stay config-independent).
    """
    keep = list(keep_buckets if keep_buckets is not None else defense_keep_buckets())
    base = d3fend_refs.filter(pl.col("bucket").is_in(keep))

    curated = defense_keep_other_curated() if curated_other is None else curated_other
    if not curated:
        return base
    drop_hosts = list(
        other_drop_hosts if other_drop_hosts is not None else defense_other_drop_hosts()
    )

    def host_kept(url: str) -> bool:
        host = urlsplit(url).netloc.lower()
        return bool(host) and not any(d in host for d in drop_hosts)

    other = d3fend_refs.filter(
        (pl.col("bucket") == "OTHER")
        & pl.col("url").map_elements(host_kept, return_dtype=pl.Boolean)
    )
    return pl.concat([base, other], how="vertical")


def tag_defensive_documents(
    documents: pl.DataFrame,
    provenance: pl.DataFrame,
    d3fend_refs: pl.DataFrame,
    prose_tagged: pl.DataFrame,
) -> pl.DataFrame:
    """Tag each deduped defensive doc with the D3FEND concept(s)/tactic/parent/bucket that cite it.

    Aggregated across every canonical URL that produced a content hash (via provenance), so a
    doc cited by several D3FEND references carries all of them.
    """
    ref_tags = d3fend_refs.select(
        "url", pl.col("concept_id").alias("d3fend_id"), "bucket"
    ).filter(pl.col("d3fend_id").str.len_chars() > 0)
    concept_meta = (
        prose_tagged.select(pl.col("tech_id").alias("d3fend_id"), "tactic", "parent_id")
        .unique(subset=["d3fend_id"], keep="first")
    )
    rolled = (
        provenance.filter(pl.col("content_hash").is_not_null())
        .join(ref_tags, on="url", how="left")
        .join(concept_meta, on="d3fend_id", how="left")
        .group_by("content_hash")
        .agg(
            pl.col("d3fend_id").drop_nulls().unique().sort().alias("d3fend_ids"),
            pl.col("tactic").drop_nulls().unique().sort().alias("tactics"),
            pl.col("parent_id").drop_nulls().unique().sort().alias("parent_ids"),
            pl.col("bucket").drop_nulls().unique().sort().alias("buckets"),
        )
    )
    return documents.join(rolled, on="content_hash", how="left")


def _write_bucket_review(refs: pl.DataFrame, sample: pl.DataFrame, out: Path) -> None:
    keep = sorted(defense_keep_buckets())
    curated = defense_keep_other_curated()
    counts = bucket_summary(refs)
    selected = select_defensive_urls(refs)["url"].n_unique()
    lines = [
        "# Defensive reference buckets — selection record",
        "",
        f"Keep set: **{', '.join(keep)}**. PATENT and MITRE_CAR are dropped (legalese / stub). "
        "`ATTACK_XREF` (attack.mitre.org) is dropped per the no-crosswalks constraint; mis-bucketed "
        "patents (patentimages/patentguru) are reclassified into PATENT. "
        + ("**OTHER: curated keep** — substantive refs kept, noise hosts dropped via "
           "`configs/corpus.yaml:defense_other_drop_hosts`."
           if curated else "`OTHER`: dropped.")
        + f" **Curated defensive selection: {selected} distinct URLs.**",
        "",
        "## Per-bucket counts",
        "",
        "| bucket | refs | status |",
        "|---|---|---|",
    ]
    drop = {"PATENT", "MITRE_CAR", "ATTACK_XREF"}
    for row in counts.iter_rows(named=True):
        b = row["bucket"]
        if b in keep:
            status = "KEEP"
        elif b in drop:
            status = "DROP"
        elif b == "OTHER":
            status = "CURATED-KEEP" if curated else "DROP"
        else:
            status = "?"
        lines.append(f"| {b} | {row['len']} | {status} |")
    lines += [
        "",
        f"## OTHER sample ({sample.height} of "
        f"{int(counts.filter(pl.col('bucket') == 'OTHER')['len'].sum() or 0)})",
        "",
        "| url | title | supports concept |",
        "|---|---|---|",
    ]
    for row in sample.iter_rows(named=True):
        title = (row["title"] or "").replace("|", "\\|")[:80]
        concept = (row["concept_label"] or "").replace("|", "\\|")[:40]
        lines.append(f"| {row['url']} | {title} | {concept} |")
    lines.append("")
    out.write_text("\n".join(lines))


def run_external_scrape() -> None:
    """Scrape the curated defensive external refs and write documents + provenance."""
    data = ROOT / "data"
    refs = pl.read_parquet(data / "d3fend_refs.parquet")
    prose_tagged = pl.read_parquet(data / "d3fend_prose_tagged.parquet")
    urls = select_defensive_urls(refs)
    print(f"defensive URLs selected: {urls['url'].n_unique()}")
    documents, provenance = scrape.scrape(
        urls["raw_url"].to_list(), cache_dir=data / "scrape_cache",
        user_agent=scrape_user_agent(),
    )
    tagged = tag_defensive_documents(documents, provenance, refs, prose_tagged)
    tagged = tagged.select([c for c in DEFENSE_DOC_COLUMNS if c in tagged.columns])
    tagged.write_parquet(data / "defensive_documents.parquet")
    provenance.write_parquet(data / "defensive_provenance.parquet")
    print(f"defensive documents: {tagged.height} | provenance rows: {provenance.height}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Defensive corpus: baseline (default) or external scrape.")
    parser.add_argument("--scrape", action="store_true",
                        help="run the curated defensive external scrape (network)")
    if parser.parse_args().scrape:
        run_external_scrape()
        return

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    prose = pl.read_parquet(data / "d3fend_prose.parquet")
    hierarchy = pl.read_parquet(data / "d3fend_hierarchy.parquet")
    refs = pl.read_parquet(data / "d3fend_refs.parquet")
    tactic_map = load_tactic_map(ROOT / "inputs" / "d3fend.csv")

    tagged = tag_prose_with_tactics(prose, tactic_map, hierarchy)
    tagged.write_parquet(data / "d3fend_prose_tagged.parquet")

    other = sample_other(refs, 50)
    other.write_parquet(data / "d3fend_other_sample.parquet")
    _write_bucket_review(refs, other, ROOT / "reports" / "defensive_bucket_review.md")

    kept = select_defensive_urls(refs)
    n_tac = int(tagged["tactic"].is_not_null().sum())
    print(f"prose rows tagged:    {tagged.height} ({n_tac} with a D3FEND tactic)")
    print("tactic distribution (definitions map cleanly):")
    for row in (
        tagged.filter(pl.col("tactic").is_not_null())
        .group_by("tactic").len().sort("len", descending=True).iter_rows(named=True)
    ):
        print(f"   {row['tactic']:10s} {row['len']}")
    print(f"refs kept for scrape: {kept['url'].n_unique()} distinct URLs "
          f"(buckets {sorted(kept['bucket'].unique().to_list())})")
    print("see reports/defensive_bucket_review.md for the curated selection record")


if __name__ == "__main__":
    main()
