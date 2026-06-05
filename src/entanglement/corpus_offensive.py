"""Offensive external-literature corpus: ATT&CK-ontology URL selection -> scrape -> tag.

The offensive corpus's external layer is the literature ATT&CK cites. Selection is by
**ATT&CK ontology object type** (deterministic, no LLM judge): keep references cited by
operational/methodological objects (techniques, software, procedures), drop references
cited *only* by attribution objects (groups, campaigns); mitigations + the `mitre-attack`
self-source are already excluded upstream. After scraping (shared pipeline), each deduped
document is tagged with the technique(s)/parents/tactics that cite it, the citing-object
types, and domain.

This replaces the earlier WMDP LLM-judge filter; `quality.py` remains in the repo as an
optional *analysis-stage* operational-depth tagger (unwired here). Execution (the scrape)
is deferred — this module's selection + tagging logic is the gated-but-ready part.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from entanglement import scrape
from entanglement.config import scrape_user_agent

ROOT = Path(__file__).resolve().parents[2]

# Citing-object types whose references we keep (operational/methodological).
KEEP_CITING_TYPES = frozenset({"technique", "software", "procedure"})

OFFENSIVE_DOC_COLUMNS = [
    "content_hash", "url", "raw_url", "extractor", "n_chars",
    "tech_ids", "parent_ids", "tactics", "citing_types", "domain", "text",
]


def collect_offensive_urls(offensive_refs: pl.DataFrame) -> pl.DataFrame:
    """One row per canonical URL kept under the ontology object-type rule.

    Groups `offensive_refs` (one row per url×citing-object) to per-url citing-type set,
    technique ids, and tactics; keeps a URL iff it is cited by ≥1 of
    {technique, software, procedure} (drops URLs cited *only* by group/campaign).
    """
    grouped = offensive_refs.group_by("url").agg(
        pl.col("raw_url").first(),
        pl.col("citing_type").unique().sort().alias("citing_types"),
        pl.col("tech_id").drop_nulls().unique().sort().alias("tech_ids"),
        pl.col("tactics").list.explode().drop_nulls().unique().sort().alias("tactics"),
    )
    return grouped.filter(
        pl.col("citing_types").map_elements(
            lambda ts: bool(set(ts) & KEEP_CITING_TYPES), return_dtype=pl.Boolean
        )
    )


def tag_documents(
    documents: pl.DataFrame,
    provenance: pl.DataFrame,
    url_tags: pl.DataFrame,
    techniques: pl.DataFrame,
    domain: str = "enterprise",
) -> pl.DataFrame:
    """Attach tech_ids / parent_ids / tactics / citing_types / domain to each deduped doc.

    Tags are aggregated across *every* canonical URL that produced a given content hash
    (provenance maps url -> content_hash), so a doc cited by several objects carries all.
    """
    joined = (
        provenance.filter(pl.col("content_hash").is_not_null())
        .join(url_tags.select("url", "tech_ids", "citing_types"), on="url", how="left")
    )
    # technique rollup -> parents + tactics
    tech_lookup = techniques.select("tech_id", "parent_id", "tactics")
    rolled = (
        joined.explode("tech_ids").filter(pl.col("tech_ids").is_not_null())
        .join(tech_lookup, left_on="tech_ids", right_on="tech_id", how="left")
        .group_by("content_hash")
        .agg(
            pl.col("tech_ids").unique().sort().alias("tech_ids"),
            pl.col("parent_id").drop_nulls().unique().sort().alias("parent_ids"),
            pl.col("tactics").list.explode().drop_nulls().unique().sort().alias("tactics"),
        )
    )
    # citing-type rollup (separate: software/group/campaign refs carry no tech_id)
    ctypes = (
        joined.explode("citing_types").filter(pl.col("citing_types").is_not_null())
        .group_by("content_hash")
        .agg(pl.col("citing_types").unique().sort().alias("citing_types"))
    )
    return (
        documents.join(rolled, on="content_hash", how="left")
        .join(ctypes, on="content_hash", how="left")
        .with_columns(pl.lit(domain).alias("domain"))
    )


def main() -> None:
    data = ROOT / "data"
    cache = data / "scrape_cache"
    offensive_refs = pl.read_parquet(data / "offensive_refs.parquet")
    techniques = pl.read_parquet(data / "attack_techniques.parquet")

    url_tags = collect_offensive_urls(offensive_refs)
    print(f"offensive URLs selected (ontology object-type filter): {url_tags.height}")
    documents, provenance = scrape.scrape(
        url_tags["raw_url"].to_list(), cache_dir=cache, user_agent=scrape_user_agent()
    )
    print(f"scraped: {documents.height} unique docs / {provenance.height} urls attempted")

    tagged = tag_documents(documents, provenance, url_tags, techniques)
    tagged = tagged.select([c for c in OFFENSIVE_DOC_COLUMNS if c in tagged.columns])
    tagged.write_parquet(data / "offensive_documents.parquet")
    provenance.write_parquet(data / "offensive_provenance.parquet")
    print(f"offensive documents: {tagged.height} | provenance rows: {provenance.height}")


if __name__ == "__main__":
    main()
