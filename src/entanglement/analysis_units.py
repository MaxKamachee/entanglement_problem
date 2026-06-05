"""Unified, normalized, bucket-labeled analysis units across all three corpora.

For the separability experiments we need offense / dual / defense text at *comparable*
granularity and with surface format stripped — otherwise an embedding/probe separates
on length, markup, or source-format artifacts rather than capability (the confound
flagged throughout). This module pulls every corpus layer through the same
`units.clean_text` + `units.resegment` normalization and labels each unit with its
bucket (offense/dual/defense), layer (provenance), and a coarse topic/tactic tag.

Output: `data/analysis_units.parquet`. Downstream (`separability.py`) embeds these and
measures offense/defense separability + where the dual substrate sits. Balance is applied
at *sampling* time downstream (per the corpus-balance decision), not here.

**Framework-metadata prune (default on).** Offense and defense are kept *symmetric and
external-reference-only*: the framework-internal cataloging layers — ATT&CK STIX `uses`
procedure examples (`offense/procedure`, terse attribution data) and D3FEND in-hand
definitions/abstracts (`defense/prose`, framework cataloging) — are dropped. They are
off-target for capability measurement and, per the source-confound diagnostic
(`scripts/diag_source_tsne.py`), their distinctive register dominated the representation
geometry. The prune is config-driven (`prune_framework_metadata`, default true) and applied
in the pure builder so it is reproducible from source, not a one-off parquet mutation.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from entanglement.units import clean_text, resegment

ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_UNIT_COLUMNS = [
    "unit_id", "bucket", "layer", "topic", "n_chars", "text",
]

# Framework-internal cataloging layers dropped by the prune (bucket, layer): kept symmetric.
_FRAMEWORK_METADATA_LAYERS = [("offense", "procedure"), ("defense", "prose")]

# (parquet file, bucket, layer, text column, topic/tactic source column, row-filter or None)
# row-filter = (column, value): keep only rows where df[column] == value, if the column exists.
_SOURCES = [
    ("attack_procedures.parquet", "offense", "procedure", "text", "tactics", None),
    ("offensive_documents.parquet", "offense", "offense_external", "text", "tactics", None),
    ("d3fend_prose_tagged.parquet", "defense", "prose", "text", "tactic", None),
    # defense external = D3FEND-cited docs only; the MS supplement is a separate layer below, so filter
    # it out here (after integration defensive_documents also holds supplement_github rows).
    ("defensive_documents.parquet", "defense", "defense_external", "text", "tactics",
     ("source_category", "d3fend_cited")),
    ("defensive_supplement_github.parquet", "defense", "external_supplement", "text", "topic", None),
    ("substrate_units.parquet", "dual", "substrate", "text", "topic", None),
]


def _first_topic(value) -> str | None:
    """Coarse topic/tactic tag: first element of a list column, or a scalar, else None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, pl.Series)):
        vals = [v for v in list(value) if v]
        return vals[0] if vals else None
    return str(value) or None


def _prune_framework_metadata(units: pl.DataFrame) -> pl.DataFrame:
    """Drop framework-internal cataloging layers (offense/procedure, defense/prose).

    Pure + deterministic; keeps offense and defense external-reference-only and symmetric.
    """
    keep = pl.lit(True)
    for bucket, layer in _FRAMEWORK_METADATA_LAYERS:
        keep = keep & ~((pl.col("bucket") == bucket) & (pl.col("layer") == layer))
    return units.filter(keep)


def build_analysis_units(
    data_dir: Path = ROOT / "data",
    *,
    target: int = 3000,
    hard_max: int = 4000,
    prune_framework_metadata: bool = True,
) -> pl.DataFrame:
    """Normalize every corpus layer into comparable units, labeled by bucket/layer/topic.

    With ``prune_framework_metadata=True`` (default), offense/procedure and defense/prose
    rows are dropped (external-reference-only, symmetric). Pass ``False`` for the full pool.
    """
    from entanglement.units import content_hash

    rows: list[dict] = []
    for fname, bucket, layer, text_col, topic_col, row_filter in _SOURCES:
        path = data_dir / fname
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        if row_filter is not None and row_filter[0] in df.columns:
            df = df.filter(pl.col(row_filter[0]) == row_filter[1])
        has_topic = topic_col in df.columns
        for r in df.iter_rows(named=True):
            topic = _first_topic(r.get(topic_col)) if has_topic else None
            for unit_text in resegment(clean_text(r[text_col] or ""), target=target, hard_max=hard_max):
                rows.append({
                    "unit_id": content_hash(unit_text),
                    "bucket": bucket,
                    "layer": layer,
                    "topic": topic,
                    "n_chars": len(unit_text),
                    "text": unit_text,
                })
    units = pl.DataFrame(rows, schema={
        "unit_id": pl.String, "bucket": pl.String, "layer": pl.String,
        "topic": pl.String, "n_chars": pl.Int64, "text": pl.String,
    })
    if prune_framework_metadata:
        units = _prune_framework_metadata(units)
    # dedup identical units (cross-layer/bucket): keep first occurrence
    return units.unique(subset=["unit_id"], keep="first", maintain_order=True)


def main() -> None:
    from entanglement import config as cfg

    data = ROOT / "data"
    prune = cfg.prune_framework_metadata()
    units = build_analysis_units(data, prune_framework_metadata=prune)
    print(f"prune_framework_metadata={prune} (offense/procedure + defense/prose "
          f"{'dropped' if prune else 'kept'}) -> {units.height} units")

    if cfg.cleanup_enabled():
        from entanglement.cleanup import apply_cleanup
        before = units.height
        units, log = apply_cleanup(units, **cfg.cleanup_params())
        log.write_parquet(data / "cleanup_drops.parquet")
        print(f"cleanup: dropped {before - units.height} units -> {units.height}")
        for row in log.group_by("reason").len().sort("len", descending=True).iter_rows(named=True):
            print(f"   {row['reason']:24s} {row['len']}")

    units.write_parquet(data / "analysis_units.parquet")
    print(f"wrote analysis_units.parquet ({units.height} units)")
    print(f"analysis units: {units.height}  | chars: {units['n_chars'].sum():,}")
    print("by bucket:")
    for row in units.group_by("bucket").agg(
        pl.len().alias("units"), pl.col("n_chars").median().alias("med")
    ).sort("bucket").iter_rows(named=True):
        print(f"   {row['bucket']:8s} {row['units']:6d} units  median {int(row['med'])}c")
    print("by layer:")
    for row in units.group_by("layer").len().sort("len", descending=True).iter_rows(named=True):
        print(f"   {row['layer']:18s} {row['len']}")


if __name__ == "__main__":
    main()
