"""Cross-bucket contamination check: are any documents near-duplicated ACROSS buckets?

Within-bucket dedup is handled (exact `unit_id` globally + offense near-dup collapse). This checks the
remaining risk: a document that appears near-identically in *two* buckets would corrupt the
offense/dual/defense labels (the same text can't be both offensive and defensive evidence). Reuses the
audit's MinHash/LSH. Expectation ~0 (the buckets come from disjoint sources). Report-only by default;
if cross-bucket pairs are found it lists them and (with --drop) removes the lower-priority-bucket member
(keep-order offense > dual > defense) and rewrites the corpus.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from entanglement.corpus_audit import minhash_signatures, near_duplicate_pairs

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "data" / "analysis_units.parquet"
KEEP_ORDER = {"offense": 0, "dual": 1, "defense": 2}   # lower rank = higher priority (kept)


def cross_bucket_pairs(units: pl.DataFrame, threshold: float = 0.7) -> list[tuple]:
    """Near-dup pairs (est. Jaccard ≥ threshold) whose two members are in different buckets."""
    sigs = minhash_signatures(units["text"].to_list())
    buckets = units["bucket"].to_list()
    ids = units["unit_id"].to_list()
    out = []
    for i, j, est in near_duplicate_pairs(sigs, threshold=threshold):
        if buckets[i] != buckets[j]:
            out.append((ids[i], buckets[i], ids[j], buckets[j], est))
    return out


def drop_ids_for(pairs: list[tuple]) -> set[str]:
    """unit_ids to drop: the member in the lower-priority bucket of each cross-bucket pair."""
    drop = set()
    for id_i, b_i, id_j, b_j, _ in pairs:
        drop.add(id_j if KEEP_ORDER[b_j] > KEEP_ORDER[b_i] else id_i)
    return drop


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Cross-bucket contamination check.")
    ap.add_argument("--drop", action="store_true", help="remove offenders + rewrite the corpus")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()

    units = pl.read_parquet(UNITS)
    pairs = cross_bucket_pairs(units, args.threshold)

    lines = ["# Cross-bucket contamination check", "",
             f"MinHash/LSH over {units.height} units (est. Jaccard ≥{args.threshold}). A *cross-bucket* "
             "near-dup pair would corrupt the offense/dual/defense labels.", "",
             f"**Found {len(pairs)} cross-bucket near-duplicate pairs.**", ""]
    if pairs:
        lines += ["| bucket A | unit A | bucket B | unit B | est. Jaccard |",
                  "|---|---|---|---|---:|"]
        for id_i, b_i, id_j, b_j, est in pairs[:200]:
            lines.append(f"| {b_i} | {id_i[:12]} | {b_j} | {id_j[:12]} | {est:.3f} |")
        lines.append("")
    else:
        lines += ["✅ No cross-bucket contamination — the buckets are cleanly separated by provenance "
                  "(as expected from disjoint sources).", ""]
    (ROOT / "reports" / "cross_bucket_contamination.md").write_text("\n".join(lines))
    print(f"cross-bucket near-dup pairs: {len(pairs)}")

    if pairs and args.drop:
        drop = drop_ids_for(pairs)
        cleaned = units.filter(~pl.col("unit_id").is_in(list(drop)))
        cleaned.write_parquet(UNITS)
        print(f"dropped {len(drop)} offenders -> {cleaned.height} units")
    elif pairs:
        print("re-run with --drop to remove offenders (lower-priority bucket member).")


if __name__ == "__main__":
    main()
