#!/usr/bin/env python
"""Grade SAQ answer dumps with the Haiku judge (off-GPU) + per-region accuracy comparison.

Takes one or more `*.saq_answers.parquet` files (written by eval_suite on the pod), grades each
via `entanglement.saq.grade_saqs_batch` (cheap Haiku judge), writes `<name>.graded.parquet`, and
prints per-region (attack/defend/substrate) accuracy for each — side by side so a c0-vs-cN
before/after is readable at a glance. Needs ANTHROPIC_API_KEY.

Usage: uv run python scripts/grade_saq.py runs/smoke/wmdp_c0.saq_answers.parquet \
                                            runs/smoke/wmdp_c20.saq_answers.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entanglement.saq import grade_saqs_batch  # noqa: E402


def grade_file(path: Path) -> pl.DataFrame:
    answered = pl.read_parquet(path)
    verdicts = grade_saqs_batch(answered, poll_interval=10.0)
    graded = answered.join(verdicts, on="saq_id", how="left")
    out = path.with_suffix(".graded.parquet")
    graded.write_parquet(out)
    print(f"  graded {graded.height} -> {out.name}", flush=True)
    return graded


def main() -> None:
    files = [Path(p) for p in sys.argv[1:]]
    if not files:
        sys.exit("usage: grade_saq.py <file.saq_answers.parquet> [more...]")

    per_region = {}
    overall = {}
    for f in files:
        tag = f.name.replace(".saq_answers.parquet", "")
        print(f"grading {tag} ...", flush=True)
        g = grade_file(f)
        agg = g.group_by("region").agg(pl.col("correct").mean().round(3).alias("acc"),
                                       pl.len().alias("n")).sort("region")
        per_region[tag] = {r["region"]: (r["acc"], r["n"]) for r in agg.iter_rows(named=True)}
        overall[tag] = round(g["correct"].mean(), 3)

    tags = [f.name.replace(".saq_answers.parquet", "") for f in files]
    regions = sorted({r for t in per_region.values() for r in t})
    print("\n=== SAQ accuracy by region (Haiku-judged) ===")
    print(f"{'region':12s} " + "  ".join(f"{t:>12s}" for t in tags))
    for reg in regions:
        cells = []
        for t in tags:
            acc, n = per_region[t].get(reg, ("-", 0))
            cells.append(f"{acc} (n={n})")
        print(f"{reg:12s} " + "  ".join(f"{c:>12s}" for c in cells))
    print(f"{'OVERALL':12s} " + "  ".join(f"{overall[t]!s:>12s}" for t in tags))


if __name__ == "__main__":
    main()
