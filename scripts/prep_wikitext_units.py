#!/usr/bin/env python
"""Prep a wikitext retain pool — the canonical RMU/cais-rmu retain set (LOCAL, no GPU).

The unlearning-tax experiment contrasts two retain sets per domain:
  * wikitext  — generic text, the WMDP/cais-rmu default (the "blunt" retain anchor)
  * substrate — the domain's own WMDP retain corpus (the "targeted" retain anchor)

This builds the wikitext side, normalized with the SAME unit pipeline as the WMDP corpora
(`units.clean_text` + `units.resegment`, target 3000 / hard_max 4000, >=250-char floor,
unit_id dedup) so the retain anchor is at equal granularity across arms. Bucket = "retain"
so it drops straight into unlearn_rmu's --retain-buckets retain.

Source: Salesforce/wikitext, wikitext-2-raw-v1 (the cais-rmu retain config), train split.
Capped at --max-source-rows (seed 0) for a bounded, shippable pool.

Writes data/wikitext_units.parquet (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entanglement.scrape import content_hash          # noqa: E402
from entanglement.units import clean_text, resegment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "wikitext_units.parquet"
MIN_CHARS = 250


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="wikitext-2-raw-v1")
    p.add_argument("--max-source-rows", type=int, default=20000,
                   help="cap raw wikitext rows before resegment (seed 0)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", args.config, split="train")
    d = pl.DataFrame({"text": ds["text"]}).filter(pl.col("text").str.len_chars() > MIN_CHARS)
    n_raw = d.height
    if d.height > args.max_source_rows:
        d = d.sample(n=args.max_source_rows, seed=args.seed)
    print(f"wikitext {args.config}: {d.height} source rows (from {n_raw} non-trivial; seed={args.seed})",
          flush=True)

    rows: list[dict] = []
    for r in d.iter_rows(named=True):
        for seg in resegment(clean_text(r["text"] or ""), target=3000, hard_max=4000):
            if len(seg) < MIN_CHARS:
                continue
            rows.append({"unit_id": content_hash(seg), "bucket": "retain",
                         "layer": "wikitext", "topic": None, "n_chars": len(seg), "text": seg})

    units = pl.DataFrame(rows, schema={"unit_id": pl.String, "bucket": pl.String,
                                       "layer": pl.String, "topic": pl.String,
                                       "n_chars": pl.Int64, "text": pl.String})
    units = units.unique(subset=["unit_id"], keep="first", maintain_order=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    units.write_parquet(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({units.height} retain units, median "
          f"{int(units['n_chars'].median())}c)")


if __name__ == "__main__":
    main()
