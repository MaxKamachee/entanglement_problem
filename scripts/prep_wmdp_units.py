#!/usr/bin/env python
"""Prep WMDP-cyber forget/retain corpora for the geometry calibration run (LOCAL, no GPU).

Downloads the cyber forget/retain text corpora from `cais/wmdp-corpora` at a PINNED
revision, verifies sha256, and normalizes them with the SAME pipeline as our corpus
(`units.clean_text` + `units.resegment`, target 3000 / hard_max 4000, ≥250-char floor,
unit_id dedup) so the geometry comparison is apples-to-apples at equal granularity.

This is a read-only side analysis vs our corpus: it writes only
`inputs/wmdp_corpora/*.parquet` (raw, gitignored) and `data/wmdp_cyber_units.parquet`.

Next step (GPU, one-shot on a pod — same flow as the original extraction):
    # files onto the box: scripts/extract_hidden_states.py + data/wmdp_cyber_units.parquet
    pip install torch "transformers>=4.43" "polars>=1.0" pyyaml
    export HF_TOKEN=hf_...
    python extract_hidden_states.py --corpus wmdp_cyber_units.parquet \
        --buckets forget retain \
        --out wmdp_cyber_forget_retain.parquet --manifest wmdp_extraction_manifest.yaml
    # copy wmdp_cyber_forget_retain.parquet back to data/hidden_states/
Then: `uv run python scripts/diag_wmdp_geometry.py` -> reports/wmdp_cyber_geometry_baseline.md
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entanglement.scrape import content_hash          # noqa: E402
from entanglement.units import clean_text, resegment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "inputs" / "wmdp_corpora"
OUT = ROOT / "data" / "wmdp_cyber_units.parquet"

# Pinned dataset revision (cais/wmdp-corpora, lastModified 2024-04-25) + file checksums.
REVISION = "daf89fa9b618b63a624228061a9cebacca88009c"
FILES = {
    "cyber-forget-corpus": "d6ea02facf8bfce704a41fbbb5555e261ca093aa15904f595176c33662780e0c",
    "cyber-retain-corpus": "6706314bedc410f5513f108ca687a7f7798e9f12be0a811ec282946495d8dfc1",
}
MIN_CHARS = 250   # tiny-fragment floor (matches our cleanup floor)


def fetch(name: str, expected_sha: str) -> Path:
    dest = RAW_DIR / f"{name}.parquet"
    if not dest.exists():
        url = (f"https://huggingface.co/datasets/cais/wmdp-corpora/resolve/"
               f"{REVISION}/{name}/train-00000-of-00001.parquet")
        print(f"downloading {name} @ {REVISION[:12]} ...", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if got != expected_sha:
        sys.exit(f"sha256 mismatch for {name}: got {got}, expected {expected_sha}")
    return dest


def main() -> None:
    rows: list[dict] = []
    for split, name in (("forget", "cyber-forget-corpus"), ("retain", "cyber-retain-corpus")):
        d = pl.read_parquet(fetch(name, FILES[name]))
        print(f"{name}: {d.height} source docs")
        for r in d.iter_rows(named=True):
            for seg in resegment(clean_text(r["text"] or ""), target=3000, hard_max=4000):
                if len(seg) < MIN_CHARS:
                    continue
                rows.append({"unit_id": content_hash(seg), "bucket": split,
                             "layer": "wmdp_cyber", "topic": None,
                             "n_chars": len(seg), "text": seg})
    units = pl.DataFrame(rows, schema={"unit_id": pl.String, "bucket": pl.String,
                                       "layer": pl.String, "topic": pl.String,
                                       "n_chars": pl.Int64, "text": pl.String})
    units = units.unique(subset=["unit_id"], keep="first", maintain_order=True)
    units.write_parquet(OUT)
    for r in units.group_by("bucket").agg(pl.len().alias("units"),
                                          pl.col("n_chars").median().alias("med")).sort("bucket").iter_rows(named=True):
        print(f"  {r['bucket']:7s} {r['units']:6d} units  median {int(r['med'])}c")
    print(f"wrote {OUT.relative_to(ROOT)} ({units.height} units)")


if __name__ == "__main__":
    main()
