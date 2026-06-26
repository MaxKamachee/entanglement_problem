#!/usr/bin/env python
"""Prep WMDP forget/retain corpora for the geometry calibration run (LOCAL, no GPU).

Supports both domains via `--domain {cyber,bio}`. Downloads the forget/retain text
corpora from their PINNED revisions, verifies sha256, and normalizes them with the
SAME pipeline as our corpus (`units.clean_text` + `units.resegment`, target 3000 /
hard_max 4000, >=250-char floor, unit_id dedup) so the bio<->cyber comparison is
apples-to-apples at equal granularity.

Sources
-------
cyber  forget  cais/wmdp-corpora @ daf89fa   cyber-forget-corpus  (1 shard, public)
       retain  cais/wmdp-corpora @ daf89fa   cyber-retain-corpus  (1 shard, public)
bio    forget  cais/wmdp-bio-forget-corpus @ 5a786ed  (2 shards, GATED -> needs HF auth)
       retain  cais/wmdp-corpora @ daf89fa   bio-retain-corpus    (4 shards, public)

Bio is much larger than cyber (~24k full papers forget, ~60k docs retain vs cyber's
1k/4.5k). Resegmenting all of it would make a ~400MB units parquet you'd then ship to
a GPU pod only to sample 200/bucket from it. So `--max-source-docs` caps the per-split
source-doc pool (fixed seed) BEFORE resegmenting. A 200-doc separability probe is
statistically unaffected by whether it's drawn from a 20k- or 500k-unit pool; the cap
just keeps the corpus shippable. The cap is a size-forced deviation and is recorded in
the printed summary so the bio<->cyber comparison stays honest. Cyber defaults to no cap
(it's already small), bio defaults to 3000/split.

Downloads use auth-aware `hf_hub_download` (required: the bio forget repo is gated).

This is a read-only side analysis: it writes only the HF cache (raw, gitignored) and
`data/wmdp_{domain}_units.parquet`.

Next step (GPU, one-shot on a pod -- same flow as the original extraction):
    # files onto the box: scripts/extract_hidden_states.py + data/wmdp_<domain>_units.parquet
    pip install torch "transformers>=4.43" "polars>=1.0" pyyaml
    export HF_TOKEN=hf_...
    python extract_hidden_states.py --corpus wmdp_bio_units.parquet \
        --buckets forget retain \
        --out wmdp_bio_forget_retain.parquet --manifest wmdp_bio_extraction_manifest.yaml
    # copy wmdp_bio_forget_retain.parquet back to data/hidden_states/
Then: `uv run python scripts/diag_wmdp_geometry.py` -> reports/wmdp_cyber_geometry_baseline.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from entanglement.scrape import content_hash          # noqa: E402
from entanglement.units import clean_text, resegment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_CHARS = 250   # tiny-fragment floor (matches our cleanup floor)

# Pinned (repo, revision, file, sha256) per domain/split. sha256 = sha256 of the file
# bytes as served at the pinned revision (verified 2026-06-25 with a valid HF token).
WMDP_CORPORA_REV = "daf89fa9b618b63a624228061a9cebacca88009c"   # cais/wmdp-corpora (2024-04-25)
BIO_FORGET_REV = "5a786ed1041e56fef2d4ed26c1239f12b73a68eb"     # cais/wmdp-bio-forget-corpus (2025-05-29)

DOMAINS: dict[str, dict[str, list[dict]]] = {
    "cyber": {
        "forget": [
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "cyber-forget-corpus/train-00000-of-00001.parquet",
             "sha": "d6ea02facf8bfce704a41fbbb5555e261ca093aa15904f595176c33662780e0c"},
        ],
        "retain": [
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "cyber-retain-corpus/train-00000-of-00001.parquet",
             "sha": "6706314bedc410f5513f108ca687a7f7798e9f12be0a811ec282946495d8dfc1"},
        ],
    },
    "bio": {
        "forget": [
            {"repo": "cais/wmdp-bio-forget-corpus", "rev": BIO_FORGET_REV,
             "file": "data/train-00000-of-00002.parquet",
             "sha": "918b4b66f13fde0807cfb308a4d3d3f68533e184a66732e3b9f8e7c8bee1dd62"},
            {"repo": "cais/wmdp-bio-forget-corpus", "rev": BIO_FORGET_REV,
             "file": "data/train-00001-of-00002.parquet",
             "sha": "3abf82955b42ebd114b08effc447e40f2204730038292654e2654351784c47fe"},
        ],
        "retain": [
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "bio-retain-corpus/train-00000-of-00004.parquet",
             "sha": "76511131f0170b1c8bcd681d3dc37c9ffb4b4c5d1d00622f3ba35a55855b897d"},
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "bio-retain-corpus/train-00001-of-00004.parquet",
             "sha": "9638fba9e0451a89b4a7dc2ceeca6aa0f4402a82c9d1fd0401e8e707fe090510"},
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "bio-retain-corpus/train-00002-of-00004.parquet",
             "sha": "3aefb924ab20a0cb91ff4e6794eec4bba84933275abd0ec16e3e2f10af29ea43"},
            {"repo": "cais/wmdp-corpora", "rev": WMDP_CORPORA_REV,
             "file": "bio-retain-corpus/train-00003-of-00004.parquet",
             "sha": "73393a526796ac54bb24eb36a6a5a383e622d43d6ee17d0063010964534aa8e3"},
        ],
    },
}
DEFAULT_MAX_SOURCE_DOCS = {"cyber": None, "bio": 3000}


def fetch(spec: dict) -> Path:
    """Auth-aware download of one pinned file; verify sha256 of the bytes."""
    path = Path(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset",
                                revision=spec["rev"]))
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != spec["sha"]:
        sys.exit(f"sha256 mismatch for {spec['repo']}::{spec['file']}: "
                 f"got {got}, expected {spec['sha']}")
    return path


def load_split(specs: list[dict], max_source_docs: int | None, seed: int) -> pl.DataFrame:
    """Read + concat all shards of a split (text column only), optionally cap source docs."""
    frames = [pl.read_parquet(fetch(s)).select("text") for s in specs]
    d = pl.concat(frames, how="vertical")
    n_raw = d.height
    if max_source_docs is not None and d.height > max_source_docs:
        d = d.sample(n=max_source_docs, seed=seed)
    return d, n_raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAINS), default="cyber")
    p.add_argument("--max-source-docs", type=int, default=None,
                   help="cap source docs/split before resegment (default: per-domain "
                        "-- cyber None, bio 3000). Use 0 for no cap.")
    p.add_argument("--seed", type=int, default=0, help="seed for the source-doc cap sample")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    domain = args.domain
    cap = args.max_source_docs
    if cap is None:
        cap = DEFAULT_MAX_SOURCE_DOCS[domain]
    elif cap == 0:
        cap = None
    out = ROOT / "data" / f"wmdp_{domain}_units.parquet"

    rows: list[dict] = []
    for split in ("forget", "retain"):
        d, n_raw = load_split(DOMAINS[domain][split], cap, args.seed)
        capped = f" (capped from {n_raw}, seed={args.seed})" if cap and n_raw > cap else ""
        print(f"{domain}-{split}: {d.height} source docs{capped}", flush=True)
        for r in d.iter_rows(named=True):
            for seg in resegment(clean_text(r["text"] or ""), target=3000, hard_max=4000):
                if len(seg) < MIN_CHARS:
                    continue
                rows.append({"unit_id": content_hash(seg), "bucket": split,
                             "layer": f"wmdp_{domain}", "topic": None,
                             "n_chars": len(seg), "text": seg})

    units = pl.DataFrame(rows, schema={"unit_id": pl.String, "bucket": pl.String,
                                       "layer": pl.String, "topic": pl.String,
                                       "n_chars": pl.Int64, "text": pl.String})
    units = units.unique(subset=["unit_id"], keep="first", maintain_order=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    units.write_parquet(out)
    for r in units.group_by("bucket").agg(pl.len().alias("units"),
                                          pl.col("n_chars").median().alias("med")).sort("bucket").iter_rows(named=True):
        print(f"  {r['bucket']:7s} {r['units']:6d} units  median {int(r['med'])}c")
    print(f"wrote {out.relative_to(ROOT)} ({units.height} units)")


if __name__ == "__main__":
    main()
