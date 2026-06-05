"""Freeze the analysis-units corpus to a versioned v1 artifact + manifest.

Validates invariants (refuses to stamp on any violation), then writes `data/corpus_manifest.yaml`
(version, counts, sha256, config snapshot, source/license tiers) and copies the frozen parquet to
`data/analysis_units_v1.parquet`. Downstream MCQ + unlearning experiments pin v1 so their numbers are
reproducible against a fixed corpus.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from entanglement import config
from entanglement.corpus_validate import validate_corpus

ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / "data" / "analysis_units.parquet"
FROZEN = ROOT / "data" / "analysis_units_v1.parquet"
MANIFEST = ROOT / "data" / "corpus_manifest.yaml"

# Source → (redistribution tier, license note). Text of scraped/third-party sources is NOT
# redistributed; the released artifact is URLs + content hashes + the pipeline.
_SOURCE_TIERS = {
    "offense_external": "URL+hash only (ATT&CK-cited third-party literature; mixed licenses)",
    "defense_external": "URL+hash only (D3FEND-cited third-party literature; mixed licenses)",
    "external_supplement": "GitHub blue-team repos via tarball: SOC-Assistant-Guide (MIT, redistributable w/ attribution) + BlueTeam-Tools (no license -> URL+hash only, text not redistributed)",
    "substrate": "mixed: RFCs/MDN/FIPS redistributable; OSTEP/Boneh/IntelSDM/Nmap pointer+hash only",
}


def freeze() -> dict:
    df = pl.read_parquet(UNITS)
    violations = validate_corpus(df)
    if violations:
        raise SystemExit("corpus validation FAILED — refusing to freeze:\n  - "
                         + "\n  - ".join(violations))

    sha = hashlib.sha256(UNITS.read_bytes()).hexdigest()
    by_bucket = dict(df.group_by("bucket").len().sort("bucket").iter_rows())
    by_layer = dict(df.group_by("layer").len().sort("layer").iter_rows())
    manifest = {
        "version": "v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": "data/analysis_units.parquet",
        "frozen_parquet": "data/analysis_units_v1.parquet",
        "sha256": sha,
        "n_units": df.height,
        "n_chars": int(df["n_chars"].sum()),
        "by_bucket": by_bucket,
        "by_layer": by_layer,
        "config": {
            "prune_framework_metadata": config.prune_framework_metadata(),
            "cleanup_enabled": config.cleanup_enabled(),
            "cleanup": config.cleanup_params(),
            "github_supplements": config.github_supplements(),
        },
        "source_tiers": _SOURCE_TIERS,
        "provenance_notes": [
            "Offense/defense are external-reference-only (framework-internal procedure/prose pruned).",
            "Offense cleaned: tiny IOC dumps dropped + MinHash near-dup collapse @0.85.",
            "Dual cleaned: >50% non-prose dumps + bad short fragments dropped.",
            "Defense supplemented with GitHub blue-team repos (SOC-Assistant-Guide + BlueTeam-Tools), "
            "tagged into D3FEND tactics (topic); inferred tactic, authoritative d3fend_ids left empty.",
            "Cross-bucket contamination checked and removed (keep-order offense>dual>defense); 0 remain.",
            "Standing caveat: bucket separability is topic/register-confounded; valence claims require "
            "the unlearning-tax experiment, not separability.",
        ],
    }
    MANIFEST.write_text(yaml.safe_dump(manifest, sort_keys=False))
    shutil.copyfile(UNITS, FROZEN)
    return manifest


def main() -> None:
    m = freeze()
    print(f"froze {m['version']}: {m['n_units']} units {m['by_bucket']}")
    print(f"  sha256 {m['sha256'][:16]}…  -> {FROZEN.relative_to(ROOT)} + {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
