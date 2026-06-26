#!/usr/bin/env bash
# One-shot pod setup: install deps + (re)build the units parquets locally on the pod.
# Run from the repo root AFTER you've authenticated HF (the bio forget corpus is gated):
#     hf auth login          # or: export HF_TOKEN=hf_...
#     bash scripts/pod_setup.sh
set -euo pipefail

echo ">> installing deps"
pip install -q "torch" "transformers>=4.43" datasets polars accelerate peft huggingface_hub numpy

echo ">> building data (cyber + wikitext are public; bio forget is gated -> needs HF auth)"
python scripts/prep_wmdp_units.py --domain cyber
python scripts/prep_wmdp_units.py --domain bio
python scripts/prep_wikitext_units.py

echo ">> data ready:"
ls -lh data/*.parquet
echo ">> done. Next: see docs/POD_RUN.md"
