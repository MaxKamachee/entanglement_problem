#!/usr/bin/env bash
# One-shot pod setup: install deps + (re)build the units parquets locally on the pod.
# Run from the repo root AFTER you've authenticated HF (the bio forget corpus is gated):
#     hf auth login          # or: export HF_TOKEN=hf_...
#     bash scripts/pod_setup.sh
set -euo pipefail

echo ">> disk check (need ~25-27 GB free for deps + zephyr weights + dataset cache)"
df -h "${HF_HOME:-$HOME}" . | sed 's/^/   /'
FREE_GB=$(df -Pk "${HF_HOME:-$HOME}" | awk 'NR==2{print int($4/1024/1024)}')
echo ">> free on cache target: ${FREE_GB} GB  (HF_HOME=${HF_HOME:-<default ~/.cache/huggingface>})"
if [ "${FREE_GB}" -lt 30 ]; then
  echo "!! WARNING: <30 GB free. Set HF_HOME to a roomy volume, e.g.:  export HF_HOME=/workspace/hf"
  echo "!! (continuing in 5s — Ctrl-C to abort)"; sleep 5
fi

echo ">> installing deps"
pip install -q "torch" "transformers>=4.43" datasets polars accelerate peft huggingface_hub numpy

echo ">> building data (cyber + wikitext are public; bio forget is gated -> needs HF auth)"
python scripts/prep_wmdp_units.py --domain cyber
python scripts/prep_wmdp_units.py --domain bio
python scripts/prep_wikitext_units.py

echo ">> data ready:"
ls -lh data/*.parquet
echo ">> done. Next: see docs/POD_RUN.md"
