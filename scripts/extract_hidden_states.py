#!/usr/bin/env python
"""Phase 1 (GPU): extract Llama-3.1-8B hidden states for the three-way corpus — ONE-SHOT, on a pod.

Self-contained: depends only on torch / transformers / polars / pyyaml, and imports nothing from
the `entanglement` package, so it runs on a bare RunPod box. This is the simplest path when you
already have an SSH-able GPU pod — no provisioning SDK, no payload limits. (A Flash/serverless
variant lives in `extract_hidden_states_flash.py` for when you DON'T have a box.)

What it does
------------
1. Loads `meta-llama/Llama-3.1-8B` (gated -> needs HF_TOKEN + accepted license) in bfloat16.
2. Uniform-stratified-samples N docs/corpus (offense/dual/defense) from the three-way corpus,
   fixed seed -> identical protocol to the BGE pilot, so the results are directly comparable.
3. Per doc: tokenize (truncate 2048), forward pass with output_hidden_states under no_grad,
   masked-mean-pool the requested layers -> one 4096-d vector per (doc, layer). A single forward
   pass yields all hidden states, so extracting 3 layers vs 9 costs the same GPU time.
4. Writes a long-format parquet + a manifest with everything needed for reproducibility.

One-shot run on the pod
-----------------------
    # get the two files onto the box (scp, or runpodctl send, or paste):
    #   extract_hidden_states.py   +   data/analysis_units.parquet
    pip install torch "transformers>=4.43" "polars>=1.0" pyyaml
    export HF_TOKEN=hf_...                      # account that ACCEPTED the Llama-3.1 license
    python extract_hidden_states.py --corpus analysis_units.parquet
    # then copy data/hidden_states/llama31_8b_three_way.parquet back to your laptop.

Defaults reproduce the planned run: 200 docs/corpus, layers 4/16/28, seed 0, masked-mean pooling.
Pass `--layers 0 4 8 12 16 20 24 28 31` for a full depth curve (free — same forward pass).
A 24 GB GPU is enough (bf16 weights ~16 GB); 600 short docs take a few minutes.

Output columns: doc_id, corpus_label, subcap, source_id, layer, embedding(list[f32], 4096).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYERS = [4, 16, 28]          # early / mid / late (hidden_states index; 0 = embeddings)
BUCKETS = ("offense", "dual", "defense")
MAX_TOKENS = 2048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default="data/analysis_units.parquet",
                   help="three-way corpus parquet (cols: unit_id,bucket,layer,topic,text)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None, help="pin a specific HF commit (default: main)")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="hidden_states indices (0=embeddings, k=output of block k)")
    p.add_argument("--buckets", nargs="+", default=list(BUCKETS),
                   help="bucket labels to sample (default: offense dual defense; "
                        "e.g. 'forget retain' for the WMDP-cyber calibration run)")
    p.add_argument("--n-per-corpus", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/hidden_states/llama31_8b_three_way.parquet")
    p.add_argument("--manifest", default="data/hidden_states/extraction_manifest.yaml")
    return p.parse_args(argv)


def stratified_sample(units, n_per_corpus: int, seed: int, buckets=BUCKETS):
    """Uniform sample of up to ``n_per_corpus`` rows per bucket. Deterministic given the seed."""
    import polars as pl

    parts = []
    for bucket in buckets:
        b = units.filter(pl.col("bucket") == bucket)
        parts.append(b.sample(n=min(n_per_corpus, b.height), seed=seed) if b.height else b)
    return pl.concat(parts)


def _resolve_revision(model, fallback) -> str:
    rev = getattr(model.config, "_commit_hash", None)
    return str(rev) if rev else (fallback or "unknown")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import polars as pl
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"corpus not found: {corpus_path} (scp data/analysis_units.parquet to the box)")
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("WARNING: no HF_TOKEN in env — gated Llama-3.1 load will fail.", file=sys.stderr)

    units = pl.read_parquet(corpus_path)
    sample = stratified_sample(units, args.n_per_corpus, args.seed, buckets=args.buckets)
    counts = dict(sample.group_by("bucket").len().iter_rows())
    print(f"sampled {sample.height} docs {counts} (seed={args.seed})", flush=True)

    print(f"loading {args.model} (bf16)...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision,
            torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True,
        ).eval()
    except Exception as exc:
        sys.exit(f"model load failed ({type(exc).__name__}: {exc}); check HF_TOKEN + license + VRAM.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # Llama has none; safe — we mask-pool

    n_hidden = model.config.num_hidden_layers + 1
    bad = [li for li in args.layers if li < 0 or li >= n_hidden]
    if bad:
        sys.exit(f"layer indices {bad} out of range 0..{n_hidden - 1}")

    doc_ids = sample["unit_id"].to_list()
    labels = sample["bucket"].to_list()
    subcaps = sample["topic"].to_list()
    source_ids = sample["layer"].to_list()          # provenance layer -> output `source_id`
    texts = sample["text"].to_list()
    device = next(model.parameters()).device

    rows: list[dict] = []
    total_tokens = 0
    for start in range(0, len(texts), args.batch_size):
        enc = tokenizer(texts[start:start + args.batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=args.max_tokens).to(device)
        mask = enc["attention_mask"]
        total_tokens += int(mask.sum().item())
        with torch.no_grad():
            out = model(**enc)
        m = mask.unsqueeze(-1).to(torch.float32)
        denom = m.sum(dim=1).clamp(min=1.0)
        for li in args.layers:
            pooled = ((out.hidden_states[li].to(torch.float32) * m).sum(dim=1) / denom).cpu().numpy()
            for bi in range(pooled.shape[0]):
                gi = start + bi
                rows.append({
                    "doc_id": doc_ids[gi], "corpus_label": labels[gi], "subcap": subcaps[gi],
                    "source_id": source_ids[gi], "layer": int(li),
                    "embedding": pooled[bi].tolist(),
                })
        done = min(start + args.batch_size, len(texts))
        if done % 50 < args.batch_size:
            print(f"  {done}/{len(texts)} docs ({total_tokens:,} tokens)", flush=True)

    out_df = pl.DataFrame(rows, schema={
        "doc_id": pl.String, "corpus_label": pl.String, "subcap": pl.String,
        "source_id": pl.String, "layer": pl.Int64, "embedding": pl.List(pl.Float32),
    })
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    print(f"wrote {out_path} ({out_df.height} rows = {len(texts)} docs x {len(args.layers)} layers)",
          flush=True)

    manifest = {
        "model": args.model, "revision_requested": args.revision,
        "revision_resolved": _resolve_revision(model, args.revision),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "layers_extracted": list(args.layers), "n_per_corpus": args.n_per_corpus,
        "sample_counts": counts, "seed": args.seed, "pooling": "masked_mean",
        "max_tokens": args.max_tokens, "dtype": "bfloat16", "corpus": str(corpus_path),
        "total_tokens": total_tokens, "n_documents": len(texts),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote {man_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
