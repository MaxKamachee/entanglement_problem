#!/usr/bin/env python
"""Phase 1 (GPU): extract Llama-3.1-8B hidden states for the three-way corpus via RunPod Flash.

Model-representation analog of the BGE separability pilot. Instead of "write a script and run
it on a pod yourself", this uses the **RunPod Flash** SDK: the GPU work lives in an
`@Endpoint`-decorated async function that Flash provisions and runs on a serverless A100; the
sampling, payload assembly, and file IO run locally.

Architecture (driven by Flash's 10 MB queue-payload limit)
----------------------------------------------------------
* INPUT is sampled **locally** (we have polars), so only a ~0.4 MB 600-row subset crosses the
  wire — the full 21 MB corpus never leaves the laptop.
* OUTPUT (≈30 MB of float32 hidden states for 3 layers) exceeds the 10 MB cap, so the endpoint
  returns **one layer at a time** as a base64 float16 `.npy` blob (~6.5 MB each). The local
  driver calls the endpoint once per layer (a warm worker keeps the model loaded across calls)
  and reassembles the full long-format table locally.

What the remote function does, per layer
----------------------------------------
load Llama-3.1-8B (bf16, gated -> needs HF_TOKEN) -> tokenize (truncate 2048) -> forward pass
with output_hidden_states under no_grad -> masked-mean-pool the one requested layer -> return
the (n_docs, 4096) matrix as float16.

Prerequisites
-------------
* `RUNPOD_API_KEY` in the environment (or `flash login`) — Flash auth / provisioning.
* `HF_TOKEN` in the environment — an account that has ACCEPTED the gated Llama-3.1 license. It
  is forwarded to the remote worker via `env=` (never hardcoded, never written to disk).

Run (from the project root, on a machine with Flash auth)
---------------------------------------------------------
    export RUNPOD_API_KEY=...           # or: flash login
    export HF_TOKEN=hf_...              # license-accepted account
    flash dev                           # starts the dev server; provisions on first call
    # then, in another shell, drive the extraction:
    python scripts/extract_hidden_states.py

`flash dev` discovers the `@Endpoint` function and provisions the A100 on the first `await`.
(`local=True` can be set on the decorator for a CPU smoke test if torch is installed locally.)

Outputs (written locally)
-------------------------
* data/hidden_states/llama31_8b_three_way.parquet  — long format, one row per (doc, layer):
      doc_id, corpus_label, subcap, source_id, layer, embedding(list[f32], 4096)
* data/hidden_states/extraction_manifest.yaml      — model + resolved revision, layers, seed,
      pooling, sample counts, total tokens, timestamp.

Defaults reproduce the planned run: 200 docs/corpus, layers 4/16/28, seed 0, masked-mean pooling.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path

from runpod_flash import Endpoint, GpuGroup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_LAYERS = [4, 16, 28]          # early / mid / late (hidden_states index; 0 = embeddings)
BUCKETS = ("offense", "dual", "defense")
MAX_TOKENS = 2048
INPUT_TEXT_CAP = 12000                # ~chars; 2048 tokens of English is well under this


# --------------------------------------------------------------------------------------------
# Local helpers (no GPU) — sampling + payload (de)serialization. Importable for offline tests.
# --------------------------------------------------------------------------------------------
def stratified_sample(units, n_per_corpus: int, seed: int, buckets=BUCKETS):
    """Uniform sample of up to ``n_per_corpus`` rows per bucket. Deterministic given the seed.

    Identical protocol to the BGE pilot (``separability.balanced_sample``) so the
    representation-space result is directly comparable. ``buckets`` selects which labels
    to sample (default offense/dual/defense; pass 'forget retain' for the WMDP runs).
    """
    import polars as pl

    parts = []
    for bucket in buckets:
        b = units.filter(pl.col("bucket") == bucket)
        parts.append(b.sample(n=min(n_per_corpus, b.height), seed=seed) if b.height else b)
    return pl.concat(parts)


def encode_sample(sample) -> str:
    """Serialize the (doc_id, text) subset to a base64 parquet string for the wire (<1 MB)."""
    import polars as pl

    buf = io.BytesIO()
    sample.select(
        pl.col("unit_id").alias("doc_id"),
        pl.col("text").str.slice(0, INPUT_TEXT_CAP).alias("text"),
    ).write_parquet(buf, compression="zstd")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_embeddings(emb_npy_b64: str):
    """Decode the remote float16 .npy blob back to a float32 (n_docs, 4096) numpy array."""
    import numpy as np

    arr = np.load(io.BytesIO(base64.b64decode(emb_npy_b64)))
    return arr.astype(np.float32)


# --------------------------------------------------------------------------------------------
# Remote GPU function — runs on a RunPod A100. ALL heavy imports inside (cloudpickle).
# --------------------------------------------------------------------------------------------
@Endpoint(
    name="extract-hidden-states-v2",              # fresh name -> clean worker + current code
    gpu=GpuGroup.AMPERE_80,                       # A100 80GB
    workers=(1, 1),                               # one warm worker for this one-shot job
    idle_timeout=180,                             # keep warm across the 3 per-layer calls
    dependencies=["torch", "transformers>=4.43", "polars", "pyarrow"],
    env={"HF_TOKEN": os.environ.get("HF_TOKEN", ""),    # forwarded from local env
         # avoid allocator fragmentation across the per-layer / cross-corpus calls on a
         # warm worker (a full bf16 batch can't grow into freed-but-fragmented blocks).
         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    execution_timeout_ms=0,                       # unlimited: gated ~16 GB load + inference
)
async def extract_layer(sample_parquet_b64: str, layer: int, model_name: str,
                        revision, max_tokens: int) -> dict:
    import base64 as _b64
    import io as _io

    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Cache the loaded model in worker globals so the 3 per-layer calls reload it at most once.
    g = globals()
    cache_key = f"_HS_MODEL::{model_name}::{revision}"
    if cache_key not in g:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token         # Llama has none; safe — we mask-pool
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.bfloat16, output_hidden_states=True,
        ).to("cuda").eval()
        g[cache_key] = (tok, mdl)
    tokenizer, model = g[cache_key]

    n_hidden = model.config.num_hidden_layers + 1
    if layer < 0 or layer >= n_hidden:
        raise ValueError(f"layer {layer} out of range 0..{n_hidden - 1}")

    sample = pl.read_parquet(_io.BytesIO(_b64.b64decode(sample_parquet_b64)))
    doc_ids = sample["doc_id"].to_list()
    texts = sample["text"].to_list()

    torch.cuda.empty_cache()                                  # start from a clean allocator
    vecs: list[np.ndarray] = []
    total_tokens = 0
    batch_size = 4
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_tokens).to("cuda")
        mask = enc["attention_mask"]
        total_tokens += int(mask.sum().item())
        with torch.no_grad():
            out = model(**enc, use_cache=False)               # no KV cache: we only forward once
        h = out.hidden_states[layer].to(torch.float32)        # (B, T, H)
        m = mask.unsqueeze(-1).to(torch.float32)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # masked mean -> (B, H)
        vecs.append(pooled.cpu().numpy().astype(np.float16))
        del enc, mask, out, h, m, pooled                      # release per-batch GPU tensors
    torch.cuda.empty_cache()

    emb = np.concatenate(vecs, axis=0)                         # (n_docs, 4096) float16
    buf = _io.BytesIO()
    np.save(buf, emb)
    revision_resolved = getattr(model.config, "_commit_hash", None) or (revision or "unknown")
    return {
        "doc_ids": doc_ids,
        "emb_npy_b64": _b64.b64encode(buf.getvalue()).decode("ascii"),
        "layer": int(layer),
        "revision_resolved": str(revision_resolved),
        "total_tokens": total_tokens,
        "n_docs": len(doc_ids),
        "hidden_size": int(model.config.hidden_size),
    }


# --------------------------------------------------------------------------------------------
# Local driver
# --------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(ROOT / "data" / "analysis_units.parquet"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=None)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--buckets", nargs="+", default=list(BUCKETS),
                   help="bucket labels to sample (default offense dual defense; "
                        "pass 'forget retain' for the WMDP bio/cyber runs)")
    p.add_argument("--n-per-corpus", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "data" / "hidden_states" /
                                        "llama31_8b_three_way.parquet"))
    p.add_argument("--manifest", default=str(ROOT / "data" / "hidden_states" /
                                             "extraction_manifest.yaml"))
    return p.parse_args(argv)


async def run_extraction(args) -> None:
    import polars as pl
    import yaml

    corpus = Path(args.corpus)
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN not set locally; the gated Llama load will fail remotely.",
              flush=True)

    units = pl.read_parquet(corpus)
    sample = stratified_sample(units, args.n_per_corpus, args.seed, buckets=args.buckets)
    counts = dict(sample.group_by("bucket").len().iter_rows())
    print(f"sampled {sample.height} docs {counts} (seed={args.seed}); "
          f"extracting layers {args.layers} on RunPod A100...", flush=True)

    # local metadata, keyed by doc_id, to re-attach to the returned embeddings
    meta = {r["unit_id"]: r for r in sample.select(
        "unit_id", "bucket", "topic", "layer").iter_rows(named=True)}
    sample_b64 = encode_sample(sample)

    frames = []
    revision_resolved = None
    total_tokens = None
    for layer in args.layers:
        res = await extract_layer(sample_b64, layer, args.model, args.revision, args.max_tokens)
        emb = decode_embeddings(res["emb_npy_b64"])           # (n_docs, 4096) float32
        revision_resolved = res["revision_resolved"]
        total_tokens = res["total_tokens"]
        frames.append(pl.DataFrame({
            "doc_id": res["doc_ids"],
            "corpus_label": [meta[d]["bucket"] for d in res["doc_ids"]],
            "subcap": [meta[d]["topic"] for d in res["doc_ids"]],
            "source_id": [meta[d]["layer"] for d in res["doc_ids"]],
            "layer": [layer] * res["n_docs"],
            "embedding": [emb[i].tolist() for i in range(emb.shape[0])],
        }))
        print(f"  layer {layer}: {res['n_docs']} docs x {res['hidden_size']} dims", flush=True)

    out_df = pl.concat(frames)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    print(f"wrote {out_path} ({out_df.height} rows = {sample.height} docs x {len(args.layers)} layers)",
          flush=True)

    manifest = {
        "model": args.model,
        "revision_requested": args.revision,
        "revision_resolved": revision_resolved,
        "layers_extracted": list(args.layers),
        "n_per_corpus": args.n_per_corpus,
        "sample_counts": counts,
        "seed": args.seed,
        "pooling": "masked_mean",
        "max_tokens": args.max_tokens,
        "dtype_compute": "bfloat16",
        "dtype_transport": "float16",
        "gpu": "AMPERE_80 (A100 80GB)",
        "corpus": str(corpus),
        "total_tokens_per_layer_pass": total_tokens,
        "n_documents": sample.height,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"wrote {man_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    asyncio.run(run_extraction(parse_args(argv)))


if __name__ == "__main__":
    main()
