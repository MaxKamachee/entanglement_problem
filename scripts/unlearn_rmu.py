#!/usr/bin/env python
"""RMU unlearning (GPU, one-shot on a pod) — Llama-3.1-8B-Instruct, WMDP reference algorithm.

Self-contained (torch/transformers/polars only; no `entanglement` imports), same run pattern
as scripts/extract_hidden_states.py. Implements Representation Misdirection for Unlearning
(Li et al. 2024, cais/rmu): steer the FORGET set's mid-layer activations toward a fixed random
control vector (scaled by the steering coefficient), while L2-anchoring the RETAIN set's
activations to the frozen base model. Only the down_proj weights of a few layers around
`--layer` are updated.

Used by the WMDP-vs-ours smoke test. The forget/retain corpora are passed as parquet +
bucket filters, so the SAME script serves both arms:
  Arm A (WMDP):  --forget-parquet wmdp_cyber_units.parquet --forget-buckets forget \
                 --retain-parquet wmdp_cyber_units.parquet --retain-buckets retain
  Arm B (ours):  --forget-parquet analysis_units_v1.parquet --forget-buckets offense \
                 --retain-parquet analysis_units_v1.parquet --retain-buckets dual defense

Saves the unlearned model to --out (HF save_pretrained). coeff 0 is meaningless here (no
steering) — the sweep driver treats coeff 0 as the base-model baseline and does NOT call this.

CPU smoke (mechanics only, no GPU, ~1 min):
  python scripts/unlearn_rmu.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --forget-parquet data/analysis_units_v1.parquet --forget-buckets offense \
    --retain-parquet data/analysis_units_v1.parquet --retain-buckets dual defense \
    --layer 2 --coeff 20 --steps 3 --batch-size 2 --max-tokens 64 --out /tmp/rmu_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--forget-parquet", required=True)
    p.add_argument("--forget-buckets", nargs="+", required=True)
    p.add_argument("--retain-parquet", required=True)
    p.add_argument("--retain-buckets", nargs="+", required=True)
    p.add_argument("--layer", type=int, default=7, help="layer whose activations are matched/steered")
    p.add_argument("--update-layers", type=int, default=3, help="# layers (ending at --layer) whose down_proj trains")
    p.add_argument("--coeff", type=float, required=True, help="steering coefficient (control-vector norm)")
    p.add_argument("--alpha", type=float, default=1200.0, help="retain-anchor weight")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output dir for the unlearned model")
    return p.parse_args(argv)


def load_texts(parquet: str, buckets: list[str], min_chars: int = 200) -> list[str]:
    """Pure: texts whose `bucket` is in `buckets`, from a units parquet (testable, no torch)."""
    import polars as pl

    df = pl.read_parquet(parquet)
    df = df.filter(pl.col("bucket").is_in(buckets) & (pl.col("n_chars") >= min_chars))
    return df["text"].to_list()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.coeff <= 0:
        sys.exit("coeff must be > 0 (coeff 0 = base model; the sweep driver handles that without RMU).")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    forget = load_texts(args.forget_parquet, args.forget_buckets)
    retain = load_texts(args.retain_parquet, args.retain_buckets)
    if not forget or not retain:
        sys.exit(f"empty corpus: forget={len(forget)} retain={len(retain)}")
    print(f"forget={len(forget)} retain={len(retain)} | layer={args.layer} coeff={args.coeff} "
          f"alpha={args.alpha} steps={args.steps}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(trainable: bool):
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        m.train(trainable)
        m.requires_grad_(False)
        return m

    updated, frozen = load(True), load(False)
    run_rmu(updated, frozen, tok, forget, retain, layer=args.layer, update_layers=args.update_layers,
            coeff=args.coeff, alpha=args.alpha, steps=args.steps, lr=args.lr,
            batch_size=args.batch_size, max_tokens=args.max_tokens)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    updated.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved unlearned model -> {out}", flush=True)
    return 0


def run_rmu(updated, frozen, tok, forget, retain, *, layer, update_layers, coeff,
            alpha, steps, lr, batch_size, max_tokens) -> None:
    """In-place RMU on `updated` (frozen = reference). No disk I/O — used by main() and the
    in-memory point runner (run_point.py) so no 16GB checkpoint is ever written."""
    import torch

    n_layers = updated.config.num_hidden_layers
    if not (0 <= layer < n_layers):
        raise ValueError(f"layer {layer} out of range 0..{n_layers - 1}")
    update_ids = list(range(max(0, layer - update_layers + 1), layer + 1))
    params = []
    for i in update_ids:
        w = updated.model.layers[i].mlp.down_proj.weight
        w.requires_grad_(True)
        params.append(w)
    opt = torch.optim.AdamW(params, lr=lr)
    device = next(updated.parameters()).device
    cv = torch.rand(1, 1, updated.config.hidden_size, device=device, dtype=torch.bfloat16)
    cv = cv / cv.norm() * coeff

    def acts(model, texts, idx):
        batch = texts[idx % len(texts): idx % len(texts) + batch_size] or texts[:batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(device)
        out = model(**enc, output_hidden_states=True)
        return out.hidden_states[layer + 1], enc["attention_mask"].unsqueeze(-1)

    mse = torch.nn.MSELoss()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        uf, mf = acts(updated, forget, step)
        unlearn = mse(uf * mf, cv.expand_as(uf) * mf)
        ur, mr = acts(updated, retain, step)
        with torch.no_grad():
            fr, _ = acts(frozen, retain, step)
        retain_loss = mse(ur * mr, fr.to(ur.dtype) * mr) * alpha
        (unlearn + retain_loss).backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"  step {step}: unlearn={unlearn.item():.4f} retain={retain_loss.item():.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
