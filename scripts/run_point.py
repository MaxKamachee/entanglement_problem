#!/usr/bin/env python
"""One sweep point IN MEMORY (GPU, pod): optional RMU then eval, NO checkpoint written.

Disk-safe replacement for the save-checkpoint-then-reload flow: the pod's 30GB container
disk can't hold two 8B copies, so we never serialize a model. Loads the model (+ a frozen
reference iff coeff>0), runs RMU in place (run_rmu), then evaluates the same in-memory model
(run_evals) and writes only the small JSON (+ SAQ-answers parquet). Reuses unlearn_rmu.run_rmu
and eval_suite.run_evals — no logic duplicated.

Called per (arm, coeff) by run_smoke_sweep.py as a fresh subprocess (clean GPU memory each).
coeff 0 = base model (no RMU). Example:
  python scripts/run_point.py --model meta-llama/Llama-3.1-8B-Instruct --tag ours_c100 \
    --coeff 100 --layer 7 --alpha 1200 --steps 500 \
    --forget-parquet data/analysis_units_v1.parquet --forget-buckets offense \
    --retain-parquet data/analysis_units_v1.parquet --retain-buckets dual defense \
    --out runs/smoke/ours_c100.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_suite  # noqa: E402
import unlearn_rmu  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--coeff", type=float, required=True)
    p.add_argument("--out", required=True)
    # RMU params (used iff coeff > 0)
    p.add_argument("--forget-parquet")
    p.add_argument("--forget-buckets", nargs="+")
    p.add_argument("--retain-parquet")
    p.add_argument("--retain-buckets", nargs="+")
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--update-layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    # eval params (consumed by eval_suite.run_evals)
    p.add_argument("--saq", default="data/saq_eval.parquet")
    p.add_argument("--n-mcq", type=int, default=200)
    p.add_argument("--n-mmlu", type=int, default=200)
    p.add_argument("--n-mbpp", type=int, default=50)
    p.add_argument("--n-saq", type=int, default=0)
    p.add_argument("--n-degen", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load():
        return AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    device_map="auto")

    model = load()
    if args.coeff > 0:
        model.train(True)
        model.requires_grad_(False)
        frozen = load()
        frozen.train(False)
        frozen.requires_grad_(False)
        forget = unlearn_rmu.load_texts(args.forget_parquet, args.forget_buckets)
        retain = unlearn_rmu.load_texts(args.retain_parquet, args.retain_buckets)
        print(f"RMU {args.tag}: forget={len(forget)} retain={len(retain)} "
              f"layer={args.layer} coeff={args.coeff} alpha={args.alpha} steps={args.steps}", flush=True)
        unlearn_rmu.run_rmu(model, frozen, tok, forget, retain, layer=args.layer,
                            update_layers=args.update_layers, coeff=args.coeff, alpha=args.alpha,
                            steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                            max_tokens=args.max_tokens)
        del frozen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    eval_suite.run_evals(model, tok, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
