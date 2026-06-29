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
import circuit_breakers  # noqa: E402
import eval_suite  # noqa: E402
import npo  # noqa: E402
import unlearn_rmu  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["rmu", "circuit_breakers", "npo"], default="rmu",
                   help="strength = --coeff (rmu) / --cb-alpha (cb) / --npo-steps (npo); 0 = base")
    # corpora (used iff strength > 0)
    p.add_argument("--forget-parquet")
    p.add_argument("--forget-buckets", nargs="+")
    p.add_argument("--retain-parquet")
    p.add_argument("--retain-buckets", nargs="+")
    # RMU params
    p.add_argument("--coeff", type=float, default=0.0)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--update-layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    # circuit-breaker params (--cb-alpha is the CB strength axis)
    p.add_argument("--cb-target-layers", type=int, nargs="+", default=[10, 20])
    p.add_argument("--cb-alpha", type=float, default=0.0)
    p.add_argument("--cb-steps", type=int, default=300)
    p.add_argument("--cb-lr", type=float, default=1e-4)
    # NPO params (--npo-steps is the NPO strength axis)
    p.add_argument("--npo-steps", type=int, default=0)
    p.add_argument("--npo-lr", type=float, default=2e-4)
    p.add_argument("--npo-beta", type=float, default=0.1)
    p.add_argument("--npo-retain-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    # eval params (consumed by eval_suite.run_evals)
    p.add_argument("--domain", default="cyber",
                   help="eval domain: selects offense WMDP subject + MMLU neighbor subjects")
    p.add_argument("--saq", default="data/saq_eval.parquet")
    p.add_argument("--n-mcq", type=int, default=200)
    p.add_argument("--n-neighbor", type=int, default=100)
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

    strength = {"rmu": args.coeff, "circuit_breakers": args.cb_alpha,
                "npo": args.npo_steps}[args.method]
    model = load()
    if strength > 0:
        forget = unlearn_rmu.load_texts(args.forget_parquet, args.forget_buckets)
        retain = unlearn_rmu.load_texts(args.retain_parquet, args.retain_buckets)
        print(f"{args.method} {args.tag}: forget={len(forget)} retain={len(retain)} "
              f"strength={strength}", flush=True)
        if args.method == "rmu":
            model.train(True)
            model.requires_grad_(False)
            frozen = load()
            frozen.train(False)
            frozen.requires_grad_(False)
            unlearn_rmu.run_rmu(model, frozen, tok, forget, retain, layer=args.layer,
                                update_layers=args.update_layers, coeff=args.coeff, alpha=args.alpha,
                                steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                                max_tokens=args.max_tokens)
            del frozen
        elif args.method == "circuit_breakers":   # LoRA RR merged into weights
            model.requires_grad_(False)
            model = circuit_breakers.run_cb(model, tok, forget, retain,
                                            target_layers=args.cb_target_layers, steps=args.cb_steps,
                                            lr=args.cb_lr, alpha=args.cb_alpha,
                                            batch_size=args.batch_size, max_tokens=args.max_tokens)
        else:  # npo — LoRA NPO merged into weights
            model.requires_grad_(False)
            model = npo.run_npo(model, tok, forget, retain, steps=args.npo_steps, lr=args.npo_lr,
                                beta=args.npo_beta, retain_weight=args.npo_retain_weight,
                                batch_size=args.batch_size, max_tokens=args.max_tokens)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.eval()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    eval_suite.run_evals(model, tok, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
