#!/usr/bin/env python
"""One relearning point IN MEMORY (GPU, pod): RMU-unlearn -> LoRA-finetune -> recovery curve.

2nd entanglement measure (WMDP App B.6 / Fig 15): after unlearning, finetune and watch offense
recover. relearn corpus = forget (adversarial robustness) or retain (the novel entanglement probe:
does offense revive from *legitimate* same-domain text alone?).

No checkpoints written: load model (+frozen), RMU in place, LoRA-wrap, then finetune in blocks,
evaluating offense + same-domain neighbor after each block -> JSON with the recovery series.

Example (pod):
  python scripts/run_relearn.py --model HuggingFaceH4/zephyr-7b-beta --domain cyber \
    --coeff 6.5 --forget-parquet data/wmdp_cyber_units.parquet --forget-buckets forget \
    --retain-parquet data/wikitext_units.parquet --retain-buckets retain \
    --relearn-parquet data/wmdp_cyber_units.parquet --relearn-buckets retain \
    --relearn-steps 200 --eval-every 50 --out runs/relearn/cyber_relretain.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_suite  # noqa: E402
import relearn  # noqa: E402
import unlearn_rmu  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--domain", required=True, choices=sorted(eval_suite.DOMAIN_EVALS))
    p.add_argument("--out", required=True)
    # unlearn (RMU)
    p.add_argument("--coeff", type=float, required=True)
    p.add_argument("--forget-parquet", required=True)
    p.add_argument("--forget-buckets", nargs="+", required=True)
    p.add_argument("--retain-parquet", required=True)
    p.add_argument("--retain-buckets", nargs="+", required=True)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--update-layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    # relearn (LoRA finetune)
    p.add_argument("--relearn-parquet", required=True)
    p.add_argument("--relearn-buckets", nargs="+", required=True)
    p.add_argument("--relearn-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--relearn-lr", type=float, default=1e-4)
    # eval
    p.add_argument("--n-mcq", type=int, default=300)
    p.add_argument("--n-neighbor", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def quick_eval(model, tok, domain, n_mcq, n_nbr) -> dict:
    """Fast offense + headline-neighbor accuracy, reusing eval_suite primitives."""
    from datasets import load_dataset
    de = eval_suite.DOMAIN_EVALS[domain]
    ds = load_dataset("cais/wmdp", de["offense"], split=f"test[:{n_mcq}]")
    off = eval_suite._mcq_accuracy(model, tok, [(r["question"], r["choices"], int(r["answer"]),
                                                 de["offense_subject"]) for r in ds])
    accs = {}
    for subj in de["neighbors"]:
        ds = load_dataset("cais/mmlu", subj, split=f"test[:{n_nbr}]")
        accs[subj] = eval_suite._mcq_accuracy(model, tok, [(r["question"], r["choices"],
                                              int(r["answer"]), subj) for r in ds])
    return {"offense_mcq": round(off, 4),
            "neighbor_mmlu": round(sum(accs.values()) / max(1, len(accs)), 4),
            "neighbor_by_subject": {k: round(v, 4) for k, v in accs.items()}}


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(trainable):
        m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
        m.train(trainable); m.requires_grad_(False)
        return m

    # 1) RMU unlearn
    model = load(True)
    frozen = load(False)
    forget = unlearn_rmu.load_texts(args.forget_parquet, args.forget_buckets)
    retain = unlearn_rmu.load_texts(args.retain_parquet, args.retain_buckets)
    print(f"RMU {args.domain} coeff={args.coeff}: forget={len(forget)} retain={len(retain)}", flush=True)
    unlearn_rmu.run_rmu(model, frozen, tok, forget, retain, layer=args.layer,
                        update_layers=args.update_layers, coeff=args.coeff, alpha=args.alpha,
                        steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                        max_tokens=args.max_tokens)
    del frozen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    series = [{"relearn_steps": 0, **quick_eval(model, tok, args.domain, args.n_mcq, args.n_neighbor)}]
    print(f"after unlearn: {series[-1]}", flush=True)

    # 2) LoRA relearn in blocks, eval after each
    model = relearn.lora_wrap(model)
    relearn_texts = unlearn_rmu.load_texts(args.relearn_parquet, args.relearn_buckets)
    done = 0
    while done < args.relearn_steps:
        block = min(args.eval_every, args.relearn_steps - done)
        relearn.train_steps(model, tok, relearn_texts, steps=block, lr=args.relearn_lr,
                            batch_size=args.batch_size, max_tokens=args.max_tokens)
        done += block
        model.eval()
        series.append({"relearn_steps": done, **quick_eval(model, tok, args.domain, args.n_mcq, args.n_neighbor)})
        print(f"after {done} relearn steps: {series[-1]}", flush=True)

    out = {"model": args.model, "domain": args.domain, "coeff": args.coeff,
           "relearn_parquet": args.relearn_parquet, "relearn_buckets": args.relearn_buckets,
           "relearn_steps": args.relearn_steps, "series": series}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
