#!/usr/bin/env python
"""MCQ-based gradient-alignment entanglement metric — uniform across bio/cyber/chem, NO corpus.

Functional (not static) entanglement probe: for each domain build the answer-gradient
g = mean over items of ∇θ[ −log p(correct letter | question) ] on the base model, for three roles:
  offense  = WMDP-X questions
  neighbor = MMLU same-domain subjects (the legit knowledge unlearning would collaterally damage)
  general  = MMLU-all (off-domain baseline)
Then:
  entanglement(X) = cos(g_offense, g_neighbor) − cos(g_offense, g_general)
i.e. EXCESS same-domain gradient alignment over the shared MCQ/reasoning baseline (the control that
the static geometry probe lacked). High = pushing offense drags the same-domain neighbor with it.

Gradients are taken wrt the RMU layers' down_proj (default 5/6/7) to stay light and match where
unlearning acts. Works for chem with no forget corpus — only MCQs, which exist for all three domains.
Validate: cos-excess should rank cyber > bio, tracking the causal unlearning tax; chem then places
on the same scale. Resolves the chem corpus gap for the *predictive* lens.

CPU smoke (tiny model):
  python scripts/gradient_align.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --layers 1 --domains cyber --n-mcq 4 --n-neighbor 2 --n-general 4 --out /tmp/grad.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_suite  # noqa: E402  (DOMAIN_EVALS, LETTERS, format_mcq_prompt)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--domains", nargs="+", default=["bio", "cyber", "chem"],
                   choices=sorted(eval_suite.DOMAIN_EVALS))
    p.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7],
                   help="down_proj layers to take gradients wrt (match RMU)")
    p.add_argument("--n-mcq", type=int, default=150)
    p.add_argument("--n-neighbor", type=int, default=80)
    p.add_argument("--n-general", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    model.requires_grad_(False)
    params = []
    for i in args.layers:
        w = model.model.layers[i].mlp.down_proj.weight
        w.requires_grad_(True)
        params.append(w)
    device = next(model.parameters()).device

    def letter_id(ltr):
        ids = tok.encode(ltr, add_special_tokens=False) or tok.encode(" " + ltr, add_special_tokens=False)
        return ids[0]

    def role_grad(items):
        """mean over items of grad(-log p(correct letter)) wrt params, returned as one flat vector."""
        acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
        n = 0
        for q, choices, ans, subj in items:
            enc = tok(eval_suite.format_mcq_prompt(q, choices, subj), return_tensors="pt").to(device)
            logits = model(**enc).logits[0, -1]
            loss = -torch.log_softmax(logits.float(), dim=-1)[letter_id(eval_suite.LETTERS[ans])]
            g = torch.autograd.grad(loss, params, retain_graph=False)
            for a, gi in zip(acc, g):
                a += gi.detach().float()
            n += 1
        return torch.cat([(a / max(1, n)).flatten() for a in acc])

    def items_from(ds_args, n, subject):
        ds = load_dataset(*ds_args, split=f"test[:{n}]")
        return [(r["question"], r["choices"], int(r["answer"]),
                 subject if subject is not None else r.get("subject", "knowledge")) for r in ds]

    cos = torch.nn.functional.cosine_similarity
    result = {"model": args.model, "layers": args.layers, "domains": {}}
    for dom in args.domains:
        de = eval_suite.DOMAIN_EVALS[dom]
        g_off = role_grad(items_from(("cais/wmdp", de["offense"]), args.n_mcq, de["offense_subject"]))
        nbr_items = []
        for subj in de["neighbors"]:
            nbr_items += items_from(("cais/mmlu", subj), args.n_neighbor, subj)
        g_nbr = role_grad(nbr_items)
        g_gen = role_grad(items_from(("cais/mmlu", "all"), args.n_general, None))
        c_nbr = float(cos(g_off, g_nbr, dim=0))
        c_gen = float(cos(g_off, g_gen, dim=0))
        result["domains"][dom] = {"cos_off_neighbor": round(c_nbr, 4),
                                  "cos_off_general": round(c_gen, 4),
                                  "entanglement": round(c_nbr - c_gen, 4)}
        print(f"{dom}: cos(off,nbr)={c_nbr:.4f} cos(off,gen)={c_gen:.4f} "
              f"entanglement(excess)={c_nbr - c_gen:+.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    ranked = sorted(result["domains"].items(), key=lambda kv: -kv[1]["entanglement"])
    print("ranking (most→least entangled):", [d for d, _ in ranked])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
