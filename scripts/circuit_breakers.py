#!/usr/bin/env python
"""Circuit Breakers (Representation Rerouting) unlearning — 2nd method for method-independence.

Faithful reimplementation of Zou et al. 2024 ("Improving Alignment and Robustness with Circuit
Breakers", GraySwanAI/circuit-breakers): train LoRA adapters so that on FORGET data the model's
hidden states at target layers are rerouted *away* from their original direction (rerouting loss =
ReLU(cosine(h_lora, h_orig))), while on RETAIN data they are preserved (L2 to the original reps).
The retain/rerouting coefficients are scheduled linearly over training. After training, the LoRA is
merged into the weights — so the safeguard "lives in the weights the attacker controls" (the paper's
point), and our durability eval then finetunes on top of it.

Self-contained (torch + peft). `run_cb` mirrors `unlearn_rmu.run_rmu` so the durability harness
(run_relearn.py) dispatches by `--method`. Original reps are obtained by disabling the LoRA adapters
(no separate frozen copy needed).

NOTE on hyperparameters: layer set / alpha / lr should be matched to the reference repo for a
production result; defaults here are reasonable for a 7-8B model and validated only for mechanics.

CPU smoke (tiny model):
  python scripts/circuit_breakers.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --forget-parquet data/wmdp_cyber_units.parquet --forget-buckets forget \
    --retain-parquet data/wikitext_units.parquet --retain-buckets retain \
    --target-layers 1 --steps 3 --batch-size 2 --max-tokens 64 --out /tmp/cb_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import unlearn_rmu  # noqa: E402  (reuse load_texts)


def run_cb(model, tok, forget, retain, *, target_layers, steps, lr, alpha,
           batch_size, max_tokens):
    """In-LoRA circuit-breaker training, then merge into weights. Returns the merged model."""
    import torch
    from peft import LoraConfig, get_peft_model

    layers = list(target_layers)
    cfg = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers,
    )
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device

    def encode(texts, idx):
        batch = texts[idx % len(texts): idx % len(texts) + batch_size] or texts[:batch_size]
        return tok(batch, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_tokens).to(device)

    for step in range(steps):
        progress = (step + 1) / steps
        c_cb = alpha * progress             # ramp rerouting up
        c_ret = alpha * (1.0 - progress)    # ramp retain down
        opt.zero_grad(set_to_none=True)

        # FORGET: reroute (push LoRA reps off the original direction)
        enc_f = encode(forget, step)
        out_cb = model(**enc_f, output_hidden_states=True)
        with torch.no_grad(), model.disable_adapter():
            out_orig_f = model(**enc_f, output_hidden_states=True)
        mf = enc_f["attention_mask"]
        loss_cb = 0.0
        for L in layers:
            cos = torch.nn.functional.cosine_similarity(
                out_cb.hidden_states[L + 1], out_orig_f.hidden_states[L + 1].detach(), dim=-1)
            loss_cb = loss_cb + (torch.relu(cos) * mf).sum() / mf.sum().clamp(min=1)
        loss_cb = loss_cb / len(layers)

        # RETAIN: preserve (keep LoRA reps near original)
        enc_r = encode(retain, step)
        out_ret = model(**enc_r, output_hidden_states=True)
        with torch.no_grad(), model.disable_adapter():
            out_orig_r = model(**enc_r, output_hidden_states=True)
        mr = enc_r["attention_mask"].unsqueeze(-1)
        loss_ret = 0.0
        for L in layers:
            d = out_ret.hidden_states[L + 1] - out_orig_r.hidden_states[L + 1].detach()
            loss_ret = loss_ret + ((d * d).sum(-1, keepdim=True) * mr).sum() / mr.sum().clamp(min=1)
        loss_ret = loss_ret / len(layers)

        loss = c_cb * loss_cb + c_ret * loss_ret
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"  cb step {step}: cb={float(loss_cb.detach()):.4f} "
                  f"retain={float(loss_ret.detach()):.4f} (c_cb={c_cb:.1f} c_ret={c_ret:.1f})", flush=True)

    return model.merge_and_unload()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--forget-parquet", required=True)
    p.add_argument("--forget-buckets", nargs="+", required=True)
    p.add_argument("--retain-parquet", required=True)
    p.add_argument("--retain-buckets", nargs="+", required=True)
    p.add_argument("--target-layers", type=int, nargs="+", default=[10, 20])
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.requires_grad_(False)
    forget = unlearn_rmu.load_texts(args.forget_parquet, args.forget_buckets)
    retain = unlearn_rmu.load_texts(args.retain_parquet, args.retain_buckets)
    print(f"CB: forget={len(forget)} retain={len(retain)} layers={args.target_layers}", flush=True)
    model = run_cb(model, tok, forget, retain, target_layers=args.target_layers, steps=args.steps,
                   lr=args.lr, alpha=args.alpha, batch_size=args.batch_size, max_tokens=args.max_tokens)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print(f"saved CB model -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
