#!/usr/bin/env python
"""Relearning finetune for the 2nd entanglement measure (anchored to WMDP App B.6 / Fig 15).

After RMU unlearning, finetune the model and watch offense recover. Two probes:
  * retain-only FT  — does offense revive from *legitimate* same-domain text alone? If so, the
    offensive and legitimate knowledge are entangled (the novel entanglement measure).
  * forget-set FT   — the paper's adversarial robustness test (Fig 15 showed cyber recovers).

Faithful-enough + tractable: LoRA finetune (standard for relearning/recovery attacks; full FT of a
7B with AdamW optimizer state barely fits alongside the model). Next-token LM loss on the corpus.

Self-contained (torch/peft only); shipped to the Flash worker and exec'd, like unlearn_rmu/eval_suite.
`train_steps` runs N gradient steps in place on a (LoRA-wrapped) model; the driver interleaves it
with offense/neighbor evals to produce a recovery-vs-steps curve.
"""

from __future__ import annotations


def lora_wrap(model):
    """Wrap with LoRA on attn+MLP projections; only LoRA params train."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, cfg)


def train_steps(model, tok, texts, *, steps, lr, batch_size, max_tokens) -> float:
    """N next-token LM gradient steps over `texts` (cycled). Returns mean loss. In place."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    device = next(model.parameters()).device
    model.train()
    total, n = 0.0, 0
    for step in range(steps):
        i = (step * batch_size) % max(1, len(texts))
        batch = texts[i:i + batch_size] or texts[:batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(device)
        opt.zero_grad(set_to_none=True)
        out = model(**enc, labels=enc["input_ids"])
        out.loss.backward()
        opt.step()
        total += float(out.loss.item()); n += 1
        if step % 20 == 0 or step == steps - 1:
            print(f"  relearn step {step}: loss={out.loss.item():.4f}", flush=True)
    return total / max(1, n)
