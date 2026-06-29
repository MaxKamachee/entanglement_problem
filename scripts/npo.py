#!/usr/bin/env python
"""NPO (Negative Preference Optimization) unlearning — reliable second method for the tax sweep.

Zhang et al. 2024. Forget loss pushes the model's sequence-likelihood on the forget set BELOW the
reference model's, gently (bounded, unlike raw gradient ascent):
    L_forget = -(2/β) * mean[ log σ( -β (logπ_θ(x) - logπ_ref(x)) ) ]
plus a retain cross-entropy term to preserve utility. Trained via LoRA; the reference logprob is the
SAME model with adapters disabled (no separate frozen copy). After training the LoRA is merged.

Unlike circuit breakers (a subtle cosine-rerouting objective that our reimpl couldn't drive), NPO is
a direct likelihood push that LoRA follows reliably — so it actually removes offense. The unlearning
*strength* axis is the number of training steps.

Self-contained (torch + peft). `run_npo` mirrors unlearn_rmu.run_rmu / circuit_breakers.run_cb so the
harness dispatches by --method.
"""

from __future__ import annotations


def _seq_logprob(model, enc):
    """Mean per-token logprob of each sequence (teacher-forced), shape (B,)."""
    import torch
    out = model(**enc)
    logits = out.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = enc["attention_mask"][:, 1:].float()
    return (logp * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def run_npo(model, tok, forget, retain, *, steps, lr, beta, retain_weight, batch_size, max_tokens):
    """LoRA NPO unlearning, then merge into weights. Returns the merged model."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device

    def enc_of(texts, idx):
        batch = texts[idx % len(texts): idx % len(texts) + batch_size] or texts[:batch_size]
        return tok(batch, return_tensors="pt", padding=True, truncation=True,
                   max_length=max_tokens).to(device)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        # NPO forget loss
        ef = enc_of(forget, step)
        lp_theta = _seq_logprob(model, ef)
        with torch.no_grad(), model.disable_adapter():
            lp_ref = _seq_logprob(model, ef)
        logratio = lp_theta - lp_ref                       # (B,)
        loss_npo = -(2.0 / beta) * F.logsigmoid(-beta * logratio).mean()
        # retain cross-entropy (preserve utility)
        er = enc_of(retain, step)
        loss_ret = model(**er, labels=er["input_ids"]).loss
        loss = loss_npo + retain_weight * loss_ret
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"  npo step {step}: npo={float(loss_npo.detach()):.4f} "
                  f"retain={float(loss_ret.detach()):.4f}", flush=True)

    return model.merge_and_unload()
