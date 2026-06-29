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
    """Mean per-token logprob (teacher-forced), (B,). Memory-efficient: no full-vocab float32
    softmax — gather the label logit and subtract logsumexp (a reduction)."""
    import torch
    logits = model(**enc).logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    sel = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # (B,T)
    lse = torch.logsumexp(logits, dim=-1)                       # (B,T) reduction, no (B,T,V) copy
    mask = enc["attention_mask"][:, 1:].to(logits.dtype)
    return ((sel - lse) * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def run_npo(model, tok, forget, retain, *, steps, lr, beta, retain_weight, batch_size, max_tokens):
    """LoRA NPO unlearning, then merge into weights. Returns the merged model.

    Memory-managed for 128k-vocab models on ~40GB GPUs: efficient logprob, capped batch/seqlen,
    and SEPARATE backward on the forget vs retain terms so only one forward graph is alive at a time."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    bs = min(2, batch_size)            # NPO holds a full-vocab forward graph -> keep batch small
    mt = min(256, max_tokens)
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def enc_of(texts, idx):
        batch = texts[idx % len(texts): idx % len(texts) + bs] or texts[:bs]
        return tok(batch, return_tensors="pt", padding=True, truncation=True,
                   max_length=mt).to(device)

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        # --- forget term: backward immediately so its graph is freed before the retain forward ---
        ef = enc_of(forget, step)
        lp_theta = _seq_logprob(model, ef)
        with torch.no_grad(), model.disable_adapter():
            lp_ref = _seq_logprob(model, ef)
        loss_npo = -(2.0 / beta) * F.logsigmoid(-beta * (lp_theta - lp_ref)).mean()
        loss_npo.backward()
        # --- retain term (separate graph) ---
        er = enc_of(retain, step)
        loss_ret = retain_weight * model(**er, labels=er["input_ids"]).loss
        loss_ret.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            print(f"  npo step {step}: npo={float(loss_npo.detach()):.4f} "
                  f"retain={float(loss_ret.detach()):.4f}", flush=True)

    return model.merge_and_unload()
