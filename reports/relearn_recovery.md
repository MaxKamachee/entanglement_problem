# Relearning: offense recovery under finetuning (2nd entanglement measure)

After RMU-unlearn (step 0), LoRA-finetune and watch offense (WMDP MCQ) recover. `relforget` = re-teach the forbidden corpus (adversarial, paper Fig 15); `relretain` = finetune on **legitimate same-domain** text only — offense reviving from that is the entanglement signal. Recovery normalized to base offense where available.

| condition | offense@unlearn | offense@final | neighbor@unlearn→final | offense recovered | frac→base |
|---|--:|--:|--:|--:|--:|
| bio_relforget | 0.293 | 0.480 | 0.442→0.566 | 0.187 | 0.694 |
| cyber_relforget | 0.260 | 0.427 | 0.355→0.585 | 0.167 | 1.170 |
| bio_relretain | 0.287 | 0.477 | 0.435→0.596 | 0.190 | 0.689 |
| cyber_relretain | 0.273 | 0.397 | 0.340→0.570 | 0.123 | 0.955 |

## Headline
**Durability, not absolute recovery, is the entanglement metric** — absolute offense-recovered is confounded by how much offense each domain started with (bio's base is higher). The right question is whether the unlearning *stuck*: fraction of base offense recovered (→1.0 = fully reverses) and offense remaining below base.

- **Entanglement probe (retain-only FT):** fraction of base recovered — bio 0.689, cyber 0.955 → recovers more completely in **cyber** (absolute, for reference: bio 0.190, cyber 0.123).
- **Adversarial recovery (forget FT):** fraction of base recovered — bio 0.694, cyber 1.170 → recovers more completely in **cyber** (absolute, for reference: bio 0.187, cyber 0.167).
- **Interpretation:** offense reviving from legitimate same-domain text alone is the entanglement signal. Cyber recovers 0.955 of base from retain-only FT vs bio 0.689 → SUPPORTS cyber being more entangled (consistent with the unlearning-tax result + WMDP Fig 15). Caveat: cyber's base offense is near chance, so its small headroom makes the fraction noisier (can exceed 1.0).

## Figure

![relearn_recovery](figures/relearn_recovery.png)