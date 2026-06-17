# RMU unlearning smoke — shakedown result (2026-06-17)

**Status: pipeline validated end-to-end; numbers NOT scientifically interpretable.** A go/no-go
shakedown of the WMDP-vs-ours RMU harness on Llama-3.1-8B-Instruct. Only 2 of the planned points
completed before the run hit a pod disk limit and we stopped to redesign (in-memory, no
checkpoints): `wmdp_c0` (base) and `wmdp_c20` (RMU on WMDP forget/retain, steering coeff 20).
This is the **WMDP arm only, single mild coefficient** — not a Pareto, not the partition test.

## What completed (WMDP arm)

| axis | c0 (base) | c20 (RMU) | note |
|---|---:|---:|---|
| WMDP-cyber MCQ (offense, logprob) | 0.500 | 0.435 | mild offense drop; **the one clean signal** |
| MMLU (general canary, logprob) | 0.455 | 0.455 | general capability untouched |
| MBPP pass@1 (benign coding) | 0.04 | 0.04 | floored at base → uninformative |
| SAQ attack (Haiku-judged) | 0.157 | 0.10 | **invalid — see refusal confound** |
| SAQ defend | 0.24 | 0.20 | n=25, base low → noisy |
| SAQ substrate | 0.30 | 0.067 | n=30 → noisy (9→2 correct) |
| degeneracy guard | clean | clean | no broken text at either point |

## Three design-critical findings (the real value of the shakedown)

1. **Offense SAQ measures safety refusal, not capability.** The aligned Instruct model **refused
   46% of offensive SAQs** at base ("I can't provide information on hacking…"), graded as wrong;
   defend/substrate had **0% refusal**. Worse, RMU *lowered* the attack refusal rate (46% → 16%),
   so the attack-SAQ delta conflates capability change with alignment erosion. **→ Use the MCQ
   eval (logprob over A–D, refusal-robust; already built in `src/entanglement/mcq.py`) as the
   offense capability measure, not free-generation SAQ.** This is why WMDP/RMU use MCQ. SAQ stays
   useful only for benign/defensive axes (0% refusal).
2. **Eval is under-powered.** Base SAQ accuracy 0.1–0.3 on n=25–70/region → per-region deltas are
   within noise. MBPP floored at 0.04 (terse prompt + BPE decode). Real run needs larger n and a
   fixed MBPP prompt (or drop it).
3. **RMU is unstable above the low coeff regime.** c=20 held (retain mse ~2e-4); c=100 diverged
   (retain loss → 243). Real run needs gradient clipping + a low/dense coeff range (~[0,5,10,20,40]).

## What the one clean point shows (with heavy caveats)

At a mild coefficient, RMU on WMDP removed a little offense (WMDP-MCQ −0.065) with **no** general
collateral damage (MMLU flat) and no degeneracy. Consistent with the well-behaved low-coeff corner.
Nothing here speaks to the project's actual hypothesis (does the three-way partition beat a binary
split) — that requires the within-corpus partition test, not this confounded WMDP-arm single point.

## Implications for the real run
- Offense/defense capability via **MCQ** (refusal-robust); SAQ only for benign/defensive context.
- **Within-corpus arms** (ours-binary vs ours-three-tier) to test the partition cleanly; WMDP as a
  sanity anchor, not an arm.
- Grad clipping + low/dense coeffs; larger eval n; fix or drop MBPP.

Artifacts: `runs/smoke/wmdp_c0.json`, `wmdp_c20.json`, `*.saq_answers.graded.parquet`. Harness:
`scripts/{run_smoke_sweep,run_point,unlearn_rmu,eval_suite,grade_saq}.py`.
