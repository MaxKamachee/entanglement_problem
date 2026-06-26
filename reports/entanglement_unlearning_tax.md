# Entanglement unlearning-tax: bio vs cyber, wikitext vs substrate retain

**Question.** How much legitimate same-domain ability do you lose to remove a given amount of offensive ability (RMU), per domain — and does a targeted (same-domain substrate) retain set beat the blunt (wikitext) one? Quantifies WMDP Figure 11.

**Setup.** `HuggingFaceH4/zephyr-7b-beta`, RMU layer 7 / α=1200, MLP down_proj on layers 5-7, 500 steps, coeff sweep. Forget = WMDP-{domain}; retain = wikitext vs WMDP-{domain} substrate. Eval (lm-eval-harness 0-shot, paper-comparable): offense = WMDP MCQ (↓ good); neighbor = same-domain MMLU (college_X far + near-X canary, ↑ kept); general MMLU canary; degeneracy guard. Normalized to per-domain base.

## Base model (validation vs paper Table 2 / Fig 11)
| domain | WMDP offense | neighbor (mean) | general MMLU |
|---|---|---|---|
| bio | 0.562 | 0.590 | 0.537 |
| cyber | 0.403 | 0.580 | 0.537 |

## Sweep (normalized vs base; ✗ = degeneracy-flagged)
| arm | coeff | offense_removed | neighbor_kept | general_kept | coherent |
|---|--:|--:|--:|--:|:--:|
| bio_wikitext | 2 | 0.416 | 0.949 | 0.972 | ✓ |
| bio_wikitext | 4 | 0.488 | 0.807 | 0.977 | ✓ |
| bio_wikitext | 6.5 | 0.544 | 0.868 | 0.967 | ✓ |
| bio_wikitext | 10 | 0.424 | 0.863 | 0.977 | ✓ |
| bio_wikitext | 20 | 0.496 | 0.753 | 0.902 | ✗ |
| bio_substrate | 2 | 0.048 | 0.983 | 0.986 | ✓ |
| bio_substrate | 4 | 0.000 | 1.006 | 0.967 | ✓ |
| bio_substrate | 6.5 | 0.040 | 1.017 | 0.981 | ✓ |
| bio_substrate | 10 | 0.152 | 0.978 | 0.958 | ✓ |
| bio_substrate | 20 | 0.096 | 0.930 | 0.953 | ✓ |
| cyber_wikitext | 2 | 0.803 | 0.862 | 0.944 | ✓ |
| cyber_wikitext | 4 | 0.869 | 0.655 | 0.935 | ✓ |
| cyber_wikitext | 6.5 | 1.049 | 0.612 | 0.944 | ✓ |
| cyber_wikitext | 10 | 1.049 | 0.466 | 0.963 | ✗ |
| cyber_wikitext | 20 | 0.934 | 0.509 | 0.540 | ✗ |
| cyber_substrate | 2 | 0.000 | 0.966 | 0.935 | ✓ |
| cyber_substrate | 4 | 0.443 | 0.948 | 0.930 | ✓ |
| cyber_substrate | 6.5 | 0.000 | 1.000 | 0.916 | ✓ |
| cyber_substrate | 10 | 0.754 | 0.922 | 0.940 | ✓ |
| cyber_substrate | 20 | 0.770 | 0.819 | 0.963 | ✓ |

## Headline: tax + asymmetry
Tax = same-domain neighbor *lost* per unit offense removed (lower = more precise; coherent points only). An arm that never removes ≥0.2 offense is marked **suppressed** (the retain anchor blocks unlearning) — its 'tax' is not meaningful.

| domain | wikitext tax | substrate tax | substrate effect |
|---|--:|--:|--|
| bio | 0.276 | — | **suppressed** (max offense removed 0.15) |
| cyber | 0.327 | 0.163 | halves tax (gain 0.164) |

- **Cross-domain asymmetry (wikitext RMU):** cyber tax 0.327 vs bio tax 0.276 → cyber is **more entangled** (loses 0.050 more same-domain neighbor per unit offense removed).
- **Precision payoff is domain-specific:** in **cyber** the substrate retain set cuts the tax from 0.327 to 0.163 while staying coherent (a Pareto point wikitext can't reach); in **bio** the substrate retain instead **suppresses unlearning** (max offense removed 0.15). The targeted retain set helps precisely in the more-entangled domain — and is counterproductive in the less-entangled one.

## Figures

![entanglement_pareto](figures/entanglement_pareto.png)

![entanglement_fig11](figures/entanglement_fig11.png)