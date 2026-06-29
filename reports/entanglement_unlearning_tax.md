# Entanglement unlearning-tax (seed-aggregated)

Tax = legit same-domain ability lost per unit offense removed (per-seed slope, mean ± SE). Methods: RMU, Circuit Breakers.

## Base model
| domain | WMDP offense | neighbor (mean) | general MMLU |
|---|--:|--:|--:|
| bio | 0.550 | 0.685 | 0.450 |
| cyber | 0.465 | 0.590 | 0.450 |

## Tax ± SE (over seeds) by method × domain × retain
| method | domain | retain | tax ± SE | seeds | max off removed |
|---|---|---|--:|--:|--:|
| RMU | bio | wikitext | 0.158 ± 0.000 | 1 | 0.500 |
| RMU | cyber | wikitext | 0.206 ± 0.000 | 1 | 0.628 |
| RMU | cyber | substrate | -0.012 ± 0.000 | 1 | 0.349 |
| Circuit Breakers | cyber | wikitext | 0.000 ± 0.000 | 1 | 0.023 |

## Cross-domain asymmetry (wikitext arm)
- **RMU:** cyber 0.206±0.000 vs bio 0.158±0.000 → gap +0.049 (beyond combined SE).

## Figures

![entanglement_pareto](figures/entanglement_pareto.png)

![entanglement_fig11](figures/entanglement_fig11.png)