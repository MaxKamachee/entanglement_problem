# Entanglement unlearning-tax (RMU + Circuit Breakers)

Safety tax = legitimate same-domain ability lost per unit offense removed. Methods: RMU.

## Base model
| domain | WMDP offense | neighbor (mean) | general MMLU |
|---|--:|--:|--:|
| bio | 0.562 | 0.590 | 0.537 |
| cyber | 0.403 | 0.580 | 0.537 |

## Tax by method × domain × retain (slope; coherent points)
| method | domain | retain | tax | max offense removed |
|---|---|---|--:|--:|
| RMU | bio | wikitext | 0.276 | 0.544 |
| RMU | bio | substrate | 0.280 | 0.152 |
| RMU | cyber | wikitext | 0.327 | 1.049 |
| RMU | cyber | substrate | 0.163 | 0.770 |

## Cross-domain asymmetry (wikitext arm)
- **RMU:** bio tax 0.276 vs cyber 0.327 → more entangled: **cyber**.

## Figures

![entanglement_pareto](figures/entanglement_pareto.png)

![entanglement_fig11](figures/entanglement_fig11.png)