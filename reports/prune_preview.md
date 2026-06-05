# External-only prune — counts + GPU-free geometry preview

## Pre/post-prune corpus counts (`data/analysis_units.parquet`)

| bucket | layer dropped | pre-prune | post-prune |
|---|---|---:|---:|
| offense | procedure (15,924) | 34,711 | **18,787** |
| defense | prose (825) | 3,579 | **2,754** |
| dual | — | 3,760 | **3,760** |
| **total** | | 42,050 | **25,301** |

Offense and defense are now register-symmetric: both median ~2,900 chars (was 132 vs 2,919).
Reproducible via `prune_framework_metadata: true` (configs/corpus.yaml) in `analysis_units.build_analysis_units`.

## GPU-free preview (existing hidden states, external-only, balanced ~99/bucket)

`scripts/diag_prune_preview.py` filters the already-extracted `llama31_8b_three_way.parquet` to
external-only and re-probes. **Small subset (99 offense docs from the original 200-sample) → directional,
not final; the rigorous test is a fresh 200-external/bucket re-extraction.**

| layer | 3-way: full → pruned | off-vs-def: full → pruned | within-dual topic (6-way) |
|---:|:--|:--|:--|
| 4 | 0.923 → 0.916 | 0.932 → 0.949 | 0.840 |
| 16 | 0.938 → 0.943 | 0.953 → 0.944 | 0.955 |
| 28 | 0.952 → 0.946 | 0.950 → 0.944 | 0.970 |

## Verdict

**The prune did NOT collapse separability.** Predicted 0.70–0.85; observed ~0.92–0.95 — essentially
unchanged. Removing the 132-char procedure register was correct *hygiene* (offense/defense are now
comparable) but it was **not** the main driver of separability. The dominant axis is **topic/subject
matter**, which the prune does not touch: within the dual bucket alone the model still recovers the 6
source-topics at **0.97**, and offense/dual/defense remain ~0.94 separable external-only.

**Implication for the contribution claim:** the geometry does **not** support a strong "buckets are
distinct in representation space ⇒ capabilities separable" claim — the separability is explained by topic,
not offensive/defensive valence. Separability should be retired as evidence for the thesis. The valence
question requires **topic-matched** comparison (offense vs defense *within the same topic*) and the
**unlearning tax** (capability eval), where topic is held constant. Do not proceed to MCQ/unlearning under
a separability-based contribution claim; proceed under a topic-controlled design.
