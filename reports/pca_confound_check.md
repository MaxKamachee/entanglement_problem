# PCA confound check + substrate-sharing robustness (NULL RESULT)

Ran 2026-06-08 on the existing **pre-supplement** hidden states
(`data/hidden_states/llama31_8b_three_way.parquet`: Llama-3.1-8B masked-mean pooled,
200 docs/bucket, layers 4/16/28, 4096-d). Reproduce with
`uv run python scripts/diag_pca_confound.py`.

## Check 1 — three-way separability after dropping top-k principal components

Per layer: remove the top-k unsupervised principal directions (SVD on the centered
600-doc matrix), then 5-fold stratified logistic probe (StandardScaler, seed 0).
Chance = 0.333. Cells: accuracy / macro-F1.

| layer | k=0 (base) | drop top-1 | drop top-2 | drop top-3 |
|------:|:----------:|:----------:|:----------:|:----------:|
| 4  | 0.935 / 0.935 | 0.938 / 0.938 | 0.888 / 0.888 | **0.332 / 0.335** (chance) |
| 16 | 0.932 / 0.931 | 0.932 / 0.932 | 0.795 / 0.794 | **0.208 / 0.215** (below chance) |
| 28 | 0.950 / 0.950 | 0.957 / 0.957 | 0.938 / 0.938 | **0.808 / 0.810** |

**Reading.** At the deepest layer (28) the offense/dual/defense distinction is
**distributed** — still 0.81 after removing the top 3 PCs. At layers 4/16 it
**collapses to chance**, i.e. shallow/mid-layer separability is concentrated in a few
dominant variance directions (consistent with topic/register/source-format confounds).
Dropping only the top-1 PC slightly *helps* everywhere → the single largest axis is
classification-irrelevant (magnitude/length); the confoundable structure is in PCs 2–3.
The deep-layer signal (where unlearning acts) is the robust one.

## Check 2 — substrate-domain sharing is NOT robust (null result)

Nearest dual-domain centroid (cosine) for each of the 200 offense + 200 defense docs,
layer 28, under three transforms of the same embeddings:

| domain | uncentered off / def | centered off / def | top-3-PC-removed off / def |
|---|---:|---:|---:|
| recon | **181** / 45 | 71 / 25 | 41 / 16 |
| crypto | 0 / 5 | 0 / 8 | **98** / 2 |
| os_internals | 5 / 76 | 4 / 29 | 19 / 40 |
| networking | 9 / 63 | 3 / 16 | 8 / 10 |
| web | 3 / 9 | 70 / 79 | 26 / 48 |
| architecture | 2 / 2 | 52 / 43 | 8 / **84** |

**Reading.** The headline "offense collapses onto the recon substrate (181/200)"
**does not survive**: it drops to 71 under mean-centering alone and 41 after top-3-PC
removal, while other domains reshuffle arbitrarily (crypto 0→98, architecture-defense
2→84). The assignment was riding low-dimensional global structure (shared mean + top
PCs = register/format/magnitude), not substrate semantics.

**Verdict: no substrate-sharing claim is supported by document-embedding geometry.**
Coarse bucket separation is real (Check 1, layer 28); fine-grained substrate-domain
attribution is confound-dominated. Establishing genuine shared-substrate reliance
requires confound-controlled or causal designs — the Fisher-overlap pilot
(parameter-space importance overlap) and ultimately the unlearning-tax experiment.

**Caveats.** Pre-supplement 200/bucket sample; thin dual-domain centroids (recon n=13,
architecture n=12 at layer 28); hard nearest-centroid assignment.
