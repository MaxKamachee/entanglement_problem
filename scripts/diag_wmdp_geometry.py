#!/usr/bin/env python
"""WMDP-cyber geometry calibration — binary forget/retain separability under PC removal.

Method calibration + possible benchmark critique, NOT entanglement evidence and NOT a
fresh geometry attempt on our corpus. Runs the SAME protocol as our PCA confound check
(reports/pca_confound_check.md) on WMDP-cyber's published forget/retain text corpora,
head-to-head with our corpus's offense-vs-defense split.

Protocol (identical to ours): Llama-3.1-8B masked-mean pooled hidden states, layers
4/16/28, 200 docs/split (seed 0), 5-fold stratified logistic probe (StandardScaler,
seed 0). Binary task -> chance = 0.5 (our original check was 3-way, chance 0.333).
Confound removal: k=0 is mean-centering-only; drop-top-1/2/3 removes the top
unsupervised principal directions of the SAME 2-class matrix being probed.

Protocol caveat (stated upfront, applies to BOTH columns equally): in a 2-class
matrix, the between-class direction is typically among the top PCs, so within-task
PC removal partially removes class signal *mechanically*. The comparison is still
apples-to-apples — both splits get the identical treatment — and our corpus is also
shown under global-PC removal (PCs from the 600-doc 3-way matrix) for context.

Inputs:  data/hidden_states/wmdp_cyber_forget_retain.parquet  (GPU step — see
         scripts/prep_wmdp_units.py docstring for the one-shot pod command)
         data/hidden_states/llama31_8b_three_way.parquet      (existing)
Output:  reports/wmdp_cyber_geometry_baseline.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
WMDP_HIDDEN = ROOT / "data" / "hidden_states" / "wmdp_cyber_forget_retain.parquet"
OURS_HIDDEN = ROOT / "data" / "hidden_states" / "llama31_8b_three_way.parquet"
REPORT = ROOT / "reports" / "wmdp_cyber_geometry_baseline.md"
LAYERS = [4, 16, 28]
KS = [0, 1, 2, 3]


def remove_top_pcs(X: np.ndarray, k: int, basis: np.ndarray | None = None) -> np.ndarray:
    """Center X, then project out the top-k right-singular directions of `basis` (default X)."""
    Xc = X - X.mean(0)
    if k == 0:
        return Xc
    B = basis - basis.mean(0) if basis is not None else Xc
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt[:k]
    return Xc - (Xc @ V.T) @ V


def probe(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    r = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
    return r["test_accuracy"].mean(), r["test_f1_macro"].mean()


def ladder(h: pl.DataFrame, labels: tuple[str, str], label_col: str = "corpus_label") -> dict:
    """{layer: [(acc, f1) for k in KS]} for a binary split, within-task PCs."""
    out: dict[int, list[tuple[float, float]]] = {}
    for layer in LAYERS:
        s = h.filter((pl.col("layer") == layer) & pl.col(label_col).is_in(labels))
        X = np.array(s["embedding"].to_list())
        y = s[label_col].to_numpy()
        out[layer] = [probe(remove_top_pcs(X, k), y) for k in KS]
    return out


def fmt_row(layer: int, cells: list[tuple[float, float]]) -> str:
    return f"| {layer} | " + " | ".join(f"{a:.3f} / {f:.3f}" for a, f in cells) + " |"


def main() -> None:
    if not WMDP_HIDDEN.exists():
        sys.exit(f"{WMDP_HIDDEN.relative_to(ROOT)} not found — run the GPU extraction first "
                 "(see scripts/prep_wmdp_units.py docstring for the one-shot pod command).")
    wmdp_h = pl.read_parquet(WMDP_HIDDEN)
    ours_h = pl.read_parquet(OURS_HIDDEN)
    n_wmdp = dict(wmdp_h.filter(pl.col("layer") == LAYERS[0]).group_by("corpus_label").len().iter_rows())

    wmdp = ladder(wmdp_h, ("forget", "retain"))
    ours = ladder(ours_h, ("offense", "defense"))

    # context panel: ours under GLOBAL PCs (computed on the full 600-doc 3-way matrix)
    ctx_rows = []
    for layer in LAYERS:
        s_all = ours_h.filter(pl.col("layer") == layer)
        X_all = np.array(s_all["embedding"].to_list())
        lab = s_all["corpus_label"].to_numpy()
        m = np.isin(lab, ("offense", "defense"))
        cells = []
        for k in KS:
            Xp = remove_top_pcs(X_all, k)[m]
            cells.append(probe(Xp, lab[m]))
        ctx_rows.append((layer, cells))

    head = " | ".join(["k=0 (center only)", "drop top-1", "drop top-2", "drop top-3"])
    L = ["# WMDP-cyber geometry baseline — binary forget/retain separability under PC removal", ""]
    L.append("**Scope.** Method calibration + possible benchmark critique. Binary forget/retain "
             "split; does NOT test the three-way substrate-entanglement hypothesis (WMDP has no "
             "substrate bucket). Separability ≠ entanglement.")
    L.append("")
    L.append("**Data.** `cais/wmdp-corpora` @ `daf89fa9b618b63a624228061a9cebacca88009c` "
             "(2024-04-25): cyber-forget 1,000 docs / cyber-retain 4,473 docs (sha256-pinned, "
             "see `scripts/prep_wmdp_units.py`). Normalized with our exact unit pipeline "
             "(clean_text + resegment ~3k chars) → forget 3,917 / retain 14,738 units; sampled "
             f"{n_wmdp} per split (seed 0). Comparison: hazardous-cyber forget text vs retain text.")
    L.append("")
    L.append("**Protocol.** Identical to `reports/pca_confound_check.md`: Llama-3.1-8B "
             "masked-mean pooled, layers 4/16/28, 5-fold stratified logistic probe. **Binary → "
             "chance = 0.5** (the original check was 3-way, chance 0.333). k=0 is the "
             "mean-centering-only condition. *Protocol caveat (applies to both columns equally):* "
             "in a 2-class matrix the between-class direction is typically among the top PCs, so "
             "within-task PC removal partially removes class signal mechanically; the head-to-head "
             "is still apples-to-apples, and our corpus is also shown under global-PC removal for context.")
    L.append("")
    L.append("## Head-to-head: within-task PC removal (acc / macro-F1, chance 0.5)")
    L.append("")
    L.append("### WMDP-cyber: forget vs retain")
    L.append(f"| layer | {head} |")
    L.append("|---:|" + ":---:|" * len(KS))
    for layer in LAYERS:
        L.append(fmt_row(layer, wmdp[layer]))
    L.append("")
    L.append("### Our corpus: offense vs defense (same within-pair protocol)")
    L.append(f"| layer | {head} |")
    L.append("|---:|" + ":---:|" * len(KS))
    for layer in LAYERS:
        L.append(fmt_row(layer, ours[layer]))
    L.append("")
    L.append("### Context — our offense-vs-defense under GLOBAL PCs (from the 600-doc 3-way matrix)")
    L.append(f"| layer | {head} |")
    L.append("|---:|" + ":---:|" * len(KS))
    for layer, cells in ctx_rows:
        L.append(fmt_row(layer, cells))
    L.append("")
    L.append("Reference, our 3-way probe (chance 0.333): L28 base 0.950 → drop-3 **0.808** "
             "(robust); L4/L16 collapse to ≤ chance by drop-3 (`reports/pca_confound_check.md`).")

    # sober, threshold-flagged interpretation
    w28 = wmdp[28]
    o28 = ours[28]
    L.append("")
    L.append("## Interpretation (sober)")
    surv = w28[2][0] > 0.7 and w28[3][0] > 0.65
    coll = w28[2][0] < 0.6
    if surv:
        verdict = ("WMDP-cyber forget/retain separability **survives** within-task PC removal at the "
                   "deep layer — its split is comparatively distributed, unlike ours, which collapses "
                   "under the same treatment.")
    elif coll:
        verdict = ("WMDP-cyber forget/retain separability **collapses** under within-task PC removal, "
                   "matching our corpus: in both, the binary split is concentrated in a few dominant "
                   "variance directions (consistent with register/topic confounds, with the mechanical-"
                   "removal caveat above applying to both).")
    else:
        verdict = ("WMDP-cyber lands **between** survival and collapse — partial low-dimensional "
                   "concentration; treat as inconclusive and inspect per-layer numbers.")
    L.append(f"- WMDP L28: base {w28[0][0]:.3f} → drop-2 {w28[2][0]:.3f} → drop-3 {w28[3][0]:.3f}. "
             f"Ours L28 (same protocol): {o28[0][0]:.3f} → {o28[2][0]:.3f} → {o28[3][0]:.3f}.")
    L.append(f"- {verdict}")
    L.append("- Either way this is separability calibration only: no entanglement claim, and no "
             "substrate-domain analysis was run on WMDP (no substrate labels; that analysis already "
             "produced a confound artifact on our data).")

    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT.relative_to(ROOT)}")
    for layer in LAYERS:
        print(f"  WMDP L{layer}: " + "  ".join(f"k{k}={a:.3f}" for k, (a, _) in zip(KS, wmdp[layer])))


if __name__ == "__main__":
    main()
