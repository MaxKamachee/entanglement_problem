#!/usr/bin/env python
"""PCA confound check on the Llama hidden states (reproduces reports/pca_confound_check.md).

Check 1: 3-way offense/dual/defense probe accuracy after removing the top-k unsupervised
principal components per layer — is separability distributed, or riding a few dominant
(confoundable) variance directions?

Check 2: substrate-domain sharing via nearest dual-domain centroid (cosine), layer 28,
under uncentered / centered / top-3-PC-removed transforms — is the domain assignment
stable, or an artifact of low-dimensional global structure?

Result (2026-06-08, pre-supplement hidden states): deep-layer separability IS robust
(0.81 after top-3 removal at L28); substrate-domain assignment is NOT (recon dominance
181 -> 71 -> 41; null result). See the report for the full reading.
"""

from __future__ import annotations

import collections

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HIDDEN = "data/hidden_states/llama31_8b_three_way.parquet"
LAYERS = [4, 16, 28]
SHARING_LAYER = 28


def remove_top_pcs(X: np.ndarray, k: int) -> np.ndarray:
    """Center, then project out the top-k right-singular directions (k=0: center only)."""
    Xc = X - X.mean(0)
    if k == 0:
        return Xc
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    return Xc - (Xc @ V.T) @ V


def probe(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """5-fold stratified logistic probe -> (accuracy, macro-F1)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    r = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
    return r["test_accuracy"].mean(), r["test_f1_macro"].mean()


def _norm(M: np.ndarray) -> np.ndarray:
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def domain_sharing(X: np.ndarray, labels: np.ndarray, domains_col: np.ndarray) -> dict:
    """Nearest dual-domain centroid (cosine) counts for offense + defense docs."""
    Xn = _norm(X)
    dual = labels == "dual"
    dom = domains_col[dual]
    domains = sorted(set(dom.tolist()))
    C = np.array([_norm(Xn[dual][dom == d].mean(0, keepdims=True))[0] for d in domains])
    out: dict = {"domains": domains}
    for b in ("offense", "defense"):
        nn = (Xn[labels == b] @ C.T).argmax(1)
        out[b] = collections.Counter(domains[i] for i in nn)
    return out


def main() -> None:
    h = pl.read_parquet(HIDDEN)

    print("=== Check 1: 3-way probe acc/F1 after dropping top-k PCs (chance 0.333) ===")
    print("  layer |     k=0     |   drop 1    |   drop 2    |   drop 3")
    for layer in LAYERS:
        s = h.filter(pl.col("layer") == layer)
        X = np.array(s["embedding"].to_list())
        y = s["corpus_label"].to_numpy()
        cells = []
        for k in (0, 1, 2, 3):
            a, f = probe(remove_top_pcs(X, k), y)
            cells.append(f"{a:.3f}/{f:.3f}")
        print(f"    {layer:3d} | " + " | ".join(cells))

    print(f"\n=== Check 2: substrate sharing, layer {SHARING_LAYER} (/200 docs each) ===")
    s = h.filter(pl.col("layer") == SHARING_LAYER)
    X = np.array(s["embedding"].to_list())
    labels = np.array(s["corpus_label"].to_list())
    sub = np.array(s["subcap"].to_list())
    runs = [("uncentered", X), ("centered", remove_top_pcs(X, 0)), ("top-3-removed", remove_top_pcs(X, 3))]
    shared = [domain_sharing(M, labels, sub) for _, M in runs]
    domains = shared[0]["domains"]
    header = " | ".join(f"{name} off/def" for name, _ in runs)
    print(f"  {'domain':14s} | {header}")
    for d in domains:
        cells = " | ".join(f"{r['offense'].get(d, 0):4d} /{r['defense'].get(d, 0):4d}" for r in shared)
        print(f"  {d:14s} | {cells}")


if __name__ == "__main__":
    main()
