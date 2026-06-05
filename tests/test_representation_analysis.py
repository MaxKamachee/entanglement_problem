"""Tests for the representation-analysis helpers (offline, no model, synthetic embeddings)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")

from entanglement.representation_analysis import (  # noqa: E402
    analyze_layer,
    out_of_fold_confidence,
    spread_normalized_distances,
    within_class_spread,
)


def _synthetic_df(n=40, dim=16, seed=0):
    """3 separated Gaussian blobs (offense/dual/defense) x 2 fake layers, long format.

    Dual is placed *between* offense and defense so the 'dual between' checks should pass.
    Layer 8 is more separated than layer 4 (so a depth trend is detectable if asserted).
    """
    rng = np.random.default_rng(seed)
    centers = {"offense": np.r_[3.0, np.zeros(dim - 1)],
               "dual": np.r_[1.5, 1.5, np.zeros(dim - 2)],
               "defense": np.r_[0.0, 3.0, np.zeros(dim - 2)]}
    rows = []
    for layer, spread in ((4, 0.6), (8, 0.3)):
        for b, c in centers.items():
            pts = rng.normal(0, spread, (n, dim)) + c
            for i in range(n):
                rows.append({"doc_id": f"{b}-{layer}-{i}", "corpus_label": b,
                             "subcap": None, "source_id": "x", "layer": layer,
                             "embedding": pts[i].tolist()})
    return pl.DataFrame(rows)


def test_within_class_spread_positive_finite():
    df = _synthetic_df()
    X = np.asarray(df.filter(pl.col("layer") == 4)["embedding"].to_list())
    y = np.asarray(df.filter(pl.col("layer") == 4)["corpus_label"].to_list())
    spread = within_class_spread(X, y)
    assert set(spread) == {"offense", "dual", "defense"}
    assert all(np.isfinite(v) and v >= 0 for v in spread.values())


def test_dual_sits_between_normalized():
    df = _synthetic_df()
    X = np.asarray(df.filter(pl.col("layer") == 4)["embedding"].to_list())
    y = np.asarray(df.filter(pl.col("layer") == 4)["corpus_label"].to_list())
    nd = spread_normalized_distances(X, y)["normalized"]
    assert nd["offense-defense"] > nd["offense-dual"]
    assert nd["offense-defense"] > nd["defense-dual"]


def test_out_of_fold_confidence_in_range():
    df = _synthetic_df()
    X = np.asarray(df.filter(pl.col("layer") == 4)["embedding"].to_list())
    y = np.asarray(df.filter(pl.col("layer") == 4)["corpus_label"].to_list())
    true_prob, pred = out_of_fold_confidence(X, y, folds=4)
    assert true_prob.shape == y.shape and pred.shape == y.shape
    assert true_prob.min() >= 0.0 and true_prob.max() <= 1.0


def test_analyze_layer_shape_and_separability():
    df = _synthetic_df()
    res = analyze_layer(df, 4)
    # well-separated blobs -> high 3-way accuracy, dual between, sensible borderline fractions
    assert res["three_way_acc"] >= 0.85
    assert res["dual_between"] is True
    assert set(res["borderline_frac"]) == {"offense", "dual", "defense"}
    assert all(0.0 <= v <= 1.0 for v in res["borderline_frac"].values())
    # one confidence row per doc in this layer (3 buckets x 40)
    assert len(res["doc_rows"]) == 120
    assert all(0.0 <= r["true_class_prob"] <= 1.0 for r in res["doc_rows"])
