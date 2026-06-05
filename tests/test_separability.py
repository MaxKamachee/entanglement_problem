"""Tests for the separability pilot helpers (no embedding-model download)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytest.importorskip("sklearn")

from entanglement.separability import (  # noqa: E402
    TRUNC_CHARS,
    balanced_sample,
    centroid_cosine_distances,
    probe,
)


def _units(n_off, n_dual, n_def):
    rows = (
        [{"bucket": "offense", "text": "o" * 50}] * n_off
        + [{"bucket": "dual", "text": "d" * 50}] * n_dual
        + [{"bucket": "defense", "text": "f" * 50}] * n_def
    )
    return pl.DataFrame(rows)


def test_balanced_sample_caps_per_bucket():
    s = balanced_sample(_units(100, 80, 30), per_bucket=50)
    counts = dict(s.group_by("bucket").len().iter_rows())
    assert counts["offense"] == 50 and counts["dual"] == 50
    assert counts["defense"] == 30          # smaller bucket taken whole


def test_balanced_sample_truncates():
    s = balanced_sample(
        pl.DataFrame([{"bucket": "offense", "text": "x" * 5000}]), per_bucket=10
    )
    assert s["text"].str.len_chars().max() == TRUNC_CHARS


def test_centroids_locate_dual_between():
    # offense ~ axis-0, defense ~ axis-1, dual ~ diagonal (between)
    X = np.array([[1.0, 0.0]] * 5 + [[0.7, 0.7]] * 5 + [[0.0, 1.0]] * 5)
    buckets = ["offense"] * 5 + ["dual"] * 5 + ["defense"] * 5
    d = centroid_cosine_distances(X, buckets)
    assert d["offense-defense"] > d["offense-dual"]
    assert d["offense-defense"] > d["defense-dual"]   # dual lies between


def test_probe_separates_clean_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (40, 8)), rng.normal(5, 0.1, (40, 8))])
    y = np.array(["offense"] * 40 + ["defense"] * 40)
    res = probe(X, y, folds=4)
    assert res["accuracy"] > 0.95 and res["baseline"] == 0.5


def test_project_2d_shape():
    import numpy as np
    from entanglement.viz_separability import project_2d
    X = np.vstack([np.random.default_rng(0).normal(c, 0.3, (12, 16)) for c in (0, 4, 8)])
    proj = project_2d(X)
    assert proj.shape == (36, 2)
