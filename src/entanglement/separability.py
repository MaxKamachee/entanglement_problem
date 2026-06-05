"""Separability pilot: do offense / defense / dual units separate in embedding space?

Reads the unified `analysis_units.parquet`, balanced-samples per bucket (so the
44:1 doc asymmetry doesn't drive the result), truncates each unit to a fixed char
budget (so *length* isn't the signal), then embeds two ways:

* **lexical** — TF-IDF (a surface-vocabulary baseline), and
* **semantic** — a sentence-embedding model (fastembed BGE-small, ONNX, best-effort).

Comparing the two answers the load-bearing question: is offense/defense separability
just disjoint vocabulary, or does it persist semantically? It then measures a linear
probe (offense-vs-defense + 3-way, cross-validated) and centroid cosine distances to
locate the **dual substrate** relative to offense and defense.

This is a *data*-separability measurement. Per the project's standing caveat, high
separability here does NOT establish *capability* separability — that requires the
unlearning-tax experiment. The pilot's job is to characterize the geometry and locate
the substrate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
TRUNC_CHARS = 2000   # cap each unit before embedding so length isn't the signal


def balanced_sample(units: pl.DataFrame, per_bucket: int, seed: int = SEED) -> pl.DataFrame:
    """Sample up to ``per_bucket`` units from each bucket (offense/dual/defense)."""
    parts = []
    for bucket in ("offense", "dual", "defense"):
        b = units.filter(pl.col("bucket") == bucket)
        parts.append(b.sample(n=min(per_bucket, b.height), seed=seed) if b.height else b)
    return pl.concat(parts).with_columns(pl.col("text").str.slice(0, TRUNC_CHARS))


def embed_tfidf(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(max_features=20000, stop_words="english",
                        sublinear_tf=True, min_df=2).fit_transform(texts)
    return X, "tfidf"


def embed_semantic(texts: list[str]):
    """fastembed BGE-small (ONNX). Raises if the model/runtime is unavailable."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    arr = np.array(list(model.embed(texts)))
    return arr, "semantic(bge-small)"


def probe(X, y: np.ndarray, folds: int = 5) -> dict:
    """Cross-validated logistic-regression separability: accuracy + macro-F1 + confusion."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold

    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    preds = np.empty_like(y)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
    labels = sorted(set(y.tolist()))
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "labels": labels,
        "confusion": confusion_matrix(y, preds, labels=labels).tolist(),
        "baseline": 1.0 / len(labels),
    }


def centroid_cosine_distances(X, buckets: list[str]) -> dict:
    """Cosine distance between L2-normalized bucket centroids."""
    from sklearn.preprocessing import normalize

    Xn = normalize(X)
    b = np.array(buckets)
    cent = {k: np.asarray(Xn[b == k].mean(axis=0)).ravel() for k in ("offense", "dual", "defense")}

    def cos_dist(u, v):
        return float(1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    return {
        "offense-defense": cos_dist(cent["offense"], cent["defense"]),
        "offense-dual": cos_dist(cent["offense"], cent["dual"]),
        "defense-dual": cos_dist(cent["defense"], cent["dual"]),
    }


def run_backend(sample: pl.DataFrame, embed_fn) -> dict:
    texts = sample["text"].to_list()
    buckets = sample["bucket"].to_list()
    X, name = embed_fn(texts)
    y3 = np.array(buckets)
    mask = y3 != "dual"
    Xod = X[mask] if hasattr(X, "shape") else X[np.where(mask)[0]]
    return {
        "backend": name,
        "offense_vs_defense": probe(Xod, y3[mask]),
        "three_way": probe(X, y3),
        "centroids": centroid_cosine_distances(X, buckets),
    }


def _fmt(res: dict) -> list[str]:
    od = res["offense_vs_defense"]
    tw = res["three_way"]
    c = res["centroids"]
    closer = "offense" if c["offense-dual"] < c["defense-dual"] else "defense"
    return [
        f"### {res['backend']}",
        "",
        f"- **offense vs defense probe:** accuracy {od['accuracy']:.3f}, macro-F1 {od['macro_f1']:.3f} "
        f"(chance {od['baseline']:.2f})",
        f"- **3-way (offense/dual/defense) probe:** accuracy {tw['accuracy']:.3f}, macro-F1 "
        f"{tw['macro_f1']:.3f} (chance {tw['baseline']:.2f}); labels {tw['labels']}, confusion {tw['confusion']}",
        f"- **centroid cosine distances:** offense–defense {c['offense-defense']:.3f}, "
        f"offense–dual {c['offense-dual']:.3f}, defense–dual {c['defense-dual']:.3f}",
        f"- **dual substrate sits closer to `{closer}`**; "
        + ("it lies *between* offense and defense (both bucket–dual distances < offense–defense)."
           if max(c["offense-dual"], c["defense-dual"]) < c["offense-defense"]
           else "it is NOT strictly between offense and defense."),
        "",
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offense/defense/dual separability pilot.")
    parser.add_argument("--per-bucket", type=int, default=1500)
    args = parser.parse_args()

    units = pl.read_parquet(ROOT / "data" / "analysis_units.parquet")
    sample = balanced_sample(units, args.per_bucket)
    counts = dict(sample.group_by("bucket").len().iter_rows())
    print(f"balanced sample: {counts}  (truncated to {TRUNC_CHARS} chars/unit)")

    results = [run_backend(sample, embed_tfidf)]
    print(f"[lexical] off-vs-def acc={results[0]['offense_vs_defense']['accuracy']:.3f}")
    try:
        results.append(run_backend(sample, embed_semantic))
        print(f"[semantic] off-vs-def acc={results[-1]['offense_vs_defense']['accuracy']:.3f}")
    except Exception as exc:  # model download / runtime unavailable
        print(f"[semantic] skipped: {type(exc).__name__}: {exc}")

    lines = [
        "# Separability pilot — offense / dual / defense in embedding space",
        "",
        f"Balanced sample (per bucket): {counts}; each unit truncated to {TRUNC_CHARS} chars so "
        "length is not the signal. **Data-separability only** — not capability separability "
        "(that needs the unlearning-tax experiment).",
        "",
    ]
    for r in results:
        lines += _fmt(r)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "separability_pilot.md").write_text("\n".join(lines))
    print("wrote reports/separability_pilot.md")


if __name__ == "__main__":
    main()
