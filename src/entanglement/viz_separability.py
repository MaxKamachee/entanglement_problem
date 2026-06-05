"""Visualize the separability pilot: 2D embedding projection + probe accuracy + confusion.

Recomputes the pilot deterministically (same seed/sample as `separability.py`), then renders a
3-panel figure to `reports/figures/separability_pilot.png`:

  (A) 2D t-SNE projection of the semantic embeddings, colored by bucket — the headline
      "do offense / dual / defense separate, and where does the substrate sit?" view.
  (B) linear-probe accuracy (lexical vs semantic; offense-vs-defense vs 3-way) against chance.
  (C) 3-way confusion matrix (semantic) — which buckets get confused.

A viz subsample (`--per-bucket`, default 800) keeps t-SNE snappy; the canonical accuracy numbers
live in `reports/separability_pilot.md` (1,500/bucket).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from entanglement.separability import balanced_sample, embed_semantic, embed_tfidf, probe

ROOT = Path(__file__).resolve().parents[2]
BUCKET_COLORS = {"offense": "#d62728", "dual": "#7f7f7f", "defense": "#1f77b4"}


def project_2d(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """PCA-50 -> t-SNE-2 (standard pipeline for sentence embeddings)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n = X.shape[0]
    pre = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
    return TSNE(n_components=2, init="pca", random_state=seed,
                perplexity=min(30, max(5, n // 4))).fit_transform(pre)


def make_figure(sample: pl.DataFrame, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    texts = sample["text"].to_list()
    buckets = np.array(sample["bucket"].to_list())
    Xsem, _ = embed_semantic(texts)
    Xtf, _ = embed_tfidf(texts)

    mask = buckets != "dual"
    od_sem = probe(Xsem[mask], buckets[mask])
    od_tf = probe(Xtf[np.where(mask)[0]], buckets[mask])
    tw_sem = probe(Xsem, buckets)
    tw_tf = probe(Xtf, buckets)
    proj = project_2d(Xsem)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # (A) 2D projection
    ax = axes[0]
    for b, color in BUCKET_COLORS.items():
        pts = proj[buckets == b]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.45, c=color, label=b, edgecolors="none")
    ax.set_title("(A) Semantic embeddings (t-SNE) by bucket")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, loc="best", frameon=True)

    # (B) probe accuracy bars
    ax = axes[1]
    groups = ["offense\nvs defense", "3-way\n(off/dual/def)"]
    x = np.arange(len(groups))
    w = 0.35
    lex = [od_tf["accuracy"], tw_tf["accuracy"]]
    sem = [od_sem["accuracy"], tw_sem["accuracy"]]
    ax.bar(x - w / 2, lex, w, label="lexical (TF-IDF)", color="#9467bd")
    ax.bar(x + w / 2, sem, w, label="semantic (BGE)", color="#2ca02c")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.axhline(1 / 3, ls=":", c="gray", lw=1)
    ax.text(1.4, 0.5, "chance (2-way)", fontsize=7, c="gray", va="bottom", ha="right")
    ax.text(1.4, 1 / 3, "chance (3-way)", fontsize=7, c="gray", va="bottom", ha="right")
    for xi, (lv, sv) in enumerate(zip(lex, sem)):
        ax.text(xi - w / 2, lv + 0.01, f"{lv:.2f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, sv + 0.01, f"{sv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CV accuracy")
    ax.set_title("(B) Linear-probe separability")
    ax.legend(loc="lower right", fontsize=8)

    # (C) 3-way confusion (semantic), row-normalized
    ax = axes[2]
    labels = tw_sem["labels"]
    cm = np.array(tw_sem["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("(C) 3-way confusion (semantic, row-norm)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Offense / Dual-substrate / Defense — separability pilot "
                 f"({dict(sample.group_by('bucket').len().iter_rows())}, data-separability only)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {"offense_vs_defense_semantic": od_sem["accuracy"], "three_way_semantic": tw_sem["accuracy"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the separability pilot figure.")
    parser.add_argument("--per-bucket", type=int, default=800)
    args = parser.parse_args()

    units = pl.read_parquet(ROOT / "data" / "analysis_units.parquet")
    sample = balanced_sample(units, args.per_bucket)
    out = ROOT / "reports" / "figures" / "separability_pilot.png"
    stats = make_figure(sample, out)
    print(f"wrote {out.relative_to(ROOT)}  (off-vs-def semantic acc {stats['offense_vs_defense_semantic']:.3f})")


if __name__ == "__main__":
    main()
