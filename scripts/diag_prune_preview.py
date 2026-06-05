"""GPU-FREE preview of the external-only prune, using the ALREADY-extracted hidden states.

The full re-extraction (200 external docs/bucket on the pod) is the rigorous test. This preview
just filters the existing `llama31_8b_three_way.parquet` to external-only (drops the procedure +
prose rows already present in the 200-doc sample), balances per bucket, and re-runs the probe /
centroid geometry — to see *directionally* whether dropping the procedure register lowers
separability and changes the offense-outlier structure. Sample is small (~99/bucket), so treat the
numbers as indicative, not final.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from entanglement.representation_analysis import analyze_layer
from entanglement.separability import probe
from entanglement.viz_separability import BUCKET_COLORS, project_2d

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "data" / "hidden_states" / "llama31_8b_three_way.parquet"
DROP_SOURCES = {"procedure", "prose"}            # framework-internal cataloging
FULL_3WAY = {4: 0.923, 16: 0.938, 28: 0.952}      # pre-prune (full sample) for comparison
FULL_BIN = {4: 0.932, 16: 0.953, 28: 0.950}


def balance(df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    n = df.group_by("corpus_label").len()["len"].min()
    parts = [df.filter(pl.col("corpus_label") == c).sample(n=n, seed=seed)
             for c in ("offense", "dual", "defense")]
    return pl.concat(parts), n


def main(layer_for_fig: int = 28):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pl.read_parquet(HS).filter(~pl.col("source_id").is_in(list(DROP_SOURCES)))
    print("external-only rows per (layer, bucket):")
    print(df.group_by(["layer", "corpus_label"]).len().sort(["layer", "corpus_label"]))

    print("\nlayer | 3-way: full -> pruned | off-vs-def: full -> pruned | dual_between | "
          "closer | within-dual topic")
    fig_df = None
    for L in sorted(df["layer"].unique().to_list()):
        sub = df.filter(pl.col("layer") == L)
        bal, n = balance(sub)
        res = analyze_layer(bal, L)
        d = sub.filter(pl.col("corpus_label") == "dual")
        Xd = np.asarray(d["embedding"].to_list(), dtype=np.float64)
        topic_acc = probe(Xd, np.asarray(d["subcap"].to_list()), folds=4)["accuracy"]
        print(f"  {L:3d} | {FULL_3WAY.get(L, float('nan')):.3f} -> {res['three_way_acc']:.3f}      "
              f"| {FULL_BIN.get(L, float('nan')):.3f} -> {res['binary_acc']:.3f}        "
              f"| {str(res['dual_between']):5s} | {res['dual_closer_to']:7s} | {topic_acc:.3f}  "
              f"(balanced n={n}/bucket)")
        if L == layer_for_fig:
            fig_df = bal

    # t-SNE at the requested layer (external-only, balanced), colored by bucket
    X = np.asarray(fig_df["embedding"].to_list(), dtype=np.float64)
    y = np.asarray(fig_df["corpus_label"].to_list())
    proj = project_2d(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    for b, c in BUCKET_COLORS.items():
        p = proj[y == b]
        ax.scatter(p[:, 0], p[:, 1], s=16, alpha=0.6, c=c, label=b, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2)
    ax.set_title(f"External-only prune PREVIEW — layer {layer_for_fig} (t-SNE, balanced)")
    out = ROOT / "reports" / "figures" / f"diag_prune_preview_layer{layer_for_fig:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
