"""Quick confound diagnostic: is the model clustering by document SOURCE/topic, not valence?

Recolors the existing hidden-states t-SNE by `source_id` (provenance/register) and projects the
DUAL bucket alone colored by `subcap` (topic = its six source families). If dual fragments into its
six sources, the separability is substantially source/topic structure, not offense/defense valence.
Also probes: predict source_id (overall) and predict topic within dual — high accuracy = confound.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from entanglement.separability import probe
from entanglement.viz_separability import project_2d

ROOT = Path(__file__).resolve().parents[1]
HS = ROOT / "data" / "hidden_states" / "llama31_8b_three_way.parquet"

SOURCE_COLORS = {
    "procedure": "#d62728", "offense_external": "#ff9896",
    "prose": "#1f77b4", "defense_external": "#aec7e8",
    "substrate": "#7f7f7f",
}
TOPIC_COLORS = {
    "crypto": "#9467bd", "os_internals": "#2ca02c", "web": "#ff7f0e",
    "networking": "#17becf", "recon": "#8c564b", "architecture": "#e377c2",
}


def mat(df, layer):
    sub = df.filter(pl.col("layer") == layer)
    X = np.asarray(sub["embedding"].to_list(), dtype=np.float64)
    return sub, X


def main(layer=28):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pl.read_parquet(HS)

    # ---- probes across layers (numbers) ----
    print("layer | source_id 5-way acc | within-dual topic 6-way acc")
    for L in sorted(df["layer"].unique().to_list()):
        sub, X = mat(df, L)
        src_acc = probe(X, np.asarray(sub["source_id"].to_list()), folds=5)["accuracy"]
        d = sub.filter(pl.col("corpus_label") == "dual")
        Xd = np.asarray(d["embedding"].to_list(), dtype=np.float64)
        topic_acc = probe(Xd, np.asarray(d["subcap"].to_list()), folds=4)["accuracy"]
        print(f"  {L:3d} | {src_acc:.3f}              | {topic_acc:.3f}")

    # ---- figure at the requested layer ----
    sub, X = mat(df, layer)
    proj = project_2d(X)
    src = np.asarray(sub["source_id"].to_list())

    dual = sub.filter(pl.col("corpus_label") == "dual")
    Xd = np.asarray(dual["embedding"].to_list(), dtype=np.float64)
    proj_d = project_2d(Xd)
    topic = np.asarray(dual["subcap"].to_list())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    for s, c in SOURCE_COLORS.items():
        p = proj[src == s]
        ax.scatter(p[:, 0], p[:, 1], s=12, alpha=0.6, c=c, label=s, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title(f"(A) layer {layer} — all 600, colored by source_id (register)")

    ax = axes[1]
    for t, c in TOPIC_COLORS.items():
        p = proj_d[topic == t]
        ax.scatter(p[:, 0], p[:, 1], s=18, alpha=0.7, c=c, label=t, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title(f"(B) layer {layer} — DUAL only, colored by subcap (6 sources)")

    fig.suptitle("Source/topic confound check — does structure track provenance, not valence?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ROOT / "reports" / "figures" / f"diag_source_tsne_layer{layer:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
