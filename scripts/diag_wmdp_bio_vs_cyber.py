#!/usr/bin/env python
"""WMDP bio vs cyber — cross-domain forget/retain entanglement geometry.

The headline question: is WMDP-cyber's forget/retain boundary *more entangled* (harder
to separate, more confound-dependent) than WMDP-bio's? WMDP App D asserts this
qualitatively (cyber knowledge is mostly dual-use so unlearning hurts defenders; bio
knowledge is more separably-offensive) but puts no numbers on it. This script measures
the asymmetry with the SAME confound-controlled protocol we used for the cyber baseline,
applied identically to both domains so the bio<->cyber delta is apples-to-apples.

Protocol (identical to reports/wmdp_cyber_geometry_baseline.md and pca_confound_check.md):
Llama-3.1-8B masked-mean pooled hidden states, layers 4/16/28, 200 docs/split (seed 0),
5-fold stratified logistic probe (StandardScaler). Binary forget/retain -> chance 0.5.
Confound control: k=0 is mean-centering only; drop-top-1/2/3 projects out the top
unsupervised principal directions of the SAME 2-class matrix being probed (within-task).

Reading the result: a split whose accuracy STAYS high after PC removal is *distributed*
(genuinely separable beyond the few dominant variance/topic/register directions); a split
that COLLAPSES toward chance was riding low-dimensional confounds. The entanglement
asymmetry = bio survives PC removal where cyber collapses (or bio collapses *less*).

CAVEAT (carried from pca_confound_check.md): separability != entanglement. Geometry is
suggestive, not proof. The causal test is the unlearning tax (does forgetting offense
collaterally degrade defensive/benign capability?) and relearning speed -- not embedding
geometry. The t-SNE panels here are ILLUSTRATION ONLY; t-SNE distances are not metric, so
no claim is read off cluster spacing. The numbers come from the PC-removal ladder.
Protocol caveat (applies to both domains equally): in a 2-class matrix the between-class
direction is typically among the top PCs, so within-task PC removal partially removes
class signal mechanically -- the head-to-head is still identical treatment per domain.

Inputs:  data/hidden_states/wmdp_bio_forget_retain.parquet
         data/hidden_states/wmdp_cyber_forget_retain.parquet
         (both from scripts/extract_hidden_states.py --buckets forget retain; see
          scripts/prep_wmdp_units.py docstring for the one-shot pod command.)
Output:  reports/wmdp_bio_vs_cyber_geometry.md  + figures/
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
HIDDEN = {
    "bio": ROOT / "data" / "hidden_states" / "wmdp_bio_forget_retain.parquet",
    "cyber": ROOT / "data" / "hidden_states" / "wmdp_cyber_forget_retain.parquet",
}
REPORT = ROOT / "reports" / "wmdp_bio_vs_cyber_geometry.md"
FIG_DIR = ROOT / "reports" / "figures"
LAYERS = [4, 16, 28]
KS = [0, 1, 2, 3]
LABELS = ("forget", "retain")


def remove_top_pcs(X: np.ndarray, k: int) -> np.ndarray:
    """Center X, then project out its own top-k right-singular directions (within-task)."""
    Xc = X - X.mean(0)
    if k == 0:
        return Xc
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k]
    return Xc - (Xc @ V.T) @ V


def probe(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    r = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy", "f1_macro"])
    return float(r["test_accuracy"].mean()), float(r["test_f1_macro"].mean())


def ladder(h: pl.DataFrame) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[tuple[float, float]]] = {}
    for layer in LAYERS:
        s = h.filter((pl.col("layer") == layer) & pl.col("corpus_label").is_in(LABELS))
        X = np.array(s["embedding"].to_list())
        y = s["corpus_label"].to_numpy()
        out[layer] = [probe(remove_top_pcs(X, k), y) for k in KS]
    return out


def centroid_cosdist(h: pl.DataFrame, layer: int) -> tuple[float, float]:
    """Cosine distance between forget/retain centroids, raw and after mean-centering."""
    s = h.filter((pl.col("layer") == layer) & pl.col("corpus_label").is_in(LABELS))
    X = np.array(s["embedding"].to_list())
    y = s["corpus_label"].to_numpy()

    def cd(M: np.ndarray) -> float:
        f = M[y == "forget"].mean(0)
        r = M[y == "retain"].mean(0)
        cos = f @ r / (np.linalg.norm(f) * np.linalg.norm(r) + 1e-12)
        return float(1 - cos)

    return cd(X), cd(X - X.mean(0))


def fmt_row(layer: int, cells: list[tuple[float, float]]) -> str:
    return f"| {layer} | " + " | ".join(f"{a:.3f} / {f:.3f}" for a, f in cells) + " |"


def make_figures(lads: dict, hs: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    # --- Fig 1: PC-removal ladder, bio vs cyber, per layer ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dom in zip(axes, ("bio", "cyber")):
        lad = lads[dom]
        for layer, marker in zip(LAYERS, "osD"):
            ax.plot(KS, [lad[layer][k][0] for k in range(len(KS))], marker=marker, label=f"layer {layer}")
        ax.axhline(0.5, ls="--", c="gray", lw=1)
        ax.text(2.55, 0.515, "chance", color="gray", fontsize=8)
        ax.set_title(f"WMDP-{dom}: forget vs retain", fontsize=11)
        ax.set_xticks(KS)
        ax.set_xticklabels(["0\n(center)", "1", "2", "3"])
        ax.set_xlabel("top PCs removed")
        ax.set_ylim(0.1, 1.02)
    axes[0].set_ylabel("probe accuracy (binary)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Does the forget/retain split survive removing the biggest variance directions?", fontsize=11)
    fig.tight_layout()
    p = FIG_DIR / "wmdp_bio_vs_cyber_ladder.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))

    # --- Fig 2: t-SNE at layer 28 (ILLUSTRATION ONLY — distances not metric) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, dom in zip(axes, ("bio", "cyber")):
        s = hs[dom].filter((pl.col("layer") == 28) & pl.col("corpus_label").is_in(LABELS))
        X = np.array(s["embedding"].to_list())
        y = s["corpus_label"].to_numpy()
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        for cls, color in zip(LABELS, ("tab:red", "tab:blue")):
            m = y == cls
            ax.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.6, c=color, label=cls)
        ax.set_title(f"WMDP-{dom} — layer 28", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=9)
    fig.suptitle("t-SNE (layer 28, illustration only — distances are not metric; see PC-removal ladder for the number)",
                 fontsize=10)
    fig.tight_layout()
    p = FIG_DIR / "wmdp_bio_vs_cyber_tsne.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(str(p))
    return paths


def main() -> None:
    missing = [d for d, p in HIDDEN.items() if not p.exists()]
    if missing:
        sys.exit(f"missing hidden states for {missing} — run the GPU extraction first "
                 "(see scripts/prep_wmdp_units.py docstring for the one-shot pod command).")

    hs = {d: pl.read_parquet(p) for d, p in HIDDEN.items()}
    lads = {d: ladder(h) for d, h in hs.items()}
    counts = {d: dict(h.filter(pl.col("layer") == LAYERS[0]).group_by("corpus_label").len().iter_rows())
              for d, h in hs.items()}
    cdist = {d: {layer: centroid_cosdist(h, layer) for layer in LAYERS} for d, h in hs.items()}

    head = " | ".join(["k=0 (center only)", "drop top-1", "drop top-2", "drop top-3"])
    L = ["# WMDP bio vs cyber — cross-domain forget/retain entanglement geometry", ""]
    L.append("**Scope.** Quantifies the bio<->cyber separability asymmetry WMDP App D asserts "
             "qualitatively. Binary forget/retain split per domain, identical protocol. "
             "**Separability is geometric evidence, not a causal entanglement claim** — the causal "
             "test is the unlearning tax + relearning speed (see `reports/pca_confound_check.md`).")
    L.append("")
    L.append("**Data.** bio-forget `cais/wmdp-bio-forget-corpus` @ `5a786ed` (gated), "
             "bio-retain + cyber `cais/wmdp-corpora` @ `daf89fa` — sha256-pinned, see "
             "`scripts/prep_wmdp_units.py`. Bio source-docs capped at 3,000/split (seed 0) for "
             "shippability (cyber uncapped: 1,000 forget / 4,473 retain docs). Normalized with our "
             "unit pipeline (clean_text + resegment ~3k chars). Sampled per split (seed 0): "
             f"bio {counts['bio']}, cyber {counts['cyber']}.")
    L.append("")
    L.append("**Protocol.** Llama-3.1-8B masked-mean pooled, layers 4/16/28, 5-fold stratified "
             "logistic probe (StandardScaler). Binary → chance 0.5. k=0 = mean-centering only; "
             "drop-top-k removes the top-k within-task principal directions. *Caveat (both domains "
             "equally):* the between-class direction usually sits among the top PCs, so within-task "
             "removal mechanically removes some class signal — identical treatment per domain.")
    L.append("")
    L.append("## Head-to-head: within-task PC removal (acc / macro-F1, chance 0.5)")
    for dom in ("bio", "cyber"):
        L.append("")
        L.append(f"### WMDP-{dom}: forget vs retain")
        L.append(f"| layer | {head} |")
        L.append("|---:|" + ":---:|" * len(KS))
        for layer in LAYERS:
            L.append(fmt_row(layer, lads[dom][layer]))
    L.append("")
    L.append("## Forget/retain centroid cosine distance (raw / mean-centered)")
    L.append("| layer | bio | cyber |")
    L.append("|---:|:---:|:---:|")
    for layer in LAYERS:
        b = cdist["bio"][layer]; c = cdist["cyber"][layer]
        L.append(f"| {layer} | {b[0]:.3f} / {b[1]:.3f} | {c[0]:.3f} / {c[1]:.3f} |")

    # data-driven interpretation
    def survival(lad, layer, k):  # accuracy after dropping k PCs
        return lad[layer][k][0]
    L.append("")
    L.append("## Interpretation (data-driven)")
    for dom in ("bio", "cyber"):
        l = lads[dom]
        L.append(f"- **{dom}** L28: base {survival(l,28,0):.3f} → drop-2 {survival(l,28,2):.3f} "
                 f"→ drop-3 {survival(l,28,3):.3f}; L4 drop-3 {survival(l,4,3):.3f}, "
                 f"L16 drop-3 {survival(l,16,3):.3f}.")
    # asymmetry headline: mean drop-3 accuracy across layers
    bio_d3 = np.mean([survival(lads["bio"], layer, 3) for layer in LAYERS])
    cyb_d3 = np.mean([survival(lads["cyber"], layer, 3) for layer in LAYERS])
    delta = bio_d3 - cyb_d3
    direction = ("**bio is more separable / less entangled than cyber**" if delta > 0.05 else
                 "**cyber is more separable than bio**" if delta < -0.05 else
                 "**no clear asymmetry** between domains")
    L.append(f"- **Asymmetry (headline).** Mean drop-3 accuracy across layers: bio {bio_d3:.3f} vs "
             f"cyber {cyb_d3:.3f} (Δ = {delta:+.3f}). Under confound-controlled PC removal, {direction}. "
             "This is the WMDP App D claim, now with a number — but it remains a *geometric* measure; "
             "confirm with the unlearning-tax experiment before claiming causal entanglement.")

    fig_paths = make_figures(lads, hs)
    L.append("")
    L.append("## Figures")
    for p in fig_paths:
        rel = Path(p).relative_to(ROOT / "reports")
        L.append(f"\n![{Path(p).stem}]({rel})")

    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT.relative_to(ROOT)} + {len(fig_paths)} figures")
    print(f"  asymmetry (mean drop-3 acc): bio {bio_d3:.3f} vs cyber {cyb_d3:.3f} (Δ {delta:+.3f})")
    for dom in ("bio", "cyber"):
        for layer in LAYERS:
            print(f"  {dom} L{layer}: " + "  ".join(f"k{k}={a:.3f}" for k, (a, _) in zip(KS, lads[dom][layer])))


if __name__ == "__main__":
    main()
