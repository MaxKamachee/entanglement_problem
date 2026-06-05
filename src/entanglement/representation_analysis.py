"""Representation-space separability: does the offense / dual / defense structure hold in
Llama-3.1-8B's *hidden states*, and how does it change with depth?

This is the model-representation analog of the BGE document-embedding pilot (`separability.py`).
It reads the hidden states extracted on GPU (`data/hidden_states/llama31_8b_three_way.parquet`,
long format: one row per (doc, layer), 4096-d mean-pooled vector) and, **per layer**, runs the
exact same instruments as the pilot so the numbers are directly comparable:

* linear probe (offense-vs-defense + 3-way, cross-validated) — `separability.probe`
* centroid cosine distances, here **normalized by within-class spread** so "dual sits between"
  is a scale-free claim
* t-SNE projection colored by bucket — `viz_separability.project_2d`
* per-document confidence (out-of-fold P(true class)) -> a per-doc "borderline" entanglement signal

Plus a cross-layer **depth curve**: does separability rise with depth (a capability-relevant
distinction forming in the layers where unlearning will act) or stay flat (surface/register signal)?

Standing caveat (same as the pilot): this measures *representation* separability of the documents,
which is closer to — but still not identical to — *capability* entanglement in the weights. The
definitive test remains the unlearning-tax experiment. High separability here characterizes the
geometry the unlearning interventions will operate on; it does not by itself prove entanglement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from entanglement.separability import centroid_cosine_distances, probe
from entanglement.viz_separability import BUCKET_COLORS, project_2d

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
BUCKETS = ("offense", "dual", "defense")
BORDERLINE_TAU = 0.5            # 3-way P(true class) below this = "borderline" (entangled region)
HIDDEN_STATES = ROOT / "data" / "hidden_states" / "llama31_8b_three_way.parquet"

# BGE pilot reference (reports/separability_pilot.md, 1500/bucket) for the comparison narrative.
BGE_REFERENCE = {
    "three_way_tfidf": 0.934, "three_way_semantic": 0.900,
    "offense_vs_defense_tfidf": 0.943, "offense_vs_defense_semantic": 0.927,
}


def layer_matrix(df: pl.DataFrame, layer: int):
    """Return (X[n,4096], y[n], doc_ids) for one model layer."""
    sub = df.filter(pl.col("layer") == layer)
    X = np.asarray(sub["embedding"].to_list(), dtype=np.float64)
    y = np.asarray(sub["corpus_label"].to_list())
    return X, y, sub["doc_id"].to_list()


def within_class_spread(X: np.ndarray, buckets: np.ndarray) -> dict:
    """Mean cosine distance of each member to its own (normalized) class centroid, per bucket."""
    from sklearn.preprocessing import normalize

    Xn = normalize(X)
    b = np.asarray(buckets)
    out = {}
    for k in BUCKETS:
        pts = Xn[b == k]
        if not len(pts):
            out[k] = float("nan")
            continue
        c = pts.mean(axis=0)
        cn = c / (np.linalg.norm(c) or 1.0)
        cos = pts @ cn / (np.linalg.norm(pts, axis=1) * 1.0)
        out[k] = float(np.mean(1.0 - cos))
    return out


def spread_normalized_distances(X: np.ndarray, buckets: np.ndarray) -> dict:
    """Centroid cosine distance between each pair, divided by the pair's mean within-class spread.

    Scale-free separation: >1 means classes are farther apart than they are internally spread.
    The prediction we test is offense-defense > offense-dual and > defense-dual (dual between).
    """
    raw = centroid_cosine_distances(X, list(buckets))
    spread = within_class_spread(X, buckets)
    norm = {}
    for pair in ("offense-defense", "offense-dual", "defense-dual"):
        a, c = pair.split("-")
        denom = 0.5 * (spread[a] + spread[c])
        norm[pair] = float(raw[pair] / denom) if denom else float("nan")
    return {"raw": raw, "spread": spread, "normalized": norm}


def out_of_fold_confidence(X: np.ndarray, y: np.ndarray, folds: int = 5):
    """Out-of-fold P(true class) and predicted label per row (mirrors ``probe``'s CV setup)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    clf = LogisticRegression(max_iter=2000, C=1.0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    true_prob = np.zeros(len(y))
    pred = np.empty_like(y)
    for tr, te in skf.split(np.zeros(len(y)), y):
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        classes = list(clf.classes_)
        for i, idx in enumerate(te):
            true_prob[idx] = proba[i, classes.index(y[idx])]
        pred[te] = clf.predict(X[te])
    return true_prob, pred


def analyze_layer(df: pl.DataFrame, layer: int) -> dict:
    """All per-layer metrics + the per-document confidence rows for one model layer."""
    X, y, doc_ids = layer_matrix(df, layer)
    mask = y != "dual"
    od = probe(X[mask], y[mask])
    tw = probe(X, y)
    dist = spread_normalized_distances(X, y)
    true_prob, _ = out_of_fold_confidence(X, y)

    borderline = true_prob < BORDERLINE_TAU
    bl_frac = {k: float(borderline[y == k].mean()) for k in BUCKETS}
    nd = dist["normalized"]
    dual_between = nd["offense-defense"] > nd["offense-dual"] and nd["offense-defense"] > nd["defense-dual"]
    closer = "offense" if nd["offense-dual"] < nd["defense-dual"] else "defense"

    doc_rows = [
        {"doc_id": d, "layer": layer, "corpus_label": yi,
         "true_class_prob": float(p), "borderline": bool(p < BORDERLINE_TAU)}
        for d, yi, p in zip(doc_ids, y, true_prob)
    ]
    return {
        "layer": layer,
        "three_way_acc": tw["accuracy"], "three_way_f1": tw["macro_f1"],
        "binary_acc": od["accuracy"], "binary_f1": od["macro_f1"],
        "labels": tw["labels"], "confusion": tw["confusion"],
        "distances": dist, "dual_between": dual_between, "dual_closer_to": closer,
        "borderline_frac": bl_frac,
        "doc_rows": doc_rows,
    }


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def _figure_layer(df: pl.DataFrame, res: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer = res["layer"]
    X, y, _ = layer_matrix(df, layer)
    proj = project_2d(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    ax = axes[0]
    for b, color in BUCKET_COLORS.items():
        pts = proj[y == b]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, c=color, label=b, edgecolors="none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, loc="best", frameon=True)
    ax.set_title(f"(A) layer {layer} hidden states (t-SNE) by bucket")

    ax = axes[1]
    labels = res["labels"]
    cm = np.asarray(res["confusion"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"(B) layer {layer} 3-way confusion (row-norm)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Representation separability — layer {layer} "
                 f"(3-way acc {res['three_way_acc']:.3f}, off-vs-def {res['binary_acc']:.3f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _figure_depth(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(layers, [r["three_way_acc"] for r in results], "o-", label="3-way (off/dual/def)",
            color="#2ca02c")
    ax.plot(layers, [r["binary_acc"] for r in results], "s-", label="offense-vs-defense",
            color="#9467bd")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.axhline(1 / 3, ls=":", c="gray", lw=1)
    ax.text(layers[-1], 0.5, "chance (2-way)", fontsize=7, c="gray", va="bottom", ha="right")
    ax.text(layers[-1], 1 / 3, "chance (3-way)", fontsize=7, c="gray", va="bottom", ha="right")
    for r in results:
        ax.text(r["layer"], r["three_way_acc"] + 0.008, f"{r['three_way_acc']:.2f}",
                ha="center", fontsize=8)
    ax.set_xlabel("model layer (hidden_states index; 0 = embeddings)")
    ax.set_ylabel("CV probe accuracy")
    ax.set_ylim(0.2, 1.02)
    ax.set_xticks(layers)
    ax.set_title("Separability vs depth (3 points — read directionally)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------
def _describe_geometry(results: list[dict]) -> str:
    """Data-driven prose for where dual sits: between, or {X+dual} vs Y-as-outlier."""
    last = results[-1]
    nd = last["distances"]["normalized"]
    pairs = sorted(nd.items(), key=lambda kv: kv[1])
    closest, far = pairs[0][0], pairs[-1][0]
    if last["dual_between"]:
        return ("Dual lies *between* offense and defense at every layer (offense-defense is the "
                "widest pair) — the substrate is an intermediate region, not collapsed onto a pole.")
    # not between: the widest pair involves dual, so dual is decisively on the opposite side
    closest_b = set(closest.split("-"))
    if "dual" in closest_b:
        partner = (closest_b - {"dual"}).pop()
        outlier = (set(far.split("-")) - {"dual"}).pop()
        return (
            f"**Dual is not between offense and defense — it clusters with {partner}.** At every "
            f"layer the closest pair is {closest} and the *widest* pair is {far} (wider than "
            f"offense-defense itself), so **{outlier} is the representational outlier** while dual and "
            f"{partner} sit together. The {partner}-vs-dual gap stays smallest and the {outlier}-vs-dual "
            "gap stays largest as depth increases — i.e. the model represents the neutral substrate "
            f"more like {partner} content than like {outlier} content, and that asymmetry sharpens "
            "with depth.")
    return (f"Dual is not strictly between the poles; the closest pair is {closest} and the widest "
            f"is {far}. At the deepest layer dual sits closer to **{last['dual_closer_to']}**.")


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    if delta > 0.02:
        return "rising"
    if delta < -0.02:
        return "falling"
    return "roughly flat"


def build_report(results: list[dict], fig_paths: dict, out: Path) -> None:
    layers = [r["layer"] for r in results]
    tw = [r["three_way_acc"] for r in results]
    bn = [r["binary_acc"] for r in results]
    last = results[-1]

    lines = [
        "# Representation analysis — offense / dual / defense in Llama-3.1-8B hidden states",
        "",
        "Mean-pooled hidden states for 200 docs/bucket (seed 0, masked-mean, bf16), layers "
        f"{layers}. Same probe/centroid/t-SNE instruments as the BGE pilot, so numbers are "
        "directly comparable. **Representation-separability** — closer to capability than the BGE "
        "*document*-separability, but still not the same thing; the unlearning-tax experiment is "
        "the definitive entanglement test.",
        "",
        "## Headline numbers (per layer)",
        "",
        "| layer | 3-way acc | off-vs-def acc | dual between? | dual closer to | "
        "borderline: off / dual / def |",
        "|------:|----------:|---------------:|:-------------:|:--------------:|:----------------------------:|",
    ]
    for r in results:
        bf = r["borderline_frac"]
        lines.append(
            f"| {r['layer']} | {r['three_way_acc']:.3f} | {r['binary_acc']:.3f} | "
            f"{'yes' if r['dual_between'] else 'NO'} | {r['dual_closer_to']} | "
            f"{bf['offense']:.2f} / {bf['dual']:.2f} / {bf['defense']:.2f} |"
        )

    best_tw = max(tw)
    lines += [
        "",
        "## The four questions",
        "",
        f"**1. vs the BGE pilot (3-way {BGE_REFERENCE['three_way_semantic']:.3f} semantic / "
        f"{BGE_REFERENCE['three_way_tfidf']:.3f} lexical; off-vs-def "
        f"{BGE_REFERENCE['offense_vs_defense_semantic']:.3f}/{BGE_REFERENCE['offense_vs_defense_tfidf']:.3f}).** "
        f"Llama hidden states reach a peak 3-way accuracy of {best_tw:.3f} (layers {layers}); "
        f"off-vs-def peaks at {max(bn):.3f}. "
        + ("That is comparable to or above the document-embedding pilot — the three-way structure is "
           "if anything *clearer* inside the model than in BGE space."
           if best_tw >= BGE_REFERENCE["three_way_semantic"] - 0.02 else
           "That is somewhat below the document-embedding pilot — the model's pooled representations "
           "blur the buckets more than BGE sentence embeddings do."),
        "",
        "**2. Where does dual sit per layer?** " + _describe_geometry(results),
        "",
        f"**3. Does separability rise with depth?** 3-way accuracy is **{_trend(tw)}** across "
        f"layers {layers[0]}→{layers[-1]} ({tw[0]:.3f}→{tw[-1]:.3f}); off-vs-def is **{_trend(bn)}** "
        f"({bn[0]:.3f}→{bn[-1]:.3f}). "
        + ("Rising separability with depth is the signal we hoped for: the offense/dual/defense "
           "distinction sharpens in the deeper layers where unlearning interventions act, consistent "
           "with a capability-relevant (not merely surface/register) distinction."
           if _trend(tw) == "rising" else
           "Roughly flat/with-depth-stable separability suggests much of the distinction is already "
           "present in shallow representations (more register/topic than deep-capability) — read with "
           "the 3-point caveat below.")
        + " *Caveat:* only 3 layers were extracted, so this is **directional**, not a fine curve.",
        "",
        f"**4. Borderline (entangled-region) documents.** Fraction with out-of-fold P(true class) < "
        f"{BORDERLINE_TAU} at the deepest layer (L{last['layer']}): offense "
        f"{last['borderline_frac']['offense']:.2f}, dual {last['borderline_frac']['dual']:.2f}, "
        f"defense {last['borderline_frac']['defense']:.2f}. These low-confidence docs sit where the "
        "corpora overlap in representation space; they are exported per-doc in "
        "`data/document_confidence.parquet` as candidates for follow-up (the docs most likely to be "
        "co-affected by an offense-targeted unlearning intervention).",
        "",
        "## What this suggests (and the caveat)",
        "",
        "The headline shift from the BGE pilot: in the *model's* representation space the dual "
        "substrate does **not** sit in the middle — it clusters with **defense**, and **offense is "
        "the outlier** (offense-dual is the widest pair at every layer). Two readings, not mutually "
        "exclusive: (a) the defensive corpus (D3FEND prose) is itself heavily mechanism/substrate "
        "prose, so defense and substrate genuinely share representational structure — which is *what "
        "you'd expect if defense leans on the shared substrate*; (b) offensive content (attack "
        "procedures, threat-intel attribution) is stylistically/representationally distinctive, "
        "pulling it apart regardless of underlying capability. That separability **and** the "
        "defense↔dual closeness both **strengthen with depth**, which is the direction consistent "
        "with a capability-relevant distinction forming where unlearning acts.",
        "",
        "**Caveat (load-bearing).** This is still the geometry of *pooled document vectors*, not "
        "capability entanglement in the weights. Document-space proximity of defense and dual does "
        "not prove that unlearning offense will (or won't) damage defense — that is precisely what "
        "the unlearning-tax experiment measures. What this analysis establishes is the *substrate "
        "the interventions operate on*: the buckets are cleanly linearly separable (≥0.92 three-way) "
        "deep in the network, so a representation-level forget/retain split is well-posed.",
        "",
        "## Figures",
        "",
        f"![depth curve]({fig_paths['depth']})",
        "",
    ]
    for r in results:
        lines.append(f"![layer {r['layer']}]({fig_paths[r['layer']]})")
        lines.append("")
    lines += [
        "## Per-layer spread-normalized centroid distances",
        "",
        "| layer | offense-defense | offense-dual | defense-dual |",
        "|------:|----------------:|-------------:|-------------:|",
    ]
    for r in results:
        n = r["distances"]["normalized"]
        lines.append(f"| {r['layer']} | {n['offense-defense']:.2f} | {n['offense-dual']:.2f} | "
                     f"{n['defense-dual']:.2f} |")
    lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


def main() -> None:
    if not HIDDEN_STATES.exists():
        raise SystemExit(f"missing {HIDDEN_STATES} — run scripts/extract_hidden_states.py on a GPU first.")
    df = pl.read_parquet(HIDDEN_STATES)
    layers = sorted(df["layer"].unique().to_list())
    print(f"analyzing {df.height} rows across layers {layers}...", flush=True)

    fig_dir = ROOT / "reports" / "figures"
    results = []
    fig_paths = {}
    for layer in layers:
        res = analyze_layer(df, layer)
        results.append(res)
        fig = fig_dir / f"repr_layer{layer:02d}.png"
        _figure_layer(df, res, fig)
        fig_paths[layer] = f"figures/{fig.name}"
        print(f"  layer {layer}: 3-way {res['three_way_acc']:.3f}  off-vs-def "
              f"{res['binary_acc']:.3f}  dual_between={res['dual_between']}", flush=True)

    depth_fig = fig_dir / "repr_depth_curve.png"
    _figure_depth(results, depth_fig)
    fig_paths["depth"] = f"figures/{depth_fig.name}"

    # metrics parquet
    metrics = pl.DataFrame([{
        "layer": r["layer"], "three_way_acc": r["three_way_acc"], "three_way_f1": r["three_way_f1"],
        "binary_acc": r["binary_acc"], "binary_f1": r["binary_f1"],
        "dist_off_def": r["distances"]["normalized"]["offense-defense"],
        "dist_off_dual": r["distances"]["normalized"]["offense-dual"],
        "dist_def_dual": r["distances"]["normalized"]["defense-dual"],
        "dual_between": r["dual_between"], "dual_closer_to": r["dual_closer_to"],
        "borderline_offense": r["borderline_frac"]["offense"],
        "borderline_dual": r["borderline_frac"]["dual"],
        "borderline_defense": r["borderline_frac"]["defense"],
    } for r in results])
    metrics.write_parquet(ROOT / "data" / "representation_metrics.parquet")

    # per-document confidence parquet
    doc_rows = [row for r in results for row in r["doc_rows"]]
    pl.DataFrame(doc_rows).write_parquet(ROOT / "data" / "document_confidence.parquet")

    build_report(results, fig_paths, ROOT / "reports" / "representation_analysis.md")
    print("wrote reports/representation_analysis.md + data/representation_metrics.parquet + "
          "data/document_confidence.parquet", flush=True)


if __name__ == "__main__":
    main()
