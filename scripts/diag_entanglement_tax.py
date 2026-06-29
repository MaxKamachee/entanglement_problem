#!/usr/bin/env python
"""Analyze the entanglement unlearning-tax sweep -> reports/entanglement_unlearning_tax.md.

Reads runs/entanglement*/*.json (run_entanglement_sweep.py), possibly mixing methods (rmu/cb/npo),
retain sets (wikitext/substrate), strengths, and SEEDS. Aggregates over seeds to put error bars on
the safety-tax curves and a confidence interval on the tax slope.

Tag formats: base_<domain>; <rmu|cb|npo>_<domain>_<wikitext|substrate>_s<strength>[_seed<n>];
legacy <domain>_<wikitext|substrate>_c<coeff> (= rmu, seed 0).

Tax = same-domain legit ability lost per unit offense removed. Computed PER SEED (slope through that
seed's coherent points) then averaged -> tax ± SE. Cross-domain asymmetry reports the bio/cyber gap
against the combined SE so you can see if it's beyond noise.

Figures: FIG 1 Pareto (offense removed vs neighbor kept) with per-point error bars, RMU/CB/NPO ×
retain × domain. FIG 2 near-neighbor bars at offense-removed≈0.6 (seed-averaged).
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "entanglement_unlearning_tax.md"
FIG_DIR = ROOT / "reports" / "figures"
CHANCE = 0.25
DOMAINS = ["bio", "cyber", "chem"]
RETAINS = ["wikitext", "substrate"]
MLABEL = {"rmu": "RMU", "cb": "Circuit Breakers", "npo": "NPO"}
MSTYLE = {"rmu": "-", "cb": "--", "npo": "-."}
RCOLOR = {"wikitext": "tab:orange", "substrate": "tab:blue"}
KEEP_DEGEN = False


def load_points(dirs=None):
    pts, bases = [], {}
    run_dirs = dirs if dirs is not None else sorted(ROOT.glob("runs/entanglement*"))
    for run_dir in run_dirs:
        for p in sorted(Path(run_dir).glob("*.json")):
            d = json.loads(p.read_text())
            tag = d.get("tag", p.stem)
            scores = d.get("scores", {})
            degen = any(v.get("any_degenerate") for v in d.get("degeneracy", {}).values())
            if tag.startswith("base_"):
                bases[tag.split("_", 1)[1]] = scores
                continue
            m = re.match(r"(rmu|cb|npo)_(bio|cyber|chem)_(wikitext|substrate)_s([\d.]+)(?:_seed(\d+))?$", tag)
            mo = re.match(r"(bio|cyber|chem)_(wikitext|substrate)_c([\d.]+)$", tag)
            if m:
                method, dom, ret, strength = m.group(1), m.group(2), m.group(3), float(m.group(4))
                seed = int(m.group(5)) if m.group(5) else 0
            elif mo:
                method, dom, ret, strength, seed = "rmu", mo.group(1), mo.group(2), float(mo.group(3)), 0
            else:
                continue
            pts.append({"method": method, "domain": dom, "retain": ret, "arm": f"{dom}_{ret}",
                        "strength": strength, "seed": seed, "scores": scores, "degenerate": degen})
    return pts, bases


def norm_point(scores, base):
    bo, bn = base.get("offense_mcq"), base.get("neighbor_mmlu")
    o, n = scores.get("offense_mcq"), scores.get("neighbor_mmlu")
    out = {}
    if bo is not None and o is not None and bo > CHANCE:
        out["off"] = max(0.0, (bo - o) / (bo - CHANCE))
    if bn and n is not None:
        out["nbr"] = n / bn
    return out


def _se(xs):
    return (st.stdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else 0.0


def agg_curve(pts, bases, method, arm):
    """Per strength: aggregate over seeds -> dict(strength -> (off_m, off_se, nbr_m, nbr_se, n, degen))."""
    dom = arm.split("_")[0]
    by_s = {}
    for p in pts:
        if p["method"] != method or p["arm"] != arm:
            continue
        n = norm_point(p["scores"], bases.get(dom, {}))
        if "off" in n and "nbr" in n:
            by_s.setdefault(p["strength"], []).append((n["off"], n["nbr"], p["degenerate"]))
    out = {}
    for s, rows in by_s.items():
        offs = [r[0] for r in rows]; nbrs = [r[1] for r in rows]
        degen = all(r[2] for r in rows) and not KEEP_DEGEN
        out[s] = (st.mean(offs), _se(offs), st.mean(nbrs), _se(nbrs), len(rows), degen)
    return dict(sorted(out.items()))


def tax_ci(pts, bases, method, arm):
    """Per-seed slope of (1-nbr) vs off through coherent points -> (mean, se, n_seeds, max_off)."""
    dom = arm.split("_")[0]
    by_seed, all_off = {}, []
    for p in pts:
        if p["method"] != method or p["arm"] != arm:
            continue
        if p["degenerate"] and not KEEP_DEGEN:
            continue
        n = norm_point(p["scores"], bases.get(dom, {}))
        if "off" in n and "nbr" in n:
            all_off.append(n["off"])
            if n["off"] > 0.02:
                by_seed.setdefault(p["seed"], []).append((n["off"], 1 - n["nbr"]))
    slopes = []
    for pairs in by_seed.values():
        sxx = sum(x * x for x, _ in pairs)
        if sxx > 0:
            slopes.append(sum(x * y for x, y in pairs) / sxx)
    if not slopes:
        return None, None, 0, (max(all_off) if all_off else 0.0)
    return st.mean(slopes), _se(slopes), len(slopes), max(all_off)


def fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def make_figures(pts, bases, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    doms = [d for d in DOMAINS if d in bases]
    paths = []

    # FIG 1: Pareto with error bars (seeds), RMU/CB/NPO × retain × domain
    fig, axes = plt.subplots(1, len(doms), figsize=(5.4 * len(doms), 4.8), squeeze=False, sharey=True)
    for ax, dom in zip(axes[0], doms):
        for method in methods:
            for retain in RETAINS:
                ag = agg_curve(pts, bases, method, f"{dom}_{retain}")
                if not ag:
                    continue
                xs = [v[0] for v in ag.values()]; xe = [v[1] for v in ag.values()]
                ys = [v[2] for v in ag.values()]; ye = [v[3] for v in ag.values()]
                lbl = f"{MLABEL[method]}/{retain}"
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", ls=MSTYLE[method],
                            color=RCOLOR[retain], alpha=0.8, ms=4, capsize=2, label=lbl)
        ax.axhline(1.0, ls=":", c="gray", lw=1)
        ax.set_title(f"WMDP-{dom}", fontsize=11)
        ax.set_xlabel("offense removed (→ chance = 1.0)")
        ax.set_xlim(-0.03, 1.1); ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("same-domain neighbor kept")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Safety tax (mean ± SE over seeds): legit same-domain ability kept vs offense removed",
                 fontsize=11)
    fig.tight_layout()
    p1 = FIG_DIR / "entanglement_pareto.png"
    fig.savefig(p1, dpi=150); plt.close(fig); paths.append(str(p1))

    # FIG 2: near-neighbor bars at offense-removed≈0.6 (seed-averaged scores)
    def op_scores(method, arm):
        ag = agg_curve(pts, bases, method, arm)
        cand = {s: v for s, v in ag.items() if not v[5]}
        if not cand:
            return None
        s_op = min(cand, key=lambda s: abs(cand[s][0] - 0.6))
        rows = [p["scores"] for p in pts if p["method"] == method and p["arm"] == arm
                and p["strength"] == s_op and (KEEP_DEGEN or not p["degenerate"])]
        if not rows:
            return None
        keys = ["offense_mcq", "general_mmlu"] + list(bases[arm.split("_")[0]]
                                                      .get("neighbor_mmlu_by_subject", {}))
        avg = {}
        for k in keys:
            vals = [(r.get(k) if k in ("offense_mcq", "general_mmlu")
                     else r.get("neighbor_mmlu_by_subject", {}).get(k)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                avg[k] = st.mean(vals)
        return avg

    fig, axes = plt.subplots(len(methods), len(doms), figsize=(5.6 * len(doms), 4.2 * len(methods)),
                             squeeze=False)
    for r, method in enumerate(methods):
        for cax, dom in zip(axes[r], doms):
            base = bases[dom]
            subj = list(base.get("neighbor_mmlu_by_subject", {}).keys())
            cats = ["offense_mcq", "general_mmlu"] + subj
            labels = ["WMDP\n(off↓)", "MMLU\nall"] + [s.replace("_", "\n") for s in subj]
            series = {"base": {k: (base.get(k) if k in ("offense_mcq", "general_mmlu")
                                   else base.get("neighbor_mmlu_by_subject", {}).get(k)) for k in cats}}
            for retain in RETAINS:
                sc = op_scores(method, f"{dom}_{retain}")
                if sc:
                    series[retain] = sc
            x = np.arange(len(cats)); w = 0.26
            for i, (name, sc) in enumerate(series.items()):
                cax.bar(x + (i - 1) * w, [sc.get(c) or 0 for c in cats], w, label=name)
            cax.axhline(CHANCE, ls="--", c="gray", lw=1)
            cax.set_title(f"{MLABEL[method]} — WMDP-{dom} (off≈0.6)", fontsize=10)
            cax.set_xticks(x); cax.set_xticklabels(labels, fontsize=7); cax.set_ylim(0, 1.0)
            cax.legend(fontsize=8)
        axes[r][0].set_ylabel("accuracy")
    fig.suptitle("Near-domain over-removal at matched offense removed (seed-averaged)", fontsize=11)
    fig.tight_layout()
    p2 = FIG_DIR / "entanglement_fig11.png"
    fig.savefig(p2, dpi=150); plt.close(fig); paths.append(str(p2))
    return paths


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="single run dir (default: glob runs/entanglement*)")
    ap.add_argument("--keep-degen", action="store_true", help="don't gate degeneracy-flagged points")
    a = ap.parse_args()
    global KEEP_DEGEN
    KEEP_DEGEN = a.keep_degen
    pts, bases = load_points([Path(a.run_dir)] if a.run_dir else None)
    if not bases:
        sys.exit("no base_<domain>.json found.")
    methods = [m for m in ("rmu", "cb", "npo") if any(p["method"] == m for p in pts)]

    L = ["# Entanglement unlearning-tax (seed-aggregated)", ""]
    L.append("Tax = legit same-domain ability lost per unit offense removed (per-seed slope, "
             "mean ± SE). Methods: " + ", ".join(MLABEL[m] for m in methods) + ".")
    L.append("")
    L.append("## Base model")
    L.append("| domain | WMDP offense | neighbor (mean) | general MMLU |")
    L.append("|---|--:|--:|--:|")
    for d in DOMAINS:
        if d in bases:
            b = bases[d]
            L.append(f"| {d} | {fmt(b.get('offense_mcq'))} | {fmt(b.get('neighbor_mmlu'))} | "
                     f"{fmt(b.get('general_mmlu'))} |")
    L.append("")
    L.append("## Tax ± SE (over seeds) by method × domain × retain")
    L.append("| method | domain | retain | tax ± SE | seeds | max off removed |")
    L.append("|---|---|---|--:|--:|--:|")
    T = {}
    for method in methods:
        for dom in DOMAINS:
            if dom not in bases:
                continue
            for retain in RETAINS:
                t, se, ns, mx = tax_ci(pts, bases, method, f"{dom}_{retain}")
                if ns == 0 and mx < 0.02:
                    continue
                T[(method, dom, retain)] = (t, se, ns)
                tstr = f"{fmt(t)} ± {fmt(se)}" if t is not None else "— (suppressed)"
                L.append(f"| {MLABEL[method]} | {dom} | {retain} | {tstr} | {ns} | {fmt(mx)} |")
    L.append("")
    L.append("## Cross-domain asymmetry (wikitext arm)")
    for method in methods:
        tb = T.get((method, "bio", "wikitext")); tc = T.get((method, "cyber", "wikitext"))
        if tb and tc and tb[0] is not None and tc[0] is not None:
            gap = tc[0] - tb[0]
            comb_se = (tb[1] ** 2 + tc[1] ** 2) ** 0.5
            sig = "beyond combined SE" if abs(gap) > comb_se else "WITHIN combined SE (not significant)"
            L.append(f"- **{MLABEL[method]}:** cyber {fmt(tc[0])}±{fmt(tc[1])} vs bio "
                     f"{fmt(tb[0])}±{fmt(tb[1])} → gap {gap:+.3f} ({sig}).")

    figs = make_figures(pts, bases, methods)
    L.append("")
    L.append("## Figures")
    for p in figs:
        L.append(f"\n![{Path(p).stem}]({Path(p).relative_to(ROOT / 'reports')})")
    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT.relative_to(ROOT)} + {len(figs)} figures; methods={methods}")


if __name__ == "__main__":
    main()
