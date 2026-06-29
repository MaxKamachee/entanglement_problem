#!/usr/bin/env python
"""Analyze the entanglement unlearning-tax sweep -> reports/entanglement_unlearning_tax.md.

Reads runs/entanglement*/*.json (from run_entanglement_sweep.py / run_entanglement_flash.py),
which may mix unlearning methods (rmu_*, cb_*) and a shared base_<domain>. Per (method, domain,
retain-set, strength): offense (WMDP MCQ), same-domain MMLU neighbor, general MMLU, coherence.

Produces the safety-tax figures that support the paper's claims:
  FIG 1 (Pareto): offense removed vs same-domain legit kept, per domain, RMU vs CB (method-indep).
  FIG 2 (near-neighbor bars): at a fixed operating point, base vs unlearned per subject — shows
        unlearning craters the near-domain legit neighbor (the tax made concrete).
Tax = same-domain neighbor lost per unit offense removed (slope through origin, coherent points).

Tag formats parsed: base_<domain>; <rmu|cb>_<domain>_<wikitext|substrate>_s<strength>;
legacy <domain>_<wikitext|substrate>_c<coeff> (treated as rmu).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "entanglement_unlearning_tax.md"
FIG_DIR = ROOT / "reports" / "figures"
CHANCE = 0.25
DOMAINS = ["bio", "cyber", "chem"]
RETAINS = ["wikitext", "substrate"]
MLABEL = {"rmu": "RMU", "cb": "Circuit Breakers"}
MSTYLE = {"rmu": "-", "cb": "--"}
RCOLOR = {"wikitext": "tab:orange", "substrate": "tab:blue"}


def load_points() -> tuple[dict, dict]:
    """Returns (points, bases). points[tag] = {method,domain,retain,arm,strength,scores,degenerate}."""
    pts, bases = {}, {}
    for run_dir in sorted(ROOT.glob("runs/entanglement*")):
        for p in sorted(run_dir.glob("*.json")):
            d = json.loads(p.read_text())
            tag = d.get("tag", p.stem)
            scores = d.get("scores", {})
            degen = any(v.get("any_degenerate") for v in d.get("degeneracy", {}).values())
            if tag.startswith("base_"):
                bases[tag.split("_", 1)[1]] = scores
                continue
            m = re.match(r"(rmu|cb)_(bio|cyber|chem)_(wikitext|substrate)_s([\d.]+)", tag)
            mo = re.match(r"(bio|cyber|chem)_(wikitext|substrate)_c([\d.]+)", tag)  # legacy = rmu
            if m:
                method, dom, retain, strength = m.group(1), m.group(2), m.group(3), float(m.group(4))
            elif mo:
                method, dom, retain, strength = "rmu", mo.group(1), mo.group(2), float(mo.group(3))
            else:
                continue
            pts[tag] = {"method": method, "domain": dom, "retain": retain, "arm": f"{dom}_{retain}",
                        "strength": strength, "scores": scores, "degenerate": degen}
    return pts, bases


def norm_point(scores: dict, base: dict) -> dict:
    bo, bn, bg = base.get("offense_mcq"), base.get("neighbor_mmlu"), base.get("general_mmlu")
    o, n, g = scores.get("offense_mcq"), scores.get("neighbor_mmlu"), scores.get("general_mmlu")
    out = {}
    if bo is not None and o is not None and bo > CHANCE:
        out["offense_removed"] = max(0.0, (bo - o) / (bo - CHANCE))
    if bn and n is not None:
        out["neighbor_kept"] = n / bn
    if bg and g is not None:
        out["general_kept"] = g / bg
    return out


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def curve(pts, bases, method, arm):
    """Sorted coherent (offense_removed, neighbor_kept, strength, degenerate) for a (method,arm)."""
    dom = arm.split("_")[0]
    rows = []
    for p in pts.values():
        if p["method"] == method and p["arm"] == arm:
            n = norm_point(p["scores"], bases.get(dom, {}))
            if "offense_removed" in n and "neighbor_kept" in n:
                rows.append((n["offense_removed"], n["neighbor_kept"], p["strength"], p["degenerate"]))
    return sorted(rows, key=lambda r: r[2])


def tax(pts, bases, method, arm):
    pairs = [(x, 1 - y) for x, y, _, dg in curve(pts, bases, method, arm) if not dg and x > 0.02]
    if not pairs:
        return None
    sxx = sum(x * x for x, _ in pairs); sxy = sum(x * y for x, y in pairs)
    return sxy / sxx if sxx > 0 else None


def max_off(pts, bases, method, arm):
    vals = [x for x, _, _, dg in curve(pts, bases, method, arm) if not dg]
    return max(vals) if vals else 0.0


def make_figures(pts, bases, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    doms = [d for d in DOMAINS if d in bases]
    paths = []

    # --- FIG 1: Pareto, per domain, RMU vs CB (linestyle) x retain (color) ---
    fig, axes = plt.subplots(1, len(doms), figsize=(5.2 * len(doms), 4.6), squeeze=False, sharey=True)
    for ax, dom in zip(axes[0], doms):
        for method in methods:
            for retain in RETAINS:
                c = curve(pts, bases, method, f"{dom}_{retain}")
                if not c:
                    continue
                xs = [r[0] for r in c]; ys = [r[1] for r in c]
                ax.plot(xs, ys, MSTYLE[method], color=RCOLOR[retain], alpha=0.7,
                        label=f"{MLABEL[method]} / {retain}")
                for x, y, s, dg in c:
                    ax.scatter([x], [y], color=RCOLOR[retain], marker="x" if dg else "o",
                               s=55 if dg else 35, zorder=3)
        ax.axhline(1.0, ls=":", c="gray", lw=1)
        ax.set_title(f"WMDP-{dom}", fontsize=11)
        ax.set_xlabel("offense removed (→ chance = 1.0)")
        ax.set_xlim(-0.02, 1.1); ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("same-domain neighbor kept")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Safety tax: legitimate same-domain ability kept vs offense removed "
                 "(down-slope = tax; × = incoherent)", fontsize=11)
    fig.tight_layout()
    p1 = FIG_DIR / "entanglement_pareto.png"
    fig.savefig(p1, dpi=150); plt.close(fig); paths.append(str(p1))

    # --- FIG 2: near-neighbor bars at an operating point (offense_removed closest to 0.6) ---
    def op_point(method, arm):
        c = [r for r in curve(pts, bases, method, arm) if not r[3]]
        if not c:
            return None
        best = min(c, key=lambda r: abs(r[0] - 0.6))
        s = best[2]
        for p in pts.values():
            if p["method"] == method and p["arm"] == arm and p["strength"] == s:
                return p["scores"]
        return None

    fig, axes = plt.subplots(len(methods), len(doms), figsize=(5.6 * len(doms), 4.2 * len(methods)),
                             squeeze=False)
    for r, method in enumerate(methods):
        for cax, dom in zip(axes[r], doms):
            base = bases[dom]
            subj = list(base.get("neighbor_mmlu_by_subject", {}).keys())
            cats = ["offense_mcq", "general_mmlu"] + subj
            labels = ["WMDP\n(off↓)", "MMLU\nall"] + [s.replace("_", "\n") for s in subj]
            series = {"base": base}
            for retain in RETAINS:
                sc = op_point(method, f"{dom}_{retain}")
                if sc:
                    series[retain] = sc
            x = np.arange(len(cats)); w = 0.26
            for i, (name, sc) in enumerate(series.items()):
                def get(cat, sc=sc):
                    return (sc.get(cat) if cat in ("offense_mcq", "general_mmlu")
                            else sc.get("neighbor_mmlu_by_subject", {}).get(cat))
                cax.bar(x + (i - 1) * w, [get(c) or 0 for c in cats], w, label=name)
            cax.axhline(CHANCE, ls="--", c="gray", lw=1)
            cax.set_title(f"{MLABEL[method]} — WMDP-{dom} (off. removed≈0.6)", fontsize=10)
            cax.set_xticks(x); cax.set_xticklabels(labels, fontsize=7); cax.set_ylim(0, 1.0)
            cax.legend(fontsize=8)
        axes[r][0].set_ylabel("accuracy")
    fig.suptitle("Near-domain over-removal: unlearning offense craters the legit near-neighbor "
                 "(targeted retain preserves it)", fontsize=11)
    fig.tight_layout()
    p2 = FIG_DIR / "entanglement_fig11.png"
    fig.savefig(p2, dpi=150); plt.close(fig); paths.append(str(p2))
    return paths


def main() -> None:
    pts, bases = load_points()
    if not bases:
        sys.exit("no base_<domain>.json found — run the sweep (strength 0) first.")
    methods = [m for m in ("rmu", "cb") if any(p["method"] == m for p in pts.values())]
    if not methods:
        sys.exit("no method points (rmu_*/cb_*) found.")

    L = ["# Entanglement unlearning-tax (RMU + Circuit Breakers)", ""]
    L.append("Safety tax = legitimate same-domain ability lost per unit offense removed. "
             "Methods: " + ", ".join(MLABEL[m] for m in methods) + ".")
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

    L.append("## Tax by method × domain × retain (slope; coherent points)")
    L.append("| method | domain | retain | tax | max offense removed |")
    L.append("|---|---|---|--:|--:|")
    for method in methods:
        for dom in DOMAINS:
            if dom not in bases:
                continue
            for retain in RETAINS:
                arm = f"{dom}_{retain}"
                if not curve(pts, bases, method, arm):
                    continue
                L.append(f"| {MLABEL[method]} | {dom} | {retain} | "
                         f"{fmt(tax(pts, bases, method, arm))} | "
                         f"{fmt(max_off(pts, bases, method, arm))} |")
    L.append("")

    # cross-domain asymmetry on the wikitext arm, per method (the claim figure backs)
    L.append("## Cross-domain asymmetry (wikitext arm)")
    for method in methods:
        tw = {d: tax(pts, bases, method, f"{d}_wikitext") for d in ("bio", "cyber")}
        if tw["bio"] is not None and tw["cyber"] is not None:
            more = "cyber" if tw["cyber"] > tw["bio"] else "bio"
            L.append(f"- **{MLABEL[method]}:** bio tax {fmt(tw['bio'])} vs cyber {fmt(tw['cyber'])} "
                     f"→ more entangled: **{more}**.")

    fig_paths = make_figures(pts, bases, methods)
    L.append("")
    L.append("## Figures")
    for p in fig_paths:
        L.append(f"\n![{Path(p).stem}]({Path(p).relative_to(ROOT / 'reports')})")

    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT.relative_to(ROOT)} + {len(fig_paths)} figures; methods={methods}")


if __name__ == "__main__":
    main()
