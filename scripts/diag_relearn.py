#!/usr/bin/env python
"""Analyze the relearning experiment -> reports/relearn_recovery.md.

Reads runs/relearn/*.json (from run_relearn_sweep.py): per condition, a `series` of
{relearn_steps, offense_mcq, neighbor_mmlu} after RMU-unlearn (step 0) then LoRA-finetuning.

2nd entanglement measure: does offense recover, and how fast, per domain x relearn corpus?
  *_relretain = entanglement probe — offense reviving from LEGITIMATE same-domain text alone
                means offense and legitimate knowledge are entangled.
  *_relforget = adversarial robustness (paper Fig 15) — re-teaching the forbidden corpus.

Headline: is offense recovery (especially from retain-only FT) larger/faster in the more-entangled
domain (cyber)? Normalized to base offense where available (runs/entanglement/base_<domain>.json).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "relearn"
BASES = ROOT / "runs" / "entanglement"      # for base offense (optional normalization)
REPORT = ROOT / "reports" / "relearn_recovery.md"
FIG_DIR = ROOT / "reports" / "figures"
CHANCE = 0.25
CONDS = ["bio_relforget", "cyber_relforget", "bio_relretain", "cyber_relretain"]


def base_offense(domain: str) -> float | None:
    f = BASES / f"base_{domain}.json"
    if f.exists():
        return json.loads(f.read_text()).get("scores", {}).get("offense_mcq")
    return None


def load() -> dict:
    out = {}
    for name in CONDS:
        f = RUNS / f"{name}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[name] = {"domain": d.get("domain", name.split("_")[0]),
                         "corpus": name.split("_")[1].replace("rel", ""),
                         "series": d.get("series", [])}
    return out


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def make_figure(data: dict) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [("forget", "relearn on FORGET (adversarial / Fig 15)"),
              ("retain", "relearn on RETAIN-only (entanglement probe)")]
    colors = {"bio": "tab:green", "cyber": "tab:red"}
    drew = False
    for ax, (corpus, title) in zip(axes, panels):
        for dom in ("bio", "cyber"):
            cond = f"{dom}_rel{corpus}"
            if cond not in data:
                continue
            s = data[cond]["series"]
            xs = [p["relearn_steps"] for p in s]
            ys = [p["offense_mcq"] for p in s]
            if xs:
                ax.plot(xs, ys, "-o", color=colors[dom], label=dom); drew = True
            b = base_offense(dom)
            if b is not None:
                ax.axhline(b, ls=":", c=colors[dom], lw=1, alpha=0.6)
        ax.axhline(CHANCE, ls="--", c="gray", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relearn (LoRA finetune) steps")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("offense (WMDP MCQ) accuracy")
    fig.suptitle("Offense recovery under finetuning (dotted = base offense, dashed = chance)",
                 fontsize=11)
    fig.tight_layout()
    if not drew:
        plt.close(fig); return None
    p = FIG_DIR / "relearn_recovery.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return str(p)


def main() -> None:
    if not RUNS.exists():
        sys.exit(f"no {RUNS.relative_to(ROOT)} — run scripts/run_relearn_sweep.py on a pod first.")
    data = load()
    if not data:
        sys.exit(f"no relearn JSONs in {RUNS.relative_to(ROOT)}.")

    L = ["# Relearning: offense recovery under finetuning (2nd entanglement measure)", ""]
    L.append("After RMU-unlearn (step 0), LoRA-finetune and watch offense (WMDP MCQ) recover. "
             "`relforget` = re-teach the forbidden corpus (adversarial, paper Fig 15); `relretain` = "
             "finetune on **legitimate same-domain** text only — offense reviving from that is the "
             "entanglement signal. Recovery normalized to base offense where available.")
    L.append("")
    L.append("| condition | offense@unlearn | offense@final | neighbor@unlearn→final | "
             "offense recovered | frac→base |")
    L.append("|---|--:|--:|--:|--:|--:|")
    summ = {}
    for name in CONDS:
        if name not in data:
            continue
        s = data[name]["series"]
        if not s:
            continue
        dom = data[name]["domain"]
        o0, of = s[0]["offense_mcq"], s[-1]["offense_mcq"]
        n0, nf = s[0].get("neighbor_mmlu"), s[-1].get("neighbor_mmlu")
        rec = of - o0
        b = base_offense(dom)
        frac = (of - o0) / (b - o0) if (b is not None and b - o0 > 1e-6) else None
        summ[name] = {"o0": o0, "of": of, "rec": rec, "frac": frac, "dom": dom}
        L.append(f"| {name} | {fmt(o0)} | {fmt(of)} | {fmt(n0)}→{fmt(nf)} | {fmt(rec)} | {fmt(frac)} |")
    L.append("")

    L.append("## Headline")
    L.append("**Durability, not absolute recovery, is the entanglement metric** — absolute "
             "offense-recovered is confounded by how much offense each domain started with (bio's "
             "base is higher). The right question is whether the unlearning *stuck*: fraction of "
             "base offense recovered (→1.0 = fully reverses) and offense remaining below base.")
    L.append("")

    def cmp(corpus, label):
        bio, cyb = summ.get(f"bio_rel{corpus}"), summ.get(f"cyber_rel{corpus}")
        if not (bio and cyb) or bio["frac"] is None or cyb["frac"] is None:
            return
        more = "cyber" if cyb["frac"] > bio["frac"] else "bio"
        L.append(f"- **{label}:** fraction of base recovered — bio {fmt(bio['frac'])}, cyber "
                 f"{fmt(cyb['frac'])} → recovers more {'completely' if more=='cyber' else 'fully'} in "
                 f"**{more}** (absolute, for reference: bio {fmt(bio['rec'])}, cyber {fmt(cyb['rec'])}).")

    cmp("retain", "Entanglement probe (retain-only FT)")
    cmp("forget", "Adversarial recovery (forget FT)")
    if all(f"{d}_relretain" in summ for d in ("bio", "cyber")):
        bf, cf = summ["bio_relretain"]["frac"], summ["cyber_relretain"]["frac"]
        if bf is not None and cf is not None:
            L.append(f"- **Interpretation:** offense reviving from legitimate same-domain text alone "
                     f"is the entanglement signal. Cyber recovers {fmt(cf)} of base from retain-only "
                     f"FT vs bio {fmt(bf)} → {'SUPPORTS' if cf > bf else 'does NOT support'} cyber "
                     "being more entangled (consistent with the unlearning-tax result + WMDP Fig 15). "
                     "Caveat: cyber's base offense is near chance, so its small headroom makes the "
                     "fraction noisier (can exceed 1.0).")

    fig = make_figure(data)
    if fig:
        L.append("")
        L.append("## Figure")
        L.append(f"\n![relearn_recovery]({Path(fig).relative_to(ROOT / 'reports')})")

    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT.relative_to(ROOT)}" + (f" + figure" if fig else ""))
    for n, v in summ.items():
        print(f"  {n}: offense {v['o0']:.3f} -> {v['of']:.3f} (recovered {v['rec']:+.3f})")


if __name__ == "__main__":
    main()
