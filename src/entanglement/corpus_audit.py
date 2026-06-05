"""Post-prune corpus quality audit for `data/analysis_units.parquet`.

Pre-MCQ gate: before generating evaluation questions we need to know the assembled corpus is
substantive technical content, not noise / filler / extraction artifacts. This module produces a
single report (`reports/corpus_quality_audit.md`) with, per bucket (offense/dual/defense):

1. 20 randomly-sampled documents (full text) for manual review.
2. Document-level quality metrics (alpha ratio, sentence/paragraph counts, mean word length,
   boilerplate-phrase count) + a flagging rule; fraction flagged per bucket.
3. Topic distribution per (bucket, topic) — ATT&CK tactic (offense), D3FEND tactic (defense),
   substrate source-topic (dual) — plus source-corpus technique/D3FEND-id coverage.
4. Near-duplicate detection (MinHash + LSH over 5-word shingles); near-dup docs per bucket/source.
5. Length distribution per (bucket, topic-source); a per-bucket histogram + short-source flags.

Plus a data-driven recommendation: which bucket needs cleanup and what specifically.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = ("offense", "dual", "defense")
SEED = 0

_WORD = re.compile(r"[A-Za-z]+")
_SENT = re.compile(r"[.!?]+")
_BOILERPLATE = [
    "subscribe", "cookie", "404", "page not found", "login required", "sign in",
    "privacy policy", "all rights reserved", "terms of service", "accept all",
    "newsletter", "advertisement", "javascript is disabled", "enable javascript",
    "back to top", "share this", "follow us", "©",
]


# ---------------------------------------------------------------------------- metrics
def doc_metrics(text: str) -> dict:
    n = len(text)
    alpha = sum(c.isalpha() for c in text)
    words = _WORD.findall(text)
    sents = [s for s in _SENT.split(text) if s.strip()]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    low = text.lower()
    bp = sum(low.count(p) for p in _BOILERPLATE)
    n_words = len(words)
    n_sents = len(sents)
    return {
        "n_chars": n,
        "alpha_ratio": round(alpha / n, 3) if n else 0.0,
        "sentence_count": n_sents,
        "paragraph_count": len(paras),
        "mean_word_len": round(sum(len(w) for w in words) / n_words, 2) if n_words else 0.0,
        "mean_sentence_words": round(n_words / n_sents, 1) if n_sents else float(n_words),
        "boilerplate_count": bp,
    }


def flag_reasons(m: dict) -> list[str]:
    r = []
    if m["alpha_ratio"] < 0.65:
        r.append("low_alpha")
    if m["boilerplate_count"] >= 2:
        r.append("high_boilerplate")
    if m["sentence_count"] >= 4 and m["mean_sentence_words"] < 5:
        r.append("choppy_sentences")
    if m["n_chars"] < 250:
        r.append("tiny")
    return r


# ---------------------------------------------------------------------------- minhash
def _shingles(text: str, k: int = 5, cap: int = 512) -> np.ndarray:
    toks = _WORD.findall(text.lower())
    if len(toks) < k:
        grams = [" ".join(toks)] if toks else [text[:64]]
    else:
        grams = [" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]
    ids = {zlib.crc32(g.encode()) for g in grams}
    arr = np.fromiter(ids, dtype=np.uint64)
    if arr.size > cap:                      # bound work for very long docs
        arr = arr[:cap]
    return arr


def minhash_signatures(texts: list[str], n_perm: int = 128) -> np.ndarray:
    P = np.uint64((1 << 61) - 1)
    rng = np.random.default_rng(SEED)
    a = rng.integers(1, int(P), size=n_perm, dtype=np.uint64)
    b = rng.integers(0, int(P), size=n_perm, dtype=np.uint64)
    sigs = np.full((len(texts), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, t in enumerate(texts):
        s = _shingles(t)
        if not s.size:
            continue
        hashed = (a[:, None] * s[None, :] + b[:, None]) % P    # (n_perm, n_shingle)
        sigs[i] = hashed.min(axis=1)
    return sigs


def near_duplicate_pairs(sigs: np.ndarray, bands: int = 32, threshold: float = 0.7) -> list[tuple]:
    n, n_perm = sigs.shape
    rows = n_perm // bands
    candidates: set[tuple[int, int]] = set()
    for bnd in range(bands):
        sub = sigs[:, bnd * rows:(bnd + 1) * rows]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            buckets.setdefault(sub[i].tobytes(), []).append(i)
        for grp in buckets.values():
            if len(grp) < 2 or len(grp) > 400:        # skip degenerate giant buckets
                continue
            for x in range(len(grp)):
                for y in range(x + 1, len(grp)):
                    candidates.add((grp[x], grp[y]))
    pairs = []
    for i, j in candidates:
        est = float(np.mean(sigs[i] == sigs[j]))
        if est >= threshold:
            pairs.append((i, j, round(est, 3)))
    return pairs


def _union_find(n: int, pairs: list[tuple]) -> dict[int, int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return {i: find(i) for i in range(n)}


# ---------------------------------------------------------------------------- figure
def _length_figure(units: pl.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, b in zip(axes, BUCKETS):
        vals = units.filter(pl.col("bucket") == b)["n_chars"].to_list()
        ax.hist(vals, bins=40, color="#4c72b0", alpha=0.8)
        ax.axvline(float(np.median(vals)), ls="--", c="k", lw=1,
                   label=f"median {int(np.median(vals))}")
        ax.set_title(f"{b}  (n={len(vals)})")
        ax.set_xlabel("n_chars")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("documents")
    fig.suptitle("Document length distribution per bucket")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------- report
def _source_coverage() -> list[str]:
    """Top ATT&CK techniques / D3FEND ids from the source doc parquets (provenance enrichment)."""
    lines = []
    specs = [
        ("offensive_documents.parquet", "tech_ids", "ATT&CK techniques (offensive_documents)"),
        ("defensive_documents.parquet", "d3fend_ids", "D3FEND techniques (defensive_documents)"),
    ]
    for fname, col, label in specs:
        path = ROOT / "data" / fname
        if not path.exists():
            continue
        d = pl.read_parquet(path)
        if col not in d.columns:
            continue
        exploded = d.explode(col).drop_nulls(col)
        top = exploded.group_by(col).len().sort("len", descending=True).head(12)
        lines.append(f"**{label}** — {exploded[col].n_unique()} distinct; top 12:")
        lines.append("")
        lines.append("| id | docs |")
        lines.append("|---|---:|")
        for r in top.iter_rows(named=True):
            lines.append(f"| {r[col]} | {r['len']} |")
        lines.append("")
    return lines


def build_report(units: pl.DataFrame, out: Path) -> None:
    # ---- per-doc metrics over the FULL corpus (for flagging stats) ----
    metrics = [doc_metrics(t) for t in units["text"].to_list()]
    reasons = [flag_reasons(m) for m in metrics]
    mdf = units.select("bucket", "topic", "layer", "n_chars").with_columns(
        pl.Series("alpha_ratio", [m["alpha_ratio"] for m in metrics]),
        pl.Series("sentence_count", [m["sentence_count"] for m in metrics]),
        pl.Series("mean_sentence_words", [m["mean_sentence_words"] for m in metrics]),
        pl.Series("boilerplate_count", [m["boilerplate_count"] for m in metrics]),
        pl.Series("flagged", [bool(r) for r in reasons]),
        pl.Series("reasons", [",".join(r) for r in reasons]),
    )

    L = ["# Corpus quality audit — post-prune `analysis_units.parquet`", "",
         f"Seed {SEED}. Buckets: " + ", ".join(
             f"{b} {units.filter(pl.col('bucket') == b).height}" for b in BUCKETS)
         + f" (total {units.height}). Pre-MCQ gate — manual review + automated flags.", ""]

    # ---- 1. flagging summary ----
    L += ["## 1. Quality metrics & flagging", "",
          "Flag rules: `low_alpha` (alpha ratio <0.65), `high_boilerplate` (≥2 boilerplate "
          "phrases), `choppy_sentences` (≥4 sentences & mean <5 words/sentence), `tiny` (<250 chars).",
          "", "| bucket | flagged | % | low_alpha | high_boilerplate | choppy | tiny | "
          "median alpha | median sent.len |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in BUCKETS:
        sub = mdf.filter(pl.col("bucket") == b)
        sub_reasons = [r for r, bk in zip(reasons, units["bucket"].to_list()) if bk == b]
        flat = [x for r in sub_reasons for x in r]
        nflag = sum(1 for r in sub_reasons if r)
        L.append(
            f"| {b} | {nflag} | {100 * nflag / sub.height:.1f}% | "
            f"{flat.count('low_alpha')} | {flat.count('high_boilerplate')} | "
            f"{flat.count('choppy_sentences')} | {flat.count('tiny')} | "
            f"{sub['alpha_ratio'].median():.3f} | {sub['mean_sentence_words'].median():.1f} |")
    L.append("")

    # ---- 2. topic distribution ----
    L += ["## 2. Topic distribution per (bucket, topic)", "",
          "offense topic = ATT&CK tactic · defense topic = D3FEND tactic · dual topic = substrate source.",
          "", "| bucket | topic | docs |", "|---|---|---:|"]
    for b in BUCKETS:
        t = (units.filter(pl.col("bucket") == b).group_by("topic").len()
             .sort("len", descending=True))
        for r in t.iter_rows(named=True):
            L.append(f"| {b} | {r['topic']} | {r['len']} |")
    L += ["", "### Source-corpus provenance coverage (pre-resegmentation)", ""]
    L += _source_coverage()

    # ---- 3. near-duplicates ----
    sigs = minhash_signatures(units["text"].to_list())
    pairs = near_duplicate_pairs(sigs)
    comp = _union_find(units.height, pairs)
    bucket_arr = units["bucket"].to_list()
    topic_arr = units["topic"].to_list()
    # a doc is a near-dup if it shares a component with >=1 other doc
    from collections import Counter
    comp_sizes = Counter(comp.values())
    dup_idx = [i for i in range(units.height) if comp_sizes[comp[i]] > 1]
    L += ["## 3. Near-duplicates (MinHash/LSH, 5-word shingles, est. Jaccard ≥0.7)", "",
          f"{len(pairs)} near-duplicate pairs; **{len(dup_idx)} documents** "
          f"({100 * len(dup_idx) / units.height:.1f}%) sit in a near-dup cluster.", "",
          "| bucket | near-dup docs | % of bucket | top topic-source |", "|---|---:|---:|---|"]
    for b in BUCKETS:
        idxs = [i for i in dup_idx if bucket_arr[i] == b]
        bn = sum(1 for x in bucket_arr if x == b)
        topc = Counter(topic_arr[i] for i in idxs).most_common(1)
        top = f"{topc[0][0]} ({topc[0][1]})" if topc else "—"
        L.append(f"| {b} | {len(idxs)} | {100 * len(idxs) / bn:.1f}% | {top} |")
    L.append("")

    # ---- 4. length distribution ----
    fig = ROOT / "reports" / "figures" / "corpus_quality_lengths.png"
    _length_figure(units, fig)
    L += ["## 4. Length distribution", "", f"![lengths](figures/{fig.name})", "",
          "Per (bucket, topic-source) length percentiles — short medians flag possible extraction failures:",
          "", "| bucket | topic | n | p10 | median | p90 |", "|---|---|---:|---:|---:|---:|"]
    pct = (units.group_by("bucket", "topic").agg(
        pl.len().alias("n"),
        pl.col("n_chars").quantile(0.10).alias("p10"),
        pl.col("n_chars").median().alias("p50"),
        pl.col("n_chars").quantile(0.90).alias("p90"),
    ).sort(["bucket", "p50"]))
    for r in pct.iter_rows(named=True):
        L.append(f"| {r['bucket']} | {r['topic']} | {r['n']} | {int(r['p10'])} | "
                 f"{int(r['p50'])} | {int(r['p90'])} |")
    L.append("")

    # ---- 5. recommendation (data-driven) ----
    flagged_pct = {b: 100 * sum(1 for r, bk in zip(reasons, bucket_arr) if bk == b and r)
                   / sum(1 for x in bucket_arr if x == b) for b in BUCKETS}
    dup_pct = {b: 100 * sum(1 for i in dup_idx if bucket_arr[i] == b)
               / sum(1 for x in bucket_arr if x == b) for b in BUCKETS}
    worst_flag = max(BUCKETS, key=lambda b: flagged_pct[b])
    worst_dup = max(BUCKETS, key=lambda b: dup_pct[b])
    L += ["## 5. Recommendation", "",
          f"- Highest flagged fraction: **{worst_flag}** ({flagged_pct[worst_flag]:.1f}%).",
          f"- Highest near-dup fraction: **{worst_dup}** ({dup_pct[worst_dup]:.1f}%).",
          "- Per-bucket flagged%: " + ", ".join(f"{b} {flagged_pct[b]:.1f}%" for b in BUCKETS)
          + "; near-dup%: " + ", ".join(f"{b} {dup_pct[b]:.1f}%" for b in BUCKETS) + ".",
          "", "_Manual review of the §6 samples should confirm whether flagged docs are genuine "
          "noise (drop) or false positives (symbol-dense but substantive, e.g. crypto/RFC notation)._", ""]

    # ---- 6. sampled full text ----
    L += ["## 6. Sampled documents (20/bucket, full text)", ""]
    for b in BUCKETS:
        sub = units.filter(pl.col("bucket") == b).sample(
            n=min(20, units.filter(pl.col("bucket") == b).height), seed=SEED)
        L.append(f"### {b}")
        L.append("")
        for r in sub.iter_rows(named=True):
            m = doc_metrics(r["text"])
            fr = flag_reasons(m)
            L.append(f"**{r['unit_id'][:12]}** · topic={r['topic']} · {m['n_chars']}c · "
                     f"alpha={m['alpha_ratio']} · sents={m['sentence_count']} · "
                     f"bp={m['boilerplate_count']}" + (f" · ⚠ {','.join(fr)}" if fr else ""))
            L.append("")
            L.append("```")
            L.append(r["text"])
            L.append("```")
            L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))


def main() -> None:
    units = pl.read_parquet(ROOT / "data" / "analysis_units.parquet")
    out = ROOT / "reports" / "corpus_quality_audit.md"
    build_report(units, out)
    print(f"wrote {out.relative_to(ROOT)} ({units.height} docs audited)")


if __name__ == "__main__":
    main()
