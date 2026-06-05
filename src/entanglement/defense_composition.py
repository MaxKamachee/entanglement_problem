"""Defense corpus composition by source → reports/defense_composition.md.

Determines whether the defensive external corpus is broadly *defensive/mechanism* content or
*compliance/policy*-flavored (which shapes how we frame the contribution claim). Classifies each
source document in `defensive_documents.parquet` by URL, and reports both **document-level** and
**unit-level** breakdowns — the distinction matters because a handful of NIST SP 800-series
standards are huge and resegment into many analysis units, so they can dominate the unit corpus
while being a small fraction of documents.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from entanglement.units import clean_text, resegment

ROOT = Path(__file__).resolve().parents[2]


def classify_defense_source(url: str) -> str:
    u = url.lower()
    host = urlparse(u).netloc.replace("www.", "")
    if "soc-assistant-guide" in u or "blueteam-tools" in u:
        return "SUPPLEMENT_GITHUB"
    if "nist.sp.800" in u or ("nist" in u and "800-" in u):
        return "NIST_SP_800"
    if "ietf.org" in host or "rfc-editor" in host:
        return "IETF_RFC"
    if "cwe.mitre.org" in host or "capec.mitre.org" in host:
        return "MITRE_CWE"
    if host in {"cisa.gov", "ic3.gov", "dni.gov", "dhs.gov", "jhuapl.edu"} or host.endswith(".gov"):
        return "US_GOV_CISA"
    if ("doi.org" in host or host.endswith(".edu") or "eprints" in host or "sei.cmu.edu" in host
            or "nebelwelt" in host or "arxiv" in host):
        return "ACADEMIC"
    if host in {"docs.microsoft.com", "learn.microsoft.com", "cisco.com", "ssh.com",
                "clang.llvm.org", "3ds.com"} or "microsoft.com" in host:
        return "VENDOR"
    return "OTHER"


def build_report(docs: pl.DataFrame, out: Path) -> None:
    rows = []
    for r in docs.iter_rows(named=True):
        cat = classify_defense_source(r["url"])
        n_units = len(resegment(clean_text(r["text"] or ""), target=3000, hard_max=4000))
        rows.append({"url": r["url"], "category": cat, "src_chars": r["n_chars"], "n_units": n_units})
    df = pl.DataFrame(rows)
    total_docs, total_units = df.height, int(df["n_units"].sum())

    agg = (df.group_by("category").agg(
        pl.len().alias("docs"),
        pl.col("n_units").sum().alias("units"),
        pl.col("src_chars").median().alias("median_src_chars"),
    ).sort("units", descending=True))

    L = ["# Defense corpus composition by source", "",
         f"Source: `defensive_documents.parquet` — **{total_docs} documents** → **{total_units} "
         "analysis units** (resegmented at ~3000 chars). Classified by URL. The unit column is what "
         "actually enters the experiments; NIST SP 800-series standards are large and dominate units "
         "while being few documents.", "",
         "| category | docs | % docs | units | % units | median src chars |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in agg.iter_rows(named=True):
        L.append(f"| {r['category']} | {r['docs']} | {100 * r['docs'] / total_docs:.0f}% | "
                 f"{r['units']} | {100 * r['units'] / total_units:.0f}% | {int(r['median_src_chars'])} |")

    nist = agg.filter(pl.col("category") == "NIST_SP_800")
    nist_docs = int(nist["docs"][0]) if nist.height else 0
    nist_units = int(nist["units"][0]) if nist.height else 0
    nist_unit_pct = 100 * nist_units / total_units if total_units else 0.0
    policy_units = int(df.filter(pl.col("category").is_in(["NIST_SP_800", "US_GOV_CISA"]))["n_units"].sum())
    policy_pct = 100 * policy_units / total_units if total_units else 0.0

    L += ["", "## Headline", "",
          f"- **NIST SP 800-series: {nist_docs}/{total_docs} documents ({100 * nist_docs / total_docs:.0f}%) "
          f"but {nist_units}/{total_units} units ({nist_unit_pct:.0f}%).**",
          f"- Compliance/policy-flavored sources (NIST SP + US-gov/CISA): **{policy_pct:.0f}% of units**.",
          "- Median source length by category: " + ", ".join(
              f"{r['category']} {int(r['median_src_chars'])}c" for r in agg.iter_rows(named=True)) + ".",
          "", "## Verdict", "",
          ("The defense unit corpus is **compliance/policy-dominated**: a few large NIST SP 800-series "
           "standards (plus US-gov/CISA) supply the majority of units. The contribution claim should be "
           "framed as *defensive-governance/compliance* content, not broad operational defensive "
           "mechanism knowledge — or the corpus should be rebalanced toward mechanism-level defensive "
           "sources before strong claims."
           if nist_unit_pct >= 40 or policy_pct >= 50 else
           "The defense unit corpus is **broadly defensive**: NIST SP / policy sources are present but do "
           "not dominate units; vendor-hardening, RFC, academic, and CWE mechanism content carry "
           "substantial weight. The 'defensive capability' framing is defensible, with the caveat that "
           "depth per D3FEND technique is thin (see audit)."), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out.relative_to(ROOT)} | NIST units {nist_unit_pct:.0f}% | policy units {policy_pct:.0f}%")


def main() -> None:
    docs = pl.read_parquet(ROOT / "data" / "defensive_documents.parquet")
    build_report(docs, ROOT / "reports" / "defense_composition.md")


if __name__ == "__main__":
    main()
