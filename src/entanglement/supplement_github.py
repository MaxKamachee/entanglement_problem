"""GitHub blue-team supplement (operational defense) — fetch repo markdown, tag into D3FEND tactics.

Replaces the Microsoft Security supplement. Brings operational defensive content into the defense
bucket from clean-licensed GitHub repos (the intended distribution channel), fetched as a **repo
tarball** (one request per repo). Each markdown doc is cleaned, content-filtered, and tagged with an
inferred **D3FEND tactic** (Model/Harden/Detect/Isolate/Deceive/Evict/Restore) so the whole defense
bucket shares one D3FEND-grounded taxonomy. Tactic is keyword-inferred (Detect-skewed, as real SOC work
is) — so we set `topic` only and leave authoritative `d3fend_ids` empty.

Sources (configs/corpus.yaml `github_supplements`): SOC-Assistant-Guide (MIT — detection/DFIR depth) +
BlueTeam-Tools (no license — tool-knowledge breadth, treated local-build-only / URL+hash release).
Text is not redistributed; the released artifact is URLs + content hashes + this pipeline.

Outputs: `inputs/github_blueteam_supplement.parquet` (full provenance), `data/defensive_supplement_github
.parquet` (build input: text+topic, read by `analysis_units._SOURCES`), and provenance rows appended to
`defensive_documents.parquet` (`source_category="supplement_github"`).
"""

from __future__ import annotations

import io
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from entanglement import config
from entanglement.scrape import content_hash
from entanglement.units import clean_text

ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "MATS-Research-Corpus-Build/1.0 (mkamachee; academic research)"
MIN_CHARS = 600                      # cleaned-text floor (drops empty/stub/TODO md)
_SKIP_PATH = re.compile(r"(?i)(^|/)(license|contributing|code_of_conduct|security|changelog|"
                        r"\.github|author|sponsors)\b")
_DEFENSE_TERMS = ["detect", "monitor", "hunt", "forensic", "malware", "incident", "siem", "log",
                  "threat", "harden", "yara", "sigma", "analysis", "vulnerab", "response", "alert"]

SUPPLEMENT_COLUMNS = ["repo", "path", "url", "fetch_timestamp", "original_title",
                      "topic", "text", "content_hash", "license"]
_LIST_COLS = ["d3fend_ids", "tactics", "parent_ids", "buckets"]

# D3FEND tactic <- keyword map (scanned over path + text; argmax; else general_defense).
_TACTIC_KEYWORDS: dict[str, list[str]] = {
    "Detect": ["detect", "detection", "hunt", "siem", "splunk", " kql", "sigma", "yara", "monitor",
               "telemetry", "forensic", "malware analysis", "analyze", "analysis", "alert", "ioc",
               "anomal", "investigat", "log analysis", "sysmon", " edr", " xdr", "triage", "packet"],
    "Harden": ["harden", "patch", "mitigat", " gpo", "baseline", "secure config", "disable",
               "prevent execution", "dmarc", "spf", "dkim", "least privilege", "attack surface reduc"],
    "Model": ["asset inventory", "network mapping", "network scan", "discovery", "enumerat",
              "threat model", "threat intelligence", "reconnaissance", "shodan", "nmap", "attack surface map"],
    "Isolate": ["isolat", "contain", "segmentation", "quarantine", "firewall rule", "network isolation"],
    "Evict": ["evict", "eradicat", "remove malware", "kill process", "delete malicious", "disable account"],
    "Restore": ["restore", "recovery", "recover deleted", "backup", "rebuild", "data recovery", "reimage"],
    "Deceive": ["honeypot", "deception", "decoy", "canary", "honeytoken"],
}


# --------------------------------------------------------------------------- pure helpers
def d3fend_tactic(text: str, path: str = "") -> str:
    blob = (path + "\n" + text).lower()
    scores = {t: sum(blob.count(k) for k in kws) for t, kws in _TACTIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_defense"


def passes_filter(path: str, text: str) -> bool:
    if _SKIP_PATH.search(path):
        return False
    if len(text) < MIN_CHARS:
        return False
    low = text.lower()
    return sum(1 for t in _DEFENSE_TERMS if t in low) >= 2


def _title(text: str, path: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if s:
            return s[:200]
    return path.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- network
def fetch_repo_markdown(owner: str, repo: str, ref: str, license_note: str, *, client) -> list[dict]:
    """Download a repo tarball (1 request) and return cleaned/filtered/tagged markdown docs."""
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{ref}"
    resp = client.get(url, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    rows, seen = [], set()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.lower().endswith(".md"):
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name   # strip "{repo}-{sha}/"
            raw = tf.extractfile(m)
            if raw is None:
                continue
            text = clean_text(raw.read().decode("utf-8", "ignore"))
            if not passes_filter(rel, text):
                continue
            h = content_hash(text)
            if h in seen:
                continue
            seen.add(h)
            rows.append({
                "repo": f"{owner}/{repo}",
                "path": rel,
                "url": f"https://github.com/{owner}/{repo}/blob/{ref}/{rel}",
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "original_title": _title(text, rel),
                "topic": d3fend_tactic(text, rel),
                "text": text,
                "content_hash": h,
                "license": license_note,
            })
    return rows


def fetch_all() -> pl.DataFrame:
    import httpx

    client = httpx.Client(headers={"user-agent": USER_AGENT})
    rows: list[dict] = []
    try:
        for r in config.github_supplements():
            docs = fetch_repo_markdown(r["owner"], r["repo"], r.get("ref", "main"),
                                       r.get("license", "unknown"), client=client)
            print(f"  {r['owner']}/{r['repo']}: {len(docs)} docs kept", flush=True)
            rows += docs
    finally:
        client.close()
    return pl.DataFrame(rows, schema={c: pl.String for c in SUPPLEMENT_COLUMNS})


# --------------------------------------------------------------------------- integrate
def integrate(df: pl.DataFrame) -> None:
    df.write_parquet(ROOT / "inputs" / "github_blueteam_supplement.parquet")
    # build input for analysis_units (text + topic)
    df.select("text", "topic", "url", "content_hash").write_parquet(
        ROOT / "data" / "defensive_supplement_github.parquet")

    # defensive_documents provenance: d3fend base (idempotent) + github rows
    dd = pl.read_parquet(ROOT / "data" / "defensive_documents.parquet")
    if "source_category" in dd.columns:
        base = dd.filter(pl.col("source_category") == "d3fend_cited").drop("source_category")
    else:
        base = dd
    base = base.with_columns(pl.lit("d3fend_cited").alias("source_category"))
    gh = df.select(
        pl.col("content_hash"),
        pl.col("url"),
        pl.col("url").alias("raw_url"),
        pl.lit("github_markdown").alias("extractor"),
        pl.col("text").str.len_chars().cast(pl.Int64).alias("n_chars"),
        *[pl.lit([], dtype=pl.List(pl.String)).alias(c) for c in _LIST_COLS],
        pl.col("text"),
        pl.lit("supplement_github").alias("source_category"),
    )
    pl.concat([base.select(gh.columns), gh], how="vertical").write_parquet(
        ROOT / "data" / "defensive_documents.parquet")


def main() -> None:
    df = fetch_all()
    if df.height == 0:
        raise SystemExit("no github supplement docs fetched")
    integrate(df)
    tac = dict(df.group_by("topic").len().sort("len", descending=True).iter_rows())
    print(f"github supplement: {df.height} docs ({df['text'].str.len_chars().sum():,} chars)")
    print(f"  D3FEND-tactic (topic) distribution: {tac}")
    print("wrote inputs/github_blueteam_supplement.parquet + data/defensive_supplement_github.parquet "
          "+ updated defensive_documents.parquet")


if __name__ == "__main__":
    main()
