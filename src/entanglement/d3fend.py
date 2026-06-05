"""Parse the D3FEND OWL into normalized reference + prose tables.

The defensive side of the Venn. Two products:

* ``d3fend_refs`` — one row per kb-reference URL (``has-link``), tagged with a
  coarse source bucket. This is the D3FEND half of the dual-use URL
  intersection.
* ``d3fend_prose`` — in-hand MITRE-authored defensive text (``definition`` +
  ``kb-abstract``), with char counts for the corpus-volume tally.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import polars as pl
import rdflib

from entanglement.config import load_vendor_hosts
from entanglement.normalize import canonicalize_url

D3F = "http://d3fend.mitre.org/ontologies/d3fend.owl#"
RDFS_LABEL = rdflib.RDFS.label
RDFS_SUBCLASS = rdflib.RDFS.subClassOf

# Security-vendor / threat-intel hosts; methodology choice, loaded from config so
# reviewers can audit it. See configs/vendor_hosts.yaml.
VENDOR_HOSTS: tuple[str, ...] = tuple(load_vendor_hosts())


def _p(name: str) -> rdflib.URIRef:
    return rdflib.URIRef(D3F + name)


def classify_source(url: str, vendor_hosts: tuple[str, ...] | list[str] = VENDOR_HOSTS) -> str:
    """Coarse source bucket from the URL host.

    Buckets: PATENT, MITRE_CAR, ATTACK_XREF, NIST, RFC, ACADEMIC, VENDOR, OTHER. The VENDOR
    check is positive (known security-vendor hosts) and runs last before the
    OTHER fallback, so the five pre-existing buckets are unaffected and only
    genuinely-unknown hosts remain in OTHER.
    """
    host = urlsplit(url).netloc.lower()
    if ("patents.google" in host or host.endswith("patentscope.wipo.int")
            or "patentimages.storage.googleapis" in host or "patentguru" in host):
        return "PATENT"
    if "car.mitre.org" in host:
        return "MITRE_CAR"
    if "attack.mitre.org" in host:
        return "ATTACK_XREF"   # out-of-framework ATT&CK content (no-crosswalks) — excluded
    if "nist.gov" in host or "csrc.nist" in host:
        return "NIST"
    if host.endswith("rfc-editor.org") or "ietf.org" in host:
        return "RFC"
    if any(h in host for h in ("arxiv.org", "doi.org", "mdpi.com", "ieee", "acm.org",
                               "springer", "usenix.org", "sciencedirect")):
        return "ACADEMIC"
    if any(h in host for h in vendor_hosts):
        return "VENDOR"
    return "OTHER"


def _local(uri: rdflib.term.Node) -> str:
    return str(uri).split("#")[-1]


def _load_graph(owl_path: str | Path) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(owl_path))
    return g


def build_d3fend_tables(
    owl_path: str | Path,
    graph: rdflib.Graph | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (d3fend_refs, d3fend_prose) as polars DataFrames.

    ``graph`` lets a caller reuse an already-parsed OWL graph (the 6 MB parse is
    the expensive step); when omitted the file is parsed here.
    """
    g = graph if graph is not None else _load_graph(owl_path)

    # --- references (has-link URLs) ---
    ref_rows: list[dict] = []
    for ref_node, link in g.subject_objects(_p("has-link")):
        url = canonicalize_url(str(link))
        if not url:
            continue
        title = g.value(ref_node, _p("kb-reference-title"))
        # concept(s) this reference supports
        concept = g.value(ref_node, _p("kb-reference-of"))
        concept_label = g.value(concept, RDFS_LABEL) if concept else None
        concept_id = g.value(concept, _p("d3fend-id")) if concept else None
        ref_rows.append(
            {
                "ref_id": _local(ref_node),
                "url": url,
                "raw_url": str(link),
                "title": str(title) if title else "",
                "concept_id": str(concept_id) if concept_id else "",
                "concept_label": str(concept_label) if concept_label else "",
                "bucket": classify_source(url),
            }
        )

    # --- prose: definitions on d3fend-id'd concepts + kb-abstracts ---
    prose_rows: list[dict] = []
    for subj, did in g.subject_objects(_p("d3fend-id")):
        defn = g.value(subj, _p("definition"))
        if defn and str(defn).strip():
            label = g.value(subj, RDFS_LABEL)
            prose_rows.append(
                {
                    "subject_id": str(did),
                    "label": str(label) if label else _local(subj),
                    "kind": "definition",
                    "text": str(defn),
                    "n_chars": len(str(defn)),
                }
            )
    for subj, abstract in g.subject_objects(_p("kb-abstract")):
        text = str(abstract).strip()
        if not text:
            continue
        label = g.value(subj, _p("kb-reference-title")) or g.value(subj, RDFS_LABEL)
        prose_rows.append(
            {
                "subject_id": _local(subj),
                "label": str(label) if label else _local(subj),
                "kind": "kb_abstract",
                "text": text,
                "n_chars": len(text),
            }
        )

    refs_df = pl.DataFrame(ref_rows).unique(subset=["ref_id", "url"])
    prose_df = pl.DataFrame(prose_rows)
    return refs_df, prose_df


def build_d3fend_hierarchy(
    owl_path: str | Path,
    graph: rdflib.Graph | None = None,
) -> pl.DataFrame:
    """Return [d3fend_id, label, parent_id] from the OWL subClassOf tree.

    A D3FEND concept's parent is the nearest ``rdfs:subClassOf`` ancestor that
    also carries a ``d3fend-id`` (skipping anonymous/tactic classes). Top-level
    techniques have no d3fend-id'd parent and get ``parent_id = self`` (mirrors
    the ATT&CK ``parent_tech_id`` convention where a top-level id is its own
    parent). The handful of multi-inheritance nodes take their first parent by
    sorted id, for determinism.
    """
    g = graph if graph is not None else _load_graph(owl_path)

    id_of: dict[rdflib.term.Node, str] = {
        subj: str(did) for subj, did in g.subject_objects(_p("d3fend-id"))
    }
    rows: list[dict] = []
    for subj, did in id_of.items():
        parents = sorted(
            id_of[par] for par in g.objects(subj, RDFS_SUBCLASS) if par in id_of
        )
        label = g.value(subj, RDFS_LABEL)
        rows.append(
            {
                "d3fend_id": did,
                "label": str(label) if label else _local(subj),
                "parent_id": parents[0] if parents else did,
            }
        )
    return pl.DataFrame(rows).unique(subset=["d3fend_id"], keep="first")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    owl = root / "inputs" / "d3fend.owl"
    out = root / "data"
    out.mkdir(exist_ok=True)

    g = _load_graph(owl)  # parse once, reuse for all three products
    refs_df, prose_df = build_d3fend_tables(owl, graph=g)
    hierarchy_df = build_d3fend_hierarchy(owl, graph=g)
    refs_df.write_parquet(out / "d3fend_refs.parquet")
    prose_df.write_parquet(out / "d3fend_prose.parquet")
    hierarchy_df.write_parquet(out / "d3fend_hierarchy.parquet")

    print(f"reference URLs:    {refs_df.height} "
          f"({refs_df['url'].n_unique()} distinct)")
    print("ref buckets:")
    for row in (refs_df.group_by("bucket").len().sort("len", descending=True)
                .iter_rows(named=True)):
        print(f"   {row['bucket']:10s} {row['len']}")
    by_kind = prose_df.group_by("kind").agg(
        pl.len().alias("n"), pl.col("n_chars").sum().alias("chars")
    )
    print("prose:")
    for row in by_kind.iter_rows(named=True):
        print(f"   {row['kind']:12s} {row['n']:5d} entries  {row['chars']:>10,} chars")
    print(f"   {'TOTAL':12s} {prose_df.height:5d} entries  "
          f"{prose_df['n_chars'].sum():>10,} chars")
    n_with_parent = int((hierarchy_df["parent_id"] != hierarchy_df["d3fend_id"]).sum())
    print(f"hierarchy:         {hierarchy_df.height} concepts "
          f"({n_with_parent} with a d3fend-id parent)")


if __name__ == "__main__":
    main()
