Do we # Corpus Card — Offensive/Defensive Cyber-Capability Entanglement Corpus

*Modeled on Datasheets for Datasets (Gebru et al.) and HuggingFace dataset cards.
**Version: v1 — all corpus construction complete** (in-hand + external layers for offensive
and defensive, plus the dual/substrate bucket). Generated 2026-06-03, completed 2026-06-04.*


## Motivation

Two technique-tagged text corpora — one offensive, one defensive — built **only** from
MITRE ATT&CK (enterprise) and MITRE D3FEND, with no crosswalks (no NIST 800-53, no
expert-named external sources, no out-of-framework ATT&CK Mitigations content). The corpora
support research on **cyber-capability entanglement**: how offensive and defensive
capability co-occur and separate in language-model-relevant text. Offensive ↔ defensive
pairing is studied through D3FEND's mappings, which target enterprise ATT&CK — hence the
enterprise-only domain scope (see Limitations).

## Composition — two corpora, two layers each

Each corpus has an **in-hand layer** (framework-authored text, no network, fully
reproducible) and an **external-literature layer** (third-party documents the frameworks
cite, scraped and filtered).

### Offensive corpus

| Layer | Source | Status |
|---|---|---|
| In-hand: procedure examples | STIX `uses` relationship descriptions (enterprise ATT&CK) | **Built** |
| External: cited literature | `external_references` URLs, **ATT&CK-ontology object-type selection** (keep technique/software/procedure; drop group/campaign-only; excl. mitigations + self) → scrape | **Built (2,704 docs)** |

- **`attack_procedures.parquet`** — **16,017** procedure examples across **599** techniques,
  **3,210,380** chars. One row per (source-object, technique) `uses` relationship carrying a
  non-empty description. Source objects: malware (9,836), intrusion-set (4,362), campaign
  (1,019), tool (800). Tagged per row: `tech_id`, `parent_id`, `tactics`, `domain`
  (`enterprise`), `source_ref`, `source_type`, `source_name`.

### Defensive corpus

| Layer | Source | Status |
|---|---|---|
| In-hand: D3FEND prose | `d3f:definition` + `d3f:kb-abstract` from the OWL | **Built** |
| External: cited literature | `d3f:kb-reference → d3f:has-link` URLs, **curated bucket keep** (NIST/RFC/ACADEMIC/VENDOR + curated OTHER) → scrape | **Built (61 docs)** |

- **`d3fend_prose_tagged.parquet`** — **828** prose rows (**493** definitions + **335**
  kb-abstracts), **292,894** chars. Tagged with D3FEND `tactic`
  (Model/Harden/Detect/Isolate/Deceive/Evict/Restore) and `parent_id`. Tactic maps cleanly
  for the **271** CSV-listed D3FEND techniques; parent (from the OWL `subClassOf` hierarchy)
  is set for all **493** definition rows. kb-abstract rows are reference-keyed and carry null
  tactic/parent by design.

## Source list — licensing & redistribution

| Source | Version | License / terms | Redistribution |
|---|---|---|---|
| MITRE ATT&CK STIX (enterprise) | 18.1 (pinned, see `inputs/SOURCES.md`) | Apache-2.0 / MITRE ATT&CK Terms of Use | Framework text redistributable with attribution |
| MITRE D3FEND OWL + CSV | 1.4.0 (pinned) | MITRE D3FEND Terms of Use | Framework text redistributable with attribution |
| Scraped third-party documents (external layers) | per-URL | **Mixed / mostly unknown** per host | **Likely restricted** — see note |

> **Redistribution note (external layers):** scraped full text is third-party content under
> heterogeneous, mostly-unknown licenses. The intended release ships **URL + content SHA-256
> + tags + provenance** for external documents, **not** the raw text, unless a per-host
> license permits redistribution. The in-hand layers (ATT&CK/D3FEND framework text) are
> redistributable with attribution.

## Per-corpus volume statistics

| Corpus | Layer | Docs | Chars | Status |
|---|---|---|---|---|
| Offensive | procedures (in-hand) | 16,017 | 3,210,380 | Built |
| Offensive | cited literature (external) | 2,704 | 55,547,488 | **Built** |
| Defensive | D3FEND prose (in-hand) | 828 | 292,894 | Built |
| Defensive | cited literature (external) | 61 | 8,558,771 | **Built** |
| Dual/substrate | curated sources (in-hand) | 2,670 chunks | 7,719,504 | Built |

Per-tactic breakdown (defensive in-hand, definitions with a mapped tactic): Detect 90,
Isolate 57, Harden 55, Model 27, Evict 19, Restore 12, Deceive 11. Per-domain (offensive):
enterprise (ICS excluded; schema retains `domain` for later ICS). Token counts: TODO (post-tokenizer).

**Scrape attrition (provenance `reason` tallies).** Offensive: 3,767 URLs → 2,805 fetched OK →
2,704 unique docs (74% success); drops: stub 369, dead 297, robots_disallow 118, non_text 85,
errors 93. Defensive: 141 URLs → 61 docs (43% success); drops dominated by **robots_disallow 50**
(.gov/standards/vendor hosts block bots), stub 13, non_text 6, errors 11, dead 4.

**Offense/defense external asymmetry is now concrete and severe:** 2,704 vs 61 docs (~44:1 by count;
~6.5:1 by chars, since defensive docs are large standards/specs). Combined with the in-hand asymmetry,
this is the central methodological feature of the corpus — see Known limitations. (2 documents are
shared offense↔defense by content hash — near-zero textual dual-use, as expected.)

## External reference inventory (pre-scrape)

D3FEND `kb-reference` URLs by source bucket — drives the defensive external scrape:

| Bucket | Refs | Disposition |
|---|---|---|
| PATENT | 148 | DROP (legalese; incl. patentimages/patentguru reclassified out of OTHER) |
| OTHER | 95 | **CURATED-KEEP** — substantive kept (82), noise hosts dropped (`defense_other_drop_hosts`) |
| MITRE_CAR | 89 | DROP (stub content) |
| VENDOR | 24 | KEEP |
| ACADEMIC | 15 | KEEP |
| NIST | 11 | KEEP |
| RFC | 10 | KEEP |
| ATTACK_XREF | 2 | DROP (attack.mitre.org — no-crosswalks) |

**Curated defensive external selection: 141 distinct URLs** (60 named-bucket + 82 curated OTHER −
denylisted noise; see `reports/defensive_bucket_review.md`). Decision: curated keep (2026-06-04).

Offensive external URL inventory (`offensive_refs.parquet`, 3,885 distinct canonical URLs by citing
type — technique 2,163 · procedure 1,788 · software 1,085 · group 701 · campaign 101): selection keeps
refs cited by technique/software/procedure and drops the 118 cited *only* by group/campaign →
**3,767 URLs** to scrape. Excludes mitigations + `mitre-attack` self-refs. Replaces the WMDP
construction-filter (see Known limitations).

## Known limitations

- **Defensive thinness.** Removing PATENT/MITRE_CAR/ATTACK_XREF leaves ~60 named-bucket refs; the
  **curated OTHER keep** (2026-06-04) folds in ~82 substantive OTHER refs (TPM/CET/CWE/STIGs/CISA/
  academic), dropping ~13 noise hosts → **141 curated defensive refs**. Still modest vs offensive.
- **Framework-bounded coverage.** Only ATT&CK + D3FEND; no crosswalks. Capability that no
  framework cites is invisible to this corpus by construction.
- **Enterprise-only.** ICS and mobile excluded. D3FEND maps to enterprise ATT&CK, so ICS
  offensive techniques would be entanglement-orphans. The pipeline is domain-aware; ICS can
  be added later.
- **VENDOR bucket is a heuristic allowlist** (`configs/vendor_hosts.yaml`) and will
  under-capture long-tail threat-intel hosts.
- **Offensive selection is structural, not operational.** The WMDP LLM-judge filter was **replaced**
  by deterministic ATT&CK ontology object-type selection (keep technique/software/procedure refs, drop
  group/campaign-only). This is reproducible and LLM-free but *gentle* (~5% drop) — the offensive
  external set still contains non-operational vendor-blog/news that techniques cite. An operational-depth
  axis is recoverable later via the (now-optional, unwired) `quality.py` rater if an experiment needs it.
- **Scrape attrition (expected, post-scrape).** Paywalled/login hosts (IEEE, ScienceDirect,
  some patents) will 403; stubs <2000 chars dropped. The provenance tables record every drop.

## Intended uses / out-of-scope uses

- **Intended:** research on offensive/defensive capability entanglement in LMs; measuring
  capability co-occurrence/separation; methodology for framework-grounded corpus construction.
- **Out of scope:** operational offensive use; as a how-to source for conducting attacks. The
  WMDP rubric exists to *measure* operational capability for research, not to curate an
  attack toolkit. Redistribution of scraped offensive full text is intentionally avoided.

## Per-document metadata schema

**`attack_procedures.parquet`** (offensive in-hand) — `PROCEDURE_COLUMNS`:
`proc_id` (16-char sha256 of source_ref|target_ref), `tech_id`, `parent_id`,
`tactics` (list[str]), `domain`, `source_ref`, `source_type`, `source_name`, `text`, `n_chars`.

**`d3fend_prose_tagged.parquet`** (defensive in-hand) — `DEFENSE_PROSE_COLUMNS`:
`tech_id` (D3FEND id, or ref-local-name for kb-abstracts), `label`, `kind`
(`definition`|`kb_abstract`), `tactic` (nullable), `parent_id` (nullable), `text`, `n_chars`.

**`offensive_documents.parquet`** (external) — `OFFENSIVE_DOC_COLUMNS`:
`content_hash`, `url`, `raw_url`, `extractor`, `n_chars`, `tech_ids`, `parent_ids`, `tactics`,
`citing_types` (technique/software/procedure/…), `domain`, `text`. (No WMDP score — replaced by
ontology object-type selection.)

**`defensive_documents.parquet`** (external) — `DEFENSE_DOC_COLUMNS`:
`content_hash`, `url`, `raw_url`, `extractor`, `n_chars`, `d3fend_ids`, `tactics`, `parent_ids`,
`buckets`, `text`.

**`{offensive,defensive}_provenance.parquet`** — `PROVENANCE_COLUMNS`:
`url`, `raw_url`, `fetch_ts`, `http_status`, `content_hash`, `extractor`, `n_chars`,
`success`, `reason` (`""`|`robots_disallow`|`dead`|`non_text`|`stub`|`error:<type>`).

**`d3fend_other_sample.parquet`** — OTHER-bucket review sample: `url`, `title`, `concept_label`.

**`substrate_corpus.parquet`** (dual bucket — provenance) — `SUBSTRATE_COLUMNS`:
`chunk_id` (sha256 of text = dedup key), `source_ids` (list — cross-source dups tag both),
`source_name`, `topic` (∈ networking/os_internals/architecture/web/crypto/recon), `version_pin`,
`text`, `n_chars`, `chunk_position` (structural locator), `extractor`, `extraction_warnings` (list),
`license`.

**`substrate_units.parquet`** (normalized analysis units) — `UNIT_COLUMNS`:
`unit_id` (sha256 of normalized text), `parent_id` (source `chunk_id` — provenance link),
`source_ids`, `source_name`, `topic`, `version_pin`, `unit_position` (`<chunk_position>#<window>`),
`n_chars`, `text` (format-stripped + de-wrapped + re-windowed), `license`. Derived from the provenance
corpus for the separability experiments; see Preprocessing.

---

# Substrate corpus — the "dual" bucket (offense / DUAL / defense)

## Motivation
The third bucket of the offense/dual/defense Venn. **Substrate = operational prerequisite knowledge**
— networking, OS, architecture, web, crypto, and recon *mechanism* documentation that both offensive
and defensive practitioners depend on. The contribution this corpus enables is the claim that **the
shared substrate is the locus of offense/defense entanglement** — dual-use *by application* (neutral
content, dual use), as opposed to dual-use-in-content (which our offense/defense citation overlap
showed is nearly empty: 1 shared URL of 2,163). It differs from `wmdp-cyber-retain` and generic retain
sets by being *topic-curated, mechanism-level, and provenance-pinned*, so the dual-use behavior is an
empirical output of topic-defined input rather than a labeling assumption.

## Composition (per-source — grown one unit at a time)

| Source | id | type | topic(s) | chunks | chars | license / redistribution | status |
|---|---|---|---|---|---|---|---|
| IETF RFCs (9293/1035/8446/9110/4271/4253) | `rfcs` | rfc_text | networking, web, crypto | 610 | 1,439,526 | IETF Trust BCP 78 — **redistributable** w/ attribution | ✅ built |
| FIPS 197 (AES) | `fips197` | pdf_single | crypto | 47 | 61,323 | US-gov **public domain** | ✅ built |
| MDN Web Security | `mdn_web_security` | github_markdown | web | 44 | 409,452 | CC-BY-SA 2.5 — **redistributable** | ✅ built |
| Nmap Network Scanning | `nmap_book` | html_book_chapters | recon | 88 | 770,441 | free-to-read — **pointer+hash only** | ✅ built |
| OSTEP | `ostep` | pdf_chapter_collection | os_internals | 840 | 1,448,274 | free-to-read — **pointer+hash only** | ✅ built |
| Boneh & Shoup Applied Cryptography (v0.6) | `boneh` | pdf_single (full book) | crypto | 751 | 3,021,799 | free-to-read — **pointer+hash only** | ✅ built |
| Intel SDM Vol 3A (conceptual CH2–7) | `intel_sdm` | pdf_chapter_extract | architecture | 290 | 568,689 | Intel © — **pointer+hash only** | ✅ built |

**Per-source dual-use rationale (one sentence each):**
- **`rfcs`** — Canonical protocol mechanism specs (TCP/DNS/TLS/HTTP/BGP/SSH) are read identically by
  attackers crafting/abusing protocol behavior and defenders building detection and hardening; the
  spec itself carries no valence, only the application does.
- **`fips197`** — The AES specification is the shared cryptographic primitive that attackers analyze for
  weaknesses/side-channels and defenders implement for confidentiality; the algorithm description is
  valence-neutral mechanism that both sides must understand to operate on it.
- **`mdn_web_security`** — MDN's web-security docs describe the browser security models (same-origin,
  CORS, CSP, passkeys) *and* the attacks against them (clickjacking, CSRF, MITM) in one place — the same
  mechanism knowledge defenders configure and attackers bypass, making it substrate by application.
- **`ostep`** — OS internals (scheduling, virtual memory, concurrency, file systems, distribution) are
  the foundation both privilege-escalation/memory-corruption offense and sandboxing/access-control
  defense are built on; the textbook teaches the mechanisms (incl. systems code) with no valence.
- **`nmap_book`** — Network scanning/reconnaissance methodology (host discovery, port/version/OS
  detection, NSE) is the canonical dual-use capability — defenders inventory and audit their own
  networks with the exact techniques attackers use to map targets; the book also covers defending
  against scans, so attack and defense sit side by side.
- **`boneh`** — Foundational cryptography (ciphers, MACs, hashing, key exchange, public-key, protocols)
  is the shared primitive layer attackers attack (cryptanalysis, protocol weaknesses) and defenders
  build on (secure construction); the graduate text teaches the mechanisms with no valence.
- **`intel_sdm`** — CPU privilege rings, segmentation, paging, and interrupt/exception handling are the
  architectural substrate beneath both exploitation (ROP, ring transitions, page-table abuse) and
  runtime defense (SMEP/SMAP, virtualization-based security); the manual specifies the mechanism only.

## Source provenance — exact locations & version pinning
Every substrate chunk records `extractor` (how) and `version_pin` (which version); the manifest
`configs/substrate_sources.yaml` is the canonical record of where each source comes from, and each
`data/build_reports/<id>.yaml` records `handler`, `source_location`, and `source_version_pin`.

| id | exact location (where) | handler (how) | version pin |
|---|---|---|---|
| `rfcs` | `https://www.rfc-editor.org/rfc/rfc{9293,1035,8446,9110,4271,4253}.txt` | rfc_text | RFC number (immutable) |
| `fips197` | `https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf` | pdf_single | download-date + sha256(PDF) |
| `mdn_web_security` | `github.com/mdn/content` :: `files/en-us/web/security/` | github_markdown | git commit SHA (`8dd50fa9…`) |
| `ostep` | `https://pages.cs.wisc.edu/~remzi/OSTEP/{stem}.pdf` (41 chapter stems pinned in YAML) | pdf_chapter_collection | per-chapter download-date + sha256 |
| `nmap_book` | `https://nmap.org/book/toc.html` (index) → `https://nmap.org/book/{stem}.html` | html_book_chapters | per-page download-date + sha256(HTML) |

Exact chapter/page selection (OSTEP stems, Nmap deny-prefixes) lives in the YAML manifest; reruns are
deterministic from the pinned versions.

## Collection process
Framework-provenance for offense/defense (ATT&CK/D3FEND citations); **curated, version-pinned sourcing
for the dual/substrate bucket** (manifest in `configs/substrate_sources.yaml`, generic handlers per
`type` in `src/entanglement/substrate.py`). Each source is built as one reviewed unit (handler + test +
build + `data/build_reports/<id>.yaml` + this card section + a dated METHODOLOGY entry).

## Preprocessing
Structural-boundary chunking (RFC sections; later: chapters/doc-sections), never character windows.
Dedup by content hash post-extraction (keep first; cross-source dup → one row tagging both `source_ids`).
Density policy is per-source: `flag_only` for legitimately symbol-dense sources (RFCs — ABNF grammars,
header tables, reference lists) never auto-drops on alpha-ratio; `standard` sources drop <0.5. RFC
extraction strips `[Page N]` footers, running headers, and form-feeds; drops table-of-contents and
sub-80-char fragments.

**Analysis-units layer (`src/entanglement/units.py` → `substrate_units.parquet`).** The structural
chunks above are the provenance record; a separate normalized layer is derived for the embedding/
separability experiments, because raw chunks span ~400× in size (a length signal that dominates
embeddings) and carry source-specific surface formats (RFC column-wrapping, pypdf kerning, markdown/
HTML) that a model can separate on instead of content. The units layer (a) format-strips — removes
markdown/HTML markup, de-wraps hard-wrapped lines into flowing prose, collapses whitespace; (b)
re-windows each chunk at paragraph/sentence (then word) boundaries to a consistent ~3000-char target,
hard-capped at 4000 (≈1000 tokens) — this also splits the oversized RFC appendices; (c) drops
boilerplate residue (RFC front matter, NIST process/changelog sections). Every unit keeps `parent_id`
→ the source chunk, so analysis traces back to provenance. Result: **870 units**, median 1,684 chars,
max 3,999 (0 over-cap); per-source medians now 990–3,110 (vs 982–7,208 raw). **Caveat:** geometric
separability of these units must still be validated against the actual unlearning tax — comparable
granularity and stripped format reduce, but do not by themselves prove, that separability reflects
*capability* rather than residual surface signal.

**Unified analysis corpus (`src/entanglement/analysis_units.py` → `analysis_units.parquet`).** All three
buckets are assembled into one normalized, bucket-labeled table for the separability/representation
experiments. **External-reference-only, symmetric design (default `prune_framework_metadata: true`):**
the framework-internal cataloging layers are dropped — `offense/procedure` (STIX `uses` attribution
examples) and `defense/prose` (D3FEND in-hand definitions/abstracts) — keeping only the cited-literature
external references on each side plus the dual substrate. Motivation: the source-confound diagnostic
(`scripts/diag_source_tsne.py`) showed framework-metadata register (132-char, named-entity-dense
procedures; short, sometimes off-valence D3FEND prose) dominated the representation geometry rather than
offensive/defensive valence. The source parquets above are unchanged; this prune affects only the
assembled analysis corpus, and is reversible via the config flag. Offense and defense are now
register-symmetric (both median ~2,900 chars). **Caveat:** this fixes the register asymmetry but not the
topic confound (buckets still differ by subject matter; dual still fragments into its 6 source-topics) —
valence must be tested by topic-matched comparison + the unlearning tax, never separability alone.

**GitHub blue-team supplement (operational defense).** To add operational defensive content (the D3FEND
external layer was governance/compliance-heavy), two clean-licensed GitHub repos are scraped via repo
**tarball** (one request each; GitHub = intended distribution), cleaned, content-filtered, and tagged into
**D3FEND tactics** so the whole defense bucket shares one D3FEND-grounded taxonomy (`topic`):
- **`H3llKa1ser/SOC-Assistant-Guide`** (MIT) — detection-engineering / DFIR *depth* (Splunk/KQL detections,
  threat hunting, forensics, AD attack scenarios, cloud).
- **`A-poc/BlueTeam-Tools`** (no license) — blue-team tool-knowledge *breadth* (per-tool description +
  usage across the SOC toolchain).
Scraped 2026-06-04: **101 docs → 146 units** (`supplement_github.py`, config `github_supplements`).
**D3FEND tactic is keyword-inferred** (Detect-skewed, as SOC work is) → `topic` set, authoritative
`d3fend_ids` left empty. **License/redistribution:** SOC-Assistant-Guide is MIT (redistributable w/
attribution); BlueTeam-Tools has no license → **local build only, URL+hash release, text not
redistributed** (released artifact = URLs + content hashes + `supplement_github.py`). **Limitations:**
two community repos, modest volume; broader (academic/vendor) source diversity is future work. (An earlier
Microsoft Security blog supplement and an Elastic Security Labs source were dropped — MS replaced by these
cleaner-licensed repos; Elastic's ToU bars scrapers. Both documented in METHODOLOGY.)

**Frozen v1 (`data/analysis_units_v1.parquet` + `data/corpus_manifest.yaml`).** Counts: offense **18,174**
/ dual **3,572** / defense **2,900** (= **24,646** units; defense = 2,754 D3FEND-cited + 146 supplement).
Defense composition: NIST+US-gov **68%** of units, GitHub-supplement ~6%, other ~26% — the supplement adds
operational **Detect** depth (Detect units 142→274) but is small, so defense stays NIST-heavy *by raw
count*; the compliance tilt is handled at **MCQ-sampling time** (cap NIST's share of generated questions),
not by padding the corpus. Cross-bucket contamination checked: **0** offense↔defense near-duplicates (the
blue-team repos aren't ATT&CK-cited). The manifest pins the sha256, per-bucket/layer counts, and the full
config snapshot; MCQ + unlearning experiments run against v1.

## Distribution
Redistributable as text: `rfcs` (IETF BCP 78), `mdn_web_security` (CC-BY-SA 2.5), `fips197` (US-gov PD).
Pointer+hash only (free-to-read, redistribution not granted — verify before release): `ostep`, `boneh`,
`intel_sdm`, `nmap_book`. The `license` field is propagated to every chunk.

## Volume statistics (final — 6 sources, all 6 topics)
Provenance: **2,670 chunks / 7,719,504 chars**. Analysis units: **3,761 units / ~7.3M chars** (max 3,999c).

| topic | units | sources |
|---|---|---|
| crypto | 1,496 | boneh, fips197, RFC 8446 |
| os_internals | 916 | ostep |
| web | 412 | mdn_web_security, RFC 9110 |
| architecture | 340 | intel_sdm |
| recon | 302 | nmap_book |
| networking | 292 | RFCs 9293/1035/4271/4253 |

Per-source quality sweep (all coherent): median alpha-ratio 0.86–0.94 across sources; ~10 chunks
< 0.5 alpha (RFC packet diagrams, FIPS hex matrices) are flagged-and-kept (symbol-dense by nature),
not garbled.

## Limitations (substrate)
- **`crypto` is overweight** (1,496 units) — the full Boneh book (751 chunks / 3M chars) dominates. Fine
  for a substrate pool, but if topic balance matters for an experiment, subsample Boneh.
- **`architecture` rests on a single source** (Intel SDM Vol 3A CH2–7); `os_internals` on a single
  source (OSTEP). Breadth within those topics is one-author-deep.
- **~10 symbol-dense chunks** (RFC diagrams, FIPS S-box/hex) and **4 units with a `U+FFFD`** glyph from
  PDF extraction are retained (0.1%); cosmetic small-caps kerning (`S UBBYTES`) survives in some
  PDF-sourced provenance chunks (units normalize ligatures but not kerning).
- **Reference/IANA RFC sections + NIST process boilerplate dropped** from the units layer (10 chunks)
  but retained in provenance.
- Coverage is curated, not exhaustive; English-only; snapshot-pinned (RFC numbers immutable; PDFs/HTML
  by download-date + sha256; MDN by git commit).
- **Redistribution split:** text-redistributable = `rfcs`, `mdn_web_security`, `fips197`; pointer+hash
  only (free-to-read, redistribution not granted) = `ostep`, `boneh`, `intel_sdm`, `nmap_book`.
- **Placement not yet validated**: that each source sits *between* offense and defense in embedding/
  gradient space remains an experiment-stage check (esp. the poles MDN→defensive, Nmap→offensive); and
  per the earlier caveat, geometric separability must be validated against the unlearning tax, not
  reported alone.

## Maintenance
Add a source = add a `configs/substrate_sources.yaml` entry + handler (if a new `type`) + test + run +
build report + card row + METHODOLOGY entry. Rebuilds are deterministic from pinned versions.
