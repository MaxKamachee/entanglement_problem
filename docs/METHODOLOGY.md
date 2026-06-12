# Methodology

Living record of how the entanglement corpus is constructed. Appended every
stage. The corpus is the primary contribution, so every source, filter,
threshold, and decision is documented here for reproducibility and review.

---

## Sources (pinned)

See `inputs/SOURCES.md` for checksums and full provenance.

- **MITRE D3FEND** ontology, tag `1.4.0` (2026-03-31). In-hand `d3fend.owl`,
  `d3fend.csv`, `d3fend-full-mappings.csv` (served by the live D3FEND API;
  self-identify as 1.4.0 via `owl:versionInfo`). D3FEND 1.4.0 maps against
  ATT&CK 18.x.
- **MITRE ATT&CK** Enterprise STIX `enterprise-attack-18.1.json`, pinned to
  match D3FEND 1.4.0's mapped ATT&CK version (avoids version-skew phantom
  edges). The dist artifacts are *build outputs* (not release assets); the
  reproducibility anchor is the tag + checksums, with a documented optional
  Docker-rebuild upgrade.

Decision — **ATT&CK 18.1 over 18.0/19**: D3FEND 1.4.0's CHANGELOG states it
updated ATT&CK to 18.1. Verified empirically: 324/325 mapped technique ids
resolve against 18.1; the single miss (`T1142` "Keychain") is a *deprecated*
ATT&CK id, not version drift. No ICS bundle needed (no unresolved ICS ids).

---

## Stage 1 — Parse sources into the tagging graph

Modules: `attack.py`, `d3fend.py`, `mappings.py`, with the shared
`normalize.py` spine (URL canonicalization + technique-id normalization; 16
unit tests).

URL canonicalization (`normalize.canonicalize_url`): lowercase scheme+host,
strip leading `www.`, drop fragment + trailing slash, keep only allowlisted
query params `{doi, arxiv, arxivid}` (sorted). Rationale: the dual-use signal
is a URL set-intersection between ATT&CK and D3FEND references; inconsistent
canonicalization would split a shared document into two singletons. We err
toward *under*-merging (drop most query params) since over-merging two distinct
docs is worse than dropping a rare query-keyed one.

Technique-id normalization: canonical full id (`T1003.001`, uppercased) plus a
derived `parent_id` (`T1003`), so joins can roll up to parent or stay at leaf.

Parsed volumes:
- ATT&CK: **691** techniques (216 top-level, 475 sub), **2,163** distinct
  offensive reference URLs.
- D3FEND OWL: **391** distinct reference URLs; in-hand prose **292,894 chars**
  (493 `definition` + 335 `kb-abstract`).
- Mappings: **14,003** edges, **149** D3FEND techniques ↔ **325** ATT&CK ids.

Finding — **D3FEND references are sparse and low-quality**: of 391 URLs, 143
PATENT (37%) + 89 MITRE_CAR (23%) = 59%; only ~36 are NIST/RFC/academic. This
is why reference-only corpus construction under-delivers on the defensive side.

---

## Stage "Venn" — reference-URL categorization (diagnostic)

Module: `categorize.py`. Classified all reference URLs into Offensive /
Defensive / Dual-Use using shared-citation + artifact-contestedness (p75).

Finding — **document-level dual-use is essentially zero**: the ATT&CK and
D3FEND reference URL sets intersect in **exactly 1 URL** (Microsoft's "Registry
Key Security and Access Rights" — a textbook shared-mechanism document). This
killed citation-based dual-use and motivated the pivot to topic-driven
retrieval with framework-provenance labeling.

---

## Stage A — Subcapability taxonomy + entanglement profile + topic partition

Module: `subcapabilities.py`. Config: `configs/subcapabilities.yaml`.
Outputs: `reports/entanglement_profile.md`, `data/technique_subcap.parquet`,
`data/topic_partition.parquet`. Tests: `tests/test_subcapabilities.py`.

**Taxonomy.** 14 ATT&CK enterprise tactics → 10 offensive subcapabilities;
7 D3FEND tactics → 5 defensive subcapabilities (full partitions). This refines
the spec's rough 7 offensive subcaps into a complete, gradient-preserving set
so recon / initial-access / impact are not collapsed.

**Multi-tactic assignment rule.** A technique spanning tactics in different
subcaps is assigned to the **earliest tactic in kill-chain order**
(`attack_tactic_priority`). Deterministic and documented. Result: every one of
the 691 techniques assigned, **0 orphans**.

**Contestedness.** Per ATT&CK technique = distinct D3FEND counterparts via the
14k mappings, taken as `max(exact-id count, parent-rollup count)` so a
sub-technique inherits its parent's mapped breadth. Rolled up per subcap as
mean/max.

**Dual-use threshold.** An ATT&CK technique is a dual-use *topic* if its
contestedness ≥ the **75th percentile of the mapped-technique distribution**
(= **28.0** D3FEND counterparts) OR it is in a graph-blind subcap. Choice of
p75 follows the spec's "above 75th percentile" rule; recorded as a tunable
parameter in the config.

**Topic partition (drives Stage B).** offensive **469** · dual-use **222** ·
defensive **271** topics. ATT&CK techniques are offensive unless contested/
graph-blind (then dual-use); D3FEND techniques (all 271, name coalesced across
the three sparse hierarchy columns of `d3fend.csv`) are defensive.

**Key finding — the reconnaissance paradox.** The `reconnaissance` subcap
(ATT&CK Reconnaissance + Resource-Development, 92 techniques) has **zero**
D3FEND mappings — the graph is structurally blind to pre-compromise activity —
yet reconnaissance is the canonical dual-use capability (scanning / enumeration
/ OSINT shared by red and blue teams). It is therefore flagged dual-use **by
construction**, not by the graph. This is the central evidence that
graph-contestedness alone cannot define dual-use, and the reason document-level
content checking (deferred Stage C) is needed rather than optional.

Entanglement gradient (mean D3FEND counterparts/technique, abbreviated):
persistence_escalation 25.5 · credential_access 22.4 · defense_evasion 17.9 ·
command_and_control 14.7 · initial_access 13.1 · execution 12.9 ·
collection_exfiltration 10.2 · discovery_lateral_movement 7.2 · impact 6.6 ·
reconnaissance 0.0 (graph-blind).

### Decision log
- ATT&CK STIX pinned to 18.1 (matches D3FEND 1.4.0); ICS excluded (empirically
  unneeded). `T1142` deprecated-id miss accepted.
- Lightweight source pinning (checksums + tag) over Docker rebuild for v0.
- Offensive subcaps expanded to 10 (from spec's ~7) to preserve the gradient.
- Multi-tactic techniques: earliest-kill-chain-tactic wins.
- Dual-use topic threshold: p75 of mapped contestedness (28.0), tunable.
- recon/resource-dev forced dual-use (graph-blind), per the recon paradox.
- **Stage C (document-level dual-use labeling) deferred** — user deciding the
  method. Stages A/B proceed on framework-provenance labels.

---

## Stage B (part 1) — In-hand corpus assembly

Module: `retrieve.py` (`assemble_inhand`). Output: `data/corpus_inhand.parquet`.
Tests: `tests/test_retrieve.py`.

Assembled the authoritative text already in hand into one provenance-labeled
corpus: ATT&CK technique descriptions (offensive / dual-use topics) + D3FEND
definitions (defensive / dual-use topics). Region inherited from each topic's
framework provenance (Stage A partition); `label_method = framework_provenance`.
Per-doc provenance: `doc_id`, topic, framework, region, subcap, source_bucket,
source_ref (constructed MITRE URL), fetch_date, n_chars.

Decision — **stub filter NOT applied to in-hand prose**: the <2000-char drop
targets scraped web stubs; ATT&CK/D3FEND prose is curated and often shorter.

Volumes (962 docs, 1,006,230 chars):
- offensive 469 docs / 637,574 chars; dual-use 222 / 337,144; defensive 271 /
  31,512.
- by bucket: ATTACK_DESC 691 / 974,718; D3FEND_DEF 271 / 31,512.

Finding — **severe offense/defense volume asymmetry (~20:1)**. ATT&CK
descriptions dwarf D3FEND definitions. Even adding the 219,038 chars of
unattached kb-abstract prose, the defensive side stays ~10× smaller. This is
the content-side confirmation of the Stage-1 reference-poverty finding and the
quantitative case that external defensive retrieval (NIST 800-53 in
particular) is required, not optional.

### Decision log (Stage B)
- In-hand corpus assembled first as a reproducible, no-network baseline.
- D3FEND kb-abstracts (219k chars) not yet topic-mapped; to attach via
  `kb-reference-of` in a later pass.
- Next: external retrieval — NIST 800-53 OSCAL (defensive backbone, no API key)
  + per-topic arXiv (academic, no API key) — both reproducible without search
  keys, sidestepping the search-API question.

---

## Stage C: Technique-tagged corpora (offensive procedures + defensive prose; external layers wired)

A second, parallel framing alongside the subcapability/Venn work above: two
**technique-tagged** corpora built directly to the spec — offensive (ATT&CK) and
defensive (D3FEND), each with an in-hand layer (no network) and an external
literature layer (scrape + filter, wired but run as a separate gated step). New
modules are additive; the Stage A/B outputs are untouched.

### Domain scope — enterprise only
Mobile excluded (out of scope). **ICS excluded**: D3FEND maps only to enterprise
ATT&CK, so ICS offensive techniques would be entanglement-orphans (no defensive
counterpart). Enterprise `enterprise-attack-18.1.json` alone has exactly 17,270
`uses` relationships, matching the spec's ~17,270 estimate. `procedures.py` is
domain-aware (`{domain: bundle_path}`) so ICS can be added later by dropping in a
bundle — the `domain` column is retained for that.

### Offensive in-hand: procedure examples (`procedures.py`)
Procedure examples are the `description` field on STIX `uses` relationships whose
target is an active (non-revoked/deprecated) technique. Of 17,270 `uses` rels,
16,017 target active attack-patterns and all carry a non-empty description →
**16,017 procedures, 599 techniques, 3,210,380 chars** (`attack_procedures.parquet`).
Sources: malware 9,836, intrusion-set 4,362, campaign 1,019, tool 800. Tagged with
`tech_id`/`parent_id`/`tactics`/`domain`/source metadata. `proc_id =
sha256(source_ref|target_ref)[:16]` for idempotency.

### Defensive in-hand: tagged D3FEND prose (`corpus_defensive.py`)
The existing `d3fend_prose` (definitions + kb-abstracts) is tagged with D3FEND
tactic and parent technique. **Tactic** comes from `inputs/d3fend.csv` (`D3FEND
Tactic` column; reuses the Stage-A path) and maps cleanly for the 271 CSV-listed
techniques. **Parent** comes from the OWL `subClassOf` hierarchy (`d3fend.py:
build_d3fend_hierarchy`, 466 concepts, 435 with a d3fend-id parent) and is set for
all 493 definition rows. kb-abstract rows are reference-keyed → null tactic/parent
by design. Output `d3fend_prose_tagged.parquet` (828 rows).

### VENDOR bucket + config-as-methodology
`classify_source` gains a **VENDOR** bucket (additive, after ACADEMIC, before the
OTHER fallback — the five existing buckets are byte-for-byte unchanged). The host
list lives in `configs/vendor_hosts.yaml`, not in code, so reviewers can audit it.
Likewise `defense_keep_buckets`, `wmdp_keep_threshold`, and scraper politeness live
in `configs/corpus.yaml`. After regeneration, buckets are PATENT 143, OTHER 102,
MITRE_CAR 89, VENDOR 24, ACADEMIC 15, NIST 11, RFC 10.

### Decision — OTHER bucket reviewed, not auto-dropped
The defensive external filter keeps NIST/RFC/ACADEMIC/VENDOR and drops
PATENT/MITRE_CAR. `OTHER` (102 refs) is **not** dropped pre-emptively: removing
PATENT+MITRE_CAR already leaves the defensive external set thin, and a sample shows
OTHER carries real capability content (Intel CET spec, LUKS on-disk format, RHEL
STIGs, CISA eviction guidance, man pages, academic PDFs). `corpus_defensive.main()`
emits `reports/defensive_bucket_review.md` (per-bucket counts + 50-row OTHER sample)
and `data/d3fend_other_sample.parquet`; the keep/drop decision on OTHER is made from
that review **before** any scrape runs.

### External layer (wired, deferred): shared scraper + WMDP filter
- `scrape.py` — shared fetch/extract/provenance. trafilatura (HTML) / pypdf (PDF),
  1 req/s per host, robots.txt (fail-open), dedup by canonical URL then content
  SHA-256, drop dead/non-text/stub(<2000 chars). On-disk cache keyed by
  canonical-URL hash makes runs idempotent/resumable; a URL cited by both corpora is
  fetched once. Emits `documents` + `provenance` tables.
- `quality.py` — WMDP-style 0-10 operational-capability rubric (offensive external
  only, keep ≥ threshold). claude-sonnet-4-6 via the Message Batches API, idempotent
  by `custom_id = content_hash`, rubric carried in a cached system block, structured
  JSON output. No quality filter on the defensive corpus.
- `corpus_offensive.py` — collect `external_references` URLs (excl. `mitre-attack`)
  → scrape → WMDP filter → tag documents (tech_ids/parents/tactics/domain, aggregated
  across every URL that produced a content hash).

Outputs (post-run): `{offensive,defensive}_documents.parquet` +
`{offensive,defensive}_provenance.parquet`.

### Decision log (Stage C)
- Baseline (in-hand + bucket review + corpus card) produced first as a fully
  offline, reproducible artifact; scrape + WMDP are a separate gated step.
- Methodology parameters moved to `configs/` (auditable, not buried in source).
- `docs/CORPUS_CARD.md` built from the start (Datasheets-style), populated from the
  baseline with post-scrape placeholders.
- **Gate before scraping:** review `reports/defensive_bucket_review.md` and decide
  the OTHER-bucket policy.

---

## 2026-06-04 — Substrate corpus (dual bucket): framework + RFC source unit

**What was added.** The "dual" bucket of the offense/dual/defense Venn: a curated *substrate* corpus of
operational prerequisite knowledge (networking/OS/architecture/web/crypto/recon mechanism docs).
Thesis enabled: the shared substrate is the *locus* of offense/defense entanglement — dual-use by
application (neutral content, dual use), distinct from the near-empty dual-use-in-content (1 shared URL
of 2,163 in the offense/defense citation overlap). New: `configs/substrate_sources.yaml` (manifest =
source of truth), `src/entanglement/substrate.py` (generic per-`type` handlers), `tests/test_substrate_fetch.py`.
First source unit built: **`rfcs`** (RFC 9293/1035/8446/9110/4271/4253) → **610 chunks, 1,439,526 chars**
into `data/substrate_corpus.parquet`; report at `data/build_reports/rfcs.yaml`.

**Choices made (with alternatives considered).**
- **TLS 1.3 (8446) topic = `crypto`** (alt: `networking`). The RFC is dominantly the cryptographic
  handshake / key schedule / AEAD; tagged crypto. Reversible — edit the YAML and rebuild.
- **Per-source `density_policy`** (alt: the plan's per-*topic* policy). Symbol-density is a property of
  the *source format*, not the topic (the HTTP RFC is `topic: web` yet carries ABNF grammars). So the
  flag-don't-drop policy is keyed per source in the YAML (`flag_only` for RFCs) rather than per topic.
  This supersedes the plan's topic-keyed default and avoids dropping legitimately dense web/RFC content.
- **`source_ids` is a list** (schema fix vs the singular `source_id` in the original spec) so cross-source
  duplicate chunks collapse to one row tagging both sources. "Append" = read-existing → concat →
  content-hash dedup → rewrite (polars parquet is immutable).
- **RFC chunking = section boundaries.** Strip `[Page N]` footers, `^RFC \d+` running headers, form-feeds;
  drop table-of-contents chunks (>50% dot-leader lines) and fragments <80 chars. Verified: 0 residual
  page/header artifacts across all 610 chunks; core mechanism sections (e.g. 9293 §3.1 Header Format)
  extract cleanly.
- **Min chunk size 80 chars** (alt: the 2000-char scraped-web stub filter) — substrate chunks are
  *sections*, legitimately short; the 2000-char floor is for web stubs and does not apply here.

**Dropped / flagged content.** 38 dropped (35 too_short, 3 table_of_contents). 31 chunks flagged
`low_text_density` and **kept** (flag_only) — these are reference lists / header-table sections, which
are legitimately symbol-dense.

**Known item to revisit.** 9 References/IANA-Considerations sections are retained but are low-value
substrate; candidate for a drop rule in a later refinement. Surfaced, not yet acted on.

**Expected impact.** RFCs anchor the networking/web/crypto end of the substrate at mechanism level with a
clean, fully-redistributable (IETF BCP 78), immutably-pinned source — the strongest dual-use rationale of
the six locked sources, so a good first unit.

### 2026-06-04 — Substrate: FIPS 197 (AES) via pdf_single + PDF chunking hardening

**What was added.** Source unit **`fips197`** (FIPS 197 AES standard, NIST) via a new `pdf_single`
handler (download → pypdf extract → document-section split). topic=crypto, license=US-gov public domain
(fully redistributable), version_pin = download-date + sha256 of the PDF bytes (in the build report).
Result: **47 chunks, 61,323 chars**.

**Extraction-quality issue found and fixed (the plan's "flag PDF extraction" gate).** The first build
produced **98 chunks** — over-split. Diagnosis: FIPS standards open with federal-register *announcement*
boilerplate ("1. Name of Standard. … 5. Maintenance Agency. … 10. Patents. … 14. Inquiries"), each a
short `N. Title` line that the generic numbered-header splitter mistook for a body section, plus
ToC-vs-body label collisions (40 positions appeared >1×; 39/98 chunks were sub-150-char noise).
**Fix:** a per-source `min_chunk_chars` override (PDFs carry more short-fragment noise — boilerplate,
captions, ToC lines — than plain-text RFCs); set `min_chunk_chars: 250` for FIPS. Rebuild → 47 chunks
(0 tiny, 35 substantive, median 982 chars, collisions 40→4), losing only ~6K chars of boilerplate.
This knob is generic and will be reused/tuned for the harder PDFs (OSTEP, Boneh, Intel SDM).
Alternatives considered: `body_start_marker`/`body_end_marker` trimming (cleaner but needs a per-doc
string and more handling) — deferred; the char floor was sufficient here.

**Residual limitation (documented, not blocking).** ~8 longer NIST front/back-matter boilerplate items
(announcement 3/6/8/11/12/15, references 12/13/15/19) clear the 250-char floor and remain; they are
low-value process text, not crypto substrate. `Appendix A` collides (2×). AES S-box / state-array tables
extract as flagged `low_text_density` and are kept (flag_only, crypto). pypdf introduces minor kerning
artifacts ("S UBBYTES") in small-caps headings; content remains coherent.

**Decision.** Accepted FIPS as-is for v1 (35 substantive crypto chunks + minor boilerplate residue),
documented rather than perfected — consistent with "honestly documented over complete." The clean RFC
baseline was preserved during diagnosis (FIPS reverted from the corpus, then re-appended after the fix).

### 2026-06-04 — Substrate: MDN Web Security via github_markdown

**What was added.** Source unit **`mdn_web_security`** (MDN Web Docs `files/en-us/web/security/`) via a
new `github_markdown` handler: sparse + shallow + blobless clone of just the subtree
(`git clone --depth 1 --filter=blob:none --sparse` then `sparse-checkout set`), one chunk per `.md`
file. topic=web, license=CC-BY-SA 2.5 (redistributable), version_pin = `git mdn/content@<HEAD sha>`
(recorded: `8dd50fa9…`). Result: **44 chunks, 409,452 chars, 0 dropped, no quality flags** — markdown
prose is alpha-dense and extracts cleanly.

**Choices.** Clone strategy avoids pulling the full ~50k-file mdn/content repo (sparse subtree only).
Preprocessing strips YAML frontmatter (`---`…`---`) and KumaScript macros (`{{…}}`) — verified 0 leaks
of either across all 44 chunks. Per-file chunking (vs splitting on `##` headers) per the plan; MDN files
are already topic-scoped (one concept per file), so file == chunk is the natural structural boundary.

**Dual-use note (strengthens the thesis).** The subtree contains an `attacks/` directory (clickjacking,
CSRF, IDOR, MITM, phishing) alongside the defensive mechanism docs — the same source documents both the
attacks and the protections, concrete evidence of dual-use-by-co-location at the substrate layer. This
also tempers the pre-build concern that MDN would lean purely defensive; it carries explicit attack
content too. (Empirical placement check remains a Day-4 item.)

### 2026-06-04 — Normalized analysis-units layer (separability-experiment prep)

**What was added.** `src/entanglement/units.py` → `data/substrate_units.parquet`, a normalized,
comparable-granularity, format-stripped view of the substrate corpus for the embedding/gradient
separability experiments (RQ2/3). The provenance corpus (`substrate_corpus.parquet`) is unchanged;
units carry `parent_id` back to the source chunk.

**Why.** A Day-1 quality audit found two confounds that would contaminate "does geometry predict the
unlearning tax?": (1) **granularity** — raw chunks span ~400× (93–40,246 chars; per-source medians
982 FIPS … 7,208 MDN), and chunk length is a dominant embedding signal; (2) **format fingerprinting** —
each source has a distinct surface format (RFC fixed-column wrapping, pypdf kerning, markdown/HTML;
44/44 MDN chunks carry `](` links, 27/44 code fences), so a model can separate sources — and thus the
offense/dual/defense buckets — on format rather than capability (the data-vs-capability-separability
trap at the formatting layer).

**How.** (a) format-strip: remove HTML tags, markdown link/image/heading/list/emphasis/fence syntax
(keeping link text and code bodies), de-wrap hard-wrapped lines into flowing prose, collapse
whitespace; (b) re-window at paragraph→sentence→word boundaries to a ~3000-char target, hard-capped at
4000 (≈1000 tokens) — also splits the 3 oversized RFC appendices (40,246/32,810/32,400 chars); (c) drop
boilerplate residue (RFC `front` chunks; NIST announcement/changelog sections via a heading regex).

**Result.** 701 provenance chunks → **870 analysis units**; 9 boilerplate chunks dropped, 0 duplicate
units; chars min/median/max 89/1,684/3,999 (0 over the 4,000 cap); per-source medians 990–3,110 (down
from 982–7,208). Parent linkage verified: 0 orphan units.

**Bugs found + fixed during the build.** (i) `_hardsplit` split only on sentence boundaries, so
punctuation-free blobs (ABNF grammars, hex/ASCII tables) produced >cap units (max 15,108) — added a
word-boundary fallback. (ii) the short-tail merge (`units[-1] + tail`) could push the last unit over
the cap (max 4,415) — gated the merge on staying ≤ hard_max. Both locked by tests.

**Decisions / limitations.** Re-windowing is *within-chunk* only (split large, keep small) to preserve
provenance and topic purity — so short structural sections (e.g. FIPS ~990c) stay smaller than
re-windowed MDN units (~3,110c); residual per-source spread is ~3× (vs 400× raw), deemed acceptable.
Cross-chunk merging of small same-source sections is a possible future refinement. De-wrapping flattens
ASCII packet diagrams (a small, already-low-value fraction). **The layer reduces but does not eliminate
the separability confound — geometric separability must still be validated against the measured
unlearning tax (RQ3), never reported alone as evidence of capability separability.**

### 2026-06-04 — Substrate: OSTEP via pdf_chapter_collection

**What was added.** Source unit **`ostep`** (OSTEP textbook) via a new `pdf_chapter_collection` handler
(per-chapter download → pypdf extract → document-section split, reusing `split_doc_sections` +
`_emit_chunks`). topic=os_internals, license = free-to-read / **pointer+hash only** (redistribution not
granted), version_pin = per-chapter download-date + sha256. Result: **840 chunks, 1,448,274 chars**
across 41 chapters; 458 too-short fragments dropped (`min_chunk_chars: 250`); density `flag_only` to
preserve systems code listings.

**Scoping decision.** Chapter list discovered from the OSTEP index page (68 PDFs) and curated to the 41
substantive mechanism chapters — virtualization (cpu/vm/vmm), concurrency (threads), persistence (file),
distribution (dist). **Excluded:** Socratic dialogues, front-matter (preface/toc/dedication), labs, and
the **security section** (security-intro/authentication/access/crypto/distributed) — dropped to keep the
`os_internals` topic pure and avoid double-counting crypto (covered by `boneh`/`fips197`). This mirrors
the "Intel SDM conceptual chapters only" curation the spec already endorses. Chapter stems are pinned in
`configs/substrate_sources.yaml` (the source of truth).

**Quality.** Spot-checked: prose sections coherent (e.g. cpu-sched-mlfq §8.2); **code listings preserved**
as substrate (e.g. threads-cv Figure 30.6 Put/Get routines — kept via flag_only despite low alpha
density). ~20 sections/chapter reflects OSTEP's genuine section richness + ASIDE/TIP/CRUX boxes, not
pathological over-split; the dropped fragments are those boxes + figure captions. Minor cosmetic pypdf
kerning ("th at", "a nd") in some chunks; content remains coherent.

**Units-layer bug fixed (exposed by OSTEP).** `units.is_boilerplate` had dropped *all* `position=="front"`
chunks — correct for RFC/FIPS front matter but it would wrongly drop **OSTEP chapter intros** (the
content before a chapter's first numbered section). Changed to **content-based** detection: match NIST
process-item text and IETF RFC front-matter markers ("Internet Engineering Task Force", "Status of This
Memo", "Request for Comments:", etc.), not the position label. OSTEP intros are now retained.

### 2026-06-04 — Substrate: Nmap Network Scanning via html_book_chapters

**What was added.** Source unit **`nmap_book`** (Nmap Network Scanning, free online edition) via a new
`html_book_chapters` handler: discover chapter stems from the ToC (`discover_chapter_stems`, relative
`.html` links minus a denylist), fetch each page (1 req/s), extract main text with trafilatura, one
chunk per page. topic=recon, license = free-to-read / **pointer+hash only**, version_pin = per-page
download-date + sha256 of fetched HTML. Result: **88 pages, 770,441 chars**, median 6,503 chars/page,
0 dropped.

**Curation.** The ToC exposes 119 relative pages; denied front/back matter, install, GUI (zenmap*), man
pages (ndiff/ncat/nping), and the large reference data-file dumps (nmap-os-db, nmap-services,
service-probes, etc.) via `deny_prefixes` in the YAML. Kept the recon *methodology*: host discovery,
port scanning, scan methods (SYN/connect/UDP/idle/FIN-XMAS/ACK/window/Maimon/IP-protocol/FTP-bounce),
version + OS detection (vscan*/osdetect*), NSE scripting, performance, and the firewall/IDS **evasion +
defense** chapters (firewalls*, firewall-subversion, nmap-defenses-*). Like MDN, this source carries
both attack and defense content, reinforcing dual-use-by-co-location for the recon topic.

**Quality.** Spot-checked: trafilatura cleanly extracts chapter prose (e.g. host-discovery-techniques —
coherent methodology text, nav/chrome stripped). Per-page chunks are large (median 6.5k chars); the
units layer re-windows them to ~3000c.

**Bug fixed.** The Nmap YAML entry initially lacked a source-level `version_pin` (per-page pins carry the
real shas), which `_build_report` requires — added a descriptive source-level pin.

### 2026-06-04 — Substrate: Boneh & Shoup Applied Cryptography (full book, pdf_single)

**Deviation surfaced (per "surface before deviating").** The spec assumed `toc.cryptobook.us` hosts
per-chapter PDFs; it does not — it links to a **single full-book PDF** (`BonehShoup_0_6.pdf`, v0.6,
~900pp) plus older versions. Same source (Boneh–Shoup *A Graduate Course in Applied Cryptography*), but
the form is one PDF, so the handler is **`pdf_single` on the full book** (reusing existing code), not
`pdf_chapter_collection`. No source substitution — only a handler adaptation to the actual distribution.
topic=crypto, license = free-to-read draft / **pointer+hash only**, version_pin = v0.6 + download-date +
sha256. Result: **751 chunks, 3,021,799 chars** (799 too-short fragments dropped, `min_chunk_chars: 250`).

**Quality (better than feared for a math-heavy text).** It's a clean digital PDF (not scanned), so pypdf
extracts well: alpha-ratio median 0.90, **zero chunks below 0.55** (no math-garble-dominated chunks),
only 4 chunks (1%) flagged low-density. Spot-checked: prose sections coherent (e.g. §2.1.2 Perfect
Security); only inline protocol diagrams partially garble while their surrounding explanation stays
intact. Residual: typographic ligatures (`deﬁned`→define) survive extraction — cosmetic; candidate for
a ligature-normalization pass in `units.clean_text` at Day 4.

**Scope note.** The whole book is ingested (all of it is foundational crypto substrate), unlike Intel
SDM where we deliberately take conceptual chapters only and drop opcode tables.

### 2026-06-04 — Substrate: Intel SDM Vol 3A (conceptual chapters) via pdf_chapter_extract

**What was added.** Source unit **`intel_sdm`** (Intel 64 & IA-32 SDM Vol 3A, System Programming Guide
Part 1) via a new `pdf_chapter_extract` handler: download the single Vol 3A PDF, extract pinned page
ranges, document-section split each. topic=architecture, license = Intel © / **pointer+hash only**,
version_pin = Vol 3A (getContent/671190) + download-date + sha256 (`373170…`). Result: **290 chunks,
568,689 chars** (129 too-short fragments dropped).

**Source identification + page-range discovery (the hard part, resolved).** Intel serves the SDM via
`cdrdv2.intel.com/v1/dl/getContent/<id>` redirects (no direct .pdf). Resolved the Vol 3A id by parsing
the SDM page's anchor titles: **671190 = "Volume 3A: System Programming Guide, Part 1"** (671447 =
combined 3A-D; 671427/671506/671269 = 3B/3C/3D). Downloaded Vol 3A (524pp) and **discovered chapter
body start pages programmatically** (scanning page text for `Vol. 3A N-1 CHAPTER N <TITLE>`, skipping
the ToC), then pinned the conceptual chapters in the YAML:
CH2 architecture overview (59–88), CH3 segmentation/memory mgmt (89–104), CH4 linear-address (105–112),
CH5 paging (113–168), CH6 protection (169–202), CH7 interrupt/exception handling (203–260). This maps
exactly to the spec's "privilege/protection, paging, memory management, exceptions, interrupts" and
**stops at CH8 (FRED)** — dropping the specialized/reference bulk (user interrupts, task/MP mgmt, init,
APIC, cache, MMX, VMX, SMM, appendices), i.e. the "drop opcode tables/appendices" intent.

**Quality (clean — the highest-risk source came out well).** Digital, well-typeset PDF → alpha-ratio
median 0.93, **zero chunks below 0.55** (no table-garble), 0 low-density flags. Spot-checked: §6.10.4
(ARPL caller-privilege checking) and §5.1.4 (CPUID paging enumeration) extract as coherent architecture
prose. Selecting conceptual chapters only avoided the opcode-table noise that whole-PDF ingestion would
have introduced.

**All six locked sources are now built; the `architecture` topic is filled.**

### 2026-06-04 — Day 4: corpus-wide quality pass + finalization

**Substrate build complete: 6 sources, 6 handler types, all 6 topics.** Provenance 2,670 chunks /
7,719,504 chars → 3,761 normalized analysis units (max 3,999c, 0 orphans).

**Ligature repair.** Added typographic-ligature normalization to `units.clean_text` (ﬁ→fi, ﬂ→fl,
ﬀ/ﬃ/ﬄ/ﬅ/ﬆ) — survives PDF extraction in Boneh/FIPS/Intel ("deﬁned"→"defined"). Verified 0 ligature
chars remain in units; provenance keeps the raw extraction.

**Per-source quality sweep (spot-checked all sources).** Median alpha-ratio 0.86–0.94 across the six;
all sampled snippets coherent (RFC SSH/ABNF/SOA, OSTEP FFS/CVs, Boneh Bleichenbacher/BLS, Intel
paging/RIP/CR3, Nmap timing/port-selection, MDN phishing/CSP, FIPS ShiftRows/InvMixColumns). The only
<0.5-alpha chunks (~10: RFC packet diagrams, FIPS hex/S-box) are flagged-and-kept (symbol-dense, not
garbled). 4 units (0.1%) carry a `U+FFFD` glyph from PDF extraction; small-caps kerning ("S UBBYTES")
survives in some PDF provenance chunks (cosmetic, headings only).

**Integrity (re-verified):** schema/nulls/`chunk_id==sha256`/uniqueness clean on both tables; build
reports match corpus counts; cross-source dups 0; boilerplate matcher = 10 genuine drops (RFC IETF/
IANA, FIPS process), OSTEP intros preserved; format-strip removed all markdown/HTML/page artifacts from
units (MDN `](` 44→0).

**Documentation finalized.** CORPUS_CARD: per-source rows, dual-use rationales, source-provenance table
(exact URLs + pinning), volume statistics, limitations, distribution/redistribution. Build reports
carry `handler` + `source_location` + version pin. This METHODOLOGY log has a dated entry per source.

**Known limitations carried forward (documented, not blocking):** crypto overweight (Boneh full book);
architecture/os_internals each single-source; placement-in-embedding-space validation deferred to the
experiment stage (and must be checked against the unlearning tax, not reported as separability alone).

### 2026-06-04 — Defensive OTHER bucket: curated keep + two classifier fixes

**Decision.** OTHER (the uncategorized D3FEND-citation bucket) gets a **curated keep**: fold in the
substantive refs (standards/specs/academic/gov/vendor hardening — TPM, Intel CET, seccomp, CFI, CWE,
K8s hardening, STIGs, CISA playbooks, academic PDFs), drop only noise hosts. Chosen over strict-drop
because removing PATENT/MITRE_CAR already leaves the defensive external set thin, and the OTHER sample
showed it's the richest defensive content.

**Two correctness fixes the review surfaced** (`classify_source`, additive — five original buckets
byte-unchanged): (1) mis-bucketed patents — `patentimages.storage.googleapis` and `patentguru` now
classify as PATENT (were leaking into OTHER); (2) new `ATTACK_XREF` bucket for `attack.mitre.org`,
dropped per the no-crosswalks constraint (out-of-framework ATT&CK Mitigations content). `cwe.mitre.org`
/`car.mitre.org` kept distinct (CWE→OTHER-kept, CAR→MITRE_CAR-drop).

**Mechanism (config-as-methodology).** `configs/corpus.yaml`: `defense_keep_other_curated: true` +
`defense_other_drop_hosts` (substring denylist of 9 noise hosts: wikipedia, blogspot, gartner, ghacks,
networkworld, safetydetectives, sap-press, sans, biometric-solutions). `select_defensive_urls` keeps
`bucket ∈ keep_set OR (bucket==OTHER AND host ∉ denylist)`. Two-stage: denylist now + the scraper's
<2000-char stub filter later catches residual thin noise.

**Result.** After regenerating `d3fend_refs.parquet`: PATENT 148, OTHER 95 (was 102), ATTACK_XREF 2.
Curated defensive selection = **141 distinct URLs** (VENDOR 24 + ACADEMIC 15 + NIST 11 + RFC 10 + 82
curated OTHER; 13 denylisted OTHER dropped, 0 patents/MITRE_CAR/ATTACK_XREF). Bucket review report
updated to a selection record. This is **selection only** — the defensive scrape remains gated.

### 2026-06-04 — Offensive external: ATT&CK ontology object-type selection (replaces WMDP filter)

**Decision.** The WMDP LLM-judge filter (keep operational-capability ≥7) is **replaced** by deterministic
ATT&CK-ontology selection. Rationale: "cited by ATT&CK" ≠ operational capability, but the *ontology
classifies techniques, not the cited literature's depth* — so an LLM judge was needed for true depth.
Instead we use the cheaper, reproducible, LLM-free signal the ontology *does* give: the **citing object
type**. References cited by operational/methodological objects (techniques, software, procedures) are
kept; references cited *only* by attribution objects (groups, campaigns) are dropped.

**Implementation.** `attack.build_offensive_refs` walks all citing object types + the `uses`
(procedure) relationships, emitting one row per (canonical-url × citing-object) tagged with
`citing_type`/`citing_id`/`citing_name`/`tech_id`/`tactics` (→ `data/offensive_refs.parquet`).
`corpus_offensive.collect_offensive_urls` groups per URL and keeps those whose citing-type set
intersects {technique, software, procedure}. `tag_documents` carries `citing_types` alongside
tech/parent/tactics/domain. `main()` no longer calls the WMDP rater; `quality.py` is unwired (kept as
an optional analysis-stage operational-depth tagger).

**Result (enterprise, post-canonicalization).** 3,885 distinct URLs by citing type: technique 2,163,
procedure 1,788, software 1,085, group 701, campaign 101. **Selection keeps 3,767** and drops 118
cited only by group/campaign. Mitigations (course-of-action) and `mitre-attack` self-refs excluded.

**Transparency / limitation.** This is a *gentle, structural* filter (~3% drop), not WMDP's aggressive
operational concentration — the kept set still includes non-operational vendor-blog/news that techniques
cite. An operational-depth axis is recoverable later via the optional rater. (Stated in the corpus card.)
This is **selection only**; the offensive scrape remains a gated network step.

### 2026-06-04 — External scrape complete: ALL corpus construction done

The shared scraper (`scrape.py`: trafilatura/pypdf, 1 req/s per host, robots-respecting, content-hash
dedup, <2000-char stub filter, full provenance) was run for both external layers.

**Offensive external** (3,767 ontology-selected URLs): **2,704 unique documents, 55.5M chars**; 74%
fetch success. Attrition by `reason`: stub 369, dead 297, robots_disallow 118, non_text 85, errors 93.
Tagged with tech_ids (2,631/2,704)/parent_ids/tactics/citing_types/domain.

**Defensive external** (141 curated URLs): **61 unique documents, 8.56M chars**; 43% success — drops
dominated by **robots_disallow (50/141)**: .gov/standards/vendor hosts (CISA, NIST-adjacent, TCG, etc.)
broadly disallow bots. Tagged with d3fend_ids/tactics/parent_ids/buckets.

**Bug fixed mid-run (single-point-of-stall).** The scrape hung at URL ~26: `urllib.robotparser.read()`
has no timeout, so one host with an unresponsive `robots.txt` froze the whole sequential run. Fixed by
routing robots through the httpx client with a 10s timeout (fail-open). Page fetches were already
bounded (30s). The cache (`data/scrape_cache/`, keyed by canonical-URL hash) made the kill/fix/resume
free — done URLs skip instantly. Also added a stderr progress bar to `scrape.scrape()`.

**Asymmetry, now empirical.** External offense:defense = 2,704 : 61 docs (~44:1 by count; ~6.5:1 by
chars). With the in-hand layers (16,017 procedures vs 828 prose rows), the offense/defense volume
asymmetry — flagged from the start — is the corpus's defining feature and a first-order consideration
for any balanced experiment. 2 documents are shared offense↔defense by content hash (near-zero textual
dual-use, consistent with the framework-citation finding).

**Status: corpus construction complete** — offensive (in-hand + external), defensive (in-hand +
external), dual/substrate (6 sources). Next is the experiment stage (separability, MCQ eval, unlearning).

### 2026-06-04 — Separability pilot (experiment stage begins)

First experiment on the completed corpus. `analysis_units.py` unifies all layers into one
normalized, bucket-labeled table (`analysis_units.parquet`, 42,050 units: offense 34,711 / dual
3,760 / defense 3,579) via `units.clean_text`+`resegment` so length/format/source artifacts aren't
the signal. `separability.py` balanced-samples 1,500/bucket, truncates to 2,000 chars, embeds two
ways (TF-IDF lexical + fastembed BGE-small semantic), and runs a cross-validated logistic probe +
centroid cosine distances.

**Findings (`reports/separability_pilot.md`).** Offense vs defense is highly separable — **lexical
94.3%**, **semantic 92.7%** (chance 50%); 3-way offense/dual/defense 93.4% / 90.0% (chance 33%). The
semantic accuracy ≈ lexical accuracy → separability is **not merely surface vocabulary**; the buckets
are distinct in meaning. The **dual substrate sits between offense and defense** in both spaces (both
bucket–dual centroid distances < the offense–defense distance), roughly equidistant (which side it's
marginally closer to flips by backend: lexical→defense, semantic→offense).

**Caveat (load-bearing).** This is **data separability**, not capability separability — the buckets
come from disjoint sources, so high separability is expected and does not establish that offensive and
defensive *capabilities* are separable in a model's weights. That requires the unlearning-tax
experiment; geometric separability must be validated against the tax, never reported as capability
separability on its own. The structurally meaningful result here is that the substrate is geometrically
intermediate, consistent with the substrate-as-shared-middle thesis.

Deps added: scikit-learn, fastembed (ONNX BGE-small; no torch).

**Figure:** `reports/figures/separability_pilot.png` — (A) t-SNE of semantic embeddings by bucket (offense/dual/defense distinct, dual bridging between), (B) probe accuracy vs chance (lexical vs semantic), (C) 3-way confusion (defense most leaky toward dual). Rendered by `src/entanglement/viz_separability.py` (matplotlib).

## 2026-06-04 — Representation-space separability (Llama-3.1-8B hidden states)

Moved the separability question from *document embeddings* (BGE) into the *model's own representation
space*, since that is where the planned unlearning interventions (RMU/NPO/…) operate. Two stages.

**Stage 1 — extraction (GPU).** `scripts/extract_hidden_states.py` (self-contained one-shot pod
script; a RunPod-Flash/serverless variant is kept at `extract_hidden_states_flash.py` for the no-box
case). Model identity: **`meta-llama/Llama-3.1-8B`**, bfloat16, `output_hidden_states=True`, eval mode.
Revision pin: the script records the resolved HF commit hash in its manifest (this run's manifest was
not transferred back; the parquet is self-describing). **Sampling:** uniform stratified by bucket, **200
docs/corpus, seed 0** — identical protocol to the BGE pilot for comparability (no within-corpus
provenance balancing). **Pooling:** **masked mean** over real tokens (attention-mask-weighted, padding
excluded; pad=eos so pad positions never enter the pooled vector); last-token noted as the considered
alternative. **Layers:** hidden_states indices **4 / 16 / 28** (early/mid/late; index 0 = embeddings,
k = output of block k). One forward pass yields all 33 states, so more layers are free compute — only 3
were transferred this round; a finer depth curve is a free re-run (`--layers 0 4 8 12 16 20 24 28 31`).
Output `data/hidden_states/llama31_8b_three_way.parquet` (long format, 1,800 rows = 600 docs × 3 layers;
columns `doc_id, corpus_label, subcap, source_id, layer, embedding[4096]`). Files moved laptop↔pod with
`runpodctl send/receive` — the `ssh.runpod.io` proxy only allows interactive PTY, so agent scp/exec was
not possible. Validated: balanced 200/bucket/layer, 0 NaNs, 600 distinct doc_ids, mean L2 norm rises
with depth (14.7→16.1→28.0 = expected residual-stream signature).

**Stage 2 — analysis (local, no GPU).** `src/entanglement/representation_analysis.py` reuses the pilot
instruments (`separability.probe`, `centroid_cosine_distances`, `viz_separability.project_2d`) per layer,
plus new metrics: centroid distances **normalized by within-class spread** (scale-free "dual between"
test) and per-document **out-of-fold P(true class)** → a "borderline" flag (< 0.5) marking docs in the
overlap region. Outputs `reports/representation_analysis.md` (+ per-layer and depth-curve figures),
`data/representation_metrics.parquet`, `data/document_confidence.parquet`. Tests:
`tests/test_representation_analysis.py` (offline, synthetic blobs; no model).

**Findings.** Three-way probe accuracy **0.923 → 0.938 → 0.952** across layers 4→16→28 (off-vs-def
0.932→0.953→0.950) — **comparable to or above** the BGE pilot's 0.90 semantic / 0.93 lexical, and
**rising with depth** (the direction consistent with a capability-relevant, not merely surface, split).
Headline shift from the pilot: in representation space the dual substrate is **not between** offense and
defense — `offense-dual` is the *widest* pair at every layer (0.52/0.92/0.98, wider than
offense-defense), so **dual clusters with defense and offense is the outlier**, an asymmetry that
sharpens with depth. Dual is the tightest cluster (lowest borderline fraction, 0.04→0.02); defense is
leakiest (0.17→0.07). **Caveat (unchanged, load-bearing):** this is geometry of pooled *document*
vectors, not capability entanglement in weights — it establishes that a representation-level
forget/retain split is well-posed (buckets ≥0.92 linearly separable deep in the net), not that
unlearning offense will/won't tax defense. That remains the unlearning-tax experiment.

**Figures:** `reports/figures/repr_layer{04,16,28}.png` (per-layer t-SNE + row-normalized confusion),
`reports/figures/repr_depth_curve.png` (3-way + binary accuracy vs depth; 3 points → read directionally).

## 2026-06-04 — Source-confound diagnostic + symmetric external-only prune

**Diagnostic.** Re-colored the Llama hidden-state t-SNE by `source_id` (provenance/register) and probed
within the dual bucket (`scripts/diag_source_tsne.py`). Result: the geometry tracks **source/topic, not
valence**. Within DUAL alone a probe recovers which of the 6 substrate sources a doc came from at **0.97**
(layer 28) — *higher* than the 3-way offense/dual/defense accuracy (0.95); `source_id` 5-way is 0.95 and
*rises with depth* (0.91→0.95), the same trend as the valence probe. Inspection showed why: offense was
~46% `procedure` units (132-char, 44%-capitalized-token, citation-tagged STIX attribution) — a register
unlike anything else — while defense `prose` (D3FEND definitions) was short and sometimes off-valence
(a sampled unit was generic statistics; another was `CreateRemoteThread`, i.e. offense-relevant). The
separability we had been reporting was substantially a topic+register artifact.

**Prune decision.** External references on both sides; framework-internal cataloging (STIX procedures,
D3FEND prose) excluded as off-target metadata that does not constitute the operational/mechanism content
capturing capability. Symmetric prune motivated by the source-confound diagnostic showing
framework-metadata register dominated the geometry. Implemented as `prune_framework_metadata` (config flag
in `configs/corpus.yaml`, default true) applied in the pure builder `analysis_units.build_analysis_units`
(reproducible from source, not a parquet mutation): drops `(offense, procedure)` + `(defense, prose)`.
Test `tests/test_analysis_units.py::test_prune_framework_metadata_drops_procedure_and_prose` asserts both
vanish under the default and survive under `prune_framework_metadata=False`.

**Counts (pre → post prune).** offense 34,711 → **18,787** (procedure 15,924 dropped); defense 3,579 →
**2,754** (prose 825 dropped); dual 3,760 (unchanged). Total 42,050 → 25,301. Offense/defense are now
register-symmetric (both median ~2,900 chars, vs the old 132 vs 2,919 asymmetry).

**Scope / caveat.** This removes the worst *register* confound and makes the two sides comparable. It does
NOT resolve the within-dual topic fragmentation (dual is still 6 topic islands) nor the broader topic
confound (buckets still differ by subject matter). The geometry must be re-measured on the pruned corpus
(re-extract hidden states, same protocol) and is expected to drop from ≈0.95; the decisive valence test
remains topic-matched comparison + the unlearning tax, not separability.

## 2026-06-04 — Pre-MCQ quality audit + cleanup filters

**Audit.** `src/entanglement/corpus_audit.py` → `reports/corpus_quality_audit.md`: per-bucket flagging
(alpha ratio, sentence/paragraph counts, boilerplate phrases), topic distribution, MinHash near-dup
detection, length histograms, and 20 full-text samples/bucket. Finding: offense (scraped threat-intel)
carried the most noise — IOC/hash-dump fragments, web-scrape near-duplicates (same report cited by many
techniques), and symbol-dense-but-substantive content (YARA rules, code) that trips `low_alpha`/
`high_boilerplate` flags as false positives. Dual carried a few non-prose dumps (Nmap traceroute,
base64 image blobs) and short truncation/exercise fragments. Defense was cleanest.

**Cleanup (config-driven, reproducible — `src/entanglement/cleanup.py`, gated by `cleanup_enabled` +
`cleanup:` block in `configs/corpus.yaml`, applied in `analysis_units.main` after the prune).** Offense:
drop `tiny` units (<250 chars, IOC/hash dumps); MinHash near-duplicate collapse at Jaccard ≥0.85 (keep
one representative/cluster). Dual: drop units >50% non-prose (composite — `nonprose_ratio` >0.5 catches
debug/traceroute dumps; `longtoken_ratio` >0.3 catches base64 blobs whose letter-runs evade the prose
ratio); for fragments <500 chars, drop exercise prompts + mid-sentence truncations (no terminal
punctuation), keep coherent prose. Thresholds calibrated against the two named exemplars (Nmap
traceroute `93e07dc3739e` nonprose 0.584; Boneh base64 `7eaa3ef734c8` longtoken 0.955) vs normal crypto
prose (nonprose ≤0.35, longtoken 0). Defense untouched. Drop log: `data/cleanup_drops.parquet`.

**Counts.** Pruned 25,301 → cleaned **24,500** (801 dropped): offense 18,787→**18,174** (229 tiny + 384
near-dup); dual 3,760→**3,572** (53 non-prose + 135 bad-short-fragment); defense 2,754 unchanged. Named
exemplar docs confirmed removed; 0 offense units <250 chars remain. Re-audit: offense flagged 16.5%→15.7%
(`tiny`→0; residual is preserved YARA/code `low_alpha` false positives), dual 10.1%→8.4%; near-dup docs
overall 592→160 (residual offense near-dups sit in the 0.7–0.85 band, below the 0.85 collapse threshold
by design). Tests: `tests/test_cleanup.py`.

## 2026-06-04 — Microsoft Security supplement + cross-bucket contamination + frozen v1

**Supplement (operational defense).** Defense was 72% NIST/compliance units (`defense_composition.md`).
Added Microsoft Security blog operational content via the WordPress **REST API only** (Phase-1 decision:
robots allowed but the website ToU bars scraping with no research carve-out, so we use the
intentionally-published API; release = URL+hash only, text not redistributed). `supplement_microsoft.py`:
~1,655 posts → 1,140 after a strict content filter (≥1500 chars, defensive-ops terminology, marketing +
product-pitch-title rejection), tagged with a 12-tag blue-team taxonomy (`subcap`). `integrate_supplement.py`
adds a `source_category` column to `defensive_documents.parquet` (backfill `d3fend_cited`; MS rows
`supplement_microsoft`) and a reproducible `analysis_units._SOURCES` entry (`layer="external_supplement"`,
topic=`subcap`; the `defensive_documents` entry is filtered to `d3fend_cited` to avoid double-count).

**Cap + composition trade-off.** MS posts are long (~3.8 units/post); all 1,140 would dominate defense.
Capped (seed 0) to `supplement_target_govfrac=0.43` → **505 posts / 1,835 units**. Result: NIST+gov **43%**,
MS **40%**, other **17%**. The spec's simultaneous "MS 25–30% / other 25–30%" is arithmetically
**infeasible** — the existing non-NIST defense pool is only ~780 units, so 'other' can't reach 25–30%
alongside NIST+gov≤45%; MS necessarily becomes the operational plurality. Prioritized the headline goal
(de-tilt NIST+gov below 45%); documented; a deferred 2nd vendor source would rebalance 'other'.

**Cross-bucket contamination.** `contamination.py` (MinHash/LSH across buckets) found **62 offense↔
MS-supplement near-duplicate pairs** — Microsoft threat-intel blogs are *also* ATT&CK-cited, so the
supplement re-introduced offense content under a defense label. Dropped the defense-side member
(keep-order offense>dual>defense); 0 remain. `reports/cross_bucket_contamination.md`.

**Frozen v1.** `corpus_validate.validate_corpus` (schema, all buckets, no empty text, no dup `unit_id`,
n_chars bounds, `unit_id==content_hash(text)`) gates `corpus_freeze`, which writes
`data/corpus_manifest.yaml` (version, counts, **sha256**, full config snapshot, source/license tiers) and
`data/analysis_units_v1.parquet`. **v1 = offense 18,174 / dual 3,572 / defense 4,527 = 26,273 units.**
Regeneration order: `integrate_supplement → analysis_units → contamination --drop → corpus_freeze`. MCQ +
unlearning pin v1. Tests: `tests/test_supplement_microsoft.py`, `tests/test_corpus_validate.py`.

## 2026-06-04 — Replaced Microsoft supplement with GitHub blue-team repos (D3FEND-tactic tagged)

**Decision (user):** drop the Microsoft Security supplement entirely (deleted: `supplement_microsoft.py`,
`integrate_supplement.py`, their tests, `inputs/microsoft_security_supplement.parquet`,
`data/defensive_supplement_microsoft.parquet`, the MS reports; `defensive_documents.parquet` reverted to
the 61 D3FEND-cited rows). Rationale: prefer clean-licensed community GitHub content over the MS vendor-blog
ToU situation. (Elastic Security Labs was also assessed and rejected — its ToU explicitly bans scrapers and
it has no volume API; recorded as future work.)

**New supplement (`supplement_github.py`, config `github_supplements`).** Two repos fetched via GitHub
**tarball** (one request/repo = intended distribution): `H3llKa1ser/SOC-Assistant-Guide` (MIT —
detection/DFIR depth) and `A-poc/BlueTeam-Tools` (no license — tool-knowledge breadth, treated
local-build-only / URL+hash release, text not redistributed). Each `.md` is `clean_text`-normalized,
content-filtered (≥600 chars + ≥2 defensive terms; skip LICENSE/CONTRIBUTING/etc.), and tagged with an
**inferred D3FEND tactic** via keyword map → `topic` (Detect/Harden/Model/Isolate/Evict/Restore/Deceive;
`general_defense` fallback). **Tactic is inferred, so authoritative `d3fend_ids` are left empty.** This
unifies the defense bucket under one D3FEND-tactic taxonomy (the D3FEND-cited docs already carry tactics),
replacing the MS-era ad-hoc 12-tag blue-team taxonomy. Result: **101 docs → 146 units**; supplement tactic
mix Detect 90 / Model 3 / Harden 3 / Isolate 3 / Restore 2 (Detect-skewed, realistic).

**Integration + composition.** Added as `analysis_units._SOURCES` entry
(`defensive_supplement_github.parquet` → `layer="external_supplement"`, topic=`topic`); the
`defensive_documents` entry stays filtered to `d3fend_cited`; MS provenance rows replaced by
`source_category="supplement_github"`. Defense = 2,754 D3FEND-cited + 146 supplement = **2,900 units**.
Composition: NIST+US-gov **68%** by raw count — the supplement is small, so it adds operational **Detect**
depth (Detect units 142→274) but does not dilute the NIST mega-docs much. **Decision: the compliance tilt
is handled at MCQ-sampling time (cap NIST's share of generated questions), not by padding the corpus.**

**Contamination + freeze.** Cross-bucket contamination check: **0** offense↔defense near-dups (blue-team
repos aren't ATT&CK-cited, unlike MS threat-intel). Re-frozen v1: **offense 18,174 / dual 3,572 / defense
2,900 = 24,646 units** (`data/analysis_units_v1.parquet` + `corpus_manifest.yaml`, sha256-pinned).
Regeneration order: `supplement_github → analysis_units → contamination → corpus_freeze`. Tests:
`tests/test_supplement_github.py`, `tests/test_corpus_validate.py`.

## 2026-06-05 — Stage 4: WMDP-style MCQ eval (3 regions, doc-level sourcing)

**Purpose.** The v1 capability benchmark the unlearning experiments (step 5) consume. Module
`src/entanglement/mcq.py`; tests `tests/test_mcq.py`; config `configs/corpus.yaml` `mcq:` →
`config.mcq_config()`. All Sonnet, Batch API. Verified offline (172 tests, `FakeClient`); generation
awaits the API key.

**Three regions, generated here (MMLU is the far-general control in step 5, not generated).**
- **attack** — offensive capability (forget target). Source `offensive_documents.parquet`.
- **defend** — *operational* defensive capability (near-neighbor; the entanglement claim). Source
  `defensive_documents.parquet` (D3FEND-cited) + `defensive_supplement_github.parquet`. **Policy/compliance
  sources (NIST_SP_800, US_GOV_CISA) are excluded** (`classify_defense_source(url)`), so questions probe
  detection/hardening/isolation/response *mechanism*, not governance. This is the key fix: under uniform
  sampling NIST dominated `Model`/`Isolate` (60–84% of units), making defend a compliance quiz.
- **substrate** — shared crypto/OS/networking knowledge, the likely *mechanism* of entanglement. Source
  `substrate_corpus.parquet`.

**Doc-level sourcing (replaces the earlier analysis-units sourcing).** `build_sections` reads the three
document corpora and `clean_text`+`resegment`s each doc into coherent ~4000-char sections (hard_max 6000),
capping `max_sections_per_doc=3` for cross-doc diversity. Coherent passages beat the embedding-normalized
units (which split sections), and the document tables carry provenance (`url`→source_category, `tactics`),
which the slim frozen `analysis_units` had dropped. **Cell assignment:** attack = the doc's **kill-chain-
primary** ATT&CK tactic (earliest in `KILL_CHAIN`; fixes the alphabetical "collection" inflation of the old
first-list-element rule); defend = D3FEND tactic with **Deceive/Evict/Restore → `active_response`**;
substrate = `topic`. Yield (n=25, oversample 2.0): **25 cells, ~1,133 sources** — attack 14 (all ~50, exfil
36), substrate 6 (all 50), defend 5 (Detect 50/Harden 46/Isolate 29; **Model 13 + active_response 9 come in
under N — reported, not padded:** operational non-NIST defensive content is genuinely thin there → flagged
as v2 corpus-expansion work).

**Pipeline.** `select_sections` (deterministic head by `section_id` hash) → `generate_mcqs_batch`
(`custom_id=section_id`; cached `GEN_SYSTEM` WMDP rubric: operational/self-contained/transferable, 4 distinct
options, no giveaways; region-aware framing perform/detect-counter/apply-mechanism) → `critique_mcqs_batch`
(graded **1–5** on `operational_focus`, `answer_defensibility`, `distractor_quality` + booleans
`self_contained`,`single_correct`; **keep = all scores ≥3 ∧ both booleans**; `corrected_correct_index` fixes
salvageable mislabels) → **contamination filter** (hashed 8-gram shingle set of the full corpus; drop any
stem sharing a verbatim 8-gram = memorization guard) → **answer-position balance** (deterministic permute so
correct index is ~uniform A–D) → `trim_to_n`. A **cost-cap** guard estimates batch spend before submitting
and aborts over `max_cost_usd` (default $50). Surviving sections → `data/mcq_source_units.parquet`
(held-out; step 5 MUST exclude from unlearning training).

**Outputs.** `data/mcq_eval.parquet` (`MCQ_COLUMNS`: region/cell/source_category/source_url/options/
correct_index/critic scores/…), `data/mcq_source_units.parquet`, `reports/mcq_eval_report.md` (per-cell
counts + mean critic scores, contamination drops, answer-position distribution, 5 examples/cell).

**Run.** Pilot: `ANTHROPIC_API_KEY=… uv run python -m entanglement.mcq --smoke` (3/cell ≈ 150 sources, ~$1.3)
→ review → full `uv run python -m entanglement.mcq` (n=25, ~$10). Verbatim prompts: `GEN_SYSTEM` /
`CRITIC_SYSTEM` in `src/entanglement/mcq.py`.

## 2026-06-11 — WMDP-cyber geometry calibration (side analysis, read-only vs corpus)

Calibrated the PCA-confound protocol against WMDP-cyber's published forget/retain text corpora
(`cais/wmdp-corpora` @ `daf89fa9`, sha256-pinned; forget 1,000 / retain 4,473 docs → our unit
pipeline → 3,917 / 14,738 units; 200/split, seed 0; Llama-3.1-8B layers 4/16/28; binary probe,
chance 0.5). Identical treatment for both corpora; within-task PC removal carries a stated
mechanical-removal caveat. **Result** (`reports/wmdp_cyber_geometry_baseline.md`, reproduce via
`scripts/prep_wmdp_units.py` + pod extraction + `scripts/diag_wmdp_geometry.py`): WMDP survives
drop-3 at L4/L16 (0.80/0.79) but collapses at L28 (0.97 → 0.25 by drop-2); ours collapses at every
layer under the same within-pair protocol (L28 0.95 → 0.22) while surviving global-PC removal
(L28 0.94 @ drop-3). No single verdict: the splits' low-dimensionality differs by depth. Method
calibration only — separability ≠ entanglement; no substrate analysis on WMDP (no labels).
