# Project: Offense/Defense Cyber Capability Entanglement in LLMs

Measurement infrastructure for studying how entangled offensive and defensive
cyber capabilities are inside open-weight LLMs, and whether that entanglement
predicts the collateral-damage cost of unlearning offensive capability.

**The corpus is the primary contribution** — a MITRE-grounded, three-way-tagged
cyber corpus + a WMDP-style MCQ benchmark generated from it. Entanglement findings
exist to show the corpus is load-bearing. Methodology must be communicable and
continuously verified, because reviewers trust the corpus by reading how it was built.

## Research questions

- **RQ1** — How inherent are offense/defense tradeoffs across domains, and how do
  we quantify entanglement?
- **RQ2** — Does embedding/representation distance predict entanglement? Does
  entanglement predict the unlearning tradeoff (the "tax")?
- **RQ3** — How well do decomposed (per-subcapability) and aggregate evals correlate?
- **RQ4** — Policy implications of inherent tradeoffs for capability suppression.

## The minimal-paper arc (current plan)

1. **Get cyber data** ✓ — corpus v1 frozen (offense 18,174 / dual 3,572 / defense
   2,900 = 24,646 units; `data/analysis_units_v1.parquet`, sha256-pinned in
   `data/corpus_manifest.yaml`). Defense corpus is treated as **final**.
2. **Crude retain/forget split (baseline):** forget = offense, retain = defense.
3. **Smart harmless/dual/harmful split (the method):** the dual substrate already
   exists as the corpus's dual bucket.
4. **MCQ evals (Stage 4, current):** generate offense + defense WMDP-style MCQs
   (`src/entanglement/mcq.py`). Benign/dual capability is measured off-the-shelf
   (MMLU `computer_security` + full MMLU) in step 5, not generated.
5. **Unlearning experiments (RMU + MCU):** show the three-way split yields more
   precise, targeted interventions than the crude baseline. Needs GPU.

Models: **Llama-3.1-8B** primary; **Gemma2-9B** + **Qwen3-8B** cross-reference.
Unlearning methods: **MCU** + **RMU**. **Cyber first, bio later.**

## Standing facts / caveats (load-bearing)

- **Three-band eval design.** Generate only attack (forget target) + defend
  (near-neighbor harmless) MCQs. The defend eval must be valence-isolated and
  topic-aligned — MMLU `computer_security` can't replace it (valence-mixed,
  aggregate). Dual/benign is off-the-shelf.
- **Separability is topic/register-confounded.** Offense/dual/defense separability
  measures topic/source, not valence. Valence claims require the unlearning-tax
  experiment, not separability. Use topic-matched designs.
- **Defense corpus is compliance-heavy** (~68% NIST/policy by unit count even after
  the GitHub blue-team supplement). Frame defense as governance/compliance unless
  rebalanced; the NIST tilt is handled at MCQ-sampling time, not by padding.
- **In representation space the dual substrate clusters with defense; offense is the
  outlier** — and the split sharpens with depth (where unlearning acts).

## Working norms (override defaults)

- **Enter plan mode often** — align on approach before non-trivial implementation;
  prefer asking over inferring on design forks.
- **Document + test every stage.** New code ships focused unit tests in the same
  step (extend the `tests/` pattern; API-dependent paths use an injectable fake
  client, see `test_quality.py`/`test_mcq.py`). No stage is done until `uv run
  pytest` is green. Append `docs/METHODOLOGY.md` each stage (sources, every
  filter/threshold + why, verbatim LLM prompts, anomalies, decision log).
- **Provenance on every doc**; tagging must be auditable per-document.
- **Pin all external sources by version + checksum** (`inputs/SOURCES.md`).
- **Use `uv`** (not bare pip). **Type-annotate.** **Prefer `polars` over `pandas`**
  for heavy joins. Keep methodology choices in `configs/*.yaml`, not source.
- **Commit frequently** after verified changes; push to the private remote to back
  up. **Never commit `data/` or `inputs/`** (non-redistributable scraped text);
  only `data/corpus_manifest.yaml` is tracked.
- **Surface anomalies** rather than smoothing them over.

See `README.md` for layout and `docs/METHODOLOGY.md` for the authoritative
stage-by-stage record.
