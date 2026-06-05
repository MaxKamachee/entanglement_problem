# Entanglement

Measurement infrastructure (corpus + benchmark + analysis pipeline) for studying
how deeply entangled offensive and defensive cyber capabilities are inside
open-weight LLMs, and whether that entanglement predicts the collateral-damage
cost of capability suppression (unlearning).

The **corpus is the primary contribution**: a MITRE-grounded, three-way-tagged
(offense / dual-substrate / defense) cyber corpus, frozen and sha256-pinned, plus
a WMDP-style MCQ benchmark generated from it. The entanglement findings demonstrate
the corpus is load-bearing.

- **Methodology (authoritative, stage-by-stage):** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- **Corpus card (sources, licensing, caveats):** [`docs/CORPUS_CARD.md`](docs/CORPUS_CARD.md)
- **Pinned source versions + checksums:** [`inputs/SOURCES.md`](inputs/SOURCES.md)

## Layout

- `src/entanglement/` — the pipeline. Each module is either a tested library or a
  `python -m entanglement.<module>` stage (see METHODOLOGY for the run order).
- `tests/` — `uv run pytest` (offline; API-dependent paths are tested with a fake client).
- `configs/` — methodology parameters kept out of code so they're reviewable
  (`corpus.yaml`, `subcapabilities.yaml`, …).
- `data/`, `inputs/` — **not committed** (non-redistributable scraped text +
  regenerable intermediates). The released artifact is the pipeline + URL lists +
  hashes; only `data/corpus_manifest.yaml` (the frozen-corpus manifest) is tracked.

## Setup

```bash
uv sync --extra dev
uv run pytest            # full suite
```

## Pipeline (high level — see docs/METHODOLOGY.md for each stage)

1. **Parse sources** → ATT&CK + D3FEND tables (`attack.py`, `d3fend.py`).
2. **Stage A** — subcapability taxonomy + topic partition (`subcapabilities.py`).
3. **Build corpora** — offensive / defensive / substrate + GitHub blue-team
   supplement (`corpus_offensive.py`, `corpus_defensive.py`, `substrate.py`,
   `supplement_github.py`).
4. **Assemble analysis units** — normalize + resegment + prune + cleanup
   (`analysis_units.py`, `units.py`, `cleanup.py`), check cross-bucket
   contamination (`contamination.py`), then **freeze v1** (`corpus_freeze.py` →
   `data/analysis_units_v1.parquet` + `corpus_manifest.yaml`).
5. **Analysis** — document/representation separability (`separability.py`,
   `representation_analysis.py`; hidden states extracted on GPU via
   `scripts/extract_hidden_states*.py`).
6. **MCQ benchmark** — WMDP-style offense+defense eval (`mcq.py`; needs
   `ANTHROPIC_API_KEY`):

   ```bash
   ANTHROPIC_API_KEY=… uv run python -m entanglement.mcq --smoke   # pilot, 3/cell
   uv run python -m entanglement.mcq                               # full, 25/cell
   ```

7. **Unlearning experiments** (RMU/MCU; separate, GPU) — measure the
   offense-forget / defense-retain tradeoff against the MCQ eval + off-the-shelf
   MMLU. *(Upcoming.)*

Current corpus: **v1** — offense 18,174 / dual 3,572 / defense 2,900 = 24,646
units (sha256-pinned in `data/corpus_manifest.yaml`).
