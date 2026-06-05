# Entanglement

Measurement infrastructure (corpus + benchmark + analysis pipeline) for studying
how deeply entangled offensive and defensive cyber capabilities are inside
open-weight LLMs, and whether that entanglement predicts the collateral-damage
cost of capability suppression.

See `CLAUDE.md` for the full project spec.

## Reproduce Phase 1

```bash
uv sync --extra dev
uv run python -m entanglement.pin_sources   # Stage 0: pin + verify source data
# ... (edge-table build, coverage report — added as stages land)
```

Pinned source versions and checksums live in `inputs/SOURCES.md`.
