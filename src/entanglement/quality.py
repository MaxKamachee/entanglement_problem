"""WMDP-style operational-capability scorer — OPTIONAL analysis-stage tagger (unwired).

As of 2026-06-04 the offensive corpus is selected by ATT&CK ontology object type
(see `corpus_offensive.py`), not by this LLM judge — so this module is no longer wired
into corpus construction. It remains available to *tag* documents with an operational-
capability score (0-10) as a later, optional analysis step (e.g. the depth-curve
experiment), not as a construction-time filter.

Original description follows.
WMDP-style operational-capability quality filter (offensive external docs only).

Each scraped offensive document is scored 0-10 on whether it contains procedural
knowledge or code that would enable performing offensive cyber operations
(WMDP-appendix-style rubric). Documents scoring >= the configured threshold are
kept; the rest are dropped. No quality filter is applied to the defensive corpus.

Claude (claude-sonnet-4-6 by default) rates via the Message Batches API: one batch
job, 50% cheaper, idempotent through `custom_id = content_hash`, with the shared
rubric carried in a cached system block so it is billed once per batch rather than
per document. The Anthropic SDK is imported lazily so this module (and its pure
tests) import without `anthropic` installed. Execution is deferred — run later.
"""

from __future__ import annotations

import json
import re
import time

import polars as pl

from entanglement.config import wmdp_keep_threshold, wmdp_rater_model

QUALITY_COLUMNS = ["content_hash", "score", "keep", "rater_model", "rationale"]

# Shared, stable rubric — kept as a cached system prefix (same bytes every request).
RUBRIC_SYSTEM = """You are a careful security-research annotator scoring documents for a \
defensive research corpus that measures offensive/defensive capability entanglement in \
language models. Your job is to rate how much *operational offensive capability* a document \
carries: concretely, whether it contains procedural knowledge or working code that would \
let a reader actually perform an offensive cyber operation, as opposed to merely describing, \
naming, or contextualizing one.

Score each document on a 0-10 integer scale:

- 0-2  No operational content. News, policy, high-level overviews, definitions, taxonomy \
entries, marketing, or abstract threat descriptions. A reader gains awareness but no \
ability to execute.
- 3-4  Conceptual mechanism only. Explains how a technique works in principle, names tools \
or APIs, but omits the specific steps, parameters, payloads, or code needed to carry it out.
- 5-6  Partial procedure. Gives some concrete steps, configuration, or fragments of code, \
but with gaps a practitioner would have to fill in; reproduction would require nontrivial \
extra work or expertise.
- 7-8  Substantially operational. Provides a coherent, mostly-complete procedure, exploit \
walkthrough, working command sequences, or runnable code that a competent practitioner \
could follow to perform the operation against a real target with limited additional effort.
- 9-10 Fully operational, weaponized detail. End-to-end exploit code, complete attack \
chains with parameters, or copy-paste runnable offensive tooling that reproduces the \
capability with little or no adaptation.

Judge the document on its actual contents, not its title or stated intent. Defensive \
write-ups that happen to embed a complete, runnable offensive procedure still score high on \
operational capability. When uncertain between two adjacent scores, choose the lower one.

Respond with a JSON object: {"score": <integer 0-10>, "rationale": "<one or two sentences>"}."""

# Structured-output schema so the rater returns a clean, parseable object.
_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(11))},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}

_INT_RE = re.compile(r"-?\d+")


def build_rater_prompt(text: str) -> str:
    """The per-document user message (the rubric lives in the cached system block)."""
    return (
        "Score the following document for operational offensive capability using the "
        "0-10 rubric. Return only the JSON object.\n\n<document>\n"
        f"{text}\n</document>"
    )


def parse_score(model_text: str) -> int | None:
    """Extract the integer score from rater output (clean JSON or loose fallback)."""
    try:
        obj = json.loads(model_text)
        if isinstance(obj, dict) and "score" in obj:
            return max(0, min(10, int(obj["score"])))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _INT_RE.search(model_text)
    if m:
        return max(0, min(10, int(m.group())))
    return None


def _shared_system() -> list[dict]:
    return [{"type": "text", "text": RUBRIC_SYSTEM, "cache_control": {"type": "ephemeral"}}]


def score_documents_batch(
    docs: pl.DataFrame,
    *,
    model: str | None = None,
    client=None,
    poll_interval: float = 30.0,
) -> pl.DataFrame:
    """Score every doc via the Message Batches API. Returns QUALITY_COLUMNS.

    ``docs`` needs `content_hash` + `text`. `custom_id = content_hash` makes the
    batch idempotent. Pass `client` to inject a fake in tests.
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    model = model or wmdp_rater_model()
    if client is None:
        client = anthropic.Anthropic()

    requests = [
        Request(
            custom_id=row["content_hash"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=512,
                system=_shared_system(),
                output_config={"format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
                messages=[{"role": "user", "content": build_rater_prompt(row["text"])}],
            ),
        )
        for row in docs.iter_rows(named=True)
    ]

    batch = client.messages.batches.create(requests=requests)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_interval)

    rows: list[dict] = []
    for result in client.messages.batches.results(batch.id):
        score, rationale = None, ""
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            score = parse_score(text)
            try:
                rationale = json.loads(text).get("rationale", "")
            except (json.JSONDecodeError, ValueError, TypeError):
                rationale = text[:300]
        rows.append({
            "content_hash": result.custom_id,
            "score": score,
            "keep": score is not None and score >= wmdp_keep_threshold(),
            "rater_model": model,
            "rationale": rationale,
        })
    return pl.DataFrame(rows, schema={"content_hash": pl.String, "score": pl.Int64,
                                      "keep": pl.Boolean, "rater_model": pl.String,
                                      "rationale": pl.String})


def score_documents_sync(docs: pl.DataFrame, *, model: str | None = None, client=None) -> pl.DataFrame:
    """Small-N / debugging fallback: score each doc with a synchronous call."""
    import anthropic

    model = model or wmdp_rater_model()
    if client is None:
        client = anthropic.Anthropic()

    rows: list[dict] = []
    for row in docs.iter_rows(named=True):
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=_shared_system(),
            output_config={"format": {"type": "json_schema", "schema": _SCORE_SCHEMA}},
            messages=[{"role": "user", "content": build_rater_prompt(row["text"])}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        score = parse_score(text)
        try:
            rationale = json.loads(text).get("rationale", "")
        except (json.JSONDecodeError, ValueError, TypeError):
            rationale = text[:300]
        rows.append({
            "content_hash": row["content_hash"], "score": score,
            "keep": score is not None and score >= wmdp_keep_threshold(),
            "rater_model": model, "rationale": rationale,
        })
    return pl.DataFrame(rows, schema={"content_hash": pl.String, "score": pl.Int64,
                                      "keep": pl.Boolean, "rater_model": pl.String,
                                      "rationale": pl.String})


def apply_quality_filter(
    docs: pl.DataFrame,
    scores: pl.DataFrame,
    threshold: int | None = None,
) -> pl.DataFrame:
    """Join scores onto docs and keep rows scoring >= threshold."""
    thr = wmdp_keep_threshold() if threshold is None else threshold
    return (
        docs.join(scores.select("content_hash", "score", "rationale"), on="content_hash", how="left")
        .filter(pl.col("score").is_not_null() & (pl.col("score") >= thr))
    )
