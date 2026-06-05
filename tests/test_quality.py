"""Tests for the WMDP rater — pure parsing/filtering + a mocked batch client."""

from __future__ import annotations

import polars as pl
import pytest

from entanglement.quality import (
    QUALITY_COLUMNS,
    RUBRIC_SYSTEM,
    apply_quality_filter,
    build_rater_prompt,
    parse_score,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 8, "rationale": "complete exploit"}', 8),
        ('{"score": 0, "rationale": "news"}', 0),
        ("the score is 7 out of 10", 7),       # loose fallback
        ('{"score": 15}', 10),                  # clamped high
        ('{"score": -3}', 0),                   # clamped low
        ("no number here at all", None),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


def test_build_rater_prompt_embeds_text_not_rubric():
    p = build_rater_prompt("PAYLOAD-XYZ")
    assert "PAYLOAD-XYZ" in p
    # rubric lives in the cached system block, not the per-doc prompt
    assert RUBRIC_SYSTEM not in p


def test_apply_quality_filter_keeps_threshold():
    docs = pl.DataFrame({
        "content_hash": ["a", "b", "c", "d"],
        "text": ["...", "...", "...", "..."],
    })
    scores = pl.DataFrame({
        "content_hash": ["a", "b", "c", "d"],
        "score": [9, 7, 6, None],
        "rationale": ["x", "y", "z", "w"],
    })
    kept = apply_quality_filter(docs, scores, threshold=7)
    assert set(kept["content_hash"].to_list()) == {"a", "b"}  # 6 and null dropped


# --- batch flow against a fake Anthropic client ---

class _Result:
    def __init__(self, custom_id, text):
        self.custom_id = custom_id
        self.result = type("R", (), {
            "type": "succeeded",
            "message": type("M", (), {
                "content": [type("B", (), {"type": "text", "text": text})()]
            })(),
        })()


class FakeBatches:
    def __init__(self, results):
        self._results = results

    def create(self, requests):
        self._ids = [r["custom_id"] for r in requests]
        return type("Batch", (), {"id": "batch_1", "processing_status": "in_progress"})()

    def retrieve(self, _id):
        return type("Batch", (), {"id": "batch_1", "processing_status": "ended"})()

    def results(self, _id):
        return self._results


class FakeMessages:
    def __init__(self, results):
        self.batches = FakeBatches(results)


class FakeClient:
    def __init__(self, results):
        self.messages = FakeMessages(results)


def test_score_documents_batch_with_fake_client(monkeypatch):
    pytest.importorskip("anthropic")
    from entanglement import quality

    docs = pl.DataFrame({"content_hash": ["h1", "h2"], "text": ["exploit code", "news article"]})
    fake = FakeClient([
        _Result("h1", '{"score": 9, "rationale": "runnable exploit"}'),
        _Result("h2", '{"score": 1, "rationale": "just news"}'),
    ])
    out = quality.score_documents_batch(docs, model="claude-sonnet-4-6", client=fake, poll_interval=0)
    assert out.columns == QUALITY_COLUMNS
    by_hash = {r["content_hash"]: r for r in out.iter_rows(named=True)}
    assert by_hash["h1"]["score"] == 9 and by_hash["h1"]["keep"] is True
    assert by_hash["h2"]["score"] == 1 and by_hash["h2"]["keep"] is False
