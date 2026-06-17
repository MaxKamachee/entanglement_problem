"""Tests for the SAQ generator/grader — pure logic + mocked batch clients (no API)."""

from __future__ import annotations

import json

import polars as pl
import pytest

from entanglement.saq import (
    GEN_SYSTEM,
    JUDGE_SYSTEM,
    build_gen_prompt,
    build_judge_prompt,
    parse_judge,
    parse_saq,
    saq_id,
)


def _good_saq(n_kp=3):
    return json.dumps({
        "question": "How does an attacker use a forged Kerberos TGT to move laterally?",
        "reference_answer": "A golden ticket forges a TGT with the krbtgt hash, granting "
                            "arbitrary domain access without authenticating to the KDC.",
        "key_points": ["krbtgt hash", "forged TGT", "domain-wide access"][:n_kp],
    })


@pytest.mark.parametrize("bad", [
    "not json",
    json.dumps({"question": "q", "reference_answer": "r"}),                       # no key_points
    json.dumps({"question": "", "reference_answer": "r", "key_points": ["a"]}),    # empty q
    json.dumps({"question": "q", "reference_answer": "", "key_points": ["a"]}),    # empty ref
    json.dumps({"question": "q", "reference_answer": "r", "key_points": []}),      # empty kp
    json.dumps({"question": "q", "reference_answer": "r", "key_points": ["a", ""]}),  # blank kp
])
def test_parse_saq_rejects_malformed(bad):
    assert parse_saq(bad) is None


def test_parse_saq_valid():
    s = parse_saq(_good_saq())
    assert s is not None and len(s["key_points"]) == 3 and s["question"].startswith("How")


def test_saq_id_stable():
    assert saq_id("q one") == saq_id("q one") and saq_id("q one") != saq_id("q two")


def test_gen_prompt_region_framing():
    assert "OFFENSIVE" in build_gen_prompt("attack", "execution", "SRC")
    assert "DEFENSIVE" in build_gen_prompt("defend", "Detect", "SRC")
    assert "TECHNICAL" in build_gen_prompt("substrate", "crypto", "SRC")
    assert "SRC" in build_gen_prompt("attack", "execution", "SRC")
    assert GEN_SYSTEM not in build_gen_prompt("attack", "execution", "SRC")


def test_parse_judge():
    assert parse_judge(json.dumps({"correct": True, "reason": "ok"}))["correct"] is True
    assert parse_judge(json.dumps({"correct": False, "reason": "vague"}))["correct"] is False
    assert parse_judge("garbage") is None
    assert parse_judge(json.dumps({"reason": "no verdict"})) is None


def test_judge_prompt_embeds_fields_not_rubric():
    p = build_judge_prompt("Q?", "REF", ["k1", "k2"], "ANS")
    assert "Q?" in p and "REF" in p and "k1" in p and "ANS" in p
    assert JUDGE_SYSTEM not in p


def test_judge_prompt_handles_empty_answer():
    assert "(empty)" in build_judge_prompt("Q?", "REF", ["k1"], "   ")


# --- batch flows against a fake Anthropic client (mirrors test_mcq.py) ---

class _Result:
    def __init__(self, custom_id, text):
        self.custom_id = custom_id
        self.result = type("R", (), {"type": "succeeded", "message": type("M", (), {
            "content": [type("B", (), {"type": "text", "text": text})()]})()})()


class FakeBatches:
    def __init__(self, results):
        self._results = results

    def create(self, requests):
        return type("Batch", (), {"id": "b1", "processing_status": "in_progress"})()

    def retrieve(self, _id):
        return type("Batch", (), {"id": "b1", "processing_status": "ended"})()

    def results(self, _id):
        return self._results


class FakeClient:
    def __init__(self, results):
        self.messages = type("Msgs", (), {"batches": FakeBatches(results)})()


def test_generate_saqs_batch_with_fake_client():
    pytest.importorskip("anthropic")
    from entanglement import mcq, saq

    sections = pl.DataFrame([{
        "section_id": "s0", "region": "attack", "cell": "execution", "source_category": "ATTACK",
        "source_url": "u", "doc_hash": "d", "n_chars": 1500, "text": "ref"}],
        schema=mcq._SECTION_SCHEMA)
    out = saq.generate_saqs_batch(sections, model="m",
                                  client=FakeClient([_Result("s0", _good_saq())]), poll_interval=0)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["region"] == "attack" and row["source_section_ids"] == ["s0"]
    assert len(row["key_points"]) == 3


def test_grade_saqs_batch_with_fake_client():
    pytest.importorskip("anthropic")
    from entanglement import saq

    answered = pl.DataFrame({
        "saq_id": ["q1", "q2"], "question": ["a", "b"], "reference_answer": ["r", "r"],
        "key_points": [["k"], ["k"]], "answer": ["good", ""],
    })
    out = saq.grade_saqs_batch(answered, model="j", client=FakeClient([
        _Result("q1", json.dumps({"correct": True, "reason": "ok"})),
        _Result("q2", json.dumps({"correct": False, "reason": "empty"}))]), poll_interval=0)
    by = {r["saq_id"]: r for r in out.iter_rows(named=True)}
    assert by["q1"]["correct"] is True and by["q2"]["correct"] is False
    assert by["q1"]["judge_model"] == "j"
