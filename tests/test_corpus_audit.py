"""Tests for the corpus quality-audit helpers (pure, offline)."""

from __future__ import annotations

from entanglement.corpus_audit import (
    doc_metrics,
    flag_reasons,
    minhash_signatures,
    near_duplicate_pairs,
)


def test_doc_metrics_substantive_text():
    txt = ("The TCP three-way handshake establishes a reliable connection between two hosts. "
           "The client sends a SYN segment to begin the exchange and propose an initial sequence "
           "number. The server replies with a SYN-ACK segment acknowledging the client and "
           "supplying its own sequence number. The client then acknowledges with a final ACK "
           "segment, after which the connection is established and data transfer may begin.")
    m = doc_metrics(txt)
    assert m["n_chars"] > 250
    assert m["alpha_ratio"] > 0.7
    assert m["sentence_count"] == 4
    assert m["boilerplate_count"] == 0
    assert flag_reasons(m) == []


def test_flag_low_alpha_and_boilerplate():
    noisy = "===|||  4o4  ||| === " * 5 + " subscribe to our newsletter. accept all cookies."
    m = doc_metrics(noisy)
    reasons = flag_reasons(m)
    assert "low_alpha" in reasons or "high_boilerplate" in reasons


def test_flag_tiny():
    assert "tiny" in flag_reasons(doc_metrics("short text."))


def test_minhash_detects_near_duplicate():
    base = "the quick brown fox jumps over the lazy dog near the river bank every morning today"
    dup = base + " and then it rests"          # ~near-identical
    diff = "completely unrelated content about cryptographic key exchange and elliptic curves here"
    sigs = minhash_signatures([base, dup, diff], n_perm=64)
    pairs = near_duplicate_pairs(sigs, bands=16, threshold=0.6)
    flagged = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in flagged          # base ~ dup
    assert (0, 2) not in flagged and (1, 2) not in flagged
