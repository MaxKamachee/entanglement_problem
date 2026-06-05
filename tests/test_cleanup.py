"""Tests for the post-prune cleanup filters (pure, offline)."""

from __future__ import annotations

import polars as pl

from entanglement.cleanup import (
    apply_cleanup,
    is_bad_short_fragment,
    is_nonprose,
    longtoken_ratio,
    nonprose_ratio,
)


def test_nonprose_and_longtoken_ratios():
    prose = "The cipher maps a fixed size block under a secret key to a ciphertext block."
    assert nonprose_ratio(prose) < 0.35
    assert longtoken_ratio(prose) == 0.0
    blob = "data " + "A1b2C3d4E5f6G7h8" * 8          # one >30-char base64-ish token
    assert longtoken_ratio(blob) > 0.3


def test_is_nonprose_catches_dump_and_blob_not_prose():
    traceroute = "SENT 192.168.0.21:53 > 64.13.134.52:53 ttl=8 id=4826 iplen=28 1 2 3 4 5 6 7 8 9"
    base64 = "header " + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2" * 4
    assert is_nonprose(traceroute, 0.5, 0.3) or nonprose_ratio(traceroute) > 0.4
    assert is_nonprose(base64, 0.5, 0.3)
    prose = "A block cipher is a deterministic algorithm operating on fixed length groups of bits."
    assert not is_nonprose(prose, 0.5, 0.3)


def test_is_bad_short_fragment():
    assert is_bad_short_fragment("Exercise 3.1: prove that the scheme is secure.") is True
    assert is_bad_short_fragment("the connection is established and then it") is True   # no terminal punct
    assert is_bad_short_fragment("The handshake completes after the final ACK.") is False


def _row(uid, bucket, text):
    return {"unit_id": uid, "bucket": bucket, "layer": f"{bucket}_external",
            "topic": "x", "n_chars": len(text), "text": text}


def test_apply_cleanup_drops_expected():
    body = ("the quick brown fox jumps over the lazy dog while the diligent analyst inspects "
            "the captured network traffic for anomalies across many distinct sessions today ") * 3
    rows = [
        _row("off_tiny", "offense", "hxxps[://]bad[.]site/a 1.2.3.4 deadbeef"),       # tiny
        _row("off_a", "offense", body + "one two three"),                            # near-dup pair...
        _row("off_b", "offense", body + "one two four"),                             # ...drop one
        _row("dual_blob", "dual", "x " + "QUJDREVGR0hJSktMTU5PUA" * 6),              # base64 blob
        _row("dual_good", "dual", body + "and the mechanism is described in detail."),  # keep
        _row("def_keep", "defense", body + "the control mitigates the risk effectively."),  # keep
    ]
    units = pl.DataFrame(rows)
    cleaned, log = apply_cleanup(units, offense_neardup_jaccard=0.6)
    reasons = {r["unit_id"]: r["reason"] for r in log.iter_rows(named=True)}
    assert reasons.get("off_tiny") == "offense_tiny"
    assert reasons.get("dual_blob") == "dual_nonprose"
    assert "off_a" in reasons or "off_b" in reasons          # exactly one of the near-dup pair dropped
    assert ("off_a" in reasons) != ("off_b" in reasons)
    kept = set(cleaned["unit_id"].to_list())
    assert {"dual_good", "def_keep"} <= kept                 # coherent prose + defense survive
    assert "off_tiny" not in kept and "dual_blob" not in kept
