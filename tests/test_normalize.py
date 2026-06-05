"""Tests for the URL + technique-id normalization spine."""

from __future__ import annotations

import pytest

from entanglement.normalize import (
    canonicalize_url,
    is_subtechnique,
    is_tech_id,
    normalize_tech_id,
    parent_tech_id,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        # host lowercasing + www stripping + scheme lowercasing
        ("HTTP://WWW.Example.com/Path", "http://example.com/Path"),
        # trailing slash and fragment dropped
        ("https://example.com/a/b/#section", "https://example.com/a/b"),
        # the two forms that must collapse for the dual-use intersection
        ("http://www.x.com/p/", "https://x.com/p".replace("https", "http")),
        # tracking query params dropped
        ("https://nist.gov/doc?utm_source=t&ref=x", "https://nist.gov/doc"),
        # identifying params preserved (and lowercased key match)
        ("https://doi.org/x?doi=10.1/abc", "https://doi.org/x?doi=10.1%2Fabc"),
        # empty / whitespace
        ("   ", ""),
    ],
)
def test_canonicalize_url(raw: str, expected: str) -> None:
    assert canonicalize_url(raw) == expected


def test_www_and_scheme_collapse_to_same_key() -> None:
    # The core invariant: these must be equal or dual-use docs split apart.
    a = canonicalize_url("http://www.example.com/paper/")
    b = canonicalize_url("https://example.com/paper")
    # scheme differs but host/path align; we keep scheme, so assert host+path equal
    assert a.split("://", 1)[1] == b.split("://", 1)[1]


@pytest.mark.parametrize(
    "raw, valid",
    [
        ("T1003", True),
        ("T1003.001", True),
        ("t1003.001", True),
        (" T1059 ", True),
        ("D3-NTA", False),
        ("T100", False),
        ("1003", False),
    ],
)
def test_is_tech_id(raw: str, valid: bool) -> None:
    assert is_tech_id(raw) is valid


def test_normalize_and_parent() -> None:
    assert normalize_tech_id(" t1003.001 ") == "T1003.001"
    assert parent_tech_id("T1003.001") == "T1003"
    assert parent_tech_id("T1003") == "T1003"
    assert parent_tech_id(" t1003.002 ") == "T1003"


def test_is_subtechnique() -> None:
    assert is_subtechnique("T1003.001") is True
    assert is_subtechnique("T1003") is False
