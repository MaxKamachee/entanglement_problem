"""Shared normalization spine for URLs and ATT&CK technique IDs.

Every offense/defense join keys on these, so they live in one module with
their own tests. The dual-use signal is a URL set-intersection between the
ATT&CK and D3FEND reference sets; inconsistent canonicalization would split a
genuinely-shared document into two singletons and erase it from the
intersection. We therefore err toward *under*-merging: drop all query params
except an explicit identifying allowlist.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query params that genuinely identify a document (vs. tracking/session junk).
# Most DOIs/arXiv ids live in the URL *path*, not the query, so this is small.
_IDENTIFYING_PARAMS: frozenset[str] = frozenset({"doi", "arxiv", "arxivid"})

# ATT&CK technique id: T#### optionally with a .### sub-technique suffix.
_TECH_ID_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for set-membership comparison.

    Lowercases scheme+host, strips a leading ``www.``, drops the fragment and
    any trailing slash, and keeps only allowlisted query params (sorted for
    stability). Returns the input stripped if it cannot be parsed as a URL.
    """
    raw = url.strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    # urlsplit puts scheme-less inputs entirely in `path`; leave those alone.
    if not parts.netloc:
        return raw.rstrip("/")

    scheme = (parts.scheme or "https").lower()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")

    kept = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() in _IDENTIFYING_PARAMS
    )
    query = urlencode(kept)
    return urlunsplit((scheme, host, path, query, ""))


def is_tech_id(value: str) -> bool:
    """True if `value` looks like an ATT&CK technique id (T1003 / T1003.001)."""
    return bool(_TECH_ID_RE.match(value.strip().upper()))


def normalize_tech_id(tech_id: str) -> str:
    """Normalize a technique id to canonical form (uppercase, trimmed)."""
    return tech_id.strip().upper()


def parent_tech_id(tech_id: str) -> str:
    """Return the parent technique id, dropping any sub-technique suffix.

    ``T1003.001`` -> ``T1003``; ``T1003`` -> ``T1003``.
    """
    return normalize_tech_id(tech_id).split(".", 1)[0]


def is_subtechnique(tech_id: str) -> bool:
    """True if `tech_id` is a sub-technique (has a .### suffix)."""
    return "." in normalize_tech_id(tech_id)
