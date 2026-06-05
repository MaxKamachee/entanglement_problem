"""Tests for the scraping pipeline — pure helpers + mocked fetch flow, no sockets."""

from __future__ import annotations

import pytest

from entanglement import scrape
from entanglement.scrape import (
    DOCUMENTS_COLUMNS,
    PROVENANCE_COLUMNS,
    FetchResult,
    RateLimiter,
    RobotsCache,
    classify_content_type,
    content_hash,
    dedup_urls,
    fetch_one,
    is_stub,
    results_to_frames,
)


def test_content_hash_stable_and_distinct():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


@pytest.mark.parametrize("n,expected", [(1999, True), (2000, False), (2001, False)])
def test_is_stub_boundary(n, expected):
    assert is_stub("x" * n, min_chars=2000) is expected


@pytest.mark.parametrize(
    "ct,url,expected",
    [
        ("application/pdf", "http://x/y", "pdf"),
        (None, "http://x/paper.pdf", "pdf"),
        ("text/html; charset=utf-8", "http://x/y", "html"),
        (None, "http://x/page", "html"),
        ("image/png", "http://x/y.png", "other"),
        ("application/zip", "http://x/y.zip", "other"),
    ],
)
def test_classify_content_type(ct, url, expected):
    assert classify_content_type(ct, url) == expected


def test_dedup_urls_collapses_www_slash_fragment():
    # canonicalize_url strips www./trailing-slash/fragment but preserves scheme
    out = dedup_urls([
        "https://www.example.com/page/",
        "https://example.com/page",
        "https://example.com/page#frag",
        "https://other.com/x",
    ])
    assert out == ["https://example.com/page", "https://other.com/x"]


def test_rate_limiter_sleeps_per_host():
    t = {"now": 0.0}
    slept: list[float] = []
    rl = RateLimiter(1.0, clock=lambda: t["now"], sleep=lambda s: slept.append(s))
    rl.wait("a.com")          # first call: no sleep
    rl.wait("a.com")          # immediate repeat: must sleep ~1.0
    rl.wait("b.com")          # different host: no sleep
    assert slept == [1.0]


def test_robots_cache_fail_open_and_disallow():
    # fetcher returns a robots body disallowing /private for *
    robots_body = "User-agent: *\nDisallow: /private\n"
    rc = RobotsCache("ua", fetcher=lambda url: robots_body)
    assert rc.allowed("https://h.com/public") is True
    assert rc.allowed("https://h.com/private/x") is False
    # fetch failure => fail-open (allowed)
    rc2 = RobotsCache("ua", fetcher=lambda url: None)
    assert rc2.allowed("https://h.com/anything") is True


# --- fetch_one with a fake httpx-like client ---

class FakeResp:
    def __init__(self, status, content, content_type):
        self.status_code = status
        self.content = content
        self.headers = {"content-type": content_type}


class FakeClient:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url, follow_redirects=True):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _allow_all():
    return RobotsCache("ua", fetcher=lambda url: "")


def _noop_limiter():
    return RateLimiter(0.0, clock=lambda: 0.0, sleep=lambda s: None)


def test_fetch_one_html_success(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape, "extract_html", lambda raw: "BODY " * 1000)
    client = FakeClient(FakeResp(200, b"<html>...</html>", "text/html"))
    r = fetch_one("http://h.com/a", client=client, limiter=_noop_limiter(),
                  robots=_allow_all(), cache_dir=tmp_path)
    assert r.success and r.extractor == "trafilatura" and r.reason == ""
    # cached: second call hits disk even if extractor would now differ
    monkeypatch.setattr(scrape, "extract_html", lambda raw: "DIFFERENT")
    r2 = fetch_one("http://h.com/a", client=FakeClient(Exception("boom")),
                   limiter=_noop_limiter(), robots=_allow_all(), cache_dir=tmp_path)
    assert r2.text == r.text


def test_fetch_one_stub_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr(scrape, "extract_html", lambda raw: "short")
    client = FakeClient(FakeResp(200, b"x", "text/html"))
    r = fetch_one("http://h.com/s", client=client, limiter=_noop_limiter(),
                  robots=_allow_all(), cache_dir=tmp_path)
    assert not r.success and r.reason == "stub"


def test_fetch_one_dead_and_nontext(tmp_path):
    dead = fetch_one("http://h.com/404", client=FakeClient(FakeResp(404, b"", "text/html")),
                     limiter=_noop_limiter(), robots=_allow_all(), cache_dir=tmp_path)
    assert not dead.success and dead.reason == "dead"
    img = fetch_one("http://h.com/p.png", client=FakeClient(FakeResp(200, b"\x89PNG", "image/png")),
                    limiter=_noop_limiter(), robots=_allow_all(), cache_dir=tmp_path)
    assert not img.success and img.reason == "non_text"


def test_fetch_one_robots_disallow(tmp_path):
    rc = RobotsCache("ua", fetcher=lambda url: "User-agent: *\nDisallow: /\n")
    r = fetch_one("http://h.com/x", client=FakeClient(FakeResp(200, b"x", "text/html")),
                  limiter=_noop_limiter(), robots=rc, cache_dir=tmp_path)
    assert not r.success and r.reason == "robots_disallow"


def test_results_to_frames_dedup_by_hash():
    body = "BODY " * 1000
    results = [
        FetchResult("http://a", "http://a", "t", 200, body, "trafilatura", True, ""),
        FetchResult("http://b", "http://b", "t", 200, body, "trafilatura", True, ""),  # dup content
        FetchResult("http://c", "http://c", "t", 404, None, None, False, "dead"),
    ]
    docs, prov = results_to_frames(results)
    assert docs.columns == DOCUMENTS_COLUMNS and prov.columns == PROVENANCE_COLUMNS
    assert docs.height == 1            # deduped by content_hash
    assert prov.height == 3            # every attempt recorded
    assert prov.filter(prov["success"]).height == 2


def test_extract_pdf_importorskip():
    pytest.importorskip("pypdf")
    # malformed PDF bytes -> None, not an exception
    assert scrape.extract_pdf(b"not a real pdf") is None
