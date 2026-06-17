"""Tests for the torch-free helpers in the GPU eval/unlearn scripts (no GPU, no torch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # top-level imports are stdlib only; torch is lazy
    return mod


ev = _load("eval_suite")
ul = _load("unlearn_rmu")


# --- eval_suite pure helpers ---

def test_format_mcq_prompt_and_pick_letter():
    p = ev.format_mcq_prompt("What overflows?", ["stack", "heap", "cache", "bus"])
    assert "A. stack" in p and "D. bus" in p and p.rstrip().endswith("Answer:")
    assert ev.pick_letter({"A": -2.0, "B": -0.1, "C": -3.0, "D": -5.0}) == "B"


def test_degeneracy_flags():
    assert ev.degeneracy_flags("")["empty"] is True
    assert ev.degeneracy_flags("")["degenerate"] is True
    rep = ev.degeneracy_flags("na " * 50)
    assert rep["degenerate"] is True and rep["distinct_ratio"] < 0.35
    good = ev.degeneracy_flags("A buffer overflow overwrites the saved return address to hijack "
                               "control flow by redirecting execution to attacker-controlled shellcode.")
    assert good["degenerate"] is False


def test_run_mbpp_case():
    code = "def add(a, b):\n    return a + b\n"
    assert ev.run_mbpp_case(code, ["assert add(2, 3) == 5"]) is True
    assert ev.run_mbpp_case(code, ["assert add(2, 3) == 6"]) is False
    assert ev.run_mbpp_case("def f():\n    return 1/0\n", ["assert f() == 1"]) is False


def test_run_mbpp_case_times_out():
    # infinite loop must be killed by the alarm, not hang the suite
    assert ev.run_mbpp_case("def f():\n    while True:\n        pass\n", ["assert f() == 1"], timeout_s=1) is False


def test_extract_code():
    assert ev.extract_code("```python\nx=1\n```") == "x=1\n"
    assert ev.extract_code("no fence here") == "no fence here"


# --- unlearn_rmu pure helper ---

def test_load_texts_filters_buckets_and_minchars(tmp_path):
    pq = tmp_path / "u.parquet"
    pl.DataFrame({
        "bucket": ["offense", "offense", "defense", "dual"],
        "n_chars": [1000, 50, 1000, 1000],
        "text": ["off-long", "off-short", "def-long", "dual-long"],
    }).write_parquet(pq)
    got = ul.load_texts(str(pq), ["offense", "dual"], min_chars=200)
    assert set(got) == {"off-long", "dual-long"}     # short offense + defense bucket excluded
