"""Load methodology parameters from configs/.

Methodology choices (source buckets to keep, the WMDP threshold, the vendor host
list, scraper politeness) live in ``configs/*.yaml`` rather than in source so a
reviewer can audit and revise them without touching code. Modules call these
loaders at import time to populate their module-level constants; the pure
``build_*`` functions still accept explicit values so tests stay offline and
config-independent.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


@functools.lru_cache(maxsize=None)
def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_corpus_config() -> dict[str, Any]:
    """Return the parsed configs/corpus.yaml mapping."""
    return _load_yaml(CONFIG_DIR / "corpus.yaml")


def load_vendor_hosts() -> list[str]:
    """Return the VENDOR host substrings from configs/vendor_hosts.yaml."""
    return list(_load_yaml(CONFIG_DIR / "vendor_hosts.yaml").get("vendor_hosts", []))


def defense_keep_buckets() -> frozenset[str]:
    """D3FEND source buckets to keep for the defensive external corpus."""
    return frozenset(load_corpus_config().get("defense_keep_buckets", []))


def defense_keep_other_curated() -> bool:
    """Whether to fold the curated OTHER bucket into the defensive external set."""
    return bool(load_corpus_config().get("defense_keep_other_curated", False))


def defense_other_drop_hosts() -> list[str]:
    """Noise hosts (substring match) to drop from the curated OTHER bucket."""
    return list(load_corpus_config().get("defense_other_drop_hosts", []))


def wmdp_keep_threshold() -> int:
    """Minimum WMDP operational-capability score to keep an offensive doc."""
    return int(load_corpus_config().get("wmdp_keep_threshold", 7))


def wmdp_rater_model() -> str:
    """Anthropic model id used as the WMDP rater."""
    return str(load_corpus_config().get("wmdp_rater_model", "claude-sonnet-4-6"))


def prune_framework_metadata() -> bool:
    """Whether to drop framework-internal cataloging layers (offense/procedure, defense/prose)
    when assembling analysis units — keeping offense/defense external-reference-only + symmetric."""
    return bool(load_corpus_config().get("prune_framework_metadata", True))


def github_supplements() -> list[dict]:
    """GitHub blue-team supplement repos (owner/repo/ref/license) for the defense bucket."""
    return list(load_corpus_config().get("github_supplements", []))


def cleanup_enabled() -> bool:
    """Whether to apply the post-prune offense/dual cleanup filters when assembling analysis units."""
    return bool(load_corpus_config().get("cleanup_enabled", True))


def cleanup_params() -> dict[str, Any]:
    """Thresholds for the cleanup filters (see configs/corpus.yaml `cleanup:`)."""
    c = load_corpus_config().get("cleanup", {}) or {}
    return {
        "offense_tiny_chars": int(c.get("offense_tiny_chars", 250)),
        "offense_neardup_jaccard": float(c.get("offense_neardup_jaccard", 0.85)),
        "dual_max_nonprose_ratio": float(c.get("dual_max_nonprose_ratio", 0.5)),
        "dual_max_longtoken_ratio": float(c.get("dual_max_longtoken_ratio", 0.3)),
        "dual_short_fragment_chars": int(c.get("dual_short_fragment_chars", 500)),
    }


def stub_min_chars() -> int:
    """Post-extraction stub threshold (chars); shorter documents are dropped."""
    return int(load_corpus_config().get("stub_min_chars", 2000))


def mcq_config() -> dict[str, Any]:
    """Stage-4 MCQ generation parameters (see configs/corpus.yaml `mcq:`)."""
    c = load_corpus_config().get("mcq", {}) or {}
    return {
        "gen_model": str(c.get("gen_model", "claude-sonnet-4-6")),
        "critic_model": str(c.get("critic_model", "claude-sonnet-4-6")),
        "per_topic_n": int(c.get("per_topic_n", 25)),
        "source_oversample": float(c.get("source_oversample", 2.0)),
        "seed": int(c.get("seed", 0)),
        "min_source_chars": int(c.get("min_source_chars", 800)),
        "smoke_n": int(c.get("smoke_n", 3)),
        "section_target": int(c.get("section_target", 4000)),
        "section_hard_max": int(c.get("section_hard_max", 6000)),
        "max_sections_per_doc": int(c.get("max_sections_per_doc", 3)),
        "defend_exclude_source_categories": list(
            c.get("defend_exclude_source_categories", ["NIST_SP_800", "US_GOV_CISA"])),
        "contamination_ngram": int(c.get("contamination_ngram", 8)),
        "max_cost_usd": float(c.get("max_cost_usd", 50.0)),
    }


def saq_config() -> dict[str, Any]:
    """SAQ smoke-test eval parameters (see configs/corpus.yaml `saq:`)."""
    c = load_corpus_config().get("saq", {}) or {}
    return {
        "gen_model": str(c.get("gen_model", "claude-sonnet-4-6")),
        "judge_model": str(c.get("judge_model", "claude-haiku-4-5-20251001")),
        "per_cell_n": int(c.get("per_cell_n", 20)),
        "source_oversample": float(c.get("source_oversample", 1.5)),
    }


def rmu_config() -> dict[str, Any]:
    """RMU unlearning parameters for the smoke sweep (see configs/corpus.yaml `rmu:`)."""
    c = load_corpus_config().get("rmu", {}) or {}
    return {
        "model": str(c.get("model", "meta-llama/Llama-3.1-8B-Instruct")),
        "layer": int(c.get("layer", 7)),
        "coeffs": list(c.get("coeffs", [0, 5, 20, 100, 300, 1000])),
        "alpha": float(c.get("alpha", 1200.0)),
        "steps": int(c.get("steps", 500)),
        "seed": int(c.get("seed", 0)),
        "max_tokens": int(c.get("max_tokens", 512)),
    }


def scrape_user_agent() -> str:
    return str(load_corpus_config().get("scrape_user_agent", "entanglement-research-corpus/0.1"))


def scrape_per_host_interval_sec() -> float:
    return float(load_corpus_config().get("scrape_per_host_interval_sec", 1.0))
