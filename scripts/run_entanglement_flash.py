#!/usr/bin/env python
"""Entanglement unlearning-tax sweep via RunPod Flash (serverless A100).

Same arms/eval as scripts/run_entanglement_sweep.py, but each (arm, coeff) point runs as a
Flash @Endpoint call on a serverless A100 instead of a pod subprocess — so the whole sweep is
driven from here, like scripts/extract_hidden_states_flash.py.

Single source of truth: the endpoint does NOT re-implement RMU/eval. It receives the SOURCE of
scripts/unlearn_rmu.py and scripts/eval_suite.py (both self-contained, torch-only), execs them on
the worker, and calls `run_rmu` (iff coeff>0) then `run_evals`. The forget/retain corpora are
sampled + truncated LOCALLY and sent gzip+b64 (<10MB); eval datasets (WMDP/MMLU/MBPP) are public
and downloaded on the worker.

Per point: load model fresh (+frozen iff coeff>0) -> optional RMU -> domain-parameterized eval
(offense WMDP MCQ, same-domain MMLU neighbors, general MMLU, MBPP, degeneracy guard) -> return the
scores dict. Loading fresh each point keeps RMU's weight mutation from leaking across points.

Usage (RUNPOD_API_KEY via `flash login`, then `flash dev` running in another shell):
  python scripts/run_entanglement_flash.py --model HuggingFaceH4/zephyr-7b-beta \
    --out-dir runs/entanglement --coeffs 0 2 4 6.5 10 20
Validation only (Phase A2 — base + one RMU point):
  python scripts/run_entanglement_flash.py --model HuggingFaceH4/zephyr-7b-beta \
    --out-dir runs/validate --arms cyber_wikitext --coeffs 0 6.5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json
import os
import sys
from pathlib import Path

from runpod_flash import Endpoint, GpuGroup

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import unlearn_rmu  # noqa: E402  (local import for load_texts; runs on the driver only)

ARMS = {
    "bio_wikitext": {"domain": "bio",
                     "forget_parquet": "data/wmdp_bio_units.parquet", "forget_buckets": ["forget"],
                     "retain_parquet": "data/wikitext_units.parquet", "retain_buckets": ["retain"]},
    "bio_substrate": {"domain": "bio",
                      "forget_parquet": "data/wmdp_bio_units.parquet", "forget_buckets": ["forget"],
                      "retain_parquet": "data/wmdp_bio_units.parquet", "retain_buckets": ["retain"]},
    "cyber_wikitext": {"domain": "cyber",
                       "forget_parquet": "data/wmdp_cyber_units.parquet", "forget_buckets": ["forget"],
                       "retain_parquet": "data/wikitext_units.parquet", "retain_buckets": ["retain"]},
    "cyber_substrate": {"domain": "cyber",
                        "forget_parquet": "data/wmdp_cyber_units.parquet", "forget_buckets": ["forget"],
                        "retain_parquet": "data/wmdp_cyber_units.parquet", "retain_buckets": ["retain"]},
}
TEXT_CAP = 3000   # chars per sampled doc (RMU truncates to max_tokens anyway)


def pack_texts(texts: list[str]) -> str:
    return base64.b64encode(gzip.compress(json.dumps(texts).encode())).decode("ascii")


def sample_texts(parquet: str, buckets: list[str], n: int, seed: int) -> list[str]:
    """Local: load + cap + truncate forget/retain docs for transport (deterministic)."""
    import polars as pl
    df = pl.read_parquet(ROOT / parquet)
    df = df.filter(pl.col("bucket").is_in(buckets) & (pl.col("n_chars") >= 200))
    if df.height > n:
        df = df.sample(n=n, seed=seed)
    return [t[:TEXT_CAP] for t in df["text"].to_list()]


# --------------------------------------------------------------------------------------------
# Remote GPU function — runs on a RunPod A100. Ships unlearn_rmu + eval_suite source and execs.
# --------------------------------------------------------------------------------------------
@Endpoint(
    name="entanglement-rmu-eval",
    gpu=GpuGroup.AMPERE_80,
    workers=(1, 1),
    idle_timeout=300,
    dependencies=["torch", "transformers>=4.43", "datasets", "polars", "numpy", "accelerate"],
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
         "HF_TOKEN": os.environ.get("HF_TOKEN", "")},   # rate-limit safety on dataset/model pulls
    execution_timeout_ms=0,
)
async def rmu_eval_point(cfg: dict) -> dict:
    import base64 as _b64
    import gzip as _gz
    import json as _json
    import types as _types
    from pathlib import Path as _Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # exec the shipped modules (self-contained, torch-only) -> run_rmu, run_evals
    ns_rmu: dict = {}
    exec(_b64.b64decode(cfg["rmu_src"]).decode(), ns_rmu)
    ns_eval: dict = {}
    exec(_b64.b64decode(cfg["eval_src"]).decode(), ns_eval)
    run_rmu = ns_rmu["run_rmu"]
    run_evals = ns_eval["run_evals"]

    def unpack(b: str) -> list:
        return _json.loads(_gz.decompress(_b64.b64decode(b)).decode())

    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(trainable: bool):
        m = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16,
                                                 device_map="auto")
        m.train(trainable)
        m.requires_grad_(False)
        return m

    model = load(cfg["coeff"] > 0)
    if cfg["coeff"] > 0:
        frozen = load(False)
        forget, retain = unpack(cfg["forget"]), unpack(cfg["retain"])
        run_rmu(model, frozen, tok, forget, retain, layer=cfg["layer"],
                update_layers=cfg["update_layers"], coeff=cfg["coeff"], alpha=cfg["alpha"],
                steps=cfg["steps"], lr=cfg["lr"], batch_size=cfg["batch_size"],
                max_tokens=cfg["max_tokens"])
        del frozen
        torch.cuda.empty_cache()
    model.eval()

    args = _types.SimpleNamespace(
        tag=cfg["tag"], model=cfg["model"], seed=cfg["seed"], domain=cfg["domain"],
        saq="__none__.parquet", n_mcq=cfg["n_mcq"], n_neighbor=cfg["n_neighbor"],
        n_mmlu=cfg["n_mmlu"], n_mbpp=cfg["n_mbpp"], n_saq=0, n_degen=cfg["n_degen"],
        out=str(_Path("/tmp") / f"{cfg['tag']}.json"))
    result = run_evals(model, tok, args)
    del model
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------------------------
# Local driver
# --------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--coeffs", type=float, nargs="+", default=[0, 2, 4, 6.5, 10, 20])
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--update-layers", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-forget", type=int, default=1000, help="forget docs sampled/sent per point")
    p.add_argument("--n-retain", type=int, default=1000, help="retain docs sampled/sent per point")
    p.add_argument("--n-mcq", type=int, default=400)
    p.add_argument("--n-neighbor", type=int, default=150)
    p.add_argument("--n-mmlu", type=int, default=400)
    p.add_argument("--n-mbpp", type=int, default=40)
    p.add_argument("--n-degen", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


async def run_sweep(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rmu_src = base64.b64encode((SCRIPTS / "unlearn_rmu.py").read_bytes()).decode()
    eval_src = base64.b64encode((SCRIPTS / "eval_suite.py").read_bytes()).decode()

    def base_cfg(tag, domain, coeff):
        return {"tag": tag, "model": args.model, "domain": domain, "coeff": float(coeff),
                "layer": args.layer, "update_layers": args.update_layers, "alpha": args.alpha,
                "steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
                "max_tokens": args.max_tokens, "seed": args.seed, "n_mcq": args.n_mcq,
                "n_neighbor": args.n_neighbor, "n_mmlu": args.n_mmlu, "n_mbpp": args.n_mbpp,
                "n_degen": args.n_degen, "rmu_src": rmu_src, "eval_src": eval_src,
                "forget": "", "retain": ""}

    async def point(tag, domain, coeff, arm_cfg) -> None:
        eval_json = out / f"{tag}.json"
        if eval_json.exists():
            print(f"skip {tag} (already done)", flush=True)
            return
        cfg = base_cfg(tag, domain, coeff)
        if coeff > 0:
            cfg["forget"] = pack_texts(sample_texts(arm_cfg["forget_parquet"], arm_cfg["forget_buckets"],
                                                    args.n_forget, args.seed))
            cfg["retain"] = pack_texts(sample_texts(arm_cfg["retain_parquet"], arm_cfg["retain_buckets"],
                                                    args.n_retain, args.seed))
        print(f"-> {tag} (coeff={coeff}, domain={domain}) on A100...", flush=True)
        try:
            result = await rmu_eval_point(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] {tag} failed: {type(e).__name__}: {e}", flush=True)
            return
        eval_json.write_text(json.dumps(result, indent=2))
        print(f"wrote {eval_json}  scores={result.get('scores')}", flush=True)

    # base (coeff 0): depends only on eval domain -> one per domain
    domains = sorted({ARMS[a]["domain"] for a in args.arms})
    if 0 in args.coeffs:
        for domain in domains:
            await point(f"base_{domain}", domain, 0, None)
    for arm in args.arms:
        cfg = ARMS[arm]
        for coeff in args.coeffs:
            if coeff == 0:
                continue
            await point(f"{arm}_c{coeff}", cfg["domain"], coeff, cfg)
    print(f"sweep complete -> {out}", flush=True)


def main(argv=None) -> int:
    asyncio.run(run_sweep(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
