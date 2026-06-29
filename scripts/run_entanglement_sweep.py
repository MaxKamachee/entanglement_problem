#!/usr/bin/env python
"""Drive the entanglement unlearning-tax sweep (GPU, on a pod / Flash A100).

Measures how much LEGITIMATE same-domain ability you lose to remove a given amount of
OFFENSIVE ability, per domain, and whether a targeted retain set reduces that tax. Tests
the field's clean-line assumption between harmful and benign knowledge.

For each arm x steering-coeff: coeff 0 = evaluate the BASE model (deduped per domain); coeff
> 0 = RMU (run_point.py, in-memory, no checkpoint) then the symmetric eval suite. Each point
is a fresh subprocess so GPU memory is released between runs. Resumable (skips existing JSON).

Arms = domain x retain-set (the mechanism knob):
  bio_wikitext     forget=wmdp-bio   retain=wikitext   (blunt / WMDP default)
  bio_substrate    forget=wmdp-bio   retain=wmdp-bio   (targeted, same-domain substrate)
  cyber_wikitext   forget=wmdp-cyber retain=wikitext
  cyber_substrate  forget=wmdp-cyber retain=wmdp-cyber

Eval per point (eval_suite, --domain): offense WMDP MCQ (down=unlearned), same-domain MMLU
neighbor (up=kept), general MMLU canary, MBPP (executed), degeneracy/coherence guard.

Usage (pod / Flash):
  python scripts/run_entanglement_sweep.py --out-dir runs/entanglement --coeffs 0 20 100 300 1000
CPU smoke (tiny model, tiny evals):
  python scripts/run_entanglement_sweep.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --out-dir /tmp/ent --coeffs 0 20 --steps 3 \
    --eval-args "--n-mcq 4 --n-neighbor 2 --n-mmlu 4 --n-mbpp 0 --n-saq 0 --n-degen 1"
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

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


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--method", choices=["rmu", "circuit_breakers"], default="rmu")
    p.add_argument("--strengths", "--coeffs", type=float, nargs="+", dest="strengths",
                   default=[0, 2, 4, 6.5, 10, 20],
                   help="unlearning strength axis: RMU coeff or CB alpha (0 = base model)")
    p.add_argument("--layer", type=int, default=7)        # RMU
    p.add_argument("--alpha", type=float, default=1200.0)  # RMU retain weight
    p.add_argument("--steps", type=int, default=500)       # RMU
    p.add_argument("--cb-steps", type=int, default=300)    # CB
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-args", default="", help="extra args passed through to run_point/eval_suite")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.eval_args)
    mtag = "rmu" if args.method == "rmu" else "cb"

    def point(tag: str, strength: float, domain: str, cfg: dict | None) -> None:
        eval_json = out / f"{tag}.json"
        if eval_json.exists():
            print(f"skip {tag} (already done)", flush=True)        # resume
            return
        cmd = [sys.executable, str(SCRIPTS / "run_point.py"),
               "--model", args.model, "--tag", tag, "--domain", domain, "--method", args.method,
               "--seed", str(args.seed), "--out", str(eval_json), *extra]
        if strength != 0:
            cmd += ["--forget-parquet", cfg["forget_parquet"], "--forget-buckets", *cfg["forget_buckets"],
                    "--retain-parquet", cfg["retain_parquet"], "--retain-buckets", *cfg["retain_buckets"]]
            if args.method == "rmu":
                cmd += ["--coeff", str(strength), "--layer", str(args.layer),
                        "--alpha", str(args.alpha), "--steps", str(args.steps)]
            else:
                cmd += ["--cb-alpha", str(strength), "--cb-steps", str(args.cb_steps)]
        try:
            run(cmd)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] {tag} failed ({e}); continuing", flush=True)

    # base model (strength 0) is method- AND retain-independent -> run once per domain, shared
    domains = sorted({ARMS[a]["domain"] for a in args.arms})
    if 0 in args.strengths:
        for domain in domains:
            point(f"base_{domain}", 0, domain, None)

    for arm in args.arms:
        cfg = ARMS[arm]
        for s in args.strengths:
            if s == 0:
                continue   # handled by the per-domain base above
            point(f"{mtag}_{arm}_s{s:g}", s, cfg["domain"], cfg)

    print(f"sweep complete -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
