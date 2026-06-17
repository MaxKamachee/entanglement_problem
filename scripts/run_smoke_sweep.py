#!/usr/bin/env python
"""Drive the WMDP-vs-ours RMU smoke sweep (GPU, on a pod).

For each arm x steering-coeff: coeff 0 = evaluate the BASE model (no unlearning); coeff > 0 =
run scripts/unlearn_rmu.py then scripts/eval_suite.py on the resulting checkpoint. Each step is
a fresh subprocess so GPU memory is released between runs. Collects the per-run eval JSONs into
an output dir; SAQ answers are graded off-GPU afterward (entanglement.saq.grade_saqs_batch).

Arms (only the partition/corpus varies; eval + model fixed):
  wmdp  — forget=wmdp_cyber_units[forget], retain=wmdp_cyber_units[retain]   (standard reference)
  ours  — forget=analysis_units_v1[offense], retain=analysis_units_v1[dual,defense]  (three-tier)

Usage (pod):
  python scripts/run_smoke_sweep.py --out-dir runs/smoke --coeffs 0 20 100 300 1000
CPU smoke (tiny model, tiny evals):
  python scripts/run_smoke_sweep.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
    --out-dir /tmp/sweep --coeffs 0 20 --steps 3 --eval-args "--n-mcq 4 --n-mmlu 4 --n-mbpp 2 --n-saq 4"
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

ARMS = {
    "wmdp": {"forget_parquet": "data/wmdp_cyber_units.parquet", "forget_buckets": ["forget"],
             "retain_parquet": "data/wmdp_cyber_units.parquet", "retain_buckets": ["retain"]},
    "ours": {"forget_parquet": "data/analysis_units_v1.parquet", "forget_buckets": ["offense"],
             "retain_parquet": "data/analysis_units_v1.parquet", "retain_buckets": ["dual", "defense"]},
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--coeffs", type=float, nargs="+", default=[0, 20, 100, 300, 1000])
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--alpha", type=float, default=1200.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-args", default="", help="extra args passed through to eval_suite.py")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.eval_args)

    for arm in args.arms:
        cfg = ARMS[arm]
        for coeff in args.coeffs:
            tag = f"{arm}_c{int(coeff)}"
            eval_json = out / f"{tag}.json"
            if eval_json.exists():
                print(f"skip {tag} (already done)", flush=True)   # resume: keep prior points
                continue
            ckpt = out / "ckpts" / tag
            try:
                if coeff == 0:
                    model_path = args.model        # base model = the c=0 baseline point
                else:
                    model_path = str(ckpt)
                    run([sys.executable, str(SCRIPTS / "unlearn_rmu.py"),
                         "--model", args.model,
                         "--forget-parquet", cfg["forget_parquet"], "--forget-buckets", *cfg["forget_buckets"],
                         "--retain-parquet", cfg["retain_parquet"], "--retain-buckets", *cfg["retain_buckets"],
                         "--layer", str(args.layer), "--coeff", str(coeff), "--alpha", str(args.alpha),
                         "--steps", str(args.steps), "--seed", str(args.seed), "--out", model_path])
                run([sys.executable, str(SCRIPTS / "eval_suite.py"),
                     "--model", model_path, "--tag", tag, "--seed", str(args.seed),
                     "--out", str(eval_json), *extra])
            except subprocess.CalledProcessError as e:
                print(f"[WARN] {tag} failed ({e}); continuing", flush=True)
            finally:
                if coeff != 0:                      # delete the 16GB ckpt right after eval (disk bound)
                    shutil.rmtree(ckpt, ignore_errors=True)
    print(f"sweep complete -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
