#!/usr/bin/env python
"""Drive the relearning experiment (GPU, pod): 2 domains x 2 relearn corpora.

Each condition: RMU-unlearn (wikitext retain, canonical coeff) then LoRA-finetune on the relearn
corpus, evaluating offense + same-domain neighbor along the way. Fresh subprocess per condition
(clean GPU). Resumable (skips existing JSON).

  relearn=forget  -> adversarial robustness (paper Fig 15): re-teach the forbidden corpus.
  relearn=retain  -> entanglement probe: does offense revive from legitimate same-domain text alone?

Usage (pod):
  python scripts/run_relearn_sweep.py --out-dir runs/relearn --coeff 6.5 --relearn-steps 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# unlearn arm is wikitext (the arm that actually removes offense); relearn corpus varies.
CONDITIONS = {
    # relearn on the SAME domain's forget corpus -> adversarial re-teach (upper bound)
    "bio_relforget": {"domain": "bio", "relearn_parquet": "data/wmdp_bio_units.parquet",
                      "relearn_buckets": ["forget"]},
    "cyber_relforget": {"domain": "cyber", "relearn_parquet": "data/wmdp_cyber_units.parquet",
                        "relearn_buckets": ["forget"]},
    # relearn on the SAME domain's legit retain -> entanglement test
    "bio_relretain": {"domain": "bio", "relearn_parquet": "data/wmdp_bio_units.parquet",
                      "relearn_buckets": ["retain"]},
    "cyber_relretain": {"domain": "cyber", "relearn_parquet": "data/wmdp_cyber_units.parquet",
                        "relearn_buckets": ["retain"]},
    # relearn on the OTHER domain's legit retain -> CONTROL (separates entanglement from
    # generic RMU fragility: if offense revives as much from unrelated text, it's not entanglement)
    "bio_relcontrol": {"domain": "bio", "relearn_parquet": "data/wmdp_cyber_units.parquet",
                       "relearn_buckets": ["retain"]},
    "cyber_relcontrol": {"domain": "cyber", "relearn_parquet": "data/wmdp_bio_units.parquet",
                         "relearn_buckets": ["retain"]},
}
FORGET = {"bio": "data/wmdp_bio_units.parquet", "cyber": "data/wmdp_cyber_units.parquet"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--method", choices=["rmu", "circuit_breakers"], default="rmu")
    p.add_argument("--coeff", type=float, default=6.5)
    p.add_argument("--relearn-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--extra", default="", help="extra args passed to run_relearn.py")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import shlex
    extra = shlex.split(args.extra)

    mtag = "rmu" if args.method == "rmu" else "cb"
    for name in args.conditions:
        c = CONDITIONS[name]
        dom = c["domain"]
        eval_json = out / f"{mtag}_{name}.json"
        if eval_json.exists():
            print(f"skip {mtag}_{name} (already done)", flush=True)
            continue
        cmd = [sys.executable, str(SCRIPTS / "run_relearn.py"),
               "--model", args.model, "--domain", dom, "--method", args.method, "--coeff", str(args.coeff),
               "--forget-parquet", FORGET[dom], "--forget-buckets", "forget",
               "--retain-parquet", "data/wikitext_units.parquet", "--retain-buckets", "retain",
               "--relearn-parquet", c["relearn_parquet"], "--relearn-buckets", *c["relearn_buckets"],
               "--relearn-steps", str(args.relearn_steps), "--eval-every", str(args.eval_every),
               "--out", str(eval_json), *extra]
        print("+", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] {name} failed ({e}); continuing", flush=True)
    print(f"relearn sweep complete -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
