# Running the GPU experiments on a RunPod pod

A persistent pod loads the model/datasets **once** — far cheaper than Flash serverless, which
cold-started (image pull + dep install + model download) on *every* point.

## 0. Pod  (mind the DISK — this bit us before)

- GPU: **A6000 / A40 (48 GB)** is enough and cheap — RMU loads model + frozen copy (~28 GB bf16);
  relearn is LoRA. A100 80 GB also fine. ~$0.5–0.8/hr (A6000).
- **Disk budget ~25–27 GB**: deps ~7 GB + zephyr weights ~14 GB + dataset cache ~3–4 GB + parquets.
  A default 20 GB **container disk overflows** — provision one of:
  - **container disk ≥ 50 GB**, or
  - a **persistent/network volume** (usually mounted at `/workspace`) and point caches there:
    ```bash
    export HF_HOME=/workspace/hf            # model + dataset cache on the roomy volume
    ```
    Put that line in `~/.bashrc` so every shell (and `pod_setup.sh`) uses it.
- We **never write model checkpoints** (RMU/relearn run in-memory) — so disk stays flat during runs;
  the only big consumers are the one-time weight + dataset downloads above.
- `df -h` before you start; `pod_setup.sh` prints free space and warns if it's tight.

## 1. Get the code + auth

```bash
# clone the private repo (use your GitHub PAT or gh auth)
git clone https://github.com/MaxKamachee/entanglement_problem.git
cd entanglement_problem

# put caches on the roomy volume FIRST (skip only if container disk ≥ 50 GB)
export HF_HOME=/workspace/hf && echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc

# HF auth — REQUIRED: zephyr + wmdp/mmlu are public, but the bio FORGET corpus is gated
hf auth login            # paste a token from https://huggingface.co/settings/tokens
# (also accept the meta-llama license only if you later run Llama; zephyr needs no license)
```

## 2. Setup (deps + build data)

```bash
bash scripts/pod_setup.sh
# builds data/wmdp_bio_units.parquet, data/wmdp_cyber_units.parquet, data/wikitext_units.parquet
```

## 3a. (optional) Reproduce the unlearning-tax sweep

Already run ($ on Flash); re-run on the pod only if you want to regenerate. Resumable.
```bash
python scripts/run_entanglement_sweep.py \
  --model HuggingFaceH4/zephyr-7b-beta --out-dir runs/entanglement \
  --coeffs 0 2 4 6.5 10 20
python scripts/diag_entanglement_tax.py     # -> reports/entanglement_unlearning_tax.md (run anywhere)
```

## 3b. Relearning experiment (the new Phase-D run)

2 domains × 2 relearn corpora = 4 conditions. Each: RMU-unlearn (wikitext retain, c=6.5) then
LoRA-finetune, tracking offense + same-domain-neighbor recovery. ~1 hr total on one A6000.
```bash
python scripts/run_relearn_sweep.py \
  --model HuggingFaceH4/zephyr-7b-beta --out-dir runs/relearn \
  --coeff 6.5 --relearn-steps 200 --eval-every 50
```
Each `runs/relearn/<cond>.json` has a `series` of {relearn_steps, offense_mcq, neighbor_mmlu}.
- `*_relforget` = adversarial robustness (paper Fig 15): re-teach the forbidden corpus.
- `*_relretain` = entanglement probe: does offense revive from legitimate same-domain text alone?

## 3c. Phase-D: tamper-durability eval (RMU + circuit breakers) — STAGED, inspect between stages

The paper's Section-4 run. Run it in stages on a base model so a bad setting doesn't waste the
whole session. **Accept the Llama license on HF first.**

**Stage 1 — eval gate: pick the model (base vs instruct).** Base avoids refusal but may be weak at
0-shot MCQ; instruct is refusal-robust for MCQ. Measure both, use whichever scores sane + has headroom.
```bash
for M in meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct; do
  python scripts/eval_suite.py --model $M --domain cyber --tag gate_$(basename $M) \
    --n-mcq 300 --n-neighbor 100 --n-mmlu 300 --n-mbpp 0 --n-saq 0 --n-degen 0 \
    --out runs/gate/$(basename $M)_cyber.json
done
# inspect: WMDP-cyber should be ~0.4, MMLU ~0.6; if a model reads ~chance, don't use it.
```

**Stage 2 — RMU coeff calibration on the chosen model `$M`** (hyperparams are model-specific; don't
borrow Zephyr's blindly). Find a coeff that removes offense while staying coherent:
```bash
python scripts/run_entanglement_sweep.py --model $M --arms cyber_wikitext bio_wikitext \
  --coeffs 0 3 6.5 13 --out-dir runs/calib_llama
# pick the coeff with a big offense drop and clean degeneracy guard -> $C
```

**Stage 3 — durability sweep at `$C`, both methods, 6 conditions** (forget / same-domain retain /
other-domain CONTROL × bio,cyber):
```bash
python scripts/run_relearn_sweep.py --model $M --method rmu --coeff $C \
  --out-dir runs/relearn_llama --relearn-steps 300 --eval-every 50
python scripts/run_relearn_sweep.py --model $M --method circuit_breakers \
  --out-dir runs/relearn_llama --relearn-steps 300 --eval-every 50
```
Outputs: `runs/relearn_llama/{rmu,cb}_<cond>.json`. (Gradient metric / chem deferred — see notes.)

## 4. Pull results back + analyze locally

```bash
# from your laptop (runpodctl gives a one-time code pair):
runpodctl send runs/relearn/*.json      # on the pod
runpodctl receive <code>                # on your laptop, into runs/relearn/
```
Then locally (no GPU): `python scripts/diag_relearn.py` → `reports/relearn_recovery.md` (+ figure);
`python scripts/diag_entanglement_tax.py` for the sweep.

## 5. STOP THE POD

```bash
# Terminate the pod in the RunPod console (or `runpodctl stop pod <id>`) when done —
# a running pod bills continuously, unlike serverless.
```
