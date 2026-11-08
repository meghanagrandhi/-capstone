# Next semester: forecaster plan and who does what

Repo: https://github.com/meghanagrandhi/-capstone (private). Everyone: `git clone`, then `git pull` before starting any work.
Setup (GPU machines): Python 3.13 venv, then `pip install -r requirements.txt` (pinned versions, CUDA 12.8 torch).
CPU-only machine: same, but the CPU build of torch is enough (analysis needs only pandas/numpy/scipy/sklearn).

## Where we are (Nov 2026)
- 36 pilot runs: 3 models (Qwen2-VL-2B, SmolVLM-2.2B, InternVL3-2B) x 2 mixes (text10, text90) x 2 mix types (matched,
  balanced) x 3 seeds. Table: `results/tier3/baselines/run_table.csv`.
- Baselines (`scripts/baselines.py`), error predicting the final drop in G image benefit on a model never seen:
  nothing changes 2.2 pts, static 2.0, training loss 4.5, G measured at 40% of training 0.9.
- Forecaster v0 (`scripts/forecaster_v0.py`, ridge): 1.6 pts using only 5% of training (4 probe signals). Weak sketch.
- **Grid code is DONE and tested** (2026-11-08): mixes at any share, `--lr` / `--lora-r` training options, fast eval,
  grid run list + queue, extra features. What is left is mainly GPU time.

## Team decisions (fixed now; do not change after seeing results)
1. **Target:** `dG_final` = (G real - G blank accuracy at the last step) - (same at step 0), in points.
   Secondary yes/no target: "big drop" = dG_final <= -3 pts (seed noise is 1-2 pts).
2. **Models:** keep all three, including InternVL (its balanced runs show no drop: the forecaster must learn "no drop" too).
3. **Metrics:** main = MAE (pts), leave-one-model-out, using only signals up to 20% of training; success = beats "nothing
   changes" (2.2) and "static" (2.0); stretch = close to "G at 40%" (0.9). Secondary: text90-vs-text10 pair accuracy > 75%,
   Spearman, recall/precision for "big drop" (AUROC allowed as a secondary number). Practical: GPU minutes saved.
4. **Locked test set:** every grid run with text share 50 (27 runs, column `locked_test` in `configs/grid_runs.csv`).
   `baselines.py` and `forecaster_v0.py` already drop them. Nobody looks at them until the very end.

## What the grid is (`configs/grid_runs.csv`, made by `python -m scripts.make_grid`)
Per model, 47 runs: text share {10,30,50,70,90} x lr {1e-4, 2e-4} x LoRA rank {8, 16} x seeds {0,1} (40) + seed 2 at pilot
settings (5) + half-length runs at share 10/90 (2). 3 models -> 141 runs (+ 36 pilot = 177).
Mixes: `results/tier2/full/mixture_grid_text{10,30,50,70,90}.csv` (900 items, VSR 50% "yes", 0 overlap with MMStar).
Every grid run is probed during training on the SAME 64 held-out items (`grid_probe_items.csv`; the pilot used only 16,
which is part of why v0's early signals are noisy). Fast eval scores A real + G real/blank at 0/40/100% of training.

## How to run (GPU machines)
```powershell
git pull
# 1) dry run: prints the jobs, uses no GPU
& "C:\Program Files\Git\bin\bash.exe" -c "DRY=1 bash scripts/run_grid_queue.sh smolvlm"
# 2) the real queue, detached (keeps running if the terminal closes); resumable: finished runs are skipped
Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList @("scripts/run_grid_queue.sh","qwen2vl","internvl")
```
Progress: `results/grid.log` (one `=== DONE <run>` line per finished run). Per-job logs: `results/pipeline_logs/grid_*`.
Heat guard: waits for GPU < 65 C between jobs, kills a job after 2 min at >= 90 C, stops after a second overheat.
Disk guard: stops if < 3 GB free. Only the final adapter (~70 MB) of each run is kept.
To send results: zip `results/tier3/eval_grid/`, every `results/tier3/*grid_text*/` folder WITHOUT `adapters/`, and `results/grid.log`.

## Person 1: CPU-only machine (analysis + forecaster)
1. Week 1: read the code (`scripts/run_tier2.py` stage mix_grid, `scripts/run_tier3_train.py`, `scripts/run_tier3_eval_v2.py --fast`,
   `scripts/run_grid_queue.sh`, `scripts/baselines.py`, `scripts/forecaster_v0.py`) and the pilot results in `README.md`.
2. Every week: unzip the GPU zips into `results/` (merge `eval_grid/checkpoint_eval_v2.csv` rows, or rerun the eval summary),
   then `python -m scripts.baselines` and `python -m scripts.forecaster_v0`; commit and push.
3. Forecaster: features already include probe signals, text share, mix mean gap, lr, rank, loss, loss gap, grad-norm mean
   and slope (feature set "all"). Next: more probe signals, LightGBM once there are 150+ runs, SHAP-style explanations.
   Pilot check (36 runs): extra training-dynamics features did NOT help on a new model yet (MAE 2.3-3.5 vs 1.6 for probe
   only) - expected with 36 runs; recheck on the grid.
4. Never fit, tune or plot the `locked_test` runs until the final evaluation.

## Person 2: RTX 5070 (94 runs: Qwen + InternVL)
1. `git pull`, dry run, then start `run_grid_queue.sh qwen2vl internvl`.
2. About 30 min per run -> ~47 GPU hours (5-6 nights). Zip results after every ~20 runs, send to Person 1.
3. Model weights are already on this machine.

## Person 3: RTX 4060 laptop (47 runs: SmolVLM)
1. FREE DISK SPACE FIRST: C: has only ~11 GB free (2026-11-08). The grid needs ~5 GB (47 final adapters + eval files);
   aim for 20 GB+ free. (Qwen weights, ~4.5 GB, were deleted on 2026-11-01; not needed for the SmolVLM grid.)
2. `git pull`, dry run, then start `run_grid_queue.sh smolvlm`.
3. About 60 min per run (train ~29 min, probe ~8 min, eval ~20 min) -> ~47 GPU hours, overnight. Never run two GPU jobs at once.
4. Zip results the same way, send to Person 1.

## Timeline
- Week 1: everyone reads this + the code; GPU machines do a dry run and start the queues.
- Weeks 1-3: ~141 grid runs (both GPUs in parallel, ~47 GPU hours each). Person 1 merges weekly.
- Week 4 on: forecaster built and scored by Person 1 (others help); locked share-50 test at the very end.

Everything is a feasibility study: report failures plainly, never claim a result is confirmed.
