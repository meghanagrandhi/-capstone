# Next semester: forecaster plan and who does what

Repo: https://github.com/meghanagrandhi/-capstone (private). Everyone: `git clone`, then `git pull` before starting any work.
Setup (GPU machines): Python 3.13 venv, then `pip install -r requirements.txt` (pinned versions, CUDA 12.8 torch). CPU-only
machine: same, but install the CPU build of torch instead (only needed to run tests; analysis needs only pandas/numpy/scipy/sklearn).

## Where we are (end of pilot, Nov 2026)
- 36 training runs: 3 models (Qwen2-VL-2B, SmolVLM-2.2B, InternVL3-2B) x 2 mixes (text10, text90) x 2 mix types (matched,
  balanced) x 3 seeds. Table: `results/tier3/baselines/run_table.csv`.
- Simple baselines (`scripts/baselines.py`): error predicting the final drop in G image benefit, on a model never seen:
  nothing changes 2.2 pts, static 2.0, training loss 4.5, G measured at 40% of training 0.9.
- Forecaster v0 (`scripts/forecaster_v0.py`, ridge on 4 probe signals): 1.6 pts using only 5% of training. Weak, a first sketch.
- Missing: enough varied runs (~200), a grid script, faster evaluation, more features. That is this plan.

## Team decisions (fixed now, before any new runs; do not change after seeing results)
1. **Target:** `dG_final` = (G real - G blank accuracy at the last step) - (same at step 0), in points. Secondary yes/no target:
   "big drop" = dG_final <= -3 pts (seed noise is 1-2 pts).
2. **Models:** keep all three, including InternVL (its balanced runs show no drop: the forecaster must learn "no drop" too).
3. **Metrics:**
   - Main: MAE (pts) with leave-one-model-out, using only signals up to 20% of training. Success = beats "nothing changes"
     (2.2) and "static" (2.0). Stretch = close to "G at 40%" (0.9).
   - Secondary: pair accuracy (does it rank text90 vs text10 right?) > 75%; Spearman; for the yes/no target, recall and
     precision (AUROC allowed here as a secondary number).
   - Practical: GPU minutes saved by deciding at 20% instead of finishing the run.
4. **Locked test set:** all runs with text share 50% are the final test set. Nobody fits or tunes anything on them; they
   are scored once, at the end.

## Person 1: CPU-only machine (code + analysis). Do this FIRST (week 1), GPU people wait for it.
1. Grid mixes: extend `scripts/run_tier2.py` (new stage, e.g. `--stage mix_grid --shares 10 30 50 70 90`) to build mixes at any
   text share from `results/tier2/full/pool_labeled_matched.csv`, same rules as the balanced mixes (900 items, VSR 50% yes,
   same per-subset counts, overlap guard). Output `results/tier2/full/mixture_grid_text{NN}.csv`.
   Run `python tests/test_overlap.py` after.
2. Training options: `scripts/run_tier3_train.py` only accepts fixed `--mixture` names and reads lr / LoRA rank from
   `configs/config.yaml`. Add `--mixture-file`, `--lr`, `--lora-r` (alpha = 2 x r) and put them in the run name and `run_meta.json`.
3. Fast eval: add a `--fast` option to `scripts/run_tier3_eval_v2.py`: only the last checkpoint, only the G set, only real +
   blank images (skip A and swap). The early signals come from the in-training probe (`probe_log.csv`), which is cheap.
4. Grid queue: a script like `scripts/run_model_queue.sh` that reads a run list (CSV: model, share, lr, rank, steps, seed),
   skips finished runs, keeps the heat guard. Suggested grid per model: shares 10/30/50/70/90 x lr {1e-4, 2e-4} x
   rank {8, 16} x seeds {0, 1} = 40 runs; plus a few 60-step runs.
5. More features: dataset stats per mix (from the mixture CSV), gradient-norm and loss-gap features from `train_log.csv`.
6. Forecaster: extend `scripts/forecaster_v0.py` (ridge now, LightGBM once there are 150+ runs), score it exactly like
   `scripts/baselines.py`, never on the locked share-50 runs until the end.
7. Every week: merge the zips from Persons 2 and 3 into `results/`, rerun `scripts.baselines`, commit and push.
Files: `scripts/run_tier2.py`, `src/pool.py`, `src/overlap.py`, `scripts/run_tier3_train.py`, `src/train.py`,
`scripts/run_tier3_eval_v2.py`, `scripts/run_model_queue.sh`, `scripts/baselines.py`, `scripts/forecaster_v0.py`,
`configs/config.yaml`, `results/tier2/full/*`, `results/tier3/*/train_log.csv` and `probe_log.csv`.

## Person 2: RTX 5070 (~120 runs, Qwen + InternVL)
1. `git pull` after Person 1 pushes the grid code. Smoke test first: 1 short Qwen run with `--smoke`, then 1 full run + fast
   eval; send the log to Person 1 if anything fails.
2. Run the Qwen and InternVL grid with the grid queue (detached, as before). About 25 min per run, ~50 GPU hours = 4-5 nights.
3. After each batch: zip `results/tier3/<new runs>/` (without `adapters/`), the eval output folder and `results/overnight.log`,
   send to Person 1. Delete adapters you do not need to save disk (the queue already keeps only evaluated ones).
Files: everything from git; model weights download on first run (Qwen ~4.5 GB, InternVL ~4 GB; already on this machine).

## Person 3: RTX 4060 laptop (~80 runs, SmolVLM + extra Qwen seeds)
1. Re-download Qwen weights (~4.5 GB, deleted on 2026-11-01); the first run does it automatically (pinned revision in config).
2. Same smoke test as Person 2, then run the SmolVLM grid + extra Qwen seeds. About 40 min per run, ~50 GPU hours, overnight.
3. Laptop rules: keep the heat guard on, never run two GPU jobs at once, check free disk space (needs > 3 GB, adapters are big).
4. Same zip as Person 2 after each batch.

## Timeline
- Week 1: Person 1 writes the grid code; all three agree on anything not decided above.
- Weeks 2-3: Persons 2 and 3 run the grid (~200 runs); Person 1 merges weekly and adds features.
- Week 4 on: forecaster built and scored by Person 1 (others help once GPU runs finish); locked share-50 test at the very end.

Everything is a feasibility study: report failures plainly, never claim a result is confirmed.
