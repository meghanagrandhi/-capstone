# Hidden visual-grounding regression pilot (feasibility only)

Status (2026-11-01): **Tiers 1-3 done on three models** (Qwen2-VL-2B, SmolVLM-2.2B, InternVL3-2B), 3 seeds each, plus paired-image
tests (MMVP, NaturalBench). No forecasting model, no AUROC, nothing here is confirmed evidence: the result is a small,
model-dependent signal (see "Results in one paragraph" at the bottom).

## Setup (Windows, native Python 3.13 + CUDA torch; WSL was not used)
```powershell
python -m venv --system-site-packages .venv      # inherits torch 2.11+cu128 and transformers 5.10.2
.\.venv\Scripts\python -m pip install peft bitsandbytes datasets accelerate pyarrow
```

## Layout
- `configs/config.yaml`: model id, max_pixels, n_items, seeds, paths (single source of config)
- `src/vlm.py`: `VLM` interface (`load`, `score_options`, `answer_nll`) + `Qwen2VL` (4-bit NF4, sdpa), `SmolVLM` and `InternVL` (bf16) wrappers;
  `select_model(cfg, "smolvlm")` swaps in the `model_smolvlm` config block (CLI: `--model smolvlm|internvl`, run names get a `smolvlm_` / `internvl_` prefix)
- `src/data.py`: MMStar loading and option parsing (3 question formats; fails loudly on anything else)
- `src/eval.py`: option-letter log-likelihood evaluator, per-item CSV cache (resumable)
- `scripts/run_tier1.py`: Tier 1 entry point
- `results/tier1/{smoke,full}/`: `per_item.csv`, `summary.json`, `results_table.csv`, `tier1_bar.png`

## Run
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\.venv\Scripts\python -m scripts.run_tier1 --smoke    # 10 items, 1 shuffle seed
.\.venv\Scripts\python -m scripts.run_tier1            # 500 items, seed 0 + shuffle seeds 1-5, real + blank
```
Reruns skip rows already in `per_item.csv` (key: id, seed, condition), so they are cheap and resumable.

## Notes
- MMStar options are embedded in the question string; items with other than 4 options (1 of 1500) are skipped so chance is a uniform 25%.
- Scoring: log-prob of the option letter A-D as the next token (one forward pass), not free generation.
- Seed 0 = canonical option order; seeds 1-5 shuffle item order and option order. Option log-probs are stored in original option order.
- Blank image = uniform gray (127,127,127) at the original image size.

## Tier 1 extras
```powershell
.\.venv\Scripts\python -m scripts.analyze_tier1        # Wilson CIs, McNemar, blind k-of-6 histogram, letter distributions
.\.venv\Scripts\python -m scripts.run_gpool [--smoke]  # G candidate pool: blind k-of-6 labels for the 999 MMStar items outside the 500-item sample
```
Model is pinned to revision `895c3a49bc3fa70a340399125c650a463535e71c` (configs/config.yaml).

## Tier 2 (training pool + mixtures)
```powershell
.\.venv\Scripts\python -m scripts.run_tier2 --stage pool [--smoke]   # stream 4 Cauldron subsets, overlap-filter vs all of MMStar
.\.venv\Scripts\python -m scripts.run_tier2 --stage label [--smoke]  # loss-gap labeling (~90 min for 3161 items; CPU-bound)
.\.venv\Scripts\python -m scripts.run_tier2 --stage mix [--smoke]    # quantile thresholds, histogram, 2 mixtures, hard overlap asserts
.\.venv\Scripts\python tests/test_overlap.py                          # planted-overlap test: the guard must raise
```
Outputs in `results/tier2/full/`: `pool_labeled.csv`, `mixture_text10.csv`, `mixture_text90.csv`, `gap_hist.png`, `summary.json`.
Known confound: the text-answerable / image-necessary classes are largely separated by source subset (see summary.json).

## Tier 3 (QLoRA mini-pilot) - done
Training: matched mixtures (`mixture_matched_text{10,90}.csv`, 250 items per Cauldron subset, so source subset is not a
confound), LoRA r=16 on language-model linear layers only, lr 2e-4, batch 8, 120 steps (Qwen) / 125 (SmolVLM, InternVL), seeds 0,1,2.
```powershell
.\.venv\Scripts\python -m scripts.run_tier3_train --mixture matched_text90 --seed 1 [--model smolvlm|internvl] [--smoke]
.\.venv\Scripts\python -m scripts.build_eval_sets --all               # A = 500 calibration items, G = 581 blind-failed items
.\.venv\Scripts\python -m scripts.run_tier3_eval_v2 --runs matched_text90_s1 [--model smolvlm|internvl] [--pcts 0 40 100]
.\.venv\Scripts\python -m scripts.run_paired_bench --runs matched_text10 matched_text90 ... [--model smolvlm|internvl]
.\.venv\Scripts\python -m scripts.analyze_tier3_v2 [--model smolvlm|internvl]   # bootstrap CIs, McNemar, seed signs, A split
.\.venv\Scripts\python -m scripts.analyze_paired_bench
.\.venv\Scripts\python -m scripts.analyze_yes_bias                     # yes/no balance of mixes, "Yes" rate, same-answer by question type
.\.venv\Scripts\python -m scripts.run_tier2 --stage mix_balanced        # follow-up: mixes with 50% "yes" in VSR (150 VSR items, 900 total)
#   then: bash scripts/run_model_queue.sh qwen2vl|smolvlm|internvl balanced (eval v2 at 0/5/10/20/40/100% in results/tier3/eval_v2_balanced, then paired benchmarks)
#   then: analyze_paired_bench / analyze_yes_bias --dir results/tier3/paired_bench_balanced --mix balanced
.\.venv\Scripts\python -m scripts.plot_final                           # results/figures/fig1-4
.\.venv\Scripts\python -m scripts.analyze_tier3_v2 --mix balanced [--model smolvlm|internvl]  # bootstrap CIs, balanced runs
.\.venv\Scripts\python -m scripts.baselines                          # run table (36 runs) + simple baselines -> results/tier3/baselines
.\.venv\Scripts\python -m scripts.plot_baselines                     # results/figures/fig5_balanced_curves, fig6_baselines
.\.venv\Scripts\python -m scripts.plot_architecture                    # results/figures/architecture.png (built vs proposed)
.\.venv\Scripts\python -m scripts.demo --part eval|data|results [--cached]   # review demo, see DEMO.md
.\.venv\Scripts\python tests/smoke_model.py smolvlm|internvl          # letter-scoring + accuracy sanity check
```
Long jobs were run detached (`Start-Process` + Git bash) via `scripts/run_eval_v2_remaining.sh`, `run_paired_bench.sh`,
`run_model_queue.sh <model>` (retries, disk guard, GPU heat guard, cool-down between jobs); log in `results/overnight.log`.

Outputs: `results/tier3/<run>/` (adapters, train/probe logs), `results/tier3/eval_v2/` (per-item CSVs, `checkpoint_eval_v2.csv`),
`results/tier3/analysis_v2{,_smolvlm,_internvl}/`, `results/tier3/paired_bench/` (`paired_bench_stats.csv`, `excluded_overlap.json`).

### Deviations and caveats (state these when reporting)
- **SmolVLM uses bf16 LoRA, not 4-bit QLoRA**: 4-bit NF4 dropped its MMStar accuracy from 40% to 29% (100-item check).
  Its option scoring starts the reply with `" Answer:"` (`score_reply_prefix`), otherwise 98% of probability goes to the word "Answer".
- **InternVL3 also uses bf16 LoRA** (kept comparable with SmolVLM; 4-bit not tested) and one 448 px tile per image
  (`crop_to_patches: false`, 256 image tokens, same budget as Qwen).
- SmolVLM and InternVL trained 125 steps (config) vs 120 for Qwen (from `step_sizing.json`); evaluated at 3 checkpoints, Qwen at 5.
- All three models are trained on the same Qwen-labelled mixtures and evaluated on the same A/G items.
- G was selected from **Qwen's** blind failures on option orders 0-5 and evaluated on order 0, so Qwen's G-blank accuracy at
  step 0 is biased low (3.8% vs 7.3% on fresh orders) and drifts up after any training; compare mixtures with each other.
- The seed-0 Qwen runs were trained before LoRA init was seeded; seeds 1 and 2 (and all SmolVLM and InternVL runs) are fully seeded.
- Paired benchmarks: 2 NaturalBench samples dropped by the overlap guard (false positives, dropped anyway), 2 skipped as
  malformed in the source data; 300 MMVP + 1192 NaturalBench questions remain.
- **The two mixtures differ in yes/no balance** (found after the runs): 75% vs 36% "yes" among VSR answers in text10 vs
  text90. Tier 2 matched source datasets but not answer type. Affects all models; see `yes_bias.csv`. All three models were
  retrained on balanced mixes (SmolVLM here; Qwen and InternVL on a teammate's RTX 5070, log `results/overnight_friend_5070.log`).
- 18+ comparisons per model without multiple-testing correction.

### Results in one paragraph
Qwen2-VL: the 90%-text-answerable run ends below the 10% run on A accuracy (-2.6 pts, CI -4.3..-1.0), G image benefit
(real-blank -2.5, CI -4.6..-0.2) and MMVP (answers change with the image 4.2 pts less often), all 3/3 seeds in the
predicted direction. SmolVLM: A accuracy is stable and G image benefit points the same way (-1.6 to -1.8, CIs include 0),
but on MMVP and NaturalBench the effect reverses (the 10% run ignores the image more; mostly one seed). The NaturalBench
reversal is confined to yes/no questions and follows the mixes' yes/no imbalance: SmolVLM's text10 runs say "Yes" 80% of
the time (seed 1: 88%; truth 50%) vs 70% for text90; on multiple-choice questions it goes the predicted way. Retrained on
yes/no-balanced mixes (`--stage mix_balanced`, `run_model_queue.sh smolvlm balanced`), SmolVLM's paired result flips to the
predicted direction (same-answer +7.8 pts MMVP, CI 2.2..13.3; +15.4 NaturalBench, CI 12.5..18.5; 3/3 seeds), but its balanced
text90 runs then say "Yes" 90% of the time and pick option A on 75% of MMVP, so for SmolVLM the paired tests mostly track a
default-answer bias that is very sensitive to the mix. InternVL3: A accuracy
is flat (-0.5, CI includes 0) while G accuracy and image benefit drop (-2.6 to -4.2, all three CIs exclude 0, 3/3 seeds);
its paired-image tests lean the same way as Qwen's but CIs include 0. Effects are 2-5
points: feasible to measure, not confirmed. Figures: `results/figures/`.

Do not run GPU jobs concurrently on this laptop: two processes time-slice the GPU and both slow down a lot.
