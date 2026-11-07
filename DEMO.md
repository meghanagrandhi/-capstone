# Review 2 demo run sheet (about 8 minutes)

Run from the project folder in PowerShell. Set this once per terminal:
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
```
Nothing here trains a model. The forecasting model is **not built**: show it only on the architecture figure, as the next phase.

## 1. Visual dependence evaluation (live, ~45 s, GPU)
```powershell
.\.venv\Scripts\python -W ignore -m scripts.demo --part eval
```
- One MMStar question ("what emotion does this emoji express?") is scored with the real image, a gray image and another item's image.
  Open `results/demo/eval_real.png`, `eval_blank.png` and `eval_swap.png` side by side.
- SmolVLM: real image gives "sad" (100%, right). Gray image gives "angry" (81%, wrong). Swapped image, two smiling men shaking hands, gives "happy" (95%, wrong).
- Then the saved 500-item result: Qwen2-VL 45.4% with the real image vs 26.4% blank (chance 25%), McNemar p = 4.9e-12.
  All three base models: real / blank / swap = Qwen 45.4 / 26.4 / 23.0, SmolVLM 40.0 / 20.4 / 23.4, InternVL3 56.8 / 30.2 / 31.2.
- Fallback if the GPU is busy or the model is slow: add `--cached` (same output from the saved scores, no model loaded).
- Live uses SmolVLM because its weights are on disk. `--model qwen2vl` or `--model internvl` re-downloads ~4 GB each first.

## 2. Dataset preparation and mixtures (CPU, a few seconds)
```powershell
.\.venv\Scripts\python -W ignore -m scripts.demo --part data
```
- Pool: 3161 Cauldron items (VSR, A-OKVQA, COCO-QA, TallyQA); 33 duplicate images and 6 MMStar overlaps removed.
- Label = loss gap (answer loss with a gray image minus with the real image). Thresholds are the 40th and 60th percentile within each subset.
- Examples (images in `results/demo/`): text-answerable "Which person has a legendary story involving this animal?" -> "Lady Godiva"
  (prior knowledge is enough); image-necessary "Where is most of the kids weight?" -> "On skis".
- Mixtures: 1000 items, 250 per subset, 10% vs 90% text-answerable; mean loss gap 1.17 vs 0.09.
- The leakage guard runs live and prints "0 overlaps" for both mixtures (it raises an error on any overlap).

## 3. LoRA training and evaluation (CPU, a few seconds)
```powershell
.\.venv\Scripts\python -W ignore -m scripts.demo --part results
```
- LoRA r=16, lr 2e-4, batch 8, 120-125 steps; 3 models x 2 mixtures x 3 seeds = 18 runs. Training loss falls in every run.
- Main example, **InternVL3**, 90% minus 10% run: overall accuracy -0.5 (CI includes 0); G accuracy -2.6, G image benefit
  -3.7 (vs gray) and -4.2 (vs swap), all CIs exclude 0, 3/3 seeds.
- Be ready for: Qwen2-VL's **overall** accuracy also dropped (-2.6, CI excludes 0), so for Qwen accuracy did not stay stable.
  SmolVLM: no CI excludes 0 on MMStar, and its paired-image tests track its answer bias (report, follow-up section).

## 4. Figures (open the files)
`results/figures/`: `fig1_blindness.png`, `fig3_contrast_forest.png` (the main one), `fig2_training_curves.png`,
`fig4_A_split.png`, `architecture.png` (built vs proposed).

Say once: feasibility sketch, not evidence. 3 seeds, about 20 contrasts per model, no multiple-testing correction.
