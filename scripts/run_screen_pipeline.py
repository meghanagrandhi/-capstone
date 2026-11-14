"""Sequential pipeline for visual-grounding regression experiment:
- 8 training runs (SmolVLM-Instruct, BF16, LoRA r=16, lr=0.0002, 62 steps, seed 0, ckpts 0 & 100%)
- Fast evaluation (0% and 100% ckpts, A real, G real, G blank) -> results/tier3/eval_screen/
- All progress logged to results/screen.log
- Never runs concurrent GPU jobs.
"""
import glob
import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PY = [sys.executable, "-u", "-m"]
LOGDIR = os.path.join(ROOT, "results", "pipeline_logs")
os.makedirs(LOGDIR, exist_ok=True)
SCREEN_LOG = os.path.join(ROOT, "results", "screen.log")
EVAL_OUT = os.path.join(ROOT, "results", "tier3", "eval_screen")
os.makedirs(EVAL_OUT, exist_ok=True)

ENV = dict(os.environ, HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONIOENCODING="utf-8")
T0 = time.time()


def log_both(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{(time.time()-T0)/60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(SCREEN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_stage(name, args):
    log_both(f"=== START: {name} ===")
    t = time.time()
    log_file = os.path.join(LOGDIR, f"{name}.log")
    
    with open(log_file, "w", encoding="utf-8") as f_out, open(SCREEN_LOG, "a", encoding="utf-8") as f_screen:
        p = subprocess.Popen(PY + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV, text=True, encoding="utf-8")
        for line in p.stdout:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except Exception:
                pass
            f_out.write(line)
            f_screen.write(line)
        p.wait()
        rc = p.returncode

    duration = (time.time() - t) / 60
    log_both(f"=== END: {name} rc={rc} ({duration:.2f} min) ===")
    if rc != 0:
        log_both(f"ERROR: {name} failed with return code {rc}")
        sys.exit(rc)


def main():
    mixtures = [
        "screen_baseline_text10",
        "screen_baseline_text90",
        "screen_okvqa_text10",
        "screen_okvqa_text90",
        "screen_textvqa_text10",
        "screen_textvqa_text90",
        "screen_textcaps_text10",
        "screen_textcaps_text90",
    ]
    
    runs = [f"smolvlm_{m}" for m in mixtures]

    log_both("Starting sequential training for 8 runs...")
    for mix in mixtures:
        run_name = f"smolvlm_{mix}"
        run_dir = os.path.join("results", "tier3", run_name)
        ckpt62 = os.path.join(run_dir, "adapters", "step_0062")
        meta_file = os.path.join(run_dir, "run_meta.json")
        
        if os.path.exists(ckpt62) and os.path.exists(meta_file):
            log_both(f"Skipping {run_name}: already completed ({ckpt62} exists).")
            continue
            
        run_stage(
            f"train_{mix}",
            [
                "scripts.run_tier3_train",
                "--model", "smolvlm",
                "--mixture", mix,
                "--steps", "62",
                "--checkpoint-pcts", "0", "100"
            ]
        )

    log_both("All 8 training runs finished. Starting fast evaluation...")
    run_stage(
        "screen_evaluation",
        [
            "scripts.run_tier3_eval_v2",
            "--model", "smolvlm",
            "--runs", *runs,
            "--pcts", "0", "100",
            "--fast",
            "--out", EVAL_OUT
        ]
    )

    log_both("Evaluation finished successfully.")
    
    # Check evaluation summary
    eval_csv = os.path.join(EVAL_OUT, "checkpoint_eval_v2.csv")
    if os.path.exists(eval_csv):
        import pandas as pd
        df = pd.read_csv(eval_csv)
        log_both(f"Generated evaluation summary at {eval_csv} ({len(df)} rows).")

    # Clean up adapter files to produce submission package without bulky adapters
    log_both("Packaging: removing adapter weights (.safetensors / .bin) to keep submission folders clean.")
    removed_bytes = 0
    for run in runs:
        for adapter_file in glob.glob(os.path.join("results", "tier3", run, "adapters", "**", "*.safetensors"), recursive=True):
            try:
                removed_bytes += os.path.getsize(adapter_file)
                os.remove(adapter_file)
            except Exception as e:
                log_both(f"Warning: could not remove {adapter_file}: {e}")
    log_both(f"Removed {removed_bytes / 1e6:.1f} MB of adapter weights from results/tier3/.")
    log_both("COMPLETE: All 8 runs trained, evaluated, and packaged successfully.")


if __name__ == "__main__":
    main()
