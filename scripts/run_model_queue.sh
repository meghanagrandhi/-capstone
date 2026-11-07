#!/usr/bin/env bash
# One model (arg: qwen2vl|smolvlm|internvl): 6 training runs (2 matched mixtures x 3 seeds) -> eval v2 (A/G real/blank/swap) -> paired
# benchmarks. Optional 2nd arg "balanced": the answer-balanced mixtures instead, with eval v2 at 0/5/10/20/40/100% of training in
# results/tier3/eval_v2_balanced, then paired benchmarks in results/tier3/paired_bench_balanced.
# Detached-safe (started with Start-Process, see README). 5-min cool-down pause between GPU jobs.
# Step 0 = base model for every run (LoRA B starts at zero), so its eval files are computed once and copied.
# Log: results/overnight.log
cd "$(dirname "$0")/.."
M=${1:?usage: run_model_queue.sh qwen2vl|smolvlm|internvl [matched|balanced]}
MX=${2:-matched}
PFX=${M}_; STEPS=""
[ $M = qwen2vl ] && { PFX=""; STEPS="--steps 120"; }  # Qwen runs have no name prefix and 120 steps (step_sizing.json)
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/overnight.log
mkdir -p results/pipeline_logs
freegb(){ df -BG /c | tail -1 | awk '{gsub("G","",$4); print $4}'; }
gputemp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' \r'; }
# heat guard: every job waits for the GPU to cool below COOL_C; a job is killed if the GPU stays >= HOT_C for
# 2 minutes, then retried after cooling; a second overheat stops the whole queue.
COOL_C=65; HOT_C=90; OVERHEATS=0
cooldown(){ sleep 120; while [ "$(gputemp)" -ge $COOL_C ]; do sleep 30; done; }
runstep(){
  name=$1; shift
  for try in 1 2 3; do
    if [ "$(freegb)" -lt 3 ]; then echo "=== FAILED $name: only $(freegb) GB free on C:" >> $L; exit 1; fi
    cooldown
    echo "=== START $name try $try $(date +%H:%M:%S) ($(freegb) GB free, GPU $(gputemp)C)" >> $L
    .venv/Scripts/python -W ignore -u -m "$@" > results/pipeline_logs/${M}_${name}_try$try.log 2>&1 &
    pid=$!; hot=0; killed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      t=$(gputemp)
      if [ "$t" -ge $HOT_C ]; then hot=$((hot + 1)); else hot=0; fi
      if [ $hot -ge 4 ]; then
        echo "=== TOO HOT: GPU ${t}C for 2 min, stopping $name $(date +%H:%M:%S)" >> $L
        kill -9 $pid; killed=1; break
      fi
    done
    wait $pid; rc=$?
    if [ $killed -eq 1 ]; then
      OVERHEATS=$((OVERHEATS + 1))
      [ $OVERHEATS -ge 2 ] && { echo "=== STOPPED: overheated twice, queue halted $(date '+%F %H:%M:%S')" >> $L; exit 1; }
      continue
    fi
    echo "=== END $name try $try rc=$rc $(date +%H:%M:%S)" >> $L
    [ $rc -eq 0 ] && return 0
    grep -A3 "Traceback\|Error:" results/pipeline_logs/${M}_${name}_try$try.log | tail -6 >> $L
    sleep 20
  done
  echo "=== FAILED $name" >> $L; exit 1
}
RUNS="${PFX}${MX}_text10 ${PFX}${MX}_text90 ${PFX}${MX}_text10_s1 ${PFX}${MX}_text90_s1 ${PFX}${MX}_text10_s2 ${PFX}${MX}_text90_s2"
echo "=== MODEL QUEUE ${M} ${MX} START $(date '+%F %H:%M:%S')" >> $L
for seed in 0 1 2; do
  for mix in ${MX}_text10 ${MX}_text90; do
    run=${PFX}${mix}$([ $seed -ne 0 ] && echo _s$seed)
    [ -f results/tier3/$run/run_meta.json ] && continue  # resumable: finished runs are skipped
    runstep train_$run scripts.run_tier3_train --model ${M} --mixture $mix --seed $seed $STEPS
    # disk: keep only the adapters that get evaluated (0, 5, 10, 20, 40, 100% of 120 or 125 steps)
    for d in results/tier3/$run/adapters/step_*; do
      case $d in *step_0000|*step_0006|*step_0012|*step_0024|*step_0025|*step_0048|*step_0050|*step_0120|*step_0125) ;; *) rm -rf "$d";; esac
    done
  done
done
if [ $MX = balanced ]; then
  E=results/tier3/eval_v2_balanced; mkdir -p $E
  first=${PFX}balanced_text10
  # base model (step 0) is the same as in the matched runs: reuse its cached per-item files (the probe differs, so not its json)
  for f in results/tier3/eval_v2/${PFX}matched_text10_step0000_*.csv; do cp -n "$f" "$E/$(basename "${f/matched_text10/balanced_text10}")"; done
  for run in $RUNS; do
    [ $run != $first ] && for f in $E/${first}_step0000_*; do cp -n "$f" "${f/$first/$run}"; done
    runstep eval_balanced_$run scripts.run_tier3_eval_v2 --model ${M} --runs $run --pcts 0 5 10 20 40 100 --probe-mixes matched balanced --out $E
  done
  P=results/tier3/paired_bench_balanced; mkdir -p $P
  # base model (step 0) is the same as in the matched runs: reuse its cached per-item file
  cp -n results/tier3/paired_bench/${PFX}matched_text10_step0000.csv $P/${PFX}balanced_text10_step0000.csv
  runstep paired_bench_${M}_balanced scripts.run_paired_bench --model ${M} --runs $RUNS --out $P
  echo "=== MODEL QUEUE ${M} ${MX} COMPLETE $(date '+%F %H:%M:%S')" >> $L
  exit 0
fi
E=results/tier3/eval_v2
first=${PFX}matched_text10
runstep eval_v2_$first scripts.run_tier3_eval_v2 --model ${M} --runs $first --pcts 0 40 100
for run in $RUNS; do
  [ $run = $first ] && continue
  for f in $E/${first}_step0000_*; do cp -n "$f" "${f/$first/$run}"; done
  runstep eval_v2_$run scripts.run_tier3_eval_v2 --model ${M} --runs $run --pcts 0 40 100
done
runstep paired_bench_${M} scripts.run_paired_bench --model ${M} --runs $RUNS
echo "=== MODEL QUEUE ${M} ${MX} COMPLETE $(date '+%F %H:%M:%S')" >> $L
