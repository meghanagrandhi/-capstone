#!/usr/bin/env bash
# Overnight Tier 3 v2: bigger eval sets, seeds 1-2 for both matched mixtures, eval v2 (A/G real, G swap, 100-item probe)
# for all 6 runs. Each stage: up to 3 tries, 20 s pause, stops if C: has < 2 GB free. Log: results/overnight.log
# Usage (from repo root): bash scripts/run_overnight.sh
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/overnight.log
mkdir -p results/pipeline_logs
S=$(python -c "import json;print(json.load(open('results/tier3/step_sizing.json'))['steps_chosen'])")
freegb(){ df -BG /c | tail -1 | awk '{gsub("G","",$4); print $4}'; }
runstep(){
  name=$1; shift
  for try in 1 2 3; do
    sleep 20
    if [ "$(freegb)" -lt 2 ]; then echo "=== FAILED $name: only $(freegb) GB free on C:" >> $L; exit 1; fi
    echo "=== START $name try $try $(date +%H:%M:%S) ($(freegb) GB free)" >> $L
    .venv/Scripts/python -W ignore -u -m "$@" > results/pipeline_logs/ov_${name}_try$try.log 2>&1
    rc=$?
    echo "=== END $name try $try rc=$rc $(date +%H:%M:%S)" >> $L
    [ $rc -eq 0 ] && return 0
    grep -A3 "Traceback\|Error:" results/pipeline_logs/ov_${name}_try$try.log | tail -6 >> $L
  done
  echo "=== FAILED $name" >> $L; exit 1
}
echo "=== OVERNIGHT START $(date '+%F %H:%M:%S') steps=$S" >> $L
runstep eval_sets_v2 scripts.build_eval_sets --all
for seed in 1 2; do
  for mix in matched_text10 matched_text90; do
    runstep train_${mix}_s$seed scripts.run_tier3_train --mixture $mix --seed $seed --steps $S
  done
done
for run in matched_text10 matched_text90 matched_text10_s1 matched_text90_s1 matched_text10_s2 matched_text90_s2; do
  runstep eval_v2_$run scripts.run_tier3_eval_v2 --runs $run
done
echo "=== OVERNIGHT COMPLETE $(date '+%F %H:%M:%S')" >> $L
