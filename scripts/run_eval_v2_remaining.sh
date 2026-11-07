#!/usr/bin/env bash
# Resume eval v2 for all 6 runs (cached results are skipped, nothing is retrained). Meant to be started DETACHED
# (see README) so it survives the Claude Code session closing. Log: results/overnight.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/overnight.log
freegb(){ df -BG /c | tail -1 | awk '{gsub("G","",$4); print $4}'; }
runstep(){
  name=$1; shift
  for try in 1 2 3; do
    sleep 20
    if [ "$(freegb)" -lt 2 ]; then echo "=== FAILED $name: only $(freegb) GB free on C:" >> $L; exit 1; fi
    echo "=== START $name try $try $(date +%H:%M:%S) ($(freegb) GB free)" >> $L
    .venv/Scripts/python -W ignore -u -m "$@" > results/pipeline_logs/ov2_${name}_try$try.log 2>&1
    rc=$?
    echo "=== END $name try $try rc=$rc $(date +%H:%M:%S)" >> $L
    [ $rc -eq 0 ] && return 0
    grep -A3 "Traceback\|Error:" results/pipeline_logs/ov2_${name}_try$try.log | tail -6 >> $L
  done
  echo "=== FAILED $name" >> $L; exit 1
}
echo "=== EVAL RESUME START $(date '+%F %H:%M:%S')" >> $L
for run in matched_text10 matched_text90 matched_text10_s1 matched_text90_s1 matched_text10_s2 matched_text90_s2; do
  runstep eval_v2_$run scripts.run_tier3_eval_v2 --runs $run
done
echo "=== OVERNIGHT COMPLETE $(date '+%F %H:%M:%S')" >> $L
