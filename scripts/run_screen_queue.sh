#!/usr/bin/env bash
# Dataset-combination screen, option B (laptop, SmolVLM): 5 mixes x text share 10/90, seed 0, half length (62 steps).
# Eval: start and end checkpoints only (0 and 100% of training). Resumable; same safety as run_grid_queue.sh.
# Start detached (PowerShell):  Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList @("scripts/run_screen_queue.sh","smolvlm") -WindowStyle Hidden
# Log: results/screen.log ; per-job logs: results/pipeline_logs/screen_* ; eval output: results/tier3/eval_screen/
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: run_screen_queue.sh smolvlm"; exit 1; }
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/screen.log
E=results/tier3/eval_screen
mkdir -p results/pipeline_logs $E
freegb(){ df -BG . | tail -1 | awk '{gsub("G","",$4); print $4}'; }
gputemp(){ nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1 | tr -d ' \r'; }
COOL_C=${COOL_C:-65}; HOT_C=${HOT_C:-90}; OVERHEATS=0
cooldown(){ sleep ${COOL_S:-120}; while [ "$(gputemp)" -ge $COOL_C ]; do sleep 30; done; }
runstep(){
  name=$1; shift
  [ -n "$DRY" ] && { echo "DRY $name: python -m $*"; return 0; }  # DRY=1: print the jobs, run nothing
  for try in 1 2 3; do
    if [ "$(freegb)" -lt 3 ]; then echo "=== FAILED $name: only $(freegb) GB free" >> $L; exit 1; fi
    cooldown
    echo "=== START $name try $try $(date '+%F %H:%M:%S') ($(freegb) GB free, GPU $(gputemp)C)" >> $L
    .venv/Scripts/python -W ignore -u -m "$@" > results/pipeline_logs/screen_${name}_try$try.log 2>&1 &
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
    grep -A3 "Traceback\|Error:" results/pipeline_logs/screen_${name}_try$try.log | tail -6 >> $L
    sleep 20
  done
  echo "=== FAILED $name" >> $L; exit 1
}
echo "=== SCREEN QUEUE $* START $(date '+%F %H:%M:%S')" >> $L
for m in aokvqa_only cocoqa_only vsr_only size450; do  # tallyqa_only runs on the other laptop
  for s in 10 90; do
    run=smolvlm_screen_${m}_text${s}_st62
    mix=screen_${m}_text$s
    final=0062
    [ -f $E/${run}_step${final}_G_blank.csv ] && continue  # finished (train + eval)
    if [ ! -f results/tier3/$run/run_meta.json ]; then
      runstep train_$run scripts.run_tier3_train --model smolvlm --mixture $mix --steps 62 < /dev/null
    fi
    runstep eval_$run scripts.run_tier3_eval_v2 --model smolvlm --runs $run --fast --pcts 0 100 --out $E < /dev/null
    [ -n "$DRY" ] && continue  # dry run must not delete anything
    for d in results/tier3/$run/adapters/step_*; do [ "$d" = "results/tier3/$run/adapters/step_$final" ] || rm -rf "$d"; done
    [ -n "$DRY" ] && continue
    echo "=== DONE $run $(date '+%F %H:%M:%S')" >> $L
  done
done
echo "=== SCREEN QUEUE $* COMPLETE $(date '+%F %H:%M:%S')" >> $L
