#!/usr/bin/env bash
# Forecaster grid queue (next semester). Args: one or more models (qwen2vl smolvlm internvl); runs every row of
# configs/grid_runs.csv for those models, in file order: train -> fast eval (A real, G real/blank at 0/40/100%) ->
# delete adapters except the final one. Resumable: a run with a finished eval is skipped.
# Same safety as run_model_queue.sh: disk guard, GPU heat guard, cool-down between jobs, 3 tries per job.
# Start detached on Windows (PowerShell):
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList @("scripts/run_grid_queue.sh","qwen2vl","internvl")
# Log: results/grid.log ; per-job logs: results/pipeline_logs/ ; eval output: results/tier3/eval_grid/
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "usage: run_grid_queue.sh qwen2vl|smolvlm|internvl [...]"; exit 1; }
PY=.venv/Scripts/python; [ -x .venv/bin/python ] && PY=.venv/bin/python  # Windows or Linux venv
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/grid.log
E=results/tier3/eval_grid
mkdir -p results/pipeline_logs $E
[ -f configs/grid_runs.csv ] || $PY -m scripts.make_grid
for m in grid_text10 grid_text30 grid_text50 grid_text70 grid_text90; do
  [ -f results/tier2/full/mixture_$m.csv ] || $PY -m scripts.run_tier2 --stage mix_grid
done
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
    $PY -W ignore -u -m "$@" > results/pipeline_logs/grid_${name}_try$try.log 2>&1 &
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
    grep -A3 "Traceback\|Error:" results/pipeline_logs/grid_${name}_try$try.log | tail -6 >> $L
    sleep 20
  done
  echo "=== FAILED $name" >> $L; exit 1
}
echo "=== GRID QUEUE $* START $(date '+%F %H:%M:%S')" >> $L
# run name = what run_tier3_train.py builds: [model_]grid_textNN[_lr..][_r..][_st..][_sN]
while IFS=, read -r model share lr rank steps seed locked <&3; do
  case " $* " in *" $model "*) ;; *) continue;; esac
  pfx=${model}_; [ $model = qwen2vl ] && pfx=""
  def=125; [ $model = qwen2vl ] && def=120
  hp=""
  [ "$($PY -c "print(float('$lr') != 2e-4)")" = True ] && hp="${hp}_lr$($PY -c "print(f'{float(\"$lr\"):g}')")"
  [ $rank != 16 ] && hp="${hp}_r$rank"
  [ $steps != $def ] && hp="${hp}_st$steps"
  run=${pfx}grid_text${share}${hp}$([ $seed -ne 0 ] && echo _s$seed)
  final=$(printf "%04d" $steps)
  [ -f $E/${run}_step${final}_G_blank.csv ] && continue  # finished (train + eval)
  if [ ! -f results/tier3/$run/run_meta.json ]; then
    runstep train_$run scripts.run_tier3_train --model $model --mixture grid_text$share --lr $lr --lora-r $rank \
      --steps $steps --seed $seed < /dev/null
  fi
  runstep eval_$run scripts.run_tier3_eval_v2 --model $model --runs $run --fast --out $E < /dev/null
  # keep only the final adapter (~70 MB per run)
  for d in results/tier3/$run/adapters/step_*; do [ "$d" = "results/tier3/$run/adapters/step_$final" ] || rm -rf "$d"; done
  [ -n "$DRY" ] && continue
  echo "=== DONE $run (locked_test=$locked) $(date '+%F %H:%M:%S')" >> $L
done 3< <(tail -n +2 configs/grid_runs.csv | tr -d '\r')  # not a pipe, so a failed job (exit) stops the whole queue
echo "=== GRID QUEUE $* COMPLETE $(date '+%F %H:%M:%S')" >> $L
