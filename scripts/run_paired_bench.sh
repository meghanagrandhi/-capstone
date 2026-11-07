#!/usr/bin/env bash
# Detached runner for the paired-image benchmarks (MMVP + NaturalBench). Log: results/overnight.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_DEACTIVATE_ASYNC_LOAD=1
L=results/overnight.log
echo "=== PAIRED BENCH START $(date '+%F %H:%M:%S') ($(df -BG /c | tail -1 | awk '{print $4}') free)" >> $L
for try in 1 2 3; do
  .venv/Scripts/python -W ignore -u -m scripts.run_paired_bench --runs matched_text10 matched_text90 matched_text10_s1 matched_text90_s1 matched_text10_s2 matched_text90_s2 > results/pipeline_logs/paired_bench_try$try.log 2>&1
  rc=$?; echo "=== END paired_bench try $try rc=$rc $(date +%H:%M:%S)" >> $L
  [ $rc -eq 0 ] && { echo "=== PAIRED BENCH COMPLETE $(date '+%F %H:%M:%S')" >> $L; exit 0; }
  sleep 20
done
echo "=== FAILED paired_bench" >> $L
