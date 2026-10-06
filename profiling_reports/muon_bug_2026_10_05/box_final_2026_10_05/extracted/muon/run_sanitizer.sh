#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
T=$HOME/gautam_k/muon; CS=$(command -v compute-sanitizer)
echo "=== compute-sanitizer on test_muon_rsag_HEAD | $(date "+%F %T") | GPU 2 | $CS"
for tool in memcheck initcheck racecheck; do
  echo "--- $tool"
  mpirun -np 1 $CS --tool $tool --print-limit 20 $T/test_muon_rsag_HEAD > $T/logs/sanitizer_$tool.txt 2>&1
  echo "exit $?"; grep -E "ERROR SUMMARY|RACECHECK SUMMARY|Uninitialized|Invalid|hazard|PASS bit|FAIL .*differ" $T/logs/sanitizer_$tool.txt | sed -E "s/ \(worst.*//" | sort | uniq -c | head -12
done
echo "=== done $(date "+%T")"
