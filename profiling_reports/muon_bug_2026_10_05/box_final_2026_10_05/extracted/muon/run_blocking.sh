#!/usr/bin/env bash
# Same HEAD binary, CUDA_LAUNCH_BLOCKING=1: every kernel launch waits until it finishes.
export CUDA_VISIBLE_DEVICES=2 CUDA_LAUNCH_BLOCKING=1
T=$HOME/gautam_k/muon; N=${1:-20}; fails=0
echo "=== arm HEAD + CUDA_LAUNCH_BLOCKING=1 | $(date "+%F %T") | GPU 2 | binary $(md5sum $T/test_muon_rsag_HEAD | cut -c1-12)"
for i in $(seq 1 $N); do
  out=$T/logs/run_HEADblocking_$i.txt
  mpirun -np 1 $T/test_muon_rsag_HEAD > $out 2>&1; rc=$?
  [ $rc -ne 0 ] && fails=$((fails+1))
  printf "run %2d exit %d | %s\n" $i $rc "$(grep -E "^\[(sync publish|staged refresh)\] (FAIL|PASS)" $out | sed -E "s/ \(worst.*//" | tr "\n" " ")"
done
echo "--- arm HEAD+blocking: $fails of $N runs FAILED"
