#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; X=$T/xcheck; N=${1:-10}
cd $BT
cmd=$(make --no-print-directory -n build-mpi FILE=$X/test_muon_rsag_x.cpp SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- "-o _runner_ " | tail -1)
cmd=${cmd/-o _runner_ /-o $X/test_x }
echo "=== cross-check (latest code) | $(date "+%F %T") | GPU 2 | $N runs"
eval "$cmd" > $X/compile.log 2>&1 || { echo COMPILE FAILED; tail -20 $X/compile.log; exit 1; }
for i in $(seq 1 $N); do mpirun -np 1 $X/test_x > $X/run_$i.txt 2>&1; done
for k in "sync publish" "sync RB-vs-RB" "sync RS-vs-RS" "sync RB2-vs-RS2" "staged refresh" "staged RB-vs-RB" "staged RS-vs-RS" "staged RB2-vs-RS2"; do
  printf "%-20s FAIL in %2d of %d runs\n" "$k" "$(grep -l "^\[$k\] FAIL" $X/run_*.txt | wc -l)" "$N"
done
echo "=== done $(date "+%T")"
