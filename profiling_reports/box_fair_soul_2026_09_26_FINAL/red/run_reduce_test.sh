#!/bin/bash
# GPU reduction test on the box.  usage:  bash run_reduce_test.sh before|after
#   before = current libtensor.so (unpatched)       after = rebuilt libtensor.so (patched)
# One process per case (a crash cannot hide the next case). One log: /root/gautam_k/red/reduce_<tag>.log
# "after" also runs compute-sanitizer: memcheck on case 0 (the old crash) and cases 37/40/42 (Generic,
# transposed), racecheck on case 4 (last axis) and case 29 (multi-CTA last axis, full report kept).
set -uo pipefail
tag=${1:?usage: run_reduce_test.sh before|after}
T=/root/gautam_k/red; LOG=$T/reduce_$tag.log; BIN=$T/trm_$tag
cd /root/gautam_k/BluTrain/Tensor-Implementations || exit 1
{
echo "=== GPU reduction test ($tag)  $(date '+%F %T')  lib $(git rev-parse --short HEAD)"
md5sum src/UnaryOps/cuda/ReductionImplGPU.cu include/ops/helpers/ReductionKernels.cuh
ls -la --time-style=+%F_%T lib/libtensor.so
echo "--- compile (CPU only)"
if ! /usr/local/cuda/bin/nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include \
    -DWITH_CUDA -DWITH_MYBLAS -IBluBridge-BLAS/include -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr \
    -gencode arch=compute_90a,code=sm_90a -o $BIN $T/test_reduce_mean_gpu.cu -L/usr/local/cuda/lib64 -Llib -L../Profiler/lib \
    -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda > $T/compile_$tag.log 2>&1; then
  echo "COMPILE FAILED:"; tail -20 $T/compile_$tag.log; exit 1
fi
export LD_LIBRARY_PATH=$PWD/lib:$PWD/../Profiler/lib:${LD_LIBRARY_PATH:-}
n=$($BIN list); echo "--- $n cases on GPU 0"
for i in $(seq 0 $((n - 1))); do
  r=$(CUDA_VISIBLE_DEVICES=0 timeout 180 $BIN $i 2>&1 | grep "^CASE")
  [ -n "$r" ] || r="CASE|#$i|CRASH|no output (aborted or timed out)"
  echo "$i $r"
done
if [ "$tag" = after ]; then
  CS=/usr/local/cuda/bin/compute-sanitizer
  echo "--- compute-sanitizer memcheck, case 0";  CUDA_VISIBLE_DEVICES=0 timeout 300 $CS --tool memcheck  $BIN 0  2>&1 | grep -E "ERROR SUMMARY|Invalid|CASE"
  echo "--- compute-sanitizer racecheck, case 4";  CUDA_VISIBLE_DEVICES=0 timeout 600 $CS --tool racecheck $BIN 4  2>&1 | grep -E "RACECHECK SUMMARY|ERROR SUMMARY|hazard|CASE" | head -5
  echo "--- compute-sanitizer racecheck, case 29 (both sides of each hazard)"; CUDA_VISIBLE_DEVICES=0 timeout 600 $CS --tool racecheck --racecheck-report all $BIN 29 > $T/racecheck29_$tag.txt 2>&1; grep -E "RACECHECK SUMMARY|Race reported|Error:|Warning:|and (Read|Write) access|CASE" $T/racecheck29_$tag.txt | head -24
  for k in 37 40 42; do echo "--- compute-sanitizer memcheck, case $k"; CUDA_VISIBLE_DEVICES=0 timeout 300 $CS --tool memcheck $BIN $k 2>&1 | grep -E "ERROR SUMMARY|Invalid|CASE"; done
fi
} 2>&1 | tee $LOG
{
echo; echo "==================== SUMMARY ($tag) ===================="
c() { grep -cE "^[0-9]+ CASE\|[^|]*\|$1\|" $LOG; }   # numbered case lines only (not the sanitizer re-runs)
echo "cases: $(grep -cE '^[0-9]+ CASE' $LOG)   ok: $(c ok)   WRONG: $(c WRONG)   CRASH: $(c CRASH)   EXCEPTION: $(c EXCEPTION)"
} | tee -a $LOG
echo "log: $LOG"
