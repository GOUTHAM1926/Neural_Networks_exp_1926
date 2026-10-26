#!/bin/bash
# 2026-10-17: free-clock timing trace of the forward K1 (library vj2_TRK1 = K1 + clock stamps, deep/make_trace.py; output unchanged,
# same SASS shape as K1). Normal forward + backward bench, 1000 iterations (>= 4 s, power-settled clock); launch 900 (and 950, a
# second process, for repeatability) is written to a file. Dense B8 18/6 and packed real_rows_752. About 2 min, one GPU.
# Read on the laptop with deep/trace_read.py.   usage: bash trace_1017.sh <gpu> [lib folder, default vj2_TRK1]
GK=$HOME/gautam_K; export PATH=/usr/local/cuda/bin:$PATH
G=${1:?gpu}; LIB=${2:-vj2_TRK1}
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LIBS=$GK/libs
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/trace_1017_$LIB; mkdir -p $L   # one folder per library (K1's first run: runs/trace_1017)
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
got=$(LD_LIBRARY_PATH=$LIBS/$LIB:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
[ "$got" = "$LIBS/$LIB/libtensor.so" ] || { say "ABORT: the binary loads '$got', not $LIBS/$LIB"; exit 3; }
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader) | $LIB $(md5sum $LIBS/$LIB/libtensor.so | cut -c1-8)"
for shape in dense packed; do
  if [ $shape = dense ]; then a="VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6"; else a="VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan"; fi
  for at in 900 950; do
    (cd $TI && env LD_LIBRARY_PATH=$LIBS/$LIB:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB $a VARLEN_BENCH_ITERS=1000 \
      V62_TRACE_FILE=$L/trace_${shape}_$at.bin V62_TRACE_AT=$at timeout 600 $TI/tva_a > $L/trace_${shape}_$at.log 2>&1); rc=$?
    say "trace $shape launch $at exit=$rc | $(grep -h 'V62_TRACE' $L/trace_${shape}_$at.log | head -1) | fwd $(grep -h 'dir=fwd' $L/trace_${shape}_$at.log | grep -o 'med_ms=[0-9.]*')"
  done
done
say "TRACE DONE"
