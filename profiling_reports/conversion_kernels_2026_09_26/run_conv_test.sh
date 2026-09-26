#!/bin/bash
# GPU as_type test on the box.  usage:  bash run_conv_test.sh before|after
# One log: /root/gautam_k/convk/conv_<tag>.log (PAIR lines, TIME lines, SUMMARY)
set -uo pipefail
tag=${1:?usage: run_conv_test.sh before|after}
T=/root/gautam_k/convk; LOG=$T/conv_$tag.log; BIN=$T/tat_$tag
cd /root/gautam_k/BluTrain/Tensor-Implementations || exit 1
{
echo "=== GPU as_type test ($tag)  $(date '+%F %T')  lib $(git rev-parse --short HEAD)"
md5sum src/Kernels/cuda/misc/ConversionKernels.cu
ls -la --time-style=+%F_%T lib/libtensor.so
echo "--- compile (CPU only)"
if ! /usr/local/cuda/bin/nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include \
    -DWITH_CUDA -DWITH_MYBLAS -IBluBridge-BLAS/include -std=c++17 -Xcompiler=-fPIC -O3 --expt-relaxed-constexpr \
    -gencode arch=compute_90a,code=sm_90a -o $BIN $T/test_as_type_gpu.cu -L/usr/local/cuda/lib64 -Llib -L../Profiler/lib \
    -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda > $T/compile_$tag.log 2>&1; then
  echo "COMPILE FAILED:"; tail -30 $T/compile_$tag.log; exit 1
fi
export LD_LIBRARY_PATH=$PWD/lib:$PWD/../Profiler/lib:${LD_LIBRARY_PATH:-}
echo "--- run on GPU 0"
CUDA_VISIBLE_DEVICES=0 timeout 900 $BIN
echo "--- exit code $?"
} 2>&1 | tee $LOG
echo "log: $LOG"
