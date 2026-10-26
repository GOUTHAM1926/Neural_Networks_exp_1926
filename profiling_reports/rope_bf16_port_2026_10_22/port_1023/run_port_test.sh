#!/bin/bash
# Port test (2026-10-23): build + run test_rope_port.cu with the LIBRARY's flags and CUDA 13.0 (the box's default
# /usr/local/cuda is 13.4; the library and the SFT run use 13.0). GPU: CUDA_VISIBLE_DEVICES (set by the caller).
# usage: CUDA_VISIBLE_DEVICES=3 bash run_port_test.sh <Tensor-Implementations dir> 2>&1 | tee port_test_<box>_<gpu>.log
set -e
TI=${1:?usage: $0 <Tensor-Implementations dir>}
cd "$(dirname "$0")"
NVCC=/usr/local/cuda-13.0/bin/nvcc
sed -e 's|^namespace cuda {$|namespace cuda { namespace ref {|' -e 's|^} // namespace cuda$|} } // namespace cuda|' \
    ref/FusedRMSRoPE_bf16_stageB.cu > ref_ns.cu
[ "$(grep -c '^namespace cuda { namespace ref {$' ref_ns.cu)" = 1 ] && [ "$(grep -c '^} } // namespace cuda$' ref_ns.cu)" = 1 ] \
    || { echo "ABORT: the reference was not renamed into namespace ref"; exit 3; }
echo "reference $(md5sum < ref/FusedRMSRoPE_bf16_stageB.cu | cut -c1-8) | port $(md5sum < new_lib.cu | cut -c1-8) | $($NVCC --version | tail -1)"
$NVCC -std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
      -I"$TI/include" -I. -o test_rope_port test_rope_port.cu
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}"
./test_rope_port
