#!/bin/bash
# Port test (2026-10-24): build + run test_rope_port.cu with the LIBRARY's flags and CUDA 13.0, on the CURRENT library
# file (copied to new_lib.cu here, md5 printed). GPU: CUDA_VISIBLE_DEVICES (set by the caller).
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash run_port_test.sh <Tensor-Implementations dir> [extra nvcc -D flags] 2>&1 | tee port_test_<box>_<gpu>.log
set -e
TI=${1:?usage: $0 <Tensor-Implementations dir> [-DBF16_BWD_DEPTH=.. -DBF16_RED_U=.. -DBF16_RED_STAGES=..]}
shift
cd "$(dirname "$0")"
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}; [ -x "$NVCC" ] || NVCC=$(command -v nvcc || ls -d /usr/local/cuda*/bin/nvcc | sort -V | tail -1)
cp "$TI/src/Kernels/cuda/norm/FusedRMSRoPE_bf16.cu" new_lib.cu
sed -e 's|^namespace cuda {$|namespace cuda { namespace ref {|' -e 's|^} // namespace cuda$|} } // namespace cuda|' \
    ref/FusedRMSRoPE_bf16_stageB.cu > ref_ns.cu
[ "$(grep -c '^namespace cuda { namespace ref {$' ref_ns.cu)" = 1 ] && [ "$(grep -c '^} } // namespace cuda$' ref_ns.cu)" = 1 ] \
    || { echo "ABORT: the reference was not renamed into namespace ref"; exit 3; }
echo "reference $(md5sum < ref/FusedRMSRoPE_bf16_stageB.cu | cut -c1-8) | library file $(md5sum < new_lib.cu | cut -c1-8) | flags: $* | $($NVCC --version | tail -1)"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -1 | tr -d .)   # 90 = H100/H200, 100/103 = B200/B300
$NVCC -std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_${CC}a,code=sm_${CC}a "$@" \
      -I"$TI/include" -I. -o test_rope_port test_rope_port.cu
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}"
./test_rope_port
