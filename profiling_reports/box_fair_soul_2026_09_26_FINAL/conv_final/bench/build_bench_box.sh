#!/bin/bash
# Build all three benchmarks on the box (CPU only; nothing runs on the GPU here).
#   usage: ./build_bench_box.sh <OLD Tensor-Implementations dir> <NEW Tensor-Implementations dir>
# Uses the same compiler and flags as the library build (/usr/local/cuda/bin/nvcc, CUDA 13.2 on the box).
set -e
OLD=$(cd "$1" && pwd); NEW=$(cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
NVCC=/usr/local/cuda/bin/nvcc
grep -q "exponent > 0x8E" "$OLD/include/dtype/Types.h" || { echo "OLD tree is not the old code"; exit 1; }
grep -q "arithmetic_to_bfloat16" "$NEW/include/dtype/Types.h" || { echo "NEW tree is not the patched code"; exit 1; }

echo "== 1/3 CPU micro-benchmark"
"$HERE/build_cpu.sh" "$OLD/include" "$NEW/include" /tmp/bench_cpu

echo "== 2/3 GPU micro-benchmark"
FL="-std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -I$HERE"
$NVCC $FL -DOwnTensor=OwnTensorOld -DBENCH_NS=gbench_old -I"$OLD/include" -c "$HERE/bench_gpu_body.cu" -o /tmp/bench_gpu_old.o
$NVCC $FL -DBENCH_NS=gbench_new -I"$NEW/include" -c "$HERE/bench_gpu_body.cu" -o /tmp/bench_gpu_new.o
$NVCC $FL -Xcompiler -fopenmp -I"$NEW/include" -c "$HERE/bench_gpu_main.cu" -o /tmp/bench_gpu_main.o
$NVCC $FL -Xcompiler -fopenmp /tmp/bench_gpu_main.o /tmp/bench_gpu_old.o /tmp/bench_gpu_new.o -o /tmp/bench_gpu

echo "== 3/3 library benchmark (compiled against the NEW headers; the library is picked at run time, no rpath)"
cd "$NEW"
$NVCC -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include \
      -DWITH_CUDA -DWITH_MYBLAS -IBluBridge-BLAS/include -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 \
      --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -o /tmp/bench_lib "$HERE/bench_lib.cu" \
      -L/usr/local/cuda/lib64 -Llib -L../Profiler/lib -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp \
      -lnvidia-ml -lprofiler -lblublas -lcuda
readelf -d /tmp/bench_lib | grep -qi "rpath\|runpath" && { echo "bench_lib has an rpath: the library choice would not be controlled"; exit 1; }
# the program must not define any Types.h conversion itself: the loader could bind the OLD
# library's calls to the program's (NEW) copy, and the "old" run would measure new code
if nm -C --defined-only /tmp/bench_lib | grep -E "OwnTensor::detail::|OwnTensor::bfloat16_t::|OwnTensor::float16_t::"; then
    echo "bench_lib defines Types.h conversion code (listed above): old/new would not be separated"; exit 1
fi
echo "built: /tmp/bench_cpu /tmp/bench_gpu /tmp/bench_lib"
