#!/bin/bash
# Build the CPU micro-benchmark: old and new conversion code in one program.
#   usage: ./build_cpu.sh <OLD include dir> <NEW include dir> [out binary]
# Uses the library's CPU flags (Tensor-Implementations Makefile TENSOR_CXXFLAGS).
set -e
OLD_INC=$1; NEW_INC=$2; OUT=${3:-./bench_cpu}
HERE=$(cd "$(dirname "$0")" && pwd)
FLAGS="-std=c++2a -O3 -fopenmp -mavx2 -mfma -mf16c -I$HERE"
grep -q "exponent > 0x8E" "$OLD_INC/dtype/Types.h" || { echo "OLD include does not contain the old clamp: wrong folder"; exit 1; }
grep -q "arithmetic_to_bfloat16" "$NEW_INC/dtype/Types.h" || { echo "NEW include is not the patched code: wrong folder"; exit 1; }
g++ $FLAGS -DOwnTensor=OwnTensorOld -DBENCH_NS=bench_old -I"$OLD_INC" -c "$HERE/bench_cpu_body.cpp" -o /tmp/bench_cpu_old.o
g++ $FLAGS -DBENCH_NS=bench_new -I"$NEW_INC" -c "$HERE/bench_cpu_body.cpp" -o /tmp/bench_cpu_new.o
g++ $FLAGS -c "$HERE/bench_cpu_main.cpp" -o /tmp/bench_cpu_main.o
g++ -fopenmp /tmp/bench_cpu_main.o /tmp/bench_cpu_old.o /tmp/bench_cpu_new.o -o "$OUT"
# the two copies must not share any conversion symbol
if nm -C /tmp/bench_cpu_old.o | grep -q " OwnTensor::"; then echo "old object still has OwnTensor:: symbols: rename failed"; exit 1; fi
echo "built $OUT"
