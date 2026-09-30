#!/usr/bin/env bash
# Build + run the ZeRO-1 tests. Mirrors dist/FSDP/tests/run_ddp_new_tests.sh.
set -euo pipefail

ROOT=/root/gautam_k/BluTrain
CUDA=/usr/local/cuda
ZERO=$ROOT/dist/zero
OUT=$ZERO/build
mkdir -p "$OUT"

INC=(
  -I"$ZERO/include"
  -I"$ROOT/dist"
  -I"$ROOT/dist/communication/include"
  -I"$ROOT/Tensor-Implementations/include"
  -I"$ROOT/Profiler/include"
  -I"$CUDA/include"
  -I/usr/include
)
CXXFLAGS=(-std=c++2a -fPIC -O2 -g -DWITH_CUDA -fopenmp)

LDFLAGS=(
  -L"$ROOT/dist/lib" -L"$ROOT/Tensor-Implementations/lib" -L"$ROOT/Profiler/lib"
  -L"$CUDA/lib64" -L"$CUDA/lib64/stubs" -L/usr/lib/x86_64-linux-gnu
  -Wl,-rpath,"$ROOT/dist/lib" -Wl,-rpath,"$ROOT/Tensor-Implementations/lib"
  -Wl,-rpath,"$ROOT/Profiler/lib" -Wl,-rpath,"$CUDA/lib64"
  -Wl,-rpath,/usr/lib/x86_64-linux-gnu
)
LDLIBS=(-ldist -ltensor -lprofiler -lcudart -lcuda -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz)

echo "== compiling ShardedOptimizer.cpp =="
mpic++ "${CXXFLAGS[@]}" "${INC[@]}" -c "$ZERO/src/ShardedOptimizer.cpp" -o "$OUT/ShardedOptimizer.o"

echo "== compiling test_zero.cpp =="
mpic++ "${CXXFLAGS[@]}" "${INC[@]}" -c "$ZERO/tests/test_zero.cpp" -o "$OUT/test_zero.o"

echo "== linking test_zero =="
mpic++ -fPIC -o "$OUT/test_zero" "$OUT/test_zero.o" "$OUT/ShardedOptimizer.o" \
  "${LDFLAGS[@]}" "${LDLIBS[@]}"

echo "== running test_zero (W=1) =="
"$OUT/test_zero"
