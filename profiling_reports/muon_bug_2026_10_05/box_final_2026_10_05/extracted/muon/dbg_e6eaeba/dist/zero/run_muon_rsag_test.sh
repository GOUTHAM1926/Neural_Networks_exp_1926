#!/usr/bin/env bash
# Build + run the ShardedMuon reduce-scatter/all-gather equivalence test.
#
# Unlike build_and_test.sh this derives the repo root from its own location, so
# it works on any checkout. Runs at every world size the box can provide up to
# 2 — which is where the bit-exact assertion applies (see the test's header for
# why W>=3 only gets a tolerance).
set -euo pipefail

ZERO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$ZERO/../.." && pwd)"
CUDA="${CUDA_HOME:-/usr/local/cuda}"
OUT="$ZERO/build"
mkdir -p "$OUT"

INC=(
  -I"$ZERO/include"
  -I"$ROOT/dist"
  -I"$ROOT/dist/communication/include"
  -I"$ROOT/Tensor-Implementations/include"
  -I"$ROOT/Profiler/include"
  -I"$CUDA/include"
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

echo "== compiling ShardedMuon.cpp =="
mpic++ "${CXXFLAGS[@]}" "${INC[@]}" -c "$ZERO/src/ShardedMuon.cpp" -o "$OUT/ShardedMuon.o"

echo "== compiling test_muon_rsag.cpp =="
mpic++ "${CXXFLAGS[@]}" "${INC[@]}" -c "$ZERO/tests/test_muon_rsag.cpp" -o "$OUT/test_muon_rsag.o"

echo "== linking =="
mpic++ -fPIC -o "$OUT/test_muon_rsag" "$OUT/test_muon_rsag.o" "$OUT/ShardedMuon.o" \
  "${LDFLAGS[@]}" "${LDLIBS[@]}"

# libtensor.so pulls in libblublas.so by bare soname, which the link-time rpath
# does not resolve for a transitive dependency.
export LD_LIBRARY_PATH="$ROOT/Tensor-Implementations/lib:$ROOT/dist/lib:$ROOT/Profiler/lib:$CUDA/lib64:${LD_LIBRARY_PATH:-}"

NGPU="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
MAXNP="${MAX_NP:-2}"
[ "$NGPU" -lt "$MAXNP" ] && MAXNP="$NGPU"

for NP in $(seq 1 "$MAXNP"); do
  echo
  echo "== running test_muon_rsag (W=$NP) =="
  mpirun --allow-run-as-root -np "$NP" "$OUT/test_muon_rsag"
done
