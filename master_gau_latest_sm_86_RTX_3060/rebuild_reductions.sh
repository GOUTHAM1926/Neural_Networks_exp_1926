#!/usr/bin/env bash
# Surgically recompile reduction units from current source and relink libtensor.so,
# reusing all other known-good objects (full `make` is blocked by an unrelated
# CudaCachingAllocator.cpp regression).
set -e
R=/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau_latest_sm_86_RTX_3060
NVCC=/usr/local/cuda-13.0/bin/nvcc
CPP="-Iinclude -I/usr/include -DWITH_CUDA"
CXXF="-std=c++2a -fPIC -O3 -fopenmp -mavx2 -mfma -mf16c"
NVF="-std=c++17 -Xcompiler=-fPIC -arch=sm_86 --use_fast_math -O3 --expt-relaxed-constexpr -lineinfo"
O=$R/lib/objects
cd "$R"

echo "[1/5] nvcc -dc ReductionImplGPU.cu"
$NVCC $CPP $NVF -dc -c src/UnaryOps/cuda/ReductionImplGPU.cu -o $O/src/UnaryOps/cuda/ReductionImplGPU.o 2>&1 | grep -v "warning:" | grep -v "note:" | grep -iE "error" || true
echo "[2/5] g++ ReductionUtils.cpp"
g++ $CPP $CXXF -c src/UnaryOps/cpu/ReductionUtils.cpp -o $O/src/UnaryOps/cpu/ReductionUtils.o 2>&1 | grep -iE "error" || true
echo "[3/5] g++ Reduction.cpp"
g++ $CPP $CXXF -c src/UnaryOps/cpu/Reduction.cpp -o $O/src/UnaryOps/cpu/Reduction.o 2>&1 | grep -iE "error" || true
echo "[4/5] device-link"
$NVCC $NVF -dlink $O/src/Views/ContiguousKernel.o $O/src/UnaryOps/cuda/ReductionImplGPU.o -o $O/device_link.o
echo "[5/5] relink libtensor.so"
ALL=$(find $O -name '*.o' ! -name 'device_link.o' | tr '\n' ' ')
$NVCC -shared $NVF $ALL $O/device_link.o -L/usr/lib64 -L$R/lib -Xlinker -rpath -Xlinker '$ORIGIN/lib' \
  -lcudart -ltbb -lcurand -lcublas -lcublasLt -lgomp -o $R/lib/libtensor.so
echo "DONE relink"
