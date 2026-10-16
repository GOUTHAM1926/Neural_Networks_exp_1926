#!/bin/bash
# usage: sass.sh <src.cu (relative to TI)> <out.o>   -- compiles one .cu for sm_90a like the library build
TI=$HOME/Downloads/BluTrain_sft/Tensor-Implementations
cd $TI && nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda-13.0/include -DWITH_CUDA \
  -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
  -Xptxas -v -c "$1" -o "$2" 2> "$2.log"; rc=$?
grep -E 'error|warning' "$2.log" | grep -v 'Function properties' | head -20
grep -B1 -A3 'Compiling entry' "$2.log" | grep -E 'Compiling entry|registers|spill' | sed 's/ptxas info    : //' | paste - - - 2>/dev/null | head -20
exit $rc
