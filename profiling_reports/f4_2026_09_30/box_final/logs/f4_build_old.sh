#!/usr/bin/env bash
# F4, step 2: build the OLD code (a3aba3e + TI b975be5), CPU only. cuBLAS backend (default).
export PATH=/usr/local/cuda/bin:$PATH
cd ~/gautam_k/BluTrain || exit 1
echo "==== build start $(date "+%F %T")  BluTrain $(git log -1 --format=%h)  TI $(git -C Tensor-Implementations log -1 --format=%h)"
nvcc --version | tail -1
time make -j44 build-mpi FILE=bluscript_zero_ckpt.cpp
echo "==== build exit=$?  end $(date "+%F %T")"
ls -la _runner_ Tensor-Implementations/lib/libtensor.so dist/lib/libdist.so 2>&1
