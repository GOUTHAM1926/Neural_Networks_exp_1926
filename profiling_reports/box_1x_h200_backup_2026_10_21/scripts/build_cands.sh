#!/bin/bash
GK=$HOME/gautam_k   # 2026-10-19: ONE folder on the 8x node: ~/gautam_k -> /localstorage/gautam_k (10-17: gautam_K)
export PATH=/usr/local/cuda/bin:$PATH
# 2026-10-16: build each forward candidate into its own library folder (CPU only, no GPU). For every <name>: copy
# ~/gautam_k/scripts/cands/fwd_<name>.cu over the tree's GQA_fwd_sm90.cu, make, keep lib/libtensor.so as
# $GK/libs/vj2_<name>/. 10-17: the tree is left on the LAST name (put F3c last = the laptop master's GQA_fwd_sm90.cu).
# usage: bash build_cands.sh [tva] F2 C1 F3c   (10-17 deep run: tva TR KOexp KOres KOepi F3c)   (tva = also rebuild the test binary tva_a, after the builds)
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; C=$GK/scripts/cands; F=$TI/src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu
cd $TI || exit 1
TVA=0; [ "$1" = tva ] && { TVA=1; shift; }
for n in "$@"; do
  [ -f $C/fwd_$n.cu ] || { echo "missing $C/fwd_$n.cu"; exit 1; }
  cp $C/fwd_$n.cu $F && touch $F
  make -j$(nproc) > /tmp/build_$n.log 2>&1 || { echo "BUILD FAILED for $n (see /tmp/build_$n.log)"; tail -5 /tmp/build_$n.log; exit 1; }
  mkdir -p $GK/libs/vj2_$n && cp lib/libtensor.so $GK/libs/vj2_$n/
  echo "$n: source $(md5sum $F | cut -c1-8) | library $(md5sum lib/libtensor.so | cut -c1-8)"
done
if [ $TVA = 1 ]; then   # the test binary, as built on this box on 10-15 (records sft_packed_pipeline_2026_10_15/box_runs/a1_1015_tests)
  [ -f tva_a ] && cp tva_a tva_a.bak_$(date +%m%d)
  nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA -std=c++17 -Xcompiler="-fPIC" --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a -o tva_a Tests/kernel_tests/test_varlen_attention.cu -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -Llib -L../Profiler/lib -Xlinker -rpath -Xlinker '$ORIGIN/lib' -Xlinker -rpath -Xlinker '$ORIGIN/../../Profiler/lib' -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lcuda > /tmp/build_tva.log 2>&1 \
    && echo "tva_a: test source $(md5sum Tests/kernel_tests/test_varlen_attention.cu | cut -c1-8) | binary $(md5sum tva_a | cut -c1-8)" || { echo "tva_a BUILD FAILED"; tail -5 /tmp/build_tva.log; exit 1; }
fi
