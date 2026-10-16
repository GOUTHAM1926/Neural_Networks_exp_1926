#!/bin/bash
# 2026-10-16: build each forward candidate into its own library folder (CPU only, no GPU). For every <name>: copy
# ~/gautam_k/scripts/cands/fwd_<name>.cu over the tree's GQA_fwd_sm90.cu, make, keep lib/libtensor.so as
# ~/gautam_k/libs/vj2_<name>/. The tree is left with fwd_F2.cu (the current best) at the end.
# usage: bash build_cands_1016.sh E1 F3c ...
TI=/root/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations; C=/root/gautam_k/scripts/cands; F=$TI/src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu
cd $TI || exit 1
for n in "$@" F2; do
  [ -f $C/fwd_$n.cu ] || { echo "missing $C/fwd_$n.cu"; exit 1; }
  cp $C/fwd_$n.cu $F && touch $F
  make -j$(nproc) > /tmp/build_$n.log 2>&1 || { echo "BUILD FAILED for $n (see /tmp/build_$n.log)"; tail -5 /tmp/build_$n.log; exit 1; }
  if [ $n != F2 ] || [ ! -d ~/gautam_k/libs/vj2_F2 ]; then mkdir -p ~/gautam_k/libs/vj2_$n && cp lib/libtensor.so ~/gautam_k/libs/vj2_$n/; fi
  echo "$n: source $(md5sum $F | cut -c1-8) | library $(md5sum lib/libtensor.so | cut -c1-8)"
done
