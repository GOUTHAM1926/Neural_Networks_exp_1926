#!/bin/bash
# 2026-10-16 scoreboard, DENSE (normal causal attention), 13 shapes, one GPU (2x H200 box, root).
#  ours = v62 forward + Vj2 backward (default) and + Vj1d backward (USE_VJ2=0); each twice, interleaved around the libraries
#  libraries = bench_step6_libs.py --dense: fa3, fa3_det, fa4, sdpa_cudnn, sdpa_flash (FA2)   (same as 10-13 t1)
# usage: bash sb_dense.sh <gpu>
G=${1:?gpu}; TI=/root/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations; S=/root/gautam_k/step6
PY=/home/operator/gautam_k/benv/bin/python; L=/root/gautam_k/sft_v2/runs/sb_1016; mkdir -p $L
export CUDA_VISIBLE_DEVICES=$G
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status_dense; }
say "dense start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) | bench $(md5sum $S/bench_step6_libs.py | cut -c1-8)"
ours() { cd $TI && env $2 LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 900 $TI/tva_a > $L/$1.log 2>&1; say "$1 exit=$? ($(grep -c '^BENCH' $L/$1.log) BENCH lines, $(grep -o 'backward mode: [^|]*' $L/$1.log | head -1))"; }
ours dense_vj2_r1  USE_VJ2=1
ours dense_vj1d_r1 USE_VJ2=0
cd $S && timeout 1500 $PY $S/bench_step6_libs.py --dense > $L/dense_libs.log 2>&1; say "dense_libs exit=$? ($(grep -c '^BENCH' $L/dense_libs.log) BENCH lines)"
ours dense_vj2_r2  USE_VJ2=1
ours dense_vj1d_r2 USE_VJ2=0
say "DENSE DONE"
