#!/bin/bash
# 2026-10-16 scoreboard, PACKED (SFT rows, per-chat masking), the 7 layouts of segments_1012.txt, one GPU.
#  ours = v62 forward (seqlens) + Vj2 backward (default) and + Vj1d backward (USE_VJ2=0), VARLEN_ONLY=BENCH (dense row
#         and seqlens plan); each twice, interleaved around the libraries
#  libraries (same as 10-13 t2): fa3_varlen (+det), fa4_varlen, cudnn_varlen_gqa (CUDNN_GQA=1), flex_docmask, torch_varlen (FA2)
# usage: bash sb_packed.sh <gpu>
G=${1:?gpu}; TI=/root/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations; S=/root/gautam_k/step6
PY=/home/operator/gautam_k/benv/bin/python; L=/root/gautam_k/sft_v2/runs/sb_1016; mkdir -p $L
LAB=/root/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
export CUDA_VISIBLE_DEVICES=$G
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status_packed; }
say "packed start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) | segments $(md5sum $S/segments_1012.txt | cut -c1-8)"
ours() { cd $TI && env $2 LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCH VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_EXPORT=$L/$1.segments.txt timeout 1200 $TI/tva_a > $L/$1.log 2>&1; say "$1 exit=$? ($(grep -c '^BENCH' $L/$1.log) BENCH lines) | segments same as 10-12: $(diff <(sort $S/segments_1012.txt) <(sort $L/$1.segments.txt) > /dev/null 2>&1 && echo YES || echo NO)"; }
ours packed_vj2_r1  USE_VJ2=1
ours packed_vj1d_r1 USE_VJ2=0
cd $S && env BENCH_ONLY=fa3_varlen,fa4_varlen,cudnn_varlen_gqa,flex_docmask,torch_varlen CUDNN_GQA=1 timeout 1800 $PY $S/bench_step6_libs.py $S/segments_1012.txt > $L/packed_libs.log 2>&1
say "packed_libs exit=$? ($(grep -c '^BENCH' $L/packed_libs.log) BENCH lines, $(grep -c 'GATE.*-> PASS' $L/packed_libs.log) gate PASS, $(grep -c INVALID $L/packed_libs.log) INVALID)"
ours packed_vj2_r2  USE_VJ2=1
ours packed_vj1d_r2 USE_VJ2=0
say "PACKED DONE"
