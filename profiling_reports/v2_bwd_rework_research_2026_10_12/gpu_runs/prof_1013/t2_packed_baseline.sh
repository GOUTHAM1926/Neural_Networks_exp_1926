#!/bin/bash
# 2026-10-13 profiling round, test 2: same-session PACKED (SFT) speed baseline on segments_1012.txt (7 layouts), GPU 2.
#  ours = v62 forward (A, B) + Vj1d backward (Vj2 is dense-only), lib f5 snapshot, VARLEN_ONLY=BENCH
#  libraries: fa3_varlen (+_det), fa4_varlen, cudnn_varlen_gqa (CUDNN_GQA=1, probe PASS on 10-12), flex_docmask, torch_varlen (FA2)
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; S=/home/operator/gautam_k/step6
export CUDA_VISIBLE_DEVICES=2
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/prof_1013/status; }
say "t2 start | segments $(md5sum $S/segments_1012.txt | cut -c1-8)"
cd $TI && env LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCH VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin VARLEN_BENCH_EXPORT=/home/operator/gautam_k/v2_work/logs/prof_1013/2_segments_check.txt timeout 1200 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/prof_1013/2_ours_packed.log 2>&1
say "t2 ours exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/prof_1013/2_ours_packed.log) lines) | segments same as 10-12: $(diff <(sort $S/segments_1012.txt) <(sort /home/operator/gautam_k/v2_work/logs/prof_1013/2_segments_check.txt) > /dev/null 2>&1 && echo YES || echo NO)"
cd $S && env BENCH_ONLY=fa3_varlen,fa4_varlen,cudnn_varlen_gqa,flex_docmask,torch_varlen CUDNN_GQA=1 timeout 1500 /home/operator/gautam_k/benv/bin/python $S/bench_step6_libs.py $S/segments_1012.txt > /home/operator/gautam_k/v2_work/logs/prof_1013/2_libs_packed.log 2>&1
say "t2 libs exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/prof_1013/2_libs_packed.log) lines, $(grep -c 'GATE.*-> PASS' /home/operator/gautam_k/v2_work/logs/prof_1013/2_libs_packed.log) gate PASS, $(grep -c INVALID /home/operator/gautam_k/v2_work/logs/prof_1013/2_libs_packed.log) INVALID)"
say "T2 DONE"
