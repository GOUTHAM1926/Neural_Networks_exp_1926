#!/bin/bash
# 2026-10-13 G3 round, test 1: quick CLASSIC on g123 (Vj2 a956e789 = G1+G2+G3 LSE loads before the S GEMM; fwd 060e6946; lib 1b9f444d). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t1 start g123 lib $(md5sum $SN/g123/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
cd $TI && env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g123/lib:$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g123_1013/1_quick_g123.log 2>&1
echo "$(date +%H:%M:%S) t1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/g123_1013/1_quick_g123.log) | $(grep -c 'ALL RUN CHECKS PASS' /home/operator/gautam_k/v2_work/logs/g123_1013/1_quick_g123.log) all-pass" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
echo "T1 DONE" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
