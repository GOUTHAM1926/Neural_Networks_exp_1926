#!/bin/bash
# 2026-10-13 G2 round, test 1: quick CLASSIC on g2 (uniform warpgroup index in v62 fwd 060e6946 + Vj2 bwd bed7731f; lib 1a00c378). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t1 start g2 lib $(md5sum $SN/g2/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
cd $TI && env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g2/lib:$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g2_1013/1_quick_g2.log 2>&1
echo "$(date +%H:%M:%S) t1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/g2_1013/1_quick_g2.log) | $(grep -c 'ALL RUN CHECKS PASS' /home/operator/gautam_k/v2_work/logs/g2_1013/1_quick_g2.log) all-pass" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
echo "T1 DONE" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
