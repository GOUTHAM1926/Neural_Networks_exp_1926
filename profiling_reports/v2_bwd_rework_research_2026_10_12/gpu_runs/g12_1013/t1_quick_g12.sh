#!/bin/bash
# 2026-10-13 G1 round, test 1: quick CLASSIC on g12 (Vj2 bwd G1 full-width dV/dK + G2, 22f130e2; fwd G2 060e6946; lib 41dda051). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t1 start g12 lib $(md5sum $SN/g12/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
cd $TI && env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g12/lib:$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g12_1013/1_quick_g12.log 2>&1
echo "$(date +%H:%M:%S) t1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/g12_1013/1_quick_g12.log) | $(grep -c 'ALL RUN CHECKS PASS' /home/operator/gautam_k/v2_work/logs/g12_1013/1_quick_g12.log) all-pass" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
echo "T1 DONE" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
