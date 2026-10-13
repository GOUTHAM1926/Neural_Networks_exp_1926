#!/bin/bash
# 2026-10-13 test 1 of the f5 hour: quick CLASSIC on f5 (fix 5 = epilogue swizzle, on f4). Snapshot f5 = lib a07ca28b, Vj2 826ae057. GPU 7.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t1 start f5 lib $(md5sum $SN/f5/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
cd $TI && env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f5_1013/1_quick_f5.log 2>&1
echo "$(date +%H:%M:%S) t1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/vj2f5_1013/1_quick_f5.log) | $(grep -c 'ALL RUN CHECKS PASS' /home/operator/gautam_k/v2_work/logs/vj2f5_1013/1_quick_f5.log) all-pass" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
echo "T1 DONE" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
