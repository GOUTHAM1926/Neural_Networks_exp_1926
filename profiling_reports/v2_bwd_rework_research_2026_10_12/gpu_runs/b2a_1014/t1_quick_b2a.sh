#!/bin/bash
# 2026-10-14 B-2a test 1: quick CLASSIC (144) on b2a (Vj2 85eb2d0f: FA3 wait order between the dQ and dK GEMMs; lib f1213d72). GPU 7.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t1 start b2a lib $(md5sum $SN/b2a/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
cd $TI && env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/b2a/lib:$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/b2a_1014/1_quick_b2a.log 2>&1
echo "$(date +%H:%M:%S) t1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/b2a_1014/1_quick_b2a.log) | $(grep -c 'ALL RUN CHECKS PASS' /home/operator/gautam_k/v2_work/logs/b2a_1014/1_quick_b2a.log) all-pass" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
echo "T1 DONE" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
