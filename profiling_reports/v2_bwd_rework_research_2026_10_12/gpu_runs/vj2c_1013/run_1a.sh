#!/bin/bash
# 2026-10-13 test 1a: quick CLASSIC on the fixed Vj2 (GQA_bwd_vj2_sm90.cu md5 bbd977be), GPU 7, one test only.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations
export LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7
echo "$(date +%H:%M:%S) start lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) vj2.cu $(md5sum $TI/src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu | cut -c1-8) tva_vj2 $(md5sum $TI/tva_vj2 | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1410/status
cd $TI && env USE_VJ2=1 VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_1410/1a_smoke.log 2>&1
echo "$(date +%H:%M:%S) 1a exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/vj2c_1013_1410/1a_smoke.log)" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1410/status
echo DONE >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1410/status
