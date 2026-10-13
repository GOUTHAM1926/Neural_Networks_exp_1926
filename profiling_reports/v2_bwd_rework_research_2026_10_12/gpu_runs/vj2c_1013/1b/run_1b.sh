#!/bin/bash
# 2026-10-13 test 1b: compute-sanitizer memcheck / synccheck / racecheck on sanitize_vj2 (B=1, 6/2 heads, T=256, hd 128),
# backward = fixed Vj2 (USE_VJ2=1, GQA_bwd_vj2_sm90.cu md5 bbd977be). GPU 7, one test only.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations
SAN=/usr/local/cuda/bin/compute-sanitizer
export LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7 USE_VJ2=1
echo "$(date +%H:%M:%S) start lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) vj2.cu $(md5sum $TI/src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu | cut -c1-8) sanitize_vj2 $(md5sum $TI/sanitize_vj2 | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1b_1416/status
for tool in memcheck synccheck racecheck; do
  timeout 420 $SAN --tool $tool --print-limit 50 $TI/sanitize_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_1b_1416/1b_${tool}_vj2.log 2>&1
  echo "$(date +%H:%M:%S) $tool exit=$? | $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|sanitize_vj2:' /home/operator/gautam_k/v2_work/logs/vj2c_1013_1b_1416/1b_${tool}_vj2.log | tr '\n' ' ')" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1b_1416/status
done
echo DONE >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_1b_1416/status
