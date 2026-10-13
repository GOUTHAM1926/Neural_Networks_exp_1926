#!/bin/bash
# 2026-10-13 test 2a: speed of the fixed Vj2 (GQA_bwd_vj2_sm90.cu md5 bbd977be). GPU 7, one test only.
#  0. reachability: nsys kernel list of sanitize_vj2 with USE_VJ2=1 (must show gqa_bwd_vj2_kernel)
#  1. BENCHDENSE Vj1d, Vj2, Vj1d, Vj2 (13 dense shapes, same session)
#  2. libraries on the same 13 shapes: bench_step6_libs.py --dense
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations
LBL=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
NSYS=/usr/local/cuda/bin/nsys
export LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/status; }
say "start lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) vj2.cu $(md5sum $TI/src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu | cut -c1-8) tva_vj2 $(md5sum $TI/tva_vj2 | cut -c1-8)"
env USE_VJ2=1 timeout 120 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2 $TI/sanitize_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2.out 2>&1
CUDA_VISIBLE_DEVICES= $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2 /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2.nsys-rep > /dev/null 2>&1
say "reach: $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2_cuda_gpu_kern_sum.csv) csv lines | vj2 kernel lines: $(grep -c gqa_bwd_vj2_kernel /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/reach_vj2_cuda_gpu_kern_sum.csv)"
for r in 1 2; do
  env VARLEN_ONLY=BENCHDENSE VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/2a_vj1d_r$r.log 2>&1; say "vj1d_r$r exit=$?"
  env USE_VJ2=1 VARLEN_ONLY=BENCHDENSE VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/2a_vj2_r$r.log 2>&1; say "vj2_r$r exit=$?"
done
cd /home/operator/gautam_k/step6 && timeout 900 /home/operator/gautam_k/benv/bin/python /home/operator/gautam_k/step6/bench_step6_libs.py --dense > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2a_1424/2a_libs_dense.log 2>&1
say "libs exit=$?"
say DONE
