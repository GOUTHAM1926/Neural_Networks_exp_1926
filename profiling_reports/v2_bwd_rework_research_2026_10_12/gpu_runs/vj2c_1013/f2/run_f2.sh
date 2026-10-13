#!/bin/bash
# 2026-10-13 Phase 1.5, fix 2 (LSE loads between the S and dP GEMM issues, FA3 order), on top of fix 1. GPU 7.
#  snapshots: f1 = lib 84c91617 (Vj2 a4f7d5f6), f2 = lib 6ffda947 (Vj2 ab6e5cbb)
#  1 quick CLASSIC on f2 (gate) | 2 BENCHDENSE f1/f2/f1/f2 | 3 nsys f2 B8 18/6 | 4 ncu f2 main kernel
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
LBL=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
lp() { echo "$SN/$1/lib:$TI/lib:$TI/../Profiler/lib"; }
export CUDA_VISIBLE_DEVICES=7 USE_VJ2=1
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/status; }
say "start f1 lib $(md5sum $SN/f1/lib/libtensor.so | cut -c1-8) f2 lib $(md5sum $SN/f2/lib/libtensor.so | cut -c1-8) tva_vj2 $(md5sum $TI/tva_vj2 | cut -c1-8)"
cd $TI
env LD_LIBRARY_PATH=$(lp f2) VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/1_quick_f2.log 2>&1
say "1 quick f2 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/1_quick_f2.log)"
grep -q "ALL RUN CHECKS PASS" /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/1_quick_f2.log || { say "STOP: quick check failed"; say DONE; exit 1; }
for r in 1 2; do for s in f1 f2; do
  env LD_LIBRARY_PATH=$(lp $s) VARLEN_ONLY=BENCHDENSE VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/2_${s}_r$r.log 2>&1; say "2 ${s}_r$r exit=$?"; done; done
env LD_LIBRARY_PATH=$(lp f2) VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=5 timeout 300 /usr/local/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2.out 2>&1; say "3 nsys f2 exit=$?"
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2 /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2.nsys-rep > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2.stats.out 2>&1; say "3 kern csv $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/nsys_f2_cuda_gpu_kern_sum.csv) lines"
env LD_LIBRARY_PATH=$(lp f2) VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 /usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.out 2>&1; say "4 ncu exit=$?"
R=$(ls /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.ncu-rep* 2>/dev/null | head -1)
for p in raw details; do CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page $p > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.$p.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_import.err; done
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.source.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_import.err
say "4 ncu exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.raw.csv) source $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f2_1013_1627/ncu_vj2.source.csv) lines"
say DONE
