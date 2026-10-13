#!/bin/bash
# 2026-10-13 Phase 1.5, fix 1 (dQ staging in fragment order + 1-D bulk reduce-add, FA3 scheme). GPU 7, one test at a time.
#  snapshots: base = v2_bbd977be (lib d2659657), f1 = fix 1 (lib 84c91617, GQA_bwd_vj2_sm90.cu a4f7d5f6)
#  1 quick CLASSIC on f1 (gate: ALL RUN CHECKS PASS) | 2 BENCHDENSE base/f1/base/f1 | 3 nsys f1 B8 18/6 | 4 ncu f1 main kernel
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
LBL=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
lp() { echo "$SN/$1/lib:$TI/lib:$TI/../Profiler/lib"; }
export CUDA_VISIBLE_DEVICES=7 USE_VJ2=1
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/status; }
say "start base lib $(md5sum $SN/v2_bbd977be/lib/libtensor.so | cut -c1-8) f1 lib $(md5sum $SN/f1/lib/libtensor.so | cut -c1-8) tva_vj2 $(md5sum $TI/tva_vj2 | cut -c1-8)"
cd $TI
env LD_LIBRARY_PATH=$(lp f1) VARLEN_QUICK=1 VARLEN_ONLY=CLASSIC VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/1_quick_f1.log 2>&1
say "1 quick f1 exit=$? | $(grep 'checks passed' /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/1_quick_f1.log)"
grep -q "ALL RUN CHECKS PASS" /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/1_quick_f1.log || { say "STOP: quick check failed"; say DONE; exit 1; }
for r in 1 2; do for s in base f1; do d=$([ $s = base ] && echo v2_bbd977be || echo f1)
  env LD_LIBRARY_PATH=$(lp $d) VARLEN_ONLY=BENCHDENSE VARLEN_STAGE0_LABELS=$LBL timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/2_${s}_r$r.log 2>&1; say "2 ${s}_r$r exit=$?"; done; done
env LD_LIBRARY_PATH=$(lp f1) VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=5 timeout 300 /usr/local/cuda/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1.out 2>&1; say "3 nsys f1 exit=$?"
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1 /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1.nsys-rep > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1.stats.out 2>&1; say "3 kern csv $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/nsys_f1_cuda_gpu_kern_sum.csv) lines"
env LD_LIBRARY_PATH=$(lp f1) VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 /usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.out 2>&1; say "4 ncu exit=$?"
R=$(ls /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.ncu-rep* 2>/dev/null | head -1)
for p in raw details; do CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page $p > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.$p.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_import.err; done
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.source.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_import.err
say "4 ncu exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.raw.csv) source $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2f1_1013_1614/ncu_vj2.source.csv) lines"
say DONE
