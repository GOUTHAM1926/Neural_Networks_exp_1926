#!/bin/bash
# 2026-10-13 profiling round, test 1: same-session DENSE speed baseline, 13 shapes, GPU 2.
#  ours = v62 forward + final Vj2 backward (snapshot f5, lib a07ca28b, USE_VJ2=1), twice, around one libraries run
#  libraries = bench_step6_libs.py --dense: fa3, fa3_det, fa4, sdpa_cudnn, sdpa_flash (FA2)
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
export CUDA_VISIBLE_DEVICES=2
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/prof_1013/status; }
say "t1 start | GPU 2 uuid $(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader) | f5 lib $(md5sum $SN/f5/lib/libtensor.so | cut -c1-8) | bench $(md5sum /home/operator/gautam_k/step6/bench_step6_libs.py | cut -c1-8)"
cd $TI && env USE_VJ2=1 LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/prof_1013/1_ours_r1.log 2>&1; say "t1 ours_r1 exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/prof_1013/1_ours_r1.log) lines)"
cd /home/operator/gautam_k/step6 && timeout 900 /home/operator/gautam_k/benv/bin/python /home/operator/gautam_k/step6/bench_step6_libs.py --dense > /home/operator/gautam_k/v2_work/logs/prof_1013/1_libs.log 2>&1; say "t1 libs exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/prof_1013/1_libs.log) lines)"
cd $TI && env USE_VJ2=1 LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/prof_1013/1_ours_r2.log 2>&1; say "t1 ours_r2 exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/prof_1013/1_ours_r2.log) lines)"
say "T1 DONE"
