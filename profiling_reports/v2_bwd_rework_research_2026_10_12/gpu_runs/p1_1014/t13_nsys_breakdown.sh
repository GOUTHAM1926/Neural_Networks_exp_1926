#!/bin/bash
# 2026-10-14 Phase 1 test 1.3: where the time goes in one backward call (main kernel + helper kernels + memory fills),
# dense B8 18/6 T4096, GPU 7. ours = Vj2 g123 (lib 1b9f444d, USE_VJ2=1), 5 timed iters after 10 warm-up;
# libraries = bench_step6_libs.py --dense dense_B8_H18x6 (BENCH_ITERS=5; fa3, fa3_det, fa4, cudnn, fa2 all run).
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; S=/home/operator/gautam_k/step6
NSYS=/usr/local/cuda/bin/nsys
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/p1_1014/status; }
say "t13 start | g123 lib $(md5sum $SN/g123/lib/libtensor.so | cut -c1-8)"
cd $TI && env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g123/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=5 timeout 300 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/p1_1014/nsys_ours $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/p1_1014/nsys_ours.out 2>&1
say "t13 nsys ours exit=$?"
cd $S && env CUDA_VISIBLE_DEVICES=7 BENCH_ITERS=5 timeout 400 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/p1_1014/nsys_libs /home/operator/gautam_k/benv/bin/python $S/bench_step6_libs.py --dense dense_B8_H18x6 > /home/operator/gautam_k/v2_work/logs/p1_1014/nsys_libs.out 2>&1
say "t13 nsys libs exit=$?"
for r in nsys_ours nsys_libs; do
  for rep in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_gpu_trace; do
    CUDA_VISIBLE_DEVICES= $NSYS stats --report $rep --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/p1_1014/$r /home/operator/gautam_k/v2_work/logs/p1_1014/$r.nsys-rep > /home/operator/gautam_k/v2_work/logs/p1_1014/$r.$rep.out 2>&1
  done
  say "t13 $r: kern $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/${r}_cuda_gpu_kern_sum.csv) mem $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/${r}_cuda_gpu_mem_time_sum.csv 2>/dev/null) trace $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/${r}_cuda_gpu_trace.csv) lines"
done
say "T13 DONE"
