#!/bin/bash
# 2026-10-13 test 2b, second half (the first half was stopped at 14:39): GPU 7, one test only.
#  export kernel tables of the 14:36/14:38 nsys reports (Vj1d, Vj2; CPU only), nsys FA3 + FA3-det (B8 18/6),
#  ncu --set full on gqa_bwd_vj2_kernel (11th launch) + raw/details/source exports. Fixed Vj2 md5 bbd977be.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations
PY=/home/operator/gautam_k/benv/bin/python
NSYS=/usr/local/cuda/bin/nsys
NCU=/usr/local/cuda/bin/ncu
export LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/status; }
say "start lib $(md5sum $TI/lib/libtensor.so | cut -c1-8) vj2.cu $(md5sum $TI/src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu | cut -c1-8) tva_vj2 $(md5sum $TI/tva_vj2 | cut -c1-8)"
for r in nsys_vj1d nsys_vj2; do
  CUDA_VISIBLE_DEVICES= $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/$r /home/operator/gautam_k/v2_work/logs/vj2c_1013_2b_1434/$r.nsys-rep > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/$r.stats.out 2>&1
  say "$r (14:3x report) kern csv $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/${r}_cuda_gpu_kern_sum.csv) lines"
done
cd /home/operator/gautam_k/step6 && env BENCH_ONLY=fa3 BENCH_ITERS=5 timeout 300 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3 $PY /home/operator/gautam_k/step6/bench_step6_libs.py --dense dense_B8_H18x6 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3.out 2>&1; say "nsys_fa3 exit=$?"
CUDA_VISIBLE_DEVICES= $NSYS stats --report cuda_gpu_kern_sum --format csv --force-export=true --force-overwrite=true --output /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3 /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3.nsys-rep > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3.stats.out 2>&1
say "nsys_fa3 kern csv $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/nsys_fa3_cuda_gpu_kern_sum.csv) lines"
cd $TI && env USE_VJ2=1 VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 $NCU --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.out 2>&1; say "ncu exit=$?"
R=$(ls /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.ncu-rep* 2>/dev/null | head -1)
for p in raw details; do CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page $p > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.$p.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_import.err; done
CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.source.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_import.err
say "ncu exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.raw.csv) source $(grep -c . /home/operator/gautam_k/v2_work/logs/vj2c_1013_2c_1509/ncu_vj2.source.csv) lines"
say DONE
