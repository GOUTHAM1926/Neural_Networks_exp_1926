#!/bin/bash
# 2026-10-13 G3 round, test 3: ncu --set full per-instruction on g123 backward (11th launch), dense B8 18/6. GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t3 start" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
cd $TI && env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g123/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 /usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_g123_bwd $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_g123_bwd.out 2>&1
echo "$(date +%H:%M:%S) t3 ncu exit=$?" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
R=$(ls /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_g123_bwd.ncu-rep* 2>/dev/null | head -1)
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_g123_bwd.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_g123_bwd.source.csv 2>> /home/operator/gautam_k/v2_work/logs/g123_1013/ncu_import.err
echo "T3 DONE" >> /home/operator/gautam_k/v2_work/logs/g123_1013/status
