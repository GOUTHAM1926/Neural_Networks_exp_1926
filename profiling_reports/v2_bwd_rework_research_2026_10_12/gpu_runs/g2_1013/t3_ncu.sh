#!/bin/bash
# 2026-10-13 G2 round, test 3: ncu --set full per-instruction on g2, dense B8 18/6: v62 forward (gqa_v62_sm90, 11th launch) and Vj2 backward (11th). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
NCU="/usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite"
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status; }
say "t3 start"
cd $TI
for k in fwd bwd; do
  [ $k = fwd ] && RX="regex:^gqa_v62_sm90\$" || RX="regex:^gqa_bwd_vj2_kernel\$"
  env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g2/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 $NCU -k "$RX" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_g2_$k $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_g2_$k.out 2>&1
  say "t3 ncu $k exit=$?"
  R=$(ls /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_g2_$k.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_g2_$k.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_import.err
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_g2_$k.source.csv 2>> /home/operator/gautam_k/v2_work/logs/g2_1013/ncu_import.err
done
say "T3 DONE"
