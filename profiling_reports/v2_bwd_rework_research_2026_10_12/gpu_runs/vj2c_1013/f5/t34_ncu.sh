#!/bin/bash
# 2026-10-13 tests 3+4 of the f5 hour: ncu --set full on gqa_bwd_vj2_kernel (11th launch, dense B8 18/6), f5 then f4s4. GPU 7.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; NCU=/usr/local/cuda/bin/ncu
cd $TI
for v in f5 f4s4; do
  echo "$(date +%H:%M:%S) ncu $v start lib $(md5sum $SN/$v/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
  env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/$v/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 $NCU --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_$v $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_$v.out 2>&1
  echo "$(date +%H:%M:%S) ncu $v exit=$?" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
  R=$(ls /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_$v.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_$v.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_import.err
  CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_$v.source.csv 2>> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/ncu_import.err
  [ $v = f5 ] && echo "T3 DONE" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
done
echo "T4 DONE" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
