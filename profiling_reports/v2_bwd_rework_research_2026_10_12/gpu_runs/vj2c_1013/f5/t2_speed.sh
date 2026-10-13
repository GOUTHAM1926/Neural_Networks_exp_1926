#!/bin/bash
# 2026-10-13 test 2 of the f5 hour: BENCHDENSE f4, f5, f4s4, f4, f5, f4s4 (same session). GPU 7.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t2 start libs f4 $(md5sum $SN/f4/lib/libtensor.so | cut -c1-8) f5 $(md5sum $SN/f5/lib/libtensor.so | cut -c1-8) f4s4 $(md5sum $SN/f4s4/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
cd $TI
for r in 1 2; do for s in f4 f5 f4s4; do
  env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/$s/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/vj2f5_1013/2_${s}_r$r.log 2>&1
  echo "$(date +%H:%M:%S) t2 ${s}_r$r exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/vj2f5_1013/2_${s}_r$r.log) BENCH lines)" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
done; done
echo "T2 DONE" >> /home/operator/gautam_k/v2_work/logs/vj2f5_1013/status
