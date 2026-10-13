#!/bin/bash
# 2026-10-13 G2 round, test 2: BENCHDENSE f5, g2, f5, g2 (same session, fwd + bwd per shape). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t2 start f5 $(md5sum $SN/f5/lib/libtensor.so | cut -c1-8) g2 $(md5sum $SN/g2/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
cd $TI
for r in 1 2; do for s in f5 g2; do
  env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/$s/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g2_1013/2_${s}_r$r.log 2>&1
  echo "$(date +%H:%M:%S) t2 ${s}_r$r exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/g2_1013/2_${s}_r$r.log) lines)" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
done; done
echo "T2 DONE" >> /home/operator/gautam_k/v2_work/logs/g2_1013/status
