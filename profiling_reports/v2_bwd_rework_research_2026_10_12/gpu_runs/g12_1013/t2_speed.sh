#!/bin/bash
# 2026-10-13 G1 round, test 2: BENCHDENSE g2, g12, g2, g12 (same session; forward identical in both = noise check). GPU 2.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t2 start g2 $(md5sum $SN/g2/lib/libtensor.so | cut -c1-8) g12 $(md5sum $SN/g12/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
cd $TI
for r in 1 2; do for s in g2 g12; do
  env CUDA_VISIBLE_DEVICES=2 USE_VJ2=1 LD_LIBRARY_PATH=$SN/$s/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/g12_1013/2_${s}_r$r.log 2>&1
  echo "$(date +%H:%M:%S) t2 ${s}_r$r exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/g12_1013/2_${s}_r$r.log) lines)" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
done; done
echo "T2 DONE" >> /home/operator/gautam_k/v2_work/logs/g12_1013/status
