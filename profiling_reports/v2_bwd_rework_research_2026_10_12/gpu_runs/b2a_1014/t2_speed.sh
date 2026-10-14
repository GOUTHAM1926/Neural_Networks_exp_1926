#!/bin/bash
# 2026-10-14 B-2a test 2: BENCHDENSE g123, b2a alternated 6 times (same session; forward identical code = noise check). GPU 7.
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots
echo "$(date +%H:%M:%S) t2 start g123 $(md5sum $SN/g123/lib/libtensor.so | cut -c1-8) b2a $(md5sum $SN/b2a/lib/libtensor.so | cut -c1-8)" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
cd $TI
for r in 1 2 3 4 5 6; do for s in g123 b2a; do
  env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/$s/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE timeout 600 $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/b2a_1014/2_${s}_r$r.log 2>&1
  echo "$(date +%H:%M:%S) t2 ${s}_r$r exit=$? ($(grep -c '^BENCH' /home/operator/gautam_k/v2_work/logs/b2a_1014/2_${s}_r$r.log) lines)" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
done; done
echo "T2 DONE" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status
