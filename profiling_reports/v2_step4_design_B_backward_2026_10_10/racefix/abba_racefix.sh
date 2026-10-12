#!/bin/bash
# Same-session A/B of the tile-0 barrier fix (2026-10-10). G7 timing of the SAME test binary, alternating
# two library builds: lib_nofix/ (snapshot before the fix) and lib/ (with the fix). 12 rounds, ABBA order.
# The forward is untouched by the fix, so its A/B is the null control.
# usage: abba_racefix.sh <logname>     (run from anywhere; GPU 7)
set -u
cd ~/gautam_k/v2_work/Tensor-Implementations || exit 1
OUT=~/gautam_k/v2_work/logs/$1
PROF=$(cd ../Profiler/lib && pwd)
: > $OUT
for v in nofix fix; do
  D=$PWD/lib; [ $v = nofix ] && D=$PWD/lib_nofix
  got=$(LD_LIBRARY_PATH=$D:$PROF ldd ./tva_step34 | awk '/libtensor/ {print $3}')
  echo "VARIANT $v loads $got md5 $(md5sum $got | cut -c1-8) (expected $D/libtensor.so)" | tee -a $OUT
  [ "$got" = "$D/libtensor.so" ] || { echo "ABORT: $v resolves the wrong libtensor" | tee -a $OUT; exit 3; }
done
for r in $(seq 1 12); do
  if [ $((r % 2)) -eq 1 ]; then order="nofix fix"; else order="fix nofix"; fi
  for v in $order; do
    D=$PWD/lib; [ $v = nofix ] && D=$PWD/lib_nofix
    echo "=== ROUND $r VARIANT $v $(date +%H:%M:%S) $(nvidia-smi -i 7 --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader)" >> $OUT
    LD_LIBRARY_PATH=$D:$PROF CUDA_VISIBLE_DEVICES=7 VARLEN_ONLY=G7 timeout 120 ./tva_step34 2>&1 | grep -E "B=8 T=4096|threw|error" >> $OUT
  done
done
echo "DONE $(date +%H:%M:%S)" >> $OUT
