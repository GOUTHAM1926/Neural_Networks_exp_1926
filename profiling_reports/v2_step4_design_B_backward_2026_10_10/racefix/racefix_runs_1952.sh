#!/bin/bash
cd ~/gautam_k/v2_work/Tensor-Implementations || exit 1
G=~/gautam_k/v2_work/logs
LBL=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
PROF=$(cd ../Profiler/lib && pwd)
N=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | wc -l)
echo "pre-check GPU 7: processes $N $(date +%H:%M:%S)" > $G/racefix_runs_1952.status
[ "$N" = 0 ] || { echo ABORT busy >> $G/racefix_runs_1952.status; exit 9; }
for M in default USE_FLASH; do
  L=$G/G_quick_racefix_${M}_h200_gpu7_1010_1952.log
  echo "libtensor resolved: $(LD_LIBRARY_PATH=$PWD/lib:$PROF ldd ./tva_step34 | awk '/libtensor/ {print $3}') md5 $(md5sum lib/libtensor.so | cut -c1-8)" > $L
  if [ $M = default ]; then E=""; else E="USE_FLASH=1"; fi
  env $E LD_LIBRARY_PATH=$PWD/lib:$PROF CUDA_VISIBLE_DEVICES=7 VARLEN_QUICK=1 VARLEN_REPEATS=1000 VARLEN_STAGE0_LABELS=$LBL timeout 1500 ./tva_step34 >> $L 2>&1
  echo "$M exit=$? $(date +%H:%M:%S)" >> $G/racefix_runs_1952.status
done
bash $G/abba_racefix.sh abba_racefix_h200_gpu7_1010_1952.log
echo "abba exit=$? $(date +%H:%M:%S)" >> $G/racefix_runs_1952.status
echo ALLDONE >> $G/racefix_runs_1952.status
