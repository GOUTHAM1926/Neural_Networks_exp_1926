#!/bin/bash
G=~/gautam_k/v2_work/logs
while kill -0 4004118 2>/dev/null; do sleep 10; done
cd ~/gautam_k/v2_work/Tensor-Implementations || exit 1
PROF=$(cd ../Profiler/lib && pwd)
N=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | wc -l)
L=$G/G9_diag7_D10_h200_gpu7_1010_1955.log
echo "pre-check GPU 7: processes $N $(date +%H:%M:%S); libtensor md5 $(md5sum lib/libtensor.so | cut -c1-8)" > $L
[ "$N" = 0 ] || { echo ABORT busy >> $L; exit 9; }
LD_LIBRARY_PATH=$PWD/lib:$PROF CUDA_VISIBLE_DEVICES=7 VARLEN_ONLY=G9 VARLEN_CASE=D10 VARLEN_DIAG_LIST=455:4057,492:4094,791:2998,1102:3537,1140:2283,1262:1005,1359:1714 VARLEN_DIAG_ROW=791 VARLEN_DIAG_T0=2998 VARLEN_DIAG_REPEAT=3 VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 ./tva_diag >> $L 2>&1
echo "diag exit=$? $(date +%H:%M:%S)" >> $L
