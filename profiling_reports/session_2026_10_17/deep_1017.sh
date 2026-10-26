#!/bin/bash
# 2026-10-17 DEEP MEASUREMENT of the forward (F3c) vs FA3 / FA4 / cuDNN, one GPU, one session (~60 min). Goal: losses that ADD UP
# to the whole free-clock gap (work doc §45.20-21). Parts, in this order:
#  C. trace (1-2 min): the V62_TRACE build of F3c, normal forward + backward bench at free clocks; launch 900 of 1000 (and 950
#     in a second process) is written with per-pair clock64 phase stamps: dense and packed -> where our time goes, per phase,
#     per SM, and the tail (SMs that finish early)
#  B. ncu forward (~12 min, locked clock, diagnostic): ours F3c, FA3, FA4, cuDNN, dense + packed, --set full + SASS source
#  D. knock-out race (~25 min, free clocks, FORWARD ONLY): F3c vs TR (stamp cost) vs KOexp / KOres / KOepi (the most each part
#     can cost); 12 rounds, rotating order, 4000 dense / 8000 packed forwards per run (>= 4 s)
#  A. scoreboard (~15 min, free clocks): ours F3c vs FA3 / FA4 / cuDNN, forward + backward, kernel-only (nsys), clocks logged
# usage: bash deep_1017.sh <gpu> [parts, default "C B D A"]
GK=$HOME/gautam_k; export PATH=/usr/local/cuda/bin:$PATH
G=${1:?gpu}; PARTS=${2:-"C B D A"}
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; S=$GK/step6; SC=$GK/scripts; LIBS=$GK/libs
PY=$GK/benv/bin/python; NCU=/usr/local/cuda/bin/ncu
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/deep_1017; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
for n in F3c TR KOexp KOres KOepi; do [ -f $LIBS/vj2_$n/libtensor.so ] || { say "ABORT: $LIBS/vj2_$n missing (build first)"; exit 3; }; done
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader) | parts $PARTS | tva_a $(md5sum $TI/tva_a | cut -c1-8)"
OURS="USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB"
for part in $PARTS; do
case $part in
C)
  for shape in dense packed; do
    if [ $shape = dense ]; then a="VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6"; else a="VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan"; fi
    for at in 900 950; do
      (cd $TI && env LD_LIBRARY_PATH=$LIBS/vj2_TR:$TI/../Profiler/lib $OURS $a VARLEN_BENCH_ITERS=1000 V62_TRACE_FILE=$L/trace_${shape}_$at.bin V62_TRACE_AT=$at \
        timeout 600 $TI/tva_a > $L/trace_${shape}_$at.log 2>&1); rc=$?
      say "C trace $shape launch $at exit=$rc | $(grep -h 'V62_TRACE' $L/trace_${shape}_$at.log | head -1) | $(grep -h 'dir=fwd' $L/trace_${shape}_$at.log | grep -o 'med_ms=[0-9.]*')"
    done
  done ;;
B)
  ncu_run() {   # name, kernel regex, launch-skip, dir, then the command
    local n=$1 k=$2 sk=$3 d=$4; shift 4
    (cd $d && timeout 1200 $NCU --set full --target-processes all --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip $sk --launch-count 1 -o $L/ncu_$n "$@" > $L/ncu_$n.out 2>&1); local rc=$?
    local R=$(ls $L/ncu_$n.ncu-rep* 2>/dev/null | head -1)
    for pg in raw details; do CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page $pg > $L/ncu_$n.$pg.csv 2>> $L/ncu_$n.import.err; done
    CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page source --print-source sass > $L/ncu_$n.source.csv 2>> $L/ncu_$n.import.err
    say "B ncu $n exit=$rc | profiled $(grep -c '==PROF== Profiling' $L/ncu_$n.out) | raw $(grep -c . $L/ncu_$n.raw.csv) lines | source $(grep -c . $L/ncu_$n.source.csv) lines"
  }
  RIV='FlashAttnFwdSm90|FlashAttentionForwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_fprop'
  ncu_run ours_dense  'gqa_v62_sm90<.int.0>' 3 $TI env LD_LIBRARY_PATH=$LIBS/vj2_F3c:$TI/../Profiler/lib $OURS VARLEN_BENCH_ITERS=1 VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 $TI/tva_a
  ncu_run ours_packed 'gqa_v62_sm90<.int.1>' 3 $TI env LD_LIBRARY_PATH=$LIBS/vj2_F3c:$TI/../Profiler/lib $OURS VARLEN_BENCH_ITERS=1 VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan $TI/tva_a
  for c in fa3 fa4 cudnn; do
    ncu_run ${c}_dense  "$RIV" 10 $S $PY $SC/rivals_free_1016.py dense $c 1
    ncu_run ${c}_packed "$RIV" 10 $S $PY $SC/rivals_free_1016.py packed $c $S/segments_1012.txt real_rows_752 1
  done ;;
D)
  AB_QUICK=0 AB_NCU=0 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 \
    bash $SC/ab_multi.sh $G ko_1017 vj2_F3c vj2_TR vj2_KOexp vj2_KOres vj2_KOepi
  say "D knock-out race exit=$? (status in runs/ko_1017/status)" ;;
A)
  bash $SC/scoreboard_free.sh $G sbfree_1017 vj2_F3c vj2_F3c
  say "A scoreboard exit=$? (runs/sbfree_1017/summary.txt)" ;;
esac
done
say "DEEP DONE"
