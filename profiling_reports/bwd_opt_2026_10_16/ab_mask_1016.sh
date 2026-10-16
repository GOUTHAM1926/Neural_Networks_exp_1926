#!/bin/bash
# 2026-10-16 backward fix 1: the causal mask as its own block behind one branch (GQA_bwd_vj2_sm90.cu, md5 1b76873c).
# Before, the mask ran on every step (predicated code): ncu B8 18/6 = 705M instructions vs FA3 625M; the fix skips 73
# instructions per warp per unmasked step (local SASS), about 83M of the 705M. One GPU of the shared 2x H200 box.
#  1. quick suite on the NEW library (G2 dense, G3-G6, G10/G11 packed, G7): correctness, about 2 min
#  2. same-session race OLD vs NEW library, 12 rounds, order old/new alternating (ABBA), our model's two shapes:
#     dense B8 18/6 and packed real_rows_752; 1000 iterations per run (>= 4 s of GPU work, so the clock settles under
#     the 700 W cap as in training), about 6 min
#  3. ncu of the NEW backward main kernels (instructions, cycles at the locked clock), about 1 min
#  GPU board: claims the GPU first (exit 2 = refused -> nothing starts), releases it at the end, also after a crash.
# usage: bash ab_mask_1016.sh <gpu> [new lib folder under ~/gautam_k/libs] [run name] [old lib folder]   (about 10 min)
#   10-16 run 1: vj2_maskblk / ab_mask_1016 (mask block after the exp loop); run 2: vj2_maskv3 / ab_maskv3_1016
#   (two exp copies behind one branch: masked on the diagonal steps, plain on the others; source md5 cc3a6125)
#   run 3: vj2_fwdg1 / ab_fwdg1_1016 / old vj2_maskv3 (forward G1: P@V as 8 m64n128k16; GQA_fwd_sm90.cu md5 86da0e1f)
#   run 4: vj2_fwdf1 / ab_fwdf1_1016 / old vj2_fwdg1 (forward F1: FA3's loop order; GQA_fwd_sm90.cu md5 79093fe2)
#   run 5: vj2_fwdg5 / ab_fwdg5_1016 / old vj2_fwdf1 (forward G5: dynamic work order; GQA_fwd_sm90.cu md5 8e3f54ca)
#   run 6: vj2_fwdf2 / ab_fwdf2_1016 / old vj2_fwdg5 (forward F2: always rescale O + uniform unit/stage; md5 b6c054a8)
#   run 7: vj2_fwdf3 / ab_fwdf3_1016 / old vj2_fwdf2 (forward F3: FA3 hand-over position + registers 24/240; md5 7b9ad666)
#   run 7 HUNG (packed: an inactive group fetched the next unit before its tail; quick suite timeout, 16:57-17:12 IST)
#   run 8: vj2_fwdf3b / ab_fwdf3b_1016 / old vj2_fwdf2 (F3 with the deadlock fixed; md5 ea5d9f75)
#   run 9: vj2_fwdf3c / ab_fwdf3c_1016 / old vj2_fwdf2 (F3 hand-over with the old 40/232 registers; md5 c6a67163)
#   step 3 profiles the backward AND the forward main kernels of the new library.
G=${1:?gpu}; B=/root/gpu_board/gpuboard
TI=/root/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations; NCU=/usr/local/cuda/bin/ncu
OLD=/root/gautam_k/libs/${4:-vj2_old}; NEW=/root/gautam_k/libs/${2:-vj2_maskblk}
LAB=/root/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=/root/gautam_k/sft_v2/runs/${3:-ab_mask_1016}; mkdir -p $L; cd $TI
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
# the binary must load the library of the folder it is pointed at, or the race compares a library with itself
for d in $OLD $NEW; do
  got=$(LD_LIBRARY_PATH=$d:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$d/libtensor.so" ] || { say "ABORT: pointed at $d, the binary loads '$got'"; exit 3; }
done
$B claim $G --note "gautam: attention backward fix A/B + tests, ~10 min" --hours 1 || { echo "claim of GPU $G refused, not starting" | tee -a $L/status; exit 2; }
trap "$B release $G" EXIT
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | old $(md5sum $OLD/libtensor.so | cut -c1-8) new $(md5sum $NEW/libtensor.so | cut -c1-8)"
run() {   # lib folder, log name, then env settings
  local d=$1 log=$2; shift 2
  env LD_LIBRARY_PATH=$d:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB "$@" timeout 900 $TI/tva_a > $L/$log 2>&1
}
# ---- 1. correctness
run $NEW quick_new.log VARLEN_QUICK=1; rc=$?
say "quick suite NEW exit=$rc | $(grep -E 'checks passed' $L/quick_new.log | head -1)"
[ $rc = 0 ] || { say "STOP: quick suite failed, no race"; exit 4; }
# ---- 2. race
for r in $(seq -w 1 12); do
  if [ $((10#$r % 2)) = 1 ]; then order="old new"; else order="new old"; fi
  for v in $order; do
    d=$OLD; [ $v = new ] && d=$NEW
    run $d race_${r}_${v}_dense.log  VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1000
    run $d race_${r}_${v}_packed.log VARLEN_ONLY=BENCH      VARLEN_CASE=real_rows_752  VARLEN_BENCH_ITERS=1000
  done
  say "race round $r done ($order)"
done
med() { grep -h "$1" $L/race_*_$2_$3.log | grep "dir=$4 " | grep -o 'med_ms=[0-9.]*' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)] " ms (" NR " runs)"}'; }
for v in old new; do
  say "race $v | dense bwd $(med impl=ours_dense $v dense bwd) fwd $(med impl=ours_dense $v dense fwd) | packed bwd $(med impl=ours_A_plan $v packed bwd) fwd $(med impl=ours_A_plan $v packed fwd)"
done
# ---- 3. ncu of the new main kernels
ncu_one() {   # name, kernel regex, then env settings
  local n=$1 k=$2; shift 2
  timeout 900 $NCU --set full --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip 3 --launch-count 1 -o $L/ncu_$n \
    env LD_LIBRARY_PATH=$NEW:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=1 "$@" $TI/tva_a > $L/ncu_$n.out 2>&1; local rc=$?
  CUDA_VISIBLE_DEVICES= $NCU --import $L/ncu_$n.ncu-rep --csv --page raw > $L/ncu_$n.raw.csv 2> /dev/null
  CUDA_VISIBLE_DEVICES= $NCU --import $L/ncu_$n.ncu-rep --csv --page details > $L/ncu_$n.details.csv 2> /dev/null
  CUDA_VISIBLE_DEVICES= $NCU --import $L/ncu_$n.ncu-rep --csv --page source --print-source sass > $L/ncu_$n.source.csv 2> /dev/null
  say "ncu $n exit=$rc | kernels profiled $(grep -c '==PROF== Profiling' $L/ncu_$n.out)"
}
ncu_one new_dense  'gqa_bwd_vj2_kernel<.bool.0>' VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6
ncu_one new_packed 'gqa_bwd_vj2_kernel<.bool.1>' VARLEN_ONLY=BENCH      VARLEN_CASE=real_rows_752
ncu_one new_fwd_dense  'gqa_v62_sm90<.int.0>' VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6
ncu_one new_fwd_packed 'gqa_v62_sm90<.int.1>' VARLEN_ONLY=BENCH      VARLEN_CASE=real_rows_752
say "AB DONE"
