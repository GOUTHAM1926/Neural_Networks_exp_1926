#!/bin/bash
GK=$HOME/gautam_K
export PATH=/usr/local/cuda/bin:$PATH
# 2026-10-19: the SM clock and the power each FORWARD actually runs at, free clocks (work doc 45.49). Why: in the 10-19
# scoreboard every kernel ran at the 700 W cap (692-695 W), and SKV took 1.2% fewer cycles than K5q under ncu but was not
# faster in the free-clock race: if a change only removes idle cycles, the power per cycle rises and the clock may drop.
# This measures the clock each library gets. Per shape, each entry runs ~16 s of forwards back to back, in the order
# A B C .. C B A (every entry twice, mirrored); nvidia-smi every 500 ms (clocks.sm, power.draw.instant, the sw_power_cap
# flag, temperature, utilisation) + each run's start / end time in runs.txt. Read with deep/clock_power_read.py (samples
# from 4 s after a run's start to its end).
# Entries: vj2_<name> = ours from ~/gautam_K/libs (tva_a, as ab_multi.sh), fa3 = FA3 from benv (rivals_free_1016.py with
# RIVAL_FWD_ONLY=1, the scoreboard's inputs).
# usage: bash clock_power_check.sh <gpu> <run name> <entry> ...   (~1.3 min per entry)
G=${1:?gpu}; RUN=${2:?run name}; shift 2; ENT=("$@")
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LD=$GK/libs; PY=$GK/benv/bin/python; S=$GK/step6; SC=$GK/scripts
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/$RUN; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
for n in "${ENT[@]}"; do
  case $n in
    fa3) [ -x $PY ] && [ -f $SC/rivals_free_1016.py ] && [ -f $S/bench_step6_libs.py ] || { say "ABORT: benv / rivals script / step6 missing"; exit 3; } ;;
    *) got=$(LD_LIBRARY_PATH=$LD/$n:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
       [ "$got" = "$LD/$n/libtensor.so" ] || { say "ABORT: pointed at $LD/$n, the binary loads '$got'"; exit 3; } ;;
  esac
done
export CUDA_VISIBLE_DEVICES=$G
say "clock/power check start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used,power.limit --format=csv,noheader) | ${ENT[*]}"
nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,power.draw.instant,clocks_event_reasons.sw_power_cap,temperature.gpu,utilization.gpu \
  --format=csv,noheader,nounits -lms 500 > $L/clocks.csv 2>&1 &
SMI=$!; trap "kill $SMI 2> /dev/null" EXIT
ts() { date +'%Y/%m/%d %H:%M:%S.%3N'; }   # the same clock and format as nvidia-smi's timestamp
ORD=("${ENT[@]}" $(printf '%s\n' "${ENT[@]}" | tac)); i=0
for shape in dense packed; do
  for n in "${ORD[@]}"; do
    i=$((i + 1)); log=$L/cp_$(printf %02d $i)_${shape}_$n.log; t0=$(ts)
    if [ $n = fa3 ]; then
      if [ $shape = dense ]; then (cd $S && RIVAL_FWD_ONLY=1 timeout 300 $PY $SC/rivals_free_1016.py dense fa3 16400 > $log 2>&1)
      else (cd $S && RIVAL_FWD_ONLY=1 timeout 300 $PY $SC/rivals_free_1016.py packed fa3 $S/segments_1012.txt real_rows_752 34000 > $log 2>&1); fi
    else
      if [ $shape = dense ]; then a="VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=15000"
      else a="VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_ITERS=35000 VARLEN_BENCH_IMPL=ours_A_plan"; fi
      (cd $TI && env LD_LIBRARY_PATH=$LD/$n:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_DIR=fwd $a timeout 300 $TI/tva_a > $log 2>&1)
    fi
    rc=$?; echo "$shape $n $t0 | $(ts) | rc=$rc | $(basename $log)" >> $L/runs.txt
  done
  say "clock/power $shape done (${ORD[*]})"
done
say "CLOCK CHECK DONE"
