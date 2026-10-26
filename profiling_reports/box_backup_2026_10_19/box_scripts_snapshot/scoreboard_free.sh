#!/bin/bash
GK=$HOME/gautam_K   # 2026-10-19: the new 2x box's folder is /root/gautam_K (capital K); 10-17: box-independent
export PATH=/usr/local/cuda/bin:$PATH
# 2026-10-16 free-clock, kernel-only scoreboard: ours (F2 dense, F3b packed; backward version 3 in both) vs FA3 / FA4 / cuDNN,
# forward + backward, dense B8 18/6 and packed real_rows_752. One contender per process under nsys (kernel times only, no
# Python overhead), forward + backward back to back, >= 4 s of GPU work per run (power window), ABBA order per shape
# (ours fa3 fa4 cudnn | cudnn fa4 fa3 ours), nvidia-smi clock / power log per run. The traces are summarised ON THE BOX
# (scoreboard_free.py) and deleted after; only summary.txt, status and the clock logs need to come back.
# Claims the GPU on the board first (refused -> nothing starts), releases it at the end, also after a crash.
# usage: bash scoreboard_free.sh <gpu> <run name> <dense lib folder> <packed lib folder> [shapes]
# 10-19 (new box, CUDA 13.4): ours = vj2_K5q for both shapes (forward K5q, backward version 3); rivals' env = benv_setup.sh.
#   e.g. setsid nohup bash /root/gautam_K/scripts/scoreboard_free.sh 0 sb_1019 vj2_K5q vj2_K5q > /dev/null 2>&1 < /dev/null &
# 10-17: ours packed now 2000 iterations like the rivals' packed runs (1000 was ~2 s, under the 4 s power window), ours
# packed runs only ours_A_plan (VARLEN_BENCH_IMPL); the GPU board is used only where it exists.
G=${1:?gpu}; RUN=${2:?run}; DL=${3:?dense lib}; PL=${4:?packed lib}; SHAPES=${5:-"dense packed"}; B=/root/gpu_board/gpuboard
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; S=$GK/step6; SC=$GK/scripts
PY=$GK/benv/bin/python; NSYS=/usr/local/cuda/bin/nsys; LIBS=$GK/libs
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/$RUN; mkdir -p $L; cd $GK
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
for d in $LIBS/$DL $LIBS/$PL; do
  got=$(LD_LIBRARY_PATH=$d:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$d/libtensor.so" ] || { say "ABORT: pointed at $d, the binary loads '$got'"; exit 3; }
done
[ -f $SC/rivals_free_1016.py ] && [ -f $S/bench_step6_libs.py ] && [ -f $S/segments_1012.txt ] || { say "ABORT: scripts missing"; exit 3; }
if [ -x $B ]; then $B claim $G --note "gautam: free-clock attention scoreboard (ours vs FA3/FA4/cuDNN), ~10 min" --hours 1 || { echo "claim of GPU $G refused, not starting" | tee -a $L/status; exit 2; }; trap "$B release $G" EXIT; fi
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | ours dense $DL $(md5sum $LIBS/$DL/libtensor.so | cut -c1-8) packed $PL $(md5sum $LIBS/$PL/libtensor.so | cut -c1-8)"
clk_on()  { TZ=Asia/Kolkata nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 100 > $L/$1.clocks.csv 2>&1 & CLK=$!; }
clk_off() { kill $CLK 2>/dev/null; wait $CLK 2>/dev/null; }
run() {   # name, working dir, then the command
  local n=$1 d=$2; shift 2
  clk_on $n
  (cd $d && timeout 900 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $L/$n "$@" > $L/$n.out 2>&1); local rc=$?
  clk_off
  CUDA_VISIBLE_DEVICES= $NSYS stats --report cuda_gpu_trace --format csv --force-export=true --force-overwrite=true --output $L/$n $L/$n.nsys-rep > $L/$n.stats.out 2>&1
  rm -f $L/$n.sqlite
  say "$n exit=$rc | trace rows $(grep -c . $L/${n}_cuda_gpu_trace.csv 2>/dev/null) | busy clock samples $(grep -c ' %$' $L/$n.clocks.csv)"
}
OURS="LD_LIBRARY_PATH=$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=1000"
for shape in $SHAPES; do
  for r in 1 2; do
    if [ $r = 1 ]; then order="ours fa3 fa4 cudnn"; else order="cudnn fa4 fa3 ours"; fi
    for c in $order; do
      if [ $c = ours ]; then
        if [ $shape = dense ]; then run ${shape}_ours_r$r $TI env $OURS LD_LIBRARY_PATH=$LIBS/$DL:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 $TI/tva_a
        else run ${shape}_ours_r$r $TI env $OURS LD_LIBRARY_PATH=$LIBS/$PL:$TI/../Profiler/lib VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan VARLEN_BENCH_ITERS=2000 $TI/tva_a; fi
      else
        if [ $shape = dense ]; then run ${shape}_${c}_r$r $S $PY $SC/rivals_free_1016.py dense $c 1000
        else run ${shape}_${c}_r$r $S $PY $SC/rivals_free_1016.py packed $c $S/segments_1012.txt real_rows_752 2000; fi
      fi
    done
  done
done
python3 $SC/scoreboard_free.py $L > $L/summary.txt 2>&1; say "summary exit=$? ($(grep -c . $L/summary.txt) lines)"
say "SCOREBOARD DONE"
