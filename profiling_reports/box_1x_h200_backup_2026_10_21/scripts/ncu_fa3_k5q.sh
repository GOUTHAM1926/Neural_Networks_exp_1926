#!/bin/bash
# 2026-10-19: ncu (--set full; ncu's default locked base clock) of the attention FORWARD kernels on ONE box / compiler / clock:
# FA3 (benv: flash_attn_3 3.0.0 built with nvcc 13.4) vs ours K5q (libs/vj2_K5q a804ea04, nvcc 13.4), dense B8 x 4096 18/6 and
# packed real_rows_752. Same settings as ab_multi.sh's ncu (4th launch, one kernel), inputs as in the 10-19 scoreboard
# (rivals_free_1016.py / tva_a). Why: the dense forward is the only entry we lose (1.084x FA3, work doc 45.43); 45.41 showed
# that stall shares mislead, so the comparison must be about tensor-pipe idle time, which needs FA3 measured the same way.
# Exports raw + source (SASS) CSV on the box; ~3-4 min on one GPU.
# usage: setsid nohup bash /root/gautam_K/scripts/ncu_fa3_k5q.sh <gpu> > /dev/null 2>&1 < /dev/null &
GK=$HOME/gautam_K; G=${1:?gpu}; RUN=ncu_fa3_k5q_1019
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; NCU=/usr/local/cuda/bin/ncu; PY=$GK/benv/bin/python; S=$GK/step6; SC=$GK/scripts
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin; LIB=$GK/libs/vj2_K5q
L=$GK/sft_v2/runs/$RUN; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
got=$(LD_LIBRARY_PATH=$LIB:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
[ "$got" = "$LIB/libtensor.so" ] || { say "ABORT: tva_a loads '$got', not $LIB/libtensor.so"; exit 3; }
[ -x $PY ] && [ -f $SC/rivals_free_1016.py ] && [ -f $S/bench_step6_libs.py ] || { say "ABORT: benv / rivals script / step6 missing"; exit 3; }
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | ours vj2_K5q $(md5sum $LIB/libtensor.so | cut -c1-8) | ncu $($NCU --version | tail -1)"
prof() {   # name, kernel regex, working dir, then the command
  local n=$1 k=$2 d=$3; shift 3
  (cd $d && timeout 900 $NCU --set full --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip 3 --launch-count 1 \
     -o $L/$n "$@" > $L/$n.out 2>&1)
  local rc=$? rep=$(ls $L/$n.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= $NCU --import $rep --csv --page raw > $L/$n.raw.csv 2> /dev/null
  CUDA_VISIBLE_DEVICES= $NCU --import $rep --csv --page source --print-source sass > $L/$n.source.csv 2> /dev/null
  say "$n exit=$rc | kernels profiled $(grep -c '==PROF== Profiling' $L/$n.out) | raw $(grep -c . $L/$n.raw.csv) lines | source $(grep -c . $L/$n.source.csv) lines"
}
OURS="LD_LIBRARY_PATH=$LIB:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=1"
prof ours_dense  'gqa_v62_sm90<.int.0>' $TI env $OURS VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 $TI/tva_a
prof fa3_dense   'FlashAttnFwdSm90'     $S  $PY $SC/rivals_free_1016.py dense fa3 10
prof ours_packed 'gqa_v62_sm90<.int.1>' $TI env $OURS VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan $TI/tva_a
prof fa3_packed  'FlashAttnFwdSm90'     $S  $PY $SC/rivals_free_1016.py packed fa3 $S/segments_1012.txt real_rows_752 10
say "NCU DONE"
