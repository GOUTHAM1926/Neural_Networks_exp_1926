#!/bin/bash
GK=$HOME/gautam_k   # 2026-10-17: box-independent (2-node box: /home/operator/gautam_k; 2x box: /root/gautam_k)
export PATH=/usr/local/cuda/bin:$PATH
# 2026-10-16: several forward candidates against one base library in ONE GPU session (free clocks).
#  1. quick suite (correctness) per candidate; a failing candidate is dropped from the race
#  2. race: 12 rounds; each round runs every library once (dense B8 18/6 + packed real_rows_752, 1000 iterations each,
#     >= 4 s of GPU work), in an order that rotates every round (round r starts at library r mod K, odd rounds reversed)
#  3. ncu of each candidate's forward main kernels (dense + packed; diagnostic only, locked clock)
#     10-17: this box's ncu writes .ncu-repz (zstd), so the import takes whichever of .ncu-rep / .ncu-repz exists
# usage: bash ab_multi.sh <gpu> <run name> <base lib folder> <candidate lib folder> ...   (lib folders under ~/gautam_k/libs)
# 10-17 switches (environment): AB_QUICK=0 no quick suite (measurement builds that are wrong on purpose), AB_NCU=0 no ncu,
#   AB_ENV="K=V ..." extra settings for every race run (VARLEN_BENCH_DIR=fwd: forward only), AB_DENSE_ITERS / AB_PACKED_ITERS
#   (defaults 1000 / 2000 = >= 4 s of forward + backward; forward only needs ~4000 / 8000 for the same 4 s)
#   AB_ROUNDS (default 12), AB_QUICK_LIBS="vj2_X ..." quick suite only for these candidates (the others race unchecked)
AB_ROUNDS=${AB_ROUNDS:-12}; AB_QUICK_LIBS=${AB_QUICK_LIBS:-}
AB_QUICK=${AB_QUICK:-1}; AB_NCU=${AB_NCU:-1}; AB_ENV=${AB_ENV:-}; DI=${AB_DENSE_ITERS:-1000}; PI=${AB_PACKED_ITERS:-2000}
G=${1:?gpu}; RUN=${2:?run name}; shift 2; LIBS_ALL=("$@"); B=/root/gpu_board/gpuboard
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; NCU=/usr/local/cuda/bin/ncu; LD=$GK/libs
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/$RUN; mkdir -p $L; cd $TI
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
for n in "${LIBS_ALL[@]}"; do
  got=$(LD_LIBRARY_PATH=$LD/$n:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$LD/$n/libtensor.so" ] || { say "ABORT: pointed at $LD/$n, the binary loads '$got'"; exit 3; }
done
if [ -x $B ]; then $B claim $G --note "gautam: attention forward candidates race, ~15 min" --hours 1 || { echo "claim of GPU $G refused, not starting" | tee -a $L/status; exit 2; }; trap "$B release $G" EXIT; fi
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | $(for n in "${LIBS_ALL[@]}"; do echo -n "$n $(md5sum $LD/$n/libtensor.so | cut -c1-8)  "; done)"
run() { local n=$1 log=$2; shift 2; env LD_LIBRARY_PATH=$LD/$n:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB "$@" timeout 600 $TI/tva_a > $L/$log 2>&1; }
OK=("${LIBS_ALL[0]}")
for n in "${LIBS_ALL[@]:1}"; do
  [ $AB_QUICK = 0 ] && { OK+=("$n"); continue; }
  [ -n "$AB_QUICK_LIBS" ] && [[ " $AB_QUICK_LIBS " != *" $n "* ]] && { say "quick suite $n skipped (AB_QUICK_LIBS)"; OK+=("$n"); continue; }
  run $n quick_$n.log VARLEN_QUICK=1; rc=$?
  say "quick suite $n exit=$rc | $(grep -E 'checks passed' $L/quick_$n.log | head -1)"
  [ $rc = 0 ] && OK+=("$n") || say "DROPPED $n (quick suite failed)"
done
K=${#OK[@]}
for r in $(seq -f %02g 1 $AB_ROUNDS); do   # two digits always: multi_stats.py reads race_NN_
  ord=(); for i in $(seq 0 $((K-1))); do ord+=("${OK[$(( (10#$r + i) % K ))]}"); done
  [ $((10#$r % 2)) = 0 ] && ord=($(printf '%s\n' "${ord[@]}" | tac))
  for n in "${ord[@]}"; do
    run $n race_${r}_${n}_dense.log  VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=$DI $AB_ENV
    run $n race_${r}_${n}_packed.log VARLEN_ONLY=BENCH      VARLEN_CASE=real_rows_752  VARLEN_BENCH_ITERS=$PI VARLEN_BENCH_IMPL=ours_A_plan $AB_ENV
  done
  say "race round $r done (${ord[*]})"
done
[ $AB_NCU = 0 ] && OK_NCU=() || OK_NCU=("${OK[@]:1}")
for n in "${OK_NCU[@]}"; do
  for shape in dense packed; do
    if [ $shape = dense ]; then k='gqa_v62_sm90<.int.0>'; a="VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6"; else k='gqa_v62_sm90<.int.1>'; a="VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan"; fi
    timeout 600 $NCU --set full --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip 3 --launch-count 1 -o $L/ncu_${n}_$shape \
      env LD_LIBRARY_PATH=$LD/$n:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=1 $a $TI/tva_a > $L/ncu_${n}_$shape.out 2>&1
    CUDA_VISIBLE_DEVICES= $NCU --import $(ls $L/ncu_${n}_$shape.ncu-rep* | head -1) --csv --page raw > $L/ncu_${n}_$shape.raw.csv 2> /dev/null
    CUDA_VISIBLE_DEVICES= $NCU --import $(ls $L/ncu_${n}_$shape.ncu-rep* | head -1) --csv --page source --print-source sass > $L/ncu_${n}_$shape.source.csv 2> /dev/null
    say "ncu $n $shape | kernels profiled $(grep -c '==PROF== Profiling' $L/ncu_${n}_$shape.out)"
  done
done
say "MULTI DONE"
