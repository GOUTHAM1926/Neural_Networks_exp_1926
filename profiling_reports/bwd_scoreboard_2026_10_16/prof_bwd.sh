#!/bin/bash
# 2026-10-16 backward profiling (Vj2 vs FA3 / FA4 / cuDNN), one GPU of the shared 2x H200 box.
#  Part A, nsys (free clocks, as in training) + an nvidia-smi log of SM clock / power every 100 ms during each run:
#    where the time of one backward call goes (main kernel + helper kernels), and the clock each contender runs at.
#  Part B, ncu --set full (ncu fixes the clock): the backward main kernels, per instruction.
#  Shapes: dense B8 18/6 T4096 (our model) and packed real_rows_752 (typical SFT rows, the seqlens plan).
#  GPU board: claims the GPU first (exit 2 = refused -> nothing starts), releases it at the end, also after a crash.
# usage: bash prof_bwd.sh <gpu> [all|ncu_ours]   (all: about 15-20 min; ncu_ours: only our two ncu steps, about 2 min)
G=${1:?gpu}; P=${2:-all}; B=/root/gpu_board/gpuboard
TI=/root/gautam_k/sft_v2/BluTrain_sft/Tensor-Implementations; S=/root/gautam_k/step6
PY=/home/operator/gautam_k/benv/bin/python; NSYS=/usr/local/cuda/bin/nsys; NCU=/usr/local/cuda/bin/ncu
LAB=/root/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=/root/gautam_k/sft_v2/runs/prof_1016; mkdir -p $L; cd /root/gautam_k
$B claim $G --note "gautam: attention backward nsys+ncu profiling, ~20 min" --hours 1 || { echo "claim of GPU $G refused, not starting" | tee -a $L/status; exit 2; }
trap "$B release $G" EXIT
export CUDA_VISIBLE_DEVICES=$G
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | lib $(md5sum $TI/lib/libtensor.so | cut -c1-8)"
OURS="LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=50"
clk_on()  { TZ=Asia/Kolkata nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 100 > $L/$1.clocks.csv 2>&1 & CLK=$!; }
clk_off() { kill $CLK 2>/dev/null; wait $CLK 2>/dev/null; }
nsys_run() {   # name, then the command (run from its own directory)
  local n=$1; shift
  clk_on $n
  timeout 600 $NSYS profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $L/$n "$@" > $L/$n.out 2>&1; local rc=$?
  clk_off
  for rep in cuda_gpu_kern_sum cuda_gpu_trace; do
    CUDA_VISIBLE_DEVICES= $NSYS stats --report $rep --format csv --force-export=true --force-overwrite=true --output $L/$n $L/$n.nsys-rep > $L/$n.$rep.out 2>&1
  done
  say "nsys $n exit=$rc | kernels $(grep -c . $L/${n}_cuda_gpu_kern_sum.csv 2>/dev/null) lines | clock samples $(grep -c MHz $L/$n.clocks.csv)"
}
# ---- Part A: nsys + clock log
if [ "$P" = all ]; then
cd $TI && nsys_run ours_dense  env $OURS VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 $TI/tva_a
cd $TI && nsys_run ours_packed env $OURS VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 $TI/tva_a
cd $S  && nsys_run libs_dense  env BENCH_ONLY=fa3,fa4,sdpa_cudnn $PY $S/bench_step6_libs.py --dense dense_B8_H18x6
cd $S  && nsys_run libs_packed env BENCH_ONLY=fa3_varlen,fa4_varlen,cudnn_varlen_gqa CUDNN_GQA=1 $PY $S/bench_step6_libs.py $S/segments_1012.txt real_rows_752
fi
# ---- Part B: ncu, one launch of each backward main kernel after the warm-up
ncu_run() {    # name, kernel regex, launch-skip, launch-count, then the command
  local n=$1 k=$2 sk=$3 lc=$4; shift 4
  timeout 1500 $NCU --set full --target-processes all --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip $sk --launch-count $lc -o $L/ncu_$n "$@" > $L/ncu_$n.out 2>&1; local rc=$?
  local R=$(ls $L/ncu_$n.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page raw > $L/ncu_$n.raw.csv 2>> $L/ncu_$n.import.err
  CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page details > $L/ncu_$n.details.csv 2>> $L/ncu_$n.import.err
  CUDA_VISIBLE_DEVICES= $NCU --import $R --csv --page source --print-source sass > $L/ncu_$n.source.csv 2>> $L/ncu_$n.import.err
  say "ncu $n exit=$rc | kernels profiled $(grep -c '==PROF== Profiling' $L/ncu_$n.out) | raw $(grep -c . $L/ncu_$n.raw.csv) lines"
}
# ncu writes a bool template argument as (bool)0 / (bool)1, not false / true ('.' matches the brackets)
cd $TI && ncu_run ours_dense  'gqa_bwd_vj2_kernel<.bool.0>' 3 1 env $OURS VARLEN_BENCH_ITERS=1 VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 $TI/tva_a
cd $TI && ncu_run ours_packed 'gqa_bwd_vj2_kernel<.bool.1>' 3 1 env $OURS VARLEN_BENCH_ITERS=1 VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 $TI/tva_a
[ "$P" = all ] || { say "PROF DONE ($P)"; exit 0; }
cd $S  && ncu_run libs_dense  'FlashAttnBwdSm90|FlashAttentionBackwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_bprop' 0 10 env BENCH_WARM=0 BENCH_ITERS=1 BENCH_ONLY=fa3,fa4,sdpa_cudnn $PY $S/bench_step6_libs.py --dense dense_B8_H18x6
cd $S  && ncu_run libs_packed 'FlashAttnBwdSm90|FlashAttentionBackwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_bprop' 0 10 env BENCH_WARM=0 BENCH_ITERS=1 BENCH_ONLY=fa3_varlen,fa4_varlen,cudnn_varlen_gqa CUDNN_GQA=1 $PY $S/bench_step6_libs.py $S/segments_1012.txt real_rows_752
say "PROF DONE"
