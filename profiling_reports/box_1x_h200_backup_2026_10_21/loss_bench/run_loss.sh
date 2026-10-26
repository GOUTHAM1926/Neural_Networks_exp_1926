#!/bin/bash
# 2026-10-17 (2x box, GPU 0 = ours; the default): the SFT loss -- OURS (forward save_stats + backward with stats, the trainer's pair)
# vs PyTorch eager (fp32 upcast, bf16), torch.compile, Liger, Apple CCE -- same shape, labels and logits (see loss_rivals_1017.py).
#  1. build ours' bench (bench_sparse_ce_sft.cu, the 2026-10-09 file, unchanged; CPU only) against a FIXED copy of libtensor.so
#     (libs/vj2_K5q: the same loss kernels as the tree; the tree's lib/ is rebuilt by the attention runs on GPU 0)
#  2. ours: 6 cases x (F, B, F+B, Fv, Fv+Bo), 2000 warm-up + 500 timed calls per pass (the long passes run >= 4 s)
#  3. the others: Part 1 (loss only, 6 cases) and Part 2 (lm_head + loss, 3 cases), >= 4 s warm-up per library and case
#  4. nvidia-smi clock / power / memory log every 200 ms; everything packed into /root/gautam_k/loss_results.tgz
# ~10 min. usage: nohup bash /root/gautam_k/loss_bench/run_loss.sh [gpu, default 0 = ours] > /dev/null 2>&1 &
G=${1:-0}; GK=/root/gautam_k; TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LB=$GK/loss_bench; LIB=$GK/libs/vj2_K5q
PY=$GK/lossenv/bin/python; LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin; R=$GK/sft_v2/runs; L=$R/loss_1017
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p $L && cd $LB || exit 1
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
[ -x $PY ] && [ -f $LAB ] && [ -f $LIB/libtensor.so ] && [ -f $LB/loss_rivals_1017.py ] || { say "ABORT: missing env / labels / library / script"; exit 3; }
nvcc -I$TI/include -I$TI/../Profiler/include -I$TI/../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
  -std=c++17 -Xcompiler="-fPIC" --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
  -o $LB/bench_sparse_ce_sft $LB/bench_sparse_ce_sft.cu -L/usr/local/cuda/lib64 -L$LIB -L$TI/../Profiler/lib \
  -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lcuda > $L/build.log 2>&1 \
  || { say "BUILD FAILED (see build.log)"; tail -5 $L/build.log; exit 1; }
export LD_LIBRARY_PATH=$LIB:$TI/../Profiler/lib
got=$(ldd $LB/bench_sparse_ce_sft | awk '/libtensor.so/ {print $3}')
[ "$got" = "$LIB/libtensor.so" ] || { say "ABORT: the bench loads '$got', not $LIB/libtensor.so"; exit 3; }
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader) | lib $(md5sum $LIB/libtensor.so | cut -c1-8) | bench $(md5sum $LB/bench_sparse_ce_sft.cu | cut -c1-8) | rivals $(md5sum $LB/loss_rivals_1017.py | cut -c1-8)"
TZ=Asia/Kolkata nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 200 > $L/clocks.csv 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
$LB/bench_sparse_ce_sft $LAB --warmup 2000 --iters 500 > $L/ours.log 2>&1
say "ours exit=$? | $(grep -c '=> PASS' $L/ours.log) of 6 cases PASS"
$PY $LB/loss_rivals_1017.py $LAB --ours-log $L/ours.log > $L/rivals.log 2>&1
say "rivals exit=$? | $(grep -c '^LOSSBENCH' $L/rivals.log) result lines | $(grep -c '^SKIP\|^FAIL' $L/rivals.log) skipped / failed"
kill $CLK 2>/dev/null
cd $R && tar czf $GK/loss_results.tgz loss_1017 && say "PACKED loss_results.tgz"
