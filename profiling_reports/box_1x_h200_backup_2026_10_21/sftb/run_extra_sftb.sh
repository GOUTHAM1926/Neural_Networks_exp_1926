#!/bin/bash
# 2026-10-21 (1x H200, GPU 0), after run_all_sftb.sh: A3b = the trainer's IN-PLACE loss backward timed next to the
# out-of-place entry the other loss benches call (bench_sparse_ce_sft_v3.cu, NEW library only), on the padded and on the
# packed labels, with both gradients' checksums (must be bit-identical). ~5 min. Status lines go to sftb/run_status.
# usage: setsid nohup bash ~/gautam_k/sftb/run_extra_sftb.sh > /dev/null 2>&1 < /dev/null &
GK=$HOME/gautam_k; S=$GK/sftb; TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; PL=$TI/../Profiler/lib; NEW=$GK/libs/sft_new
PD=$GK/sft_data/stage0_ultrachat_5k_T4096; PK=$GK/sft_data/stage0_ultrachat_5k_T4096_packed
R=$S/runs; ST=$S/run_status; X=$S/bin; L=$R/A3b_loss_inplace
mkdir -p $L $GK/tmp
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $ST; }
pgrep -f "[r]un_all_sftb.sh" > /dev/null && { say "A3b ABORT: run_all_sftb.sh is still running"; exit 3; }
[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] || { say "A3b ABORT: GPU 0 busy: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; exit 2; }
nvcc -I$TI/include -I$TI/../Profiler/include -I$TI/../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
  -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
  -o $X/bench_ce_v3 $S/bench_sparse_ce_sft_v3.cu -L/usr/local/cuda/lib64 -L$NEW -L$PL \
  -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda > $L/build.log 2>&1 \
  || { say "A3b BUILD FAILED: $(grep -m2 -iE 'error|undefined' $L/build.log | tr '\n' ' ' | cut -c1-250)"; exit 1; }
got=$(LD_LIBRARY_PATH=$NEW:$PL ldd $X/bench_ce_v3 | awk '/libtensor.so/ {print $3}')
[ "$got" = "$NEW/libtensor.so" ] || { say "A3b ABORT: bench loads '$got', not $NEW/libtensor.so"; exit 3; }
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > $L/clocks.csv 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
for set in padded packed; do
  lab=$PD/train/labels.bin; [ $set = packed ] && lab=$PK/train/labels.bin
  LD_LIBRARY_PATH=$NEW:$PL $X/bench_ce_v3 $lab --warmup 2000 --iters 500 > $L/$set.log 2>&1
  say "A3b $set exit=$? | $(grep -c '=> PASS' $L/$set.log) of 6 cases PASS | in-place gradient bit-identical in $(grep -c 'gradient=BIT-IDENTICAL' $L/$set.log) of $(grep -c '^INPLACE' $L/$set.log) cases"
done
kill $CLK 2>/dev/null
cd $S && tar czf $GK/sftb_results.tgz --exclude='*.nsys-rep' --exclude='*.sqlite' --exclude='*_cuda_gpu_trace.csv' runs run_status 2>/dev/null
say "A3b DONE | results re-packed $GK/sftb_results.tgz"
