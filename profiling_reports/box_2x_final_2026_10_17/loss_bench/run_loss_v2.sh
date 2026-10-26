#!/bin/bash
# 2026-10-17 (2x box, GPU 0 = ours by default): the loss BACKWARD fix (LossKernels.cu sparseCENormalize_from_stats: real
# 16-byte loads / stores for bf16 -- they were eight 2-byte accesses each -- and 4 loads in flight per thread), old vs new.
#  1. build the library from the tree (the new LossKernels.cu must already be copied in; CPU only) -> libs/loss_v2/
#  2. build bench_sparse_ce_sft_v2.cu (= the 10-09 bench + a checksum of the trainer pair's gradient per case)
#  3. run it 4 times on the same GPU: OLD library (libs/vj2_K5q), NEW, OLD, NEW -- same binary, only LD_LIBRARY_PATH changes
#  4. summary: F / B / F+B per case and run, and whether the gradient checksums of OLD and NEW are equal (= bit-identical)
# ~15 min (build ~5, 4 runs x ~2.3). usage: nohup bash /root/gautam_k/loss_bench/run_loss_v2.sh [gpu, default 0] > /dev/null 2>&1 &
G=${1:-0}; GK=/root/gautam_k; TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LB=$GK/loss_bench
OLD=$GK/libs/vj2_K5q; NEW=$GK/libs/loss_v2; LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
R=$GK/sft_v2/runs; L=$R/loss_v2_1017; mkdir -p $L
export PATH=/usr/local/cuda/bin:$PATH
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
F=$TI/src/Kernels/cuda/loss/LossKernels.cu
grep -q "loss_ld16" $F || { say "ABORT: the tree's LossKernels.cu is the OLD one (copy the new file first)"; exit 3; }
say "tree LossKernels.cu $(md5sum $F | cut -c1-8) | GQA_fwd_sm90.cu $(md5sum $TI/src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu | cut -c1-8)"
cd $TI && make -j$(nproc) > $L/build_lib.log 2>&1 || { say "LIBRARY BUILD FAILED (see build_lib.log)"; tail -5 $L/build_lib.log; exit 1; }
mkdir -p $NEW && cp lib/libtensor.so $NEW/ && say "library built: $NEW/libtensor.so $(md5sum $NEW/libtensor.so | cut -c1-8) (old $(md5sum $OLD/libtensor.so | cut -c1-8))"
nvcc -I$TI/include -I$TI/../Profiler/include -I$TI/../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
  -std=c++17 -Xcompiler="-fPIC" --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
  -o $LB/bench_sparse_ce_sft_v2 $LB/bench_sparse_ce_sft_v2.cu -L/usr/local/cuda/lib64 -L$NEW -L$TI/../Profiler/lib \
  -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lcuda > $L/build_bench.log 2>&1 \
  || { say "BENCH BUILD FAILED (see build_bench.log)"; tail -5 $L/build_bench.log; exit 1; }
export CUDA_VISIBLE_DEVICES=$G
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader)"
TZ=Asia/Kolkata nvidia-smi -i $G --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 200 > $L/clocks.csv 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
for run in old_r1 new_r1 old_r2 new_r2; do
  lib=$OLD; [ "${run%_*}" = new ] && lib=$NEW
  got=$(LD_LIBRARY_PATH=$lib:$TI/../Profiler/lib ldd $LB/bench_sparse_ce_sft_v2 | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$lib/libtensor.so" ] || { say "ABORT: $run loads '$got', not $lib/libtensor.so"; exit 3; }
  LD_LIBRARY_PATH=$lib:$TI/../Profiler/lib $LB/bench_sparse_ce_sft_v2 $LAB --warmup 2000 --iters 500 > $L/$run.log 2>&1
  say "$run exit=$? | $(grep -c '=> PASS' $L/$run.log) of 6 cases PASS | lib $(md5sum $lib/libtensor.so | cut -c1-8)"
done
kill $CLK 2>/dev/null
python3 - $L > $L/summary.txt 2>&1 <<'EOF'
import re, sys
L = sys.argv[1]; runs = ["old_r1", "new_r1", "old_r2", "new_r2"]; d = {}
for r in runs:
    cur = None
    for line in open(f"{L}/{r}.log"):
        m = re.match(r"== (\w+)\s", line)
        if m: cur = m.group(1); d.setdefault(cur, {})[r] = {}; continue
        if not cur: continue
        for k, pat in (("F", r"\s+F   fwd save_stats\s+median\s+([0-9.]+)"), ("B", r"\s+B   bwd with stats\s+median\s+([0-9.]+)"),
                       ("FB", r"\s+F\+B trainer pair\s+median\s+([0-9.]+)"), ("cs", r"\s+grad checksum: ([0-9a-f]+)")):
            m = re.match(pat, line)
            if m: d[cur][r][k] = m.group(1)
print(f"{'case':12s} | {'B old r1/r2':>17s} {'B new r1/r2':>17s} {'B new/old':>9s} | {'F+B old':>8s} {'F+B new':>8s} {'new/old':>7s} | gradient")
for c, v in d.items():
    f = lambda r, k: float(v[r][k])
    bo = (f("old_r1", "B") + f("old_r2", "B")) / 2; bn = (f("new_r1", "B") + f("new_r2", "B")) / 2
    fo = (f("old_r1", "FB") + f("old_r2", "FB")) / 2; fn = (f("new_r1", "FB") + f("new_r2", "FB")) / 2
    cs = {v[r]["cs"] for r in runs}
    print(f"{c:12s} | {f('old_r1','B'):8.4f}/{f('old_r2','B'):8.4f} {f('new_r1','B'):8.4f}/{f('new_r2','B'):8.4f} {bn / bo:9.3f} | "
          f"{fo:8.4f} {fn:8.4f} {fn / fo:7.3f} | {'BIT-IDENTICAL (all 4 checksums equal)' if len(cs) == 1 else 'DIFFERENT: ' + str(cs)}")
EOF
say "summary written ($(grep -c BIT-IDENTICAL $L/summary.txt) of 6 cases bit-identical)"
cd $R && tar czf $GK/loss_v2_results.tgz loss_v2_1017 && say "PACKED loss_v2_results.tgz"
