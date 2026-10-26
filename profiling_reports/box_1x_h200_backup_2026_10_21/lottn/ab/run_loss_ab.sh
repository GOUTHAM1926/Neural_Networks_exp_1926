#!/bin/bash
# 2026-10-21 (1x H200 root@86.38.238.55, its only GPU = 0): the LOSS changes of dist_side_gautam_lottn, OLD vs NEW library.
#  1. build test_loss_pretrain_ab.cu once (NEW tree's headers; only API both libraries have; no rpath) -- CPU, ~1 min
#  2. run it OLD, NEW, OLD, NEW for bf16 logits (what the production YAML runs: mixed_precision.bf16_logits) and fp32 logits
#     (the trainer's built-in default without that YAML section), the same
#     binary each time, only LD_LIBRARY_PATH changes; ldd must show the expected libtensor.so or the run aborts
#  3. summary: kernel F / B medians, autograd backward time and allocator peaks, and whether the loss bits and all
#     gradient checksums are equal across the 4 runs (= bit-identical)
# ~3 min. Needs both libraries built (lottn/old and lottn/new: make profiler tensor dist) and the val shard.
# usage: setsid nohup bash ~/gautam_k/lottn/ab/run_loss_ab.sh > /dev/null 2>&1 < /dev/null &
GK=$HOME/gautam_k; AB=$GK/lottn; TOK=$GK/pt_data/edu/edufineweb_val_00000.bin
R=$AB/runs/loss_ab_$(TZ=Asia/Kolkata date +%m%d_%H%M); mkdir -p $R
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $R/status; }
for t in old new; do
  [ -f $AB/$t/Tensor-Implementations/lib/libtensor.so ] || { say "ABORT: $AB/$t has no built libtensor.so"; exit 3; }
done
TI=$AB/new/Tensor-Implementations
say "libs: old $(md5sum < $AB/old/Tensor-Implementations/lib/libtensor.so | cut -c1-8) new $(md5sum < $TI/lib/libtensor.so | cut -c1-8)"
cd $TI && nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
  -DWITH_MYBLAS -IBluBridge-BLAS/include -std=c++17 -Xcompiler="-fPIC" --use_fast_math -O3 --expt-relaxed-constexpr \
  -gencode arch=compute_90a,code=sm_90a -o $R/test_loss_pretrain_ab $AB/ab/test_loss_pretrain_ab.cu \
  -L/usr/local/cuda/lib64 -Llib -L../Profiler/lib -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml \
  -lprofiler -lblublas -lcuda > $R/build_test.log 2>&1 || { say "TEST BUILD FAILED (build_test.log)"; tail -5 $R/build_test.log; exit 1; }
say "test built"
busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { say "ABORT: GPU 0 is in use: $(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; exit 3; }
export CUDA_VISIBLE_DEVICES=0
say "start | $(nvidia-smi -i 0 --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader)"
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 200 > $R/clocks.csv 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
for run in old_r1 new_r1 old_r2 new_r2; do
  T=$AB/${run%_*}; LDP=$T/Tensor-Implementations/lib:$T/Profiler/lib
  got=$(LD_LIBRARY_PATH=$LDP ldd $R/test_loss_pretrain_ab | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$T/Tensor-Implementations/lib/libtensor.so" ] || { say "ABORT: $run loads '$got', not $T/Tensor-Implementations/lib/libtensor.so"; exit 3; }
  for mode in f32 bf16; do
    LD_LIBRARY_PATH=$LDP $R/test_loss_pretrain_ab $mode $TOK --warmup 50 --iters 200 > $R/${run}_$mode.log 2>&1
    say "$run $mode exit=$? | $(grep -o 'reps_identical=[01]' $R/${run}_$mode.log)"
  done
done
kill $CLK 2>/dev/null
python3 - $R > $R/summary.txt 2>&1 <<'EOF'
import re, sys
R = sys.argv[1]; runs = ["old_r1", "new_r1", "old_r2", "new_r2"]
def kv(line): return dict(re.findall(r"(\w+)=(\S+)", line))
for mode in ("f32", "bf16"):
    K, A = {}, {}
    for r in runs:
        for line in open(f"{R}/{r}_{mode}.log"):
            if line.startswith("KERNEL "): K[r] = kv(line)
            if line.startswith("AUTOGRAD "): A[r] = kv(line)
    if len(K) < 4 or len(A) < 4:
        print(f"== {mode}: MISSING RESULT LINES (runs with KERNEL {sorted(K)}, AUTOGRAD {sorted(A)})"); continue
    m = lambda D, side, k: (float(D[side + "_r1"][k]) + float(D[side + "_r2"][k])) / 2
    print(f"== {mode} logits (N = 32768 rows x V = 50304)")
    for k, name in (("F_med_ms", "kernel forward with stats"), ("B_med_ms", "kernel backward with stats"),
                    ("fwd_ms", "autograd loss forward"), ("bwd_ms", "autograd backward (loss + cast + lm_head GEMMs)")):
        D = K if k in K["old_r1"] else A
        o, n = m(D, "old", k), m(D, "new", k)
        print(f"  {name:48s} old {o:9.4f} ms  new {n:9.4f} ms  new/old {n / o:6.3f}"
              f"   (runs: {' '.join(D[r][k] for r in runs)})")
    for k in ("active_peak_MiB", "allocated_peak_MiB", "reserved_peak_MiB"):
        o, n = m(A, "old", k), m(A, "new", k)
        print(f"  {k:48s} old {o:9.1f}     new {n:9.1f}     new-old {n - o:+9.1f} MiB")
    for name, D, keys in (("kernel", K, ("loss_bits", "max_cs", "sum_cs", "grad_cs")),
                          ("autograd", A, ("loss_bits", "wgrad_cs", "xgrad_cs"))):
        for k in keys:
            vals = {D[r][k] for r in runs}
            print(f"  {name + ' ' + k:48s} {'BIT-IDENTICAL in all 4 runs' if len(vals) == 1 else 'DIFFERENT: ' + str(sorted(vals))}")
    print(f"  {'reps identical inside each run':48s} {' '.join(A[r]['reps_identical'] for r in runs)}")
EOF
say "summary written: $R/summary.txt"
cat $R/summary.txt
