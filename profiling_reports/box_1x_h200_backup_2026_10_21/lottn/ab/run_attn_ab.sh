#!/bin/bash
# 2026-10-21 (1x H200 root@86.38.238.55, its only GPU = 0): the ATTENTION changes of dist_side_gautam_lottn.
#  1. bench_attn_ab.cu built against each tree (OLD's and NEW's calls have different signatures, so one binary per tree;
#     ldd must show the tree's own libtensor.so), run OLD, NEW, OLD, NEW at the production shape (B 8, 18/6 heads,
#     T 4096, hd 128): forward / backward time and output checksums. ~1 min.
#  2. our attention suite (ab/test_varlen_attention.cu = the laptop master's Tests/kernel_tests copy) built against NEW:
#     quick = VARLEN_QUICK=1 (the smoke subset, minutes); full = everything incl. the 1000-repeat race test at 8x4096 (the
#     QPU adoption gate; 605/605 on 10-13 in 1,301 s on Verda, longer where the CPU references run on fewer cores).
# usage: setsid nohup bash ~/gautam_k/lottn/ab/run_attn_ab.sh [quick|full] > /dev/null 2>&1 < /dev/null &
MODE=${1:-quick}; GK=$HOME/gautam_k; AB=$GK/lottn; LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
R=$AB/runs/attn_ab_${MODE}_$(TZ=Asia/Kolkata date +%m%d_%H%M); mkdir -p $R
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $R/status; }
case $MODE in quick|full) ;; *) say "ABORT: mode must be quick or full"; exit 3;; esac
[ -f $LAB ] || { say "ABORT: no $LAB (the suite's packed groups need it)"; exit 3; }
NVCC_FLAGS="-std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a"
LINK="-L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib -Llib -L../Profiler/lib -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda"   # as TI run-cuda
for t in old new; do
  TI=$AB/$t/Tensor-Implementations
  [ -f $TI/lib/libtensor.so ] || { say "ABORT: $TI has no built libtensor.so"; exit 3; }
  (cd $TI && nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
     $NVCC_FLAGS -o $R/bench_attn_$t $AB/ab/bench_attn_ab.cu $LINK) > $R/build_bench_$t.log 2>&1 \
    || { say "BENCH BUILD FAILED for $t: $(grep -m3 -iE 'error|undefined' $R/build_bench_$t.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
done
TI=$AB/new/Tensor-Implementations
(cd $TI && nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
   $NVCC_FLAGS -o $R/tva_a $AB/ab/test_varlen_attention.cu $LINK) > $R/build_tva.log 2>&1 \
  || { say "SUITE BUILD FAILED: $(grep -m3 -iE 'error|undefined' $R/build_tva.log | tr '\n' ' ' | cut -c1-300)"; exit 1; }
say "built | libs old $(md5sum < $AB/old/Tensor-Implementations/lib/libtensor.so | cut -c1-8) new $(md5sum < $TI/lib/libtensor.so | cut -c1-8) | suite source $(md5sum < $AB/ab/test_varlen_attention.cu | cut -c1-8)"
busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { say "ABORT: GPU 0 is in use: $(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; exit 3; }
export CUDA_VISIBLE_DEVICES=0
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms 500 > $R/clocks.csv 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
for run in old_r1 new_r1 old_r2 new_r2; do
  t=${run%_*}; T=$AB/$t; LDP=$T/Tensor-Implementations/lib:$T/Profiler/lib
  got=$(LD_LIBRARY_PATH=$LDP ldd $R/bench_attn_$t | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$T/Tensor-Implementations/lib/libtensor.so" ] || { say "ABORT: $run loads '$got'"; exit 3; }
  LD_LIBRARY_PATH=$LDP $R/bench_attn_$t --warmup 20 --iters 200 > $R/bench_$run.log 2>&1
  say "bench $run exit=$? | $(grep -oE 'F_med_ms=\S+|B_med_ms=\S+|repeat_\S+' $R/bench_$run.log | tr '\n' ' ')"
done
python3 - $R > $R/bench_summary.txt 2>&1 <<'EOF'
import re, sys
R = sys.argv[1]; runs = ["old_r1", "new_r1", "old_r2", "new_r2"]; d = {}
for r in runs:
    for line in open(f"{R}/bench_{r}.log"):
        if line.startswith("ATTN "): d[r] = dict(re.findall(r"(\w+)=(\S+)", line))
if len(d) < 4: print("MISSING ATTN lines:", sorted(d)); sys.exit(1)
m = lambda side, k: (float(d[side + "_r1"][k]) + float(d[side + "_r2"][k])) / 2
print("attention at B 8, 18/6 heads, T 4096, hd 128, causal (OLD = BLUCCL v62 fwd + Vj1d bwd, NEW = QPU fwd + Vj2 bwd)")
for k, name in (("F_med_ms", "forward"), ("B_med_ms", "backward")):
    o, n = m("old", k), m("new", k)
    print(f"  {name:9s} old {o:7.4f} ms  new {n:7.4f} ms  new/old {n / o:6.3f}  ({(o / n - 1) * 100:+.1f}% faster)"
          f"   runs: {' '.join(d[r][k] for r in runs)}")
fo, fn = m("old", "F_med_ms") + m("old", "B_med_ms"), m("new", "F_med_ms") + m("new", "B_med_ms")
print(f"  fwd+bwd   old {fo:7.4f} ms  new {fn:7.4f} ms  new/old {fn / fo:6.3f}  ({(fo / fn - 1) * 100:+.1f}% faster)")
for side in ("old", "new"):
    same = {k: d[side + "_r1"][k] == d[side + "_r2"][k] for k in ("o_cs", "lse_cs", "dk_cs", "dv_cs", "dq_cs")}
    print(f"  {side}: r1 vs r2 outputs equal: {same} | repeat inside each run (O/LSE/dK/dV, dQ): "
          f"{d[side + '_r1']['repeat_o_lse_dk_dv']}/{d[side + '_r1']['repeat_dq']}, {d[side + '_r2']['repeat_o_lse_dk_dv']}/{d[side + '_r2']['repeat_dq']}")
EOF
cat $R/bench_summary.txt | tee -a $R/status
got=$(LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib ldd $R/tva_a | awk '/libtensor.so/ {print $3}')
[ "$got" = "$TI/lib/libtensor.so" ] || { say "ABORT: the suite loads '$got'"; exit 3; }
Q=0; [ $MODE = quick ] && Q=1
say "suite ($MODE) start"
cd $TI && LD_LIBRARY_PATH=$TI/lib:$TI/../Profiler/lib VARLEN_QUICK=$Q VARLEN_STAGE0_LABELS=$LAB timeout 14400 $R/tva_a > $R/suite_$MODE.log 2>&1
say "suite ($MODE) exit=$? | $(grep -E 'checks passed' $R/suite_$MODE.log | tail -1) | FAIL lines: $(grep -c FAIL $R/suite_$MODE.log)"
