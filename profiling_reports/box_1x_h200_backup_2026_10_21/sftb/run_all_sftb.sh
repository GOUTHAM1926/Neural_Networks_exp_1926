#!/bin/bash
# 2026-10-21 (1x H200 root@86.38.238.55, GPU 0): the SFT kernel benchmarks for Kathirvel (PLAN.md), one GPU, in this order:
#   prep  libs/sft_new = the master library (libtensor + libblublas, built by setup_sftb.sh); tva_a into the tree
#         (scoreboard_free.sh runs $TI/tva_a); builds the loss benches (v1 = the rivals' reference, v2 = + gradient checksums),
#         the RoPE bench and the RoPE positions test (CPU only)
#   S1    attention quick suite (VARLEN_QUICK=1) on the master build: correctness before any timing
#   A3    loss: ours vs PyTorch eager fp32 / bf16, torch.compile, Liger (loss only); Liger FLCE, Apple CCE (lm_head + loss)
#   A5    RMSNorm + RoPE: test_rope_positions; ours nopos / identity / packed; PyTorch eager / compile, Liger
#   B1    attention on 7 layouts (3 sets of 8 real packed rows, 4 uniform): ours dense full-row vs ours packed (BENCH);
#         FA3 / FA3-det / FA4 / cuDNN / torch varlen / Flex / SDPA on the same segments (bench_step6_libs.py)
#   A1    attention scoreboard: ours vs FA3 / FA4 / cuDNN, dense B8 18/6 + packed real_rows_752, fwd + bwd, ABBA x 2, one
#         contender per process under nsys (scoreboard_free.sh), then the steady-state summary (scoreboard_steady.py)
#   A4    loss OLD (BLUCCL lib, before the loss work) vs NEW (master), SFT shape, old / new / old / new, gradient checksums
#   B2    loss NEW on the packed rows' labels (the same chats as A4's padded rows)
#   C     SFT runs, the 5,000 stage-0 chats from blumodel_step_8000, AdamW 2e-5 (trainer defaults: warmup 0.03, cosine to
#         2e-6, val every 10 steps), 2 epochs: P = padded (B 8, 64 rows per step = 64 chats, 156 steps), K = packed (B 4,
#         20 rows per step = ~66 chats, ~151 steps); GPU memory / power / clock sampled every second
# Before each step: GPU 0 must have no process (waits up to 30 min for someone else's job, then skips the step). Each step
# logs its exit code and the next one starts. Status: ~/gautam_k/sftb/run_status (last line "ALL DONE").
# usage: setsid nohup bash ~/gautam_k/sftb/run_all_sftb.sh ["S1 A3 ..." = a subset; default all] > /dev/null 2>&1 < /dev/null &
STEPS=${1:-"S1 A3 A5 B1 A1 A4 B2 C"}
GK=$HOME/gautam_k; S=$GK/sftb; M=$GK/sft_v2/BluTrain_sft; TI=$M/Tensor-Implementations; PL=$TI/../Profiler/lib
NEW=$GK/libs/sft_new; OLD=$GK/lottn/old/Tensor-Implementations/lib; BIN=$M/bluscript_sft_bin; INIT=$GK/ckpt/blumodel_step_8000
PD=$GK/sft_data/stage0_ultrachat_5k_T4096; PK=$GK/sft_data/stage0_ultrachat_5k_T4096_packed; LAB=$PD/train/labels.bin
LB=$GK/loss_bench; SC=$GK/scripts; S6=$GK/step6; PYL=$GK/lossenv/bin/python; PYB=$GK/benv/bin/python
R=$S/runs; ST=$S/run_status; X=$S/bin
mkdir -p $R $X $GK/tmp
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $ST; }
gpu_free() {   # 0 = GPU 0 has no process; waits up to 30 min
  local i
  for i in $(seq 60); do
    [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
    [ $i = 1 ] && say "GPU 0 busy: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' ') -- waiting up to 30 min"
    sleep 30
  done
  return 1
}
clk_on()  { TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used --format=csv -lms ${2:-500} > $1 2>&1 & CLK=$!; }
clk_off() { kill $CLK 2>/dev/null; wait $CLK 2>/dev/null; }
ld_is() {   # binary, library folder: the binary must load libtensor.so from that folder
  local got; got=$(LD_LIBRARY_PATH=$2:$PL ldd $1 | awk '/libtensor.so/ {print $3}')
  [ "$got" = "$2/libtensor.so" ] || { say "ABORT step: $(basename $1) loads '$got', not $2/libtensor.so"; return 1; }
}

# ---- prep (CPU) ------------------------------------------------------------------------------------------------------
say "run start | steps: $STEPS | GPU 0 $(nvidia-smi -i 0 --query-gpu=name,memory.used,power.limit,clocks.max.sm --format=csv,noheader) | driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
mkdir -p $NEW && cp $TI/lib/libtensor.so $TI/lib/libblublas.so $NEW/ && cp $S/tva_a $TI/tva_a || { say "ABORT: copying the library / tva_a failed"; exit 3; }
say "libs/sft_new: libtensor $(md5sum < $NEW/libtensor.so | cut -c1-8) libblublas $(md5sum < $NEW/libblublas.so | cut -c1-8) | OLD (BLUCCL) libtensor $(md5sum < $OLD/libtensor.so | cut -c1-8) | tva_a $(md5sum < $TI/tva_a | cut -c1-8) | trainer $(md5sum < $BIN | cut -c1-8)"
F="-I$TI/include -I$TI/../Profiler/include -I$TI/../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a"
LK="-L/usr/local/cuda/lib64 -L$NEW -L$PL -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda"
nvcc $F -o $X/bench_ce_v1 $LB/bench_sparse_ce_sft.cu $LK > $X/build_ce_v1.log 2>&1 & p1=$!
nvcc $F -o $X/bench_ce_v2 $LB/bench_sparse_ce_sft_v2.cu $LK > $X/build_ce_v2.log 2>&1 & p2=$!
nvcc $F -o $X/bench_rope $S/bench_rope_sft.cu $LK > $X/build_rope.log 2>&1 & p3=$!
nvcc $F -o $X/test_rope_positions $TI/Tests/kernel_tests/test_rope_positions.cu $LK > $X/build_rope_test.log 2>&1 & p4=$!
for p in $p1 $p2 $p3 $p4; do wait $p; done
for b in bench_ce_v1 bench_ce_v2 bench_rope test_rope_positions; do
  [ -x $X/$b ] || say "BUILD FAILED: $b ($(grep -m2 -iE 'error|undefined' $X/build_*.log | tr '\n' ' ' | cut -c1-250))"
done
say "prep done | built: $(ls $X | grep -v '\.log$' | tr '\n' ' ')"

# ---- S1: attention quick suite ---------------------------------------------------------------------------------------
s_S1() {
  local L=$R/S1_quick_suite; mkdir -p $L
  ld_is $TI/tva_a $NEW || return
  clk_on $L/clocks.csv
  (cd $TI && LD_LIBRARY_PATH=$NEW:$PL VARLEN_QUICK=1 VARLEN_STAGE0_LABELS=$LAB timeout 3600 ./tva_a > $L/suite_quick.log 2>&1); local rc=$?
  clk_off
  say "S1 quick suite exit=$rc | $(grep -E 'backward mode' $L/suite_quick.log | head -1 | cut -c1-60) | $(grep -E 'checks passed' $L/suite_quick.log | tail -1) | FAIL lines $(grep -c FAIL $L/suite_quick.log)"
}

# ---- A3: loss vs the rivals ------------------------------------------------------------------------------------------
s_A3() {
  local L=$R/A3_loss_rivals; mkdir -p $L
  ld_is $X/bench_ce_v1 $NEW || return
  clk_on $L/clocks.csv 200
  LD_LIBRARY_PATH=$NEW:$PL $X/bench_ce_v1 $LAB --warmup 2000 --iters 500 > $L/ours.log 2>&1
  say "A3 ours exit=$? | $(grep -c '=> PASS' $L/ours.log) of 6 cases PASS | $(grep -E 'ALL CHECKS PASS|SOME CHECKS FAILED' $L/ours.log)"
  (cd $LB && timeout 3600 $PYL $LB/loss_rivals_1017.py $LAB --ours-log $L/ours.log > $L/rivals.log 2>&1); local rc=$?
  clk_off
  say "A3 rivals exit=$rc | $(grep -c '^LOSSBENCH' $L/rivals.log) result lines | $(grep -c '^SKIP\|^FAIL' $L/rivals.log) skipped / failed"
}

# ---- A5: RMSNorm + RoPE ----------------------------------------------------------------------------------------------
s_A5() {
  local L=$R/A5_rope; mkdir -p $L
  ld_is $X/bench_rope $NEW || return
  clk_on $L/clocks.csv 200
  LD_LIBRARY_PATH=$NEW:$PL timeout 1800 $X/test_rope_positions > $L/test_rope_positions.log 2>&1
  say "A5 test_rope_positions exit=$? | $(grep -iE 'passed|ALL|FAIL' $L/test_rope_positions.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
  LD_LIBRARY_PATH=$NEW:$PL timeout 1800 $X/bench_rope $PK/train --warmup 50 --iters 300 > $L/ours.log 2>&1
  say "A5 ours exit=$? | $(grep -c '^ROPE case=' $L/ours.log) cases | $(grep 'identity_vs_nopos' $L/ours.log)"
  timeout 3600 $PYL $S/bench_rope_rivals.py $PK/train --iters 300 --warm 4 > $L/rivals.log 2>&1; local rc=$?
  clk_off
  say "A5 rivals exit=$rc | $(grep -c '^ROPE impl=' $L/rivals.log) result lines | $(grep -c '^FAIL\|^SKIP' $L/rivals.log) failed / skipped"
}

# ---- B1: attention, padded full rows vs packed, 7 layouts, ours + the libraries ----------------------------------------
s_B1() {
  local L=$R/B1_attn_layouts; mkdir -p $L
  ld_is $TI/tva_a $NEW || return
  clk_on $L/clocks.csv
  (cd $TI && LD_LIBRARY_PATH=$NEW:$PL USE_VJ2=1 VARLEN_ONLY=BENCH VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=200 \
     VARLEN_BENCH_EXPORT=$L/segments.txt timeout 1800 ./tva_a > $L/ours.log 2>&1)
  say "B1 ours exit=$? | $(grep -c '^BENCH layout=' $L/ours.log) BENCH lines | segments vs segments_1012.txt: $(cmp -s $L/segments.txt $S6/segments_1012.txt && echo identical || echo DIFFERENT)"
  (cd $S6 && BENCH_ITERS=100 timeout 3600 $PYB $S6/bench_step6_libs.py $L/segments.txt > $L/rivals.log 2>&1); local rc=$?
  clk_off
  say "B1 rivals exit=$rc | $(grep -c '^BENCH layout=' $L/rivals.log) BENCH lines | $(grep -ciE 'error|traceback' $L/rivals.log) error lines"
}

# ---- A1: attention scoreboard (scoreboard_free.sh: its own status and folder) -------------------------------------------
s_A1() {
  local RUN=sb_sft_1021 L=$R/A1_scoreboard W=$GK/sft_v2/runs/sb_sft_1021; mkdir -p $L
  timeout 7200 bash $SC/scoreboard_free.sh 0 $RUN sft_new sft_new > $L/driver.log 2>&1; local rc=$?
  python3 $SC/scoreboard_steady.py $W > $W/summary_steady.txt 2>&1
  cp $W/status $W/summary.txt $W/summary_steady.txt $W/*.clocks.csv $W/*.out $L/ 2>/dev/null
  say "A1 scoreboard exit=$rc | $(grep -c 'exit=0' $W/status) of 16 runs exit 0 | summary $(grep -c . $W/summary.txt) lines, steady $(grep -c . $W/summary_steady.txt) lines"
}

# ---- A4: loss OLD vs NEW (same binary, the library switched) -----------------------------------------------------------
s_A4() {
  local L=$R/A4_loss_old_new run lib; mkdir -p $L
  clk_on $L/clocks.csv 200
  for run in old_r1 new_r1 old_r2 new_r2; do
    lib=$OLD; [ "${run%_*}" = new ] && lib=$NEW
    ld_is $X/bench_ce_v2 $lib || { clk_off; return; }
    LD_LIBRARY_PATH=$lib:$PL $X/bench_ce_v2 $LAB --warmup 2000 --iters 500 > $L/$run.log 2>&1
    say "A4 $run exit=$? | $(grep -c '=> PASS' $L/$run.log) of 6 cases PASS | lib $(md5sum < $lib/libtensor.so | cut -c1-8)"
  done
  clk_off
  python3 - $L > $L/summary.txt 2>&1 <<'EOF'
import re, sys
L = sys.argv[1]; runs = ["old_r1", "new_r1", "old_r2", "new_r2"]; d = {}
for r in runs:
    cur = None; pend = None
    for line in open(f"{L}/{r}.log"):
        m = re.match(r"\s+grad checksum: ([0-9a-f]+)", line)   # printed just BEFORE its case's '== name' header
        if m: pend = m.group(1); continue
        m = re.match(r"== (\w+)\s", line)
        if m: cur = m.group(1); d.setdefault(cur, {})[r] = {"cs": pend}; pend = None; continue
        if not cur: continue
        for k, pat in (("F", r"\s+F   fwd save_stats\s+median\s+([0-9.]+)"), ("B", r"\s+B   bwd with stats\s+median\s+([0-9.]+)"),
                       ("FB", r"\s+F\+B trainer pair\s+median\s+([0-9.]+)")):
            m = re.match(pat, line)
            if m: d[cur][r][k] = m.group(1)
print(f"{'case':12s} | {'F old':>7s} {'F new':>7s} {'new/old':>7s} | {'B old':>7s} {'B new':>7s} {'new/old':>7s} | "
      f"{'F+B old':>7s} {'F+B new':>7s} {'new/old':>7s} | gradient (4 checksums)")
for c, v in d.items():
    f = lambda r, k: float(v[r][k])
    m = lambda k, a: (f(a + "_r1", k) + f(a + "_r2", k)) / 2
    cs = {v[r]["cs"] for r in runs}
    print(f"{c:12s} | {m('F','old'):7.4f} {m('F','new'):7.4f} {m('F','new')/m('F','old'):7.3f} | {m('B','old'):7.4f} {m('B','new'):7.4f} "
          f"{m('B','new')/m('B','old'):7.3f} | {m('FB','old'):7.4f} {m('FB','new'):7.4f} {m('FB','new')/m('FB','old'):7.3f} | "
          f"{'BIT-IDENTICAL' if len(cs) == 1 else 'DIFFERENT: ' + str(cs)}")
print("ms, median of 500 calls, mean of the two runs per library; new/old < 1 = NEW faster")
EOF
  say "A4 summary: $(grep -c BIT-IDENTICAL $L/summary.txt) of 6 cases bit-identical"
}

# ---- B2: loss NEW on the packed labels ---------------------------------------------------------------------------------
s_B2() {
  local L=$R/B2_loss_packed; mkdir -p $L
  ld_is $X/bench_ce_v2 $NEW || return
  clk_on $L/clocks.csv 200
  LD_LIBRARY_PATH=$NEW:$PL $X/bench_ce_v2 $PK/train/labels.bin --warmup 2000 --iters 500 > $L/new_packed.log 2>&1; local rc=$?
  clk_off
  say "B2 exit=$rc | $(grep -c '=> PASS' $L/new_packed.log) of 6 cases PASS | $(grep -E 'labels +:' $L/new_packed.log | cut -c1-150)"
}

# ---- C: the SFT runs ---------------------------------------------------------------------------------------------------
sft_run() {   # name, data folder, rows per micro-batch, rows per optimizer step
  local n=$1 data=$2 b=$3 rps=$4 d=$R/C_$1 LDP rc
  mkdir -p $d && cd $d || return
  LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
  [ -n "$LDP" ] || { say "C $n ABORT: make print_run_ld_path printed nothing"; return; }
  gpu_free || { say "C $n SKIPPED: GPU 0 busy for 30 min"; return; }
  say "C $n start | data $(basename $data) | B $b | rows per step $rps | LD $LDP"
  TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv -l 1 > $d/gpu.csv 2>&1 &
  local MEM=$!
  ( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
    export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$data BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 \
           BLU_OPTIMIZER=adamw BLU_SFT_B=$b BLU_SFT_ROWS_PER_STEP=$rps BLU_SFT_EPOCHS=2
    timeout 7200 mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA -x BLU_INIT_FROM \
      -x BLU_ALL_BF16 -x BLU_CKPT -x BLU_OPTIMIZER -x BLU_SFT_B -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS $BIN < /dev/null > run.log 2>&1 )
  rc=$?
  kill $MEM 2>/dev/null; wait $MEM 2>/dev/null
  say "C $n exit=$rc | steps $(grep -c '^step ' run.log) | $(grep -E 'SftLoader' run.log | head -1 | cut -c1-110) | last $(grep '^step ' run.log | tail -1 | cut -c1-105) | $(grep '\[val\]' run.log | tail -1 | sed 's/^ *//') | peak GPU memory $(awk -F', ' 'NR>1 {v=$2+0; if (v>m) m=v} END {print m}' gpu.csv) MiB | $(grep -c 'Training complete' run.log) 'Training complete'"
}
s_C() {
  sft_run P_padded $PD 8 64
  sft_run K_packed $PK 4 20
}

for s in $STEPS; do
  gpu_free || { say "$s SKIPPED: GPU 0 busy for 30 min"; continue; }
  say "$s start"
  s_$s
done
cd $S && tar czf $GK/sftb_results.tgz --exclude='*.nsys-rep' --exclude='*.sqlite' --exclude='*_cuda_gpu_trace.csv' runs run_status 2>/dev/null
say "ALL DONE | results $GK/sftb_results.tgz ($(du -h $GK/sftb_results.tgz | cut -f1))"
