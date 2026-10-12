#!/bin/bash
# 2026-10-12: cuDNN packed (varlen) attention vs FA3 / FA4 and ours, on the Step 6 segments (segments_1012.txt).
# GPU 7 ONLY. Nothing here starts by itself: run one block by hand, only after the OK.
#   bash run_cudnn_varlen.sh check   CPU only (no GPU visible): self-test of the benchmark script, < 1 min
#   bash run_cudnn_varlen.sh main    GPU 7, ~5 min: (1) cuDNN probe in its own process, (2) timing of FA3 / FA4 /
#                                    cuDNN on all 7 layouts, (3) timing of ours (dense / B / A) on the same layouts
#   bash run_cudnn_varlen.sh nsys    GPU 7, ~3 min: kernel-level times on the real middle rows (real_rows_752-759),
#                                    FA3 + cuDNN (two captures, one NVTX range per contender) and ours
#   bash run_cudnn_varlen.sh nsys libs   the same without ours (~1.5 min)
# Before each GPU step: GPU 7 must show no process and < 1 GiB used, else the block stops without touching it.
# Output: ~/gautam_k/step6/cudnn_varlen_1012/ (a log per step, status.txt with one line per step).
set -u
MODE=${1:-}
S=~/gautam_k/step6; O=$S/cudnn_varlen_1012; mkdir -p $O; ST=$O/status.txt
PY=~/gautam_k/benv/bin/python
TI=~/gautam_k/v2_work/Tensor-Implementations
LBL=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
SEG=$S/segments_1012.txt
say() { echo "$(date +%H:%M:%S) $*" | tee -a $ST; }

# memory.used is device-wide (it also sees other containers' work); the process list may not, so both are checked
gpu7_free() {
  local n m u
  n=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | grep -c .)
  m=$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  u=$(nvidia-smi -i 7 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  say "GPU 7: $n process(es), $m MiB used, $u% busy"
  [ "$n" = 0 ] && [ "$m" -lt 1024 ]
}
pin_gpu7() {
  export CUDA_VISIBLE_DEVICES=7
  export EXPECT_GPU_UUID=$(nvidia-smi -i 7 --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  say "pinned CUDA_VISIBLE_DEVICES=7 ($EXPECT_GPU_UUID)"
}
ours_env() {
  cd $TI
  export LD_LIBRARY_PATH=$PWD/lib:$PWD/../Profiler/lib:${LD_LIBRARY_PATH:-}
  say "ours: libtensor.so $(md5sum lib/libtensor.so | cut -c1-8), tva_step6 $(md5sum tva_step6 | cut -c1-8)"
}

# --force-overwrite: without it nsys stats keeps an existing CSV ("SKIPPED: output file ... exists") and the
# numbers silently come from an older capture (happened on the first nsys re-run, 2026-10-12).
stats_all() {   # per-kernel-name summary for every report, plus the per-NVTX-range summary where ranges exist
  local r
  for r in "$@"; do
    [ -f $O/$r.nsys-rep ] || continue
    CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      --force-overwrite=true --output $O/$r $O/$r.nsys-rep > $O/$r.stats.out 2>&1
    CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/nsys stats --report nvtx_kern_sum --format csv --force-export=true \
      --force-overwrite=true --output $O/$r $O/$r.nsys-rep > $O/$r.nvtx.out 2>&1
    say "$r: rep $(stat -c %s $O/$r.nsys-rep) bytes, kern csv $(wc -l < $O/${r}_cuda_gpu_kern_sum.csv 2>/dev/null) lines ($(grep -c cudnn_generated $O/${r}_cuda_gpu_kern_sum.csv 2>/dev/null) cuDNN attention kernels), nvtx csv $(wc -l < $O/${r}_nvtx_kern_sum.csv 2>/dev/null) lines"
  done
}

case "$MODE" in
check)
  say "check: CPU-only self-test (no GPU visible)"
  cd $S
  CUDA_VISIBLE_DEVICES= $PY bench_step6_libs.py --selftest $SEG > $O/selftest.log 2>&1
  say "selftest exit=$? | $(tail -1 $O/selftest.log)"
  ;;
main)
  gpu7_free || { say "ABORT: GPU 7 not free, nothing run"; exit 1; }
  pin_gpu7
  cd $S
  say "1/3 probe start"
  timeout 600 $PY bench_step6_libs.py --cudnn-probe $SEG > $O/probe.log 2>&1
  say "1/3 probe exit=$? | $(grep PROBE_SUMMARY $O/probe.log)"
  GQA=0; grep -q "PROBE_SUMMARY.*gqa=PASS" $O/probe.log && GQA=1
  gpu7_free || { say "ABORT before 2/3: GPU 7 not free"; exit 1; }
  say "2/3 libs timing start (CUDNN_GQA=$GQA)"
  BENCH_ONLY=fa3,fa4,cudnn CUDNN_GQA=$GQA timeout 1200 $PY bench_step6_libs.py $SEG > $O/libs.log 2>&1
  say "2/3 libs exit=$? | $(grep -c '^BENCH ' $O/libs.log) BENCH lines, $(grep -c 'GATE.*-> PASS' $O/libs.log) gate PASS, $(grep -c 'INVALID' $O/libs.log) INVALID"
  gpu7_free || { say "ABORT before 3/3: GPU 7 not free"; exit 1; }
  ours_env
  say "3/3 ours timing start"
  VARLEN_ONLY=BENCH VARLEN_STAGE0_LABELS=$LBL VARLEN_BENCH_EXPORT=$O/segments_check.txt timeout 1200 ./tva_step6 > $O/ours.log 2>&1
  say "3/3 ours exit=$? | $(grep -c '^BENCH ' $O/ours.log) BENCH lines"
  if diff <(sort $SEG) <(sort $O/segments_check.txt) > /dev/null 2>&1; then
    say "segments: ours ran the SAME layouts as segments_1012.txt"
  else
    say "segments: DIFFERENT from segments_1012.txt or missing -- do not compare ours with the libs"
  fi
  say "main ALLDONE"
  ;;
nsys)
  gpu7_free || { say "ABORT: GPU 7 not free, nothing run"; exit 1; }
  pin_gpu7
  GQA=0; grep -q "PROBE_SUMMARY.*gqa=PASS" $O/probe.log 2>/dev/null && GQA=1
  NS="/usr/local/cuda/bin/nsys profile --sample=none --cpuctxsw=none --force-overwrite=true"   # --trace set per run
  cd $S
  # Two library captures, no capture range (a first try with --capture-range=cudaProfilerApi recorded no kernels):
  # (a) FA3 + cuDNN with K/V copied to 18 heads, (b) cuDNN with the 6 KV heads. Both cuDNN variants launch kernels
  # with the same names, so they are captured apart. The per-name summary of (b) still holds the gate's one
  # wrong-mapping call (18 heads) among ~17 cuDNN calls; the NVTX summary (timed calls only) is the exact number.
  nsys_lib() {   # $1 = report name, $2 = BENCH_ONLY, $3 = CUDNN_GQA
    say "nsys $1 start (BENCH_ONLY=$2 CUDNN_GQA=$3)"
    BENCH_ONLY=$2 CUDNN_GQA=$3 BENCH_ITERS=5 BENCH_NVTX=1 timeout 900 \
      $NS --trace=cuda,nvtx -o $O/$1 $PY bench_step6_libs.py $SEG real_rows_752 > $O/$1.out 2>&1
    say "nsys $1 exit=$? | BENCH lines: $(grep -c '^BENCH' $O/$1.out) total, $(grep -c '^BENCH.*impl=cudnn' $O/$1.out) cuDNN"
  }
  nsys_lib nsys_libs_exp fa3_varlen,cudnn_varlen_exp 0
  gpu7_free || { say "ABORT before nsys gqa: GPU 7 not free"; exit 1; }
  [ "$GQA" = 1 ] && nsys_lib nsys_libs_gqa cudnn_varlen_gqa 1
  [ "${2:-}" = libs ] && { stats_all nsys_libs_exp nsys_libs_gqa; say "nsys libs ALLDONE"; exit 0; }
  gpu7_free || { say "ABORT before nsys ours: GPU 7 not free"; exit 1; }
  ours_env
  say "nsys ours start"
  VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_ITERS=5 VARLEN_STAGE0_LABELS=$LBL timeout 900 \
    $NS --trace=cuda -o $O/nsys_ours_packed ./tva_step6 > $O/nsys_ours_packed.out 2>&1
  say "nsys ours exit=$?"
  stats_all nsys_libs_exp nsys_libs_gqa nsys_ours_packed
  say "nsys ALLDONE"
  ;;
stats)   # CPU only: re-export the summaries of the reports already on disk
  stats_all nsys_libs_exp nsys_libs_gqa nsys_ours_packed
  say "stats ALLDONE"
  ;;
*)
  echo "usage: bash run_cudnn_varlen.sh check|main|nsys [libs]|stats"; exit 2
  ;;
esac
