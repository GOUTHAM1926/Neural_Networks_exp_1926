#!/bin/bash
# 2026-10-17 (2x box, GPU 0): the dense forward's step-start wait + the exp unit.
# Why: timeline TLK5q (ko_results): group 0's wait at the step start is ~100-150 or ~275-300 cycles and almost never between
# (dense: long on ~95% of the steps); FA3 / FA4 / cuDNN wait with CUTLASS's try_wait + suspend hint (SASS: NANOSLEEP.SYNCS)
# and FA3's in-loop K and V waits never retried (ncu 2026-10-13). Knock-outs: no exp -6.8%, no max -3.5..-4.1%, no rescale -2.2..-2.4%.
#  1. build W1 (K5q + CUTLASS's wait), W1s (K5q + test_wait spin), W2 (W1 + separate K / V barriers, V waited before P@V),
#     P8w / P4w (W1 + every 8th / 4th exp by the polynomial, masked keys exactly 0), TLW2 (W2 + timeline) + F3c back in the tree
#  2. quick suite (296 checks) on each candidate (a failing one is dropped), then a FORWARD-ONLY race vs K5q, 8 rounds,
#     4000 dense / 8000 packed forwards per run (>= 4 s)
#  3. the timeline trace of W2, dense + packed
#  4. everything packed into /root/gautam_k/w_results.tgz
# usage: nohup bash /root/gautam_k/scripts/run_w.sh > /dev/null 2>&1 &
S=/root/gautam_k/scripts; R=/root/gautam_k/sft_v2/runs
bash $S/build_cands.sh W1 W1s W2 P8w P4w TLW2 F3c > /root/gautam_k/build_w.log 2>&1 || exit 1
AB_NCU=0 AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 \
  bash $S/ab_multi.sh 0 w_1017 vj2_K5q vj2_W1 vj2_W1s vj2_W2 vj2_P8w vj2_P4w
bash $S/trace_1017.sh 0 vj2_TLW2
cd $R && tar czf /root/gautam_k/w_results.tgz w_1017 trace_1017_vj2_TLW2 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED w_results.tgz" >> $R/w_1017/status
