#!/bin/bash
# 2026-10-19 (new box, GPU 0): the ROW-SPLIT softmax candidates vs K5q (work doc 45.46). R8 / R16 / R32 = K5q (f35982af) with
# part A (the exps before the wait for P@V(n-1)) taken from row 0 only: 8 / 16 / 32 of its 32 scores, so row 1's max can run
# between them (FA3's schedule). Results expected bit-identical to K5q. ~20 min:
#  1. build R8 R16 R32, then F3c back in the tree (CPU, ~10 min)
#  2. quick suite (296 checks) on each candidate, then a FORWARD-ONLY race vs K5q, 8 rounds, 4000 dense / 8000 packed forwards per
#     run (>= 4 s), the order rotating every round (4 libraries), then ncu of each candidate (dense + packed)
#  3. everything packed into /root/gautam_K/r_results.tgz
# usage: setsid nohup bash /root/gautam_K/scripts/run_r_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/r_1019
bash $S/build_cands.sh R8 R16 R32 F3c > /root/gautam_K/build_r.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED (build_r.log)" >> $R/r_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_r.log)" >> $R/r_1019/status
AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 \
  bash $S/ab_multi.sh 0 r_1019 vj2_K5q vj2_R8 vj2_R16 vj2_R32
cd $R && tar czf /root/gautam_K/r_results.tgz r_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED r_results.tgz" >> $R/r_1019/status
