#!/bin/bash
# 2026-10-17 (2x box, GPU 0): what limits the dense forward loop step? (results of the KO libraries are WRONG on purpose)
#  1. build KOexp / KOmax / KOres / KOall (K5q minus the exps / the row max / the O rescale / all three) + TLK5q (K5q + per-step
#     timeline) + F3c back in the tree
#  2. FORWARD-ONLY race (VARLEN_BENCH_DIR=fwd: no backward on the wrong outputs), K5q vs the 4 knock-outs, 6 rounds,
#     4000 dense / 8000 packed forwards per run (>= 4 s), no quick suite (the knock-outs fail it by design)
#  3. the timeline trace of K5q (normal forward + backward bench; correct output), dense + packed
#  4. everything packed into /root/gautam_K/ko_results.tgz
# usage: nohup bash /root/gautam_K/scripts/run_ko.sh > /dev/null 2>&1 &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs
bash $S/build_cands.sh KOexp KOmax KOres KOall TLK5q F3c > /root/gautam_K/build_ko.log 2>&1 || exit 1
AB_QUICK=0 AB_NCU=0 AB_ROUNDS=6 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 \
  bash $S/ab_multi.sh 0 ko_1017 vj2_K5q vj2_KOexp vj2_KOmax vj2_KOres vj2_KOall
bash $S/trace_1017.sh 0 vj2_TLK5q
cd $R && tar czf /root/gautam_K/ko_results.tgz ko_1017 trace_1017_vj2_TLK5q && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED ko_results.tgz" >> $R/ko_1017/status
