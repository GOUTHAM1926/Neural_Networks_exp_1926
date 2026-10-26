#!/bin/bash
# 2026-10-19 (GPU 0): KNOCK-OUT of group 0's in-loop K/V wait (fwd_KOkv.cu, MEASUREMENT ONLY: its results are wrong on purpose).
# Question (work doc 45.47): the per-step timeline shows group 0 waiting ~299 cycles for tile n's K/V at every step start; FA3
# never waits there (separate K / V pipelines, K released right after QK). If the step does NOT get faster without any wait, the
# wait is slack and FA3's K/V pipeline would not help us; if it does, port FA3's split pipeline. ~10 min:
#  1. build KOkv, then F3c back in the tree (CPU)
#  2. no quick suite (wrong on purpose); forward-only race vs K5q, 8 rounds, A B / B A order, 4000 dense / 8000 packed forwards;
#     ncu of KOkv (tensor busy)
# usage: setsid nohup bash /root/gautam_K/scripts/run_kokv_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/kokv_1019
bash $S/build_cands.sh KOkv F3c > /root/gautam_K/build_kokv.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED" >> $R/kokv_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_kokv.log)" >> $R/kokv_1019/status
AB_QUICK=0 AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 bash $S/ab_multi.sh 0 kokv_1019 vj2_K5q vj2_KOkv
cd $R && tar czf /root/gautam_K/kokv_results.tgz kokv_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED kokv_results.tgz" >> $R/kokv_1019/status
