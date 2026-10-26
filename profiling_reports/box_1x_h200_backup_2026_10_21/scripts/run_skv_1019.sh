#!/bin/bash
# 2026-10-19 (GPU 0): SKV = K5q + FA3's split K / V pipeline (work doc 45.48): build, quick suite (296 checks incl. packed ==
# alone bit-exact; a failing candidate is dropped), forward-only race vs K5q (8 rounds, A B / B A), ncu of SKV. ~6 min.
# Expected: dense about equal (the K/V wait is slack there: KOkv -0.30%), packed up to ~-1.8% (KOkv's bound).
# usage: setsid nohup bash /root/gautam_K/scripts/run_skv_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/skv_1019
bash $S/build_cands.sh SKV F3c > /root/gautam_K/build_skv.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED ($(head -c 200 /root/gautam_K/build_skv.log))" >> $R/skv_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_skv.log)" >> $R/skv_1019/status
AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 bash $S/ab_multi.sh 0 skv_1019 vj2_K5q vj2_SKV
cd $R && tar czf /root/gautam_K/skv_results.tgz skv_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED skv_results.tgz" >> $R/skv_1019/status
