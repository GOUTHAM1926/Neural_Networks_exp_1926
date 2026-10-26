#!/bin/bash
# 2026-10-19 (GPU 0, ~9 min): SM3 = LHOd + FA3's softmax structure (work doc 45.52). Local SASS (= box SASS for this measure,
# checked on LHOd and FA3): exps issued 0 / 1 / 2 / 3-4 / 5+ instructions after the previous exp 0 / 13 / 22 / 7 / 23 (LHOd
# 4 / 21 / 13 / 5 / 22, FA3 0 / 11 / 19 / 13 / 22); same instruction counts, 168 registers, 0 spills.
#  1. build SM3, F3c back in the tree (CPU)
#  2. quick suite (296 checks incl. packed == alone bit-exact), forward-only race vs LHOd (8 rounds, A B / B A), ncu of SM3
#     (PC sampling: do the exps' "wait" stalls drop as predicted?)
#  3. forward + backward race vs LHOd (8 rounds, the scoreboard's regime), no ncu
# usage: setsid nohup bash /root/gautam_K/scripts/run_sm3_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/sm3_1019
bash $S/build_cands.sh SM3 F3c > /root/gautam_K/build_sm3.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED ($(head -c 200 /root/gautam_K/build_sm3.log))" >> $R/sm3_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_sm3.log)" >> $R/sm3_1019/status
AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 bash $S/ab_multi.sh 0 sm3_1019 vj2_LHOd vj2_SM3
AB_ROUNDS=8 AB_NCU=0 AB_QUICK=0 bash $S/ab_multi.sh 0 sm3fb_1019 vj2_LHOd vj2_SM3
cd $R && tar czf /root/gautam_K/sm3_results.tgz sm3_1019 sm3fb_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED sm3_results.tgz" >> $R/sm3_1019/status
