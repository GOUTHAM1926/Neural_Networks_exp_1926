#!/bin/bash
# 2026-10-17 (2x box, GPU 0): build K5e / K7 / K7e (+ F3c back in the tree), quick suite on all three, race vs K5q (8 rounds, free
# clocks), then the results packed into /root/gautam_k/k7_results.tgz. K5e = K5 with split 8; K7 = K5q + FA4's conditional rescale
# (O rescaled only when a row max grows by > 8 in log2 units); K7e = split 8 + conditional rescale.
# usage: nohup bash /root/gautam_k/scripts/run_k7.sh > /dev/null 2>&1 &
S=/root/gautam_k/scripts; R=/root/gautam_k/sft_v2/runs
bash $S/build_cands.sh K5e K7 K7e F3c > /root/gautam_k/build_k7.log 2>&1 || exit 1
AB_QUICK_LIBS="vj2_K5e vj2_K7 vj2_K7e" AB_ROUNDS=8 AB_NCU=0 bash $S/ab_multi.sh 0 k7_1017 vj2_K5q vj2_K5e vj2_K7 vj2_K7e
cd $R && tar czf /root/gautam_k/k7_results.tgz k7_1017 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED k7_results.tgz" >> $R/k7_1017/status
