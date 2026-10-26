#!/bin/bash
# 2026-10-17 (2x box, GPU 0): build K5 / K5q / K6 / TRK5 (+ F3c back in the tree), quick suite on K5 / K5q / K6, race vs K4b
# (8 rounds, free clocks), the K5 trace, then everything packed into /root/gautam_K/k5_results.tgz (one small file to copy).
# K5 = K4b + half softmax split + ONE P register set (0 spills); K5q = quarter split; K6 = K5 + packed: next unit's descriptor
# loaded ahead. usage: nohup bash /root/gautam_K/scripts/run_k5.sh > /dev/null 2>&1 &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs
bash $S/build_cands.sh K5 K5q K6 TRK5 F3c > /root/gautam_K/build_k5.log 2>&1 || exit 1
AB_QUICK_LIBS="vj2_K5 vj2_K5q vj2_K6" AB_ROUNDS=8 AB_NCU=0 bash $S/ab_multi.sh 0 k5_1017 vj2_K4b vj2_K5 vj2_K5q vj2_K6
bash $S/trace_1017.sh 0 vj2_TRK5
cd $R && tar czf /root/gautam_K/k5_results.tgz k5_1017 trace_1017_vj2_TRK5 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED k5_results.tgz" >> $R/k5_1017/status
