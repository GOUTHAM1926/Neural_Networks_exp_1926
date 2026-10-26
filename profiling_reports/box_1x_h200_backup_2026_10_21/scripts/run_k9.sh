#!/bin/bash
# 2026-10-17 (2x box, GPU 0): build K8 / K9 / K9h (+ F3c back in the tree), quick suite on all three, race vs K5q (8 rounds,
# free clocks), results packed into /root/gautam_K/k9_results.tgz.
# K8 = per-mode best (dense: split 8 + conditional rescale = K7e's SASS; packed: split 16 = K5q's SASS).
# K9 / K9h = K8 + every 4th / every 2nd exp computed by a degree-3 polynomial on the FP32 units (FA4's trick; fewer MUFU).
# usage: nohup bash /root/gautam_K/scripts/run_k9.sh > /dev/null 2>&1 &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs
bash $S/build_cands.sh K8 K9 K9h F3c > /root/gautam_K/build_k9.log 2>&1 || exit 1
AB_QUICK_LIBS="vj2_K8 vj2_K9 vj2_K9h" AB_ROUNDS=8 AB_NCU=0 bash $S/ab_multi.sh 0 k9_1017 vj2_K5q vj2_K8 vj2_K9 vj2_K9h
cd $R && tar czf /root/gautam_K/k9_results.tgz k9_1017 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED k9_results.tgz" >> $R/k9_1017/status
