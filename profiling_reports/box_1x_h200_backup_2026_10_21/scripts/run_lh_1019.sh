#!/bin/bash
# 2026-10-19 (GPU 0): LH / LHO = K5q + FA3's L2 cache hints (work doc 45.49). LH: Q loads evict-first, K / V loads
# evict-last, exactly FA3's (mainloop_fwd_sm90_tma_gmma_ws.hpp:753 / 767 / 820); LHO: LH + the O stores evict-first. Same
# values, only the L2 policy changes -> results bit-identical to K5q expected. ~17 min:
#  1. build LH LHO, then F3c back in the tree (CPU)
#  2. quick suite (296 checks) on each, forward-only race vs K5q (8 rounds, 3 libraries, rotating order), ncu of each
#  3. clock / power check: K5q SKV LH LHO and FA3, dense + packed forwards, ~16 s each, mirrored order (clock_power_check.sh)
#  4. everything packed into /root/gautam_K/lh_results.tgz
# usage: setsid nohup bash /root/gautam_K/scripts/run_lh_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/lh_1019
bash $S/build_cands.sh LH LHO F3c > /root/gautam_K/build_lh.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED ($(head -c 200 /root/gautam_K/build_lh.log))" >> $R/lh_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_lh.log)" >> $R/lh_1019/status
AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000 AB_PACKED_ITERS=8000 bash $S/ab_multi.sh 0 lh_1019 vj2_K5q vj2_LH vj2_LHO
bash $S/clock_power_check.sh 0 lh_1019 vj2_K5q vj2_SKV vj2_LH vj2_LHO fa3
cd $R && tar czf /root/gautam_K/lh_results.tgz lh_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED lh_results.tgz" >> $R/lh_1019/status
