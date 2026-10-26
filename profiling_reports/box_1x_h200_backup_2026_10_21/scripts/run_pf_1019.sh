#!/bin/bash
# 2026-10-19 (GPU 0, ~7 min): PF = SM3m + the producer loads a unit's first K / V tile BEFORE waiting for the Q buffer (FA3's
# order, mainloop :769 / :776 / :780; work doc 45.53). ncu PC sampling (SM3m dense): 1,498 cycles per unit at the wait for that
# tile (FA3: 50 at its first-K wait). Protocol model clean (deep/protocol_sim_pf.py); local SASS: producer order = FA3's, the
# consumer code unchanged (packed identical; dense: 3 setup instructions reordered, steady loop identical).
#  1. build PF, F3c back in the tree (CPU)
#  2. quick suite (296 checks), forward + backward race vs SM3m (8 rounds, A B / B A), ncu of PF (does that wait drop?)
# usage: setsid nohup bash /root/gautam_K/scripts/run_pf_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/pf_1019
bash $S/build_cands.sh PF F3c > /root/gautam_K/build_pf.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED ($(head -c 200 /root/gautam_K/build_pf.log))" >> $R/pf_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_pf.log)" >> $R/pf_1019/status
AB_ROUNDS=8 bash $S/ab_multi.sh 0 pf_1019 vj2_SM3m vj2_PF
cd $R && tar czf /root/gautam_K/pf_results.tgz pf_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED pf_results.tgz" >> $R/pf_1019/status
