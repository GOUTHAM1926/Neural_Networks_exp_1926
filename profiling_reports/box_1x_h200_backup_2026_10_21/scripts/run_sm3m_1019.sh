#!/bin/bash
# 2026-10-19 (GPU 0, ~9 min): SM3m = SM3 + FA3's causal-mask order (work doc 45.52). SM3's packed forward lost 0.45%: PC
# sampling showed its diagonal tiles building the mask as a bitmask (64 LOP3, +28k cycle-equivalents per SM, packed ncu).
# Local SASS: SM3m has 0 such LOP3, packed kernel 3,432 instructions (LHOd 3,504, SM3 3,688), dense 3,672 (LHOd 3,736), the
# steady loop's exp spacing unchanged (0 / 13 / 22 / 7 / 23), 168 registers, 0 spills.
#  1. build SM3m, F3c back in the tree (CPU)
#  2. quick suite on SM3m only (296 checks incl. packed == alone bit-exact; SM3 passed at 15:53)
#  3. forward + backward race LHOd / SM3 / SM3m (8 rounds, rotating order: the training-like regime), ncu of SM3 and SM3m
# usage: setsid nohup bash /root/gautam_K/scripts/run_sm3m_1019.sh > /dev/null 2>&1 < /dev/null &
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; mkdir -p $R/sm3m_1019
bash $S/build_cands.sh SM3m F3c > /root/gautam_K/build_sm3m.log 2>&1 || { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST BUILD FAILED ($(head -c 200 /root/gautam_K/build_sm3m.log))" >> $R/sm3m_1019/status; exit 1; }
echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST built: $(tr '\n' ' ' < /root/gautam_K/build_sm3m.log)" >> $R/sm3m_1019/status
AB_ROUNDS=8 AB_QUICK_LIBS="vj2_SM3m" bash $S/ab_multi.sh 0 sm3m_1019 vj2_LHOd vj2_SM3 vj2_SM3m
cd $R && tar czf /root/gautam_K/sm3m_results.tgz sm3m_1019 && echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST PACKED sm3m_results.tgz" >> $R/sm3m_1019/status
