#!/bin/bash
# 2026-10-19 21:40 (GPU = 1st argument, default 4; LIBRARY = 2nd, default vj2_QPU): the FULL test suite (test_varlen_attention
# without VARLEN_QUICK) on one library folder under ~/gautam_k/libs; made from run_pf_full_1019.sh. Was (for PF): the best forward so far
# (only its quick suite, 296 checks, has run: pf_1019). The last full run was 605/605 in 1,301 s (10-13, vj2c_1013/3, an older
# test binary). The library under test is libs/vj2_PF (built by run_pf_1019.sh, 6653f9f2); the tree's own build is not used.
# 10-19 20:00: 8x node, ONE folder: ~/gautam_k is a link to /localstorage/gautam_k. Expect 1-2 h there (its quick suite ran
# ~5x slower than on Verda: the CPU references on shared CPUs), so the time limit is 4 h.
# usage: setsid nohup bash ~/gautam_k/scripts/run_full_suite.sh 4 vj2_QPU > /dev/null 2>&1 < /dev/null &   (folder: runs/full_<lib>_<MMDD>)
export PATH=/usr/local/cuda/bin:$PATH
G=${1:-4}; N=${2:-vj2_QPU}; GK=$HOME/gautam_k; TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LD=$GK/libs; R=$GK/sft_v2/runs/full_${N}_$(date +%m%d); mkdir -p $R
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $R/status; }
got=$(LD_LIBRARY_PATH=$LD/$N:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
[ "$got" = "$LD/$N/libtensor.so" ] || { say "ABORT: the test binary loads '$got', not $LD/$N/libtensor.so"; exit 3; }
B=/root/gpu_board/gpuboard
if [ -x $B ]; then $B claim $G --note "gautam: attention full test suite, up to 2 h" --hours 3 || { say "claim of GPU $G refused, not starting"; exit 2; }; trap "$B release $G" EXIT; fi
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | $N $(md5sum < $LD/$N/libtensor.so | cut -c1-8) | tva_a $(md5sum < $TI/tva_a | cut -c1-8)"
cd $TI && CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$LD/$N:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB timeout 14400 ./tva_a > $R/full_$N.log 2>&1
rc=$?
say "full suite exit=$rc | $(grep -E 'checks passed' $R/full_$N.log | tail -1) | FAIL lines: $(grep -c FAIL $R/full_$N.log)"
cd $GK/sft_v2/runs && tar czf $GK/full_${N}_results.tgz $(basename $R) && say "PACKED full_${N}_results.tgz"
