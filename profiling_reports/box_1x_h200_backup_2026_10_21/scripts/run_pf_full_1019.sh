#!/bin/bash
# 2026-10-19 (~25 min, GPU = the first argument, default 4 = ours on the 8x node 10-19): the FULL test suite (test_varlen_attention without VARLEN_QUICK) on PF, the best forward so far
# (only its quick suite, 296 checks, has run: pf_1019). The last full run was 605/605 in 1,301 s (10-13, vj2c_1013/3, an older
# test binary). The library under test is libs/vj2_PF (built by run_pf_1019.sh, 6653f9f2); the tree's own build is not used.
# 10-19 20:00: 8x node, ONE folder: ~/gautam_k is a link to /localstorage/gautam_k
# usage: setsid nohup bash ~/gautam_k/scripts/run_pf_full_1019.sh 4 > /dev/null 2>&1 < /dev/null &
export PATH=/usr/local/cuda/bin:$PATH
G=${1:-4}; GK=$HOME/gautam_k; TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; LD=$GK/libs; R=$GK/sft_v2/runs/pf_full_1019; mkdir -p $R
LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $R/status; }
got=$(LD_LIBRARY_PATH=$LD/vj2_PF:$TI/../Profiler/lib ldd $TI/tva_a | awk '/libtensor.so/ {print $3}')
[ "$got" = "$LD/vj2_PF/libtensor.so" ] || { say "ABORT: the test binary loads '$got', not $LD/vj2_PF/libtensor.so"; exit 3; }
B=/root/gpu_board/gpuboard
if [ -x $B ]; then $B claim $G --note "gautam: attention full test suite, ~25 min" --hours 1 || { say "claim of GPU $G refused, not starting"; exit 2; }; trap "$B release $G" EXIT; fi
say "start | GPU $G $(nvidia-smi -i $G --query-gpu=name,memory.used --format=csv,noheader) | vj2_PF $(md5sum < $LD/vj2_PF/libtensor.so | cut -c1-8) | tva_a $(md5sum < $TI/tva_a | cut -c1-8)"
cd $TI && CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$LD/vj2_PF:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB timeout 3600 ./tva_a > $R/full_PF.log 2>&1
rc=$?
say "full suite exit=$rc | $(grep -E 'checks passed' $R/full_PF.log | tail -1) | FAIL lines: $(grep -c FAIL $R/full_PF.log)"
cd $GK/sft_v2/runs && tar czf $GK/pf_full_results.tgz pf_full_1019 && say "PACKED pf_full_results.tgz"
