#!/bin/bash
# 2026-10-19 (GPU 0): LHOd = LHO's L2 hints in the DENSE kernel only; the packed kernel stays K5q's (work doc 45.50).
# Local nvcc 13.0: LHOd's dense kernel SASS == LHO's, its packed kernel SASS == K5q's (0 differing lines each). ~8 min:
#  1. build LHOd, then F3c back in the tree (CPU)
#  2. the same SASS check on THIS build (CUDA 13.4): dense kernel vs vj2_LHO, packed kernel vs vj2_K5q (cuobjdump, CPU)
#  3. quick suite (296 checks), then a FORWARD + BACKWARD race vs K5q (8 rounds, A B / B A, the scoreboard's regime: the
#     backward runs between the forwards, so the clock is set by the mix, as in training); no ncu (both kernels measured)
#  4. everything packed into /root/gautam_K/lhod_results.tgz
# usage: setsid nohup bash /root/gautam_K/scripts/run_lhod_1019.sh > /dev/null 2>&1 < /dev/null &
export PATH=/usr/local/cuda/bin:$PATH
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; LB=/root/gautam_K/libs; L=$R/lhod_1019; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $L/status; }
bash $S/build_cands.sh LHOd F3c > /root/gautam_K/build_lhod.log 2>&1 || { say "BUILD FAILED ($(head -c 200 /root/gautam_K/build_lhod.log))"; exit 1; }
say "built: $(tr '\n' ' ' < /root/gautam_K/build_lhod.log)"
FN0=_ZN9OwnTensor4cuda5gqa6212gqa_v62_sm90ILi0EEEv14CUtensorMap_stS3_S3_PK13__nv_bfloat16PS4_PfiiiifPK4int4i
FN1=_ZN9OwnTensor4cuda5gqa6212gqa_v62_sm90ILi1EEEv14CUtensorMap_stS3_S3_PK13__nv_bfloat16PS4_PfiiiifPK4int4i
sass() { cuobjdump -sass -fun $2 $LB/$1/libtensor.so 2> /dev/null | sed -E 's#/\*[0-9a-f]{4,6}\*/##; s#/\* 0x[0-9a-f]+ \*/##' | grep -E '^[[:space:]]+[A-Z@{]'; }
sass vj2_LHOd $FN0 > $L/sass_LHOd_dense.txt; sass vj2_LHO $FN0 > $L/sass_LHO_dense.txt
sass vj2_LHOd $FN1 > $L/sass_LHOd_packed.txt; sass vj2_K5q $FN1 > $L/sass_K5q_packed.txt
n=$(cat $L/sass_LHOd_dense.txt $L/sass_LHO_dense.txt $L/sass_LHOd_packed.txt $L/sass_K5q_packed.txt | wc -l)
d0=$(diff $L/sass_LHOd_dense.txt $L/sass_LHO_dense.txt | grep -c '^[<>]'); d1=$(diff $L/sass_LHOd_packed.txt $L/sass_K5q_packed.txt | grep -c '^[<>]')
say "SASS check: dense LHOd vs LHO $d0 differing lines ($(wc -l < $L/sass_LHOd_dense.txt) lines), packed LHOd vs K5q $d1 ($(wc -l < $L/sass_LHOd_packed.txt) lines)"
[ $(wc -l < $L/sass_LHOd_dense.txt) -gt 1000 ] && [ $(wc -l < $L/sass_LHOd_packed.txt) -gt 1000 ] || { say "ABORT: SASS dump empty or short (kernel name not found?), total $n lines"; exit 4; }
AB_ROUNDS=8 AB_NCU=0 bash $S/ab_multi.sh 0 lhod_1019 vj2_K5q vj2_LHOd
cd $R && tar czf /root/gautam_K/lhod_results.tgz lhod_1019 && say "PACKED lhod_results.tgz"
