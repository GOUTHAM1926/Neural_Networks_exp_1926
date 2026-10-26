#!/bin/bash
# 2026-10-19 (GPU 0, ~10 min): PFT = PF + FA3's dense O store (stmatrix in the TMA layout -> fence -> warps 1-3 arrive, warp 0
# waits, one lane issues 2 TMA stores and waits for their smem read before releasing the K slot; work doc 45.55). ncu PC
# sampling: our last P.V + epilogue 2,442 cycles per unit vs FA3's 2,059 (memory +239, fences +125). Local SASS (nvcc 13.0):
# dense 2 UTMASTG, 0 LDS.128 / STG.E.128 (PF 8 + 8), 24 fewer instructions, no COLLECTIVE fallbacks; packed kernel = PF's
# except the parameter offsets (PFT has one more tensor map); TMA_O 0 = PF exactly.
#  1. build PFT, then F3c back in the tree (CPU)
#  2. the same SASS check on THIS build (CUDA 13.4, cuobjdump, CPU); stops if it fails
#  3. quick suite (296 checks), forward + backward race vs PF (8 rounds, A B / B A), ncu of PFT (dense + packed)
#  4. everything packed into /root/gautam_K/pft_results.tgz
# usage: setsid nohup bash /root/gautam_K/scripts/run_pft_1019.sh > /dev/null 2>&1 < /dev/null &
export PATH=/usr/local/cuda/bin:$PATH
S=/root/gautam_K/scripts; R=/root/gautam_K/sft_v2/runs; LB=/root/gautam_K/libs; L=$R/pft_1019; mkdir -p $L
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $L/status; }
bash $S/build_cands.sh PFT F3c > /root/gautam_K/build_pft.log 2>&1 || { say "BUILD FAILED ($(head -c 200 /root/gautam_K/build_pft.log))"; exit 1; }
say "built: $(tr '\n' ' ' < /root/gautam_K/build_pft.log)"
P3=_ZN9OwnTensor4cuda5gqa6212gqa_v62_sm90ILi%dEEEv14CUtensorMap_stS3_S3_PK13__nv_bfloat16PS4_PfiiiifPK4int4i      # PF: 3 maps
P4=_ZN9OwnTensor4cuda5gqa6212gqa_v62_sm90ILi%dEEEv14CUtensorMap_stS3_S3_S3_PK13__nv_bfloat16PS4_PfiiiifPK4int4i   # PFT: 4 maps
sass() { cuobjdump -sass -fun $(printf $2 $3) $LB/$1/libtensor.so 2> /dev/null | sed -E 's#/\*[0-9a-f]{4,6}\*/##; s#/\* 0x[0-9a-f]+ \*/##' | grep -E '^[[:space:]]+[A-Z@{]' | grep -v 'Function :' | sed -E 's/c\[0x0\]\[0x[0-9a-f]+\]/c[P]/g; s/ +$//'; }
sass vj2_PF $P3 0 > $L/sass_PF_dense.txt; sass vj2_PFT $P4 0 > $L/sass_PFT_dense.txt
sass vj2_PF $P3 1 > $L/sass_PF_packed.txt; sass vj2_PFT $P4 1 > $L/sass_PFT_packed.txt
c() { grep -c "$1" $2; }
dp=$(diff $L/sass_PF_packed.txt $L/sass_PFT_packed.txt | grep -c '^[<>]')
say "SASS check: dense PF $(wc -l < $L/sass_PF_dense.txt) instr (UTMASTG $(c UTMASTG $L/sass_PF_dense.txt), STG.E.128 $(c STG.E.128 $L/sass_PF_dense.txt), LDS.128 $(c LDS.128 $L/sass_PF_dense.txt)) | dense PFT $(wc -l < $L/sass_PFT_dense.txt) instr (UTMASTG $(c UTMASTG $L/sass_PFT_dense.txt), STG.E.128 $(c STG.E.128 $L/sass_PFT_dense.txt), LDS.128 $(c LDS.128 $L/sass_PFT_dense.txt), COLLECTIVE $(c COLLECTIVE $L/sass_PFT_dense.txt)) | packed PFT vs PF $dp differing lines ($(wc -l < $L/sass_PFT_packed.txt) instr, parameter offsets ignored)"
[ $(wc -l < $L/sass_PFT_dense.txt) -gt 1000 ] && [ $(wc -l < $L/sass_PF_packed.txt) -gt 1000 ] || { say "ABORT: SASS dump empty or short (kernel name not found?)"; exit 4; }
[ $(c UTMASTG $L/sass_PFT_dense.txt) = 2 ] && [ $(c STG.E.128 $L/sass_PFT_dense.txt) = 0 ] && [ $(c COLLECTIVE $L/sass_PFT_dense.txt) = 0 ] && [ $dp = 0 ] || { say "ABORT: SASS check failed (expected dense UTMASTG 2, STG.E.128 0, COLLECTIVE 0; packed 0 differing lines)"; exit 4; }
AB_ROUNDS=8 bash $S/ab_multi.sh 0 pft_1019 vj2_PF vj2_PFT
cd $R && tar czf /root/gautam_K/pft_results.tgz pft_1019 && say "PACKED pft_results.tgz"
