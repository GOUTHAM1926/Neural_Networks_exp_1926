#!/bin/bash
# 2026-10-19 (8x node, CPU ONLY, ~8 min, no GPU): build the last forward candidates QP / QPU / QPD / QPUD (work doc 45.58), F3c
# back in the tree, then the SASS gate on THIS build: QP's dense and packed steady loops must equal PF's and every candidate
# kernel must hold the 4 Q prefetches (UTMAPF). The libraries' md5s are recorded; run_qp_1019.sh (the GPU part) runs only after
# "GATE PASSED" and only on these exact libraries.
# usage: bash ~/gautam_k/scripts/build_qp_1019.sh
GK=$HOME/gautam_k; S=$GK/scripts; LB=$GK/libs; L=$GK/sft_v2/runs/qp_1019; mkdir -p $L $GK/tmp   # cuobjdump needs a writable TMPDIR
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
say "build start (CPU only) | QP QPU QPD QPUD, then F3c"
bash $S/build_cands.sh QP QPU QPD QPUD F3c > $GK/build_qp.log 2>&1 || { say "BUILD FAILED ($(tail -c 300 $GK/build_qp.log | tr '\n' ' '))"; exit 1; }
say "built: $(tr '\n' ' ' < $GK/build_qp.log)"
python3 $S/loop_sass.py $LB/vj2_PF/libtensor.so $LB/vj2_QP/libtensor.so $LB/vj2_QPU/libtensor.so $LB/vj2_QPD/libtensor.so $LB/vj2_QPUD/libtensor.so > $L/sass_gate.txt 2>&1
same=$(grep -E '^ +vj2_QP +kernel' $L/sass_gate.txt | grep -c ' 0 lines differ')
pref=$(grep -E '^ +vj2_QP[UD]* +kernel' $L/sass_gate.txt | grep -c 'UTMAPF 0/0/4')
say "SASS gate: QP loops equal to PF's in $same of 2 kernels; Q prefetch x4 in $pref of 8 candidate kernels | $(grep -E '^ +vj2_' $L/sass_gate.txt | awk '{print $1, "kernel " $3, "loop " $7, $9 " differ;"}' | tr '\n' ' ')"
if [ "$same" = 2 ] && [ "$pref" = 8 ]; then
  (cd $LB && md5sum vj2_PF/libtensor.so vj2_QP/libtensor.so vj2_QPU/libtensor.so vj2_QPD/libtensor.so vj2_QPUD/libtensor.so) > $L/gate_libs.md5
  say "GATE PASSED | libraries: $(awk '{print substr($2,1,index($2,"/")-1) " " substr($1,1,8)}' $L/gate_libs.md5 | tr '\n' ' ')"
else
  say "GATE FAILED (see $L/sass_gate.txt): do not run the GPU part"; exit 4
fi
