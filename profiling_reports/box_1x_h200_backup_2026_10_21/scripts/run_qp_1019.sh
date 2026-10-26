#!/bin/bash
# 2026-10-19 (8x node, GPU = first argument, default 4 = ours today; ~20 min of GPU). The GPU part of the last forward
# candidates (work doc 45.58), after build_qp_1019.sh has passed its SASS gate: every fix measured on its own, in ONE session:
#   QP   = PF + FA3's L2 prefetch of Q when the producer starts a pair (FA3 mainloop :681)   [steady loops = PF's]
#   QPU  = QP + the dense unit decoded by the producer (no division in the consumers)       [dense loop reordered: 143 instr]
#   QPD  = QP + the diagonal tile first, as FA3 (mainloop :1150)                             [loops reordered: 57 dense, 23 packed]
#   QPUD = all three                                                                         [132 dense, 23 packed]
# Quick suite per candidate (296 checks; a failing one is dropped), forward + backward race vs PF (8 rounds, rotating order),
# ncu of every candidate AND of PF (same box, same day); everything into ~/gautam_k/qp_results.tgz.
# usage: setsid nohup bash ~/gautam_k/scripts/run_qp_1019.sh 4 > /dev/null 2>&1 < /dev/null &
G=${1:-4}; GK=$HOME/gautam_k; S=$GK/scripts; R=$GK/sft_v2/runs; LB=$GK/libs; L=$R/qp_1019
TI=$GK/sft_v2/BluTrain_sft/Tensor-Implementations; NCU=/usr/local/cuda/bin/ncu; LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
export PATH=/usr/local/cuda/bin:$PATH
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $L/status; }
grep -q "GATE PASSED" $L/status 2> /dev/null || { echo "build_qp_1019.sh has not passed its SASS gate: not starting"; exit 4; }
(cd $LB && md5sum -c --quiet $L/gate_libs.md5) || { say "ABORT: the libraries changed since the gate"; exit 4; }
say "GPU part start | GPU $G"
AB_ROUNDS=8 bash $S/ab_multi.sh $G qp_1019 vj2_PF vj2_QP vj2_QPU vj2_QPD vj2_QPUD
export CUDA_VISIBLE_DEVICES=$G
n=vj2_PF                                     # the base too (ab_multi profiles only the candidates)
for shape in dense packed; do
  if [ $shape = dense ]; then k='gqa_v62_sm90<.int.0>'; a="VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6"; else k='gqa_v62_sm90<.int.1>'; a="VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_IMPL=ours_A_plan"; fi
  timeout 600 $NCU --set full --force-overwrite --kernel-name-base demangled -k "regex:$k" --launch-skip 3 --launch-count 1 -o $L/ncu_${n}_$shape \
    env LD_LIBRARY_PATH=$LB/$n:$TI/../Profiler/lib USE_VJ2=1 VARLEN_STAGE0_LABELS=$LAB VARLEN_BENCH_ITERS=1 $a $TI/tva_a > $L/ncu_${n}_$shape.out 2>&1
  CUDA_VISIBLE_DEVICES= $NCU --import $(ls $L/ncu_${n}_$shape.ncu-rep* | head -1) --csv --page raw > $L/ncu_${n}_$shape.raw.csv 2> /dev/null
  CUDA_VISIBLE_DEVICES= $NCU --import $(ls $L/ncu_${n}_$shape.ncu-rep* | head -1) --csv --page source --print-source sass > $L/ncu_${n}_$shape.source.csv 2> /dev/null
  say "ncu $n $shape | kernels profiled $(grep -c '==PROF== Profiling' $L/ncu_${n}_$shape.out)"
done
cd $R && tar czf $GK/qp_results.tgz qp_1019 && say "PACKED qp_results.tgz"
