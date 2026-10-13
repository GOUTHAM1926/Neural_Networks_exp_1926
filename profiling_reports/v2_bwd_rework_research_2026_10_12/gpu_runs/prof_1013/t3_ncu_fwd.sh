#!/bin/bash
# 2026-10-13 profiling round, test 3: ncu --set full, per-instruction (SASS), FORWARD, dense B8 18/6 T4096, GPU 2.
#  ours v62 (11th launch = the timed one) ; libraries FA3, FA3-det (same fwd kernel), FA4, cuDNN in one python run (BENCH_WARM=0, 1 round)
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; S=/home/operator/gautam_k/step6
NCU="/usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite"
export CUDA_VISIBLE_DEVICES=2
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/prof_1013/status; }
say "t3 start"
cd $TI && env USE_VJ2=1 LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 $NCU -k "regex:^gqa_v62_sm90\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_ours_fwd $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_ours_fwd.out 2>&1
say "t3 ours fwd exit=$?"
cd $S && env BENCH_WARM=0 BENCH_ITERS=1 timeout 900 $NCU --kernel-name-base demangled -k "regex:FlashAttnFwdSm90|FlashAttentionForwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_fprop" --launch-count 10 -o /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_fwd /home/operator/gautam_k/benv/bin/python $S/bench_step6_libs.py --dense dense_B8_H18x6 > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_fwd.out 2>&1
say "t3 libs fwd exit=$? | kernels captured: $(grep -c '==PROF== Profiling' /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_fwd.out)"
for r in ncu_ours_fwd ncu_libs_fwd; do
  R=$(ls /home/operator/gautam_k/v2_work/logs/prof_1013/$r.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/prof_1013/$r.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/$r.import.err
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page details > /home/operator/gautam_k/v2_work/logs/prof_1013/$r.details.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/$r.import.err
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/prof_1013/$r.source.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/$r.import.err
  say "t3 $r exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/$r.raw.csv) lines, source $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/$r.source.csv) lines"
done
say "T3 DONE"
