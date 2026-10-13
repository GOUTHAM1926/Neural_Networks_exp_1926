#!/bin/bash
# 2026-10-13 profiling round, test 5: ncu --set full per-instruction, PACKED forward on real_rows_752 (typical real chats), GPU 2.
#  ours = design A (gqa_v62_sm90<2>, matched by mangled name ILi2E; 11th launch = the timed one); FA3 varlen (BENCH_ONLY=fa3_varlen, 1 round)
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; S=/home/operator/gautam_k/step6
NCU="/usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite"
export CUDA_VISIBLE_DEVICES=2
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/prof_1013/status; }
say "t5 start"
cd $TI && env LD_LIBRARY_PATH=$SN/f5/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCH VARLEN_CASE=real_rows_752 VARLEN_BENCH_ITERS=1 VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 600 $NCU --kernel-name-base mangled -k "regex:gqa_v62_sm90ILi2E" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_ours_packed_fwd $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_ours_packed_fwd.out 2>&1
say "t5 ours A fwd exit=$? | captured $(grep -c '==PROF== Profiling' /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_ours_packed_fwd.out)"
cd $S && env BENCH_ONLY=fa3_varlen BENCH_WARM=0 BENCH_ITERS=1 timeout 900 $NCU --kernel-name-base demangled -k "regex:FlashAttnFwdSm90" --launch-count 2 -o /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_fa3_packed_fwd /home/operator/gautam_k/benv/bin/python $S/bench_step6_libs.py $S/segments_1012.txt real_rows_752 > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_fa3_packed_fwd.out 2>&1
say "t5 FA3 varlen fwd exit=$? | captured $(grep -c '==PROF== Profiling' /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_fa3_packed_fwd.out)"
for r in ncu_ours_packed_fwd ncu_fa3_packed_fwd; do
  R=$(ls /home/operator/gautam_k/v2_work/logs/prof_1013/$r.ncu-rep* 2>/dev/null | head -1)
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/prof_1013/$r.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/$r.import.err
  CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/prof_1013/$r.source.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/$r.import.err
  say "t5 $r exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/$r.raw.csv) lines, source $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/$r.source.csv) lines"
done
say "T5 DONE"
