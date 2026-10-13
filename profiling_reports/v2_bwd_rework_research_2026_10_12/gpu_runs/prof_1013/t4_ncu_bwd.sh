#!/bin/bash
# 2026-10-13 profiling round, test 4: ncu --set full, per-instruction (SASS), BACKWARD main kernels of FA3, FA3-det, FA4, cuDNN,
# dense B8 18/6 T4096, one python run (BENCH_WARM=0, 1 round), GPU 2. Ours (final Vj2 f5) was profiled at 17:11 (vj2c_1013/f5/ncu_f5.*).
S=/home/operator/gautam_k/step6
NCU="/usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite"
export CUDA_VISIBLE_DEVICES=2
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/prof_1013/status; }
say "t4 start"
cd $S && env BENCH_WARM=0 BENCH_ITERS=1 timeout 1500 $NCU --kernel-name-base demangled -k "regex:FlashAttnBwdSm90|FlashAttentionBackwardSm90|cudnn_generated_fort_native_sdpa_sm90_flash_bprop" --launch-count 10 -o /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd /home/operator/gautam_k/benv/bin/python $S/bench_step6_libs.py --dense dense_B8_H18x6 > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.out 2>&1
say "t4 libs bwd exit=$? | kernels captured: $(grep -c '==PROF== Profiling' /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.out)"
R=$(ls /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.ncu-rep* 2>/dev/null | head -1)
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page details > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.details.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.source.csv 2>> /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.import.err
say "t4 exports: raw $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.raw.csv) lines, source $(grep -c . /home/operator/gautam_k/v2_work/logs/prof_1013/ncu_libs_bwd.source.csv) lines"
say "T4 DONE"
