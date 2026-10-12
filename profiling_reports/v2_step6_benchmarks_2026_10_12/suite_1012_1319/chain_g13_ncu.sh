#!/bin/bash
# 2026-10-12: (1) after G12 ends and GPU 7 has room, run G13; (2) after the whole suite + G13, the narrowed
# packed ncu capture (FA3 only) on an otherwise empty GPU 7. Status lines -> chain.status.
G=~/gautam_k/v2_work/logs/suite_1012_1319; ST=$G/chain.status; : > $ST
say() { echo "$(date +%H:%M:%S) $*" >> $ST; }
cd ~/gautam_k/v2_work/Tensor-Implementations
export LD_LIBRARY_PATH=$PWD/lib:$PWD/../Profiler/lib:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=7
G12P=$(awk '$1=="G12"{print $2}' $G/pids.txt); say "waiting for G12 pid $G12P"
while kill -0 $G12P 2>/dev/null; do sleep 15; done
while [ $(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits) -gt 100000 ]; do sleep 15; done
say "G13 start (GPU 7 memory $(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader))"
VARLEN_ONLY=G13 VARLEN_STAGE0_LABELS=/home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin timeout 5400 ./tva_g13 > $G/G13.log 2>&1 &
G13P=$!; echo "G13 $G13P" >> $G/pids.txt; wait $G13P; say "G13 exit=$?"
for p in $(awk '{print $2}' $G/pids.txt); do while kill -0 $p 2>/dev/null; do sleep 15; done; done
say "suite + G13 done; GPU 7 procs $(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | wc -l)"
[ $(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | wc -l) = 0 ] || { say "GPU 7 not empty: ncu skipped"; exit 0; }
cd ~/gautam_k/step6; O=$S/ncu_6c; O=~/gautam_k/step6/ncu_6c
BENCH_ONLY=fa3 BENCH_WARM=0 BENCH_ITERS=1 /usr/local/cuda/bin/ncu --set full --target-processes all --force-overwrite --kernel-name-base demangled -k "regex:FlashAttnFwdSm90|FlashAttnBwdSm90" --launch-count 8 -o $O/fa3_packed ~/gautam_k/benv/bin/python bench_step6_libs.py segments_1012.txt real_rows_752 > $O/fa3_packed.out 2>&1
say "ncu fa3_packed exit=$? rep $(stat -c %s $O/fa3_packed.ncu-repz 2>/dev/null) bytes"
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $O/fa3_packed.ncu-repz --csv --page raw > $O/fa3_packed.raw.csv 2> $O/fa3_packed.import.err
say "ALLDONE raw $(wc -l < $O/fa3_packed.raw.csv) lines"
