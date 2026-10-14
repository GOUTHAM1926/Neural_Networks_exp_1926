#!/bin/bash
# 2026-10-14 B-2a test 3 (re-run, ncu part only). The b2ali line-info build was done at 12:01 (kernel SASS identical to b2a,
# 1760/1760); that first run was killed during ncu at the user's request. GPU 0 (ours from 10-14 14:13): ncu --set full on gqa_bwd_vj2_kernel
# (11th launch, dense B8 18/6) with snapshot b2ali; export raw + source (sass and cuda). Same as test 1.4 (p1_1014/t14_lineinfo_ncu.sh part b).
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; A=src/Kernels/cuda/attention/arch
export PATH=/usr/local/cuda/bin:$PATH
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/b2a_1014/status; }
say "t3b start | snapshot b2ali lib $(md5sum $SN/b2ali/lib/libtensor.so | cut -c1-8)"
n=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -c .); m=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$n" = 0 ] && [ "$m" -lt 1024 ] || { say "STOP: GPU 0 busy ($n procs, $m MiB) before ncu"; say "T3 DONE"; exit 1; }
env CUDA_VISIBLE_DEVICES=0 USE_VJ2=1 LD_LIBRARY_PATH=$SN/b2ali/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 /usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.out 2>&1
say "t3b: ncu exit=$?"
R=$(ls /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.ncu-rep* 2>/dev/null | head -1)
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_sass.csv 2>> /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source cuda > /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_cuda.csv 2>> /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source cuda,sass > /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_cuda_sass.csv 2>> /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.import.err
say "t3b: exports raw $(grep -c . /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.raw.csv) sass $(grep -c . /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_sass.csv) cuda $(grep -c . /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_cuda.csv) cuda+sass $(grep -c . /home/operator/gautam_k/v2_work/logs/b2a_1014/ncu_b2ali.source_cuda_sass.csv) lines"
say "T3 DONE"
