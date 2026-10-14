#!/bin/bash
# 2026-10-14 Phase 1 test 1.4: per-C++-line stall map of the Vj2 main kernel (g123), GPU 7.
#  a) CPU: rebuild ONLY GQA_bwd_vj2_sm90.cu with -lineinfo (instruction -> source line map; code unchanged), snapshot g123li,
#     check the kernel SASS is identical to the normal build, then restore the normal library.
#  b) GPU 7: ncu --set full on gqa_bwd_vj2_kernel (11th launch, dense B8 18/6) with that snapshot; export raw + source (sass and cuda).
TI=/home/operator/gautam_k/v2_work/Tensor-Implementations; SN=/home/operator/gautam_k/v2_work/snapshots; A=src/Kernels/cuda/attention/arch
export PATH=/usr/local/cuda/bin:$PATH
say() { echo "$(date +%H:%M:%S) $*" >> /home/operator/gautam_k/v2_work/logs/p1_1014/status14; }
cd $TI
say "t14 start | src vj2 $(md5sum $A/GQA_bwd_vj2_sm90.cu | cut -c1-8) lib $(md5sum lib/libtensor.so | cut -c1-8)"
cuobjdump -sass lib/objects/$A/GQA_bwd_vj2_sm90.o | awk '/Function : .*gqa_bwd_vj2_kernel/{on=1} /Function : /&&!/gqa_bwd_vj2_kernel/{on=0} on' | grep -oE '/\*[0-9a-f]{4}\*/ +[^;]+;' | sed 's/ \+/ /g' > /home/operator/gautam_k/v2_work/logs/p1_1014/sass_normal.txt
touch $A/GQA_bwd_vj2_sm90.cu
make -j$(nproc) tensor SM_ARCH=90 NVCC="/usr/local/cuda/bin/nvcc -lineinfo" > /home/operator/gautam_k/v2_work/logs/p1_1014/build_lineinfo.log 2>&1; say "t14 a: lineinfo build exit=$?"
cuobjdump -sass lib/objects/$A/GQA_bwd_vj2_sm90.o | awk '/Function : .*gqa_bwd_vj2_kernel/{on=1} /Function : /&&!/gqa_bwd_vj2_kernel/{on=0} on' | grep -oE '/\*[0-9a-f]{4}\*/ +[^;]+;' | sed 's/ \+/ /g' > /home/operator/gautam_k/v2_work/logs/p1_1014/sass_lineinfo.txt
say "t14 a: kernel SASS lines normal $(wc -l < /home/operator/gautam_k/v2_work/logs/p1_1014/sass_normal.txt) lineinfo $(wc -l < /home/operator/gautam_k/v2_work/logs/p1_1014/sass_lineinfo.txt) | identical: $(cmp -s /home/operator/gautam_k/v2_work/logs/p1_1014/sass_normal.txt /home/operator/gautam_k/v2_work/logs/p1_1014/sass_lineinfo.txt && echo YES || echo NO) | line-table sections: $(cuobjdump -elf lib/objects/$A/GQA_bwd_vj2_sm90.o 2>/dev/null | grep -c 'debug_line')"
mkdir -p $SN/g123li/lib && cp -p lib/libtensor.so $SN/g123li/lib/ && say "t14 a: snapshot g123li lib $(md5sum $SN/g123li/lib/libtensor.so | cut -c1-8)"
touch $A/GQA_bwd_vj2_sm90.cu; make -j$(nproc) tensor SM_ARCH=90 > /home/operator/gautam_k/v2_work/logs/p1_1014/build_restore.log 2>&1; say "t14 a: normal library restored exit=$? lib $(md5sum lib/libtensor.so | cut -c1-8)"
cmp -s /home/operator/gautam_k/v2_work/logs/p1_1014/sass_normal.txt /home/operator/gautam_k/v2_work/logs/p1_1014/sass_lineinfo.txt || { say "STOP: lineinfo changed the kernel code"; say "T14 DONE"; exit 1; }
n=$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader | grep -c .); m=$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$n" = 0 ] && [ "$m" -lt 1024 ] || { say "STOP: GPU 7 busy ($n procs, $m MiB) before ncu"; say "T14 DONE"; exit 1; }
env CUDA_VISIBLE_DEVICES=7 USE_VJ2=1 LD_LIBRARY_PATH=$SN/g123li/lib:$TI/lib:$TI/../Profiler/lib VARLEN_ONLY=BENCHDENSE VARLEN_CASE=dense_B8_H18x6 VARLEN_BENCH_ITERS=1 timeout 420 /usr/local/cuda/bin/ncu --set full --import-source yes --target-processes all --force-overwrite -k "regex:^gqa_bwd_vj2_kernel\$" --launch-skip 10 --launch-count 1 -o /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li $TI/tva_vj2 > /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.out 2>&1
say "t14 b: ncu exit=$?"
R=$(ls /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.ncu-rep* 2>/dev/null | head -1)
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page raw > /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.raw.csv 2>> /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source sass > /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_sass.csv 2>> /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source cuda > /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_cuda.csv 2>> /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.import.err
CUDA_VISIBLE_DEVICES= /usr/local/cuda/bin/ncu --import $R --csv --page source --print-source cuda,sass > /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_cuda_sass.csv 2>> /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.import.err
say "t14 b: exports raw $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.raw.csv) sass $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_sass.csv) cuda $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_cuda.csv) cuda+sass $(grep -c . /home/operator/gautam_k/v2_work/logs/p1_1014/ncu_g123li.source_cuda_sass.csv) lines"
say "T14 DONE"
