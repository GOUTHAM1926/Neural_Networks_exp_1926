#!/bin/bash
# 2026-10-21 (1x H200 root@86.38.238.55): CPU-only setup for the SFT benchmarks (PLAN.md). No GPU use.
#   1. base checkpoint: HF navingv/curriculum_8btokens/blumodel_step_8000, only metadata.bin + data_rank_0.bin (6.5 GB)
#   2. the laptop master tree (SFT kernels: QPU + Vj2 + packed rows, loss fixes, RoPE positions): library, the SFT trainer
#      bluscript_sft_bin, the attention suite tva_a
#   3. the loss rivals' venv (lossenv: torch + liger-kernel + cut-cross-entropy) and the attention rivals' venv (benv:
#      FA3 + FA4 from source, cuDNN through torch)
# All in parallel (10-19 record: the rivals' env took ~5-7 min clean -- FA3 is built only for bf16 / hd 128 / sm90, ~3-4.5 min);
# our library (~15 min) is the long pole, so the whole setup is ~20 min. Status: $GK/sftb/setup_status.
# usage: setsid nohup bash ~/gautam_k/sftb/setup_sftb.sh > /dev/null 2>&1 < /dev/null &
GK=$HOME/gautam_k; S=$GK/sftb; ST=$S/setup_status; M=$GK/sft_v2/BluTrain_sft; TI=$M/Tensor-Implementations
mkdir -p $S $GK/tmp $GK/ckpt/blumodel_step_8000
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $ST; }
say "setup start | cores $(nproc) | disk free $(df -h ~ | awk 'NR==2 {print $4}')"

# 1. checkpoint (in the background)
(
  D=$GK/ckpt/blumodel_step_8000; U=https://huggingface.co/navingv/curriculum_8btokens/resolve/main/blumodel_step_8000
  for f in metadata.bin data_rank_0.bin; do
    curl -sSL --fail --retry 5 -C - -o "$D/$f" "$U/$f" || { say "CKPT FAILED: $f"; exit 1; }
  done
  say "CKPT DONE | metadata.bin $(stat -c %s $D/metadata.bin) B md5 $(md5sum < $D/metadata.bin | cut -c1-8) (expect 129127 / 5119b20d) | data_rank_0.bin $(stat -c %s $D/data_rank_0.bin) B (expect 6469415424)"
) &
CK=$!

# 3. the rivals' environments, in the background alongside our build
( bash $GK/loss_bench/loss_env_setup.sh > $S/lossenv.log 2>&1
  say "LOSSENV exit=$? | $(grep -E 'imports used|SETUP DONE|FAILED' $GK/lossenv_setup.log | tail -2 | tr '\n' ' ')" ) &
LE=$!
( bash $GK/scripts/benv_setup.sh > $S/benv.log 2>&1
  say "BENV exit=$? | $(tail -2 $GK/benv_setup.log | tr '\n' ' ')" ) &
BE=$!

# 2. our build: library, trainer, attention suite
cd $M || { say "ABORT: no $M (rsync the master first)"; exit 3; }
make -j32 blublas profiler tensor dist SM_ARCH=90 > $S/build_lib.log 2>&1; rc=$?   # = all without ensure-blurun (PATH change)
say "LIB build exit=$rc | libtensor $(md5sum < $TI/lib/libtensor.so 2>/dev/null | cut -c1-8) | LossKernels.cu $(md5sum < $TI/src/Kernels/cuda/loss/LossKernels.cu | cut -c1-8) (expect 064590b4) | GQA_fwd_sm90.cu $(md5sum < $TI/src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu | cut -c1-8) (expect 7e4197ea)"
[ $rc = 0 ] || { say "ABORT: library build failed (sftb/build_lib.log)"; exit 1; }
cmd=$(make -n run-mpi FILE=bluscript_sft.cpp NP=1 SM_ARCH=90 2>/dev/null | grep -- '-o _runner_ bluscript_sft.cpp' | head -1 | sed 's/-o _runner_ /-o bluscript_sft_bin /')
[ -n "$cmd" ] || { say "ABORT: could not get the trainer's compile line from make -n run-mpi"; exit 1; }
echo "$cmd" > $S/build_trainer_cmd.sh
CPATH=$GK/local/include LIBRARY_PATH=$GK/local/lib bash $S/build_trainer_cmd.sh > $S/build_trainer.log 2>&1; rc=$?
say "TRAINER build exit=$rc | bluscript_sft_bin $(md5sum < $M/bluscript_sft_bin 2>/dev/null | cut -c1-8) | bluscript_sft.cpp $(md5sum < $M/bluscript_sft.cpp | cut -c1-8)"
cd $TI && nvcc -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA \
  -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a \
  -o $S/tva_a Tests/kernel_tests/test_varlen_attention.cu -L/usr/local/cuda/lib64 -Llib -L../Profiler/lib \
  -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lblublas -lcuda > $S/build_tva.log 2>&1
say "TVA build exit=$? | test source $(md5sum < Tests/kernel_tests/test_varlen_attention.cu | cut -c1-8)"

wait $LE $BE $CK
say "SETUP DONE"
