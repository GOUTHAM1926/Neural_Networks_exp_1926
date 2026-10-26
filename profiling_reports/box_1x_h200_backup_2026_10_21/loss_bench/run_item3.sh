#!/bin/bash
# 2026-10-17 (2x box, GPU 0 = ours by default): item 3 -- the loss gradient written over the logits (opt-in) + the double
# allocation in the CUDA loss backward removed -- checked standalone and in the SFT trainer (work doc 45.37).
# The new files must already be copied into the box tree (laptop rsync). Phases (2nd argument, default all):
#   standalone (~8 min): build the library -> libs/loss_v3; test_loss_grad_inplace.cu (lm_head + loss through autograd at
#     8 x 4096 x 50,304 bf16, real stage-0 targets) in 3 processes: OLD library (libs/loss_v2 = item 2 only) gradient into a
#     new tensor, NEW library switch off, NEW library switch on. Same loss + gradient checksums expected; allocator peaks.
#   trainer (~10 min: ~2 min of builds, then 6 runs of ~1-1.5 min): build the NEW trainer (bluscript_sft_bin_item3) and the
#     OLD one (bluscript_sft_old.cpp = today's trainer without the item-3 lines, old LossOps.h from loss_bench/old_trainer/old_inc,
#     linked to libs/loss_v2 = before item 3, double allocation), then runs old / new-off / new-on, twice, random init (no
#     BLU_INIT_FROM: memory does not depend on the weights) on a tiny SFT set: the first 64 train / 16 val rows of stage 0,
#     MADE ON THE LAPTOP and copied to $GK/sft_data/tiny_item3 (this box has only train/labels.bin of stage 0; the first try
#     made empty files here). BLU_SFT_EPOCHS=4 -> 4 steps of 8 micro-batches (the default 2 epochs = 2 steps, warmup 1, is
#     refused by the trainer's decay-tail check). Stops at the first run that fails. Allocator [MEM] peaks + step losses.
# usage: nohup bash /root/gautam_k/loss_bench/run_item3.sh [gpu] [standalone|trainer|all] > /dev/null 2>&1 &
G=${1:-0}; PH=${2:-all}; GK=/root/gautam_k; R=$GK/sft_v2/BluTrain_sft; TI=$R/Tensor-Implementations; LB=$GK/loss_bench
OLD=$GK/libs/loss_v2; NEW=$GK/libs/loss_v3; LAB=$GK/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
L=$GK/sft_v2/runs/item3_1017; mkdir -p $L
export PATH=/usr/local/cuda/bin:$PATH
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $L/status; }
fail() { say "FAILED: $*"; exit 1; }
grep -q "sparse_ce_backward_cuda_vec_with_stats_inplace" $TI/src/Kernels/cuda/loss/LossKernels.cu && \
grep -q "grad_into_logits_" $TI/src/autograd/backward/LossBackward.cpp && grep -q "grad_into_logits" $TI/src/autograd/operations/LossOps.cpp && \
grep -q "grad_into_logits" $TI/include/autograd/operations/LossOps.h && grep -q "BLU_LOSS_GRAD_INPLACE" $R/bluscript_sft.cpp && \
[ -f $TI/Tests/kernel_tests/test_loss_grad_inplace.cu ] || fail "the box tree does not have all item-3 files (copy them first)"
[ -f $OLD/libtensor.so ] && [ -f $LAB ] || fail "missing $OLD/libtensor.so or $LAB"

if [ $PH = standalone ] || [ $PH = all ]; then
  say "phase standalone | tree LossKernels.cu $(md5sum $TI/src/Kernels/cuda/loss/LossKernels.cu | cut -c1-8) LossBackward.cpp $(md5sum $TI/src/autograd/backward/LossBackward.cpp | cut -c1-8) LossOps.cpp $(md5sum $TI/src/autograd/operations/LossOps.cpp | cut -c1-8)"
  cd $TI && make -j$(nproc) > $L/build_lib.log 2>&1 || fail "library build (build_lib.log)"
  mkdir -p $NEW && cp lib/libtensor.so $NEW/ && say "library: $NEW/libtensor.so $(md5sum $NEW/libtensor.so | cut -c1-8) (old $(md5sum $OLD/libtensor.so | cut -c1-8))"
  NVF="-I$TI/include -I$TI/../Profiler/include -I$TI/../dist/Distributed_Checkpointing/include -I/usr/local/cuda/include -DWITH_CUDA -std=c++17 -Xcompiler=-fPIC --use_fast_math -O3 --expt-relaxed-constexpr -gencode arch=compute_90a,code=sm_90a"
  NVL="-L/usr/local/cuda/lib64 -L$TI/../Profiler/lib -ltensor -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler -lcuda"
  nvcc $NVF -o $LB/t_inplace_new $TI/Tests/kernel_tests/test_loss_grad_inplace.cu -L$NEW $NVL > $L/build_test_new.log 2>&1 || fail "test build new (build_test_new.log)"
  nvcc $NVF -DLEGACY_LOSS_API -o $LB/t_inplace_old $TI/Tests/kernel_tests/test_loss_grad_inplace.cu -L$OLD $NVL > $L/build_test_old.log 2>&1 || fail "test build legacy (build_test_old.log)"
  for arm in old_out new_out new_in; do
    lib=$NEW; bin=$LB/t_inplace_new; mode=${arm#*_}; [ ${arm%_*} = old ] && { lib=$OLD; bin=$LB/t_inplace_old; }
    got=$(LD_LIBRARY_PATH=$lib:$TI/../Profiler/lib ldd $bin | awk '/libtensor.so/ {print $3}')
    [ "$got" = "$lib/libtensor.so" ] || fail "$arm loads '$got', not $lib/libtensor.so"
    CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$lib:$TI/../Profiler/lib $bin $LAB $mode > $L/test_$arm.log 2>&1
    say "test $arm exit=$? | $(grep -o 'loss=[^ ]* wgrad_cs=[^ ]* xgrad_cs=[^ ]*\|allocated_peak_MiB=[^ ]*\|active_peak_MiB=[^ ]*\|reps_identical=[01]' $L/test_$arm.log | paste -sd' ')"
  done
fi

if [ $PH = trainer ] || [ $PH = all ]; then
  say "phase trainer | bluscript_sft.cpp $(md5sum $R/bluscript_sft.cpp | cut -c1-8)"
  TD=$GK/sft_data/tiny_item3
  for f in train/inputs.bin:524288 train/labels.bin:524288 train/valid_counts.bin:256 \
           val/inputs.bin:131072 val/labels.bin:131072 val/valid_counts.bin:64; do
    [ "$(stat -c %s $TD/${f%:*} 2>/dev/null)" = "${f#*:}" ] || fail "$TD/${f%:*} is not ${f#*:} bytes (copy the tiny set from the laptop)"
  done
  grep -q '"N": 64' $TD/train/meta.json && grep -q '"N": 16' $TD/val/meta.json || fail "$TD/*/meta.json do not say N 64 / 16"
  say "tiny set ok: $(cat $TD/train/*.bin $TD/val/*.bin | md5sum | cut -c1-8)"
  cd $R && make -j$(nproc) all SM_ARCH=90 > $L/build_all.log 2>&1 || fail "top-level build (build_all.log)"
  CMD=$(make -n run-mpi FILE=bluscript_sft.cpp SM_ARCH=90 2>/dev/null | grep -m1 -- '-o _runner_' | sed 's/-o _runner_/-o bluscript_sft_bin_item3/')
  [ -n "$CMD" ] || fail "could not get the trainer compile line from 'make -n run-mpi'"
  echo "$CMD" > $L/build_trainer_cmd.sh
  bash $L/build_trainer_cmd.sh > $L/build_trainer.log 2>&1 || fail "trainer build (build_trainer.log)"
  say "trainer built: bluscript_sft_bin_item3 $(md5sum $R/bluscript_sft_bin_item3 | cut -c1-8)"
  [ -f $R/bluscript_sft_old.cpp ] && [ -f $LB/old_trainer/old_inc/autograd/operations/LossOps.h ] || fail "old trainer files missing"
  CMD_OLD=$(echo "$CMD" | sed "s# bluscript_sft.cpp # bluscript_sft_old.cpp #; s#-o bluscript_sft_bin_item3#-o bluscript_sft_bin_old#; s#^\([^ ]*\) #\1 -I$LB/old_trainer/old_inc -L$OLD #")
  echo "$CMD_OLD" > $L/build_trainer_old_cmd.sh
  bash $L/build_trainer_old_cmd.sh > $L/build_trainer_old.log 2>&1 || fail "old trainer build (build_trainer_old.log)"
  say "old trainer built: bluscript_sft_bin_old $(md5sum $R/bluscript_sft_bin_old | cut -c1-8)"
  LDP=$(make -C $R --no-print-directory print_run_ld_path)
  for arm in old_r1 off_r1 on_r1 old_r2 off_r2 on_r2; do
    v=0; [ ${arm%_*} = on ] && v=1; bin=$R/bluscript_sft_bin_item3; ldp=$LDP; tl=$NEW
    [ ${arm%_*} = old ] && { bin=$R/bluscript_sft_bin_old; ldp=$OLD:$LDP; tl=$OLD; }
    got=$(LD_LIBRARY_PATH=$ldp ldd $bin | awk '/libtensor.so/ {print $3}')
    [ "$(readlink -f $got)" = "$(readlink -f $([ $tl = $OLD ] && echo $OLD/libtensor.so || echo $TI/lib/libtensor.so))" ] || fail "$arm loads '$got'"
    d=$L/trainer_$arm; mkdir -p $d && cd $d
    for x in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$x"; done
    env CUDA_VISIBLE_DEVICES=$G LD_LIBRARY_PATH=$ldp BLU_SFT_DATA=$TD BLU_SFT_EPOCHS=4 BLU_ALL_BF16=1 BLU_CKPT=0 BLU_OPTIMIZER=adamw \
      BLU_LOSS_GRAD_INPLACE=$v mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA \
      -x BLU_SFT_EPOCHS -x BLU_ALL_BF16 -x BLU_CKPT -x BLU_OPTIMIZER -x BLU_LOSS_GRAD_INPLACE $bin < /dev/null > run.log 2>&1
    rc=$?
    say "trainer $arm exit=$rc | lib $(md5sum $got | cut -c1-8) | $(grep -m1 '^\[MEM\]' run.log) | $(grep -cE '^step +[0-9]+ ' run.log) step lines"
    [ $rc = 0 ] && grep -q '^\[MEM\]' run.log || fail "trainer $arm (trainer_$arm/run.log)"
  done
fi
cd $GK/sft_v2/runs && tar czf $GK/item3_results.tgz item3_1017 && say "PACKED item3_results.tgz"
