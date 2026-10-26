#!/bin/bash
# 2026-10-21 (1x H200 root@86.38.238.55, its only GPU = 0): OLD vs NEW TRAINER A/B for dist_side_gautam_lottn.
# The colleagues' production trainer (MODE=production_zero: train/production_zero/bluscript_zero_ckpt.cpp, built per tree by
# `make _ablation_compile` into lottn/pz_old and lottn/pz_new) with their production YAML, ONE change: validation_freq 4
# (ab/pz_ab_config.yaml; its parallelism.config resolves next to it: ab/bluscript_zero_ckpt_parallelism.yaml, an UNCHANGED
# copy -- BLU_PARALLELISM is parallelism.enabled, an int, never a path). No BLU_MAX_STEPS: the real LR schedule (8000 steps, 200 warmup), every run is stopped after step
# $STOP, so both sides see the same LR at every step. Env only: data = the edufineweb shards linked as the 4 domains
# (pt_data/mix), no checkpoints, no monitoring stack, no CSV log. One GPU = world size 1, so GRAD_ACCUM = 32 micro-steps
# of B 8 x T 4096 per step (1,048,576 tokens), exactly the per-GPU work of the production micro-batch.
# Order OLD, NEW, OLD, NEW. Per run: the trainer log; nvidia-smi samples every 500 ms (memory.used = the peak memory);
# the libtensor.so / libdist.so the binary loads (checked with ldd before the run, the run aborts on a mismatch).
# ~30 min at STOP 12 (~6-7 min per run). Then compare_trainer_ab.py writes summary.txt.
# usage: setsid nohup bash ~/gautam_k/lottn/ab/ab_trainer.sh [stop_step, default 12] > /dev/null 2>&1 < /dev/null &
STOP=${1:-12}; GK=$HOME/gautam_k; AB=$GK/lottn; CFG=$AB/ab/pz_ab_config.yaml
R=$AB/runs/trainer_ab_$(TZ=Asia/Kolkata date +%m%d_%H%M); mkdir -p $R
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $R/status; }
for t in old new; do [ -x $AB/pz_$t ] || { say "ABORT: no trainer binary $AB/pz_$t"; exit 3; }; done
[ -f $CFG ] || { say "ABORT: no $CFG"; exit 3; }
busy=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$busy" = 0 ] || { say "ABORT: GPU 0 is in use: $(nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; exit 3; }
export CUDA_VISIBLE_DEVICES=0 BLU_MIX_ROOT=$GK/pt_data/mix BLU_CKPT=0 BLU_MONITOR=0 BLU_LOG_ENABLED=0
XS=$(env | grep -oE '^BLU_[A-Z0-9_]+=' | tr -d '=' | sed 's/^/-x /' | tr '\n' ' ')   # every BLU_* variable reaches the rank
say "env: $(env | grep -E '^BLU_' | sort | tr '\n' ' ')"
say "start | STOP $STOP | $(nvidia-smi -i 0 --query-gpu=name,memory.used,utilization.gpu,clocks.sm --format=csv,noheader)"
for run in old_r1 new_r1 old_r2 new_r2; do
  t=${run%_*}; T=$AB/$t; BIN=$AB/pz_$t
  LDP=$(cd $T && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
  lt=$(LD_LIBRARY_PATH=$LDP ldd $BIN | awk '/libtensor.so/ {print $3}'); ld=$(LD_LIBRARY_PATH=$LDP ldd $BIN | awk '/libdist.so/ {print $3}')
  case "$lt|$ld" in "$T/Tensor-Implementations/lib/libtensor.so|$T/dist/"*) ;; *) say "ABORT: $run loads '$lt' and '$ld' (tree $T)"; exit 3;; esac
  say "$run | bin $(md5sum < $BIN | cut -c1-8) | libtensor $(md5sum < $lt | cut -c1-8) $lt"
  TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,memory.used,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -lms 500 > $R/$run.smi.csv 2>&1 &
  SMI=$!
  cd $R
  LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
    $XS $BIN --config=$CFG > $R/$run.log 2>&1 &
  PID=$!; t0=$(date +%s); why="died"   # PID = mpirun itself (a plain command in the background, not a subshell)
  while kill -0 $PID 2>/dev/null; do
    if grep -qE "^step +$STOP \|" $R/$run.log; then why="reached step $STOP"; break; fi
    [ $(( $(date +%s) - t0 )) -gt 1500 ] && { why="TIMEOUT (25 min)"; break; }
    sleep 5
  done
  kill -TERM $PID 2>/dev/null; sleep 10; kill -KILL $PID 2>/dev/null; wait $PID 2>/dev/null
  for p in $(pgrep -f "$BIN --config"); do kill -KILL $p 2>/dev/null; done
  sleep 5; kill $SMI 2>/dev/null
  peak=$(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2 + 0 > m) m = $2 + 0} END {print m}' $R/$run.smi.csv)
  say "$run: $why after $(( $(date +%s) - t0 )) s | $(grep -cE '^step +[0-9]+ \|' $R/$run.log) step lines | peak GPU memory $peak MiB"
  left=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$left" = 0 ] || { say "ABORT: GPU 0 still has $left process(es) after $run"; exit 3; }
done
python3 $AB/ab/compare_trainer_ab.py $R $STOP > $R/summary.txt 2>&1
say "summary written: $R/summary.txt"
cat $R/summary.txt
