#!/bin/bash
# swa24_arms_par.sh -- 2026-11-20 ~20:50 IST, step 13 (the SFT arms on Tulu 3). The user: "can't we use other GPUs?".
# swa23_arms_seq.sh runs the arms one after another on GPU 7 (C, then D, then A; ~47 min each). This script starts D and
# A NOW, side by side, one GPU each (the two GPUs the user names), while C finishes on GPU 7.
# Settings identical to C: the base350 / common / ARGS lines are copied from swa23_arms_seq.sh unchanged (NP=1, ROWS 20 x
# B 4, 1 epoch of m2_tulu_A, LR 1e-4 cosine, warm-up 3%, Muon, val every 500 steps, EXPORT=1; the pushed code, binary
# bin/sft_swa).
# No clash with swa23_arms_seq.sh, nothing running is touched:
#   - each arm goes into the folder swa23_arms_seq.sh would use ($AR/<arm>), claimed with an atomic mkdir; when C ends,
#     swa23_arms_seq.sh finds D's folder, prints "ABORT: ... exists" and exits by itself (GPU 7 is then free);
#   - D is claimed only while C's trainer is still running: after C ends, swa23_arms_seq.sh needs at least 5 s (its GPU
#     check) before it looks for D's folder, so the two scripts can never both start D. If C has already ended, D is
#     left to swa23_arms_seq.sh (GPU 7) and only A starts here;
#   - an arm whose folder exists already is skipped; this script uses its own TMPDIR (tmp_b).
# GPU check first (each named GPU: no process in 2 samples 5 s apart); taken -> STOP and report (never wait, never kill).
# usage: setsid nohup bash swa24_arms_par.sh <gpu for D> <gpu for A> > arms24.out 2>&1 < /dev/null &
GD=${1:?gpu for D}; GA=${2:?gpu for A}; H=/localstorage/gautam_k/swa9_1120; X=$H/bin; T=$H/bt; ST=$H/status
Y=/localstorage/gautam_k/deps/yaml/usr; DATA=/localstorage/gautam_k/sft135/data/m2_tulu_A; AR=$H/arms
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST [swa24 arms g$GD,$GA] $*" | tee -a $ST; }
procs() { local u; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$u"; }
[ "$GD" = "$GA" ] && { say "ABORT: D and A need two different GPUs"; exit 2; }
for g in $GD $GA; do [ "$g" = 7 ] && { say "ABORT: GPU 7 runs arm C (swa23_arms_seq.sh)"; exit 2; }; done
for f in $X/sft_swa $H/raw/kv4/meta.json $H/raw/kv4_swa3/meta.json $DATA/train/meta.json $DATA/val/meta.json; do
  [ -e $f ] || { say "ABORT: missing $f; GPUs left free"; exit 2; }; done
for g in $GD $GA; do
  if [ "$(procs $g)" != 0 ] || { sleep 5; [ "$(procs $g)" != 0 ]; }; then say "STOP: GPU $g is taken (not waiting, nothing touched)"; exit 5; fi
done
mkdir -p $AR $H/tmp_b; export PATH=/usr/local/cuda/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID TMPDIR=$H/tmp_b
LDR="$(cat $H/ldp.txt):$Y/lib/x86_64-linux-gnu"
# ---- the same three lines as swa23_arms_seq.sh (do not change one without the other)
base350="N_LAYERS=28 ROPE_THETA=50000 D_MODEL=1024 Q_HEADS=8 KV_HEADS=4 FFN_HIDDEN=2560 VOCAB=49280 EMBED_SCALE=1"
common="SFT_DATA=$DATA ROWS=20 B=4 EPOCHS=1 LR=1e-4 WARMUP=0.03 VAL_FREQ=500 OPT=muon EXPORT=1 NP=1 BIN=$X/sft_swa"
declare -A ARGS; ARGS[C_swa3_win]="kv4_swa3 SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4"; ARGS[D_swa3_full]="kv4_swa3 SWA_WINDOW=0 SWA_MISMATCH_OK=1"; ARGS[A_kv4_full]="kv4"
# ----
declare -A GPU; GPU[D_swa3_full]=$GD; GPU[A_kv4_full]=$GA
say "start | binary sft_swa $(md5sum < $X/sft_swa | cut -c1-8) | data $DATA | D on GPU $GD, A on GPU $GA (C stays on GPU 7)"
launch() {   # launch <arm> <gpu> <base> VAR=value ...   (the folder is claimed by the caller)
  local arm=$1 g=$2 base=$3; shift 3; local rd=$AR/$arm
  ( cd $T && env LD_LIBRARY_PATH=$LDR RUN_DIR=$rd GPUS=$g INIT_FROM_RAW=$H/raw/$base $base350 $common "$@" \
        bash sft/run_sft.sh > $AR/run_sft_$arm.out 2>&1 ) &
  echo $! > $AR/pid_$arm
  say "launched $arm on GPU $g: base $base, $*"
}
mine=""
c_alive() { [ -f $AR/pid_C_swa3_win ] && kill -0 $(cat $AR/pid_C_swa3_win) 2> /dev/null; }
for arm in D_swa3_full A_kv4_full; do
  if [ $arm = D_swa3_full ] && ! c_alive; then say "$arm: C has already ended, so swa23_arms_seq.sh starts D on GPU 7; not launched here"; continue; fi
  if ! mkdir $AR/$arm 2> /dev/null; then say "$arm: folder exists (started elsewhere): not launched here"; continue; fi
  launch $arm ${GPU[$arm]} ${ARGS[$arm]}; mine="$mine $arm"
done
[ -z "$mine" ] && { say "nothing launched here"; rm -rf $H/tmp_b; exit 0; }
t0=$(date +%s); declare -A seen
while :; do
  alive=0
  for arm in $mine; do
    kill -0 $(cat $AR/pid_$arm) 2> /dev/null && alive=$((alive + 1))
    lg=$AR/$arm/run.log; [ -f $lg ] || continue
    if [ -z "${seen[start_$arm]}" ] && grep -q -E '^  attention       :' $lg; then
      seen[start_$arm]=1
      say "$arm: $(grep -m1 '^  attention       :' $lg | sed 's/^ *attention *: //' | cut -c1-110)"
      grep -m1 -E "window record|has no window record|WARNING \(BLU_SWA_MISMATCH_OK" $lg | cut -c1-220 | sed 's/^/    /' | tee -a $ST
    fi
    for v in $(grep -oE '\[val\] step [0-9]+ \| val_loss [0-9.]+' $lg | awk '{print $3 ":" $6}'); do
      s=${v%%:*}; [ -n "${seen[$arm.$s]}" ] && continue; seen[$arm.$s]=1
      dt=$(grep -E '^step ' $lg | tail -50 | awk '{for (i = 1; i <= NF; i++) if ($i == "dt") print $(i + 1)}' | sort -g | awk '{a[NR] = $1} END {print (NR ? a[int((NR + 1) / 2)] : "-")}')
      say "$arm: val step $s = ${v#*:} | last step $(grep -E '^step ' $lg | tail -1 | awk '{print $2}') | step ms ~$dt"
    done
    if grep -q -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg 2> /dev/null && [ -z "${seen[err_$arm]}" ]; then
      seen[err_$arm]=1; say "$arm: ERROR line: $(grep -m1 -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg | cut -c1-200)"; fi
  done
  [ $alive = 0 ] && break
  sleep 30
done
for arm in A_kv4_full C_swa3_win D_swa3_full; do
  lg=$AR/$arm/run.log
  say "$arm done: steps $(grep -c -E '^step ' $lg 2> /dev/null) | final $(grep -oE '\[val\] step [0-9]+ \| val_loss [0-9.]+' $lg 2> /dev/null | tail -1) | export $(ls -la $AR/$arm/final.safetensors 2> /dev/null | awk '{print $5 " B"}') | checkpoints $(ls -d $AR/$arm/checkpoints/*_step_* 2> /dev/null | wc -l)"
done
rm -rf $H/tmp_b
say "SWA24 ARMS DONE ($(( ($(date +%s) - t0) / 60 )) min) | GPUs: $GD $(procs $GD), $GA $(procs $GA), 7 $(procs 7) process(es)"
