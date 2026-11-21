#!/bin/bash
# swa27_armB.sh -- 2026-11-21 ~11:30 IST: arm B of the SFT window study (SWA doc sections 6 and 12.2): the FULL-ATTENTION
# base kv4, fine-tuned WITH the window (W 1,024, every 4th layer global), BLU_SWA_MISMATCH_OK=1 (the base's record says
# window 0). The user, 2026-11-21: "let's run that B too" (dropped on 2026-11-20 19:13 IST as "not our case"; run now to
# complete the 2 x 2 grid: A = kv4 + full, B = kv4 + window, C = kv4_swa3 + window, D = kv4_swa3 + full).
# Where: P015 GPU 3 (the user: "gpu 3 in p015 is ours"; P035's 8 GPUs are held by Sanjay's run since 00:47 IST). The
# inputs come from P035 (swa27_copy_p015.sh, md5-checked) at the same paths, plus P035's CUDA 13.4 runtime libraries in
# $H/cuda134, put first on the library path, so B runs with the libraries A / C / D ran with (on P035 the folder does
# not exist and the system's 13.4 is used).
# Settings identical to A / C / D: base350 / common copied unchanged from swa24_arms_par.sh (= swa23_arms_seq.sh); the
# same binary (sft_swa md5 3af3e48d), data m2_tulu_A, 1 epoch = 5,652 steps, val every 500 steps, checkpoints every
# 1,000 (newest 2 kept), EXPORT=1; launched through sft/run_sft.sh as they were.
# After the training, as swa25b_eval.sh did for A / C / D (the same commands, the same suite):
#   convert (CPU) sft/blutrain_to_hf.py -> hf_out/B_kv4_win; its config must be LLLG x 7, use_sliding_window True, 1024;
#   B_fullattn    B's weights with full attention in every layer (B served with its pretraining attention);
#   vg_B          hf_val_groups.py: native (= the window) and other (= full attention): total / trainer rows / groups;
#   ev_B          lm-eval, the 10-23 dev suite (ifeval, hellaswag, arc_easy, arc_challenge, piqa, winogrande, mmlu +
#                 gsm8k first 400), --apply_chat_template --fewshot_as_multiturn, batch 32, HF transformers;
#   ev_Bfull      the same suite on B_fullattn: on the second GPU if one is given (at the same time as ev_B), else after.
# Checks (a failure stops our run at once, our PIDs only, and says why):
#   start    GPU(s) clean (no process and < 1 GiB used, 2 samples 5 s apart; all GPUs logged); inputs present; binary
#            md5 3af3e48d; the library path of run_sft.sh (make print_run_ld_path) == ldp.txt of P035; every library of
#            sft_swa resolves, the CUDA ones from cuda134 when it exists;
#   start-up the trainer's attention line must be "sliding window ON: the last 1024 tokens ... (7 global, 21 windowed):
#            LLLG x 7" and the mismatch WARNING (base window 0, this run 1024) must be printed;
#   step 0   B's step-0 val = the kv4 base with the window, HF 2026-11-20 (valgroups/base_kv4.json, mode other, trainer
#            rows: 1.678098), within 0.0005 (our trainer = HF within 0.00007 in step 9; full attention would be 1.6723);
#   end      5,652 steps, a final val, final.safetensors, no error line; after vg_B: HF native trainer rows == the
#            trainer's final val within 0.0005.
# The evaluation waits for $H/eval_inputs.ok (written by swa27_copy_p015.sh eval after its md5 check passes; the eval
# inputs are copied while B trains). A file $H/STOP_B: no new job starts (a running one finishes).
# usage: bash swa27_armB.sh check <gpu> [<second gpu>]   (CPU only: inputs, libraries, GPU state, the exact command)
#        setsid nohup bash swa27_armB.sh run <gpu> [<second gpu>] > /localstorage/gautam_k/swa9_1120/armB27.out 2>&1 < /dev/null &
MODE=${1:?check or run}; G1=${2:?gpu}; G2=${3:-}
R=/localstorage/gautam_k; H=$R/swa9_1120; X=$H/bin; T=$H/bt; ST=$H/status; AR=$H/arms; HO=$H/hf_out; EV=$H/evals
VG=$H/valgroups; J=$H/jobs_B; Y=$R/deps/yaml/usr; S=$R/sft135; DATA=$S/data/m2_tulu_A
PYP=$S/prepenv/bin/python3; PYE=$S/evalenv/bin/python3; LME=$S/evalenv/bin/lm_eval; TOK=$S/tokenizer_base/tokenizer.json
TASKS=ifeval,hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu; GSM_LIMIT=400
ARM=B_kv4_win; BASE=kv4; NSTEPS=5652; WANT_CFG="LLLGLLLGLLLGLLLGLLLGLLLGLLLG True 1024"
WANT_ATTN="sliding window ON: the last 1024 tokens of each chat (< row length 4096), every 4th layer global (7 global, 21 windowed): LLLGLLLGLLLGLLLGLLLGLLLGLLLG"
BOX=$(hostname | grep -oE 'p0[0-9]+')
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST [swa27 armB $BOX] $*" | tee -a $ST; }
uuid() { nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader; }
procs() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$(uuid $1)"; }
mem() { nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9; }
clean() { [ "$(procs $1)" = 0 ] && [ "$(mem $1)" -lt 1024 ]; }
gpu_ok() { clean $1 && { sleep 5; clean $1; }; }
allgpus() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | awk -F', ' '{printf "%s:%s/%s ", $1, $2, $3}'; }
tree_pids() {   # a pid and all its descendants
  local all; all=$(ps -e -o pid=,ppid=); local q="$1" out="$1" p c
  while [ -n "$q" ]; do set -- $q; p=$1; shift; q="$*"
    for c in $(echo "$all" | awk -v p=$p '$2 == p {print $1}'); do out="$out $c"; q="$q $c"; done; done
  echo $out
}
stop_tree() {   # stop_tree <pid> <why>: our process tree only (TERM, then KILL)
  local tp; tp=$(tree_pids $1); say "STOPPING our processes ($2): $tp"; kill -TERM $tp 2> /dev/null; sleep 10; kill -KILL $tp 2> /dev/null
}
fail() { say "FAIL: $*"; exit 1; }
[ "$G1" = "$G2" ] && fail "the two GPUs must differ"

# ---- start checks (both modes; CPU only)
for f in $X/sft_swa $T/sft/run_sft.sh $T/sft/blutrain_to_hf.py $H/ldp.txt $H/raw/$BASE/meta.json $DATA/train/meta.json $DATA/val/meta.json $Y/lib/x86_64-linux-gnu; do
  [ -e $f ] || fail "missing $f"; done
b=$(md5sum < $X/sft_swa | cut -c1-8); [ $b = 3af3e48d ] || fail "sft_swa md5 $b, not 3af3e48d (the binary of A / C / D)"
grep -q '"swa_window": *"0"' $H/raw/$BASE/meta.json || fail "raw/$BASE/meta.json does not record window 0"
export PATH=/usr/local/cuda/bin:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID
LDR="$(cat $H/ldp.txt):$Y/lib/x86_64-linux-gnu"
[ -d $H/cuda134 ] && LDR="$H/cuda134:$LDR"
LDP=$(cd $T && make -s print_run_ld_path SM_ARCH=90 2> /dev/null | tail -n 1)   # what run_sft.sh puts first
[ "$LDP" = "$(cat $H/ldp.txt)" ] || fail "run_sft.sh library path '$LDP' differs from P035's ldp.txt '$(cat $H/ldp.txt)'"
LDD=$(LD_LIBRARY_PATH=$LDP:$LDR ldd $X/sft_swa)
echo "$LDD" | grep -q 'not found' && fail "unresolved libraries: $(echo "$LDD" | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
for l in libcudart.so.13 libcublas.so.13 libcublasLt.so.13 libcurand.so.10 libcufile.so.0; do
  got=$(echo "$LDD" | awk -v l=$l '$1 == l {print $3}')
  if [ -d $H/cuda134 ]; then [ "$got" = "$H/cuda134/$l" ] || fail "$l resolves to $got, not $H/cuda134/$l"; fi
done
for l in libtensor.so libblublas.so libdist.so libprofiler.so libyaml-cpp.so.0.8; do
  got=$(echo "$LDD" | awk -v l=$l '$1 == l {print $3}'); [ -n "$got" ] || fail "$l not linked?"; done
say "checks passed | sft_swa 3af3e48d | library path = P035's | CUDA libraries from $(echo "$LDD" | awk '$1 == "libcublasLt.so.13" {print $3}' | xargs dirname) | GPUs now: $(allgpus)"
ref0=$($PYE -c "import json; r = [x for x in json.load(open('$VG/base_kv4.json')) if x['mode'] == 'other']; print('%.6f' % r[0]['loss']['trainer_rows'])" 2> /dev/null || true)
[ -n "$ref0" ] || ref0=1.678098   # the 2026-11-20 value (valgroups/base_kv4.json is copied with the eval inputs)
base350="N_LAYERS=28 ROPE_THETA=50000 D_MODEL=1024 Q_HEADS=8 KV_HEADS=4 FFN_HIDDEN=2560 VOCAB=49280 EMBED_SCALE=1"
common="SFT_DATA=$DATA ROWS=20 B=4 EPOCHS=1 LR=1e-4 WARMUP=0.03 VAL_FREQ=500 OPT=muon EXPORT=1 NP=1 BIN=$X/sft_swa"
ARGSB="SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4 SWA_MISMATCH_OK=1"
CMD="cd $T && env LD_LIBRARY_PATH=$LDR RUN_DIR=$AR/$ARM GPUS=$G1 INIT_FROM_RAW=$H/raw/$BASE $base350 $common $ARGSB bash sft/run_sft.sh"
if [ "$MODE" = check ]; then
  say "check only: would run on GPU $G1${G2:+ (+ ev_Bfull on GPU $G2)}: $CMD"
  for g in $G1 $G2; do say "GPU $g: $(procs $g) process(es), $(mem $g) MiB used"; done
  [ -e $AR/$ARM ] && say "NOTE: $AR/$ARM exists already: run would abort"
  [ -f $H/eval_inputs.ok ] && say "eval inputs: present (eval_inputs.ok)" || say "eval inputs: not yet (the run waits for eval_inputs.ok after the training)"
  if [ -x $PYE ]; then say "eval env: $($PYE -c 'import torch, transformers, lm_eval; print("torch", torch.__version__, "| transformers", transformers.__version__, "| lm_eval ok")' 2>&1 | tail -1)"; fi
  if [ -x $PYP ]; then $PYP $T/sft/blutrain_to_hf.py --help > /dev/null 2>&1 && say "converter: imports ok (prepenv)" || say "converter: --help FAILED in prepenv"; fi
  say "step-0 reference (kv4 base with the window, HF): $ref0"
  exit 0
fi
[ "$MODE" = run ] || fail "mode must be check or run"

# ---- run: GPU check, claim the folder, launch
for g in $G1 $G2; do gpu_ok $g || { say "STOP: GPU $g is not clean ($(procs $g) process(es), $(mem $g) MiB): not waiting, nothing touched | GPUs: $(allgpus)"; exit 5; }; done
mkdir -p $AR $HO $EV $VG $J $H/tmp_B
mkdir $AR/$ARM 2> /dev/null || fail "$AR/$ARM exists (started before?): not launched"
export TMPDIR=$H/tmp_B
say "start | B = base $BASE + window 1024 / every 4, override | GPU $G1${G2:+ (+ GPU $G2 for ev_Bfull)} | $CMD"
( eval "$CMD" > $AR/run_sft_$ARM.out 2>&1 ) &
PIDT=$!; echo $PIDT > $AR/pid_$ARM
lg=$AR/$ARM/run.log; t0=$(date +%s); declare -A seen
while kill -0 $PIDT 2> /dev/null; do
  sleep 30
  [ -f $lg ] || continue
  if [ -z "${seen[attn]}" ] && grep -q -E '^  attention       :' $lg; then
    seen[attn]=1; a=$(grep -m1 -E '^  attention       :' $lg | sed 's/^ *attention *: //')
    w=$(grep -m1 -E 'WARNING \(BLU_SWA_MISMATCH_OK=1' $lg)
    say "start-up: attention: $a"; say "start-up: ${w:-NO WARNING LINE}"
    [ "$a" = "$WANT_ATTN" ] || { stop_tree $PIDT "the attention line is not the window layout"; fail "attention line '$a'"; }
    echo "$w" | grep -q 'sliding window 0 .*BLU_SWA_WINDOW=1024' || { stop_tree $PIDT "no mismatch warning"; fail "warning line '$w'"; }
  fi
  for v in $(grep -oE '\[val\] step [0-9]+ \| val_loss [0-9.]+' $lg | awk '{print $3 ":" $6}'); do
    s=${v%%:*}; [ -n "${seen[v$s]}" ] && continue; seen[v$s]=1
    dt=$(grep -E '^step ' $lg | tail -50 | awk '{for (i = 1; i <= NF; i++) if ($i == "dt") print $(i + 1)}' | sort -g | awk '{a[NR] = $1} END {print (NR ? a[int((NR + 1) / 2)] : "-")}')
    say "val step $s = ${v#*:} | last step $(grep -E '^step ' $lg | tail -1 | awk '{print $2}') | step ms ~$dt | $(( ($(date +%s) - t0) / 60 )) min"
    if [ $s = 0 ]; then
      ok=$(awk -v a=${v#*:} -v b=$ref0 'BEGIN {d = a - b; if (d < 0) d = -d; print (d <= 0.0005 ? "yes" : "no"), d}')
      say "step-0 check: B ${v#*:} vs the kv4 base with the window (HF) $ref0: diff ${ok#* } -> $([ ${ok%% *} = yes ] && echo PASS || echo FAIL)"
      [ ${ok%% *} = yes ] || { stop_tree $PIDT "step-0 val off the reference"; fail "step-0 val ${v#*:} vs $ref0"; }
    fi
  done
  if [ -z "${seen[err]}" ] && grep -q -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg; then
    seen[err]=1; say "ERROR line: $(grep -m1 -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg | cut -c1-200)"; fi
done
if [ -z "${seen[err]}" ] && grep -q -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg 2> /dev/null; then   # lines of the last 30 s
  seen[err]=1; say "ERROR line: $(grep -m1 -i -E 'fatal|\bnan\b|out of memory|illegal|ERROR:' $lg | cut -c1-200)"; fi
n=$(grep -c -E '^step ' $lg 2> /dev/null); fin=$(grep -oE '\[val\] step [0-9]+ \| val_loss [0-9.]+' $lg 2> /dev/null | tail -1)
sz=$(stat -c %s $AR/$ARM/final.safetensors 2> /dev/null || echo 0)
say "training ended ($(( ($(date +%s) - t0) / 60 )) min): steps $n | final $fin | export $sz B | checkpoints $(ls -d $AR/$ARM/checkpoints/*_step_* 2> /dev/null | wc -l)"
[ "$n" = $NSTEPS ] && [ -n "$fin" ] && [ "$sz" -gt 1000000000 ] && [ -z "${seen[err]}" ] || fail "the training did not end cleanly (see $lg)"
tv=$(echo "$fin" | awk '{print $NF}')

# ---- evaluation inputs (copied while B trained)
while [ ! -f $H/eval_inputs.ok ]; do
  [ -f $H/STOP_B ] && fail "STOP_B while waiting for the eval inputs"
  [ -z "${seen[wait]}" ] && { seen[wait]=1; say "waiting for the eval inputs (eval_inputs.ok)"; }; sleep 30
done
export NLTK_DATA=$H/nltk_data HF_HOME=$H/hfcache TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1 TMPDIR=$H/tmp_eB
mkdir -p $TMPDIR
for f in $PYP $PYE $LME $TOK $H/hf_val_groups.py $H/hf/$BASE/config.json $NLTK_DATA/tokenizers/punkt_tab; do [ -e $f ] || fail "missing $f"; done

# ---- convert (CPU), B_fullattn
$PYP $T/sft/blutrain_to_hf.py --export $AR/$ARM/final.safetensors --base-hf $H/hf/$BASE --tokenizer $TOK --out $HO/$ARM \
    > $J/convert_$ARM.log 2>&1 || fail "convert failed (jobs_B/convert_$ARM.log)"
got=$($PYP -c "import json; c = json.load(open('$HO/$ARM/config.json')); lt = c.get('layer_types') or []; print(''.join('G' if t == 'full_attention' else 'L' for t in lt), c.get('use_sliding_window'), c.get('sliding_window'))")
[ "$got" = "$WANT_CFG" ] || fail "hf_out/$ARM config attention '$got', expected '$WANT_CFG'"
touch $HO/$ARM/done; say "converted -> hf_out/$ARM | config: $got"
rm -rf $HO/B_fullattn; mkdir -p $HO/B_fullattn
for f in $HO/$ARM/*; do case $(basename $f) in config.json|done) ;; *) ln -s $f $HO/B_fullattn/ ;; esac; done
$PYP -c "
import json; c = json.load(open('$HO/$ARM/config.json')); n = c['num_hidden_layers']
c['use_sliding_window'], c['sliding_window'], c['layer_types'] = False, None, ['full_attention'] * n
json.dump(c, open('$HO/B_fullattn/config.json', 'w'), indent=2)" || fail "B_fullattn config"
touch $HO/B_fullattn/done; say "made hf_out/B_fullattn (B's weights, full attention in all 28 layers)"

# ---- GPU jobs
evone() {   # name folder
  local O=$EV/$1 rc rc2; mkdir -p $O
  $LME --model hf --model_args "pretrained=$2,dtype=bfloat16" --tasks "$TASKS" --apply_chat_template --fewshot_as_multiturn \
      --batch_size 32 --output_path $O --log_samples > $O/run.txt 2>&1; rc=$?
  $LME --model hf --model_args "pretrained=$2,dtype=bfloat16" --tasks gsm8k --limit $GSM_LIMIT --apply_chat_template \
      --fewshot_as_multiturn --batch_size 32 --output_path $O/gsm8k --log_samples >> $O/run.txt 2>&1; rc2=$?
  [ $rc = 0 ] && [ $rc2 = 0 ]
}
evsum() { echo "$1: $(grep -E '^\|(ifeval|gsm8k|hellaswag|arc_easy|arc_challenge|piqa|winogrande|mmlu) ' $EV/$1/run.txt 2> /dev/null | awk -F'|' '{gsub(/ /,"",$2); gsub(/ /,"",$6); gsub(/ /,"",$8); printf "%s(%s) %s  ", $2, $6, $8}' | cut -c1-330)"; }
waitclean() { local i; for i in $(seq 1 12); do gpu_ok $1 && return 0; sleep 5; done; return 1; }
P2=""
if [ -n "$G2" ] && [ ! -f $H/STOP_B ]; then
  if waitclean $G2; then
    ( export CUDA_VISIBLE_DEVICES=$G2; evone B_fullattn $HO/B_fullattn; echo $? > $J/ev_Bfull.rc ) > /dev/null 2>&1 < /dev/null &
    P2=$!; say "start ev_Bfull on GPU $G2"
  else say "GPU $G2 is not clean now: ev_Bfull will run on GPU $G1 after ev_B"; G2=""; fi
fi
[ -f $H/STOP_B ] && fail "STOP_B before vg_B"
waitclean $G1 || fail "GPU $G1 is not clean after the training ($(procs $G1) process(es), $(mem $G1) MiB): not waiting"
say "start vg_B on GPU $G1"
CUDA_VISIBLE_DEVICES=$G1 $PYE $H/hf_val_groups.py $HO/$ARM $DATA --name B --json $VG/B.json >> $VG/B.log 2>&1 || fail "vg_B (valgroups/B.log)"
grep -h -E '^HF val groups' $VG/B.log | cut -d'|' -f2-10 | sed 's/^/    /' | tee -a $ST
hv() { $PYE -c "import json; r = [x for x in json.load(open('$VG/B.json')) if x['mode'] == '$1']; print('%.6f' % r[0]['loss']['$2'])"; }
ok=$(awk -v a=$(hv native trainer_rows) -v b=$tv 'BEGIN {d = a - b; if (d < 0) d = -d; print (d <= 0.0005 ? "PASS" : "FAIL"), d}')
say "check: HF native trainer rows $(hv native trainer_rows) vs trainer final val $tv: diff ${ok#* } -> ${ok%% *}"
[ -f $H/STOP_B ] && fail "STOP_B before ev_B"
say "start ev_B on GPU $G1"
( export CUDA_VISIBLE_DEVICES=$G1; evone B $HO/$ARM ) && say "end ev_B exit 0" || say "ERROR line: ev_B failed (evals/B/run.txt)"
say "    $(evsum B)"
if [ -n "$P2" ]; then
  wait $P2; say "end ev_Bfull exit $(cat $J/ev_Bfull.rc 2> /dev/null)"
elif [ ! -f $H/STOP_B ]; then
  say "start ev_Bfull on GPU $G1"
  ( export CUDA_VISIBLE_DEVICES=$G1; evone B_fullattn $HO/B_fullattn ) && say "end ev_Bfull exit 0" || say "ERROR line: ev_Bfull failed"
fi
say "    $(evsum B_fullattn)"
rm -rf $H/tmp_B $H/tmp_eB
say "SWA27 ARM B DONE ($(( ($(date +%s) - t0) / 60 )) min since the launch) | GPUs: $(allgpus)"
