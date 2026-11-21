#!/bin/bash
# swa25b_eval.sh -- 2026-11-20 ~21:50 IST: = swa25_eval.sh with NLTK_DATA set. The lm-eval runs of swa25_eval.sh failed
# when IFEval was scored: NLTK refuses its punkt_tab data in the eval venv (copied from P032, owned by another account,
# uid 1004) and in /home/operator/nltk_data (writable by other accounts). Fix: a private copy owned by us ($H/nltk_data,
# the extracted punkt_tab, no write for group / others), NLTK_DATA pointing to it (checked: sent_tokenize / word_tokenize
# work). The val-group jobs and the conversions of swa25_eval.sh are done (their done markers are kept, so they are
# skipped); the 4 lm-eval jobs run again. Same GPUs (7, 4, 5), same 00:00 IST limit.
# (header of swa25_eval.sh:) swa25_eval.sh -- 2026-11-20 ~21:15 IST, step 13 of the SWA plan: the evaluation of the SFT arms, scheduled (the user:
# "after these runs, run the remaining tests too ... schedule those too"). Each job starts by itself as soon as its inputs
# exist and one of OUR GPUs is free: GPU 7 (arm C ended 20:58), GPUs 4 and 5 (arms D and A, ending ~21:38). A GPU is used
# only when no process at all is on it (2 samples 5 s apart) and no job of ours runs there; a GPU taken by anyone else is
# skipped (never killed, never waited on in a way that blocks the other GPUs). Nothing else on the box is touched.
#
# 1. CPU, at once: the lm-eval task data downloaded into our own HF_HOME ($H/hfcache; P035 has no copy: the 135M evals ran
#    on P032).
# 2. CPU, per arm, once its trainer has ended and final.safetensors exists: sft/blutrain_to_hf.py (the pushed exporter:
#    bf16 Qwen3 folder + the run's window fields + our tulu-plain-v1 chat template) -> hf_out/<arm>. Its config is checked
#    (C: window 1024, LLLG x 7; A, D: full attention in all 28 layers); a wrong one is an ERROR and that model is not
#    evaluated. For C also hf_out/C_fullattn: the same weights (links), full attention in every layer (what vLLM and
#    SGLang serve for our layout).
# 3. GPU jobs, in this order of priority, each on one free GPU of ours:
#    vg_*  hf_val_groups.py on the Tulu val split: native attention and the other type; total + groups (chats <= / > 1,024
#          tokens; positions 0-1k / 1k-2k / 2k+). Also the two bases (kv4_swa3, kv4) as the reference. The native total
#          must equal the trainer's val loss of the same weights (checked in the final summary).
#    ev_*  lm-eval, the 10-23 dev suite of the 135M work unchanged (eval_queue_v3.sh): ifeval, hellaswag, arc_easy,
#          arc_challenge, piqa, winogrande, mmlu + gsm8k (first 400); --apply_chat_template --fewshot_as_multiturn,
#          batch 32, HF transformers backend (vLLM ignores per-layer windows). Models: C, D, A, C_fullattn.
# A file $H/STOP_EVAL: no new job starts (running ones finish).
# Time limit (the user, 2026-11-20 ~21:20: "max time is 12:00, don't run after that even though pending runs are there"):
# a job starts only if its estimate (vg_bases 10 min, other vg 6 min, ev 50 min) ends before 00:00 IST; at 00:00 every job
# of ours still running is stopped (our processes only, by exact PID) and the script ends. When all jobs are done the
# script ends and leaves the GPUs free (nothing else is started).
# usage: setsid nohup bash swa25_eval.sh > eval25.out 2>&1 < /dev/null &
H=/localstorage/gautam_k/swa9_1120; T=$H/bt; AR=$H/arms; HO=$H/hf_out; EV=$H/evals; VG=$H/valgroups; ST=$H/status; J=$H/jobs
S=/localstorage/gautam_k/sft135; PYP=$S/prepenv/bin/python3; PYE=$S/evalenv/bin/python3; LME=$S/evalenv/bin/lm_eval
DATA=$S/data/m2_tulu_A; TOK=$S/tokenizer_base/tokenizer.json; GPUS="7 4 5"
TASKS=ifeval,hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu; GSM_LIMIT=400
JOBS="vg_C vg_bases vg_D vg_A ev_C ev_D ev_A ev_Cfull"
export NLTK_DATA=$H/nltk_data   # swa25b: the private punkt_tab copy (see the header)
export HF_HOME=$H/hfcache TMPDIR=$H/tmp_e TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p $HO $EV $VG $HF_HOME $TMPDIR $J
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST [swa25b eval] $*" | tee -a $ST; }
procs() { local u; u=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader); nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$u"; }
for f in $PYP $PYE $LME $TOK $DATA/val/meta.json $H/hf_val_groups.py $T/sft/blutrain_to_hf.py $H/hf/kv4_swa3/config.json $H/hf/kv4/config.json $NLTK_DATA/tokenizers/punkt_tab; do
  [ -e $f ] || { say "ABORT: missing $f"; exit 2; }; done
t0=$(date +%s); DL=$(TZ=Asia/Kolkata date -d "2026-11-21 00:00:00" +%s)
est_min() { case $1 in vg_bases) echo 10 ;; vg_*) echo 6 ;; *) echo 50 ;; esac; }
tree_pids() {   # a pid and all its descendants
  local all; all=$(ps -e -o pid=,ppid=); local q="$1" out="$1" p c
  while [ -n "$q" ]; do set -- $q; p=$1; shift; q="$*"
    for c in $(echo "$all" | awk -v p=$p '$2 == p {print $1}'); do out="$out $c"; q="$q $c"; done; done
  echo $out
}
say "start | our GPUs $GPUS | jobs $JOBS | HF_HOME $HF_HOME | no job may run past 00:00 IST"

# ---- 1. task data (CPU, background)
( $PYE - > $J/predownload.log 2>&1 <<'EOF'
from lm_eval.tasks import TaskManager
tm = TaskManager()
names = ["ifeval", "hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu", "gsm8k"]
d = tm.load_task_or_group(names)
print("task data ready:", ", ".join(names), "|", len(d), "entries")
EOF
  echo $? > $J/predownload.rc ) &

# ---- 2. conversions (CPU)
arm_finished() { [ -f $AR/$1/final.safetensors ] && [ -f $AR/pid_$1 ] && ! kill -0 $(cat $AR/pid_$1) 2> /dev/null; }
convert() {   # arm base
  local arm=$1 base=$2 out=$HO/$1 got want
  rm -rf $out
  $PYP $T/sft/blutrain_to_hf.py --export $AR/$arm/final.safetensors --base-hf $H/hf/$base --tokenizer $TOK --out $out \
      > $J/convert_$arm.log 2>&1 || { say "ERROR line: convert $arm failed (jobs/convert_$arm.log)"; touch $out.failed; return 1; }
  got=$($PYP -c "import json; c = json.load(open('$out/config.json')); lt = c.get('layer_types') or []; print(''.join('G' if t == 'full_attention' else 'L' for t in lt), c.get('use_sliding_window'), c.get('sliding_window'))")
  if [ $arm = C_swa3_win ]; then want="LLLGLLLGLLLGLLLGLLLGLLLGLLLG True 1024"; else want="GGGGGGGGGGGGGGGGGGGGGGGGGGGG False None"; fi
  [ "$got" = "$want" ] || { say "ERROR line: $arm config.json attention is '$got', expected '$want': not evaluated"; touch $out.failed; return 1; }
  touch $out/done; say "converted $arm -> hf_out/$arm | config: $got | $(grep -m1 -E 'config.json window' $J/convert_$arm.log | cut -c1-80)"
}
make_full() {   # C with full attention in every layer, same weights (links)
  local src=$HO/C_swa3_win out=$HO/C_fullattn f
  rm -rf $out; mkdir -p $out
  for f in $src/*; do case $(basename $f) in config.json|done) ;; *) ln -s $f $out/ ;; esac; done
  $PYP -c "
import json; c = json.load(open('$src/config.json')); n = c['num_hidden_layers']
c['use_sliding_window'], c['sliding_window'], c['layer_types'] = False, None, ['full_attention'] * n
json.dump(c, open('$out/config.json', 'w'), indent=2)" || { say "ERROR line: C_fullattn config failed"; return 1; }
  touch $out/done; say "made hf_out/C_fullattn (C's weights, full attention in all 28 layers)"
}

# ---- 3. GPU jobs
vgone() {   # name folder
  $PYE $H/hf_val_groups.py $2 $DATA --name $1 --json $VG/$1.json >> $VG/$1.log 2>&1
}
evone() {   # name folder
  local O=$EV/$1 rc rc2; mkdir -p $O
  $LME --model hf --model_args "pretrained=$2,dtype=bfloat16" --tasks "$TASKS" --apply_chat_template --fewshot_as_multiturn \
      --batch_size 32 --output_path $O --log_samples > $O/run.txt 2>&1; rc=$?
  $LME --model hf --model_args "pretrained=$2,dtype=bfloat16" --tasks gsm8k --limit $GSM_LIMIT --apply_chat_template \
      --fewshot_as_multiturn --batch_size 32 --output_path $O/gsm8k --log_samples >> $O/run.txt 2>&1; rc2=$?
  [ $rc = 0 ] && [ $rc2 = 0 ]
}
job() {   # the command of a job
  case $1 in
    vg_C) vgone C $HO/C_swa3_win ;;
    vg_D) vgone D $HO/D_swa3_full ;;
    vg_A) vgone A $HO/A_kv4_full ;;
    vg_bases) vgone base_kv4_swa3 $H/hf/kv4_swa3 && vgone base_kv4 $H/hf/kv4 ;;
    ev_C) evone C $HO/C_swa3_win ;;
    ev_D) evone D $HO/D_swa3_full ;;
    ev_A) evone A $HO/A_kv4_full ;;
    ev_Cfull) evone C_fullattn $HO/C_fullattn ;;
  esac
}
ready() {   # inputs of a job exist
  case $1 in
    vg_bases) return 0 ;;
    vg_C|ev_C) [ -f $HO/C_swa3_win/done ] ;;
    vg_D|ev_D) [ -f $HO/D_swa3_full/done ] ;;
    vg_A|ev_A) [ -f $HO/A_kv4_full/done ] ;;
    ev_Cfull) [ -f $HO/C_fullattn/done ] ;;
  esac || return 1
  case $1 in ev_*) [ -f $J/predownload.rc ] ;; esac
}
dead() {   # a job can never run: its model failed to convert
  case $1 in
    vg_C|ev_C|ev_Cfull) [ -e $HO/C_swa3_win.failed ] ;;
    vg_D|ev_D) [ -e $HO/D_swa3_full.failed ] ;;
    vg_A|ev_A) [ -e $HO/A_kv4_full.failed ] ;;
    *) return 1 ;;
  esac
}
summary() {   # one line per finished job
  case $1 in
    vg_*) grep -h -E '^HF val groups' $VG/*.log 2> /dev/null | grep -E "\| ($(case $1 in vg_C) echo C ;; vg_D) echo D ;; vg_A) echo A ;; vg_bases) echo 'base_kv4_swa3|base_kv4' ;; esac)) \|" | cut -d'|' -f2-9 | sed 's/^/    /' | tee -a $ST ;;
    ev_*) local n; n=$(case $1 in ev_C) echo C ;; ev_D) echo D ;; ev_A) echo A ;; ev_Cfull) echo C_fullattn ;; esac)
          echo "    $n: $(grep -E '^\|(ifeval|gsm8k|hellaswag|arc_easy|arc_challenge|piqa|winogrande|mmlu) ' $EV/$n/run.txt 2> /dev/null | awk -F'|' '{gsub(/ /,"",$2); gsub(/ /,"",$6); gsub(/ /,"",$8); printf "%s(%s) %s  ", $2, $6, $8}' | cut -c1-330)" | tee -a $ST ;;
  esac
}
declare -A PID GPU_OF
is_free() {   # gpu: no job of ours on it, no process at all (2 samples 5 s apart)
  local g=$1 k; for k in "${!PID[@]}"; do [ "${GPU_OF[$k]}" = "$g" ] && kill -0 ${PID[$k]} 2> /dev/null && return 1; done
  [ "$(procs $g)" = 0 ] && { sleep 5; [ "$(procs $g)" = 0 ]; }
}
while :; do
  # the time limit: at 00:00 IST stop every job of ours still running, start nothing more
  if [ $(date +%s) -ge $DL ]; then
    for k in "${!PID[@]}"; do
      if kill -0 ${PID[$k]} 2> /dev/null; then
        tp=$(tree_pids ${PID[$k]}); say "00:00 IST limit: stopping $k on GPU ${GPU_OF[$k]} (our processes $tp)"
        kill -TERM $tp 2> /dev/null; sleep 10; kill -KILL $tp 2> /dev/null
      fi
      touch $J/$k.done
    done
    for k in $JOBS; do [ -f $J/$k.done ] || { say "00:00 IST limit: not started: $k"; touch $J/$k.done $J/$k.skipped; }; done
    break
  fi
  # conversions as arms finish
  for spec in C_swa3_win:kv4_swa3 D_swa3_full:kv4_swa3 A_kv4_full:kv4; do
    arm=${spec%%:*}; base=${spec#*:}
    [ -f $HO/$arm/done ] || [ -e $HO/$arm.failed ] && continue
    arm_finished $arm && convert $arm $base && [ $arm = C_swa3_win ] && make_full
  done
  # finished jobs
  for k in "${!PID[@]}"; do
    if ! kill -0 ${PID[$k]} 2> /dev/null; then
      rc=$(cat $J/$k.rc 2> /dev/null); say "end $k on GPU ${GPU_OF[$k]} exit ${rc:-?} ($(( ($(date +%s) - $(stat -c %Y $J/$k.start)) / 60 )) min)"; summary $k
      [ "$rc" = 0 ] || say "ERROR line: $k exit ${rc:-?} (see $([ ${k:0:2} = vg ] && echo valgroups || echo evals)/)"
      unset PID[$k]; touch $J/$k.done
    fi
  done
  # start jobs on free GPUs
  if [ ! -f $H/STOP_EVAL ]; then
    for g in $GPUS; do
      next=""
      for k in $JOBS; do [ -f $J/$k.done ] || [ -n "${PID[$k]}" ] || dead $k && continue; ready $k && { next=$k; break; }; done
      [ -z "$next" ] && break
      if [ $(( $(date +%s) + $(est_min $next) * 60 )) -gt $DL ]; then
        say "00:00 IST limit: $next not started (estimate $(est_min $next) min would end after 00:00)"; touch $J/$next.done $J/$next.skipped; continue
      fi
      is_free $g || continue
      touch $J/$next.start
      ( export CUDA_VISIBLE_DEVICES=$g; job $next; echo $? > $J/$next.rc ) > /dev/null 2>&1 < /dev/null &
      PID[$next]=$!; GPU_OF[$next]=$g; say "start $next on GPU $g"
    done
  fi
  # all done (or can never run)?
  left=0; for k in $JOBS; do [ -f $J/$k.done ] || dead $k || left=$((left + 1)); done
  [ $left = 0 ] && break
  sleep 30
done
# ---- final summary: the native totals against the trainer's val losses of the same weights
tv() { grep -oE '\[val\] step [0-9]+ \| val_loss [0-9.]+' $AR/$1/run.log | $2 -1 | awk '{print $NF}'; }
hv() { $PYE -c "import json; r = [x for x in json.load(open('$VG/$1.json')) if x['mode'] == '$2']; print('%.6f' % r[0]['loss']['total'] if r else '-')" 2> /dev/null; }
say "check: native HF total vs trainer val (same weights): C $(hv C native) vs $(tv C_swa3_win tail) | D $(hv D native) vs $(tv D_swa3_full tail) | A $(hv A native) vs $(tv A_kv4_full tail)"
say "check: bases vs the arms' step-0 val: kv4_swa3 window $(hv base_kv4_swa3 native) vs C $(tv C_swa3_win head) | kv4_swa3 full $(hv base_kv4_swa3 other) vs D $(tv D_swa3_full head) | kv4 full $(hv base_kv4 native) vs A $(tv A_kv4_full head)"
for k in $JOBS; do dead $k && say "not run: $k (its model failed to convert)"; done
rm -rf $TMPDIR
say "SWA25B EVAL DONE ($(( ($(date +%s) - t0) / 60 )) min) | GPUs: 7 $(procs 7), 4 $(procs 4), 5 $(procs 5) process(es)"
