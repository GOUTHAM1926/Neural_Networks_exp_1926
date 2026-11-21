#!/bin/bash
# swa27_setup_p015.sh -- 2026-11-21 ~12:45 IST: the inputs of arm B on P015, made mostly ON P015 instead of copied (the
# user: "can't we reproduce those in P015 instead of copying, which is taking so much time?"). Since ~11:50 everything
# that leaves the boxes towards the laptop crawls (P035 -> laptop and P015 -> laptop both 0.26-0.33 MB/s, 4 parallel
# streams 0.26 MB/s in total; laptop -> P015 still ~3.2 MB/s), so swa27_copy_p015.sh (12 GB through the laptop) would
# take ~10 h. Runs on the LAPTOP; every piece is checked against P035's own md5 lists (small, fetched once).
#   fetch    the only pieces that exist nowhere else, through the slow link, zstd -10 (~145 MB): bin/sft_swa (md5
#            3af3e48d) and the bt tree WITHOUT the 320 object files (*.o; its built libraries libtensor / libblublas /
#            libdist / libprofiler included), ldp.txt, deps/yaml, the data's two meta.json files, tokenizer_base,
#            nltk_data, hf_val_groups.py, valgroups/base_kv4.json. Check: md5 of every file, P035 == P015.
#   data     the 10 .bin files of m2_tulu_A from the LAPTOP (~/Downloads/sft135, md5-identical to P035's: checked
#            2026-11-21 12:20; only the two meta.json differ, those come with fetch). Check: all 12 files == P035's list.
#   base     Navin's HF export on P015 (/localstorage/test/BluTrain/hp350m/hf/kv4 = P035's hf/kv4 in all 7 files)
#            copied to swa9_1120/hf/kv4; prepenv = a venv with P035's prepenv versions (34 pins, PyPI); the base
#            converted with the same hf_to_blutrain.py (sha256 62b03e47..., uploaded from the laptop's BluTrain_push,
#            the step-9 code) as on 2026-11-20: --hf-dir hf/kv4 --out raw/kv4 --check. Check: raw/kv4 == P035's (284
#            files by md5; meta.json included, it records the source path and the script's sha256).
#   evalenv  a venv with P035's evalenv versions (110 pins, PyPI; torch 2.14.1 cu130 build), the lm-eval task data
#            into swa9_1120/hfcache (as swa25b_eval.sh did), import checks; then swa9_1120/eval_inputs.ok.
# Not copied: P035's CUDA 13.4 runtime libraries (465 MB compressed = ~25 min on the slow link). B uses P015's CUDA 13.0
# runtime with the SAME binary and libraries: their GPU code is in our libraries (sm_90a, built on P035); in this build
# only FusedLinearGelu_sm89.o references cuBLAS (the sm89 GELU path: the 350M is SwiGLU on H200, so B never calls it);
# cuRAND only in TensorFactory.o / RNG.o (random init / RNG state: B loads every weight from the base); all 67 cudart /
# cuRAND functions the binaries import exist in P015's 13.0 libraries (checked 12:40); same driver 595.91.07 on both.
# usage: bash swa27_setup_p015.sh fetch|data|base|evalenv
PH=${1:?fetch, data, base or evalenv}
A="ssh -p 2225 -o BatchMode=yes -o ConnectTimeout=15 operator@103.42.51.252"   # P035 (read only)
B="ssh -p 2226 -o BatchMode=yes -o ConnectTimeout=15 operator@103.42.51.252"   # P015
R=/localstorage/gautam_k; H=$R/swa9_1120; S=$R/sft135
SP=/tmp/claude-1000/-home-blu-bridge016-Downloads-Neural-Networks-exp-1926/d36b6b0d-fbee-47c9-8b10-a9773d29d8dc/scratchpad
REFS=$SP/p035_refs.txt; D=$(dirname "$(readlink -f "$0")")
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST [swa27 setup $PH] $*"; }
fail() { say "FAIL: $*"; exit 1; }
t0=$(date +%s)
$B "test -d $R -a -w $R" || fail "P015:$R missing / not writable"
case $PH in
fetch)
  P="swa9_1120/bin/sft_swa swa9_1120/bt swa9_1120/ldp.txt deps/yaml sft135/data/m2_tulu_A/train/meta.json sft135/data/m2_tulu_A/val/meta.json sft135/tokenizer_base swa9_1120/nltk_data swa9_1120/hf_val_groups.py swa9_1120/valgroups/base_kv4.json"
  say "start | slow link, zstd -10 | $P (bt without *.o)"
  $A "cd $R && tar -cf - --exclude='*.o' $P | zstd -T16 -10 -q -c" | $B "mkdir -p $R && cd $R && zstd -d -q -c | tar -xf -" || fail "stream"
  say "copied in $(( $(date +%s) - t0 )) s; checking"
  LIST="cd $R && { find $P -type f ! -name '*.o' -print0 | sort -z | xargs -0 -r -P 32 -n 64 stdbuf -oL md5sum | sort -k 2; find $P -type l -printf 'LINK %p -> %l\n' | sort; }"
  LA=$($A "$LIST" | LC_ALL=C sort); LB=$($B "$LIST" | LC_ALL=C sort)   # one sort order for both boxes
  [ "$LA" = "$LB" ] || { say "FAIL: differ ($(echo "$LA" | wc -l) vs $(echo "$LB" | wc -l))"; diff <(echo "$LA") <(echo "$LB") | head; exit 1; }
  say "check: $(echo "$LA" | wc -l) entries, P035 == P015 | sft_swa $($B "md5sum < $H/bin/sft_swa | cut -c1-8")" ;;
data)
  say "start | upload from the laptop: m2_tulu_A train/val *.bin (10 files)"
  ( cd ~/Downloads/sft135 && tar -cf - data/m2_tulu_A/train/*.bin data/m2_tulu_A/val/*.bin ) | zstd -T8 -3 -q -c \
      | $B "mkdir -p $S && cd $S && zstd -d -q -c | tar -xf -" || fail "upload"
  say "uploaded in $(( $(date +%s) - t0 )) s; checking"
  got=$($B "cd $S/data/m2_tulu_A && find train val -name '*.bin' | sort | xargs md5sum" | LC_ALL=C sort -k 2)
  want=$(grep '\.bin$' $SP/m2_p035.md5 | LC_ALL=C sort -k 2)
  [ "$got" = "$want" ] || { say "FAIL: data differ"; diff <(echo "$want") <(echo "$got"); exit 1; }
  say "check: 10 .bin files == P035's md5" ;;
base)
  say "start | hf/kv4 from Navin's export on P015, prepenv (34 pins), convert raw/kv4"
  sed -n '/##PREPENV/,/##EVALENV/p' $REFS | grep -v '^##' > $SP/prepenv_freeze.txt
  $B "mkdir -p $H/hf $H/tools && rm -rf $H/hf/kv4 && cp -a /localstorage/test/BluTrain/hp350m/hf/kv4 $H/hf/kv4 && touch $H/hf/kv4/.done" || fail "copy hf/kv4"
  got=$($B "cd $H/hf/kv4 && md5sum blu_meta.json config.json generation_config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer_config.json")
  want=$(sed -n '/##HFKV4/,/##RAWKV4/p' $REFS | grep -v '^##' | grep -v '\.done')
  [ "$(echo "$got" | sort -k 2)" = "$(echo "$want" | sort -k 2)" ] || fail "hf/kv4 differs from P035's"
  say "hf/kv4: 7 files == P035's"
  scp -q -P 2226 -o BatchMode=yes ~/Downloads/BluTrain_push/sft/hf_to_blutrain.py $SP/prepenv_freeze.txt operator@103.42.51.252:$H/tools/ || fail "scp"
  $B "sha256sum < $H/tools/hf_to_blutrain.py | cut -c1-16" | grep -qx 62b03e4740d27f50 || fail "converter sha256"
  $B "set -e; [ -x $S/prepenv/bin/python3 ] || python3 -m venv $S/prepenv; $S/prepenv/bin/python3 -m pip install -q --disable-pip-version-check -r $H/tools/prepenv_freeze.txt" > $SP/setup_prepenv.log 2>&1 \
      || { tail -5 $SP/setup_prepenv.log; fail "prepenv pip"; }
  [ "$($B "$S/prepenv/bin/python3 -m pip freeze --disable-pip-version-check")" = "$(cat $SP/prepenv_freeze.txt)" ] || fail "prepenv freeze != P035's"
  say "prepenv: $(wc -l < $SP/prepenv_freeze.txt) packages, freeze == P035's ($(( $(date +%s) - t0 )) s)"
  $B "cd $H && rm -rf raw/kv4 && mkdir -p raw logs && nice -n 19 $S/prepenv/bin/python3 tools/hf_to_blutrain.py --hf-dir $H/hf/kv4 --out $H/raw/kv4 --check > logs/convert_kv4_p015.log 2>&1" \
      || { $B "tail -5 $H/logs/convert_kv4_p015.log"; fail "convert"; }
  got=$($B "cd $H/raw/kv4 && find . -type f | sort | xargs md5sum" | LC_ALL=C sort -k 2)
  want=$(sed -n '/##RAWKV4/,/##TOK/p' $REFS | grep -v '^##' | grep . | LC_ALL=C sort -k 2)
  [ "$got" = "$want" ] || { say "FAIL: raw/kv4 differs ($(diff <(echo "$want") <(echo "$got") | grep -c '^>') files)"; diff <(echo "$want") <(echo "$got") | head -6; exit 1; }
  say "check: raw/kv4 == P035's (284 files by md5, meta.json included) | $($B "grep -m1 -E 'reference forward|embed check' $H/logs/convert_kv4_p015.log | cut -c1-100")" ;;
evalenv)
  say "start | evalenv (110 pins) + lm-eval task data"
  sed -n '/##EVALENV/,/##HFKV4/p' $REFS | grep -v '^##' > $SP/evalenv_freeze.txt
  scp -q -P 2226 -o BatchMode=yes $SP/evalenv_freeze.txt operator@103.42.51.252:$H/tools/ || fail "scp"
  $B "set -e; [ -x $S/evalenv/bin/python3 ] || python3 -m venv $S/evalenv; $S/evalenv/bin/python3 -m pip install -q --disable-pip-version-check -r $H/tools/evalenv_freeze.txt" > $SP/setup_evalenv.log 2>&1 \
      || { tail -5 $SP/setup_evalenv.log; fail "evalenv pip"; }
  [ "$($B "$S/evalenv/bin/python3 -m pip freeze --disable-pip-version-check")" = "$(cat $SP/evalenv_freeze.txt)" ] || say "NOTE: evalenv freeze differs from P035's (see below)"
  diff <(cat $SP/evalenv_freeze.txt) <($B "$S/evalenv/bin/python3 -m pip freeze --disable-pip-version-check") | head -8
  say "evalenv installed ($(( $(date +%s) - t0 )) s): $($B "$S/evalenv/bin/python3 -c 'import torch, transformers, lm_eval; print(torch.__version__, transformers.__version__, \"lm_eval ok\")'" 2>&1 | tail -1)"
  $B "cd $H && export HF_HOME=$H/hfcache HF_DATASETS_TRUST_REMOTE_CODE=1 && $S/evalenv/bin/python3 -c \"
from lm_eval.tasks import TaskManager
tm = TaskManager(); names = ['ifeval', 'hellaswag', 'arc_easy', 'arc_challenge', 'piqa', 'winogrande', 'mmlu', 'gsm8k']
d = tm.load_task_or_group(names); print('task data ready:', ', '.join(names), '|', len(d), 'entries')\" > logs/predownload_p015.log 2>&1" \
      || { $B "tail -5 $H/logs/predownload_p015.log"; fail "task data"; }
  say "$($B "tail -1 $H/logs/predownload_p015.log")"
  $B "NLTK_DATA=$H/nltk_data $S/evalenv/bin/python3 -c 'import nltk; print(len(nltk.sent_tokenize(\"One. Two.\")), \"nltk ok\")'" | grep -q "2 nltk ok" || fail "nltk"
  for f in $S/evalenv/bin/lm_eval $S/prepenv/bin/python3 $S/tokenizer_base/tokenizer.json $H/hf_val_groups.py $H/hf/kv4/config.json $H/nltk_data/tokenizers/punkt_tab $H/valgroups/base_kv4.json; do
    $B "test -e $f" || fail "missing $f"; done
  $B "touch $H/eval_inputs.ok" && say "wrote eval_inputs.ok" ;;
*) fail "unknown phase $PH" ;;
esac
say "DONE $PH in $(( $(date +%s) - t0 )) s"
