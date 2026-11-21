#!/bin/bash
# swa22_prep.sh -- 2026-11-20, P035, CPU ONLY (nice 19; no GPU is touched). The inputs of the step-9 GPU checks:
#   1. the 350M pair (Navin's HF exports hp350m/hf/{kv4_swa3, kv4}, copied from P015) converted with the step-9
#      sft/hf_to_blutrain.py (prepenv: numpy + tokenizers) into BluTrain raw folders, with --check (numpy reference
#      forward); meta.json must carry swa_window 1024 / swa_global_every 4 (kv4_swa3) and swa_window 0 (kv4);
#   2. a small train split (the first 640 packed rows of sft135/data/d0_gen_A/train = 32 optimizer steps of 20 rows) for
#      a run that ENDS (so the final export and the last checkpoint with its model record are written); val = the full
#      d0_gen_A val split (396 rows, 2,000 chats, 844 of them longer than 1,024 tokens);
#   3. the window-on / window-off val loss of the HF model is computed later on the GPU (hf_val_loss.py, evalenv).
# usage: setsid nohup bash swa22_prep.sh > prep.out 2>&1 < /dev/null &
H=/localstorage/gautam_k/swa9_1120; ST=$H/status; PY=/localstorage/gautam_k/sft135/prepenv/bin/python3
D0=/localstorage/gautam_k/sft135/data/d0_gen_A
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST [swa22 prep] $*" | tee -a $ST; }
mkdir -p $H/raw $H/data/small $H/logs
say "start | hf/kv4_swa3 $(du -sh $H/hf/kv4_swa3 | cut -f1), hf/kv4 $(du -sh $H/hf/kv4 | cut -f1)"
for m in kv4_swa3 kv4; do
  rm -rf $H/raw/$m
  nice -n 19 $PY $H/bt/sft/hf_to_blutrain.py --hf-dir $H/hf/$m --out $H/raw/$m --check > $H/logs/convert_$m.log 2>&1; rc=$?
  say "convert $m: exit $rc | $(grep -m1 '^window:' $H/logs/convert_$m.log | cut -c1-150) | $(grep -m1 'embed check' $H/logs/convert_$m.log | cut -c1-90)"
  grep -A5 'reference forward' $H/logs/convert_$m.log | sed 's/^/    /' | tee -a $ST
  [ $rc = 0 ] || { say "ABORT: conversion of $m failed: $(tail -3 $H/logs/convert_$m.log | tr '\n' ' ' | cut -c1-250)"; exit 1; }
  say "  meta.json of $m: $($PY -c "import json; m=json.load(open('$H/raw/$m/meta.json')); print('swa_window', m['swa_window'], 'swa_global_every', m['swa_global_every'], '| tensors', m['tensors'])")"
done
nice -n 19 $PY - "$D0" "$H/data/small" 640 <<'PYEOF' || { say "ABORT: small split failed"; exit 1; }
import json, os, shutil, sys
import numpy as np
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(f"{dst}/train", exist_ok=True)
m = json.load(open(f"{src}/train/meta.json")); T = m["T"]
so = np.fromfile(f"{src}/train/seq_offsets.bin", dtype=np.int64); sl = np.fromfile(f"{src}/train/seq_lengths.bin", dtype=np.int32)
for name in ("inputs.bin", "labels.bin"):
    a = np.memmap(f"{src}/train/{name}", dtype=np.uint16, mode="r", shape=(m["N"], T))
    np.ascontiguousarray(a[:n]).tofile(f"{dst}/train/{name}")
np.fromfile(f"{src}/train/valid_counts.bin", dtype=np.uint32)[:n].tofile(f"{dst}/train/valid_counts.bin")
so[: n + 1].tofile(f"{dst}/train/seq_offsets.bin"); sl[: so[n]].tofile(f"{dst}/train/seq_lengths.bin")
m["N"] = n; m["subset"] = f"the first {n} rows of {src}/train (swa22_prep.sh, 2026-11-20); the other fields are the full split's"
json.dump(m, open(f"{dst}/train/meta.json", "w"), indent=1)
if os.path.lexists(f"{dst}/val"): os.remove(f"{dst}/val")
os.symlink(f"{src}/val", f"{dst}/val")
print(f"small split: {n} train rows, {int(so[n])} chats (max {int(sl[:so[n]].max())} tokens, {int((sl[:so[n]] > 1024).sum())} longer than 1024); val -> {src}/val")
PYEOF
say "small split: $(ls $H/data/small/train | tr '\n' ' ') | val -> $(readlink $H/data/small/val)"
say "SWA22 PREP DONE (CPU only; no GPU run started)"
