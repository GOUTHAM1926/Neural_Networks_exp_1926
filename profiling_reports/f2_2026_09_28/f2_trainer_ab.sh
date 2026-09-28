#!/usr/bin/env bash
# F2 trainer A/B: the 1.7B dist_side_merge trainer (bluscript_zero_ckpt.cpp), one H200, B=4, T=4096,
# global batch 1,048,576 tokens -> 64 micro-steps per optimizer step.
#
#   OLD arm: BLUTRAIN_NO_FUSED_GRAD_ACC=1     -> grad += update.as_type(fp32)  (the pre-F2 path)
#   NEW arm: fused path, BLUTRAIN_REPORT_FUSED_GRAD_ACC=1 prints the fused-call count at exit
#
# ONE binary for both arms (only the env switch differs), same data, order OLD-NEW-OLD-NEW, STEPS steps
# each. nvidia-smi only BETWEEN runs, never during. Checkpoints off and deleted before every run.
#
#   setsid nohup bash f2_trainer_ab.sh > ~/gautam_k/f2_ab/nohup.out 2>&1 < /dev/null &
#   watch:  tail -f ~/gautam_k/f2_ab/session.log
#   stop:   kill -- -$(cat ~/gautam_k/f2_ab/script.pgid)
#   smoke:  STEPS=5 ORDER="NEW" OUT=~/gautam_k/f2_smoke bash f2_trainer_ab.sh
set -u
BT=${BT:-$HOME/gautam_k/BluTrain}
OUT=${OUT:-$HOME/gautam_k/f2_ab}
STEPS=${STEPS:-20}
ORDER=${ORDER:-"OLD NEW OLD NEW"}
BIN=${BIN:-$HOME/gautam_k/_runner_F2}
MIX=${MIX:-$HOME/gautam_k/mix}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
ps -o pgid= -p $$ | tr -d " " > "$OUT/script.pgid"
LDP=/usr/lib/x86_64-linux-gnu:$BT/Tensor-Implementations/lib:$BT/Profiler/lib:$BT/dist/lib

echo "==== F2 trainer A/B  $(date '+%F %T')  host=$(hostname)  GPU=$CUDA_VISIBLE_DEVICES  steps=$STEPS  order=[$ORDER]"
[ -x "$BIN" ] || { echo "ABORT: no binary $BIN"; exit 1; }
NSYM=$(nm -D --defined-only "$BT/Tensor-Implementations/lib/libtensor.so" | grep -c cuda_accumulate_lowp_into_fp32)
echo "F2 symbol in libtensor.so: $NSYM"; [ "$NSYM" -ge 1 ] || { echo "ABORT: lib is not the F2 build"; exit 1; }
md5sum "$BIN" "$BT/Tensor-Implementations/lib/libtensor.so"
LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-} ldd "$BIN" | grep -E "libtensor|libdist|libprofiler|libblublas"
for d in commoncrawl code math; do [ -e "$MIX/$d" ] || { echo "ABORT: $MIX/$d missing"; exit 1; }; done
echo "preflight GPU apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

i=0
for arm in $ORDER; do
  i=$((i + 1)); tag=run${i}_${arm}; log=$OUT/$tag.log
  if [ "$arm" = OLD ]; then NOF=1; REP=0; else NOF=0; REP=1; fi
  echo; echo "---- $tag start $(date +%T)  BLUTRAIN_NO_FUSED_GRAD_ACC=$NOF"
  nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  cd "$BT" && rm -rf checkpoints_bluscript
  LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-} BLU_MIX_ROOT=$MIX BLU_BF16_LOGITS=1 BLU_MAX_STEPS=$STEPS BLU_CKPT=0 \
  BLUTRAIN_NO_FUSED_GRAD_ACC=$NOF BLUTRAIN_REPORT_FUSED_GRAD_ACC=$REP \
  mpirun --allow-run-as-root -np 1 \
    -x LD_LIBRARY_PATH -x BLU_MIX_ROOT -x BLU_BF16_LOGITS -x BLU_MAX_STEPS -x BLU_CKPT \
    -x BLUTRAIN_NO_FUSED_GRAD_ACC -x BLUTRAIN_REPORT_FUSED_GRAD_ACC -x CUDA_VISIBLE_DEVICES \
    "$BIN" > "$log" 2>&1
  rc=$?
  echo "---- $tag exit=$rc end $(date +%T)  step lines=$(grep -c '^step' "$log")  resumed=$(grep -ci 'resum\|loaded DCP' "$log")"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
done

echo; echo "==== SUMMARY  (tok/s window = steps 2..$((STEPS - 1)), same for every run)"
python3 - "$OUT" "$STEPS" "$ORDER" <<'EOF'
import re, sys, statistics as st
out, steps, order = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
runs = []
for i, arm in enumerate(order, 1):
    tag = f"run{i}_{arm}"; txt = open(f"{out}/{tag}.log", errors="replace").read()
    rows = re.findall(r"^step\s+(\d+) \| train ([-\d.naninf]+) .*?\| tok/s ([\d.]+) .*?\| dt ([\d.]+) ms", txt, re.M)
    step = {int(s): (float(l), float(t), float(d)) for s, l, t, d in rows}
    win = [step[s][1] for s in range(2, steps) if s in step]
    mem = re.search(r"\[MEM\] (.*)", txt); cnt = re.search(r"\[grad-accum\].*: (\d+)", txt)
    runs.append(dict(tag=tag, arm=arm, n=len(step), step=step, win=win, mem=mem.group(1) if mem else "MISSING",
                     cnt=int(cnt.group(1)) if cnt else None))
for r in runs:
    w = r["win"]
    print(f"{r['tag']:<10} steps={r['n']:<3} tok/s mean={st.mean(w):9.1f} median={st.median(w):9.1f} "
          f"(n={len(w)})" if w else f"{r['tag']:<10} steps={r['n']} NO STEPS")
    print(f"           [MEM] {r['mem']}")
    if r["cnt"] is not None:
        mic = steps * 64 - 1
        print(f"           fused calls={r['cnt']}  -> per micro-step (after the first) = {r['cnt'] / mic:.3f}"
              f"  (expected: the number of bf16 weight tensors, a whole number)")
for arm in ("OLD", "NEW"):
    m = [st.mean(r["win"]) for r in runs if r["arm"] == arm and r["win"]]
    if m: print(f"{arm}: mean tok/s per run = {', '.join(f'{x:.1f}' for x in m)}  -> {st.mean(m):.1f}")
o = [st.mean(r["win"]) for r in runs if r["arm"] == "OLD" and r["win"]]
n = [st.mean(r["win"]) for r in runs if r["arm"] == "NEW" and r["win"]]
if o and n: print(f"NEW vs OLD tok/s: {100 * (st.mean(n) / st.mean(o) - 1):+.2f}%   "
                  f"(OLD run-to-run spread: {100 * (max(o) - min(o)) / st.mean(o):.2f}%)")
# loss: NEW-vs-OLD difference compared with OLD-vs-OLD (the trainer is not bit-reproducible run to run)
def loss_gap(a, b):
    common = [s for s in a["step"] if s in b["step"]]
    return st.mean(abs(a["step"][s][0] - b["step"][s][0]) for s in common) if common else float("nan")
olds = [r for r in runs if r["arm"] == "OLD"]; news = [r for r in runs if r["arm"] == "NEW"]
if len(olds) >= 2: print(f"loss gap OLD vs OLD (noise): {loss_gap(olds[0], olds[1]):.4f}")
if len(news) >= 2: print(f"loss gap NEW vs NEW (noise): {loss_gap(news[0], news[1]):.4f}")
for a in olds:
    for b in news: print(f"loss gap {a['tag']} vs {b['tag']}: {loss_gap(a, b):.4f}")
print("step-0 loss (identical inputs, must match exactly): " + ", ".join(f"{r['tag']}={r['step'][0][0]}" for r in runs if 0 in r["step"]))
EOF
echo "==== done $(date '+%F %T')"
