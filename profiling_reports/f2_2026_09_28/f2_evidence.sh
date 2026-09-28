#!/usr/bin/env bash
# F2 evidence set 2 -- the direct effects, one run per arm (counts do not wobble run to run):
#   * allocator counters at exit ([ALLOC] line, harness_alloc_print.py)  -> allocations / frees removed
#   * nsys kernel summary over ~one optimizer step (64 micro-steps)       -> kernels removed / added,
#     their GPU time, and the total number of kernel launches
# Normalised per micro-step with the loss forward kernel, which runs exactly once per micro-step.
#
#   setsid nohup bash f2_evidence.sh > ~/gautam_k/f2_evidence_nohup.out 2>&1 < /dev/null &
#   stop: kill -- -$(cat ~/gautam_k/f2_evidence/script.pgid)
set -u
BT=${BT:-$HOME/gautam_k/BluTrain}
OUT=${OUT:-$HOME/gautam_k/f2_evidence}
STEPS=${STEPS:-6}
DELAY=${DELAY:-100}      # s after launch: past init and step 0
DUR=${DUR:-30}           # s captured: a bit more than one optimizer step (~25 s); --kill=none lets the run finish and print [ALLOC]
BIN=${BIN:-$HOME/gautam_k/_runner_F2b}
MIX=${MIX:-$HOME/gautam_k/mix}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/session.log") 2>&1
ps -o pgid= -p $$ | tr -d " " > "$OUT/script.pgid"
LDP=/usr/lib/x86_64-linux-gnu:$BT/Tensor-Implementations/lib:$BT/Profiler/lib:$BT/dist/lib

echo "==== F2 evidence  $(date '+%F %T')  host=$(hostname)  GPU=$CUDA_VISIBLE_DEVICES  steps=$STEPS  nsys delay=${DELAY}s duration=${DUR}s"
[ -x "$BIN" ] || { echo "ABORT: no binary $BIN"; exit 1; }
[ "$(strings "$BIN" | grep -c '\[ALLOC\] allocs')" -ge 1 ] || { echo "ABORT: $BIN has no [ALLOC] print"; exit 1; }
[ "$(nm -D --defined-only "$BT/Tensor-Implementations/lib/libtensor.so" | grep -c cuda_accumulate_lowp_into_fp32)" -ge 1 ] || { echo "ABORT: lib is not the F2 build"; exit 1; }
md5sum "$BIN" "$BT/Tensor-Implementations/lib/libtensor.so"
nsys --version
echo "preflight GPU apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

for arm in OLD NEW; do
  if [ "$arm" = OLD ]; then NOF=1; REP=0; else NOF=0; REP=1; fi
  echo; echo "---- $arm start $(date +%T)  BLUTRAIN_NO_FUSED_GRAD_ACC=$NOF"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  cd "$BT" && rm -rf checkpoints_bluscript
  LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-} BLU_MIX_ROOT=$MIX BLU_BF16_LOGITS=1 BLU_MAX_STEPS=$STEPS BLU_CKPT=0 \
  BLUTRAIN_NO_FUSED_GRAD_ACC=$NOF BLUTRAIN_REPORT_FUSED_GRAD_ACC=$REP \
  mpirun --allow-run-as-root -np 1 \
    -x LD_LIBRARY_PATH -x BLU_MIX_ROOT -x BLU_BF16_LOGITS -x BLU_MAX_STEPS -x BLU_CKPT \
    -x BLUTRAIN_NO_FUSED_GRAD_ACC -x BLUTRAIN_REPORT_FUSED_GRAD_ACC -x CUDA_VISIBLE_DEVICES \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --delay="$DELAY" --duration="$DUR" --kill=none \
                 --force-overwrite=true -o "$OUT/$arm" "$BIN" > "$OUT/$arm.log" 2>&1
  echo "---- $arm exit=$? end $(date +%T)  step lines=$(grep -c '^step' "$OUT/$arm.log")"
  ls -la "$OUT/$arm.nsys-rep" 2>&1
  # Reports to files, never through a pipe (a SIGPIPE'd nsys leaves a truncated report).
  nsys stats --force-export=true --report cuda_gpu_kern_sum --report cuda_api_sum --format csv \
             --output "$OUT/$arm" "$OUT/$arm.nsys-rep" > "$OUT/${arm}_stats.out" 2>&1
  echo "stats exit=$?"; ls "$OUT"/${arm}_*.csv 2>&1
done

echo; echo "==== SUMMARY"
python3 - "$OUT" "$STEPS" <<'EOF'
import csv, glob, re, sys
out, steps = sys.argv[1], int(sys.argv[2])
def table(arm, rep):
    f = glob.glob(f"{out}/{arm}_{rep}.csv")
    if not f: sys.exit(f"missing {arm}_{rep}.csv")
    return list(csv.DictReader(open(f[0])))
def kern(arm):
    d = {}
    for r in table(arm, "cuda_gpu_kern_sum"):
        d[r["Name"]] = (int(r["Instances"]), int(float(r["Total Time (ns)"])))
    return d
def launches(arm):
    return sum(int(r["Num Calls"]) for r in table(arm, "cuda_api_sum") if r["Name"].startswith("cudaLaunchKernel"))
res = {}
for arm in ("OLD", "NEW"):
    k = kern(arm)
    ruler = [v[0] for n, v in k.items() if "sparse_ce_forward_kernel_vec_save_stats" in n]
    M = sum(ruler)
    txt = open(f"{out}/{arm}.log", errors="replace").read()
    al = re.search(r"\[ALLOC\] allocs (\d+) \| frees (\d+) \| cache_hits (\d+) \| cache_misses (\d+) \| cuda_mallocs (\d+) \| splits (\d+) \| merges (\d+)", txt)
    mem = re.search(r"\[MEM\] (.*)", txt)
    fused = re.search(r"\[grad-accum\].*?: (\d+)", txt)
    res[arm] = dict(k=k, M=M, L=launches(arm), al=al, mem=mem.group(1) if mem else None,
                    fused=int(fused.group(1)) if fused else None)
    print(f"{arm}: micro-steps captured (loss-forward instances) = {M}   kernel launches = {res[arm]['L']}"
          f"  -> {res[arm]['L'] / M:.1f} per micro-step" if M else f"{arm}: NO micro-steps captured")
O, N = res["OLD"], res["NEW"]
if O["M"] and N["M"]:
    print(f"\nkernel launches per micro-step: OLD {O['L'] / O['M']:.1f}  NEW {N['L'] / N['M']:.1f}  "
          f"change {N['L'] / N['M'] - O['L'] / O['M']:+.1f}")
    names = set(O["k"]) | set(N["k"])
    rows = []
    for n in names:
        oi, ot = O["k"].get(n, (0, 0)); ni, nt = N["k"].get(n, (0, 0))
        dpi = ni / N["M"] - oi / O["M"]
        if abs(dpi) >= 0.5:
            rows.append((dpi, oi / O["M"], ni / N["M"], ot / O["M"] / 1e6, nt / N["M"] / 1e6, n))
    rows.sort()
    print("\nkernels whose count per micro-step changed (OLD -> NEW), with GPU time per micro-step:")
    t_old = t_new = 0.0
    for dpi, oc, nc, otm, ntm, n in rows:
        t_old += otm; t_new += ntm
        print(f"  {oc:8.1f} -> {nc:8.1f}  ({dpi:+7.1f})   {otm:7.3f} ms -> {ntm:7.3f} ms   {n[:110]}")
    print(f"  GPU time in these kernels per micro-step: OLD {t_old:.3f} ms  NEW {t_new:.3f} ms  change {t_new - t_old:+.3f} ms")
    tot_o = sum(v[1] for v in O["k"].values()) / O["M"] / 1e6
    tot_n = sum(v[1] for v in N["k"].values()) / N["M"] / 1e6
    print(f"  all GPU kernel time per micro-step: OLD {tot_o:.2f} ms  NEW {tot_n:.2f} ms  change {tot_n - tot_o:+.2f} ms")
mic = steps * 64 - 1
for arm in ("OLD", "NEW"):
    a = res[arm]["al"]
    print(f"\n{arm} [ALLOC]: " + (f"allocs {a.group(1)} frees {a.group(2)} cache_hits {a.group(3)} cache_misses {a.group(4)} "
                                 f"cuda_mallocs {a.group(5)} splits {a.group(6)} merges {a.group(7)}" if a else "MISSING"))
    print(f"{arm} [MEM]  : {res[arm]['mem']}")
if O["al"] and N["al"]:
    da = int(O["al"].group(1)) - int(N["al"].group(1)); df = int(O["al"].group(2)) - int(N["al"].group(2))
    print(f"\nallocations removed: {da}  frees removed: {df}  over {mic} micro-steps -> {da / mic:.3f} and {df / mic:.3f} per micro-step (expected 156)")
if N["fused"] is not None:
    print(f"fused calls (NEW): {N['fused']} -> {N['fused'] / mic:.3f} per micro-step (expected 156)")
EOF
echo "==== done $(date '+%F %T')"
