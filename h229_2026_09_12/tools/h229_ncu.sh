#!/usr/bin/env bash
# §H229 PHASE 1b -- tensor-core occupancy on all 15 production shapes, ours vs cuBLAS.
#
# WHY. Phase 1 gave throughput and power per shape. It cannot say WHY a shape is slow: a kernel
# at 95% tensor-pipe occupancy is doing all it can and only a better algorithm helps, while one
# at 60% is stalling and structure can be fixed. nsys cannot answer this -- it has kernel times
# and nothing about the SM's pipes. ncu can.
#
# ONE LAUNCH, MID-LOOP. --launch-skip 2 --launch-count 1 profiles a steady-state launch, not the
# cold first one. ncu replays the kernel, so a small iteration count is enough and correctness is
# already established by Phase 1.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-1}; export CUDA_VISIBLE_DEVICES=$GPUIDX
export H211_FILL=3
NCU=$(command -v ncu)
BIN=./build/h211_replay
CASES=${CASES:-tools/h212_cases.txt}
LOCK=${LOCK:-1200}
OUT=${OUT:-h229_ncu}; mkdir -p $OUT
M="sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_active,sm__cycles_elapsed.avg"

restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
trap restore EXIT INT TERM HUP
nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo "=== H229b ncu tensor-core sweep  GPU$GPUIDX  lock=${LOCK}MHz  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
echo

grab(){  # $1 arm  $2 sig -> "tensor sm dram l1 cycles"
  $NCU --kernel-name regex:"bgemm_sm90_kernel|nvjet" --launch-skip 2 --launch-count 1 \
       --metrics "$M" --csv $BIN "$1" "$2" 6 2>/dev/null \
  | python3 -c "
import sys,csv
d={}
for r in csv.DictReader(l for l in sys.stdin if l.startswith('\"')):
    n=r.get('Metric Name',''); v=r.get('Metric Value','')
    if n and v:
        try: d[n]=float(v.replace(',',''))
        except: pass
g=lambda k: d.get(k,0.0)
print('%.2f %.2f %.2f %.2f %.0f'%(
  g('sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active'),
  g('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
  g('dram__throughput.avg.pct_of_peak_sustained_elapsed'),
  g('l1tex__throughput.avg.pct_of_peak_sustained_active'),
  g('sm__cycles_elapsed.avg')))"
}

: > $OUT/raw.txt
printf "%-14s %-22s | %-28s | %-28s | %s\n" "site" "shape" "OURS  tensor/sm/dram/l1" "cuBLAS tensor/sm/dram/l1" "tensor delta"
printf "%s\n" "---------------------------------------------------------------------------------------------------------------------------"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  sig=$(echo "$line" | awk '{print $1}')
  site=$(echo "$line" | grep -oE '#.*' | sed 's/# *//'); [ -z "$site" ] && site="?"
  IFS='|' read -r lay M_ N_ K_ _ _ _ _ _ _ _ _ _ <<< "$sig"
  read ot os od ol oc <<< "$(grab ours   "$sig")"
  read ct cs cd cl cc <<< "$(grab cublas "$sig")"
  echo "$site|$sig|$ot|$os|$od|$ol|$oc|$ct|$cs|$cd|$cl|$cc" >> $OUT/raw.txt
  python3 - "$site" "$lay ${M_}x${N_}x${K_}" "$ot" "$os" "$od" "$ol" "$ct" "$cs" "$cd" "$cl" <<'PY'
import sys
s,sh,ot,os_,od,ol,ct,cs,cd,cl=sys.argv[1:11]
f=lambda *v: "/".join(f"{float(x):5.1f}" for x in v)
d=float(ot)-float(ct)
print(f"{s:<14} {sh:<22} | {f(ot,os_,od,ol):<28} | {f(ct,cs,cd,cl):<28} | {d:+6.1f} pts")
PY
done < "$CASES"
restore; echo; echo "[clocks free]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
