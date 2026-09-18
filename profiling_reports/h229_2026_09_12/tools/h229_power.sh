#!/usr/bin/env bash
# §H229c POWER ATTRIBUTION -- where does our +2.40% power go?
#
# Established so far: we lead tensor occupancy on 14/15 (96.59% vs 95.13%), we drive L1 LESS
# (64.73% vs 78.74%), DRAM is identical (~19%), and we are +1.72% faster standalone -- yet we
# draw +2.40% more power. Better tensor use and less data movement should cost LESS power, not
# more. So the power is going somewhere neither tensor nor memory traffic explains.
#
# Candidates this measures:
#   * threads/registers -- we launch 512 threads/block on 192x192 where cuBLAS launches 384.
#     More live registers is more static power, independent of what they compute.
#   * non-tensor instruction issue -- ALU/FMA/LSU work the tensor metric does not see.
#     feedback_vector_not_total_instructions: count the VECTOR pipes, the uniform pipe is at parity.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-1}; export CUDA_VISIBLE_DEVICES=$GPUIDX; export H211_FILL=3
BIN=./build/h211_replay; CASES=${CASES:-tools/h212_cases.txt}; LOCK=${LOCK:-1200}
OUT=${OUT:-h229_power}; mkdir -p $OUT
M="launch__registers_per_thread,launch__block_size,launch__grid_size,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,sm__cycles_elapsed.avg"
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
trap restore EXIT INT TERM HUP
nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo "=== H229c power attribution  GPU$GPUIDX  lock=${LOCK}MHz  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"; echo

grab(){
  ncu --kernel-name regex:"bgemm_sm90_kernel|nvjet" --launch-skip 2 --launch-count 1 \
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
print('%.0f %.0f %.0f %.2f %.0f %.0f %.0f %.0f %.0f'%(
 g('launch__registers_per_thread'),g('launch__block_size'),g('launch__grid_size'),
 g('sm__warps_active.avg.pct_of_peak_sustained_active'),
 g('smsp__inst_executed.sum'),g('smsp__inst_executed_pipe_alu.sum'),
 g('smsp__inst_executed_pipe_fma.sum'),g('smsp__inst_executed_pipe_lsu.sum'),
 g('sm__cycles_elapsed.avg')))"
}
: > $OUT/raw.txt
printf "%-14s | %-34s | %-34s\n" "site" "OURS reg/thr grid occ% inst/cyc" "cuBLAS reg/thr grid occ% inst/cyc"
printf "%s\n" "-------------------------------------------------------------------------------------------"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  sig=$(echo "$line" | awk '{print $1}')
  site=$(echo "$line" | grep -oE '#.*' | sed 's/# *//'); [ -z "$site" ] && site="?"
  read oR oB oG oO oI oA oF oL oC <<< "$(grab ours   "$sig")"
  read cR cB cG cO cI cA cF cL cC <<< "$(grab cublas "$sig")"
  echo "$site|$sig|$oR|$oB|$oG|$oO|$oI|$oA|$oF|$oL|$oC|$cR|$cB|$cG|$cO|$cI|$cA|$cF|$cL|$cC" >> $OUT/raw.txt
  python3 - "$site" "$oR" "$oB" "$oG" "$oO" "$oI" "$oC" "$cR" "$cB" "$cG" "$cO" "$cI" "$cC" <<'PY'
import sys
s=sys.argv[1]; oR,oB,oG,oO,oI,oC=map(float,sys.argv[2:8]); cR,cB,cG,cO,cI,cC=map(float,sys.argv[8:14])
print("%-14s | r%-4.0f t%-4.0f g%-5.0f o%-5.1f i/c %-7.2f | r%-4.0f t%-4.0f g%-5.0f o%-5.1f i/c %-7.2f"%(
 s,oR,oB,oG,oO,(oI/oC if oC else 0), cR,cB,cG,cO,(cI/cC if cC else 0)))
PY
done < "$CASES"
restore; echo; echo "[clocks free]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
