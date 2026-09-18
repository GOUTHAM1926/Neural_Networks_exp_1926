#!/usr/bin/env bash
# §H230 PHASE 2 -- shape -> kernel identity, BOTH arms, standalone.
#
# WHY. Two separate questions:
#  (a) cuBLAS's in-trainer map cannot be solved from launch counts -- two of its kernels tie at
#      138,240 launches. Only a direct per-shape capture resolves it.
#  (b) Does EITHER arm pick the same kernel standalone as it does inside the trainer? Our records
#      contain a case where standalone won 101.03% while the trainer lost 96.07% on the same
#      shapes, same box, same day. If the kernel differs, every standalone number needs a caveat.
#
# ncu is used only to READ the kernel identity (name, grid, block) of one steady-state launch.
# No timing is taken here -- Phase 1 owns timing.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-1}; export CUDA_VISIBLE_DEVICES=$GPUIDX; export H211_FILL=3
BIN=./build/h211_replay; CASES=${CASES:-tools/h212_cases.txt}
OUT=${OUT:-h230_out}; mkdir -p $OUT
ident(){  # $1 arm  $2 sig -> "name|grid|block"
  ncu --kernel-name regex:"bgemm_sm90_kernel|nvjet|cutlass" --launch-skip 2 --launch-count 1 \
      --metrics sm__cycles_elapsed.avg --csv $BIN "$1" "$2" 6 2>/dev/null \
  | python3 -c "
import sys,csv,re
for r in csv.DictReader(l for l in sys.stdin if l.startswith('\"')):
    n=r.get('Kernel Name','')
    m=re.search(r'Sm90WarpspecConfig<(\d+), (\d+)',n)
    if m:
        lay=re.search(r'Sm90Layout\)?(\d)',n); L={'0':'nn','1':'nt','2':'tn'}
        n='OURS %sx%s %s'%(m.group(1),m.group(2),L.get(lay.group(1),'?') if lay else '?')
    else:
        mm=re.search(r'(nvjet_\S+|cutlass\S+)',n); n='CUB '+(mm.group(1) if mm else n[:40])
    print('%s|%s|%s'%(n,r.get('Grid Size','?'),r.get('Block Size','?'))); break
else: print('NONE|?|?')"
}
: > $OUT/raw.txt
printf "%-14s %-24s | %-26s | %s\n" "site" "shape" "OURS (standalone)" "cuBLAS (standalone)"
printf "%s\n" "-------------------------------------------------------------------------------------------------------"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  sig=$(echo "$line" | awk '{print $1}')
  site=$(echo "$line" | grep -oE '#.*' | sed 's/# *//'); [ -z "$site" ] && site="?"
  IFS='|' read -r lay M_ N_ K_ _ _ _ _ _ _ _ _ _ <<< "$sig"
  o=$(ident ours "$sig"); c=$(ident cublas "$sig")
  echo "$site|$sig|$o|$c" >> $OUT/raw.txt
  printf "%-14s %-24s | %-26s | %s\n" "$site" "$lay ${M_}x${N_}x${K_}" "$(echo $o|cut -d'|' -f1)" "$(echo $c|cut -d'|' -f1)"
done < "$CASES"
echo; echo "[done]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
