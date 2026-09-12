#!/usr/bin/env bash
# §H229 PHASE 1 -- all 15 production shapes, ours vs cuBLAS, standalone.
#
# WHY THIS AND NOT A SWEEP. The trainer says 99.80% of cuBLAS in aggregate (§H228). A 0.20%
# aggregate can hide a shape that is 5% behind offset by one that is ahead, and the trainer runs
# shapes, not averages. This measures every shape separately and reports every one.
#
# WHAT MAKES IT TRUSTWORTHY (each from a lesson that cost us):
#   * calls blublasBgemmSM90_dispatch_* -- the SAME entry point the trainer calls, so "same
#     variant" is an identity, not a claim (§H211).
#   * ONE case, ONE arm, ONE process -- many configs in one process deflated some by 6-8% and
#     manufactured a phantom +10.28% (project_test2_measurement_traps).
#   * H211_FILL=3, independent splitmix64 per matrix -- a shared periodic buffer once produced a
#     fake +41.3% (§H212b).
#   * 6 s power window -- power needs ~4 s to plateau on this card; a 1 s window read cuBLAS
#     1.28x low (§H227). Iterations are derived per shape so every measurement gets ~6 s.
#   * correctness GATES the timing; a faster wrong kernel is not a result.
#   * per-case output only. No means (feedback_per_case_not_mean).
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-1}; export CUDA_VISIBLE_DEVICES=$GPUIDX
BIN=./build/h211_replay
CASES=${CASES:-tools/h212_cases.txt}
LOCK=${LOCK:-1200}; TARGET_S=${TARGET_S:-6}; ASSUME_TF=${ASSUME_TF:-850}   # measured ~813-828 TF on these shapes; undershooting it makes the loop shorter than the 4 s power plateau
OUT=${OUT:-h229_out}; mkdir -p $OUT
export H211_FILL=3

kill_tele(){ [ -n "${TELE:-}" ] && [ "${TELE:-0}" -gt 0 ] 2>/dev/null && kill "$TELE" 2>/dev/null; TELE=""; }
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
cleanup(){ kill_tele; restore; }
trap cleanup EXIT INT TERM HUP

STRAY=$(pgrep -cf "[n]vidia-smi.*lms" || true)
[ "${STRAY:-0}" != "0" ] && { echo "!! $STRAY stray nvidia-smi sampler(s) -- power would be corrupted"; exit 1; }

if [ "$LOCK" = "0" ]; then nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; else nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1; fi
echo "=== H229 PHASE 1  GPU$GPUIDX  lock=${LOCK}MHz  target=${TARGET_S}s/measurement  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
echo

measure(){   # $1 arm(ours|cublas)  $2 sig  $3 iters -> "tflops power belowlock%"
  nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 > /tmp/h229.csv & TELE=$!
  # output line is: "<tflops> <arm> <lay> b<n> it<n>"  -- take the last line's first field
  local tf; tf=$($BIN "$1" "$2" "$3" 2>/dev/null | grep -E "^[0-9.]+ +(ours|cublas) " | tail -1 | awk '{print $1}')
  local T=$TELE; kill_tele; wait $T 2>/dev/null
  python3 -c "
import statistics
# a killed sampler can leave NUL bytes / truncated lines in the csv -- skip them rather than die
b=[]
for l in open('/tmp/h229.csv', errors='ignore'):
    l=l.replace('\x00','').strip()
    if not l: continue
    x=l.split(',')
    if len(x)<3: continue
    try: p_,c_,u_=float(x[0]),float(x[1]),float(x[2])
    except ValueError: continue
    if u_>=95: b.append((p_,c_))
p=statistics.median([y[0] for y in b]) if b else 0
mhz=statistics.median([y[1] for y in b]) if b else 0
print('%s %.1f %.0f'%('${tf:-0}',p,mhz))"
}

: > $OUT/raw.txt
printf "%-16s %-26s | %-22s | %-22s | %s\n" "site" "shape" "OURS" "cuBLAS" "ours vs cuBLAS"
printf "%s\n" "--------------------------------------------------------------------------------------------------------------------------"
while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  sig=$(echo "$line" | awk '{print $1}')
  site=$(echo "$line" | grep -oE '#.*' | sed 's/# *//')
  [ -z "$site" ] && site="?"
  IFS='|' read -r lay M N K b _ _ _ _ _ _ _ _ <<< "$sig"
  IT=$(python3 -c "print(max(20,int($TARGET_S*$ASSUME_TF*1e12/(2.0*$M*$N*$K))))")

  chk=$($BIN check "$sig" 2>/dev/null | grep -oE '^(PASS|FAIL)' | tail -1)
  if [ "$chk" != "PASS" ]; then
    printf "%-16s %-26s | *** CORRECTNESS %s -- timing suppressed\n" "$site" "$lay $M x $N x $K" "${chk:-NO-RESULT}"
    echo "$site|$sig|CHECK:${chk:-NORESULT}" >> $OUT/raw.txt
    continue
  fi
  read otf opw olo <<< "$(measure ours   "$sig" "$IT")"
  read ctf cpw clo <<< "$(measure cublas "$sig" "$IT")"
  python3 - "$site" "$lay $M x $N x $K" "$otf" "$opw" "$olo" "$ctf" "$cpw" "$clo" "$OUT/raw.txt" "$sig" "$IT" <<'PY'
import sys
site,shape,otf,opw,olo,ctf,cpw,clo,raw,sig,it=sys.argv[1:12]
otf,opw,ctf,cpw=float(otf),float(opw),float(ctf),float(cpw)
r  = (100*otf/ctf-100) if ctf else float('nan')
oe = otf/opw if opw else 0; ce = ctf/cpw if cpw else 0
re_= (100*oe/ce-100) if ce else float('nan')
print("%-16s %-26s | %8.1f TF %7.1fW | %8.1f TF %7.1fW | tput %+6.2f%%  eff %+6.2f%%"%(
      site,shape,otf,opw,ctf,cpw,r,re_))
open(raw,"a").write("%s|%s|OK|%s|%s|%s|%s|%s|%s|%s\n"%(site,sig,otf,opw,olo,ctf,cpw,clo,it))
PY
done < "$CASES"
echo
echo "achieved clock (MHz): ours $(awk -F'|' '$3=="OK"{s+=$6;n++}END{printf "%.0f",s/n}' $OUT/raw.txt)   cuBLAS $(awk -F'|' '$3=="OK"{s+=$9;n++}END{printf "%.0f",s/n}' $OUT/raw.txt)"
restore; echo "[clocks free]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
