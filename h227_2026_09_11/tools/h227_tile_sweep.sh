#!/usr/bin/env bash
# §H227 SWEEP EVERY VARIANT OF ONE TILE ACROSS EVERY PRODUCTION SHAPE IT SERVES.
#
# WHY THIS AND NOT §H225. §H225 swept one tile on ONE shape. 128x256 serves 10 of the 15
# production shapes across THREE layouts, so "the best 128x256 variant" is not a single
# question -- it is 10 questions, and they may not have the same answer. Wave quantisation,
# k-loop length and epilogue width all differ per shape.
#
# THREE DESIGN CHANGES vs §H225, each from a measured lesson:
#
#  1. RANK BY THROUGHPUT, NOT TF/W.  §H226's null control measured power sd 0.591% and
#     throughput sd 0.058% on this rig -- power is 10x noisier. A 1-round screen ranked on
#     power/efficiency is ranking noise. Throughput is the screen metric; power is recorded
#     and reported but only decides at the CONFIRM stage.
#  2. CORRECTNESS IS CHECKED PER SHAPE.  DESC_FROZEN passed on the tile it was validated on
#     and computes wrong answers on another (§H225). Presence in the registry is not evidence
#     a config is correct on THIS shape.
#  3. EQUAL WALL TIME PER VARIANT.  Iterations are derived from 2*M*N*K so a 51-GFLOP shape
#     and a 2.5-TFLOP shape both get ~TARGET_S seconds of work. A fixed iteration count makes
#     the big shapes 50x slower and the small ones too short to have a stable power window.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
PREFIX=${PREFIX:?set PREFIX, e.g. blublasBgemmSM90_bf16_tn_128x256x64}
CASES=${CASES:?set CASES, a file of  SIG|label  lines}
LOCK=${LOCK:-1200}; VTMO=${VTMO:-90}; TARGET_S=${TARGET_S:-3.0}; ASSUME_TF=${ASSUME_TF:-600}
OUT=${OUT:-h227_out}; mkdir -p $OUT
BIN=./build/h224_sweep
cp -f /tmp/so_A_li_dh0.so build/libmyblas.so

restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
# NEVER "kill ${TELE:-0}": with TELE unset that is `kill 0`, which signals the WHOLE process
# group -- it kills the ssh session that launched the run (exit 255) and leaves the sampler
# alive. tools/h225_sweep_all.sh carries the same latent bug.
kill_tele(){ [ -n "${TELE:-}" ] && [ "${TELE:-0}" -gt 0 ] 2>/dev/null && kill "$TELE" 2>/dev/null; TELE=""; }
cleanup(){ kill_tele; pkill -f "[n]vidia-smi.*--format.*-lms" 2>/dev/null; restore; }
trap cleanup EXIT INT TERM HUP

# §H224b: a leaked sampler ran 7,545 s and froze the power sensor at 84.8 W under a 609 TF
# load. Refuse to start rather than silently measure a stuck value.
STRAY=$(pgrep -f "[n]vidia-smi.*lms" | wc -l)
if [ "$STRAY" != "0" ]; then echo "!! $STRAY stray nvidia-smi sampler(s) -- clear them first"; exit 1; fi
WATCHERS=$(pgrep -f "[w]atch.*nvidia-smi" | wc -l)
if [ "$WATCHERS" != "0" ]; then echo "!! $WATCHERS watch(1) polling nvidia-smi -- clear them first"; exit 1; fi

nm -D --defined-only build/libmyblas.so | grep -oE "${PREFIX}[A-Za-z0-9_]*" | sort -u > $OUT/vars.txt
N=$(wc -l < $OUT/vars.txt)
NC=$(grep -cvE '^\s*(#|$)' "$CASES")
echo "=== H227  $PREFIX"
echo "    variants=$N   shapes=$NC   lock=${LOCK}MHz   target=${TARGET_S}s/variant   $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
[ "$N" = "0" ] && { echo "!! zero variants matched $PREFIX -- wrong prefix"; exit 1; }

measure () {   # $1 entry, $2 sig, $3 iters ; echoes "power tflops belowlock%"
  nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits -lms 100 > /tmp/h227.csv & TELE=$!
  local tf
  if [ "$1" = "cublas" ]; then tf=$(timeout $VTMO $BIN cublas "$2" "$3" 2>/dev/null | awk '{print $1}')
  else tf=$(timeout $VTMO $BIN bench "$1" "$2" "$3" 2>/dev/null | awk '{print $1}'); fi
  [ -z "$tf" ] && tf=0
  local T=$TELE; kill_tele; wait $T 2>/dev/null
  python3 -c "
import statistics
rows=[l.split(',') for l in open('/tmp/h227.csv') if l.strip()]
b=[(float(x[0]),float(x[1])) for x in rows if len(x)>=3 and float(x[2])>=95]
p=statistics.median([y[0] for y in b]) if b else 0
lo=100.0*sum(1 for y in b if y[1] < $LOCK-5)/len(b) if b else 100.0
print('%.1f %s %.1f'%(p,'${tf:-0}',lo))"
}

nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
: > $OUT/all.txt

while read -r line; do
  case "$line" in ''|\#*) continue;; esac
  # the SIG itself contains '|', so split on the LAST one only
  label="${line##*|}"; sig="${line%|*}"
  if [ "$sig" = "$line" ] || [ -z "$label" ]; then echo "!! bad case line: $line"; exit 1; fi
  nf=$(awk -F'|' '{print NF}' <<< "$sig")
  if [ "$nf" != "13" ]; then echo "!! sig has $nf fields, expected 13: $sig"; exit 1; fi
  # iterations from the shape's own FLOPs so every variant gets the same wall time
  IT=$(python3 -c "
s='$sig'.split('|'); M,Nn,K=int(s[1]),int(s[2]),int(s[3])
f=2.0*M*Nn*K
print(max(200,int($TARGET_S*$ASSUME_TF*1e12/f)))")
  echo
  echo "--- $label   $sig   iters=$IT"
  pass=0; fail=0; tmo=0
  : > $OUT/${label}.txt
  i=0
  while read -r v; do
    i=$((i+1))
    r=$(timeout $VTMO $BIN check "$v" "$sig" 2>&1 | tail -1 | awk '{print $1}')
    [ -z "$r" ] && r=TIMEOUT
    if [ "$r" != "PASS" ]; then
      [ "$r" = "TIMEOUT" ] && tmo=$((tmo+1)) || fail=$((fail+1))
      printf "  [%3d/%3d] %-50s %s\n" "$i" "$N" "$(echo $v|sed "s/${PREFIX}_*//")" "$r"
      echo "$v|$label|CHECKFAIL:$r|0|0|0" >> $OUT/all.txt
      continue
    fi
    pass=$((pass+1))
    read p t lo <<< "$(measure "$v" "$sig" "$IT")"
    echo "$v|$p|$t|$lo" >> $OUT/${label}.txt
    echo "$v|$label|OK|$p|$t|$lo" >> $OUT/all.txt
    printf "  [%3d/%3d] %-50s %7sW %8s TF  blw%%=%s\n" "$i" "$N" "$(echo $v|sed "s/${PREFIX}_*//")" "$p" "$t" "$lo"
  done < $OUT/vars.txt
  read p t lo <<< "$(measure cublas "$sig" "$IT")"
  echo "cublas|$p|$t|$lo" >> $OUT/${label}.txt
  echo "cublas|$label|OK|$p|$t|$lo" >> $OUT/all.txt
  printf "  %-62s %7sW %8s TF  blw%%=%s\n" "cuBLAS reference" "$p" "$t" "$lo"
  # §feedback: a census must print how many it matched
  echo "  census: $pass passed, $fail wrong-answer, $tmo timed out, of $N registered"
  echo
  echo "  top 8 by THROUGHPUT (the low-noise metric):"
  sort -t'|' -k3 -gr $OUT/${label}.txt | head -8 \
    | awk -F'|' -v pre="$PREFIX" '{n=$1; sub(pre"_*","",n); printf "    %8.1f TF  %7sW  %7.4f TF/W  %s\n",$3,$2,($2>0?$3/$2:0),n}'
done < <(grep -vE '^[[:space:]]*(#|$)' "$CASES")

restore; echo "[clocks free]  $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
