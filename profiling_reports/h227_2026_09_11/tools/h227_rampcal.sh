#!/usr/bin/env bash
# §H227 POWER-RAMP CALIBRATION. The smoke gate read cuBLAS at 324.9 W and our kernels at
# 406-422 W on the SAME shape at the same ~597 TFLOPS. A 1.28x power ratio is impossible
# (§H212b measured +2.73%), and §H225 read 523.6 W at 596.5 TF on this rig. So the 1 s
# window is measuring a RAMP, not a steady state -- and the two arms ramp at different
# rates, which makes any arm-vs-arm power comparison at that window meaningless.
# Measure where power plateaus instead of assuming a number.
set -u
cd ~/gautam_k/blublas
export PATH=/usr/local/cuda/bin:$PATH
GPUIDX=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPUIDX
SIG=${SIG:-"tn|3072|16384|8192|1|0|0|0|1|0|0|0|0"}
LOCK=${LOCK:-1200}; ASSUME_TF=${ASSUME_TF:-600}
BIN=./build/h224_sweep
cp -f /tmp/so_A_li_dh0.so build/libmyblas.so
restore(){ nvidia-smi -i $GPUIDX -rgc >/dev/null 2>&1; }
kill_tele(){ [ -n "${TELE:-}" ] && [ "${TELE:-0}" -gt 0 ] 2>/dev/null && kill "$TELE" 2>/dev/null; TELE=""; }
cleanup(){ kill_tele; pkill -f "[n]vidia-smi.*--format.*-lms" 2>/dev/null; restore; }
trap cleanup EXIT INT TERM HUP
nvidia-smi -i $GPUIDX -lgc $LOCK >/dev/null 2>&1
echo "=== ramp calibration on $SIG  lock=${LOCK}MHz"
printf "%-10s %6s | %-22s | %-22s\n" "target_s" "iters" "OURS (prod _template)" "cuBLAS"
for TS in 1 2 4 8 16 24; do
  IT=$(python3 -c "
s='$SIG'.split('|'); M,N,K=int(s[1]),int(s[2]),int(s[3])
print(max(200,int($TS*$ASSUME_TF*1e12/(2.0*M*N*K))))")
  line=""
  for ARM in blublasBgemmSM90_bf16_tn_128x256x64_template cublas; do
    nvidia-smi -i $GPUIDX --query-gpu=power.draw,clocks.sm,utilization.gpu \
      --format=csv,noheader,nounits -lms 100 > /tmp/h227cal.csv & TELE=$!
    if [ "$ARM" = "cublas" ]; then tf=$($BIN cublas "$SIG" $IT 2>/dev/null | awk '{print $1}')
    else tf=$($BIN bench "$ARM" "$SIG" $IT 2>/dev/null | awk '{print $1}'); fi
    T=$TELE; kill_tele; wait $T 2>/dev/null
    r=$(python3 -c "
import statistics
rows=[l.split(',') for l in open('/tmp/h227cal.csv') if l.strip()]
b=[float(x[0]) for x in rows if len(x)>=3 and float(x[2])>=95]
# also the LAST-HALF median: if power is still ramping the two differ
h=b[len(b)//2:] if b else []
print('%.1f %.1f %d'%(statistics.median(b) if b else 0, statistics.median(h) if h else 0, len(b)))")
    set -- $r
    line="$line | ${1}W last½=${2}W n=${3} ${tf:-0}TF"
  done
  printf "%-10s %6s %s\n" "$TS" "$IT" "$line"
done
restore; echo "[clocks free]"
