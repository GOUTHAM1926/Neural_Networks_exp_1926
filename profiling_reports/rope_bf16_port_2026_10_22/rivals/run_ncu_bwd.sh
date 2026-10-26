#!/bin/bash
# Step 3 (2026-10-23): ncu of one backward call, ours vs torch.compile, on 3 cases; GPU 3 only.
# Waits until rival_bench.py is no longer running, so the two never share the GPU. Then prints one line per kernel.
# usage (box): setsid nohup bash /root/gautam_k/rope_opt/rivals/run_ncu_bwd.sh > /dev/null 2>&1 < /dev/null &
cd /root/gautam_k/rope_opt/rivals
P=ncu_bwd_progress.txt
NCU=/usr/local/cuda-13.0/bin/ncu
PY=/root/gautam_k/rivalenv/bin/python
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
while pgrep -f "[r]ival_bench.py" > /dev/null; do sleep 20; done
echo "$(now) IST start" > $P
for cs in "1 18x6" "4 18x6" "4 32x8"; do
  set -- $cs; B=$1; H=$2
  for impl in ours compile; do
    rep=ncu_bwd_${impl}_B${B}_${H}
    CUDA_VISIBLE_DEVICES=3 $NCU --profile-from-start off --set full -f -o $rep $PY ncu_bwd.py --impl $impl --B $B --heads $H \
        > $rep.out 2>&1
    echo "$(now) IST done $rep: $(tail -1 $rep.out | cut -c1-100)" >> $P
  done
done
echo "$(now) IST ALL NCU DONE" >> $P
