#!/bin/bash
# Watchdog for the Tulu 3 SFT run (2026-10-22). READ-ONLY: it reads run.log, status.txt, the process list and the disk,
# and writes one line every 5 minutes to monitor.txt. Problems get their own line starting with ALERT:
#   crash text in the log, nan/inf in the train loss or grad norm, the trainer gone before the run script wrote "end",
#   no new step for 15 minutes while the trainer is alive, less than 25 GB of disk (a checkpoint save needs 17.5 GB),
#   a checkpoint that failed to save (the trainer only warns and goes on), val rising by more than 0.02.
# It stops by itself when status.txt has the run's "end" line. It never touches the run.
# usage: setsid nohup bash ~/gautam_k/monitor_tulu3_sft.sh > /dev/null 2>&1 < /dev/null &
# look:  tail -4 ~/gautam_k/runs/tulu3_sft_1022/monitor.txt ;  grep ALERT ~/gautam_k/runs/tulu3_sft_1022/monitor.txt
RUN=$HOME/gautam_k/runs/tulu3_sft_1022
# MON_LOG / MON_ST / MON_OUT pick another run's files (the 2-GPU resume: run_resume2.log, status_resume2.txt)
LOG=${MON_LOG:-$RUN/run.log}; ST=${MON_ST:-$RUN/status.txt}; OUT=${MON_OUT:-$RUN/monitor.txt}
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
note() { echo "$(now) IST $*" >> $OUT; }
last_step=-1; last_change=$(date +%s); crash_seen=0; ckwarn_seen=0
# a resumed run appends to the same log: report only the val lines that are new from now on
n_val=$(grep -cE '\[val\] step [0-9]+ ' $LOG 2>/dev/null); n_val=${n_val:-0}
prev_val=$(grep -E '\[val\] step [0-9]+ ' $LOG 2>/dev/null | tail -1 | sed -E 's/.*val_loss ([0-9.]+).*/\1/')
note "monitor start (checks every 5 min; $n_val val lines already in the log, last val ${prev_val:-none})"
while true; do
  t=$(date +%s)
  line=$(grep -E '^step +[0-9]+ ' $LOG 2>/dev/null | tail -1)
  step=$(echo "$line" | awk '{print $2}'); [ -n "$step" ] || step=-1
  train=$(echo "$line" | awk -F'|' '{print $2}' | awk '{print $2}')
  norm=$(echo "$line" | awk -F'|' '{print $5}' | awk '{print $2}')
  dt=$(echo "$line" | awk -F'|' '{print $6}' | awk '{print $2}')
  if [ "$step" != "$last_step" ]; then last_step=$step; last_change=$t; fi
  alive=$(pgrep -f '[b]luscript_sft_bin' | wc -l)
  free=$(df -BG --output=avail $RUN | tail -1 | tr -dc 0-9)
  ended=$(tail -1 $ST 2>/dev/null | grep -c ' end |')   # the LAST status line: a resumed run appends a new start line
  # ETA from the mean step time of the last 50 steps (validation and checkpoint pauses not included)
  maxs=$(grep -m1 -oE '[0-9]+ epochs = [0-9]+ steps' $LOG 2>/dev/null | awk '{print $4}')
  avg=$(grep -E '^step +[0-9]+ ' $LOG 2>/dev/null | tail -50 | awk -F'|' '{split($6, a, " "); s += a[2]; n++} END {if (n) printf "%.0f", s / n}')
  eta=""
  if [ -n "$maxs" ] && [ -n "$avg" ] && [ "$step" -ge 0 ]; then
    left=$(( (maxs - step - 1) * avg / 1000 ))
    eta=" | ETA $(TZ=Asia/Kolkata date -d "@$((t + left))" '+%H:%M') IST ($((left / 3600))h$(( left % 3600 / 60 ))m, step $step/$maxs, mean dt ${avg} ms)"
  fi
  note "step $step | train $train | norm $norm | dt ${dt} ms | trainer procs $alive | disk ${free} GB$eta"
  # new validation results
  vals=$(grep -E '\[val\] step [0-9]+ ' $LOG 2>/dev/null)
  nv=$(echo "$vals" | grep -c 'val_loss')
  if [ "$nv" -gt "$n_val" ]; then
    while read -r v; do   # a here-string, not a pipe, so prev_val survives the loop
      note "VAL $(echo "$v" | sed 's/^ *//')"
      cur=$(echo "$v" | sed -E 's/.*val_loss ([0-9.]+).*/\1/')
      if [ -n "$prev_val" ] && awk -v a="$cur" -v b="$prev_val" 'BEGIN{exit !(a > b + 0.02)}'; then
        note "ALERT val rose from $prev_val to $cur"
      fi
      prev_val=$cur
    done <<< "$(echo "$vals" | tail -n $((nv - n_val)))"
    n_val=$nv
  fi
  # problems
  since=$(tac $LOG 2>/dev/null | sed '/^=== RESUME/q' | tac)   # only the part after the last resume marker
  if [ $crash_seen -eq 0 ] && echo "$since" | grep -q -E 'what\(\)|Aborted|Segmentation fault|CUDA error|NCCL error|out of memory'; then
    note "ALERT crash text in run.log: $(echo "$since" | grep -m1 -E 'what\(\)|Aborted|Segmentation fault|CUDA error|NCCL error|out of memory')"
    crash_seen=1
  fi
  nck=$(echo "$since" | grep -c -E '\[WARN\] checkpoint (save|flush)|checkpointing disabled')
  if [ "$nck" -gt "$ckwarn_seen" ]; then
    note "ALERT checkpoint NOT saved: $(echo "$since" | grep -E '\[WARN\] checkpoint (save|flush)|checkpointing disabled' | tail -1)"
    ckwarn_seen=$nck
  fi
  if echo "$train $norm" | grep -q -i -E 'nan|inf'; then note "ALERT nan/inf at step $step: train $train norm $norm"; fi
  if [ "$alive" -eq 0 ] && [ "$ended" -eq 0 ]; then note "ALERT the trainer is not running and the run script has not ended"; fi
  if [ "$alive" -gt 0 ] && [ $((t - last_change)) -gt 900 ]; then note "ALERT no new step for $(( (t - last_change) / 60 )) min (last step $step)"; fi
  if [ -n "$free" ] && [ "$free" -lt 25 ]; then note "ALERT only ${free} GB disk left"; fi
  if [ "$ended" -gt 0 ]; then note "run ended: $(tail -1 $ST)"; note "monitor stop"; exit 0; fi
  sleep 300
done
