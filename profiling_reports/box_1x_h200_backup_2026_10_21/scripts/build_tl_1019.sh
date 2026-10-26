#!/bin/bash
# 2026-10-19: BUILD step of the same-box per-step timeline comparison, ours vs FA3 (work doc 45.51). CPU only, ~10-15 min:
#   FA3 measurement copies at levels 2 and 3 (in parallel, each in its own folder /root/gautam_K/fa3_tl<level>, fa3_tl_build.sh)
#   and ours: TL2 / TL3 = LHOd + the same stamps (build_cands.sh; F3c goes back into the tree last).
# usage: setsid nohup bash /root/gautam_K/scripts/build_tl_1019.sh > /dev/null 2>&1 < /dev/null &   (status: /root/gautam_K/build_tl.status)
GK=/root/gautam_K; S=$GK/scripts; ST=$GK/build_tl.status
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" >> $ST; }
say "start (FA3 levels 2 + 3 in parallel, then ours TL2 TL3 F3c)"
bash $S/fa3_tl_build.sh 2 > /dev/null 2>&1 & P2=$!
bash $S/fa3_tl_build.sh 3 > /dev/null 2>&1 & P3=$!
bash $S/build_cands.sh TL2 TL3 F3c > $GK/build_tl_ours.log 2>&1; rc=$?
say "ours exit=$rc: $(tr '\n' ' ' < $GK/build_tl_ours.log)"
wait $P2; r2=$?; wait $P3; r3=$?
say "FA3 level 2 exit=$r2: $(tail -1 $GK/fa3_tl2_build.log)"
say "FA3 level 3 exit=$r3: $(tail -1 $GK/fa3_tl3_build.log)"
if [ $rc = 0 ] && [ $r2 = 0 ] && [ $r3 = 0 ]; then say "BUILD TL DONE"; else say "BUILD TL FAILED"; fi
