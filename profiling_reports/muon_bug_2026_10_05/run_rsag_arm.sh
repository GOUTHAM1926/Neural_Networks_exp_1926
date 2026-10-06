#!/usr/bin/env bash
# test_muon_rsag (Navin's ShardedMuon test) against the dist/zero sources of ONE commit, N runs, on GPU 2.
# Everything else is identical across arms: the same clone, the same libtensor/libdist build, the same
# compile line (the Makefile's own build-mpi line, read with `make -n`, which only prints).
# A commit's dist/zero is extracted OUTSIDE the repo with `git archive` (read-only for the repo).
#   usage: bash run_rsag_arm.sh <HEAD|commit> [runs]        e.g.  HEAD 20  |  a3aba3e 20  |  e6eaeba 20
set -uo pipefail
C=${1:?usage: run_rsag_arm.sh <HEAD|commit> [runs]}; N=${2:-20}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; mkdir -p "$T/logs"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
cd "$BT" || exit 1
SRC=$BT
if [ "$C" != "HEAD" ]; then
    SRC=$T/src_$C
    rm -rf "$SRC" && mkdir -p "$SRC" && git archive "$C" dist/zero | tar -x -C "$SRC" || { echo "ERROR: git archive $C"; exit 1; }
fi
BIN=$T/test_muon_rsag_$C
SUM=$T/logs/summary_$C.txt
{
echo "=== arm $C | clone HEAD $(git rev-parse --short HEAD) | TI $(git -C Tensor-Implementations rev-parse --short HEAD) | $(date '+%F %T') | GPU $CUDA_VISIBLE_DEVICES"
md5sum "$SRC"/dist/zero/src/*.c* "$SRC"/dist/zero/tests/test_muon_rsag.cpp | sed "s#$HOME#~#"
cmd=$(make --no-print-directory -n build-mpi FILE="$SRC/dist/zero/tests/test_muon_rsag.cpp" SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
[ -z "$cmd" ] && { echo "ERROR: could not read the build-mpi compile line"; exit 1; }
cmd=${cmd/-o _runner_ /-o $BIN }
if [ "$C" != "HEAD" ]; then
    cmd=${cmd//dist\/zero\/src\//$SRC\/dist\/zero\/src\/}        # that commit's optimizer sources
    cmd=${cmd/-Idist\/zero\/include/-I$SRC\/dist\/zero\/include} # ... and its headers
    for f in $(grep -o "$SRC/dist/zero/src/[A-Za-z_]*\.c[pu]*" <<<"$cmd" | sort -u); do   # a file that commit does not have
        [ -f "$f" ] || { echo "note: $f absent in $C, dropped from the compile line"; cmd=${cmd//$f/}; }
    done
fi
echo "--- compile (CPU)"; echo "$cmd" > "$T/logs/compile_cmd_$C.txt"
eval "$cmd" > "$T/logs/compile_$C.log" 2>&1 || { echo "COMPILE FAILED:"; tail -30 "$T/logs/compile_$C.log"; exit 1; }
echo "--- $N runs: mpirun -np 1"
fails=0
for i in $(seq 1 "$N"); do
    out=$T/logs/run_${C}_$i.txt
    mpirun -np 1 "$BIN" > "$out" 2>&1; rc=$?
    line=$(grep -E "^\[(sync publish|staged refresh)\] (FAIL|PASS)" "$out" | sed -E 's/ \(worst.*//' | tr '\n' ' ')
    [ $rc -ne 0 ] && fails=$((fails+1))
    printf "run %2d exit %d | %s\n" "$i" "$rc" "$line"
done
echo "--- arm $C: $fails of $N runs FAILED"
echo "--- failure signatures:"; grep -h -E "FAIL .*differ" "$T"/logs/run_${C}_*.txt | sort | uniq -c
} 2>&1 | tee "$SUM"
