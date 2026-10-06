#!/usr/bin/env bash
# Validate the V1n fix in a SCRATCH copy of Tensor-Implementations (the clone is never touched). GPU = $GPU (default 2).
#  1. cp -a the clone's TI (timestamps kept), apply patch_muon_v1n.py (md5-guarded), rebuild ONLY what changed
#  2. A/B on the SAME test binary: original libtensor vs fixed libtensor (LD_LIBRARY_PATH), alternating, N rounds
#  3. the real patched function under GPU stress (ns_probe5 part 2) and Newton-Schulz on the dumped inputs (ns_probe4)
#   usage: bash run_fix_v1n.sh [rounds]
set -uo pipefail
N=${1:-20}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; ROOT=$T/fixroot; TI=$ROOT/Tensor-Implementations; OUT=$T/logs/fix_v1n
rm -rf "$OUT"; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=${GPU:-2}
echo "=== fix V1n | clone $(git -C "$BT" rev-parse --short HEAD) | TI $(git -C "$BT/Tensor-Implementations" rev-parse --short HEAD) | $(date "+%F %T") | GPU $CUDA_VISIBLE_DEVICES"

echo "--- 1. scratch TI + patch + rebuild"
# The TI Makefile reads its neighbours (-I../Profiler/include, -I../dist/...), so the copy gets links to the clone's
# real neighbours; without them every object that includes their headers looks out of date (151 files).
rm -rf "$ROOT" "$T/ti_v1n" && mkdir -p "$ROOT" && cp -a "$BT/Tensor-Implementations" "$TI" || exit 1
for d in $(grep -o -E '\.\./[A-Za-z0-9_-]+' "$TI/Makefile" | sort -u | sed 's#\.\./##'); do
    [ -e "$BT/$d" ] && ln -s "$BT/$d" "$ROOT/$d" && echo "linked neighbour $d -> clone's $d (read only use)"
done
python3 "$T/patch_muon_v1n.py" "$TI/src/Kernels/cuda/optimizer/MuonKernels.cu" || exit 1
cd "$TI" || exit 1
mapfile -t todo < <(make -n tensor WITH_CUBLAS=1 SM_ARCH=90 2>/dev/null | grep -E " -c " | grep -o -E "src/[A-Za-z0-9_/]+\.(cu|cpp)" | sort -u)
echo "make dry run: ${#todo[@]} source file(s) to compile: ${todo[*]}"
[ "${#todo[@]}" -eq 1 ] && [ "${todo[0]}" = "src/Kernels/cuda/optimizer/MuonKernels.cu" ] \
    || { echo "STOP: expected exactly MuonKernels.cu to rebuild; not starting a large build on the shared box"; exit 1; }
make -j"$(nproc)" tensor WITH_CUBLAS=1 SM_ARCH=90 > "$OUT/build.log" 2>&1 || { echo "BUILD FAILED"; tail -20 "$OUT/build.log"; exit 1; }
sym=_ZN9OwnTensor4cuda26muon_sumsq_finalize_kernelEPKfiPf
echo "finalize kernel in fixed lib: $(strings "$TI/lib/libtensor.so" | grep -c "$sym") | in original lib: $(strings "$BT/Tensor-Implementations/lib/libtensor.so" | grep -c "$sym")"

echo "--- 2. which libtensor each run loads"
BIN=$T/test_muon_rsag_HEAD
readelf -d "$BIN" | grep -E "RPATH|RUNPATH" | sed 's/^ */   /'
orig_lib=$(ldd "$BIN" | awk '/libtensor.so/ {print $3}')
fix_lib=$(LD_LIBRARY_PATH=$TI/lib ldd "$BIN" | awk '/libtensor.so/ {print $3}')
echo "original run -> $orig_lib"; echo "fixed run    -> $fix_lib"
[ "$fix_lib" = "$TI/lib/libtensor.so" ] || { echo "STOP: LD_LIBRARY_PATH does not select the fixed lib"; exit 1; }
[ "$orig_lib" != "$fix_lib" ] || { echo "STOP: both runs would load the same lib"; exit 1; }

echo "--- 3. test_muon_rsag, same binary, original vs fixed lib, alternating, $N rounds"
fo=0; ff=0
for i in $(seq 1 "$N"); do
    mpirun -np 1 "$BIN" > "$OUT/orig_$i.txt" 2>&1; ro=$?
    mpirun -x LD_LIBRARY_PATH="$TI/lib" -np 1 "$BIN" > "$OUT/fix_$i.txt" 2>&1; rf=$?
    [ $ro -ne 0 ] && fo=$((fo+1)); [ $rf -ne 0 ] && ff=$((ff+1))
    so=$(grep -E "^\[(sync publish|staged refresh)\] (PASS|FAIL)" "$OUT/orig_$i.txt" | awk '{print $3}' | tr '\n' '/')
    sf=$(grep -E "^\[(sync publish|staged refresh)\] (PASS|FAIL)" "$OUT/fix_$i.txt" | awk '{print $3}' | tr '\n' '/')
    printf "round %2d | original lib exit %d %s | fixed lib exit %d %s\n" "$i" "$ro" "$so" "$rf" "$sf"
done
echo "RESULT: original lib $fo of $N runs FAILED | fixed lib $ff of $N runs FAILED"

echo "--- 4. the real patched function under GPU stress (ns_probe5 part 2, fixed lib)"
LD_LIBRARY_PATH=$TI/lib "$T/ns_probe5" "$T/logs/dbg2/HEAD/nsin_step4.bin" 2>&1 | sed -n '/part 2/,$p' | tee "$OUT/probe5_fixed.txt"
echo "--- 5. Newton-Schulz on the dumped inputs, 200 calls each (ns_probe4): original lib vs fixed lib"
for lib in orig fix; do
    for s in 1 2 3 4; do
        if [ $lib = fix ]; then LD_LIBRARY_PATH=$TI/lib "$T/ns_probe4" "$T/logs/dbg2/HEAD/nsin_step$s.bin" 96 96 200
        else "$T/ns_probe4" "$T/logs/dbg2/HEAD/nsin_step$s.bin" 96 96 200; fi
    done > "$OUT/probe4_$lib.txt" 2>&1
    echo "[$lib lib] library NS outputs:"; grep -E "^===|\[2\]" "$OUT/probe4_$lib.txt" | paste - - | awk '{print "   step input " $NF? $0 : ""}' | sed -E 's/.*input fingerprint ([0-9a-f]+).*\[2\][^:]*: *([0-9]+) different result.*calls: *([0-9a-f]+) x([0-9]+).*/   input \1 -> \2 result(s): \3 x\4/'
done
echo "=== done $(date '+%T')"
