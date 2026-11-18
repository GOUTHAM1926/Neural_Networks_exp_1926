#!/bin/bash
# Build the SFT trainer (train/production_zero/bluscript_sft.cpp) -> ./bluscript_sft_bin, from the BluTrain root.
#
# It is the production_zero trainer's own compile line (Makefile target _ablation_compile: same flags, includes,
# ZeRO sources and libraries as `make ... MODE=production_zero`), with the source swapped on the command line
# (a command-line TRAIN_SRC overrides the Makefile's). FUSED_RMS_ROPE=1 (the Makefile default) is required: packed
# SFT rows need the fused QK-norm + RoPE kernels, which take the per-chat positions; the sequential RoPE path has no
# positions and the trainer stops with an error if packed data meets it.
#
# usage (BluTrain root, after the libraries are built):
#   make -j$(nproc) profiler tensor dist SM_ARCH=90      # once (CPU only, several minutes)
#   bash sft/build_sft_trainer.sh [SM_ARCH, default 90] [output, default ./bluscript_sft_bin]
# The binary finds libtensor / libprofiler / libdist through its RUNPATH; libblublas.so needs LD_LIBRARY_PATH:
#   export LD_LIBRARY_PATH=$(make -s print_run_ld_path SM_ARCH=90 | tail -n 1):$LD_LIBRARY_PATH
set -e
cd "$(dirname "$0")/.."
ARCH=${1:-90}
OUT=${2:-$PWD/bluscript_sft_bin}
[ -f train/production_zero/bluscript_sft.cpp ] || { echo "run from a BluTrain checkout (train/production_zero/bluscript_sft.cpp missing)"; exit 1; }
make _ablation_compile MODE=production_zero TRAIN_SRC=train/production_zero/bluscript_sft.cpp \
     ABLATION_OUT="$OUT" FUSED_RMS_ROPE=1 SM_ARCH="$ARCH"
echo "built $OUT ($(md5sum < "$OUT" | cut -c1-8))"
