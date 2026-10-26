#!/bin/bash
# 2026-10-19 (new 2x H200 box root@31.56.109.89, CUDA 13.4): the rivals' Python environment for the attention scoreboard,
# /root/gautam_K/benv, rebuilt the way the 10-12 / 10-16 one was made (work doc 41.1):
#   torch 2.14.1+cu130 (pip: nvidia-cudnn-cu13 9.24.0.43, the cuDNN of the 10-16 scoreboard),
#   FA3 = flash-attention 94e22c9 hopper/, bf16 + head dim 128 + sm90 only, every feature of that configuration kept,
#   FA4 = flash-attn-4 from the same commit (tag fa4-v4.0.0.beta34; CuTe DSL, compiled at first use).
# FA3's setup.py uses the SYSTEM nvcc for CUDA >= 13.0 (hopper/setup.py:520), so FA3 and our kernels are built with the same
# CUDA 13.4. CUTLASS = the submodule commit 94e22c9 pins (7127592...; the laptop clone has an empty submodule folder), fetched
# with git, or as a GitHub tarball if that fails. CPU + network only (no GPU); nothing is written outside /root/gautam_K.
# usage: setsid nohup bash ~/gautam_k/scripts/benv_setup.sh > /dev/null 2>&1 < /dev/null &   (log: ~/gautam_k/benv_setup.log)
# 10-19 21:30: GK = $HOME/gautam_k (was /root/gautam_K on the Verda box of 10-19; the 8x node already has its benv)
GK=$HOME/gautam_k; E=$GK/benv; FA=$GK/flash-attention; LOG=$GK/benv_setup.log; CUT=7127592069c2fe01b041e174ba4345ef9b279671
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
# rsync kept the laptop's file owner, so git (run as root) refuses the clone ("dubious ownership"); this marks it safe for
# THIS script only (git's command-scope config, git >= 2.31; checked with git 2.43 on the box), nothing in ~/.gitconfig.
# FA3's setup.py and FA4's setuptools-scm call git as child processes and inherit it.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $LOG; }
fail() { say "FAILED: $*"; exit 1; }
[ -f $FA/hopper/setup.py ] || fail "$FA/hopper/setup.py missing (copy the flash-attention clone from the laptop first)"
say "start | flash-attention $(git -C $FA rev-parse --short HEAD 2>/dev/null) | $(nvcc --version | tail -1)"

# 1. venv + torch (the same wheels as the 10-17 lossenv)
[ -x $E/bin/python ] || python3 -m venv $E || fail "venv"
$E/bin/pip install --no-cache-dir -q --upgrade pip setuptools wheel >> $LOG 2>&1 || fail "pip upgrade"
$E/bin/pip install --no-cache-dir -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 >> $LOG 2>&1 || fail "torch"
$E/bin/pip install --no-cache-dir -q numpy packaging ninja einops psutil >> $LOG 2>&1 || fail "build tools"
say "torch ready: $($E/bin/python -c 'import torch; print(torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())' 2>&1 | tail -1)"

# 2. CUTLASS at the commit flash-attention 94e22c9 pins
if [ ! -f $FA/csrc/cutlass/include/cutlass/cutlass.h ]; then
  git -C $FA submodule update --init csrc/cutlass >> $LOG 2>&1
  if [ ! -f $FA/csrc/cutlass/include/cutlass/cutlass.h ]; then
    rm -rf $FA/csrc/cutlass && mkdir -p $FA/csrc/cutlass && \
      curl -sL https://github.com/NVIDIA/cutlass/archive/$CUT.tar.gz | tar xz --strip-components=1 -C $FA/csrc/cutlass || fail "cutlass tarball"
  fi
fi
[ -f $FA/csrc/cutlass/include/cutlass/cutlass.h ] || fail "cutlass headers missing"
got=$(git -C $FA/csrc/cutlass rev-parse HEAD 2>/dev/null || echo "tarball $CUT")
say "cutlass: $got"
case "$got" in *$CUT*) ;; *) fail "cutlass is not the pinned commit $CUT";; esac

# 2b. Python headers. The box has no python3.12-dev (first try 11:21: "flash_api_stable.cpp: fatal error: Python.h: No such
# file or directory"). Instead of installing it system-wide on a shared box, the same version's packages (3.12.3-1ubuntu0.17,
# = the installed python3.12) are downloaded and UNPACKED into $GK/pydev (apt-get download + dpkg -x: nothing is installed),
# and the compiler finds them through CPATH, for this script only.
if [ ! -f /usr/include/python3.12/Python.h ]; then
  PD=$GK/pydev
  if [ ! -f $PD/root/usr/include/python3.12/Python.h ]; then
    mkdir -p $PD/debs && (cd $PD/debs && apt-get download python3.12-dev libpython3.12-dev >> $LOG 2>&1) || fail "apt-get download of the Python headers"
    for d in $PD/debs/*.deb; do dpkg -x $d $PD/root || fail "dpkg -x $d"; done
  fi
  [ -f $PD/root/usr/include/python3.12/Python.h ] || fail "Python.h not in $PD/root"
  export CPATH=$PD/root/usr/include/python3.12:$PD/root/usr/include${CPATH:+:$CPATH}
  say "python headers: $(ls $PD/debs) unpacked in $PD (not installed)"
fi

# 3. FA3: bf16 + head dim 128 + sm90 only (as on 10-12); every feature for that configuration kept
cd $FA/hopper || fail "cd hopper"
env FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=48 NVCC_THREADS=2 \
    $E/bin/pip install --no-build-isolation --no-cache-dir -v . > $GK/benv_fa3_build.log 2>&1 || fail "FA3 build (benv_fa3_build.log)"
say "FA3 built"

# 4. FA4: flash-attn-4 from the same commit, with its CUDA 13 extra
$E/bin/pip install --no-cache-dir "$FA/flash_attn/cute[cu13]" > $GK/benv_fa4_install.log 2>&1 || fail "FA4 install (benv_fa4_install.log)"
say "FA4 installed"

# 5. imports + versions (CPU only: nothing runs on a GPU here). From $GK, not hopper/: hopper/ holds a source flash_attn_3/
# without the built _C, and Python looks in the current folder first (11:26: "No module named 'flash_attn_3._C'").
cd $GK || fail "cd $GK"
CUDA_VISIBLE_DEVICES= $E/bin/python - >> $LOG 2>&1 <<'EOF' || fail "imports"
import importlib.metadata as md
import torch, flash_attn_interface, flash_attn.cute.interface
print("CHECK torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(),
      "| flash_attn_3", md.version("flash_attn_3"), "| flash-attn-4", md.version("flash-attn-4"),
      "| nvidia-cutlass-dsl", md.version("nvidia-cutlass-dsl"), "| quack-kernels", md.version("quack-kernels"))
EOF
$E/bin/pip freeze > $GK/benv_pip_freeze.txt
say "$(grep '^CHECK' $LOG | tail -1)"
say "BENV READY | $(du -sh $E | cut -f1)"
