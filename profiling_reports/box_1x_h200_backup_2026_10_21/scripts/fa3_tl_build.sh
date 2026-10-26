#!/bin/bash
# 2026-10-19: build an FA3 MEASUREMENT copy (per-step clock stamps of the forward main loop, work doc 45.51). CPU only.
# Copies /root/gautam_K/flash-attention (benv's FA3 source: 94e22c9, CUTLASS 7127592) to /root/gautam_K/fa3_tl<level>/, puts the
# patched files in (scripts/fa3_tl/: mainloop_fwd_sm90_tma_gmma_ws.hpp + flash_fwd_hdim128_bf16_sm90_tl<level>.cu as
# instantiations/flash_fwd_hdim128_bf16_sm90.cu) and builds IN PLACE (python setup.py build_ext --inplace ->
# fa3_tl<level>/flash-attention/hopper/flash_attn_3/_C*.so). benv's installed flash_attn_3 (the scoreboard's FA3) is not touched.
# Level 2: stamps at the step start and when the scheduler barrier is passed (the loop's instructions stay FA3's: local SASS
# check, exps before the P.V wait 16 vs FA3's 13 of 66); level 3: + when QK(n) is done (27 of 66: a cross-check only).
# Switches: benv's (bf16, head dim 128, sm90 only) + no split / paged / softcap / PackGQA / append-KV / local / varlen /
# cluster: none of these is part of the dense causal kernel (FA3 uses clusters only without causal, flash_fwd_launch_template.h
# :212); the backward stays, so the trace runs in the scoreboard's forward + backward regime.
# usage: bash /root/gautam_K/scripts/fa3_tl_build.sh <2|3>   (log: /root/gautam_K/fa3_tl<level>_build.log)
LVL=${1:?level 2 or 3}; case $LVL in 2) INST_MD5=f5711129;; 3) INST_MD5=4a1faea9;; *) echo "level must be 2 or 3"; exit 2;; esac
GK=/root/gautam_K; SRC=$GK/flash-attention; D=$GK/fa3_tl$LVL; P=$GK/scripts/fa3_tl; E=$GK/benv; LOG=$GK/fa3_tl${LVL}_build.log
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'   # as benv_setup.sh (rsync kept the laptop owner)
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $LOG; }
fail() { say "FAILED: $*"; exit 1; }
[ -f $SRC/hopper/setup.py ] && [ -f $SRC/csrc/cutlass/include/cutlass/cutlass.h ] || fail "benv's FA3 source or its CUTLASS missing"
[ -x $E/bin/python ] || fail "benv missing"
# the patch was made from these exact files (laptop clone 94e22c9); stop if the box copy differs
[ "$(md5sum < $SRC/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp | cut -c1-8)" = 2ed79549 ] || fail "box mainloop header is not the patch base 2ed79549"
[ "$(md5sum < $SRC/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu | cut -c1-8)" = 1ecaa31c ] || fail "box instantiation file is not the patch base 1ecaa31c"
[ "$(md5sum < $P/mainloop_fwd_sm90_tma_gmma_ws.hpp | cut -c1-8)" = 5572bf6f ] || fail "patched header in $P is not 5572bf6f"
[ "$(md5sum < $P/flash_fwd_hdim128_bf16_sm90_tl$LVL.cu | cut -c1-8)" = $INST_MD5 ] || fail "patched instantiation in $P is not $INST_MD5"
rm -rf $D && mkdir -p $D && cp -a $SRC $D/ || fail "copy of $SRC"
H=$D/flash-attention/hopper
cp $P/mainloop_fwd_sm90_tma_gmma_ws.hpp $H/ && cp $P/flash_fwd_hdim128_bf16_sm90_tl$LVL.cu $H/instantiations/flash_fwd_hdim128_bf16_sm90.cu || fail "patch copy"
say "start level $LVL | copy of FA3 $(git -C $SRC rev-parse --short HEAD) in $D | patched 5572bf6f + $INST_MD5 | $(nvcc --version | tail -1)"
# Python headers as in benv_setup.sh 2b (unpacked, not installed)
[ -f /usr/include/python3.12/Python.h ] || export CPATH=$GK/pydev/root/usr/include/python3.12:$GK/pydev/root/usr/include${CPATH:+:$CPATH}
cd $H || fail "cd $H"
env FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM64=TRUE FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
    FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
    FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
    FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE FLASH_ATTENTION_DISABLE_CLUSTER=TRUE \
    MAX_JOBS=24 NVCC_THREADS=2 $E/bin/python setup.py build_ext --inplace > $GK/fa3_tl${LVL}_build_detail.log 2>&1 \
  || fail "build (tail of $GK/fa3_tl${LVL}_build_detail.log: $(tail -3 $GK/fa3_tl${LVL}_build_detail.log | tr '\n' ' '))"
SO=$(ls $H/flash_attn_3/_C*.so 2>/dev/null | head -1)
[ -n "$SO" ] || fail "no flash_attn_3/_C*.so after the build"
nm -D --defined-only $SO | grep -q ' fa3_tl_read$' || fail "fa3_tl_read not exported by $SO"
say "BUILT level $LVL: $SO ($(du -h $SO | cut -f1), md5 $(md5sum $SO | cut -c1-8)) | exports: $(nm -D --defined-only $SO | grep -o 'fa3_tl_[a-z]*' | sort -u | tr '\n' ' ')"
