#!/bin/bash
# 2026-10-17 (2x box, CPU + network only, no GPU work): our own Python env for the loss comparison, /root/gautam_k/lossenv.
# torch 2.14.1 for CUDA 13.0 (the version the 8x box's rival env uses; this box has CUDA 13.0 only), then liger-kernel,
# cut-cross-entropy (Apple CCE) and transformers (liger's nn.Module wrappers import it). Nothing outside /root/gautam_k is
# written: no pip cache (PIP_NO_CACHE_DIR), no system packages. ~7 GB on disk.
# usage: bash /root/gautam_k/loss_bench/loss_env_setup.sh      (log: /root/gautam_k/lossenv_setup.log)
GK=/root/gautam_k; E=$GK/lossenv; LOG=$GK/lossenv_setup.log
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $LOG; }
say "start | $(python3 --version 2>&1) | disk free $(df -h /root | awk 'NR==2 {print $4}')"
if [ ! -x $E/bin/python ]; then
  python3 -m venv $E >> $LOG 2>&1 || { say "FAILED: python3 -m venv (the python3-venv package may be missing)"; exit 1; }
fi
$E/bin/pip install -q -U pip >> $LOG 2>&1
$E/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 >> $LOG 2>&1 \
  || { say "torch 2.14.1 (cu130) not found; trying the newest cu130 torch"
       $E/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130 >> $LOG 2>&1 || { say "FAILED: torch install (see $LOG)"; exit 1; }; }
$E/bin/pip install liger-kernel cut-cross-entropy transformers >> $LOG 2>&1 || { say "FAILED: liger / cce / transformers install (see $LOG)"; exit 1; }
$E/bin/python - <<'EOF' 2>&1 | tee -a $LOG
import torch, triton
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| triton", triton.__version__)
for m in ("liger_kernel", "cut_cross_entropy", "transformers"):
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "(no __version__)"))
    except Exception as e:
        print(m, "IMPORT FAILED:", type(e).__name__, e)
try:
    from liger_kernel.transformers import LigerCrossEntropyLoss, LigerFusedLinearCrossEntropyLoss
    from cut_cross_entropy import linear_cross_entropy
    print("imports used by loss_rivals_1017.py: OK")
except Exception as e:
    print("imports used by loss_rivals_1017.py: FAILED:", type(e).__name__, e)
EOF
say "SETUP DONE | env size $(du -sh $E | cut -f1) | disk free $(df -h /root | awk 'NR==2 {print $4}')"
