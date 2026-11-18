---
name: project-ada-node-format-study-1102
description: 11-02 — 8x RTX 6000 Ada (sm_89, 48 GB) node, user SSH 10.101.0.203; plan to run the chat-format study phase 1 (198 runs) there while H200s are busy; readiness audit = NOT ready (dist build, emb bwd, scripts, sm_89 validation)
metadata:
  type: project
---

H200 nodes busy (11-02); user offered the Ada node (VS Code SSH 10.101.0.203, 8x RTX 6000 Ada 48 GB, driver 580.178.04,
CUDA 13.0) for the component-9 chat-format study (fmt_screen.sh, 198 runs, ~7 min each on H200, stop/resume via READY).
Audit (plan Q&A 192): CPU side ready (~/Downloads/sft135 data_fmt 56 GB, init_fmt, gate_fmt, base_hf).
Blockers: dist won't build for sm_89 (BluCCL multimem; ZeroKernels/grad-norm need sm_89) -> per-file arch for BluCCL;
segmented emb bwd lives in BluCCL -> cooperative on Ada; run_sft.sh/fmt_screen.sh H200 assumptions (60 GB floors,
SM_ARCH=90 path, micro-batch <= 16 rows); never validated on sm_89: BluBLAS SM89 GEMMs, Ada RMSNorm, generic attention.
User runs all box commands; ask before every box action. Related: [[project-future-blublas-tile-map-sync]].

**UPDATE 11-02 16:10:** STUDY RUNNING on all 8 GPUs (user: all 8 ours) since 16:03 IST with sft/fmt_torch_sft.py --compile
(PyTorch copy of BluTrain recipe, emulate_precision_casts on) via sft/fmt_screen_torch.sh; node folder
/mnt/volgrp03/3rd_floor/gautam_fmt (logs/status_fmt, logs/checks_status, logs/gpus.txt editable). ssh works (key added
11-02). Parity vs BluTrain H200 done (plan Q&A 193, work doc 45.132). Do NOT kill Xorg (root GDM + AnyDesk, 4 MiB/GPU).
~23 min/run, ~10 h total. Next: monitor; evaluation stage (vllm present, lm_eval missing on node python3).

**11-02 20:00:** v23 (DeepSeek V3/R1 exact) added and QUEUED (plan Q&A 195): logs/runs.txt now 208 entries (backup
runs_198.txt.bak); node code/chat_formats.py = 23 versions (old kept as chat_formats_22v.py.bak). 67/198 done at 19:30,
all exit 0. Watcher + scripts now in ~/Downloads/lc_box/fmt_study (scratchpad got wiped on resume, lost the old watcher;
monitoring gap 17:20-19:30, nothing failed). Pause/resume commands told to user (stop / gpus.txt / grid restart).

**11-03: STUDY FINISHED (208/208, work doc 45.139; report chat_format_study_report_2026_11_03.md/.docx in work docs +
~/Downloads).** Rule check vs v1 (fmt_vs_v1.py): only v17 never worse + better on m2; v11 within 0.0005 of v17.
19:10 the user chose ChatML with reserved ids (v11 layout) for safety; details in report section 9 (role words plain,
answer end id 2 recommended / id 0 acceptable -> confirm with M2, stop on 2 and 0, no new ids, start values pc, no
default system message). Next: M2-M9 on v11, v17, v1 (+v7 reference), full-length confirmation, BluTrain H200 re-run.

**11-03 19:50 eval prep (CPU only; GPU runs ONLY on the user's word, planned for 11-04 on 8x Ada):** Ada node: Python
3.10.12, torch 2.13.0+cu130, driver 580.178 / CUDA 13.0, 512 CPUs, 755 GB RAM, /mnt/volgrp03 2.0 TB free, pypi + HF
reachable, HF datasets cache already has arc / hellaswag / winogrande / mmlu / ifeval / gsm8k / piqa. vLLM 0.30.0 sits in
the SHARED ~/.local env (also our training env) -> do NOT use it; build our own venv gautam_fmt/vllmenv (vllm +
lm-eval 0.4.12). lm-eval not installed. Eval scripts: ~/Downloads/BluTrain_push/sft/fmt_{measure,lm_eval,parity,
eval_data,probes,analyze}.py + fmt_evalq.sh (written for the H200 box: change W/T/PY/VPY to the Ada paths, GPU-free
check = no compute process). To do: copy scripts, venv, eval data, probes_review.md to the user, DRY=1 queue; first
pass 54 checkpoints (v1, v10, v11, v12, v17, v7 x 3 mixes x 3 seeds). The user (19:48): use quota efficiently.

**11-11 (CPU only; the user decides every GPU run): the evaluation is SET UP and checked, nothing run on a GPU.**
- Node user blubridge, folder /mnt/volgrp03/3rd_floor/gautam_fmt; all 208 hf_fmt exports READY; evals_fmt empty.
- **vllmenv** (ours): vLLM 0.30.0, torch 2.13.0 cu130, transformers 5.20.2, triton 3.7.1, tokenizers 0.23.2,
  flashinfer 0.6.18.post1, lm-eval[ifeval,hf] 0.4.12 (accelerate 1.15.0 was missing at first: make_class() crashed).
- **Scripts** in gautam_fmt/code/sft (fmt_*.py, fmt_evalq.sh, chat_formats.py, verify_format_study.py).
- **fmt_evalq.sh edits** (laptop BluTrain_push/sft, untracked study file): T settable; ONLY regex; NLTK_DATA=$W/nltk_data
  (the shared ~/nltk_data is drwxrwxrwx, so NLTK 3.10 ignores it -> IFEval would crash); gpu_free = any compute
  process busy (Sukanya's dedup job on GPUs 0-3 since 11-11 10:29).
- **eval_data** copied (sha256 14/14).
- **Checks:** self-test 23 versions OK; val counts 2000/2000/2800 match; DRY = 54 commands.
- **First GPU step (await OK):** parity on one checkpoint:
  `cd $F/code/sft && CUDA_VISIBLE_DEVICES=4 NLTK_DATA=$F/nltk_data $V fmt_parity.py run --model $F/hf_fmt/m1_smol_v11_s1
  --format-version v11 --tokenizer $F/init_fmt/v11/tokenizer.json --val-chats $F/eval_data/val_m1_smol.jsonl
  --out $F/parity_eval/m1_smol_v11_s1 --n 200`
- **Then the queue:** `W=$F T=$F/code PY=$V VPY=$V ONLY='_v(1|7|10|11|12|17)_s[0-9]+$'
  EVAL_GPUS_FILE=$F/logs/eval_gpus.txt bash fmt_evalq.sh queue` (start with "4 5 6 7").
Work doc 45.186-45.187, plan Q&A 235-236.
**11-11 11:43: the user STOPPED the eval (15 processes stopped, GPUs 4-7 free). Parity a/b/d/e done. Now a FUTURE task: [[project_future_chat_template_eval_1111]].**
