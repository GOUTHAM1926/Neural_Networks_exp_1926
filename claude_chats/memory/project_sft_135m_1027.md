---
name: project_sft_135m_1027
description: "135M production base SFT: 24-run grid (3 mixes x 4 LRs x 2 special-token arms) prepared 10-28 on the laptop; P032 run pending the user"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-27T15:10:02.935Z
---

User (10-27 evening): SFT the 135M production base on P032, finish in ~20 min, then 3 later tests (8192 kernels,
SFT env vs other frameworks, special-token ablation A/B/C with mean-init). Stopped 10-27 ~21:00 IST: "continue tmrw".
Full detail: `sft_work_documentation.md` §45.104.

**Have:**
- Navin's HF export on the laptop: `work_docs(mark-down - (.md) files )/navin_135m_checkpoints_tokenizer/`.
  - `model.safetensors` bf16, Qwen3 names, untied `lm_head`.
  - `lm_head` = BluTrain `wte` (exact); `embed_tokens` = bf16(wte x sqrt512 fp32), 0 mismatches.
  - `training_meta.json`: 512/37/4q/2kv/hd128/ffn1408/vocab49280/theta50000/embed_scale1/tied/rms1e-5/qk1e-6, dcp_loss 2.2604.
- Tokenizer = Dinesh's `/localstorage/dinesh/smol135/tokenizers/ours_49152/tokenizer.json` (vocab, merges and specials
  equal; Navin's only adds a BOS `<|endoftext|>` post-processor).
  - Specials: 0 `<|endoftext|>` (starts every pretraining doc), 1 `<|im_start|>`, 2 `<|im_end|>`; 49152-49279 reserved.
  - The README says "SFT/DPO use ChatML".
  - Rows 1/2 are near-UNTRAINED (cos 0.93 with a reserved row).
  - `assistant` = `ass` + `istant`.
- Code (uncommitted, `~/Downloads/BluTrain_push`, dist_side_gautam_lottn; 4 files +188/-32; NOT built):
  - `bluscript_sft.cpp`: embed_scale + BLU_D_MODEL/Q_HEADS/KV_HEADS/FFN_HIDDEN/VOCAB/EMBED_SCALE + loader vocab rows;
  - `SftDataLoader.h`: end_id from meta.json (absent = 50256);
  - `run_sft.sh`: OPT/WARMUP/VAL_FREQ + shape knobs;
  - `prepare_sft_data.py`: `--tokenizer` -> chatml-v1; both self-tests pass (helper libs in scratchpad pylib).
- preproduction-135M (adhitya f37c616) = our 3b80f15 + 135M trainers. Our TI f7aa40f is a superset for SFT.

**Don't have (the order for 10-28):**
1. Weight loader for Navin's safetensors (HF -> BluTrain: transpose Linear, gate_up=[gate|up], wte=lm_head,
   q/k_norm -> q/k_gamma, skip embed_tokens), OR metadata.bin + data_rank_0..7.bin from P022.
   - P032 holds only ranks 8-15 (optimizer shards).
2. Packed attention test at 4/2 heads (never tested).
3. smol-smoltalk download + prep on P032 (revision f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc).
4. Build on P032.
5. Step-0 check: loss on pretraining val ~ dcp_loss 2.26.
6. The pretraining optimizer (Muon or AdamW): ask Navin.

**SUPERSEDED 10-28 by the grid below (our 53.0 template, not ChatML; an LR sweep, not one LR).** Old proposal: smol-smoltalk; ChatML; LR 1e-4 (SmolLM2 SFT/peak ratio 0.33-0.6; our peak 3.5e-4; NOT
2e-5); same optimizer as pretraining; 128 chats/step, 2 epochs, cosine; T 4096; P032 GPUs 0-3; ~15-20 min.

P032 at 20:36: GPUs 0-4, 6 free; 5 = someone's vLLM, 7 = Sanjay's bluscript_sft_bin (check overlap with him).

**10-28 (P032 taken by a colleague: prepare only, run nothing there until the user says).** User plan: grid = data mixes
x LRs x special-token arms. Our template = the 10-15 "53.0" Tülu variant (NOT ChatML). Arm A markers plain text, arm B
`<|system|> <|user|> <|assistant|>` = new ids 49152-54 initialised as the MEAN OF THEIR SUB-PIECES (Laguna 2605.27605
§4.1.1, verified; its 100-step embedding-only warm-up = their big-MoE fix, not built, add only if B spikes). OLMo 3 §3.5.4
verified. Pretraining optimizer = Muon (shard bytes 1,176,170,496 = Muon+AdamW state; AdamW-only would be 1.61 GB).
Base benchmarks (user sheet): ours 900B avg 34.43 vs SmolLM2-135M 34.13/34.49 -> SmolLM2-135M recipe is the reference.
- Mixes (BluTrain_push sft/mixes/): m1_smol = smol-smoltalk minus explore-instruct-rewrite (NC) ~457k; m2_tulu = Tülu
  OLMo-2 0225 minus Aya + olmo_hardcoded ~766k; m3 = m1 + Tülu open_math_2_gsm8k + personas-math-grade. LRs 3e-5 1e-4
  3e-4 1e-3, Muon, 2 epochs, warmup 0.03, 128 chats/step (rows = 128/chats-per-row). 24 runs = 3 waves of 8 (1 GPU each).
- Built + verified on the laptop (all in ~/Downloads/BluTrain_push, uncommitted; work folder ~/Downloads/sft135):
  hf_to_blutrain.py (embed check 0/25.2M; numpy 37-layer reference loss 1.450 vs wrong mappings 10.9/2.6/9.3),
  blutrain_to_hf.py (round trip 410/410 bit-identical to Navin's file; chat template), prepare_sft_data.py v3
  (--template tulu-plain-v1/tulu-special-v1, BOS id 0, strip message whitespace, --mix; GPT-2 path byte-identical 5/5;
  template vs training ids 2000/2000 both arms), ref_val_loss.py (gate), trainer BLU_INIT_FROM_RAW (exact bit check) +
  BLU_EXPORT, test_varlen VARLEN_HEADS=4/2. Trainer + test NOT compiled yet (no MPI on laptop).
- Scripts: ~/Downloads/sft135/prep_all.sh (laptop data prep, PID file), sync_to_p032.sh, sft135_p032.sh build|gate|grid.
- INCIDENT 10-28 11:32: my kill hit the laptop's `systemd --user` (PID 3247, the prep job's reparent); it restarted
  (30491). Told the user. Rule: kill only by recorded PID / own setsid group, never by pattern or parent.

**10-28 12:32 IST STAGE 1 RUNNING on P032 (all 8 GPUs; user had colleagues' jobs stopped; madhu's eval queue
run_queue.py + relaunched blueval chain killed with the user's OK).** Gates passed: attention 4/2 quick 238/0, full
287/0; trainer step-0 vs numpy ref A 1.545264/1.544934, B (v2 bin) 1.570207/1.569311; 372/372 weights bit-equal.
P032 layout /localstorage/gautam_k/sft135: BluTrain (tree), bluscript_sft_bin (09022deb, stage 1), bluscript_sft_bin_v2
(1fad2ff8, + BLU_SFT_SCHEDULE cosine|linear|constant|wsd), sft135_p032.sh (running queue; NEVER edit in place —
v2 copy = sft135_p032_v2.sh with SCHEDULE + SFT_BIN), plans/stage1.txt (24 runs), runs/, logs/status (everything),
prep_p032.sh (prepenv; phase 1 d8 A/B + m1B + m2 + m3, phase 2 d0-d7 x A/B; partcache/), eval_queue.sh (evalenv
lm-eval, SmolLM2-135M-Instruct reference), data/. Laptop watcher: scratchpad/watch_p032.sh (exits on events).
Stage 1 speed: 128 chats 0.49 s/step x 6,996 = ~57 min; 64 chats ~68 min; m1 val step 0 = 1.425918. Work doc §45.105.
Next: stage 2 (best LR,batch x 4 schedules x 2 arms) with SFT_BIN=bluscript_sft_bin_v2 via sft135_p032_v2.sh.
**10-28 13:00: P032 GPUs 0-3 ONLY are ours (user); 4-7 go to colleagues from 14:00 — never schedule there.** Queues
restarted with GPUS="0 1 2 3" (grid, old script) and EVAL_GPUS="0 1 2 3" (eval_queue_v2.sh); runs already on 4-7 end
~13:37 / ~13:58. Stage-1 time at 4 GPUs: per step ~0.18 s fixed + 0.0129 s per row -> c32 runs ~130 min; full stage 1
ends ~19:30, plan ~04:00; suggested to the user: drop the c32 row / losing LRs after the c128 results (no answer yet).

**10-28 afternoon (stage 1 STOPPED by the user 14:55; GPUs 0-5 -> 8192 benchmarks).**
- Stage-1 final val (m1 arm A cosine; base 1.4259): c128 1e-6 1.2039, 3e-6 1.1464, 1e-5 1.0804, 3e-5 1.0252, 1e-4 0.9900,
  3e-4 1.0023; c64 1e-6 1.1841, 3e-6 1.1281, 1e-5 1.0674, 3e-5 1.0280. c64 1e-4 / 3e-4 KILLED (resumable: checkpoints
  every 1000 steps in runs/s1_m1A_c64_lr1e-4, runs/s1_m1A_c64_lr3e-4_r2).
- Evals (lm-eval, chat template): low val loss != best model: lr1e-4 IFEval 0.405 (> SmolLM2-Instruct 0.366) but ARC-e
  0.492 vs base 0.543 (forgetting); lr1e-5 keeps knowledge (ARC-e 0.538), IFEval 0.354.
- HANG x2: a training run SHARING its GPU with an lm_eval stopped dead (100% util, ~127 W, no log) at step 273 and right
  after the step-1000 checkpoint. Never share a training GPU. Root cause not found (gdb blocked by ptrace_scope).
- Framework comparison traps: Navin's HF folder is UNTIED (lm_head + embed*sqrt512) = right for eval, WRONG for training;
  HF Qwen3 uses rms_norm_eps 1e-5 for q/k-norm (ours 1e-6); HF Qwen3 embedding padding_idx = pad id 0 = our BOS/EOS
  (that row would never train). User decision: Muon in ALL frameworks (port ours: Moonshot match_rms_adamw 0.2*sqrt(max),
  NS 5 bf16 after fp32 Frobenius scale eps 1e-8, nesterov 0.95, decoupled wd; grad clip covers only the AdamW group).
- 8192 benchmark setup: laptop profiling_reports/sft_benchmarks_8192_2026_10_28 -> P032 gautam_k/sftb8k (run_8k.sh lanes +
  run_8k_v2.sh shared job queue, 10 head ratios x T 8192/4096). Framework venvs gautam_k/fw/env_* share benv torch via .pth.
- **10-28 17:30 state:** 8192 benchmarks DONE (all GQA shapes beat FA3/FA4/cuDNN; work doc §45.106); framework
  comparison part 1 DONE (TRL / TRL+Liger / Unsloth match ours step by step; ours 1.75-2.8x faster; Liger -4 GB memory);
  code in profiling_reports/sft_frameworks_2026_10_28 -> P032 gautam_k/fw. NEXT: discuss the 13 plan factors ONE BY ONE
  with the user (§45.106 part 6), then schedule the whole grid OVERNIGHT on P032 (user's plan).
