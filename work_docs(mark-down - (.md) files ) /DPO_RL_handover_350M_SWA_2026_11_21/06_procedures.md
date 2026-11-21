# Procedures: from a clean box to a trained, exported, evaluated model, and on to DPO / RL

Every command below is the one our runs used (2026-11-20/21, H200, CUDA 13.x), with paths as placeholders. Working
copies of our scripts are in `scripts/`; the Python package versions are in `env/`.

---

## 1. Access and clone

GitHub access, the token and the clone commands (non-contributor and contributor): `docs/how_to_handle_sft.md`
sections 1-2. Clone `BlubridgeAI/BluTrain`, branch `dist_side_gautam_lottn`, with its submodules
(Tensor-Implementations `dd5b61a`, BluBridge-BLAS `cfa0e92`, `dist/Context_Parallelism`). Never put a GitHub token on a
shared box or in a file others can read.

## 2. Build (CPU only)

```
cd BluTrain
make -j$(nproc) profiler tensor dist SM_ARCH=90        # the libraries: libprofiler, libtensor (+ libblublas), libdist
bash sft/build_sft_trainer.sh                          # -> ./bluscript_sft_bin  (sets USE_FUSED_RMS_ROPE=1)
make -s print_run_ld_path SM_ARCH=90                   # the library path the binary needs at run time
LD_LIBRARY_PATH=$(make -s print_run_ld_path SM_ARCH=90) ldd ./bluscript_sft_bin | grep "not found"   # must print nothing
```

- yaml-cpp is needed (the trainer's config bridge). Without root: `apt-get download libyaml-cpp-dev libyaml-cpp0.8`,
  `dpkg -x` both into a local folder, and add its `usr/include` to `CPATH`, its `usr/lib/x86_64-linux-gnu` to
  `LIBRARY_PATH` for the build and to `LD_LIBRARY_PATH` for runs.
- The libraries take ~10 min on 224 CPUs, the trainer ~1 min. Build in a separate tree from a running job's tree.

## 3. Python environments

- **Preparation / conversion** (`prepare_sft_data.py`, `hf_to_blutrain.py`, `blutrain_to_hf.py`): a venv with
  `env/prepenv_freeze.txt` (numpy 2.5.4, tokenizers 0.23.3, datasets 5.1.0, huggingface_hub 2.3.0, ...):
  `python3.12 -m venv prepenv && prepenv/bin/pip install -r env/prepenv_freeze.txt`.
- **Evaluation** (transformers, lm-eval): a venv with `env/evalenv_freeze.txt` (torch 2.14.1 with CUDA 13.0,
  transformers 5.21.0, lm_eval 0.4.13, accelerate 1.15.0, ...). Rebuilt this way on a second box on 2026-11-21 in 97 s;
  it gave the same results as the original.

## 4. The data

```
prepenv/bin/python sft/prepare_sft_data.py --mix sft/mixes/m2_tulu.json --pack ffd --seq-len 4096 \
    --template tulu-plain-v1 --tokenizer <tokenizer.json> --seed 1234 --workers 32 --out <data>/m2_tulu_A
```

- This reproduces `m2_tulu_A` (made 2026-10-28): the mix, template, tokenizer, seed and packing above are the ones its
  `meta.json` records; `--workers` changes only the speed.
- `m2_tulu.json` = `allenai/tulu-3-sft-olmo-2-mixture-0225` at revision `d91a0785...`, 2,000 val chats, without the Aya
  part and the hard-coded identity chats. Output: `train/` and `val/`, each with `inputs.bin`, `labels.bin` (65535 =
  ignore), `valid_counts.bin`, `seq_lengths.bin`, `seq_offsets.bin`, `meta.json` (format, counts, drops, the tokenizer's
  sha256, the mix, the seed). `--workers N` gives the same output as one worker.
- Chats longer than the row (4,096) are dropped, never split. Every piece (role marker, message, newline) is tokenized
  on its own; labels = answer tokens + the end token after each answer.
- DPO / RL data must use the same template and tokenization (`01_checkpoints.md` section 4).

## 5. The base model into BluTrain (when starting from a Hugging Face model)

```
prepenv/bin/python sft/hf_to_blutrain.py --hf-dir <HF base folder> --out <raw dir> --check
```

`--check` runs a numpy reference forward and compares; `meta.json` records the shape and the window (from the HF
config's `use_sliding_window`, `sliding_window`, `layer_types`). Start a run from it with `INIT_FROM_RAW=<raw dir>`.
Starting from one of our checkpoints instead: `INIT_FROM=<.../blumodel_step_5652>` (no conversion).

## 6. A smoke run first (a few minutes)

`BLU_MAX_STEPS` is ignored in SFT. Make a small train split (the first 640 rows = 32 steps of 20 rows; the val split
stays whole) and run with it: `sft/ref_val_loss.py --make-subset` writes one, or the 20-line Python block in
`scripts/swa22_prep.sh` (step 2) does it. Check in the log: the start-up lines (section 8), the step-0 val loss against
a reference (`sft/ref_val_loss.py` computes it with numpy, every chat alone), the step time and memory.

## 7. The SFT run (the exact settings of the 350M arms)

```
cd BluTrain
env RUN_DIR=<new run folder> GPUS=0 INIT_FROM_RAW=<raw kv4_swa3> \
    N_LAYERS=28 ROPE_THETA=50000 D_MODEL=1024 Q_HEADS=8 KV_HEADS=4 FFN_HIDDEN=2560 VOCAB=49280 EMBED_SCALE=1 \
    SFT_DATA=<data>/m2_tulu_A ROWS=20 B=4 EPOCHS=1 LR=1e-4 WARMUP=0.03 VAL_FREQ=500 OPT=muon EXPORT=1 NP=1 \
    BIN=<path>/bluscript_sft_bin \
    SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4 \
    bash sft/run_sft.sh
```

(That is arm C. Arm A: base `kv4`, no `SWA_*`. Arm D: base `kv4_swa3`, `SWA_WINDOW=0 SWA_MISMATCH_OK=1`. Arm B: base
`kv4`, `SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4 SWA_MISMATCH_OK=1`.) `run_sft.sh` needs the libraries on `LD_LIBRARY_PATH`
(section 2) and refuses to start with < 50 GB free disk in `RUN_DIR` or a busy GPU. One epoch takes ~48 min on one
H200. Our arm scripts with all checks: `scripts/swa24_arms_par.sh`, `scripts/swa27_armB.sh`.

**Context parallelism** (rows longer than one GPU holds, or faster long rows): add `SFT_CP=2 NP=2 GPUS=0,1` (`ROWS`
a multiple of B x NP / U). For the 350M U = 2 or 4; one H200 holds up to 114,688 tokens per row without CP.

## 8. What to check in the log (`RUN_DIR/run.log`)

| Line | Must say |
|---|---|
| `start \| muon lr ... \| trainer MD5 \| N GB free` | the binary you meant (ours: `3af3e48d`) |
| `attention : ...` | the attention you meant: `sliding window ON ... (7 global, 21 windowed): LLLG...` or `normal causal attention in all 28 layers` |
| `[sft] the base's window record matches this run (...)` | the base's window = the run's; or the `WARNING (BLU_SWA_MISMATCH_OK=1, on purpose)` line for a deliberate arm; a mismatch without the override stops the run |
| `[sft] loss: label 65535 ignored, divided by the step's global trained-token count ...` | SFT mode is on |
| `[val] step 0 \| val_loss X` | the base's val loss on your val set (ours: C 1.668373, A 1.672250) |
| `step N \| train ... \| dt ... ms \| mfu ...` | ~466-483 ms per step for the 350M on one H200 |
| `[sft] WARNING: no BLU_INIT_FROM, so this run fine-tunes RANDOM weights` | appears also when starting with `INIT_FROM_RAW`; then it is harmless, the step-0 val loss shows the base was loaded |

Other files: `sft_muon.csv` (every step: losses, LR, grad norm, trained tokens), `gpu.csv` (memory, power, clocks every
30 s), `checkpoints/blumodel_step_*` (+ `model_config.json`), `final.safetensors` (with `EXPORT=1`).

## 9. Resume

`RESUME=1` with the same `RUN_DIR` and settings continues the newest checkpoint (skips exactly the micro-batches
already trained; logs to `run_resume_*.log`). Resumed runs match the stopped ones at the resume step.

## 10. Export to Hugging Face and check the export

```
prepenv/bin/python sft/blutrain_to_hf.py --export <RUN_DIR>/final.safetensors --base-hf <HF base folder> \
    --tokenizer <tokenizer.json> --out <hf out>
```

It writes bf16 `model.safetensors`, `config.json` with the run's window (`use_sliding_window`, `sliding_window`,
`layer_types`), the tokenizer and the `tulu-plain-v1` chat template. Check that the export is the trained model:

```
evalenv/bin/python scripts/hf_val_groups.py <hf out> <data>/m2_tulu_A --name <tag> --json <out.json>
```

Its `trainer_rows` total must equal the trainer's final val loss within ~0.0001 (ours: 0.00001-0.00006). It also gives
the val loss by chat length (<= / > 1,024 tokens) and by position (0-1,023 / 1,024-2,047 / 2,048+), with the model's
own attention (`native`) and the other type (`other`: window <-> full).

## 11. Benchmarks (the dev suite)

```
export HF_HOME=<cache> NLTK_DATA=<private punkt_tab copy> HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
lm_eval --model hf --model_args "pretrained=<hf out>,dtype=bfloat16" \
    --tasks ifeval,hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu --apply_chat_template --fewshot_as_multiturn \
    --batch_size 32 --output_path <out> --log_samples
lm_eval --model hf --model_args "pretrained=<hf out>,dtype=bfloat16" --tasks gsm8k --limit 400 \
    --apply_chat_template --fewshot_as_multiturn --batch_size 32 --output_path <out>/gsm8k --log_samples
```

~19 min per model on one H200. **NLTK trap:** IFEval scoring needs NLTK's `punkt_tab`, and NLTK refuses a data folder
owned by another account or writable by others: keep a private copy (yours, no group / other write) and point
`NLTK_DATA` to it. Use the HF backend for window models (vLLM ignores the per-layer window).

## 12. Start DPO from a checkpoint (BluTrain)

```
BLU_INIT_FROM=<.../swa350m_sft_C_window_window/blumodel_step_5652>  BLU_N_LAYERS=28  BLU_ROPE_THETA=50000
BLU_D_MODEL=1024  BLU_Q_HEADS=8  BLU_KV_HEADS=4  BLU_FFN_HIDDEN=2560  BLU_VOCAB=49280  BLU_EMBED_SCALE=1
BLU_SWA_WINDOW=1024  BLU_SWA_GLOBAL_EVERY=4        # C (and B); nothing for A and D
```

- Weights only are loaded; DPO starts with a fresh optimizer. The reference model = the same checkpoint, same window.
- The window check at start-up compares these flags with `model_config.json`.
- Data: the `tulu-plain-v1` format; prompt ending with `<|assistant|>\n`, answers ending with `<|endoftext|>` (id 0).
- If the DPO trainer is your own code calling our kernels: per layer `window = 1024` unless (i + 1) % 4 == 0, else 0;
  packed sequences go into the chat plan like SFT chats (`05_cp_and_swa.md` section 5).

## 13. RL

- Policy and reference from C's `hf/` (HF-based trainers) or `blumodel_step_5652/` (BluTrain).
- Rollouts: generate with the window (transformers; or vLLM / SGLang patched for per-layer windows, checked against
  transformers on a prompt > 1,024 tokens). Unpatched vLLM / SGLang silently use full attention (+0.23 loss past
  position 2,048 for C).
- Policy updates with the window (BluTrain flags of section 12, or a trainer that honours `layer_types`).
- Trained context: 4,096 tokens (RoPE theta 50,000); longer rollouts are untested.

## 14. Pitfalls we hit (and the fixes)

| Pitfall | Effect | Fix |
|---|---|---|
| `ROPE_THETA` / `EMBED_SCALE` not given | no error, a worse model (1.7B: val 1.1757 with the right theta, 1.2301 with 10,000) | always pass the base's values |
| `VOCAB` not given for the 135M / 350M tokenizer | the loader stops (data 49,280 rows vs built-in 50,304) | `VOCAB=49280` |
| Window flags differ from the base's record | the run stops at start-up | set the base's window, or `SWA_MISMATCH_OK=1` on purpose |
| A `BLU_*` flag given to `run_sft.sh` directly | silently dropped (the script clears them) | a `BIN` wrapper (`04_flags_and_settings.md` section 5) |
| `BLU_MAX_STEPS` for a short SFT run | ignored | a small data split (section 6) |
| < 50 GB free disk in `RUN_DIR` | `run_sft.sh` refuses to start | another disk (each checkpoint of the 350M is 3.8 GB, 2 kept) |
| NLTK data owned by another account | IFEval scoring fails | private `punkt_tab` copy + `NLTK_DATA` |
| vLLM / SGLang for a window model | silent full attention | transformers, or patch + logits check |
| Chats longer than 4,096 tokens | dropped at preparation | longer rows (`SFT_T`, CP) if they matter |
