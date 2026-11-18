# How to handle SFT in BluTrain

## 1. GitHub access and token

Our repositories are private. You need two things:

- **Access** to three repositories: `BlubridgeAI/BluTrain`, `BlubridgeAI/Tensor-Implementations` and `kathir-23s/BluBridge-BLAS`. Ask me for the first two and Kathir for the third.
- **A GitHub token**, which git uses in place of your password.

To make the token:

1. On github.com, open your picture (top right), then **Settings**, then **Developer settings**.
2. Open **Personal access tokens**, then **Tokens (classic)**, then **Generate new token (classic)**.
3. Tick **`repo`** and click **Generate token**.
4. Copy the token. It is shown only once.
5. If a **Configure SSO** button shows next to the token, click it and authorize BlubridgeAI.

Tell git to use your token, once per machine. Replace `<TOKEN>` with your token:

```
git config --global url."https://<TOKEN>@github.com/".insteadOf "https://github.com/"
```

Never share the token or put it in any file you commit.

## 2. Clone our branch

### Non-contributor (you only run trainings)

```
mkdir -p ~/sft_work && cd ~/sft_work
git clone -b dist_side_gautam_lottn https://github.com/BlubridgeAI/BluTrain.git
cd BluTrain
git submodule update --init --recursive
```

All three submodules are needed: Tensor-Implementations, BluBridge-BLAS inside it, and `dist/Context_Parallelism`. Since 2026-11-06 the SFT trainer build also uses the Context_Parallelism headers. Before that date this guide fetched only Tensor-Implementations (`git submodule update --init --recursive Tensor-Implementations`); corrected on 2026-11-14.

To get the latest changes later:

```
cd ~/sft_work/BluTrain
git pull
git submodule update --init --recursive
```

### Contributor (you will also push changes)

Clone exactly as above. Then, inside `BluTrain`:

```
git config user.name "Your Name"
git config user.email "your.name@blubridge.com"
git checkout -b <your_branch_name>
```

After you change files:

```
git add <the files you changed>
git commit -m "what you changed"
git push -u origin <your_branch_name>
```

Then tell me the branch name. Do not push to `dist_side_gautam_lottn` directly.

## 3. Build

```
cd ~/sft_work/BluTrain
make -j$(nproc)
```

- This builds all the libraries for the GPU in your machine. It uses only the CPU and takes a while.
- Build again after a `git pull` that changes `.cpp`, `.cu` or `.h` files.
- The SFT trainer itself is compiled by the run command in step 5.

## 4. Prepare once: Python, base model, data

```
cd ~/sft_work
python3 -m venv venv && source venv/bin/activate
pip install numpy tokenizers datasets
```

**Base model.** Put the base model folder I share with you at `~/sft_work/base_hf`, then convert it for the trainer:

```
python3 BluTrain/sft/hf_to_blutrain.py --hf-dir base_hf --out ckpt_A
```

**Data.** This downloads the chats, writes them with our chat template, and packs them into rows:

```
python3 BluTrain/sft/prepare_sft_data.py --mix BluTrain/sft/mixes/m1_smol.json \
    --tokenizer base_hf/tokenizer.json --template tulu-plain-v1 --pack ffd --out data/m1_smol_A
```

- Other data mixes are in `BluTrain/sft/mixes/`. Pass a different file to `--mix`.
- For arm B (the role markers as single tokens), use these two commands instead:

```
python3 BluTrain/sft/hf_to_blutrain.py --hf-dir base_hf --out ckpt_B --new-tokens "<|system|>,<|user|>,<|assistant|>"
python3 BluTrain/sft/prepare_sft_data.py --mix BluTrain/sft/mixes/m1_smol.json \
    --tokenizer ckpt_B/tokenizer.json --template tulu-special-v1 --pack ffd --out data/m1_smol_B
```

## 5. Run the training

```
cd ~/sft_work/BluTrain
CUDA_VISIBLE_DEVICES=0 \
BLU_N_LAYERS=37 BLU_ROPE_THETA=50000 BLU_D_MODEL=512 BLU_Q_HEADS=4 BLU_KV_HEADS=2 BLU_FFN_HIDDEN=1408 BLU_VOCAB=49280 BLU_EMBED_SCALE=1 \
BLU_SFT_DATA=$HOME/sft_work/data/m1_smol_A BLU_INIT_FROM_RAW=$HOME/sft_work/ckpt_A \
BLU_SFT_LR=1e-4 BLU_SFT_MUON_LR=1e-4 BLU_SFT_ROWS_PER_STEP=26 BLU_SFT_B=13 BLU_SFT_VAL_FREQ=200 \
BLU_CKPT_DIR=$HOME/sft_work/checkpoints_run1 BLU_EXPORT=$HOME/sft_work/run1_final.safetensors \
make run-train-mpi MODE=production_zero TRAIN_SRC=train/production_zero/bluscript_sft.cpp NP=1 RUN_NAME=run1
```

The command, line by line:

1. `CUDA_VISIBLE_DEVICES=0` picks the GPU.
2. The second line is the 135M model's shape. Never change it.
3. The third line gives the data folder and the converted base model from step 4.
4. The fourth line holds the training settings (table below).
5. The fifth line says where the checkpoints and the final model go.
6. The `make` line compiles our SFT trainer (`bluscript_sft.cpp`) and runs it with mpirun on `NP` GPUs. Everything else comes from the default config `train/production_zero/bluscript_zero_ckpt_config.yaml`; any setting you write before `make` wins over the config.

The run starts in a tmux session, and make shows its log. Ctrl-C only stops showing the log; the run keeps going. The log and the metrics are in `BluTrain/training_logs/run1_mpirun_<n>/` (`console.log` and `metrics.csv`).

| Setting | Meaning |
|---|---|
| `BLU_SFT_DATA` | the data folder from step 4 (full path) |
| `BLU_INIT_FROM_RAW` | the converted base model from step 4: `ckpt_A`, or `ckpt_B` for arm B (full path) |
| `BLU_SFT_LR`, `BLU_SFT_MUON_LR` | learning rate; set both to the same value |
| `BLU_SFT_ROWS_PER_STEP` | rows per training step (26 rows = about 128 chats for m1) |
| `BLU_SFT_B` | rows per GPU at once; at most 16, and it must divide the rows per step |
| `BLU_SFT_VAL_FREQ` | measure the val loss every N steps (the default of 10 is too often) |
| `BLU_SFT_EPOCHS` | passes over the data, default 2 |
| `BLU_SFT_WARMUP` | warm-up fraction, default 0.03 |
| `BLU_SFT_SCHEDULE` | `cosine` (default), `linear`, `constant` or `wsd` |
| `BLU_OPTIMIZER` | `muon` (default) or `adamw` |
| `BLU_CKPT_DIR` | the checkpoint folder of this run |
| `BLU_EXPORT` | the file the final model is saved to |
| `BLU_CKPT_LOAD=1` | continue a stopped run from its newest checkpoint in `BLU_CKPT_DIR` |
| `NP` | number of GPUs; for 2, use `CUDA_VISIBLE_DEVICES=0,1 ... NP=2` (rows per step must then be a multiple of 2 x `BLU_SFT_B`) |
| `RUN_NAME` | the run's name (required) |

**Watch a run:** make prints the tmux session name and the log file. Later, use `tail -f <that log file>` or `tmux attach -t <session name>`. Leave tmux with Ctrl-b, then d.

**Stop a run:** `tmux attach -t <session name>`, then Ctrl-C. To continue it later, run the same command with `BLU_CKPT_LOAD=1` added.

**After training:** turn the result into a Hugging Face model folder, for evaluation:

```
cd ~/sft_work
python3 BluTrain/sft/blutrain_to_hf.py --export run1_final.safetensors --base-hf base_hf \
    --tokenizer base_hf/tokenizer.json --out hf/run1
```

For arm B, use `--tokenizer ckpt_B/tokenizer.json`.

## 6. Notes

- Always keep the model line as it is. A wrong `BLU_ROPE_THETA` gives no error, only a worse model.
- Use a new `RUN_NAME`, `BLU_CKPT_DIR` and `BLU_EXPORT` for every run.
- Write the settings before `make` on the same command, as shown; do not `export` them. Then they apply to that run only, and a leftover setting cannot change your next run.
- Run one training per GPU, and never run anything else on that GPU at the same time. Check with `nvidia-smi` first.
- Give full paths for `BLU_SFT_DATA`, `BLU_INIT_FROM_RAW`, `BLU_CKPT_DIR` and `BLU_EXPORT`.
- More detail on how the SFT code works: `BluTrain/sft/README.md`.
- 2026-11-18: the attention backward on H100 / H200 changed (faster; the same correctness rules). Nothing changes in how you
  build or run: pull the branch again and rebuild. Its switches are described in `BluTrain/sft/README.md` section 6.3.
