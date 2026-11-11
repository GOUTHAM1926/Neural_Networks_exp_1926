# Muon or AdamW for SFT: my test, and the Moonlight reference

## 1. The answer in short

- **I use Muon for SFT:** Muon for the 2-D weight matrices plus AdamW for the embedding and the norms (the usual Muon
  split), both at the same peak learning rate.
- **Why:**
  1. Both bases I fine-tune were **pretrained with Muon**: Navin's 1.5B step-8000 model, and the 135M 900B base (checked from
     its optimizer shard size: 1,176,170,496 bytes = Muon + AdamW state; AdamW alone would be about 1.61 GB).
  2. The only paper I found that tested **all four pairs of pretraining and SFT optimizer** (Moonlight, by Moonshot AI) found
     **Muon pretraining + Muon SFT best on all four benchmarks**. For a Muon-pretrained model, switching to AdamW for SFT was
     worse on all four.
  3. In my own test, Muon at a higher learning rate matched or slightly beat AdamW on validation loss.
- **What I tested myself and what I did not:**
  - I tested the **SFT optimizer** (AdamW vs Muon at three learning rates) on a **Muon-pretrained** base.
  - I did **not** test the four pretraining x SFT pairs. That needs an AdamW-pretrained twin of the same base, which I do
    not have. The evidence for the four pairs is Moonlight's (section 3 below).

## 2. My test (7 October 2026)

### Setup

| | |
|---|---|
| base | Navin's step-8000 model (`blumodel_step_8000`, about 1.5B parameters, 26 layers, 8.4B tokens), pretrained with **Muon** on 4 GPUs |
| loading | weights only (262 / 262 weights checked against the checkpoint), new optimizer state, new learning-rate schedule |
| data | UltraChat-200k (English): 5,000 training chats; validation on the same 480 chats in every run |
| template | tulu-plain-v1 (plain-text role markers, the end token after every answer), loss on the answers only |
| rows | 4,096 tokens, one chat per row with padding (about 70% padding), 64 chats per step |
| length | 2 epochs = 156 steps |
| schedule | 3% warm-up, then cosine down to one tenth of the peak learning rate |
| Muon runs | the AdamW part (embedding and norms) used the same learning rate as the Muon part |
| hardware | every run on 4 H200 GPUs, all bf16 (`BLU_ALL_BF16=1`) |

### Results (validation loss; every run starts at 1.4432)

| optimizer, learning rate | val at step 10 | val at step 80 (end of epoch 1) | best val (step) | final val (step 155) | train loss, epoch 1 / epoch 2 |
|---|---|---|---|---|---|
| AdamW 2e-5 | 1.3497 | 1.2860 | 1.2809 (155) | 1.2809 | 1.316 / 1.239 |
| Muon 2e-5 | 1.4364 | 1.3330 | 1.3162 (155) | 1.3162 | 1.384 / 1.295 |
| Muon 5e-5 | 1.4121 | 1.2920 | 1.2843 (155) | 1.2843 | 1.340 / 1.208 |
| Muon 1.2e-4 | 1.3584 | **1.2747** | **1.2747 (80)** | 1.2820 | 1.314 / 1.084 |

### What it shows

- **Muon needs a higher learning rate than AdamW for SFT.** At the same 2e-5, Muon learned clearly slower than AdamW (final
  1.3162 vs 1.2809). At 5e-5 it about matched AdamW (1.2843 vs 1.2809). At 1.2e-4 it reached the lowest validation loss of
  all runs (1.2747 at the end of epoch 1). Muon needed about 2.5-6 times AdamW's learning rate. A likely reason, not proven:
  our Muon scales each update to the size of a settled AdamW update (Moonshot's rule, section 5), while a fresh AdamW takes
  bigger first steps.
- **A high learning rate overfits a small set.** At 1.2e-4 the best validation loss came at the end of epoch 1. In epoch 2 the
  training loss fell to 1.084 while validation rose back to 1.2820.
- All runs were stable: no NaN, and no rejected Newton-Schulz updates in Muon.

### Limits of this test

- One run each, 5,000 chats, validation loss only, no benchmarks. The best runs differ by only about 0.5%.
- AdamW was run at one learning rate (2e-5) only; Muon at three.
- The step-80 value of the 1.2e-4 run is not a finished 1-epoch run (the learning rate was still 6.6e-5 at step 80).

## 3. The reference: Moonlight (Moonshot AI, arXiv 2502.16982, section 3.5)

### Their setup

- Model: Moonlight@1.2T, the 1.2T-token checkpoint of Moonlight (a mixture-of-experts model, 2.24B activated and 15.29B total
  parameters, embeddings excluded), and its twin pretrained the same way with AdamW (Moonlight-A@1.2T).
- SFT: the Tülu 3 SFT mixture (`tulu-3-sft-mixture`), 4k sequence length, 2 epochs, learning rate decaying linearly from
  5e-5 to 0.

### Table 6: all four pairs of pretraining and SFT optimizer

| benchmark (metric, shots) | Muon pretrain + Muon SFT | AdamW pretrain + Muon SFT | Muon pretrain + AdamW SFT | AdamW pretrain + AdamW SFT |
|---|---|---|---|---|
| MMLU (EM, 0-shot CoT) | **55.7** | 55.3 | 50.2 | 52.0 |
| HumanEval (pass@1, 0-shot) | **57.3** | 53.7 | 52.4 | 53.1 |
| MBPP (pass@1, 0-shot) | **55.6** | 55.5 | 55.2 | 55.2 |
| GSM8K (EM, 5-shot) | **68.0** | 62.1 | 64.9 | 64.6 |

How to read it:
- **Muon pretraining + Muon SFT is the best on all four benchmarks.** The paper says: "a model that is both Muon-pretrained
  and Muon-finetuned outperforms others in the ablation studies."
- **For a Muon-pretrained model** (our case), switching to AdamW for SFT lost on all four: MMLU -5.5, HumanEval -4.9,
  MBPP -0.4, GSM8K -3.1 (column 3 vs column 1).
- **For an AdamW-pretrained model**, Muon SFT was mixed against AdamW SFT: MMLU +3.3, HumanEval +0.6, MBPP +0.3, GSM8K -2.5
  (column 2 vs column 4).

### Table 7: Muon SFT on a model pretrained with AdamW (Qwen2.5-7B)

Setup: the Qwen2.5-7B base, the Tülu 3 SFT mixture packed to 8k, learning rate cosine from 2e-5 to 2e-6.

| benchmark (metric, shots) | AdamW SFT | Muon SFT |
|---|---|---|
| MMLU (EM, 0-shot CoT) | 71.4 | 70.8 |
| HumanEval (pass@1, 0-shot) | 79.3 | 77.4 |
| MBPP (pass@1, 0-shot) | 71.9 | 71.6 |
| GSM8K (EM, 5-shot) | 89.8 | 85.8 |

- The paper calls this "on par". In the numbers, Muon SFT is a little lower on all four (0.3 to 4.0 points).
- Its conclusion: "it is more effective to apply Muon during the pretraining phase rather than during supervised
  fine-tuning."
- So Muon SFT gives no gain on a model pretrained with AdamW. The gain comes from keeping Muon when the model was pretrained
  with Muon.

### What it means for us

- Our bases are Muon-pretrained, so the comparison that applies is column 1 vs column 3 of Table 6: **Muon SFT**.
- Their SFT recipe (Tülu 3 mix, 2 epochs, peak learning rate 5e-5) is the recipe I used for the 1.7B Tülu 3 SFT run (section 5).
- If a future base is pretrained with AdamW, Table 7 says AdamW SFT is the safe choice for it.

## 4. What other papers say about the SFT optimizer (checked in the full texts)

| paper | pretraining optimizer | SFT optimizer, as written |
|---|---|---|
| Moonlight (arXiv 2502.16982) | Muon | tested all four pairs (section 3): Muon + Muon best |
| Kimi K2 (arXiv 2507.20534) | MuonClip | Muon: "We employ the Muon optimizer in our post-training and recommend its use for fine-tuning with K2" (section 3.1); Muon for RL too |
| GLM-4.5 (arXiv 2508.06471) | Muon (not for the embeddings, biases or RMSNorm) | not stated |
| Qwen2, Qwen2.5, Qwen3 | not stated | not stated |
| Llama 2, Llama 3, DeepSeek-V3, OLMo 2 | AdamW | not named in their SFT sections |
| LIMA, Zephyr, InstructGPT | (fine-tuned existing models) | AdamW, AdamW, Adam |
| Tülu 3 | (fine-tuned Llama 3.1) | not named |

- "Everyone uses AdamW for SFT" is not true: most papers do not name their SFT optimizer at all.
- Two pretraining-only studies, for context:
  - Wen et al. (arXiv 2509.02046): against a well-tuned AdamW, Muon's speed-up shrinks as models grow, 1.4x at 0.1B and
    1.1x at 1.2B.
  - Essential AI (arXiv 2505.02222): for the same compute, Muon reaches the same loss in less time than AdamW and stays
    data-efficient at large batch sizes.

## 5. How Muon is set up in the BluTrain SFT trainer

- `BLU_OPTIMIZER=muon` (the default of `sft/run_sft.sh`). The trainer prints:
  `MUON + AdamW (BLU_OPTIMIZER=muon) | ... momentum 0.950 wd 0.100 ns_steps 5 ... adjust_lr match_rms_adamw`.
- **Muon part, the 2-D weight matrices** (all of them, for example 222 matrices in the 135M model):
  - momentum 0.95 (Nesterov), decoupled weight decay 0.1;
  - 5 Newton-Schulz steps in bf16, after scaling by the fp32 Frobenius norm (eps 1e-8);
  - each update scaled to the size of an AdamW update, Moonshot's rule (0.2 x sqrt(max(rows, columns)));
  - one Newton-Schulz guard and the update ratio are decided on the GPU (2026-11-04).
- **AdamW part:** the embedding and the norms, betas (0.9, 0.95), eps 1e-8, weight decay 0.1; gradient clipping (1.0) covers
  this part only.
- **Learning rates used:**
  - 1.7B Tülu 3 SFT run (October 22): Muon and AdamW parts both at a peak of 5e-5 (the best finished 2-epoch Muon run of my
    test, and Moonlight's value), 3% warm-up, cosine to 5e-6, 2 epochs = 11,590 steps; validation loss 1.1757 -> 0.8501.
  - 135M SFT (October 28, stage 1, Muon, 128 chats per step, final validation loss): 1e-6 1.2039, 3e-6 1.1464, 1e-5 1.0804,
    3e-5 1.0252, 1e-4 0.9900, 3e-4 1.0023. **The lowest validation loss was not the best model.** At 1e-4, IFEval was 0.405
    (above SmolLM2-135M-Instruct's 0.366) but ARC-Easy fell to 0.492 against the base's 0.543 (forgetting); 1e-5 kept the
    knowledge (ARC-Easy 0.538) with IFEval 0.354.

## 6. Open points

- AdamW was tested at one learning rate only. A fair head-to-head needs an AdamW sweep too, then benchmarks (not only
  validation loss), on more data and more than one seed.
- The four-pair test can only be run with an AdamW-pretrained twin of the same base, which I do not have. Until then,
  Moonlight's Table 6 is the evidence.
- For any future base pretrained with AdamW, use AdamW for SFT (Moonlight Table 7).

## Sources

- Moonlight: "Muon is Scalable for LLM Training", Moonshot AI, arXiv 2502.16982, section 3.5, Tables 6 and 7:
  https://arxiv.org/abs/2502.16982
- Kimi K2, arXiv 2507.20534 (section 3.1): https://arxiv.org/abs/2507.20534
- GLM-4.5, arXiv 2508.06471: https://arxiv.org/abs/2508.06471
- Wen et al., arXiv 2509.02046: https://arxiv.org/abs/2509.02046
- Essential AI, arXiv 2505.02222: https://arxiv.org/abs/2505.02222
- My records: the SFT work documentation, sections 37.4-37.7 (the test, 7 October) and 45.105 (the 135M stage 1); my SFT
  research document, section 8.
