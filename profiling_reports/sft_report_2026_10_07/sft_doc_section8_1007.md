# 8. First SFT runs in BluTrain : what I built , the tests , the results , Muon vs AdamW , and the plan (6 and 7 Oct)

## 8.1 What I built

- **Pre-processing script (sft/prepare_sft_data.py)** :
  - loads UltraChat-200k (English only , MIT licence , the set Zephyr used) ,,, ,
  - applies the template tulu-plain-v1 , which is option A of Section 3.4 : the markers are plain text (`<|user|>\n` is 6 GPT-2 tokens and `<|assistant|>\n` is 7) , so no new special tokens , and the end token 50256 after every answer is a target ,,, ,
  - puts 65535 on the prompt and the padding labels , pads the inputs with 50256 up to T = 4096 , and drops chats longer than 4096 instead of cutting them ,,, ,
  - writes inputs.bin , labels.bin , the trained-token count of every row (valid_counts.bin) and meta.json ,,, ,
- **Stage-0 data** : 5,000 random training chats and 500 validation chats , with 4.77M trained tokens , which is 23.3% of all positions , and the average chat is 1,238 tokens , so about 70% of every 4096-token row is padding ,,, ,
- **SFT dataloader (sft/SftDataLoader.h)** :
  - the target comes from labels.bin instead of the input shifted by one ,,, ,
  - each GPU gets different rows ,,, ,
  - it gives the trainer the trained-token count of the next micro-batches before the forward ,,, ,
  - it checks every id when it opens the files ,,, ,
  - my loader test passed 5/5 ,,, ,
- **Weights-only load (BLU_INIT_FROM)** : it loads only Navin anna's step-8000 weights , with a new optimizer and a new LR schedule , and 262/262 weights match the checkpoint ,,, ,
- **SFT mode in the trainer (BLU_SFT_DATA)** :
  - the loss is divided once by the trained tokens of the whole step (all micro-batches and all GPUs) , counted before the forward and added up across GPUs (Section 3.7) ,,, ,
  - the backward starts from W (the number of GPUs) instead of 1/accumulation steps , becoz the gradients are averaged across GPUs afterwards , so nothing is divided twice ,,, ,
  - the logged train loss is the plain sum , and validation runs over the whole val set every 10 steps ,,, ,
  - settings : 64 chats per step , 2 epochs , LR 2e-5 → 2e-6 , 3% warmup , then cosine ,,, ,
  - without BLU_SFT_DATA , pretraining runs exactly as before ,,, ,

## 8.2 Tests and results (H200)

All runs start from step 8000 , with 64 chats per step , 2 epochs = 156 steps ,,, ,

| run | what it checks | result |
| :---- | :---- | :---- |
| sample run , 1 GPU , 20 steps (6 Oct) | does SFT run at all | val loss 1.446 → 1.332 , and the trained-token counts match the files exactly (60,263 at step 0 , 441,382 for val) |
| full run , 1 GPU , AdamW (7 Oct) | the full 2 epochs and the checkpoint saves | val loss 1.4464 → 1.2839 , still falling at the end , no NaN , 5.3 s per step , checkpoints saved at steps 78 and 156 |
| resume from the step-78 checkpoint | a restart continues exactly | step 78 exactly the same as without the restart , final val 1.283921 vs 1.283946 |
| pretraining check | my change did not touch pretraining | step 0 bit-for-bit the same as before (train 2.327407 , val 2.453832) |
| 2 GPUs | adding up the count across GPUs | step 0 the same as 1 GPU , including the grad norm 0.7442 , and 2x faster |
| 4 GPUs | the same with 4 GPUs | step 0 the same , and 4x faster (1.35 s per step) |
| Muon , 3 learning rates , 4 GPUs | Muon in SFT | see 8.3 |

- **The v1 tests from 6.5** :
  - test 3 (the count) , test 4 (the first loss is 1.446 , not 10.83) and test 7 pass ; for test 7 , 1 , 2 and 4 GPUs with 8 , 4 and 2 accumulation steps give the same step-0 loss and grad norm ,,, ,
  - test 2 is covered by the 136-case kernel test ,,, ,
  - tests 1 , 5 and 6 are still to do , and test 8 needs generation ,,, ,

## 8.3 Muon vs AdamW : my learning-rate sweep (7 Oct)

- The base was pretrained with Muon , and my first Muon run at the AdamW LR (2e-5) learned slower than AdamW , so I ran a sweep ,,, ,
- **Setup** :
  - 4 GPUs , the same data , 2 epochs , and the same 480 validation chats ,,, ,
  - in the Muon runs , the AdamW part (embeddings and norms) used the same LR as Muon , like Kathir's script ,,, ,
  - each run's LR decays to one tenth of its peak ,,, ,
- All runs start at val 1.4432 ,,, ,

| optimizer , LR | val at step 10 | val at step 80 (end of epoch 1) | best val (step) | final val (step 155) | train loss , epoch 1 / epoch 2 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| AdamW 2e-5 | 1.3497 | 1.2860 | 1.2809 (155) | 1.2809 | 1.316 / 1.239 |
| Muon 2e-5 | 1.4364 | 1.3330 | 1.3162 (155) | 1.3162 | 1.384 / 1.295 |
| Muon 5e-5 | 1.4121 | 1.2920 | 1.2843 (155) | 1.2843 | 1.340 / 1.208 |
| Muon 1.2e-4 | 1.3584 | **1.2747** | **1.2747 (80)** | 1.2820 | 1.314 / 1.084 |

## 8.4 Muon vs AdamW : what the papers say

I checked each paper's full text , not summaries ,,, ,

| paper | pretraining optimizer | SFT optimizer |
| :---- | :---- | :---- |
| Moonlight (arXiv 2502.16982) | Muon | It tested all 4 pairs (Table 6). Muon pretraining + Muon SFT was best on all 4 benchmarks : MMLU 55.7 , HumanEval 57.3 , MBPP 55.6 , GSM8K 68.0 , against AdamW + AdamW 52.0 , 53.1 , 55.2 , 64.6. Switching optimizer between the stages gave mixed results. On Qwen2.5-7B , which was pretrained with AdamW , Muon SFT ≈ AdamW SFT (Table 7) |
| Kimi K2 (arXiv 2507.20534) | MuonClip | Muon : "We employ the Muon optimizer in our post-training and recommend its use for fine-tuning with K2" (Section 3.1) , and Muon for RL too |
| GLM-4.5 (arXiv 2508.06471) | Muon | not stated |
| Qwen2 , Qwen2.5 , Qwen3 | not stated | not stated |
| Llama 2 , Llama 3 , DeepSeek-V3 , OLMo 2 | AdamW | not named in their SFT sections |
| LIMA , Zephyr , InstructGPT | (fine-tuned existing models) | AdamW , AdamW , Adam |
| Tülu 3 | (fine-tuned Llama 3.1) | not named |

- Two pretraining-only studies :
  - Wen et al. (arXiv 2509.02046) : against a well-tuned AdamW , Muon's speed-up shrinks as models grow , 1.4x at 0.1B and 1.1x at 1.2B ,,, ,
  - Essential AI (arXiv 2505.02222) : for the same compute , Muon reaches the same loss in less time than AdamW , and it stays data-efficient at large batch sizes ,,, ,
- One correction to the optimizer row of my settings table : Llama 2's SFT section does not name its optimizer , AdamW is named only for its pretraining and RLHF ,,, ,

## 8.5 What I learned

- **Muon needs a higher LR than AdamW for SFT** : at the same 2e-5 it learns clearly slower (1.3162 vs 1.2809) , at 5e-5 it about matches AdamW (1.2843) , and at 1.2e-4 it gives the lowest val of all runs (1.2747) ,,, ,
- **A high LR overfits a small set** : at 1.2e-4 the best val comes at the end of epoch 1 , and in epoch 2 the train loss falls to 1.08 while val goes back up to 1.287 , so with only 5,000 chats , 2 epochs at 1.2e-4 is too much ,,, ,
- **The step-80 number is not a finished 1-epoch run** : at step 80 the LR was still 6.6e-5 , so a real 1-epoch run at 1.2e-4 still has to be tested ,,, ,
- **Val loss alone can't pick the optimizer** : the best runs differ by only about 0.5% , these are one run each on a small set , and val loss doesn't show whether the answers are good or stop at the end token , so the final pick needs evals ,,, ,
- **"Everyone uses AdamW for SFT" is not true** : most papers don't name their SFT optimizer , and the only paper that tested it (Moonlight) found it best to keep the pretraining optimizer , which for us is Muon ,,, ,
- **The global denominator works** : 1 , 2 and 4 GPUs give the same step-0 loss and grad norm to the last digit , which only happens if both the count and the backward scale are right ,,, ,
- **The epoch grouping depends on the GPU count** :
  - on 1 GPU , an epoch is 625 micro-batches , which is not a whole number of 8-micro-batch steps , so epoch 2 starts one micro-batch into step 78 ,,, ,
  - on 2 and 4 GPUs , epoch 2 starts exactly at step 78 ,,, ,
  - so after step 77 , the runs differ a little by design , and I can fix it by making every epoch a whole number of steps ,,, ,
- **The val set also depends on the GPU count** : 496 chats on 1 or 2 GPUs and 480 on 4 GPUs (every GPU needs whole micro-batches) , so I only compare val numbers between runs with the same GPU count ,,, ,
- **Padding is the biggest waste** : about 70% of every row is padding , and about 77% of the logits are computed for positions that are ignored ,,, ,
- **On the box** :
  - mpirun passes the whole shell environment to the run , so an old BLU_CKPT=0 from the morning switched off checkpoints in my 6 Oct run ,,, ,
  - my run scripts now clear every BLU_ variable before each run ,,, ,

## 8.6 Kathir's PyTorch SFT script vs mine

| | mine (BluTrain , C++) | Kathir's (PyTorch) |
| :---- | :---- | :---- |
| same | base step 8000 , loss only on answers , summed loss ÷ the trained tokens of the whole step , Muon for matrices + AdamW for embeddings and norms , 3% warmup , 2 epochs | same |
| data | UltraChat , 5,000 chats | OLMo 2 SFT mix , 839,168 chats |
| batch | 64 chats per step , fixed 4096-token rows (about 70% padding) | 512 chats per step , sorted by length and cut into micro-batches of at most 16,384 tokens (little padding) |
| LR | separate LRs , cosine to one tenth | one LR (5e-5) , linear to 0 |
| logits | all positions | only the trained positions |
| shuffle | the same order every epoch | a new shuffle every epoch |
| output | BluTrain checkpoints | Hugging Face checkpoints , ready for inference |
| eval | val loss | val loss + answers to 2 prompts (base vs epoch 1 vs epoch 2) |

- What I'll take from his script : logits only at the trained positions , length-sorted micro-batches , a new shuffle every epoch , and a Hugging Face export for evals ,,, ,

## 8.7 The plan from here

1. **Evals first** :
   - get Kathir's checkpoint → Hugging Face converter ,,, ,
   - run his prompts on the base model vs my SFT checkpoint (step 156) , and check that answers stop at the end token (test 8) ,,, ,
   - then run IFEval , GSM8K , HellaSwag and MMLU on base vs SFT (Section 6.4) ,,, ,
2. **Pick the optimizer and LR by evals** :
   - a 1-epoch Muon run at 1.2e-4 , against Muon 5e-5 for 2 epochs and AdamW 2e-5 ,,, ,
   - an AdamW sweep (5e-6 , 1e-5 , 2e-5) ,,, ,
3. **Speed** : logits only at the trained positions , length-sorted micro-batches , a new shuffle every epoch , and epochs that are a whole number of steps ,,, ,
4. **More data** : about 100k chats , then the data-mix study (Section 5.5) ,,, ,
5. **Remaining v1 tests and fixes** : tests 1 , 5 and 6 (Section 6.5) , a bounds check in the loss kernel , and 65535 support in the B300 loss kernel ,,, ,
6. **Later** : packing with variable-length attention (Section 3.8) , and think-model SFT once long context is in (Section 4.3) ,,, ,

## Add to Sources

- Moonlight: Liu et al., Muon is Scalable for LLM Training, 2025. https://arxiv.org/abs/2502.16982
- Kimi K2: Kimi Team, Kimi K2: Open Agentic Intelligence, 2025. https://arxiv.org/abs/2507.20534
- GLM-4.5: GLM-4.5 Team, GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models, 2025. https://arxiv.org/abs/2508.06471
- Qwen2.5: Qwen Team, Qwen2.5 Technical Report, 2024. https://arxiv.org/abs/2412.15115
- Qwen2: Qwen Team, Qwen2 Technical Report, 2024. https://arxiv.org/abs/2407.10671
- Wen et al., Fantastic Pretraining Optimizers and Where to Find Them, 2025. https://arxiv.org/abs/2509.02046
- Essential AI, Practical Efficiency of Muon for Pretraining, 2025. https://arxiv.org/abs/2505.02222
