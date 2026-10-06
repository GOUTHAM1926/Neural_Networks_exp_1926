# SFT report edits, 2026-10-05

Page numbers are the PDF pages of these arXiv versions:
- InstructGPT 2203.02155 v1
- Zephyr 2310.16944 v1
- Chu et al. 2501.17161 v2
- MT-Bench 2306.05685 v4
- DeepSeek-R1 2501.12948 v1

## EDIT 1 — Summary, "Why it's needed": the first 3 bullets

FIND:
In the InstructGPT paper, they preferred answers from a 1.3B model trained with SFT and RLHF over the 175B base GPT-3, "despite having 100x fewer parameters" ,,, ,

REPLACE WITH:
In the InstructGPT paper, people preferred the answers of a 1.3B GPT-3 trained with SFT and RLHF over the answers of the 175B GPT-3 that was only pretrained , "despite having 100x fewer parameters" ,,, ,
    (sub-bullet) The paper's goal : make GPT-3 follow instructions , first with SFT on answers written by people , then with RLHF (Section 1.8.1) ,,, ,

FIND:
In the Zephyr paper, skipping SFT and going straight to preference tuning dropped MT-Bench from 7.34 to 4.76, and the model "struggles to learn the chat template" ,,, ,

REPLACE WITH:
In the Zephyr paper, skipping SFT and going straight to preference tuning (DPO) dropped MT-Bench from 7.00 to 4.76 (the same training , only the SFT step removed) , and the model "struggles to learn the chat template" ,,, ,
    (sub-bullet) The paper's goal : make a small 7B chat model cheaply , with SFT on answers written by bigger models , then DPO on GPT-4's ratings (Section 1.8.2) ,,, ,

FIND:
Chu et al.: when the base model does not follow instructions, "without SFT, all end-to-end RL runs fail to improve" ,,, ,

REPLACE WITH:
Chu et al.: when the base model does not follow instructions, "without SFT, all end-to-end RL runs fail to improve" ,,, ,
    (sub-bullet) The paper's goal : compare what SFT and RL each teach a model , their main finding is that SFT memorizes the training data and RL learns rules that also work on new variants , and this line is one of their side checks (Section 1.8.3) ,,, ,

## EDIT 2 — Section 1.3, the Zephyr bullet

FIND:
MT-Bench was 4.76 instead of 7.34, and AlpacaEval 30.76% instead of 90.60%.

REPLACE WITH:
MT-Bench was 4.76 instead of 7.00 , and AlpacaEval 30.76% instead of 86.07% (Table 3 , page 8 : the same SFT + DPO training with and without the SFT step , the final Zephyr-7B scores 7.34 and 90.60% in Table 1 , page 6 , but it was trained longer , so it's not the fair comparison).

## EDIT 3 — Section 1.7 table, the "go straight to DPO" row, source column

FIND:
Zephyr : MT-Bench 7.34 → 4.76, AlpacaEval 90.60% → 30.76%

REPLACE WITH:
Zephyr : MT-Bench 7.00 → 4.76, AlpacaEval 86.07% → 30.76%

## EDIT 4 — Section 2.6 table, the "DPO only" row

FIND:
MT-Bench 7.34 → 4.76

REPLACE WITH:
MT-Bench 7.00 → 4.76

## EDIT 5 — NEW Section 1.8, after Section 1.7 (after the paragraph that starts "For Navin bro's model") and before "2. Where SFT sits in the full pipeline"

### 1.8 The three papers in detail : what they did , what they compared , and the exact lines :

These are the exact lines from the 3 papers in the summary , with page numbers , so anyone can open the paper and check , and the page numbers are the PDF pages of the arXiv version I give with each link ,,, ,

#### 1.8.1 InstructGPT (Ouyang et al. , OpenAI , 2022 , https://arxiv.org/abs/2203.02155v1)

What they trained : 3 sizes , 1.3B , 6B and 175B , and "all of our models use the GPT-3 architecture" (pages 2-3) , and they start from GPT-3 that was only pretrained : "We start with the GPT-3 pretrained language models from Brown et al. (2020)" (page 8 , Section 3.5) ,,, ,

Then 3 steps on top of it (page 7) :
1. SFT : people wrote good answers to about 13k prompts , and GPT-3 was trained on them ,,, ,
2. Reward model : people ranked several answers to the same prompt (33k prompts) , and a separate model learned to predict which answer people like ,,, ,
3. RLHF (PPO) : the SFT model was trained further on 31k prompts to get a high score from the reward model , and the result is called InstructGPT ,,, ,

Who was compared with whom :

| model | size | training |
|---|---|---|
| GPT-3 | 175B | pretrained only , no SFT , no RLHF |
| InstructGPT | 1.3B | the 1.3B GPT-3 , plus SFT and RLHF |

and people preferred the 1.3B InstructGPT's answers over the 175B GPT-3's answers ,,, ,

How "better" was measured : by people , not by benchmark scores , labelers saw the answers of each model to prompts that real customers sent to the OpenAI API (from customers whose prompts were not used in training) and chose the better one : "Our main metric is human preference ratings on a held out set of prompts from the same source as our training distribution" (page 10) ,,, ,

The exact lines :
- page 1 , abstract : "In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." ,,, ,
- page 3 : "These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions. Outputs from our 175B InstructGPT are preferred to 175B GPT-3 outputs 85 ± 3% of the time, and preferred 71 ± 4% of the time to few-shot 175B GPT-3." (a few-shot prompt means a few example questions and answers put before the real question , to push the base model into answering) ,,, ,
- page 2 , Figure 1 : every model is compared with the same reference , the 175B SFT model , and my reading of the chart (approximate) is that the 175B GPT-3 wins about 0.24 of the time , the 1.3B SFT-only model about 0.35 , and the 1.3B SFT + RLHF model about 0.52 , so even SFT alone at 1.3B wins more often against that reference than the 175B base ,,, ,

What it means for me : size alone doesn't make a model useful , a small model trained on good answers beats a 100x bigger model that was only pretrained ,,, ,

#### 1.8.2 Zephyr (Tunstall et al. , Hugging Face , 2023 , https://arxiv.org/abs/2310.16944v1)

The words first :
- dSFT : SFT on answers written by a bigger model ("distilled") , here UltraChat , 1.47M dialogues generated by GPT-3.5-turbo (page 5) ,,, ,
- preference tuning : teaching the model which of 2 answers is better , and DPO is one way to do it , not the only one , RLHF with PPO (InstructGPT's way) is another , and DPO learns directly from (better , worse) answer pairs , without a reward model ,,, ,
- dDPO : DPO where the better / worse labels came from GPT-4 , not from people , UltraFeedback has 64k prompts with 4 answers each "rated by GPT-4" , and the best one is "chosen" and one of the other 3 , at random , is "rejected" (page 5) ,,, ,

The test (Table 3 , page 8) : Mistral-7B trained 4 ways :

| way | MT-Bench | AlpacaEval |
|---|---|---|
| DPO only , no SFT | 4.76 | 30.76% |
| SFT , 1 epoch | 6.64 | 85.65% |
| SFT , 2 epochs | 6.19 | 78.54% |
| SFT , then DPO | 7.00 | 86.07% |

The exact lines :
- page 7 : "without an initial SFT step (dSFT), models are not able to learn at all from feedback and perform terribly." ,,, ,
- page 13 , Appendix A.2 "SFT is a required step before DPO" : "We observed a significant reduction in performance in both the MT-Bench and AlpacaEval scores when the SFT step is skipped. After a qualitative evaluation of the MT-Bench generations, we observe that the pure DPO model struggles to learn the chat template" ,,, ,
- page 14 , Figure 7 , the example : the DPO-only model gives a correct answer and then doesn't stop , it keeps printing <|system|> <|user|> <|assistant|> again and again , and the SFT + DPO model answers and stops ,,, ,

About 7.34 : the final Zephyr-7B scores 7.34 on MT-Bench and 90.60% on AlpacaEval (Table 1 , page 6) , and its strongest model "was obtained with one epoch of SFT followed by three epochs of DPO" (page 7) , so 7.34 isn't the same training as the 4.76 run , and the fair comparison is 7.00 ,,, ,

The 2 benchmarks :
- MT-Bench (Zheng et al. , 2023 , https://arxiv.org/abs/2306.05685v4) : 80 questions with 2 turns each , in 8 categories , writing , roleplay , extraction , reasoning , math , coding , STEM and humanities (page 3 , Section 2.2 , and page 4) , the model answers a question , then a follow-up question , GPT-4 reads each answer and gives it 1 to 10 (the judge prompt is on page 14 , Figure 6) , and the score is the average , and Zephyr calls it "160 questions" because that's 80 questions x 2 turns (Zephyr page 5 , Section 4.2) ,,, ,
- AlpacaEval : 805 single questions , "mostly focused on helpfulness" , and GPT-4 compares each answer with the answer of a baseline model (text-davinci-003) , and the score is the % of wins (Zephyr page 5 , Section 4.2) ,,, ,

#### 1.8.3 Chu et al. ("SFT Memorizes, RL Generalizes" , ICML 2025 , https://arxiv.org/abs/2501.17161v2)

The paper's main question (page 1 , abstract) : does SFT or RL "memorize the training data" or learn rules that work on new variants , and their answer : RL "generalizes" , SFT "tends to memorize the training data" , and "SFT is still helpful for effective RL training: SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains." ,,, ,

The task (page 3) : "GeneralPoints" , the model gets 4 cards and must write an equation that equals 24 , using each card once , and a program checks the answer , so RL can get a reward without people ,,, ,

The test (page 7 , Section 5.4 "The Role of SFT for RL Training") : RL directly on the base Llama-3.2 , no SFT first ,,, ,

The exact lines :
- page 7 : "SFT is necessary for RL training when the backbone model does not follow instructions. Figure 9 shows that without SFT, all end-to-end RL runs fail to improve. More specifically, we observe that without SFT, the base model suffers from poor instruction following capability. A detailed failure case is provided in Figure 20 (in Appendix D.3), revealing that the base Llama-3.2-Vision-11B model tends to generate long, tangential, and unstructured responses. This issue makes it impossible to retrieve task-related information and rewards for RL training." ,,, ,
- page 8 , Figure 9 : "All trials fail due to poor instruction following capability of the base model." ,,, ,

Why it fails : RL learns only from the reward , the base model rambles and never writes the answer in the asked format , so the checker never finds an answer to score , the reward stays 0 and there's nothing to learn from , and SFT first teaches the format ,,, ,

The condition , same page 7 : "Note that due to the difference in backbone model, our results do not contradict with DeepSeek-AI et al. (2025), which suggests that SFT is unnecessary for downstream RL training." ,,, ,
- "downstream RL" means the RL step that comes later in the pipeline , and DeepSeek showed that RL can run straight on a base model with no SFT before it : DeepSeek-R1-Zero is "a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step" (DeepSeek-R1 , https://arxiv.org/abs/2501.12948v1 , page 1 , abstract) , so "SFT is unnecessary for downstream RL" means "you don't always need SFT before RL" ,,, ,
- why both can be true : it depends on the base model , DeepSeek's base (DeepSeek-V3-Base , 671B parameters in total) could already follow the format written in its prompt , "<think> reasoning process here </think> <answer> answer here </answer>" (DeepSeek-R1 v1 , page 6 , Table 1) , so RL got rewards from the start , and Chu's base (Llama-3.2-Vision-11B) couldn't , so RL got nothing ,,, ,
- even DeepSeek added SFT in the end : R1-Zero "encounters challenges such as poor readability, and language mixing" (page 1) , so the released DeepSeek-R1 starts by "collecting thousands of cold-start data to fine-tune the DeepSeek-V3-Base model" (SFT) and then runs RL (page 3) ,,, ,
- for my 1.5B base : I haven't tested yet whether it follows a format from the prompt alone , because BluTrain can't generate text yet (Section 6.3) , so SFT first is the safe choice (Section 1.7 , Section 2.6) ,,, ,

#### 1.8.4 What the three show together

| paper | what it shows |
|---|---|
| InstructGPT | without post-training , even a 100x bigger model loses to a small one fine-tuned on good answers |
| Zephyr | skipping SFT and going straight to DPO fails (MT-Bench 7.00 → 4.76) , the model doesn't learn the chat format |
| Chu et al. | skipping SFT and going straight to RL fails when the base can't follow instructions , there's no reward to learn from |

So SFT comes first , and every later step (DPO , RL) builds on it ,,, ,

## EDIT 6 — Section 3.2, add after the paragraph that ends "matters more than which one ,,, ,"

What goes wrong without it : in Zephyr , the model trained with DPO but without SFT gave a correct answer and then didn't stop , it kept printing <|system|> <|user|> <|assistant|> again and again (Zephyr page 14 , Figure 7 , Section 1.8.2) , because SFT is what teaches the model that after <|assistant|> comes an answer , then the end token , and then it stops ,,, ,

Where the template is used , in 2 places , and it has to be the same in both : in the preprocessing script , every SFT sample is wrapped in it before tokenizing (Section 3.10) , and when the model is used (chat , evals) , the question is wrapped the same way and ends with "<|assistant|>\n" , and the model writes the answer and the end token , and GPT-2's tokenizer has no chat template , so I have to define one (Section 3.4) ,,, ,

## EDIT 7 — Section 3, right under the heading "3. How SFT runs in BluTrain :", before "3.1 The whole flow :"

BluTrain is our own C++/CUDA trainer on our own tensor library , and the SFT runs there , not in a Python package , and the Python libraries are only references , I read their code to copy the right ideas ,,, ,

PyTorch's own post-training libraries are stopped : torchtune says it "is no longer actively maintained: torchtune development wound down in 2025" , and torchforge says "Development in Forge has paused. LLM training at PyTorch is being consolidated in torchtitan" (their GitHub README notices , checked in Sept 2026) ,,, ,

What I take from each :

| idea | from |
|---|---|
| count the loss only on answer tokens , ignore the prompt and padding | TRL , torchtitan , Megatron-LM , and PyTorch's ignore_index |
| divide the loss by the answer tokens of the whole step , not per micro-batch | TRL (num_items_in_batch) , torchtitan , Megatron-LM (--calculate-per-token-loss) |
| the tokenized prompt must be an exact prefix of the tokenized full sample | torchtitan |
| exactly one end token per sample , and drop samples that are too long instead of cutting them | torchtitan |
| don't compute the lm_head on prompt rows (later) | TRL (chunked_nll) |

## EDIT 8 — Sources, add one line at the end of the list

Page numbers in Section 1.8 are the PDF pages of : InstructGPT v1 , Zephyr v1 , Chu et al. v2 , MT-Bench v4 , DeepSeek-R1 v1 (https://arxiv.org/abs/2501.12948v1) ,,, ,
