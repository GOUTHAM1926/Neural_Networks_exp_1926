# SFT report: edits for Gemini in Google Docs (2026-10-05, checked)

Every quote, number and page below was checked against the paper itself: 112 claims, then a second check of every correction.

## How to use this

- **Where each edit goes.** Google Docs has no line numbers. So every edit gives:
  - the page and line in the current PDF (exported 05-10-2026 11:02, 36 pages);
  - the section;
  - the exact text to find. Press Ctrl+F and paste the "Find" text to jump to the spot.
- **Order.** Do the edits from the bottom of the document to the top: H, G, F, E, D, then the Ctrl+H fixes A, B, C, C2. That way the page numbers stay right.
- **A, B, C, C2 are small fixes.** Do them with Google Docs' own Find and replace (Ctrl+H), not with Gemini. It's exact and can't change anything else.
- **D to H add new text.** The new text is inside grey code boxes, so you can copy it exactly. For each one, either:
  - **with Gemini:** paste the "Prompt for Gemini", then the code-box text; or
  - **yourself:** put the cursor at the spot, right-click, and choose "Paste from Markdown". If you don't see it, turn it on in Tools → Preferences → "Enable Markdown". Headings, lists and tables then come out formatted.
- **Headings.** After pasting, if a new heading looks different from its neighbours ("1.7 …" for "1.8", or the smaller style for "1.8.1"), select it and pick the same style as its neighbour.
- **Paste this rule into Gemini once, before the first edit:**

```text
I will give you edits for this document one at a time. For each edit, change only what I say, at the exact place I say. Do not rewrite, shorten, reorder, reformat or "improve" any other text in the document. Keep the document's style: the same fonts, sizes and heading styles; bullets stay bullets; keep every " ,,, ," at the end of a line exactly as I wrote it; keep the first person "I". Lines that start with "#" are headings, lines that start with "-" are bullets, and lines with "|" are table rows: make them a real Google Docs table with exactly those rows and columns, without the "|" signs.
```

---

## A. Section 1.3, the Zephyr bullet (page 3, lines 10-11) — Ctrl+H

Find:

```text
MT-Bench was 4.76 instead of 7.34, and AlpacaEval 30.76% instead of 90.60%.
```

Replace with:

```text
MT-Bench was 4.76 instead of 7.00 , and AlpacaEval 30.76% instead of 86.07% (Table 3 , page 8 : the same SFT + DPO training with and without the SFT step , the final Zephyr-7B scores 7.34 and 90.60% in Table 1 , page 6 , but it was trained longer , so it's not the fair comparison).
```

Click "Replace" (1 match).

## B. The tables in Sections 1.7 and 2.6 (page 6, line 8; page 11, line 48) — Ctrl+H

- Find: `7.34 → 4.76`
- Replace with: `7.00 → 4.76`
- Click "Replace all": 2 matches, the Zephyr row of the 1.7 table and the "DPO only" row of the 2.6 table.

## C. The 1.7 table, the same Zephyr cell (page 6, line 8) — Ctrl+H

- Find: `90.60% → 30.76%`
- Replace with: `86.07% → 30.76%`
- 1 match. The cell then reads: `Zephyr : MT-Bench 7.00 → 4.76, AlpacaEval 86.07% → 30.76%`

## C2. The 1.7 table, the "what happens" cell of the same row (page 6, lines 7-9) — Ctrl+H

The paper says the model "struggles to learn" the chat template, not that it doesn't learn it.

- Find: `doesn't learn the chat template`
- Replace with: `"struggles to learn the chat template"`
- 1 match. The cell then reads: `much worse chat, "struggles to learn the chat template"`

---

## D. Summary, "Why it's needed": the first 3 bullets (page 1, lines 18-23)

**Where:** page 1, under the bold "Why it's needed:". Change the first 3 bullets. The 4th bullet ("Every open-soure technique …") stays as it is.

Prompt for Gemini:

```text
In the summary, under "Why it's needed:", replace the first three bullets (the ones that start with "In the InstructGPT paper", "In the Zephyr paper" and "Chu et al.:") with the three bullets below. Each bullet has one sub-bullet under it, one level indented. Do not change the fourth bullet ("Every open-soure technique …") or anything else.
```

New text:

```markdown
- In the InstructGPT paper, people preferred the answers of a 1.3B GPT-3 trained with SFT and RLHF over the answers of the 175B GPT-3 that was only pretrained , "despite having 100x fewer parameters" ,,, ,
    - The paper's goal : "aligning language models with user intent" (page 1) , first with SFT on answers written by people , then with RLHF on people's rankings (Section 1.8.1) ,,, ,
- In the Zephyr paper, skipping SFT and going straight to preference tuning (DPO) dropped MT-Bench from 7.00 to 4.76 (the same training , only the SFT step removed) , and the model "struggles to learn the chat template" ,,, ,
    - The paper's goal : a small 7B chat model "aligned to user intent" (page 1) , made cheaply , with no human labels and a few hours of training , with SFT on answers written by a bigger model (GPT-3.5-turbo) , then DPO on GPT-4's ratings (Section 1.8.2) ,,, ,
- Chu et al.: when the base model does not follow instructions, "without SFT, all end-to-end RL runs fail to improve" ,,, ,
    - The paper's goal : compare what SFT and RL each teach a model , their main finding is that SFT memorizes the training data and RL learns rules that also work on new variants , and this line is one of their side checks (Section 1.8.3) ,,, ,
```

---

## E. NEW Section 1.8 (page 6, between line 24 and line 25)

**Where:** page 6. After the paragraph that starts "For Navin bro's model(1.5B pretrained on 8.5B tokens)" and ends "… and I'll run the base model on the same tests as the SFT model ,,, ,". Before the heading "2. Where SFT sits in the full pipeline:".

It's long, so give it in 4 parts: E1, E2, E3, E4.

Prompt for Gemini, with E1:

```text
Insert the following new section right after the paragraph that ends with "and I'll run the base model on the same tests as the SFT model ,,, ," and before the heading "2. Where SFT sits in the full pipeline:". The "1.8" heading uses the same heading style as "1.7 What happens if SFT is skipped :", and "1.8.1" uses the next smaller heading style. Change nothing else.
```

Prompt for Gemini, with E2, E3 and E4:

```text
Insert the following right after the text you inserted last, and before the heading "2. Where SFT sits in the full pipeline:". The "1.8.x" heading uses the same style as the "1.8.1" heading. Change nothing else.
```

### E1 (1.8 and 1.8.1, InstructGPT)

```markdown
## 1.8 The three papers in detail : what they did , what they compared , and the exact lines :

These are the exact lines from the 3 papers in the summary , with page numbers , so anyone can open the paper and check , and the page numbers are the PDF pages of the arXiv version I give with each link ,,, ,

### 1.8.1 InstructGPT (Ouyang et al. , OpenAI , 2022 , https://arxiv.org/abs/2203.02155v1)

What they trained : 3 sizes , 1.3B , 6B and 175B , and "all of our models use the GPT-3 architecture" (pages 2-3) , and they start from GPT-3 that was only pretrained : "We start with the GPT-3 pretrained language models from Brown et al. (2020)" (page 8 , Section 3.5) ,,, ,

Then 3 steps on top of it (page 6 , Section 3.1 , and the prompt counts are on page 7 , Section 3.2) :

1. SFT : people wrote good answers to about 13k prompts , and GPT-3 was trained on them ,,, ,
2. Reward model : people ranked several answers to the same prompt (33k prompts) , and a separate model learned to predict which answer people like ,,, ,
3. RLHF (PPO) : the SFT model was trained further on 31k prompts to get a high score from the reward model , with some pretraining data mixed in so it doesn't get worse on public NLP tasks (the paper calls this PPO-ptx) , and that model is InstructGPT : "Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models" (page 9) ,,, ,

Who was compared with whom :

| model | size | training |
|---|---|---|
| GPT-3 | 175B | pretrained only , no SFT , no RLHF |
| InstructGPT | 1.3B | the 1.3B GPT-3 , plus SFT and RLHF |

and people preferred the 1.3B InstructGPT's answers over the 175B GPT-3's answers ,,, ,

How "better" was measured : by people , not by benchmark scores , labelers saw the models' answers to prompts that real customers sent to the OpenAI API (from customers whose prompts were not used in training) , gave each answer a 1-7 score and ranked the answers , and a model's score is how often its answer was preferred to the 175B SFT model's answer (page 10 , and the rating screen is on page 39 , Figure 12) : "Our main metric is human preference ratings on a held out set of prompts from the same source as our training distribution" (page 10) ,,, ,

The exact lines :

- page 1 , abstract : "In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters." ,,, ,
- page 3 : "These models have the same architecture, and differ only by the fact that InstructGPT is fine-tuned on our human data. This result holds true even when we add a few-shot prompt to GPT-3 to make it better at following instructions. Outputs from our 175B InstructGPT are preferred to 175B GPT-3 outputs 85 ± 3% of the time, and preferred 71 ± 4% of the time to few-shot 175B GPT-3." (a few-shot prompt means a few example questions and answers put before the real question , to push the base model into answering) ,,, ,
- page 2 , Figure 1 : every model is compared with the same reference , the 175B SFT model , and my reading of the chart (approximate) is that the 175B GPT-3 wins about 0.24 of the time , the 1.3B SFT-only model about 0.35 , and the 1.3B InstructGPT (SFT + RLHF , the PPO-ptx line) about 0.53 , so even SFT alone at 1.3B wins more often against that same reference than the 175B base ,,, ,

What it means for me : size alone doesn't make a model useful , a small model fine-tuned on human data (SFT on good answers , then RLHF on people's rankings) is preferred by people over a 100x bigger model that was only pretrained ,,, ,
```

### E2 (1.8.2, Zephyr)

```markdown
### 1.8.2 Zephyr (Tunstall et al. , Hugging Face , 2023 , https://arxiv.org/abs/2310.16944v1)

The words first :

- dSFT : SFT on answers written by a bigger model ("distilled") , here UltraChat , 1.47M dialogues generated by GPT-3.5-turbo , which Zephyr cleaned and filtered down to about 200k for SFT (page 5) ,,, ,
- preference tuning : teaching the model which of 2 answers is better , and DPO is one way to do it , not the only one , RLHF with PPO (InstructGPT's way) is another , and DPO learns directly from (better , worse) answer pairs , without training a separate reward model ,,, ,
- dDPO : DPO where the better / worse labels came from GPT-4 , not from people , UltraFeedback has 64k prompts with 4 answers each "rated by GPT-4" , and the best one is "chosen" and one of the other 3 , at random , is "rejected" (page 5) ,,, ,

The test (Table 3 , page 8 , and the 4 ways are defined on page 7) : Mistral-7B trained 4 ways :

| way | MT-Bench | AlpacaEval |
|---|---|---|
| DPO only , no SFT | 4.76 | 30.76% |
| SFT , 1 epoch | 6.64 | 85.65% |
| SFT 1 epoch , then 1 more SFT epoch on UltraFeedback's best answers | 6.19 | 78.54% |
| SFT 1 epoch , then DPO 1 epoch | 7.00 | 86.07% |

The exact lines :

- page 7 : "without an initial SFT step (dSFT), models are not able to learn at all from feedback and perform terribly." ,,, ,
- page 13 , Appendix A.2 "SFT is a required step before DPO" : "We observed a significant reduction in performance in both the MT-Bench and AlpacaEval scores when the SFT step is skipped. After a qualitative evaluation of the MT-Bench generations, we observe that the pure DPO model struggles to learn the chat template" ,,, ,
- page 14 , Figure 7 , the example (a race question in 2 turns) : in turn 1 the DPO-only model gives the right answer and then doesn't stop , it keeps printing <|system|> <|user|> <|assistant|> again and again , in turn 2 it gives a wrong answer and does the same , and the SFT + DPO model answers and stops in both turns ,,, ,

About 7.34 : the final Zephyr-7B scores 7.34 on MT-Bench and 90.60% on AlpacaEval (Table 1 , page 6) , and its strongest model "was obtained with one epoch of SFT followed by three epochs of DPO" (page 7) , so 7.34 isn't the same training as the 4.76 run , and the fair comparison is 7.00 ,,, ,

The 2 benchmarks :

- MT-Bench (Zheng et al. , 2023 , https://arxiv.org/abs/2306.05685v4) : 80 questions with 2 turns each (page 6 , Section 3.5 : "every question involves two turns") , in 8 categories , writing , roleplay , extraction , reasoning , math , coding , knowledge I (STEM) and knowledge II (humanities/social science) (page 3 , Section 2.2 , and page 4) , the model answers a question , then a follow-up question , GPT-4 reads each answer and gives it 1 to 10 , and the score is the average , the default judge prompt is on page 14 (Figure 6) , and for the score they used "our prompt templates (Figure 6, Figure 10)" (page 9) , where Figure 10 (page 17) is the 2-turn version that also gives GPT-4 a reference answer , and Zephyr calls it "160 questions" (Zephyr page 5 , Section 4.2) because that's 80 questions x 2 turns , and the MT-Bench paper writes it as "160 = 80 × 2 turns" (MT-Bench page 9 , Section 5) ,,, ,
- AlpacaEval : 805 single questions , "mostly focused on helpfulness" , and GPT-4 compares each answer with the answer of a baseline model (text-davinci-003) , and the score is the % of wins (Zephyr page 5 , Section 4.2) ,,, ,
```

### E3 (1.8.3, Chu et al. and DeepSeek)

```markdown
### 1.8.3 Chu et al. ("SFT Memorizes, RL Generalizes" , ICML 2025 , https://arxiv.org/abs/2501.17161v2)

The paper's main question (page 1 , Introduction) : "whether SFT or RL primarily memorize training data ... or whether they learn generalizable rules that can adapt to novel task variants" , and their answer (page 1 , abstract) : RL "generalizes" , SFT "tends to memorize the training data" , and "SFT is still helpful for effective RL training: SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains." ,,, ,

The task (page 3) : "GeneralPoints" , the model gets 4 cards and must write an equation that equals 24 , using each card once , and a program checks the answer , so RL can get a reward without people ,,, ,

The test (page 7 , Section 5.4 "The Role of SFT for RL Training") : RL directly on the base Llama-3.2 , no SFT first ,,, ,

The exact lines :

- page 7 : "SFT is necessary for RL training when the backbone model does not follow instructions. Figure 9 shows that without SFT, all end-to-end RL runs fail to improve. More specifically, we observe that without SFT, the base model suffers from poor instruction following capability. A detailed failure case is provided in Figure 20 (in Appendix D.3), revealing that the base Llama-3.2-Vision-11B model tends to generate long, tangential, and unstructured responses. This issue makes it impossible to retrieve task-related information and rewards for RL training." ,,, ,
- page 8 , Figure 9 : "All trials fail due to poor instruction following capability of the base model." ,,, ,

Why it fails : RL learns only from the reward , and without SFT the base model doesn't follow the instructions , it "tends to generate long, tangential, and unstructured responses" (page 7) instead of the JSON answer the prompt asks for (in Figure 20 , page 20 , it starts writing Python code and "fails to finish within finite context length") , and the paper says this "makes it impossible to retrieve task-related information and rewards for RL training" (page 7) , so RL has nothing useful to learn from , and in Figure 9 (page 8) the success rate doesn't go up , it falls to about 0 in all 3 runs (my reading of the chart) , and SFT first fixes this : "SFT stabilizes the model's output format" (page 1) ,,, ,

The condition , same page 7 : "Note that due to the difference in backbone model, our results do not contradict with DeepSeekAI et al. (2025), which suggests that SFT is unnecessary for downstream RL training." ,,, ,

- "downstream RL" means the RL step that comes later in the pipeline , and DeepSeek showed that RL can run straight on a base model with no SFT before it : DeepSeek-R1-Zero is "a model trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step" (DeepSeek-R1 v1 , https://arxiv.org/abs/2501.12948v1 , page 1 , abstract) , so "SFT is unnecessary for downstream RL" means "you don't always need SFT before RL" ,,, ,
- why both can be true : it depends on the base model , Chu et al. say the reason is "the difference in backbone model" (page 7) , DeepSeek's base (DeepSeek-V3-Base , a mixture-of-experts model with 671B parameters in total and 37B used for each token : DeepSeek-R1 v1 , page 13 , Table 4 gives these numbers for DeepSeek-V3 and DeepSeek-R1 , and both are built on DeepSeek-V3-Base) was given only a prompt template that asks for "<think> reasoning process here </think> <answer> answer here </answer>" (DeepSeek-R1 v1 , page 6 , Table 1) , and R1-Zero's AIME 2024 score was already 15.6% at the start of RL (page 6 : "jumping from an initial 15.6% to an impressive 71.0%") , so my reading (the paper doesn't say it in these words) is that some of its answers could already be checked and scored , and RL had rewards to learn from , while Chu's base (Llama-3.2-Vision-11B) had "poor instruction following capability" (Chu page 7) , so RL couldn't get usable rewards from its answers and every run failed to improve (Chu Figure 9 , page 8) ,,, ,
- and DeepSeek's newer version says the same thing about size : RL straight from a base model "is highly dependent on the underlying model capacity" , and their 7B and 16B bases "consistently failed to yield meaningful improvements" (DeepSeek-R1 v2 , https://arxiv.org/abs/2501.12948v2 , pages 62-63 , Appendix G.1) ,,, ,
- even DeepSeek added SFT in the end : R1-Zero "encounters challenges such as poor readability, and language mixing" (v1 page 1) , so the released DeepSeek-R1 starts by "collecting thousands of cold-start data to fine-tune the DeepSeek-V3-Base model" (SFT) and then runs RL (v1 page 3) ,,, ,
- for my 1.5B base : it's much smaller than the 7B and 16B bases where DeepSeek's RL-from-base failed , and I haven't tested yet whether it follows a format from the prompt alone , because BluTrain can't generate text yet (Section 6.3) , so SFT first is the safe choice (Section 1.7 , Section 2.6) ,,, ,
```

### E4 (1.8.4, the three together)

```markdown
### 1.8.4 What the three show together

| paper | what it shows |
|---|---|
| InstructGPT | without post-training , even a 100x bigger model loses to a small one fine-tuned on human data (SFT on good answers , then RLHF) |
| Zephyr | skipping SFT and going straight to DPO fails (MT-Bench 7.00 → 4.76) , and the model "struggles to learn the chat template" |
| Chu et al. | skipping SFT and going straight to RL fails when the base can't follow instructions , its long , unstructured answers give RL no usable reward , so no run improves |

So SFT comes first , and every later step (DPO , RL) builds on it ,,, ,
```

---

## F. Section 3, a note right under the "3." heading (page 13, between line 7 and line 8)

**Where:** page 13. Right under the heading "3. How SFT runs in BluTrain :". Before the heading "3.1 The whole flow :".

Prompt for Gemini:

```text
Right under the heading "3. How SFT runs in BluTrain :" and before the heading "3.1 The whole flow :", insert the following paragraphs and the table, in normal text style (not a heading). Change nothing else.
```

New text:

```markdown
BluTrain is our own C++/CUDA trainer on our own tensor library , and the SFT runs there , not in a Python package , and the Python libraries are only references , I read their code to copy the right ideas ,,, ,

PyTorch's own post-training libraries are no longer being developed (torchtune has ended , torchforge is paused) : torchtune says it "is no longer actively maintained: torchtune development wound down in 2025" , and torchforge says "Development in Forge has paused. LLM training at PyTorch is being consolidated in torchtitan" (their GitHub README notices , checked in Sept 2026) ,,, ,

What I take from each :

| idea | from |
|---|---|
| count the loss only on answer tokens , ignore the prompt and padding | TRL , torchtitan , Megatron-LM , and PyTorch's ignore_index |
| divide the loss by the answer tokens of the whole step , not per micro-batch | TRL (num_items_in_batch) , torchtitan , Megatron-LM (--calculate-per-token-loss) |
| the tokenized prompt must be an exact prefix of the tokenized full sample | torchtitan |
| every sample ends with exactly one end token (added if missing , never doubled) , and samples that are too long are dropped instead of cut | torchtitan |
| don't compute the lm_head on prompt rows (later) | TRL (chunked_nll) |
```

---

## G. Section 3.2, 2 paragraphs at the end (page 14, between line 4 and line 5)

**Where:** page 14. After the paragraph that ends "… so using one template consistently matters more than which one ,,, ,". Before the heading "3.3 The tokenizer's job :".

Prompt for Gemini:

```text
In section 3.2, right after the paragraph that ends with "matters more than which one ,,, ," and before the heading "3.3 The tokenizer's job :", insert these two paragraphs in normal text style. Change nothing else.
```

New text:

```markdown
What goes wrong without it : in Zephyr , the model trained with DPO but without SFT answered (in turn 1 correctly , in turn 2 wrongly) and then didn't stop , it kept printing <|system|> <|user|> <|assistant|> again and again (Zephyr page 14 , Figure 7 , Section 1.8.2) , and the paper only says "the pure DPO model struggles to learn the chat template" (Zephyr page 13 , Appendix A.2) , and my reading is that SFT is what teaches the model that after <|assistant|> comes an answer , then the end token , and then it stops ,,, ,

Where the template is used , in 2 places , and it has to be the same in both : in the preprocessing script , every SFT sample is wrapped in it before tokenizing (Section 3.10) , and when the model is used (chat , evals) , the question is wrapped the same way and ends with "<|assistant|>\n" , and the model writes the answer and the end token , and GPT-2's tokenizer has no chat template , so I have to define one (Section 3.4) ,,, ,
```

---

## H. Sources, one line (page 35, between line 6 and line 7)

**Where:** page 35, in "Sources", at the end of the list "Papers i collected to refer :". After the bullet that starts "OLMo 3: Team Olmo, Olmo 3, 2025." (the last bullet of that list). Before "Papers from my week-1 reading (numbers taken from my notes) :".

Prompt for Gemini:

```text
In "Sources", after the bullet that starts "OLMo 3: Team Olmo" and before the line "Papers from my week-1 reading", add one more bullet in the same style with this text. Change nothing else.
```

New text:

```markdown
- Page numbers in Section 1.8 are the PDF pages of these arXiv versions : InstructGPT v1 , Zephyr v1 , Chu et al. v2 , MT-Bench v4 (https://arxiv.org/abs/2306.05685v4) , DeepSeek-R1 v1 (https://arxiv.org/abs/2501.12948v1) , and one line from DeepSeek-R1 v2 (https://arxiv.org/abs/2501.12948v2 , pages 62-63 , Appendix G.1) , and the two DeepSeek versions have different page numbers ,,, ,
```

---

## Checks after all edits (Ctrl+F)

| Search | Expected |
|---|---|
| `7.34` | 4 matches, all in new text that explains 7.34 is the final model: 1 in the 1.3 Zephyr bullet, 3 in the 1.8.2 paragraph "About 7.34" |
| `90.60%` | 2 matches: the 1.3 Zephyr bullet and the 1.8.2 paragraph "About 7.34" |
| `7.00 → 4.76` | 3 matches: the 1.7 table, the 2.6 table, the 1.8.4 table |
| `from 7.00 to 4.76` | 1 match: the summary |
| `doesn't learn the chat template` | 0 matches |
| `1.8 The three papers in detail` | 1 match, before "2. Where SFT sits in the full pipeline" |
| `What I take from each` | 1 match, right under "3. How SFT runs in BluTrain" |
| `What goes wrong without it` | 1 match, at the end of 3.2 |
| `<https` | 0 matches (no link in angle brackets) |

Then File → Download → PDF.
