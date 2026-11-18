<!-- Markdown copy of the Google Doc "PostTraining - Research and Planning", tab "SFT : what it is" (exported 2026-10-15 13:00), with the fixes listed in sft_work_documentation.md section 45.2. Images are in post_training_research_sft_images/. -->

# SFT : what it is ? Why and How every post-training step is connected/dependent on each? and How Mid-training can be validated by RL ? and How is pretraining+midtraining+sft+DPO+RL are connected to each ?, and what's next :

This doc covers my SFT work : what SFT is and why it's needed, how it works in BluTrain , what I changed in the kernels , and what I planned to do next , and I've put the source next to every number , so that anyone can check it and the papers i referred , with links also , are listed at the end and i explained each and every topic from basics , so that anyone reading this can understand on my intentions ,,, , read the whole doc patiently ,,, ,

## Summary :

**What SFT is:** After pretraining, a model (the "base model") can only continue text, if you ask it a question, it may write more questions, or never stop, SFT (supervised fine-tuning) is more training on examples of a prompt and a good answer, so the model learns to answer, the loss is the same next-token cross-entropy used in pretraining, the only difference is that the loss is counted on the answer tokens, and the prompt and padding are ignored.

**Why it's needed:**

* In the InstructGPT paper, they preferred answers from a 1.3B model trained with SFT and RLHF over the 175B base GPT-3, "despite having 100x fewer parameters" ,,, ,
* In the Zephyr paper, skipping SFT and going straight to preference tuning dropped MT-Bench from 7.00 to 4.76, and the model "struggles to learn the chat template" ,,, ,
* Chu et al.: when the base model does not follow instructions, "without SFT, all end-to-end RL runs fail to improve" ,,, ,
* Every open-source technique , I read starts post-training with SFT : Tülu 3, OLMo 3, Llama 2 and 3, DeepSeek-R1 and Qwen3 ,,, ,

## 1. What SFT is and why it's needed:

### 1.1 Quick background : tokens, logits and the loss:

A language model does not read letters , it reads tokens. A token is a piece of text : a word, part of a word, a number or a symbol and each token has an id and the tokenizer is GPT-2's , with 50,257 ids (0 to 50256) and the model's vocab has 50,304 rows : the 50,257 real ids plus 47 unused rows , which make the size GPU-friendly ,,, ,

At every position, the model outputs one score for every vocab entry, so 50,304 scores per position, and these scores are the logits, and the Softmax turns them into probabilities that add up to 1.

Training pushes the model to give a high probability to the token that actually comes next. The loss at one position is minus the log of that probability :

```text
probability 0.90  ->  loss 0.105   (good guess)
probability 0.50  ->  loss 0.693
probability 0.01  ->  loss 4.605   (bad guess)
all 50,304 equal  ->  loss ln(50304) = 10.83   (a model that knows nothing)
```

```text
L = -(1/N) * sum over counted positions t of  log p( y_t | y_<t )
```

y_t is the correct token at position t , y_<t is everything before it, and N is the number of counted positions and the backward pass turns the loss into a gradient for each weight, and the optimizer (AdamW or Muon in BluTrain) moves the weights a small step ,,, ,

### 1.2 Pretraining and the base model:

Pretraining is the big first stage and the model reads a huge amount of text (OLMo 3 used 5.9T tokens) and learns to predict each next token and the correct answer is just the next token of the text , so nobody writes labels  and people call this self-supervised and it still has a real label and a real cross-entropy loss,,, ,

The result is a base model , it knows grammar , facts and a lot of skills , but the only thing it does is continue text the way its training documents usually continue ,,, ,

The base iam using for SFT is Navin anna’s  ~1.5B model trained on 8.4B tokens and the final checkpoint (step 8000) is on Hugging Face as navingv/curriculum_8btokens , blumodel_step_8000 , trained with Muon on 4 GPUs. It uses the GPT-2 tokenizer , vocab 50,304, causal attention with RoPE , and the same table for the input embedding and the output layer (weight tying) ,,, ,

### 1.3 What a base model does when you ask it something:

A base model doesn't know it's being asked a question and it just continues the text and it might answer , or write more questions in the same style , or write a web page around the question , or never stop, because nothing taught it where an answer ends and it has also never seen a chat format with a "user" and an "assistant" ,,, ,

Papers that measured this :

* InstructGPT (OpenAI, 2022) : "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters". The difference was only post-training: SFT on about 13k prompts, then a reward model and PPO ,,, ,
* Zephyr (Hugging Face, 2023): with preference tuning (DPO) but without SFT first, MT-Bench was 4.76 instead of 7.00, and AlpacaEval 30.76% instead of 86.07%. The authors say the model "struggles to learn the chat template" ,,, ,
* OLMo 3 (Ai2, 2025): when they put chat special tokens into midtraining data, the model printed those tokens at test time and GSM8K fell from 49.43 to 0. A chat format has to be taught on purpose ,,, ,

### 1.4 What SFT is, exactly:

SFT means you keep training the base model , but on examples of a prompt followed by a good answer , and count the loss only on the answer ,,, ,

The loss function doesn't change and what changes is which positions count ,,, ,  Llama 2 says it directly -> "we zero-out the loss on tokens from the user prompt , so as a result , we backpropagate only on answer tokens" ,,, ,

|  | pretraining | SFT |
| :---- | :---- | :---- |
| loss | next-token cross-entropy | the same |
| optimizer, forward, backward | normal | the same |
| where the correct token comes from | the text itself | an answer written by a person or a stronger model |
| positions counted in the loss | all of them | only the answer, plus the end token |
| data | billions to trillions of tokens | thousands to a few million examples |
| learning rate | pretraining schedule | much smaller (5e-6 in Tülu 3) |

**Example :** The user asks "What is the capital of France?" and the answer is "Paris." After the chat template (Section 3.2) and tokenizer (Section 3.3), it's a list of tokens and I've written the tokens as text so it's readable , in the table below each marker is shown as one entry and the newlines are left out , and in reality the row is 24 GPT-2 tokens : `<|user|>\n` is 6 tokens (`<` `|` `user` `|` `>` `\n`) , then the question (7 tokens) , then `\n` , then `<|assistant|>\n` is 7 tokens (`<` `|` `ass` `istant` `|` `>` `\n`) , then `Paris` `.` and the end token 50256 , and the input at position t is token t , and the target is token t+1 ,,, ,

| pos | input | target | counted? | label in the file |
| :---- | :---- | :---- | :---- | :---- |
| 0 | `<\|user\|>` | What | no, prompt | 65535 |
| 1 | What | is | no | 65535 |
| 2 | is | the | no | 65535 |
| 3 | the | capital | no | 65535 |
| 4 | capital | of | no | 65535 |
| 5 | of | France | no | 65535 |
| 6 | France | ? | no | 65535 |
| 7 | ? | `<\|assistant\|>` | no | 65535 |
| 8 | `<\|assistant\|>` | Paris | yes, first answer token | id of "Paris" |
| 9 | Paris | . | yes | id of "." |
| 10 | . | `<\|endoftext\|>` | yes, this teaches it to stop | 50256 |
| 11 | `<\|endoftext\|>` | (pad) | no, padding | 65535 |
| 12 | (pad) | (pad) | no | 65535 |

Only 3 of the 13 positions count and “65535” is just a marker for "don't count this position" (Section 3.5) , and the loss for this example is the average of “-log p” at positions 8, 9 and 10 ,,, ,

**Why the prompt still goes in :** The prompt isn't removed and the whole row goes through the forward pass and to predict "Paris" at position 8 , the model has to look back at "capital of France" through attention ,  and in the backward pass , the gradient from the answer flows back through attention into how the prompt was processed , and into all the weights , so , the model learns how to read and use a question , but it's never trained to write the user's question and that's also why the prompt costs compute in forward and backward even though it adds nothing to the loss ,

**Why the end token has to count :** The target at position-10 is the end-of-text token (50256) , if it were masked , the model would never learn to stop after an answer ,,, ,

### 1.5 Why a small amount of data is enough for the format :

* "format" here means how to answer : reply to the question , follow the chat layout , stop at the end , and the model already has the knowledge from pretraining , so , SFT only has to teach this answering style , which needs far less data than pretraining ,,, ,

* LIMA (Meta, 2023) : they fine-tuned a 65B LLaMa on only 1,000 hand-picked examples , with no RL , then people compared its answers with other models , LIMA was equal or better than GPT-4 in 43% of cases , Bard in 58% and DaVinci003 in 65% , their conclusion : " almost all knowledge in large language models is learned during pretraining , and only limited instruction tuning data is necessary to teach models to produce high quality output ", this idea is called the **superficial alignment hypothesis** , meaning SFT mostly changes the surface (the style of answering) , not what the model knows ,,, ,

* Llama 2 (Meta, 2023) : each "annotation" is one (prompt + answer) written by a paid annotator , they stopped at 27,540 becoz a few thousand of their own good examples beat millions of public examples , so their section is titled "Quality Is All You Need" ,,, ,

* InstructGPT (OpenAI, 2022) : their SFT set was only about 13k prompts with answers written by their labelers ,,, ,

* but in SFT , "a few thousand is enough" is only true for the format (how to answer) , when SFT also has to teach skills , like solving math step by step , writing code , refusing harmful requests and following exact instructions , it needs many good examples of each skill , which is why the Tülu-3’s SFT set has 939,344 examples and the OLMo 3 Think SFT set has 2,268,468 , both are SFT-data , just covering many skills at once , and Tülu 3 showed this directly by removing one skill's data from its SFT mix at a time : without math data GSM8K fell from 76.2 to 64.1 , without safety data , safety-tests fell from 93.1 to 74.7 , and without instruction-following data IFEval fell from 72.8 to 53.6 , so if a skill is not in the SFT-data , the model doesn't learn it (Table 10, Section 5.2) ,,, ,

less data vs more data : "less is more" does not mean small is always better , LIMA's own tests (7B model) show that better answers help (filtered beat unfiltered) , more varied questions help (Stack Exchange beat wikiHow) , but more of the same data does not (up to 16x more gave no gain , "performance ... plateaus") , so , more data helps only when it adds new kinds of examples at the same quality , and that is what Tülu 3's 939,344 examples are : 19 datasets covering chat , knowledge , math , code , safety and exact instructions , where each skill's data raised that skill's score (Table 10) ,,, ,

### 1.6 Why not only SFT ? its limits :

* SFT only copies : in SFT , the loss rewards the model for matching the given answer token by token , so it learns to reproduce exactly what is in the data , if the answers in the data are average or wrong , the model learns average or wrong answers, and nothing in SFT tells the model that a better answer exists , so , in general it can't get better than its data , RL works differently , the model writes its own answers , a checker or reward model scores them , and the model is pushed towards the higher-scoring ones , so , it can find answers better than the ones in the data (Section 2.4) ,,, ,

* SFT generalizes worse outside its training data : Chu et al. trained the same model with SFT and with RL on the same tasks , then tested it on a changed version of each task that it never saw in training ,,, ,

* GeneralPoints-L is a card game where the model must use the numbers on 4-cards to make 24 , in training J, Q and K all counted as 10 , in the test they counted as 11, 12 and 13 , starting from 11.5% , RL went up to 15.0% (+3.5%) but SFT went down to 3.4% (-8.1%) ,,, ,

* V-IRL-L is a navigation task with text directions , in training the moves were north , east , south , west , in the test they were left , right , slightly-left , slightly-right , starting from 80.8% , RL went up to 91.8% (+11.0%) but SFT dropped to 1.3% (-79.5%) ,,, ,

* so , SFT memorized the exact rules , it was trained on and broke when the rules changed , while RL learned the skill itself , which is why the paper is called "SFT Memorizes , RL Generalizes" ,,, ,

* Too much SFT before RL can lower what RL reaches : SimpleRL-Zoo did SFT on Mistral-24B with about 15K short math reasoning examples , then ran the same RL from 3 starting points : no-SFT , 100-SFT steps and 500-SFT steps , and no-SFT reached the best final result , 100 steps lower , and 500 steps the lowest , becoz SFT makes the model stick to one fixed answer style , so during RL , it tries fewer different answers and finds fewer better ones , in their words SFT "significantly limits the model's ability to explore freely during RL" , and this gets "increasingly pronounced as the number of initial SFT steps grows" ,,, ,

* SFT on weaker data hurts : OLMo 3 first trained their think model with SFT on their own strong reasoning data (Dolci Think SFT), then tried more SFT on answers written by Qwen3-32B , and every evaluation dropped ("outright hurts" , Table 21) , becoz , those answers were weaker than what the model had already learned , so copying them pulled it down , but the same Qwen3-32B answers helped when used in DPO as the "chosen" side against worse answers from the small Qwen3-0.6B as the "rejected" side , becoz DPO learns from the difference between a better and a worse answer , not by copying one answer , in their words " DPO yields gains where SFT on the same data cannot " (Section 2.3) ,,, ,

* SFT can make the model forget : if the SFT data covers only a narrow area (for example only math) or the model is trained for too many epochs , it can lose general knowledge it learned in pretraining , even in Tülu 3 , TruthfulQA "actually drops as the amount of data in the mix increases" (Figure 4) , so I'll run the base model and the SFT model on the same knowledge tests (MMLU, HellaSwag) to catch this , and keep SFT to 1-2 epochs with a mix of general data (Section 6.1) ,,, ,

### 1.7 What happens if SFT is skipped :

| instead of SFT... | what happens | source |
| :---- | :---- | :---- |
| use the base model as it is | continues text instead of answering , doesn't know a chat format, doesn't know when to stop | InstructGPT : 1.3B tuned beat 175B base |
| go straight to DPO | much worse chat, doesn't learn the chat template | Zephyr : MT-Bench 7.00 → 4.76, AlpacaEval 86.07% → 30.76% |
| go straight to RL, and the base doesn't follow instructions | RL gets no useful reward and doesn't improve | Chu et al.: "all end-to-end RL runs fail to improve" |
| go straight to RL, and the base already follows instructions | can work for reasoning, but answers are hard to read | DeepSeek-R1-Zero: "poor readability and language mixing"; SimpleRL-Zoo: Qwen2.5 bases "already exhibit instruction-following abilities" |

For Navin bro’s model (1.5B pretrained on 8.4B tokens) , I haven't measured yet whether it follows instructions , the tokenizer has only one special token (end-of-text) , so the pretraining data can't contain chat special tokens , and BluTrain can't generate text yet to test it (Section 6.3) , so I'm planning SFT first , and I'll run the base model on the same tests as the SFT model ,,, ,

## 2. Where SFT sits in the full pipeline:

```text
PRETRAINING -> MIDTRAINING -> (LONG CONTEXT) -> SFT -> DPO -> RL (RLVR / GRPO)
|------------- base model -------------|       |------- post-training -------|
```

| stage | data | size in OLMo 3 7B | what it teaches |
| :---- | :---- | :---- | :---- |
| pretraining | web text, PDFs, code | 5.9T tokens | language, facts, general skills |
| midtraining | curated math, code, QA, instruction and thinking data | 100B tokens | boosts target skills, prepares for post-training |
| long context | long documents | 50B tokens | long inputs and outputs |
| SFT | prompt + good answer | Think 45.4B tokens, Instruct 3.4B tokens | format, chat, following instructions, reasoning style |
| DPO | prompt + better answer + worse answer | Instruct DPO 259,922 prompts | which of two answers is better |
| RL | prompts with a checkable answer | Tülu 3 used 29,946 | getting checkable answers right |

Sizes are from the OLMo-3 paper , except the RL row , which is from Tülu 3 and all stages up to SFT use next-token cross-entropy ; SFT counts only the answer tokens ,,, ,

### 2.1 Midtraining:

Midtraining continues pretraining on a smaller , much better data mix : more math , code , reasoning and QA , less generic web text and same loss , OLMo 3 midtrains on 100B tokens (Dolmino Mix) , "combining our highest-quality pretraining data with substantial task data for math and code problems , general knowledge QA , instruction following , and more " ,,, ,

2 OLMo-3 findings are useful here :

* **Adding Instruction and Thinking Data :**

  When running two separate 100B-token midtraining sessions with the exact same total number of tokens -> one with instruction and thinking data and one without -> the model trained with that data performed better on every single base evaluation measure ,,, ,

The one with instruction and thinking data : this run took the standard midtraining corpus and mixed in specialized data containing instructions , problem-solving steps , and reasoning traces alongside normal text ,,, ,

The one without this data : this run used the exact same total number of tokens , but relied entirely on standard midtraining text without any instruction or thinking examples ,,, ,

* **Leaving Special Tokens for SFT :**
  one should keep special tokens (like `<|im_start|>` and `<|im_end|>`) out of the midtraining stage as when OLMo 3 included those special tokens in midtraining data , the model started printing them literally at test time , causing performance to plummet (for example, GSM8K dropped from 49.43 down to 0 , and CruxEval fell from 32.89 to 18.91) but however , when the exact same chat layout was written as ordinary text instead of special tokens , GSM8K stayed strong at 46.02 and as their report heading states : *"Leave special tokens for SFT stage" ,,, ,*

OctoThinker also reports that midtraining changes how well later RL works :

OctoThinker (arXiv 2506.20512) :

OctoThinker worked on the thought “what makes a base model good for RL” , they ran the same RL on Qwen2.5-3B and Llama-3.2-3B (GRPO on MATH8K math prompts , 16 sampled answers per prompt , answers up to 4,096 tokens , the simple template "Question:{}\nAnswer:{}"), Qwen's answers grew steadily and it improved on many math benchmarks , but Llama started its answers with "\boxed:{}" and then repeated text until it hit 4,096 tokens , and it barely improved (it even got worse on GSM8K) ,,, ,

to find out why , they ran many small midtraining experiments on Llama-3.2-3B , 20B tokens each , changing one thing at a time , and then ran the same RL on each result and compared :

(1) data quality : midtraining on high-quality math web data (MegaMath-Web-Pro) gave big RL gains, while a weaker math set (FineMath-4plus) gave only small gains and the same repeat-until-4,096 failure , in their words "High-quality math pre-training corpora play a dominant role in RL scaling" ,,, ,

(2) reasoning (CoT) question-answer data , mixed 9 parts web to 1 part QA : short reasoning answers gave no RL gain over web data alone , long reasoning answers (OpenR1-Math-220K , thinking inside `<think>...</think>`) gave big gains , but they made RL unstable , with sudden score drops and answers suddenly getting much longer ,,, ,

(3) a little instruction data (about 1% of the mix, ~0.8B tokens from Tülu 3 persona IF , WildChat and UltraChat) unlocked the gains of the short reasoning data and made answer length grow smoothly , for the long reasoning data it helped but did not stop the later collapse , so , they also used a clearer RL prompt template and raised the max answer length step by step (2,048 tokens for steps 0-200, 4,096 for steps 200-320, 8,192 for steps 320-400) , which kept RL stable ,,, ,

(4) how much midtraining : they ran one 100B-token midtraining run , saved checkpoints at 20B , 70B and 100B tokens , tested each checkpoint as a base model (right after midtraining , before any SFT or RL , with few-shot prompts , e.g. GSM8K 8-shot , i.e. 8 solved problems in the prompt before the test question , becoz a base model can't follow instructions yet and the examples show it the answer format) , then ran the same RL on each checkpoint and tested again zero-shot (only the question , no examples)

the results of (4) :

| checkpoint | base-model test (before RL) | after RL |
| :---- | :---- | :---- |
| 20B | lowest | lowest |
| 70B | clearly better than 20B | better |
| 100B | about the same as 70B | better than 70B |

Whole OctoThinker experiment table (Llama-3.2-3B) :

| # | experiment (what changed in midtraining) | base-model test (before RL) | after RL |
| :---- | :---- | :---- | :---- |
| 1 | math web data: MegaMath-Web-Pro | better than original Llama | big RL gains |
|  | math web data: FineMath-4plus | better than original Llama | small gains, answers repeat until 4,096 tokens |
| 2 | + short reasoning QA (web:QA = 9:1) | small gains | no gain over web data alone |
| 3 | + long reasoning QA (web:QA = 9:1) | small gains | big gains, but RL unstable |
| 4 | + 1% instruction data (Tülu 3 IF + WildChat + UltraChat) with short reasoning | MATH500: 41.2%, GSM8K: 71.8% | MATH500: 52.4%, GSM8K: 81.6% (better after 200 steps, length grows smoothly) |
| 5 | + 1% instruction data (Tülu 3 IF + WildChat + UltraChat) with long reasoning | MATH500: 43.6%, GSM8K: 73.2% | MATH500: 58.8%, GSM8K: 86.4% (better after 150 steps, but still collapses later) |
| 6 | budget : 20B checkpoint (one 100B-token run) | lowest | lowest |
| 7 | budget : 70B checkpoint | clearly better than 20B | better |
| 8 | budget : 100B checkpoint | about the same as 70B | better than 70B |
| 9 | final recipe : 200B constant LR + 20B decay | 10-20% better than Llama | Long-3B on par with Qwen2.5-3B |

so ,  from 20B to 70B the base model itself also got better , but from 70B to 100B the base model tests showed no gain while the RL result still improved, meaning the benefit of extra midtraining can stay hidden until RL is run , in their words "Increasing the mid-training budget can improve RL performance , even if such gains are not evident in base model evaluations " ,,, ,

and their final recipe ("Stable-then-Decay"): first 200B tokens at a constant learning rate , mostly MegaMath-Web-Pro-Max (72.5%) plus general web (DCLM 10%), text-and-code (10%), QA (5%) and code (2.5%), then 20B more tokens with the learning rate decaying (for 3B, 2e-5 → 2e-6), split into three branches, long reasoning (OpenR1 15%, AM-DeepSeek-Distilled 15%, instruction 10%, MegaMath-Web-Pro 55%), short reasoning, and a hybrid of both ,,, ,

results : every branch was 10-20% better than the original Llama on 13 math benchmarks (for example the 1B model on GSM8K went from 7.66 to 44.88) , and after the same RL , OctoThinker-Long-3B "reaches performance on par with Qwen2.5-3B", while the short and hybrid branches were slightly behind on hard benchmarks ,,, ,

what it means for us : how well RL works later depends a lot on what the base model saw in midtraining , a small amount of instruction data (~1%, written as plain text as OLMo 3 also says) helps , midtraining mixes should be compared by running RL on them (the base tests can miss the gain , which is also why OLMo 3 uses RL-Zero to compare midtraining mixes) , and for long-reasoning RL a clear prompt template and a max answer length raised step by step keep training stable ,,, ,

, Midtraining is a pretraining-side decision , not part of the SFT work and the one takeaway : if the team ever adds instruction–data to pretraining or midtraining , write it as plain text and leave the special tokens to SFT ,,, ,

### 2.2 SFT :

Covered in Section 1 , with the details in Section 3 , it's the first post-training stage , it teaches the chat format , how to answer , when to stop , and (for think models) the reasoning style , and once SFT is finished , all later training stages (such as preference tuning or reinforcement learning) rely on the assumption that the model has already mastered the basics of writing a clear , readable , and correctly formatted answer ,,, ,

### 2.3 Preference tuning (DPO):

After SFT , the model answers , but not always with its best answer , preference tuning trains on a prompt x , a chosen (better) answer y_w and a rejected (worse) answer y_l , and makes the better one more likely than the worse one ,,, ,

DPO is what Zephyr , Llama 3 , Tülu 3 and OLMo 3 use and it needs a frozen copy of the SFT model , the reference model π_ref :

```text
L_DPO = -log sigmoid( beta * [ log( pi(y_w|x) / pi_ref(y_w|x) ) - log( pi(y_l|x) / pi_ref(y_l|x) ) ] )

log pi(y|x) = sum of log-probabilities of the answer tokens (the same masked sums as SFT)
beta        = how far the model may move away from pi_ref
```

It needs no reward model and doesn't generate text during training ; it runs on fixed data (DPO paper) , Tülu 3 and OLMo 3 both use a length-normalized version ,,, ,

DPO helps on top of SFT -> Tülu 3, 8B (Table 6) :

|  | after SFT | after DPO | after RLVR (final) |
| :---- | :---- | :---- | :---- |
| average | 60.6 | 64.7 | 65.1 |
| IFEval | 72.8 | 81.1 | 82.4 |
| AlpacaEval 2 | 12.4 | 33.5 | 34.5 |
| GSM8K | 76.2 | 84.3 | 87.6 |
| MATH | 31.5 | 42.0 | 43.7 |

OLMo 3 highlights a key finding called **Delta Learning** : when using chosen answers generated by a large model (Qwen3 32B) and rejected answers from a smaller model (Qwen3 0.6B) , performing SFT directly on the chosen answers actually harms performance , causing every evaluation to drop , but however , applying DPO to those same chosen answers paired with the weaker ones is effective , proving that "DPO yields gains where SFT on the same data cannot" ,,, ,

Other preference methods I noted : IPO (regularizes harder) , KTO (works with single thumbs-up/down labels instead of pairs) , ORPO (SFT and preference in one loss) , and SimPO (no reference model) and a 2026 study found that which method wins changes with model size , so I'll measure at 1.5B instead of copying a ranking ,,, ,

### 2.4 Reinforcement learning (RL) :

In RL the model generates its own answers during training , each answer gets a reward , and the model is pushed toward higher-reward answers , becoz it learns from its own outputs , it can go past the training data , which SFT can't ,,, ,

| method | where the reward comes from | what it needs | used by |
| :---- | :---- | :---- | :---- |
| PPO (classic RLHF) | a reward model trained on human preferences | policy, reward model, value model, reference model | InstructGPT |
| GRPO | a checker or reward model; each answer is compared with a group of answers to the same prompt | no value model, several samples per prompt | DeepSeek-R1 |
| RLVR | a program checks the answer (exact math answer, unit tests, instruction constraints); correct gets a reward, wrong gets 0 | prompts with a checkable answer | Tülu 3 (with PPO), OLMo 3 |

Tülu 3's RLVR data includes 7,473 GSM8K training examples and 7,500 MATH training examples evaluated by exact match on the extracted answer , along with 14,973 instruction-following prompts featuring dedicated checkers , bringing the total to 29,946 prompts ,,, ,

One failure to know about is **reward hacking** -> In Tülu 3 runs that drifted too far from the reference model , the AI learned to satisfy the strict instruction checks while completely ignoring the actual task -> for ex : when asked to measure a pen while using the obscure word "craspedodromous" 25 times , it simply repeated that exact word 25 times , becoz reinforcement learning strictly optimizes whatever automated condition is checked , validation rules and checks must be designed with extreme care ,,, ,

->**Reward hacking** is when an AI finds a loophole to pass a test without actually doing the right thing ,

Imagine a teacher says : "I will give you full points if your essay includes the word 'craspedodromous' 25 times"  and the teacher forgets to check if the essay is actually answering the question and Instead of writing a good essay , the AI just outputs the word "craspedodromous" 25 times in a row and it technically follows the rule , so it gets a high score , but the actual output is completely useless and that is why automated checks and reinforcement learning rules have to be designed very carefully , so the AI can't cheat them ,,, ,

### 2.5 RL-Zero : RL straight from the base model :

* DeepSeek-R1-Zero learned strong reasoning this way , but had "poor readability and language mixing , occasionally combining English and Chinese within a single chain-of-thought response" and the released DeepSeek-R1 adds "thousands of cold-start data" (SFT) before RL ,,, ,
* **OLMo-3 RL-Zero :** Releases reinforcement learning models trained directly from base models across four categories: math, code, instruction following, and chat and **Math Data Filtering ->** uses 13,300 prompts , having removed any questions that the base model was already able to solve successfully 8 out of 8 times and **Prompt Templates ->** discovered that simple prompt templates significantly outperform standard post-trained templates (such as `<think></think>`) when training directly from a purely mid-trained model and **Core Purpose :** uses RL-Zero as a clean, controlled method to compare different mid-training data mixes ,,, ,

### 2.6 Is this order the best ? :

No paper proves one order is best for every model , and what they show :

For SFT first :

* Zephyr : DPO without SFT loses badly (Section 1.7) ,,, ,
* Chu et al. : "SFT is necessary for RL training when the backbone model does not follow instructions" ,,, ,
* DeepSeek-R1 : stage by stage (Table 3) , the cold-start SFT ("Dev1") made the model usable but first lowered reasoning and later-RL stages brought it back:

|  | R1-Zero (RL only) | Dev1 (+ cold-start SFT) | R1 (final) |
| :---- | :---- | :---- | :---- |
| AIME 2024 | 77.9 | 59.0 | 79.8 |
| IF-Eval | 46.6 | 71.7 | 83.3 |
| AlpacaEval 2.0 | 24.7 | 50.1 | 87.6 |
| ArenaHard | 53.6 | 77.0 | 92.3 |

In their words : "Dev1 exhibits a partial degradation in reasoning performance compared to DeepSeek-R1-Zero , most notably on the AIME benchmark". So SFT bought usability, and RL bought (and recovered) reasoning ,,, ,

For keeping SFT light before RL :

* SimpleRL-Zoo (10 base models) : In tests across 10 base models , the SimpleRL-Zoo study found that performing SFT on reasoning data *before* reinforcement learning (RL) actually reduced the final RL performance and furthermore , this -ve impact worsened with more SFT steps : for ex , on Mistral-24B , skipping SFT entirely yielded the best results , whereas , 100 SFT steps performed worse , and 500 SFT steps performed the worst ,,, ,
* Same paper , on why some bases don't need SFT : Qwen2.5 models , "due to their extensive use of synthetic data during pretraining , already exhibit instruction-following abilities and reflective behaviors" ,,, ,

For DPO and RL on top of SFT :

* Tülu 3 : SFT 60.6 -> DPO 64.7 -> RLVR 65.1 average (Section 2.3) ,,, ,
* OLMo 3 Think : "We train it via SFT , DPO , and RLVR , observing gains at each stage" ,,, ,
* OLMo 3 Instruct does better when its SFT starts from the Think SFT model : 44.5 -> 47.8 average (Table 29) ,,, ,
* A 2026 paper reports that SFT then RL , as 2-stages , beats mixing them in one stage for reasoning ,,, ,

What people actually did , and other orders people tried :

| order | who | result |
| :---- | :---- | :---- |
| SFT → reward model → PPO | InstructGPT, Llama 2 (with rejection sampling) | the original RLHF recipe |
| SFT + rejection sampling + DPO, no PPO | Llama 3 | a simpler, stable recipe |
| SFT → DPO → RLVR | Tülu 3, OLMo 3 | gains at each stage |
| cold-start SFT → RL → rejection sampling + SFT → RL | DeepSeek-R1 | strong reasoning model |
| long-CoT SFT → reasoning RL → thinking-mode fusion SFT → general RL | Qwen3 | one model with think and no-think modes |
| SFT only | LIMA | good on prompts like its training data, weaker out of distribution (Chu et al.) |
| RL only | R1-Zero, OLMo 3 RL-Zero, SimpleRL-Zoo | works if the base follows the format; hard-to-read answers; fails if it doesn't |
| DPO only | Zephyr ablation | MT-Bench 7.00 → 4.76 |
| SFT and preference in one loss | ORPO | skips the separate SFT stage |
| SFT and RL mixed in one stage | mixed-policy methods | reported worse than SFT then RL |
| think SFT, then instruct SFT | OLMo 3 Instruct | +3.3 average |

My read : pretraining -> midtraining -> SFT -> DPO -> RL is what every open recipe with published ablations uses , but it's not proven best for every model , how much SFT you need depends on the base : little , if it already follows instructions , necessary if it doesn't , for the 1.5B model , SFT first is the safe choice , I don't know yet if the base follows instructions , and RL needs generation , which BluTrain doesn't have yet  and I'd keep SFT moderate (1 to 2 epochs) and measure ,,, ,

### 2.7 Tülu 3 and OLMo 3 side by side :

Both are from Ai2(allen institute of ai) and publish every dataset , setting and ablation , so they're the two recipes , i can copy and compare against and OLMo-3 has 3 post-training paths from one base : Think, Instruct and RL-Zero :-

|  | Tülu 3 | OLMo 3 Think | OLMo 3 Instruct | OLMo 3 RL-Zero |
| :---- | :---- | :---- | :---- | :---- |
| starts from | Llama 3.1 base (8B, 70B, 405B) | OLMo 3 base (7B, 32B) after midtraining and long context | the OLMo 3 Think SFT model | the OLMo 3 base |
| stages | SFT → DPO → RLVR | SFT → DPO (Delta Learning) → RLVR (OlmoRL) | SFT → DPO → RL | RL only |
| SFT data | 939,344 prompts | 2,268,468 prompts (7B), long reasoning traces | 2,152,112 prompts, incl. function calling | none |
| SFT settings | LR 5e-6 (8B), linear, warmup 0.03, 2 epochs, batch 128, max 4,096 tokens, sum loss | 2 epochs, LR sweep, then a weighted merge of two checkpoints; packing at 32,768; batch 1M tokens | same setup, 3.4B tokens | none |
| preference data | 354,192 prompts, length-normalized DPO | pairs with a clear quality gap (Qwen3 32B vs 0.6B) | 259,922 prompts, multi-turn pairs, control of answer length | none |
| RL data | 29,946 prompts: GSM8K, MATH, instruction constraints; PPO | verifiable (math, code, IF) and non-verifiable (chat) prompts | Dolci Instruct RL | 13.3K math prompts + code, IF, chat; 16K-token responses |
| eval note | a dev suite plus a separate unseen suite | `<think>...</think>` stripped before scoring |  | pass@k with 32K-token responses, temperature 1.0 |

### 2.8 The kinds of SFT :

| choice | options | what I picked for v1 |
| :---- | :---- | :---- |
| which weights | full fine-tuning ; LoRA (small low-rank matrices, base frozen) ; QLoRA (LoRA on a 4-bit base) | full fine-tuning: 1.5B fits on one H200 and needs no new model code |
| where answers come from | people ; a stronger model (distillation) ; the model itself (Self-Instruct); rejection sampling | existing open datasets |
| what to teach | instruction following , chat, domain knowledge , reasoning , tool calls , safety | instruct first (Section 4) |
| loss | plain masked cross-entropy; chunked cross-entropy (less memory) ; DFT ; proximal SFT | plain masked cross-entropy |

## 3. How SFT runs in BluTrain :

### 3.1 The whole flow :

Steps 1 to 5 run once , offline , in Python(can be done in cpp too , python is not mandatory) ,  and steps 6 to 10 run every training step in the C++/CUDA trainer :

```text
OFFLINE (Python, once)
 1. load dataset     {"messages": [{"role":"user","content":...}, {"role":"assistant",...}]}
 2. chat template    messages -> one string with role markers                    (Section 3.2)
 3. tokenizer        string -> token ids (GPT-2 BPE, 0..50256)                     (Section 3.3, Section 3.4)
 4. labels           target = next token; 65535 on prompt and padding              (Section 3.5)
 5. pad + write      pad every row to T with 50256 -> inputs.bin, labels.bin       (Section 3.8, Section 3.10)

TRAINING (C++/CUDA, every step)
 6. dataloader       read rows of both files, count valid labels, check ids        (Section 3.9)
 7. forward          all positions, prompt included -> logits [rows, 50304]
 8. loss             masked cross-entropy, divided by the GLOBAL valid-token count (Section 3.6, Section 3.7)
 9. backward         masked rows give exactly zero gradient
10. optimizer        NaN/Inf check first, then update, checkpoint                  (Section 3.11)
```

### 3.2 Messages and the chat template :

SFT datasets store each example as a list of messages and each message has a role (system , user or assistant) and content :

```text
{"messages": [
   {"role": "system",    "content": "You are a helpful assistant."},
   {"role": "user",      "content": "What is the capital of France?"},
   {"role": "assistant", "content": "Paris."}
]}
```

The model reads one flat sequence , not a list and the chat template is the fixed rule that turns the list into one string , with a marker before each turn and the same template has to be used in training and later when the model is used , or the model sees a format it wasn't trained on ,,, ,

Tülu 3's template (exact , from their Appendix B.3 Figure 27 and their tokenizer_config.json) has two rules for the end of an answer :- an earlier answer ends with the end token and then a newline , and the last answer ends with the end token only :

```text
<|system|>\n{system}\n<|user|>\n{user1}\n<|assistant|>\n{answer1}{eos_token}\n<|user|>\n{user2}\n<|assistant|>\n{answer2}{eos_token}
```

Tülu 2 put the newline after every answer , the last one too (`{answer}{eos_token}\n`) , and Tülu 3 removed only that last newline (open-instruct PR 376) , and the newline that stays between turns is in the text but is never a training target (their masking code gives it label -100) ,,, ,

My template (tulu-plain-v1 , in sft/prepare_sft_data.py) is Tülu 3's , except that no answer has a newline after its end token , so the next turn starts right after the end token :

```text
<|system|>\n{system}\n<|user|>\n{user1}\n<|assistant|>\n{answer1}<|endoftext|><|user|>\n{user2}\n<|assistant|>\n{answer2}<|endoftext|>
```

for a one-turn chat , mine and Tülu 3's give exactly the same text , and in both , the targets are the same : the answer and the end token after it ,,, ,

at generation time the prompt ends with `<|assistant|>\n` and the model writes the rest , and it stops when it writes the end token 50256 ,,, ,

ChatML (Qwen, and the tokens tested in OLMo 3) uses `<|im_start|>` and `<|im_end|>` :-

```text
<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\nParis.<|im_end|>
```

The template changes results a little and Tülu 3 compared 5 templates on the same SFT data (Table 13 , Section 4.3.1) : 53.0 (the Tülu template with "the newlines at the end of assistant messages" replaced "with an eos token") , 52.9 (Zephyr) , 52.8 (Tülu 3 : the Tülu 2 template with the last newline removed) , 52.6 (Tülu 2) , 51.6 (Llama 3) , a spread of 1.4 points , and Tülu 3 did not use the best one , "to avoid generation inconsistency with later steps in our post-training pipeline" (DPO and RL) , so using one template consistently matters more than which one ,,, ,

Which one I use , and why : mine is the 53.0 one , if I read their sentence literally (no newline after any answer's end token) , their exact 53.0 template was never published (not in the paper , not in open-instruct's code) , so this is my reading , not confirmed , and I checked my stage-0 data : after an earlier answer's end token the next tokens are always `<` `|` `user` (10,845 times in train , 1,055 in val) , and "end token then `\n`" appears 0 times , I keep it , becoz the targets are the same as Tülu 3's , and nothing measured says Tülu 3's is better (52.8 vs 53.0) ,,, ,

The same template in every stage : SFT , DPO , RL , evaluation and inference must all use exactly this template , that means the same markers , the same newlines (no `\n` after an answer's end token) , the same end token (50256 , and generation stops on it) , the same generation prompt (`<|assistant|>\n`) , and the same way of tokenizing (piece by piece , Section 3.3) , the simplest way is that every stage calls the same function as the preprocessing script , and Tülu 3 did the same : its SFT and DPO scripts read the template from one shared table (open-instruct PR 350) , I must never reuse Tülu 3's Hugging Face template file as it is , becoz it adds the `\n` between turns that my data doesn't have , and only if I later use an outside tool (Hugging Face evaluation , vLLM , TRL) , I'll write my own `chat_template` file for this format and check token by token that it gives the same tokens as the preprocessing script ,,, ,

Tülu 3 does NOT add new special tokens for roles. Its tokenizer_config.json on Hugging Face (allenai/Llama-3.1-Tulu-3-8B-SFT) shows added_tokens_decoder has only Llama 3's own tokens (128000 `<|begin_of_text|>`, 128001 `<|end_of_text|>`, 128004 `<|finetune_right_pad_id|>`, 128006 `<|start_header_id|>`, 128007 `<|end_header_id|>`, 128008 `<|eom_id|>`, 128009 `<|eot_id|>`, 128010 `<|python_tag|>`, and the `<|reserved_special_token_N|>` ids) plus 128256 `<pad>`, which Tülu added. Strings like `<|user|>`, `<|assistant|>`, `<|system|>` are NOT registered special tokens in Tülu 3. They are tokenized as plain text (5 ordinary Llama 3 tokens each, for example `<` `|` `user` `|` `>\n` ; with our GPT-2 tokenizer they are 6, 7 and 6 tokens). That is Option A in Section 3.4. So option A is not just my fallback for now, it is also what Tülu 3 actually did.

Option B (dedicated special tokens) is what Qwen3 and OLMo 3 use, where `<|im_start|>` and `<|im_end|>` are registered special tokens (still to be checked in their tokenizer files). For our GPT-2 base, Option B means using spare ids 50257..50303.

### 3.3 The tokenizer's job :

The chat template writes the role markers into the text , and the end token goes after each answer , with option A (Section 3.4) the role markers are plain text , so the tokenizer splits each one into several ordinary GPT-2 tokens , and only the end token is one special id (50256 , which my script adds as an id , never as text) , and then the preprocessing script checks that every answer ends with the end token and the tokenizer's jobs for SFT :

1. Split text into tokens and map them to id’s and GPT-2 uses byte-level BPE :- text becomes bytes , and frequent byte sequences are merged into tokens , so , any text can be encoded and there's no "unknown" token and a space usually attaches to the next word (" Paris" is one token) ,,, ,
2. Recognize special tokens as exactly one id each , never split and GPT-2 has only one :  `<|endoftext|>` = 50256 ,,, ,
3. Handle special-token text inside user messages and if a user message literally contains "`<|endoftext|>`" ,  it has to be encoded as plain text , not as the real end token and the “tiktoken” library raises an error on this by default , so , the preprocessing has to choose explicitly ,,, ,
4. Keep track of which tokens came from assistant content , becoz , that's what builds the mask and the simplest way is to tokenize piece by piece (marker, then content, then the next marker) and record for each piece whether it's an answer ,,, ,
5. Be the same tokenizer as pretraining and the base was trained on GPT-2 id’s  and with any other tokenizer , every id means a different piece of text and the model's knowledge is useless ,,, ,

BluTrain has no tokenizer or template code in C++ , that's fine , becoz the pretraining data is also tokenized offline and stored as id’s , and SFT will do the same in the preprocessing script (Section 3.10) ,

One thing to watch : BPE can merge across a boundary , so the same text can give different tokens when it's tokenized in one go instead of piece by piece , I checked it : `\n<|user|>\n` and `\nParis` give the same tokens both ways , but a user message that ends with a newline does not , `Write a poem.\n` followed by the template's `\n` gives `\n` `\n` (2 tokens , 198 198) piece by piece , and `\n\n` (1 token , 628) in one go , my script tokenizes piece by piece (message by message , and within each message , marker then text) , becoz that's also how it knows which tokens belong to the answer (target = next token) and which to the prompt (target = 65535) , so every later step (evaluation , DPO , RL , chat) has to tokenize piece by piece in exactly the same way , and the simplest way is to call the same function ,,, ,

### 3.4 Special tokens:

Special tokens mark structure, not text. The model has to learn what they mean during SFT, because it never saw them in pretraining (or saw them only as end-of-document).

| token | meaning | used by |
| :---- | :---- | :---- |
| `<\|endoftext\|>` | end of a document; in SFT, end of the answer, so the model learns to stop | GPT-2 id 50256 , my end token after each answer ; Tülu 3 uses Llama 3's `<\|end_of_text\|>` (id 128001) for the same job |
| `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` | start of a turn by that role | Tülu 3 and me , as plain text (not registered tokens) |
| `<\|im_start\|>`, `<\|im_end\|>` | start and end of a turn (ChatML) | Qwen3; tested in OLMo 3 midtraining |
| `<think>`, `</think>` | start and end of the reasoning in a think model | DeepSeek-R1, Qwen3, OLMo 3 |
| `<answer>`, `</answer>` | the final answer after the reasoning | DeepSeek-R1-Zero's template |
| pad | filler after the answer | I use 50256 as the input pad (Section 3.8) |
| tool-call markers | start and end of a function call and its result | OLMo 3 Instruct trains function calling |

LIMA added its own end-of-turn token instead of reusing EOS ,,, , and it "plays the same role as EOS of halting generation , but avoids conflation with any other meaning that the pretrained model may have imbued into the preexisting EOS token" and in the pretraining data , 50256 means "end of a web document" in pretraining , so this is worth thinking about ,,, ,

The GPT-2 tokenizer has only one special token, so I have to choose how to write the role markers and the model does have 47-unused vocab rows (ids 50257 to 50303) because the vocab is padded from 50,257 to 50,304 ,,, ,

| option | what it means | good | bad |
| :---- | :---- | :---- | :---- |
| A. markers as plain text | write `<\|user\|>` etc. as ordinary text, which becomes several existing tokens; use the real 50256 at the end of each answer | no change to the model, tokenizer or kernels. OLMo 3 found a chat layout written as plain text didn't cause the collapse that special tokens caused in midtraining (GSM8K 46.02 vs 0) | a few extra tokens per turn; user text could contain the same characters |
| B. new special tokens in the 47 spare rows | give `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<think>`, `</think>` one id each in 50257-50303 and register them in the tokenizer | one id per marker, and user text can't fake a marker. No change to the model's shape or kernels, since every id stays below 50,304 | those rows were never trained as inputs, so SFT has to learn them; needs a custom tokenizer definition and a starting value for those rows (to test) |
| C. grow the vocab | add rows past 50,304 | unlimited new tokens | changes the embedding and output-layer shapes , the checkpoint format and GEMM shapes , not needed ,,, ,  |

What I am planning to do : option A for the first smoke test , becoz , it needs no changes and keeps the pipeline test separate from tokenizer work , then option-B for the real-instruct-model , as in Qwen3 and OLMo 3 (still to be checked) , but Tülu 3 does NOT use dedicated role tokens : its `<|user|>` , `<|assistant|>` , `<|system|>` are plain text , same as option A (checked in its tokenizer_config.json) , and here option B costs no model change , in both options , the end of every answer is the real end token 50256 , and it's a target ,,, ,

### 3.5 Labels , the shift by one , and 65535 :

In pretraining today , the loader reads one long stream of tokens , and the input is tokens[i .. i+T) and the target is tokens[i+1 .. i+T+1) , the same stream shifted by one ,,, ,

For SFT ,  the preprocessing script builds the target row itself and at each position t , the target is token t+1 if token t+1 is part of the answer (or the end token right after it) , otherwise the target is the ignore marker 65535 and the example in 1.4 shows one row ,,, ,

Why 65535 :

* The targets are stored as uint16 (0 to 65535) and that's how the dataloader stores all token ids (UInt16ShardView) ,,, ,
* Real token ids are 0 to 50303 , so nothing from 50304 to 65535 can be a real target and 65535 is the top of the range and can't be confused with a token ,,, ,
* PyTorch and TRL use -100 , which works there because their targets are int64 and in uint16 , -100 is stored as 65436 , which never equals -100 , so masking would silently do nothing ,,, ,
* The default ignore index in the kernel API is -1 and a uint16 target cast to int64 is always 0 or more , so , -1 never matches and if someone turns SFT on but forgets to set the index , nothing gets masked and the run falls back to pretraining behaviour instead of masking real tokens ,,, ,

### 3.6 What the loss kernel does with it:

The loss runs in the sparse cross-entropy CUDA kernels , one GPU block per row (per position) and this is the change , in all 5 live kernels :

```text
forward, one block per row r:
    if r >= rows: return
    if target[r] == ignore_index:          # 65535 in SFT; -1 in pretraining never matches
        loss[r] = 0; saved_max[r] = 0
        saved_sum[r] = 1.0                 # 1.0, not 0: the backward computes 1/saved_sum
        return                             # skips the whole 50,304-wide softmax
    ... normal softmax cross-entropy ...

backward, one block per row r:
    if target[r] == ignore_index: grad[r, all 50,304 columns] = 0; return
    grad[r, v] = scale * ( softmax[r, v] - (v == target[r] ? 1 : 0) )
    scale = grad_output / denominator      # see Section 3.7
```

* Masking makes the loss faster , not slower , becoz a masked row returns before the softmax over 50,304 entries ,,, ,
* saved_sum = 1.0 is required and the backward computes 1/saved_sum ; a “0” there gives infinity , and infinity*0 gives NaN , which spreads through the whole gradient ,,, ,
* The early return also stops an out-of-bounds read and the normal path reads logits[row][target];  with target 65535 on a row of 50,304 values ,  that's 15,231 elements past the end of the row and the check sits before that read ,,, ,
* It's the same design as PyTorch ATen (nll_loss compares with ignore_index and skips the row) and Liger-Kernel (default -100) , Megatron instead multiplies the loss by a 0/1 mask after computing every row, so masked rows still pay for the full softmax ,,, ,
* The CPU path does the same , so CPU and GPU agree in the parity tests ,,, ,

### 3.7 The denominator :

The loss of a step is a sum over counted tokens divided by some number , in pretraining every position counts , so dividing by the batch size is right , but in SFT , the number of counted tokens changes from example to example (short answers, long answers) , and where you divide changes how much each token counts ,,, ,

Example : one optimizer step has two micro-batches and A has 50 answer tokens with average loss 2.0. B has 500 answer tokens with average loss 1.0 ,,, ,

| method | result | weight of one token in A | weight of one token in B |
| :---- | :---- | :---- | :---- |
| right: sum everything, divide once by all 550 tokens | (50×2.0 + 500×1.0) / 550 = 1.091 | 1/550 | 1/550 |
| wrong: divide each micro-batch by its own count, then average | (2.0 + 1.0) / 2 = 1.500 | 1/100 | 1/1000 |

The wrong way gives tokens in short answers 10 times the weight of tokens in long answers , and nothing warns you : the loss is finite , it goes down , no NaN and with 2 ranks × 32 accumulation steps there are 64 micro-batches per step , and in my simulation the weights differed by 13.5x ,,, ,

What others do :

* Tülu 3 hit exactly this (§4.3.2, "Batch Aggregation") : with gradient accumulation " we weight each example equally ", without it " we weight each token equally "---> They switched to a sum loss , which "effectively weights all tokens equally (which found led to generally better performance)", and re-tuned the learning rate to 5e-6 ,,, ,
* torchtitan uses a sum loss , counts the valid tokens of the whole step in the data collator , all-reduces that one number across GPUs , and divides every micro-batch's loss by it ,,, ,
* Megatron returns the loss sum and the token count separately so they can be reduced before one division and its default mode , though , divides each micro-batch by its own count , the wrong way above ,,, ,
* PyTorch ATen returns the count (total_weight) as an output , and has a switch to return the plain sum ,,, ,

I went with torchtitan's approach, the count is known before the forward runs , because the labels are already on disk , so , nothing has to be rescaled afterwards and the count never goes back to the CPU , in the loss op , it's an optional parameter : sparse_cross_entropy_loss(logits, targets, ignore_index = -1, denom = nullptr) and with denom = nullptr it divides by batch size as before , so pretraining doesn't change at all ,,, ,

### 3.8 Padding, packing and attention:

The GPU processes a batch where every row has the same length T , but SFT examples have different lengths and two ways to fill the rows :

**v1: one example per row, padded on the right.**

```text
input : [ prompt ][ answer ][ 50256 ][ 50256 ][ 50256 ] ...      ← correct (these 50256 are padding)

target: [ 65535 x (len(prompt)-1) ][ answer ][ 50256 ][ 65535 ][ 65535 ] ...   ← correct (this 50256 is END)
```

![image1](post_training_research_sft_images/image1.png)

![image2](post_training_research_sft_images/image2.png)

In SFT, the prompt tokens' weights are updated only through the answer tokens' gradients, so they learn "help produce the right answer", never "predict the next prompt token" :

![image3](post_training_research_sft_images/image3.png)

![image4](post_training_research_sft_images/image4.png)

AND REGARDING PADDING TOKENS ID’S :

![image5](post_training_research_sft_images/image5.png)

Attention needs no change for this and I checked it in the kernel code :

* The trainer calls one attention op , GQA scaled-dot-product attention with is_causal=true (forward v62 and backward Vj1d on H200) , in bluscript_zero_ckpt.cpp:639-641 ,,, ,
* Rows in a batch never read each other ; each row has its own Q, K, V offsets (GQA_fwd_sm90.cu:389,397,434) ,,, ,
* Position i only reads positions 0 to i , so with padding at the end, a real token never reads a pad ,
* Pad positions have label 65535, so their loss and gradient are exactly zero ,,, ,
* RoPE uses one position offset (0) for the whole batch , so every example starts at position 0 (bluscript_zero_ckpt.cpp:614-616) ,,, ,

Every framework I checked (TRL, Hugging Face, NeMo, LongAlign, Llama 3) adds a document mask only when packing , and TRL pads on the right for training ,,, ,

The cost of v1 is wasted compute and Average SFT examples are roughly 90 to 1,800 tokens (Alpaca ~92, FLAN ~387, UltraChat ~568, the Tülu-v2 mix ~1,759) and padding to 4096 wastes about 57% to 98% of the matmul work , and more of the attention work ,,, ,

**v1.5: length bucketing :** Group examples of similar length and pad each micro-batch only to its longest example , rounded up to 256 and LongAlign reports about 2x and it changes the matmul shapes step to step , so i would need to check the GEMM dispatch for them ,,, ,

**v2: packing :** put several examples back to back in one row , so almost nothing is wasted and reported speedups : about 2x (Hugging Face, FLAN) , 1.9x (IBM) , 3.9-6.7x (a 70B study) , up to 6x (NeMo) and OLMo 3 trains SFT with packing at 32,768 and Llama 2 also concatenated prompts and answers to fill 4096-token rows , with a special token between segments ,,, ,

The catch is that with packing and plain causal attention , the second example in a row can read the first one and the loss still goes down , so nothing reports it and the fix is a block-diagonal (document) mask : position i can read position j only if j is in the same example and j ≤ i and positions should also restart at 0 for each example ,,, ,

* In packed pretraining , the mask improved perplexity from 9.17 to 8.41 and in-context learning accuracy from 63.5 to 70.5 on a 1.3B model (Zhao et al., ACL 2024) ,,, ,
* Llama 3 : the mask "had limited impact during in standard pre-training , but find it to be important in continued pre-training on very long sequences" ,,, ,
* I didn't find a clean ablation for SFT ,,, ,

In BluTrain , v2 is a kernel change in the attention forward (v62) and backward (Vj1d  boundary tiles need the mask , plus a guard for fully masked tiles (which give NaN today) and the dataloader would also have to give each token its example start ,,, ,

**Things I found in the code that v1 must handle :**

1. The input pad id has to be 50256 , and 65535 is only a label marker and as inputs , ids 50304 to 65534 make the embedding kernel skip the row and leave uninitialized memory (EmbeddingKernels.cu:43-50, 96-107) ; 65535 only works by accident and so the loader has to reject any input id of 50304 or more ,,, ,
2. The end token must be a target at the end of every answer , or the model never learns to stop and an example cut short loses its end token , so long examples get dropped , or have their prompt side cut ,,, ,
3. Never pad on the left in training and there's no pad mask , and it creates rows where every key is masked ,,, ,
4. No micro-batch with zero valid tokens , because 0/0 = NaN with a per-micro-batch divide and drop empty examples in preprocessing ,,, ,
5. Shape limits : the v62 forward needs T divisible by 256 (backward by 64) , and T ≤ 4096 (the RoPE cache length) ,,, ,

### 3.9 The dataloader :

Current pretraining dataloader -> dist/dataloader_dist.h reads one flat uint16 stream, copies tokens[pos .. pos+BT) as input and tokens[pos+1 .. pos+BT+1) as target , and moves forward and it doesn't know where one document ends and the next starts , and the only file check is that the size is divisible by 2 ,,, ,

For SFT it needs to :

1. Read two files of the same shape , inputs and labels , row by row and the shift is already done offline , so it steps by exactly T with no +1 ,,, ,
2. Shuffle examples each epoch and give each GPU different rows ,,, ,
3. Count the valid labels (not 65535) for each micro-batch and add them up for the whole step , so the trainer can all-reduce one number before the first forward (Section 3.7) and the preprocessing can store each row's count so nothing needs scanning ,,, ,
4. Check id’s once per file : every input id below 50,304 (hard error, see Section 3.8) and every label below 50,304 or equal to 65535 ,,, ,
5. Stop after the set number of epochs (1 or 2) instead of streaming forever ,,, ,
6. Later : bucketing by length (v1.5) , and for packing (v2) , each token's example start and position ,,, ,

### 3.10 The dataloader design :

A Python script , run once per dataset :

1. Load the dataset from Hugging Face and record its name , revision and licence ,,, ,
2. Filter : drop empty answers , non-English examples (GPT-2's tokenizer is built for English) , and for think-data , answers with `<think>` but no `</think>` (OLMo 3 does the same) ,,, ,
3. Apply the template (Section 3.2) and tokenize piece by piece (Section 3.3) , building the answer mask,,, ,
4. Build the labels : next token on answer positions and the end token , 65535 everywhere else ,,, ,
5. Drop examples longer than T , or cut the prompt side , so the end token is never lost ,,, ,
6. Pad inputs with 50256 and labels with 65535 up to T ,,, ,
7. Write inputs.bin and labels.bin (uint16, N × T) and a metadata file : N , T , tokenizer , template version , pad-id , ignore id , per-row valid counts , length-histogram , and how many examples were dropped and why ,,, ,
8. Print checks : a few random rows decoded back to text with the counted part marked , the share of valid tokens , the largest id , and that every row has the end token as a target ,,, ,

Size: 50,000 examples × 4,096 tokens × 2 bytes × 2 files ≈ 0.8 GB ,,, ,

### 3.11 The preprocessing script :

| setting | plan for v1 | why |
| :---- | :---- | :---- |
| starting weights | load the base checkpoint's weights only; new optimizer state, step 0, new LR schedule | the checkpoint is a 4-GPU Muon DCP, so it needs a weights-only loader |
| ignore index | Section 3.5 | 3.5 |
| denominator | global valid-token count of the step, as a device tensor; remove the extra 1/accumulation and cross-GPU averaging | Section 3.7 |
| learning rate | sweep, e.g. 5e-6, 1e-5, 2e-5 | Tülu 3: 5e-6 at 8B (sum loss) and 2e-6 at 70B; TRL's default is 2e-5 |
| schedule | linear decay, warmup ratio 0.03 | Tülu 3, Table 11 |
| epochs | 2 | Tülu 3, OLMo 3 and Llama 2 all use 2 (LIMA used 15 on only 1,000 examples) |
| batch | about 128 examples per optimizer step, through accumulation | Tülu 3: 128; Llama 2 : 64 |
| sequence length | 4096 (the current limit) | Tülu 3 SFT also used a max of 4,096 |
| precision | bf16 compute, fp32 master weights, bf16 logits (BLU_BF16_LOGITS=1) | tested on 24 Sept (7.5) |
| optimizer | open : AdamW or Muon | the papers use AdamW (Llama 2, Zephyr, LIMA); the base used Muon; a small test will decide |
| NaN/Inf | the check stops the run before a bad update, exit code 4 | fixed 25-26 Sept (7.6) |
| logging | loss on valid tokens, valid-token count, grad norm, LR, tokens/s, memory | Section 6.5 |

The training-script changes will be small and for a sense of scale , Tülu-3 trained its 8B-SFT model on 32 GPUs for 6 hours ,,, ,

## 4. Instruct model vs think model :

### 4.1 How the data differs :

An instruct model answers directly but a think model first writes its reasoning between `<think>` and `</think>` , then the answer and both use the same SFT loss and what changes is only the data ,,, ,

```text
INSTRUCT (assistant content):
   The capital of France is Paris.<|endoftext|>

THINK (assistant content):
   <think>The question asks for the capital of France. France's capital city is Paris.
   I should answer in one sentence.</think>
   The capital of France is Paris.<|endoftext|>
```

|  | instruct | think |
| :---- | :---- | :---- |
| what the answer has | just the answer | reasoning + answer, and both are targets, because the model has to learn to write the reasoning |
| length | short: OLMo 3 7B Instruct SFT is 3.4B tokens for 2.15M prompts | long: OLMo 3 generated traces up to 32K tokens; 7B Think SFT is 45.4B tokens for 2.27M prompts |
| sequence length | Tülu 3: 4,096 | OLMo 3: 32,768 with packing |
| where answers come from | people or strong models (Tülu 3 used GPT-4o for all its persona data; OLMo 3 replaced old answers with GPT-4.1 ones) | reasoning models: OLMo 3 used QwQ-32B for math, code and instruction following, and DeepSeek R1 for chat, safety and science |
| quality checks | filtering, decontamination | also correctness: code run against test cases (up to 16 samples per prompt, filtered), instruction constraints checked, incomplete traces dropped |
| scoring | the whole answer | the `<think>...</think>` part is stripped before scoring |

Other things worth knowing:

* Qwen3 trains one model for both modes --> A /think or /no_think flag in the user or system message picks the mode , and for no-think examples "we retain an empty thinking block in the assistant's response" , so the format stays the same ,,, ,
* OLMo 3 trains its Instruct-SFT starting from the Think-SFT model and on their instruct benchmarks the average went from 44.5 to 47.8, and MATH from 60.3 to 65.9 (Table 29) and When they reuse think data for the instruct model , they "remove the reasoning traces and special tokens" ,,, ,
* DeepSeek-R1-Zero's prompt tells the model to put the reasoning in `<think>` ... `</think>` and the answer in `<answer>` ... `</answer>`  ,,, ,

### 4.2 What each part of BluTrain has to do for a think model :

| part | instruct (v1) | think |
| :---- | :---- | :---- |
| template | role markers + end token | add `<think>`/`</think>` inside the assistant turn; for a hybrid model, an empty `<think></think>` in no-think examples plus a mode flag |
| tokenizer | option A or B (Section 3.4) | `<think>` and `</think>` should be single ids (option B), so evals can find and strip them reliably |
| preprocessing | mask the prompt; answer + end token are targets | the trace is also a target; drop traces with `<think>` but no `</think>`; keep only verified answers where possible; enforce the length limit |
| sequence length | 4096 fits most instruct examples | traces reach 32K, but the trainer is limited to T ≤ 4096 (RoPE cache and the v62 forward). Either keep only short traces, or extend the context first |
| dataloader | one example per row | rows are mostly targets, and the valid count per row varies much more, so bucketing or packing matter more |
| loss kernel | masked CE | no change, but logits grow with T: at T = 32,768 one sequence is 32,768 × 50,304 logits = 3.3 GB in bf16, so a chunked cross-entropy (like Liger's) becomes important |
| denominator | global count | matters even more: a 20,000-token trace and a 300-token answer in the same step must be weighted per token |
| attention | no change | cost grows with T²; the attention kernels and RoPE cache are only validated up to 4096 |
| evaluation | generate a short answer | generate long outputs (OLMo 3 RL-Zero evaluates with 32K-token responses) and strip the think part before scoring |

### 4.3 What I am thinking to do :

-> the best order is : think-SFT first and then instruct-SFT on top of it , since OLMo 3 trained its instruct-model starting from the think-SFT checkpoint and got +3.3 on average (Table 29) , but the trainer can't do think-SFT yet , becoz think-model traces go up to higher T (>32K) tokens while it is limited to T ≤ 4096 (RoPE cache and attention kernels) in current production blutrain (higher context lengths are under experimentations i guess) , and at those lengths it also needs a chunked cross-entropy ,,, , and so for now , I will start with instruct-SFT at 4096 , which needs no attention or context changes , tests the whole pipeline end to end , and gives a first baseline ,,, , and once long context and chunked cross-entropy are in , i will do think-SFT first and then redo instruct-SFT starting from the think checkpoint (the OLMo 3 order) , and compare it with the first instruct-model run to see if the +3.3 gain also shows up for us ,,, ,

## 5. SFT data and the data mix:

### 5.1 Where SFT data comes from , and what makes it good :

| source | how it's made | examples |
| :---- | :---- | :---- |
| people | annotators or volunteers write the answers | InstructGPT's labelers; Llama 2's 27,540 vendor annotations; OpenAssistant (Tülu 3 uses 7,132); Dolly 15k |
| a stronger model (distillation) | a strong model answers collected or generated prompts | Tülu 3's persona data ("We use GPT-4o-2024-08-06 for all our persona-driven data synthesis"); OLMo 3's traces from QwQ-32B and DeepSeek R1; UltraChat in Zephyr |
| NLP datasets rewritten as instructions | QA, classification and other tasks turned into instruction + answer | FLAN ; Tülu 3 uses 89,982 FLAN v2 examples |
| the model itself | the model writes instructions and answers, which get filtered | Self-Instruct |
| rejection sampling | generate several answers and keep the ones that pass a check | Llama 2 and 3 ; DeepSeek-R1 ; OLMo 3's code answers filtered by tests ,,, ,  |

What the papers agree on :

* Quality beats quantity , LIMA used 1,000 examples , Llama 2 found a few thousand of their own examples beat millions of third-party ones and from my week-1 notes : Magpie filtered 4M examples down to 300k , and TuluTalk beat SmolTalk and Tülu with 23% and 14% less data by filtering ,,, ,
* Decontaminate , or the scores mean nothing , as OLMo 3 removes any training example that matches at least 50% of a test question's 8-grams , following Tülu 3's procedure ,,, ,
* Check licences : OLMo 3 removed examples "with (1) non-commercial or unclear licenses" and I have to check every dataset card before company use ,,, ,
* Filter bad answers : OLMo 3 also drops incomplete reasoning , wrong answers (by running tests or checking constraints) , mentions of other model developers and date cutoffs , and heavy repetition ,,, ,
* Cover every skill you want , because removing a category removes that skill (Section 5.2) ,,, ,

### 5.2 The Tülu 3 SFT mix and its ablations:

The full Tülu-3-SFT mix is 939,344 prompts (Table 7) :

| skill | dataset | prompts |
| :---- | :---- | :---- |
| general chat | Tülu 3 Hardcoded (identity questions) | 240 |
|  | OpenAssistant | 7,132 |
|  | No Robots | 9,500 |
|  | WildChat (real user conversations) | 100,000 |
| knowledge | FLAN v2 | 89,982 |
|  | SciRIFF (science) | 10,000 |
|  | TableGPT (tables) | 5,000 |
| math | Tülu 3 Persona MATH | 149,960 |
|  | Tülu 3 Persona GSM | 49,980 |
|  | Tülu 3 Persona Algebra | 20,000 |
|  | OpenMathInstruct 2 | 50,000 |
|  | NuminaMath-TIR | 64,312 |
| coding | Tülu 3 Persona Python | 34,999 |
|  | Evol CodeAlpaca | 107,276 |
| safety | CoCoNot | 10,983 |
|  | WildJailbreak | 50,000 |
|  | WildGuardMix | 50,000 |
| multilingual | Aya | 100,000 |
| precise instruction following | Tülu 3 Persona IF | 29,980 |
| total |  | 939,344 |

The "persona" sets are synthetic : GPT-4o writes a new problem or instruction for each of many made-up user personas , which gives variety and Tülu 3 kept another 354,192 prompts for preference tuning ,,, ,

They trained the same 8B model on the full mix and with one part removed each time (Table 10) :

| mix | average | biggest drop |
| :---- | :---- | :---- |
| full mix | 60.1 |  |
| without WildChat | 58.9 | chat quality (AlpacaEval) |
| without safety data | 58.0 | safety 93.1 → 74.7 |
| without persona data | 58.6 | IFEval 72.8 → 53.6 |
| without math data | 58.2 | GSM8K 76.2 → 64.1, MATH 31.5 → 23.5 |

Two more results from the same paper :

* More data kept helping on average , and on 5%, 10%, 25%, 50%, 75% and 100% of the mix , "our full mix performs best overall" -> GSM8K gained the most from more data , while TruthfulQA "actually drops as the amount of data in the mix increases" (Figure 4) ,,, ,
* The mix matters a lot and Tülu 2's SFT mix gave 48.3 average on Llama 3 8B , and Tülu 3's gave 60.1 on the same base (Table 9) ,,, ,

### 5.3 The OLMo 3 SFT mixes :

Dolci Think SFT (Table 17) has 2,268,468 prompts for 7B and the biggest parts are OpenThoughts3+ math (752,997) , Python algorithms (466,677) , persona precise IF (223,123) , precise IF (135,792) , Nemotron code (113,777) , SYNTHETIC-2 verified (104,569) , OpenThoughts3+ STEM (99,269) , Aya (98,597) , OpenThoughts3+ code (88,900) , WildChat (83,054) , and safety (CoCoNot, WildGuardMix and WildJailbreak, 89,642 together) ,,, ,

Dolci Instruct SFT (Table 30) has 2,152,112 prompts , including WildChat (302,406) , verifiable reasoning (310,572) , tool use (227,579) , Python algorithms (186,345) , logic puzzles (159,882) , Persona MATH (149,958) and precise IF (136,833) ,,, ,

How they picked the mix : start from a small base mix of 100K examples , train separate models on the base mix + up to 100K examples of one category , and look at every benchmark ,,, ,

| Think SFT experiment (Table 18) | avg | what changed |
| :---- | :---- | :---- |
| base mix (100K OpenThoughts 3) | 39.2 | IFEval 35.7, AlpacaEval 19.0 |
| + Aya | 41.9 | AlpacaEval 30.2 |
| + WildChat and OpenAssistant | 44.2 | AlpacaEval 38.5, IFEval 48.8 |
| + persona precise IF | 45.9 | IFEval 70.4 |
| + safety | 40.9 | little change on these benchmarks |
| + SYNTHETIC-2 (verified reasoning) | 47.3 | MATH 82.0 (from 74.6) |
| base code swapped to Nemotron code | 34.5 | worse, HumanEval 22.6 |
| **Instruct SFT experiment (Table 28, on OLMo 2)** | **avg** | **what changed** |
| base mix | 29.0 | MATH 6.6, GSM8K 30.1, IFEval 61.7, AlpacaEval 5.8 |
| + FLAN | 30.3 | BBH 35.0 (from 29.5) |
| + instruction following | 30.7 | IFEval 74.1, but HumanEval 14.6 (from 23.2) |
| + math | 29.3 | MATH 14.2, GSM8K 39.7, but IFEval 54.0 |
| + code | 28.7 | HumanEval 26.8 |
| + safety | 27.0 | lower on these (safety is measured separately) |
| + science | 29.4 | MMLU 53.4 |
| + WildChat | 30.9 | AlpacaEval 19.2 |

Every category raises at least one benchmark and often lowers another , OLMo 3 : " each dataset is helpful on at least one evaluation , and so our final mix includes at least a portion of each dataset we tested ” , so a data mix is a trade-off , and it has to be judged per benchmark , not only the average ,,, ,

### 5.4 Datasets I can use :

| dataset (Hugging Face id) | size | what it is | for this work |
| :---- | :---- | :---- | :---- |
| allenai/tulu-3-sft-mixture | 939,344 | the Tülu 3 mix above | first choice: known ablations to compare against; take an English subset that fits 4096 tokens |
| HuggingFaceTB/smoltalk | ~1.1M | a mix built for small models | good fit for a 1.5B model |
| HuggingFaceH4/ultrachat_200k | ~200k | filtered UltraChat, used by Zephyr | general multi-turn chat |
| GAIR/lima | 1,000 | the LIMA set | small quality test |
| databricks/databricks-dolly-15k | 15k | human-written | human-written baseline |
| OpenAssistant/oasst1 |  | volunteer conversations | multi-turn, human |
| trl-lib/Capybara | small | TRL's demo set | pipeline smoke test |
| open-thoughts/OpenThoughts3-1.2M, allenai/Dolci-Think-SFT | large | reasoning traces | for the think model later |

Sizes with "~" are from my week-1 notes and i still need to check every size and licence on the dataset cards before using anything and examples need to be English and fit in 4,096 tokens after the template, and i will measure token lengths with the GPT-2 tokenizer before choosing , since code and math may take more GPT-2 tokens than the averages above ,,, ,

### 5.5 Data-mix experiments :

1. Base mix : about 50k general chat examples (the WildChat , OpenAssistant and No Robots parts of the Tülu 3 mix) ,,, ,
2. Add one category at a time , keeping the base fixed : math (Persona MATH/GSM, OpenMathInstruct 2), code (Persona Python, Evol CodeAlpaca), precise instruction following (Persona IF), knowledge (FLAN v2, SciRIFF), safety (CoCoNot, WildGuardMix, WildJailbreak) and this is how OLMo 3 did it ,,, ,
3. Remove one category at a time from a full mix  like Tülu 3's Table 10 ,,, ,
4. A data-size curve (5%, 10%, 25%, 50%, 100%) to see whether more data still helps at 1.5B , like Tülu 3's Figure 4 ,,, ,
5. Measure each category on its own benchmark (math : GSM8K and MATH ; code : HumanEval; instruction following : IFEval; chat : AlpacaEval ; knowledge and forgetting : MMLU and HellaSwag; safety: a refusal test) , and always the base model too ,,, ,
6. Run every arm at least twice , two identical fp32 runs of the trainer differed by 0.435 in loss on average over 20 short steps , so a single run per arm can't be trusted ,,, ,
7. Report per benchmark , not only the average , since categories trade off (Section 5.3) ,,, ,

## 6. Evaluation :

### 6.1 Two kinds of eval :

* Scoring (log-likelihood) : give the model a question and some fixed choices , and pick the choice it finds most likely and no text generation needed as the trainer already does this for HellaSwag (hellaswag_eval.h) ,,, ,
* Generation : the model writes a full answer , and a program or a judge scores it (exact answer , unit tests , instruction checks , or a strong model as judge) and this needs a generation loop , ideally with a KV cache , and BluTrain doesn't have that yet ,,, ,

* Tülu 3 maintained a separate unseen evaluation suite (including AGIEval, MMLU-Pro, GPQA, IFEval-OOD, and HREF) to test out-of-distribution generalization without contaminating hyperparameter and data decisions made on the dev suite.

A few rules : run the base model and the SFT model on the same tests (a gain in instruction following with a drop in knowledge is forgetting , not a win ); decontaminate the training data against every test I report ; and use exactly the same template at test time as in training (the same markers , no \n after an answer's end token , the prompt ends with `<|assistant|>\n` , stop at 50256 , tokenized piece by piece , Section 3.2) and Tülu 3 also kept a separate "unseen" suite (including their new IFEval-OOD and HREF) that was never used for decisions ,,, ,

### 6.2 Benchmarks :

Tülu 3's dev suite is MMLU , PopQA , TruthfulQA , HumanEval(+) , GSM8K , MATH , BBH , DROP , IFEval , AlpacaEval 2 and the safety suite ,,, ,

OLMo-3-Think also reports AIME , ZebraLogic , OMEGA , LiveCodeBench , MBPP and IFBench ,,, ,

| benchmark | what it measures | how it's scored | generation? | used by |
| :---- | :---- | :---- | :---- | :---- |
| held-out SFT loss | fit to unseen SFT examples | masked loss | no | everyone |
| HellaSwag | common-sense sentence completion | log-likelihood | no | BluTrain (already there) |
| MMLU | knowledge in 57 subjects | multiple choice | no (MC form) | Tülu 3, OLMo 3 |
| TruthfulQA | avoiding common false answers | multiple choice | no (MC form) | Tülu 3 |
| PopQA | rare facts | exact match | yes | Tülu 3 |
| BBH | hard multi-step reasoning | exact match | yes | Tülu 3, OLMo 3 |
| DROP | reading with arithmetic | exact match / F1 | yes | Tülu 3 |
| GSM8K | grade-school math word problems | exact final answer | yes | Tülu 3, OLMo 3 |
| MATH | competition math | exact final answer | yes | Tülu 3, OLMo 3 |
| HumanEval(+) | writing Python functions | unit tests (pass@1) | yes | Tülu 3, OLMo 3 |
| IFEval | following exact instructions ("write 3 paragraphs", "no commas") | a program checks 25 instruction types over ~500 prompts; no judge | yes | Tülu 3, OLMo 3 |
| AlpacaEval 2 | general chat quality | win rate vs a reference, judged by a strong model, length-controlled | yes + judge | Tülu 3, OLMo 3 |
| MT-Bench | multi-turn chat | judged by a strong model | yes + judge | Zephyr |
| Arena-Hard | hard user queries | judged by a strong model | yes + judge | DeepSeek-R1 |
| GPQA | graduate-level science | multiple choice | yes (CoT) | OLMo 3, DeepSeek-R1 |
| AIME | olympiad math | exact answer | yes (long) | OLMo 3, DeepSeek-R1 |
| LiveCodeBench | recent coding problems | unit tests | yes | OLMo 3, DeepSeek-R1 |
| unseen suite (MMLU-Pro, GPQA, AGIEval English, Deepmind Mathematics, BigCodeBench, IFEval-OOD, HREF) | the same skills on tasks designed separately from the dev suite, never used for decisions | exact match; unit tests (pass@10); instruction checks (pass@1); win rate (HREF) | yes | Tülu 3 |
| safety suite | refusing harmful requests without over-refusing safe ones | classifiers and rules (XSTest, HarmBench, Do-Anything-Now, JailbreakTrigger, WildJailbreakTest, WildGuardTest) | yes | Tülu 3 |

### 6.3 What I can run now , and what needs building :

| eval | now? | what's missing |
| :---- | :---- | :---- |
| held-out SFT loss | Yes  | a validation split |
| HellaSwag | yes | nothing |
| MMLU, TruthfulQA (multiple choice) | almost | a scoring adapter for their formats |
| GSM8K, MATH, BBH, IFEval, PopQA, DROP | no | a generation loop (greedy, stop at the end token 50256), the same template at inference, answer extraction; a KV cache for speed |
| HumanEval, LiveCodeBench | no | generation plus a sandbox to run code |
| AlpacaEval, MT-Bench, Arena-Hard | no | generation plus an external judge model (costs money, and the model's outputs leave the company) |

### 6.4 The benchmarks I'd pick, in order:

| when | benchmarks | why | needs |
| :---- | :---- | :---- | :---- |
| now | held-out SFT loss, HellaSwag, MMLU and TruthfulQA (MC) | no generation needed; they show if the model forgot knowledge, which is the main SFT risk | the SFT loader and a small scoring adapter |
| first generation work | IFEval, GSM8K | IFEval measures SFT's main goal with programs and no judge; GSM8K has exact answers; both are in the Tülu 3 and OLMo 3 suites | greedy generation that stops at the end token |
| later | HumanEval, BBH, AlpacaEval 2, the safety suite | code, reasoning, chat quality, refusals | a code sandbox, a judge model |
| think model | MATH, AIME, LiveCodeBench, GPQA | the reasoning benchmarks think models are judged on | long generation |

My guess (not measured) is that hard tests like MATH, GPQA and AIME will be near their floor for a 1.5B model trained on 8.4B tokens , so they won't separate the data mixes and  for reference , OLMo 3's instruct ablations on an OLMo 2 base scored MATH 6.6 to 14.2 , so i will pick tests that actually move at this size ,,, ,

What i mean here is that : MATH is a benchmark of competition-level math problems (high-school contest style) and that score is the % of problems solved exactly right and -

The OLMo 3 team (AllenAI) ran small test trainings to choose their instruct SFT data , so they trained an older base model , OLMo 2 , on different data mixes and scored each one (their Table 28 in olmo-3 paper) and with the basic mix , OLMo-2 solved only 6.6% of MATH and adding extra math data raised it to 14.2% and that was the best of all the mixes ,,, ,

 Why I wrote it :

OLMo 2 saw trillions of tokens in pretraining and whereas the model iam testing with (navin bro’s 1.5B model trained on 8.4B tokens) saw about 8.4 billion tokens of pretraining data , hundreds of times fewer and if even OLMo-2 only reaches 6.6–14.2% on MATH , our 1.5B model will very likely score near 0% on it , whatever SFT data we use , so a score that stays near 0 for every data mix can't tell us which mix is better and so for our model and my experimentations , MATH (and other hard ones like GPQA and AIME) is a useless benchmark for choosing data and so i should pick tests that actually move at our size as for ex : val loss on answers , IFEval-style format checks , whether answers end with the end token , HellaSwag , TruthfulQA kind of ,,, , will see ,,, , many experiments are yet to be done on these ablation tests too ,,, ,

### 6.5 Tests for v1 :

These have to fail if masking , padding or the denominator is wrong :

1. Pad-content invariance : changing the tokens after the end token (the padding) must not change the loss or any gradient, bit for bit (with the deterministic attention backward, USE_DETERMINISTIC_DQ=1) ,,, ,
2. Masked-loss reference: the kernel loss must equal a Python reference computed from the same logits over only the answer positions ,,, ,
3. Valid-token count: print it against rows × T on the first batch. It must match the preprocessing metadata. If it equals rows × T, masking never kicked in ,,, ,
4. Starting loss : the first SFT loss should be near the base model's loss on the same answers, not near ln(50304) = 10.83 ,,, ,
5. Pad invariance : one example padded to T = 256 and to T = 4096 must give the same loss and gradients ,,, ,
6. Row isolation : changing row 1 must leave row 0's logits bitwise the same ,,, ,
7. Denominator : 4 accumulation steps vs 1, and 2 GPUs vs 1, must give the same update (3.7) ,,, ,
8. Stopping : after training , the probability of the end token right after an answer should be much higher than in the base model ,,, ,

## 7. (the common doubts i too had)

**Is SFT a new loss?** No, it's the same next-token cross-entropy and the only change is that prompt and padding positions get a label that adds nothing to the loss or the gradient ,,, ,

**Is instruction tuning the same as SFT?** Basically yes and instruction tuning is SFT where the examples are instructions and their answers ,,, ,

**How much data is needed?** For format and style, a few thousand good examples (LIMA used 1,000, Llama 2 stopped at 27,540) and for many skills at once, open recipes use 1 to 2.3 million (Tülu 3 939,344, OLMo 3 Think 2,268,468) and experimentations are yet to be done on this thing ,

**Can SFT make the model better than its data?** Not really, it copies what it's shown and that's why RL comes after it ,,, ,

**Does masking slow training down?** No , a masked row returns early and skips the 50,304-wide softmax ,,, ,

**Why not just divide by batch size?** Because each micro-batch has a different number of answer tokens, and dividing per micro-batch overweights short answers (about 10x in the example in 3.7) and i divide once by the global count ,,, ,

**Why pad on the right?** The attention is causal , so with pads at the end , a real token never reads a pad and no mask is needed and left padding would need a pad mask BluTrain doesn't have ,,, ,

**Can I fine-tune Llama or Qwen instead of Navin's model?** Not today , their vocabs (Llama 3: 128,256; Qwen: ~151,936; TinyLlama: 32,000) don't match BluTrain's (50,304) , and BluTrain can't import Hugging Face weights (no safetensors or from_pretrained code) and that would be a separate model-import project ,,, ,

**Why full fine-tuning and not LoRA?** 1.5B fits on one H200 with full fine-tuning  and it needs no new modules in the engine , LoRA would need new adapter layers in the C++ model code ,,, ,

**What if an example is longer than 4096 tokens?** Drop it , or cut the prompt side and never cut the end of the answer, or the model loses its end-token target (for now iam doing this , as the padding technique is being used , and later packing with variable-length (varlen) attention will not change this : an example longer than the row still has to be dropped or cut , only a longer context length (long-context training) can take it , experiments yet to be done) ,,, ,

**Is DPO RL?** Not really as it doesn't generate during training and needs no reward model and it trains on fixed pairs of a better and a worse answer ,,, ,

**Why strip `<think>` before scoring?** The score should judge the final answer, not the working and OLMo 3 does this ,,, ,

**What is Dev-suite(Development-suite) and what is Unseen-suite in Evaluations ?**

### The core problem: "information leakage"

To understand why we need both a dev set and an unseen set, you have to understand one of the biggest traps in AI: information leakage.

When researchers build an AI, they have to make hundreds of decisions:

* "What learning rate should we use?"
* "Should we include 10% math data or 20% math data?"
* "How many epochs should we train, and which checkpoint should we keep?"

They answer these questions by running an experiment, testing the model, and picking the option that gets the highest score.

### The 3 sets (the big picture)

To keep the scores honest, researchers keep three separate sets of data. In LLM post-training (SFT), these are usually not three slices of one data pile. The dev set and the unseen set are separate lists of benchmarks.

#### 1. The training set (the engine)

**What it is:** the data the model learns from. For SFT, this is the chats (for us, 5,000 UltraChat chats). This is what goes through the GPU: the model computes gradients and updates its weights on this data.

**The flaw:** if you test the model on this data, it scores higher than it would on new data, often much higher, because it was fitted to these exact examples. The score tells you nothing about new data.

#### 2. The dev set / validation set (the steering wheel)

**What it is:** data the model never trains on. In SFT there are two kinds:

* a validation split: held-out chats from the same source, which give the validation loss (for us, 500 UltraChat chats);
* a dev suite: a list of benchmarks. Tülu 3's dev suite is MMLU, PopQA, TruthfulQA, HumanEval(+), GSM8K, MATH, BBH, DROP, IFEval, AlpacaEval 2 and safety tests.

**How it's used:** after training, the researchers test the model on the dev set. For example, they test two data mixes (Mix A and Mix B) by training two models:

* Model A scores 85% on the dev set.
* Model B scores 88% on the dev set.
* The researchers choose Mix B.

**Why it is flawed (the leakage):** the model never updated its weights on the dev set, but the researchers used the dev set to make a design choice.

* Every score has some random noise, so the option that wins is partly the one that got lucky on these particular questions.
* Because they keep making changes to push the dev score higher, the model indirectly becomes over-fitted to the dev set. It ends up tailored to the specific questions in it.
* So the dev score of the final model is higher than its real performance.

#### 3. The unseen set / test set (the final reality check)

**What it is:** another set of benchmarks, designed separately from the dev set. It tests different tasks, not just new questions of the same tasks. Tülu 3's unseen suite includes AGIEval, MMLU-Pro, GPQA, IFEval-OOD (new kinds of instructions that are not in IFEval) and HREF.

**The golden rule:** the researchers are not allowed to use the unseen set while building the model:

* not to tune hyperparameters;
* not to choose data mixtures;
* not to choose checkpoints. It stays unused while the model is being built.

**How it's used:** only when the model is finished and frozen do they test it on the unseen set, once.

**Why it's crucial:** nobody used this data to make any decision, so the unseen score is an unbiased measurement of how the finished model performs on those unseen benchmarks.

**Two limits:**

* it measures only the kinds of tasks in those benchmarks, not every real-world use;
* it stays unbiased only if it's used once. If you look at the unseen score, change something and test again, it starts leaking too, and it becomes a second dev set.

### Aren’t those dev-suite and unseen suite same ? why are they different?

If the dev set and the unseen set were the same thing, the researchers would report their final score on the exact data they used to tune and tweak the model for months. That is bad science: it artificially inflates the model's scores.

The dev set is the tool the engineers use to guide the model's development. Because it's used for choices, its score becomes biased (too high).

The unseen set is the tool that tells everyone how good the model actually is. Because it was never used for choices, its score is honest.

**Reading them together :**

* unseen close to dev → the decisions generalized;
* unseen much lower than dev → the decisions over-fitted the dev set.

**One more rule:** the training data must not contain questions from either the dev or the unseen benchmarks. Otherwise the model memorizes answers and both scores are fake. Tülu 3 removes any training example that contains 50% or more of a benchmark question's 8-token chunks (8-grams).

**Source: Tülu 3 (Ai2, 2024), evaluation section** : Lambert et al., Tülu 3: Pushing Frontiers in Open Language Model Post-Training, 2024. <https://arxiv.org/abs/2411.15124>

## 8. First SFT runs in BluTrain : what I built , the tests , the results , Muon vs AdamW , and the plan (6 and 7 Oct)

### 8.1 What I built

**Pre-processing script (sft/prepare_sft_data.py) :**

* loads UltraChat-200k (English only , MIT licence , the set Zephyr used) ,,, ,
* applies the template tulu-plain-v1 (Section 3.2) , which is option A of Section 3.4 : the markers are plain text (`<|user|>\n` is 6 GPT-2 tokens and `<|assistant|>\n` is 7) , so no new special tokens , and the end token 50256 after every answer is a target , and there's no `\n` after an answer's end token (Tülu 3's template minus that `\n` , the 53.0 one of Tülu 3's Table 13) , in stage-0 , 4,998 of the 5,000 training chats are multi-turn , with 15,845 answers , and after every earlier answer's end token the next turn's `<|user|>` starts directly ,,, ,
* puts 65535 on the prompt and the padding labels , pads the inputs with 50256 up to T = 4096 , and drops chats longer than 4096 instead of cutting them ,,, ,
* writes inputs.bin , labels.bin , the trained-token count of every row (valid_counts.bin) and meta.json ,,, ,
* Stage-0 data : 5,000 random training chats and 500 validation chats , with 4.77M trained tokens , which is 23.3% of all positions , and the average chat is 1,238 tokens , so about 70% of every 4096-token row is padding ,,, ,

**SFT dataloader (sft/SftDataLoader.h) :**

* the target comes from labels.bin instead of the input shifted by one ,,, ,
* each GPU gets different rows ,,, ,
* it gives the trainer the trained-token count of the next micro-batches before the forward ,,, ,
* it checks every id when it opens the files ,,, ,
* my loader test passed 5/5 ,,, ,

**Weights-only load (BLU_INIT_FROM) :** it loads only Navin anna's step-8000 weights , with a new optimizer and a new LR schedule , and 262/262 weights match the checkpoint ,,, ,

**SFT mode in the trainer (BLU_SFT_DATA) :**

* the loss is divided once by the trained tokens of the whole step (all micro-batches and all GPUs) , counted before the forward and added up across GPUs (Section 3.7) ,,, ,
* the backward starts from W (the number of GPUs) instead of 1/accumulation steps , becoz the gradients are averaged across GPUs afterwards , so nothing is divided twice ,,, ,
* the logged train loss is the plain sum , and validation runs over the whole val set every 10 steps ,,, ,
* settings : 64 chats per step , 2 epochs , LR 2e-5 → 2e-6 , 3% warmup , then cosine ,,, ,
* without BLU_SFT_DATA , pretraining runs exactly as before ,,, ,

### 8.2 Tests and results (H200)

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

**The v1 tests from 6.5 :**

* test 3 (the count) , test 4 (the first loss is 1.446 , not 10.83) and test 7 pass ; for test 7 , 1 , 2 and 4 GPUs with 8 , 4 and 2 accumulation steps give the same step-0 loss and grad norm ,,, ,
* test 2 is covered by the 136-case kernel test ,,, ,
* tests 1 , 5 and 6 are still to do , and test 8 needs generation ,,, ,

### 8.3 Muon vs AdamW : my learning-rate sweep (7 Oct)

The base was pretrained with Muon , and my first Muon run at the AdamW LR (2e-5) learned slower than AdamW , so I ran a sweep ,,, ,

**Setup :**

* 4 GPUs , the same data , 2 epochs , and the same 480 validation chats ,,, ,
* in the Muon runs , the AdamW part (embeddings and norms) used the same LR as Muon , like Kathir's script ,,, ,
* each run's LR decays to one tenth of its peak ,,, ,

All runs start at val 1.4432 ,,, ,

| optimizer , LR | val at step 10 | val at step 80 (end of epoch 1) | best val (step) | final val (step 155) | train loss , epoch 1 / epoch 2 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| AdamW 2e-5 | 1.3497 | 1.2860 | 1.2809 (155) | 1.2809 | 1.316 / 1.239 |
| Muon 2e-5 | 1.4364 | 1.3330 | 1.3162 (155) | 1.3162 | 1.384 / 1.295 |
| Muon 5e-5 | 1.4121 | 1.2920 | 1.2843 (155) | 1.2843 | 1.340 / 1.208 |
| Muon 1.2e-4 | 1.3584 | 1.2747 | 1.2747 (80) | 1.2820 | 1.314 / 1.084 |

### 8.4 Muon vs AdamW : what the papers say

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

**Two pretraining-only studies :**

* Wen et al. (arXiv 2509.02046) : against a well-tuned AdamW , Muon's speed-up shrinks as models grow , 1.4x at 0.1B and 1.1x at 1.2B ,,, ,
* Essential AI (arXiv 2505.02222) : for the same compute , Muon reaches the same loss in less time than AdamW , and it stays data-efficient at large batch sizes ,,, ,

One correction to the optimizer row of my settings table : Llama 2's SFT section does not name its optimizer , AdamW is named only for its pretraining and RLHF ,,, ,

### 8.5 What I learned

* Muon needs a higher LR than AdamW for SFT : at the same 2e-5 it learns clearly slower (1.3162 vs 1.2809) , at 5e-5 it about matches AdamW (1.2843) , and at 1.2e-4 it gives the lowest val of all runs (1.2747) ,,, ,
* A high LR overfits a small set : at 1.2e-4 the best val comes at the end of epoch 1 , and in epoch 2 the train loss falls to 1.08 while val goes back up to 1.287 , so with only 5,000 chats , 2 epochs at 1.2e-4 is too much ,,, ,
* The step-80 number is not a finished 1-epoch run : at step 80 the LR was still 6.6e-5 , so a real 1-epoch run at 1.2e-4 still has to be tested ,,, ,
* Val loss alone can't pick the optimizer : the best runs differ by only about 0.5% , these are one run each on a small set , and val loss doesn't show whether the answers are good or stop at the end token , so the final pick needs evals ,,, ,
* "Everyone uses AdamW for SFT" is not true : most papers don't name their SFT optimizer , and the only paper that tested it (Moonlight) found it best to keep the pretraining optimizer , which for us is Muon ,,, ,
* The global denominator works : 1 , 2 and 4 GPUs give the same step-0 loss and grad norm to the last digit , which only happens if both the count and the backward scale are right ,,, ,

**The epoch grouping depends on the GPU count :**

* on 1 GPU , an epoch is 625 micro-batches , which is not a whole number of 8-micro-batch steps , so epoch 2 starts one micro-batch into step 78 ,,, ,
* on 2 and 4 GPUs , epoch 2 starts exactly at step 78 ,,, ,
* so after step 77 , the runs differ a little by design , and I can fix it by making every epoch a whole number of steps ,,, ,

The val set also depends on the GPU count : 496 chats on 1 or 2 GPUs and 480 on 4 GPUs (every GPU needs whole micro-batches) , so I only compare val numbers between runs with the same GPU count ,,, ,

Padding is the biggest waste : about 70% of every row is padding , and about 77% of the logits are computed for positions that are ignored ,,, ,

**On the box :**

* mpirun passes the whole shell environment to the run , so an old BLU_CKPT=0 from the morning switched off checkpoints in my 6 Oct run ,,, ,

my run scripts now clear every BLU_ variable before each run ,,, ,

## Sources

Papers i collected to refer :

* InstructGPT: Ouyang et al., Training language models to follow instructions with human feedback, 2022. <https://arxiv.org/abs/2203.02155>
* LIMA: Zhou et al., LIMA: Less Is More for Alignment, 2023. <https://arxiv.org/abs/2305.11206>
* Llama 2: Touvron et al., Llama 2: Open Foundation and Fine-Tuned Chat Models, 2023. <https://arxiv.org/abs/2307.09288>
* Zephyr: Tunstall et al., Zephyr: Direct Distillation of LM Alignment, 2023. <https://arxiv.org/abs/2310.16944>
* IFEval: Zhou et al., Instruction-Following Evaluation for Large Language Models, 2023. <https://arxiv.org/abs/2311.07911>
* Llama 3: Llama Team, The Llama 3 Herd of Models, 2024. <https://arxiv.org/abs/2407.21783>
* Tülu 3: Lambert et al., Tülu 3: Pushing Frontiers in Open Language Model Post-Training, 2024. <https://arxiv.org/abs/2411.15124>
* DeepSeek-R1: DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, 2025. <https://arxiv.org/abs/2501.12948>
* Chu et al.: SFT Memorizes, RL Generalizes, ICML 2025. <https://arxiv.org/abs/2501.17161>
* SimpleRL-Zoo: Zeng et al., SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild, 2025. <https://arxiv.org/abs/2503.18892>
* Qwen3: Qwen Team, Qwen3 Technical Report, 2025. <https://arxiv.org/abs/2505.09388>
* OLMo 3: Team Olmo, Olmo 3, 2025. <https://arxiv.org/abs/2512.13961>

Papers from my week-1 reading (numbers taken from my notes) :

* Instruction Tuning for Large Language Models: A Survey, 2023. <https://arxiv.org/abs/2308.10792>
* A Survey on Post-training of Large Language Models, 2025. <https://arxiv.org/abs/2503.06072>
* A Survey of Large Language Models, 2023. <https://arxiv.org/abs/2303.18223>
* Wave Network: An Ultra-Small Language Model, 2024. <https://arxiv.org/abs/2411.02674>
* OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling, 2025. <https://arxiv.org/abs/2506.20512>
* Enhancing Training Efficiency Using Packing with Flash Attention (IBM), 2024. <https://arxiv.org/abs/2407.09105>
* On the Generalization of SFT (Dynamic Fine-Tuning, DFT), 2025. <https://arxiv.org/abs/2508.05629>
* SFT-then-RL outperforms mixed-policy methods for LLM reasoning, 2026. <https://arxiv.org/abs/2604.23747>
* Do Post-Training Algorithms Actually Differ? (scale-dependent rankings), 2026. <https://arxiv.org/abs/2603.19335>
* Zhao et al., Analysing the Impact of Sequence Composition on Language Model Pre-Training, ACL 2024. <https://arxiv.org/abs/2402.13991>
* Still to read in full: post-training data quality (<https://arxiv.org/abs/2506.06522>), SFT vs RL (<https://arxiv.org/abs/2603.13985>), Does Prompt Loss Matter? (<https://arxiv.org/abs/2401.13586>)

* UltraChat: Ding et al., Enhancing Chat Language Models by Scaling High-Quality Instructional Conversations, 2023. <https://arxiv.org/abs/2305.14233>
* Tülu 2: Ivison et al., Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2, 2023. <https://arxiv.org/abs/2311.10702>

Methods and benchmarks :

* DPO: <https://arxiv.org/abs/2305.18290> · PPO: <https://arxiv.org/abs/1707.06347> · GRPO (DeepSeekMath): <https://arxiv.org/abs/2402.03300> · Chinchilla: <https://arxiv.org/abs/2203.15556>
* FLAN: <https://arxiv.org/abs/2109.01652> · Self-Instruct: <https://arxiv.org/abs/2212.10560> · GPT-2 tokenizer: <https://github.com/openai/tiktoken>
* MMLU: <https://arxiv.org/abs/2009.03300> · GSM8K: <https://arxiv.org/abs/2110.14168> · MATH: <https://arxiv.org/abs/2103.03874> · HumanEval: <https://arxiv.org/abs/2107.03374> · BBH: <https://arxiv.org/abs/2210.09261> · TruthfulQA: <https://arxiv.org/abs/2109.07958> · GPQA: <https://arxiv.org/abs/2311.12022> · HellaSwag: <https://arxiv.org/abs/1905.07830>
* MT-Bench: <https://arxiv.org/abs/2306.05685> · AlpacaEval 2: <https://arxiv.org/abs/2404.04475> · Arena-Hard: <https://arxiv.org/abs/2406.11939> · LiveCodeBench: <https://arxiv.org/abs/2403.07974> · lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>

* Tülu 3 chat template (exact , as released) : <https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B/blob/666943798adbde0b1aff34626007e26986a3c107/tokenizer_config.json> (field `chat_template`)
* open-instruct PR 376 , "Remove newline at the end of tulu template" (merged 2024-10-10) : <https://github.com/allenai/open-instruct/pull/376> ; PR 350 , one shared template table for SFT and DPO : <https://github.com/allenai/open-instruct/pull/350>
* open-instruct label masking (`mask_labels`) : <https://github.com/allenai/open-instruct/blob/95a84bb5781c69e58ba5a4f9b265f4b504ce4f2b/open_instruct/dataset_transformation.py#L1307-L1368>
* Tülu 3 model on Hugging Face: [allenai/Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT) (checked tokenizer_config.json for role tokens and chat template string).
* Tülu 3 dataset on Hugging Face: [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture)

Papers referred for AdamW (vs) Muon selection decision :

* Moonlight: Liu et al., Muon is Scalable for LLM Training, 2025. <https://arxiv.org/abs/2502.16982>
* Kimi K2: Kimi Team, Kimi K2: Open Agentic Intelligence, 2025. <https://arxiv.org/abs/2507.20534>
* GLM-4.5: GLM-4.5 Team, GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models, 2025. <https://arxiv.org/abs/2508.06471>
* Qwen2.5: Qwen Team, Qwen2.5 Technical Report, 2024. <https://arxiv.org/abs/2412.15115>
* Qwen2: Qwen Team, Qwen2 Technical Report, 2024. <https://arxiv.org/abs/2407.10671>
* Wen et al., Fantastic Pretraining Optimizers and Where to Find Them, 2025. <https://arxiv.org/abs/2509.02046>
* Essential AI, Practical Efficiency of Muon for Pretraining, 2025. <https://arxiv.org/abs/2505.02222>

Code I was referring for post-training environment :

* PyTorch ATen aten/src/ATen/native/cuda/Loss.cu; Megatron-LM pretrain_gpt.py and finalize_model_grads.py; torchtitan (sum loss + global_valid_tokens); Liger-Kernel ops/cross_entropy.py; TRL trl/trainer/sft_trainer.py; the torchtune and torchforge README notices.
* BluTrain and Tensor-Implementations code: bluscript_zero_ckpt.cpp, dist/dataloader_dist.h, LossKernels.cu, LossBackward.cpp, LossOps.cpp, the GQA attention kernels, EmbeddingKernels.cu.
