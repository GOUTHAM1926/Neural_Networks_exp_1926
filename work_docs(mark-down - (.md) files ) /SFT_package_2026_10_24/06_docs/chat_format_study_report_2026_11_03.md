# Chat format study for the 135M SFT model: what I tested, what I found, what to choose

## 1. Summary

- Before supervised fine-tuning (SFT), a model must learn a chat format: the marks that say where each message starts
  and ends and who wrote it (system, user, assistant). Labs use very different formats and nobody had compared them on
  one model, so I tested them.
- I trained the 135M base model with 23 chat formats on 3 different data mixes with 3 seeds each: 207 training runs
  (plus 1 short check run), on the 8-GPU RTX 6000 Ada node, 2026-11-02 and 11-03. All 208 runs finished.
- Result: the format matters, but only a little (the best and worst format differ by 0.25-0.9% of the validation loss),
  and the winner changes with the data mix.
- By the decision rule I fixed before the study, only one format is never worse than today's format on any mix and
  better on one: v17, the Phi-3 style. ChatML with reserved ids as Qwen and SmolLM2 use it (v11) is within seed noise
  of v17 on every mix (at most 0.0005 apart).
- Safety: today's format (v1, plain-text markers) can never be protected against fake-turn injection. A format whose
  turn boundaries are reserved special ids can be protected. ChatML itself is not required; reserved ids are.
- My choice: **ChatML with reserved ids (the v11 layout)**: protectable, as good as the best version, no new token ids
  needed (ids 1 and 2 already exist and are special), and the most common open format. Every detail and its reason is
  in section 9. It becomes final after the next tests (stop behaviour, injection tests, benchmarks) and one
  confirmation at full length.

## 2. What a chat format is

The model only sees a list of token ids. The format decides which ids mark the turns. The same short chat in three of
the formats I tested (square brackets = one reserved id; everything else is ordinary text):

```text
v1  (today, Tülu 3 style, plain text markers)      v11 (ChatML with reserved ids, as Qwen and SmolLM2)
<|system|>                                         [im_start]system
Be brief.                                          Be brief.[im_end]
<|user|>                                           [im_start]user
Capital of France?                                 Capital of France?[im_end]
<|assistant|>                                      [im_start]assistant
Paris.[end-of-document]                            Paris.[im_end]

v17 (Phi-3 style: one reserved id per role, a reserved end id after every message)
[system]
Be brief.[end]
[user]
Capital of France?[end]
[assistant]
Paris.[end]
```

In v1 the markers like `<|user|>` are ordinary text: they are cut into several normal tokens, exactly as if a user
typed them. In v11 and v17 every marker is one reserved id.

## 3. The safety question: can a user fake a turn?

There are two kinds of prompt injection (research doc `chat_template_special_tokens_tests.md`, Part 4):

- **Trick 1, the hidden order.** The user (or a web page or email the model reads) simply writes "ignore your rules
  and ...". Nothing is faked. No format can stop this; only safety training and product design can.
- **Trick 2, the fake separator.** The user types separator text inside their message, so that part of it looks like a
  new turn: a fake system message, or a fake assistant answer that has already started ("Sure, here are the steps:").
  The model decides who is speaking only from the separators, so it can be fooled. A fake assistant turn raised
  jailbreak success by about 40% in published work (Virtual Context, 2024); Apple's Writing Tools beta was fooled in
  2024 because its role tags were plain text.

Whether trick 2 can be stopped depends on the format:

| separators | can trick 2 be stopped? |
| --- | --- |
| plain text (v1-v3, v7-v9, v15, v19) | never: typed text and the real separator are the same ids |
| reserved ids marked special (all other versions) | yes, if the product converts user and outside text with special-token parsing switched OFF: the typed text of im_start then becomes ordinary pieces, never the real id |

So the answer to "is ChatML mandatory?": no. What is required is that every turn boundary is a reserved special id,
and that the serving code keeps the switch OFF for outside text. ChatML with reserved ids (v10-v14) qualifies, and so
does v17. ChatML written as plain text (v7-v9) does not. I checked in the tokenizer library that an id added as
"special" is protected this way and one added as "not special" is not, so every structure id in my versions is special.

## 4. How I tested

| item | setting |
| --- | --- |
| model | the 135M production base (the same start weights for every run) |
| trainer | a PyTorch copy of the BluTrain SFT recipe, checked against BluTrain: loss matches within 0.01-0.05%; 100 steps against a BluTrain H200 run differ by about as much as two H200 runs differ (work doc 45.132). BluTrain itself cannot be built for these GPUs yet. |
| hardware | 8x RTX 6000 Ada (one run per GPU), about 20-23 minutes per run, about 10.5 hours of wall time over two days |
| data mixes | m1 = smol-smoltalk (SmolLM2's SFT data, many multi-turn chats); m2 = the Tülu 3 OLMo-2 mix (mostly single-turn, much math and code); m3 = m1 plus grade-school math |
| per run | 100k chats of that mix, 1 epoch, ~128 chats per step (~730-806 steps), learning rate 1e-4, Muon, cosine schedule, 3% warm-up |
| seeds | 3 per version and mix (different data order) |
| runs | 23 versions x 3 mixes x 3 seeds = 207, plus 1 short check run. The first plan had 22 versions (198 runs); the exact DeepSeek V3/R1 copy (v23) was added on 11-02. |

Fairness rules, all checked by code: the same chats for every version of a mix; the same data order per seed; every
structure id added as special; only the answers (and their end id) are trained, never the prompts.

What I measured: the **answer-token validation loss**, with end ids left out, so every version is graded on exactly
the same tokens. Lower is better.

Decision rule, fixed before the study (plan 4.28 / 4.38): a difference counts only if it has the same sign in all 3
seeds and is more than twice the seed-to-seed spread. A version may replace today's v1 only if it is never worse on any
mix.

## 5. The 23 versions

"0" = the answer ends with the pretraining end-of-document id; "S" = a separate end id; "X" = the separate end id
between turns and the end-of-document id at the very end.

| version | layout | separators | end | copy of |
| --- | --- | --- | --- | --- |
| v1 / v2 / v3 | Tülu 3 markers | plain text | 0 / S / X | v1 = Tülu 3, OLMo 2 (today's format) |
| v4 / v5 / v6 | Tülu 3 layout, one id per role | reserved | 0 / S / X | v4 = OpenAssistant, DeepSeek-R1 |
| v7 / v8 / v9 | ChatML | plain text | 0 / S / X | - |
| v10 / v11 / v12 | ChatML | reserved | 0 / S / X | v11 = SmolLM2, Qwen, Hermes 3; v12 = OLMo 3 |
| v13 / v14 | as v11 / v12, new rows started exactly as those teams did | reserved | S / X | SmolLM2 / Qwen; OLMo 3 |
| v15 | no markers at all | none | 0 | "The Price of Format" paper |
| v16 | ChatML with one id per role | reserved | S | Cohere Command R |
| v17 | one id per role, end id after every message | reserved | S | Phi-3 |
| v18 | one id per role, no end id between turns | reserved | next role id | GLM-4 |
| v19 | Tülu 3 markers, end id after every message | plain text | 0 | Zephyr |
| v20 | start id, role word, end-of-header id | reserved | S | Llama 3 |
| v21 | ChatML, system text merged into the first user message | reserved | S | Mistral v0.1 |
| v22 | v11 plus a default system message when the chat has none | reserved | S | SmolLM2, Qwen2.5 |
| v23 | DeepSeek layout (system text first, one id per turn) | reserved | 0 | DeepSeek V3 / R1 |

## 6. Results

Today's v1 scored 1.1017 (m1), 1.1589 (m2), 1.0156 (m3). The table shows each version's difference to v1, mean of 3
seeds (negative = lower loss = better). "better" / "worse" only where the decision rule says the difference is real;
"=" means within seed noise. The largest seed-to-seed spread of any version was 0.0006 (m1), 0.0010 (m2), 0.0006 (m3).

| version | m1 | m2 | m3 | never worse than v1? | protectable against trick 2? |
| --- | --- | --- | --- | --- | --- |
| v2 | -0.0002 = | +0.0003 = | -0.0002 = | yes (same as v1) | no |
| v3 | -0.0002 = | +0.0005 = | -0.0001 = | yes (same as v1) | no |
| v4 | +0.0012 worse | +0.0016 worse | +0.0005 = | no | yes |
| v5 | +0.0009 = | +0.0018 worse | +0.0003 = | no | yes |
| v6 | +0.0009 = | +0.0020 worse | +0.0004 = | no | yes |
| v7 | +0.0026 worse | -0.0066 better | -0.0002 = | no | no |
| v8 | +0.0024 worse | -0.0064 better | -0.0004 = | no | no |
| v9 | +0.0024 worse | -0.0063 better | -0.0004 = | no | no |
| v10 | +0.0011 worse | -0.0019 better | +0.0004 = | no | yes |
| v11 | +0.0010 worse | -0.0015 better | +0.0002 = | no | yes |
| v12 | +0.0011 worse | -0.0014 better | +0.0002 = | no | yes |
| v13 | +0.0013 worse | -0.0010 = | +0.0005 = | no | yes |
| v14 | +0.0012 worse | -0.0013 better | +0.0004 = | no | yes |
| v15 | +0.0015 worse | -0.0007 = | +0.0035 worse | no | no |
| v16 | +0.0007 = | -0.0006 = | +0.0012 worse | no | yes |
| **v17** | **+0.0005 =** | **-0.0016 better** | **+0.0005 =** | **yes** | **yes** |
| v18 | +0.0007 = | +0.0018 worse | +0.0004 = | no | yes |
| v19 | +0.0003 = | +0.0034 worse | +0.0013 worse | no | no |
| v20 | +0.0018 worse | +0.0015 worse | +0.0020 worse | no | yes |
| v21 | +0.0016 worse | -0.0015 better | +0.0011 worse | no | yes |
| v22 | +0.0016 worse | +0.0012 worse | +0.0006 worse | no | yes |
| v23 | +0.0016 worse | +0.0012 = | +0.0010 worse | no | yes |

Every value per seed is in the work doc (45.139) and in `fmt_final_1103.txt` / `fmt_vs_v1_1103.txt` next to the run
data.

## 7. What the results tell us

1. **The format matters, but a little.** The best and worst versions differ by 0.0028 on m1, 0.0100 on m2 and 0.0039
   on m3 (0.25%, 0.9%, 0.4% of the loss), against a seed noise of at most 0.0010.
2. **The end id does not matter.** Inside each family of three (end 0, S or X) the results differ by at most 0.0004,
   which is noise. The answer can end with the pretraining end id or with a new one.
3. **The best format depends on the data.** ChatML as plain text (v7-v9) is clearly the best on the Tülu mix (m2) and
   clearly the worst on SmolTalk (m1). This is why every version ran on 3 mixes: one mix alone would have picked
   differently.
4. **Copying a big lab's format is not automatically best for this model.** Qwen/SmolLM2 (v11), OLMo 3 (v12) and
   DeepSeek (v23) are each worse than v1 on at least one mix; Llama 3 (v20) is worse on all three; a default system
   message (v22) is worse on all three.
5. **No format at all (v15) is worse** on m1 and m3, so the markers do help the model.
6. **Only v17 passes the rule:** never worse than v1 on any mix and better on m2 (-0.0016), and it is protectable
   against trick 2. v2 and v3 tie with v1 but, like v1, can never be protected.

## 8. Which one to choose

- **Chosen: ChatML with reserved ids, the v11 layout** (as Qwen, SmolLM2 and Hermes 3).
  - Safety: every turn starts with the reserved id 1 and every message ends with a reserved id, so a product can
    block fake turns.
  - Quality: within seed noise of v17, the only version that passed the rule (v11 vs v17: m1 1.1027 / 1.1022, m2
    1.1575 / 1.1573, m3 1.0158 / 1.0161). Against today's v1: better on m2 (-0.0015), worse on m1 (+0.0010), equal on
    m3.
  - Cost: no new token ids. Ids 1 and 2 (im_start, im_end) already exist in the tokenizer and are already special.
    v17 would need 3 new role ids learned from zero.
  - Tools: the most common open format; Hugging Face chat templates, vLLM and lm-eval handle it as is.
- **Kept as the comparison: v1** (today) and **v17**, in the next tests.
- **Not ChatML as plain text (v7-v9)**, even though it wins on m2: it loses on m1 and can never be protected.

The plan (4.48) makes the choice final only after:

1. the other measurements on v11, v17, v1 (and v7 as a reference): whether the model stops correctly, how well it uses
   earlier turns, leakage, standard benchmarks through lm-eval on vLLM, the injection tests (M7: fake separators with
   the switch on and off; M7b: a secret in the system message) and over-refusal (XSTest);
2. a confirmation run at full data length;
3. a re-run in BluTrain on H200, because this study used the PyTorch copy of the recipe.

## 9. The details of the chosen format, decided one by one

The 135M tokenizer has 17 special tokens: id 0 `<|endoftext|>` (the end of every pretraining document), id 1
`<|im_start|>`, id 2 `<|im_end|>`, and ids 3-16 code and notebook markers (`<repo_name>`, `<file_sep>`,
`<jupyter_code>` and others). It has 49,152 tokens; the model's embedding has 49,280 rows, so rows 49,152-49,279 are
spare. Ids 1 and 2 occurred 0 times in the 900B pretraining tokens: their rows were never trained.

The chosen layout, with the ids:

```text
[0][1]system\nBe brief.[2]\n[1]user\nCapital of France?[2]\n[1]assistant\nParis.[2]\n[1]user\nAnd of Italy?[2]\n[1]assistant\nRome.[2]
```

| decision | choice | why |
| --- | --- | --- |
| 1. start and end of every message | id 1 (im_start) starts, id 2 (im_end) ends | reserved and special, so typed text can never produce them (with the switch OFF); they exist already |
| 2. the role names | plain words `system`, `user`, `assistant` after id 1 | one special id per role inside ChatML (v16, Cohere) was not better (worse on m3 by 0.0010); a typed role word cannot open a turn, because a turn opens only with the reserved id 1; no new ids to learn from zero |
| 3. which id ends an answer | id 2, the same as every other message | end with id 0 (v10), id 2 (v11) or id 2 then id 0 at the very end (v12) gave the same loss (within 0.0004 on every mix). Id 2 keeps every message the same and is what Qwen and SmolLM2 do; id 0 keeps its one pretraining meaning, "document finished". Using id 0 instead is also acceptable on this evidence; the stop test (M2) checks whether the model stops as reliably with the new id 2 as with the already-trained id 0, so this one is confirmed there |
| 4. stop ids at generation | both id 2 and id 0 | if a model ever writes id 0 inside an answer and the product does not stop, it may start an unrelated "document" (this happened to GPT-4 in 2023, reported by N. Carlini); Qwen and OLMo 3 also stop on both |
| 5. new token ids | none | ChatML needs none. If one is ever needed (think or tool tags), it goes into the spare rows from 49,152, added as special; never ids 3-16, which are code markers, some used in pretraining with their own meaning |
| 6. starting values of ids 1 and 2 | the average of the rows of each token's own text pieces (for im_start: the pieces of its written name) | used for every version of the screen; keeping the untrained rows as they are (what Qwen and SmolLM2 did, v13) gave the same loss (within 0.0004) |
| 7. the system message | its own turn; no default text when a chat has none | merging it into the user turn (v21) was worse on m1 and m3; a default system message (v22) was worse on all three mixes |
| 8. the start of a chat | id 0, as every pretraining document starts | the same in every version, fixed before the study |
| 9. what is trained | the answers and the end id after each answer; never the prompts | fixed before the study (work doc 44.6.2) |
| 10. serving | user and outside text converted with special-token parsing OFF; chat structure added only by the template | without this, the reserved ids protect nothing (research doc Part 4.6, tested with the tokenizer library) |

## 10. What is not done yet

- Only one measurement (the answer loss) so far; the generation and safety tests above are the next step.
- The learning-rate check of the plan (4 versions at 5e-5 and 2e-4) was not run; all runs used 1e-4.
- The 100k-chat screen is shorter than a real SFT run; the confirmation at full length is still to do.
- Whatever format is chosen, the product must convert user and outside text with special-token parsing OFF, and must
  stop generation on the format's end id and on the end-of-document id. Trick 1 needs safety training; no format fixes
  it.
- Later (decided 11-02): test whether the names of the marker tokens matter (original names, own names, random
  strings) on the chosen format.
