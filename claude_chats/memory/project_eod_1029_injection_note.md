---
name: project_eod_1029_injection_note
description: "The user asked (10-29 night) that the EOD for 2026-10-29 include the injection / ChatML / parsing switch / safety / pad explanation exactly as explained in chat (plan §4.29-4.31), in plain words"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T17:11:32.747Z
---

The user said: "save this and document like this in our document and also in today's EOD, and remember to tell like
this all the things, exactly this response regarding injections and safety and ChatML and parsing and all, and those
controls."

**What the 2026-10-29 EOD must cover on this topic** (in the EOD format of [[feedback_eod_report_format]]: lowercase,
no full stops, "i", plain words, no paper names):
- **When injection happens:** when the finished model is used in a product, not during training.
- **Who:** the company writes hidden rules; the customer types; both reach the model as one list of numbers.
- **Trick 1, the hidden order:** "ignore previous instructions" (the twitter bot 2022, bing chat 2023). No switch fixes
  it; only training does.
- **Trick 2, the fake separator:** typed separator text looks like the company's rules. Plain markers (our current
  arm a) can always be faked. ChatML's reserved numbers 1 and 2 stop it only if outside text is converted with the
  switch OFF.
- **The switch rule:** the same in training and in use. Our separators = reserved numbers; all outside text = OFF. Our
  SFT drops chats containing marker names. The DPO script's special mode converts text with the switch ON.
- **Control:** in our hands for what we build. Once open-sourced, only the format choice, the reference code and the
  training go with the model. Injection safety is now a decision criterion (M7 + new M7b), not an assumption.
- **Pad = end:** our pad and end are both id 0. We are safe because all our code (SFT and the colleagues' DPO, checked
  10-29) works by position. Future RL must be checked the same way.
- **Also that day:** the format study plan v3 (15 arms × 3 mixes + LR check). The user accepted M8, M5b, arm 15 and
  the end-label check.
- **Late night, plan 3.1:** safety tests M7b and M9 (refusing harmful prompts / not refusing safe ones, with and without
  the template), all three as decision criteria. Separator ids must be "special" (tested: the switch does not protect
  "added, not special" ids). Think/tool token research. The colleague document `chat_template_special_tokens_tests.md`
  and its `.docx`.
- **How instructions reach the model:** the company fills a system box; the product's code inserts the separator ids;
  every text is converted with the switch OFF. The model obeys the system part because training taught it, and only
  from chats that have a system message.
- **How an owner changes rules (the user asked for this exact explanation in the EOD too):**
  - Rules are plain text in the system box, kept in the product's settings and sent with every conversation; they are
    not stored in the model. Editing the text (SAVE50 → SAVE60) changes the rules at once.
  - No daily fine-tuning: SFT teaches general behaviour once.
  - The switch is never turned ON for anyone's text. Large or changing data is looked up at request time.
  - Outside text (web pages, reviews, emails) goes in with the switch OFF.
  - Plan §4.33; colleague file Part 4.12.
- **Why ChatML's 2 tokens are enough (asked after that):**
  - Trick 2 needs a boundary, i.e. ending a message and starting another. Nobody can type a reserved id with the
    switch OFF, so a typed "system" stays a word inside the customer's message.
  - Rule: every token that starts or ends a turn must be special; role words may stay plain.
  - Making only "system" special would still allow a fake assistant turn.
  - Tests: arms 4-6 vs 10-14 already compare role-as-id vs boundary-id; the P0 protectability table (F0) and M7
    variants (role word only, look-alike separators) were added.
  - Plan §4.34; colleague file Part 4.13.
- **The whole picture (asked last, 10-29 night; plan §4.35, colleague file Part 4.14).** The user wants it in the EOD
  too.
  - **Roles:** the attacker uses the tricks; the owner defends. The owner types plain rules in an admin page. Only the
    product's code writes `[1] system ... [2]`. "system" anywhere else is just a word.
  - **Trick 1 real cases:** the remoteli.io Twitter bot (2022, the Challenger claim); Bing "Sydney" (2023); the Bing
    indirect web-page demo (2023); the Chevrolet $1 Tahoe "no takesies backsies" (Dec 2023); the DPD bot swearing
    (Jan 2024).
  - **Trick 2 real cases:** Carlini, GPT-4 writing endoftext and drifting to obesity (Aug 2023); Tensor Trust players
    using im_start / im_end against GPT-3.5 Turbo (2023); Apple Intelligence beta plain role tags (Aug 2024); MetaBreak
    (Oct 2025).
  - **When trick 2 works:** the switch matters only for special separators. Plain separators can always be forged, and
    so can "added, not special" ids even with the switch OFF. With no rules set it is smaller but not harmless (a fake
    assistant turn, a fake tool result).
  - **New F0 rule:** generation stops on the arm's end id AND on id 0 (from Carlini's case).
- **Last request of 10-29 night: explain every incident in detail in the EOD too.** Use plan §4.37 / colleague file
  Part 4.15: the story of each incident (what the product was, what the attacker typed, what happened, why, what
  followed). The incidents are:
  - Goodside "Haha pwned"
  - remoteli.io
  - Bing Sydney
  - Bing web page
  - Chevrolet $1 Tahoe
  - DPD
  - Carlini GPT-4 endoftext
  - Tensor Trust im_start / im_end
  - Apple Intelligence tags
  - MetaBreak
  - plus Virtual Context and ChatBug
- **Also that night:**
  - re-read all 57 user messages since 10-28; the 10-28 chat answers that were missing were recovered into §4.36 (ids
    1 and 2 copied from SmolLM2, the 49,280 rows, the unused rows becoming one vector, how new ids learn, the stage
    table);
  - plan v4 in one place (§4.38) with its change log;
  - the F0 start plan with 13 steps (§4.39).
- **The very last question of 10-29 (plan §4.40, colleague file Part 4.16): why a typed "system" message never becomes
  `[1] system`.** The code wraps the customer's text in `[1] user ... [2]`; the role word comes from which box the text
  came from.
  - **Full path:** the admin page (behind a login) → the shop's database → the server code sets the roles → the model
    server builds the ids message by message with the switch OFF.
  - **Careless path:** render everything to one string, then switch ON, which produces a perfect fake owner message.
  - **Role-field tampering:** set roles on the server, never in the browser.
  - **Where the rules live:** the rules are not in the tokenizer files.
  - **Release checklist** added to §4.38.
- **USE THE 6-STEP TRICK-2 EXAMPLE IN THE EOD (the user asked for it explicitly).**
  - **Order:** first explain the switch, then the path: owner saves the rules → cheater types the ChatML names → the
    browser sends one field → the server sets the roles → the model server builds the ids message by message with the
    switch OFF → the model reads one owner message plus one customer message. Then show the careless version next to it.
    (Plan §4.40, colleague file Part 4.16.)
  - **Also explain:** the cheater knows the format because open models publish it, and needs only the text; never rely
    on secrecy. The separators should be reserved ids marked special: ids 1 and 2, or new ids added as special.
    Decodable and public is fine. (§4.41, Part 4.17.)
- **Source audit, 10-29 night (§4.41, colleague file Part 6.12).** Every source was traced to its official owner.
  - **Changes:** WritingPrompts comes from Facebook's official archive (not the euclaise copy). XSTest uses the authors'
    Paul/XSTest and their string-match classifier. A source manifest is written with every evaluation.
  - **Flag:** No Robots (CC-BY-NC) is in m2, for factor 3.
  - **Checked:** the four gated-repo renders: the Mistral copy is identical; Llama 3.1 = Meta's code; Gemma = Google's
    documentation.
- **The fake assistant turn and the official names (plan §4.42, colleague file Part 4.18).**
  - Safety training puts the refusal at the START of the assistant's answer, and the safety lives in the first few words
    ("shallow safety alignment", Qi et al. 2024).
  - A fake earlier assistant turn that already agreed ("Sure, here are the steps:"), or a prefilled answer start, jumps
    over that refusal point. It targets the model's own safety habit, so no system message is needed.
  - Names: both tricks = prompt injection (OWASP LLM01). Trick 1 = direct or indirect prompt injection. Trick 2 =
    special token injection (Virtual Context title; promptfoo's STI), also chat-template, role or delimiter injection.
    The fake assistant start = prefilling attack or message overflow.
- **PUT IN THE EOD (the user asked explicitly): how the cheater knows the exact format** (plan §4.43, colleague file
  Part 4.19).
  - Open models publish the template on purpose (`tokenizer_config.json` chat_template, model cards, papers, official
    pages).
  - The cheater needs only the TEXT: Tülu 3's `<|assistant|>` = [27, 91, 78191, 91, 29] whoever writes it.
  - Closed products leak it: the product names its model; trick 1 (Bing Sydney); shipped files (Apple tags); trial and
    error (Tensor Trust, promptfoo); look-alikes (MetaBreak); the model's own output (Carlini).
  - **Never rely on the format being secret** (Kerckhoffs's principle vs security through obscurity). Protection =
    special ids + switch OFF + training, and it works even with the format public.
- **Jailbreaks and the when-fooled table (plan §4.44, colleague file Part 4.20).**
  - **Six families:** role-play (DAN), fake earlier turns (Virtual Context, many-shot), prefilling and message overflow,
    adversarial suffixes, format mismatch, fine-tuning.
  - **Layers of defence:** special ids + switch OFF; roles on the server; no outside prefill; safety recovery training
    (Qi et al.: prefill success 42-57% → 3-4.5%); instruction hierarchy; output filters.
  - **Table of 7 setups × 5 attacks:** the only safe row for typed separators is special ids + switch OFF + roles on
    the server + no outside prefill.
  - **New side experiment X4:** safety recovery examples in the SFT data.
- **PUT IN THE EOD: what the format CANNOT fix** (plan §4.45 part 1, colleague file Part 4.21). Each limit with what
  reduces it:
  - hidden orders → training, X5;
  - look-alikes → training;
  - prefilling → product rule + X4;
  - roles chosen in the browser → roles on the server, R1;
  - other companies' products → format + reference code + training;
  - jailbreaks without separators → training and filters;
  - fine-tuning of released weights → cannot be enforced.
- **Roles browser vs server:** with roles on the server, the browser sends only {"message"} and the server sets the
  roles. With roles in the browser, the page sends the whole list, so the customer can edit in his own system or
  assistant message, and it gets REAL separators.
- **Safety experiments planned.** Now: M7 (e) many fake turns, M7 (f) direct prefill. Later: X4 safety recovery, X5
  injected-instruction training (StruQ-style), R1 serving-code checks + output-filter evaluation.
- **The when-fooled table = "the holy grail summary"** (user). It is now at the start of the colleague file's Part 4.
- **EOD 10-29 WRITTEN (10-29 night):** a continuation from "so today ," with 5 paragraphs and 3 tables (teams;
  when-fooled; stages and time) plus the 2 verda lines. No GPU was used that day.
- **Time estimate (plan §4.38):**
  - F1: ~15 h;
  - F2: ~3.3 h;
  - F3: ~11 h;
  - F4: ~2.6 h;
  - F0: ~1 day.
  - F0→F3 is about 2.5-3 days. The evaluation time is an estimate (15 min light / 45 min full per checkpoint); F0 step
    12 measures it.
- The user said: "we will explain in eod same response explanation only of these both". Use the content of the 10-29
  late-night chat answers (how instructions reach the model + why safety; how owners set rules) in plain EOD style.

**How to apply:** when the user says "eod" for 2026-10-29, include this, but only if their draft lists this work; if
they paste a draft, follow [[feedback-eod-only-what-user-lists]] and ask before adding topics their draft lacks.

**Why:** the user explicitly asked for it on 2026-10-29 night.

**User correction on the EOD draft (10-29 night).** "You didn't tell what the model carries though we can't control
everything; I told you to remember it."
- The EOD MUST include a "what i cannot control vs what our model carries" block:
  - reserved special separators;
  - every separator marked special;
  - training chats with owner rules, and no typed separator text in the training data;
  - stop on the end token and id 0;
  - the template and a safe-use rule shipped with the model;
  - later, safety training.
- The 4th and 5th paragraphs must be MINIMAL and cover component 9 only (chat template / special-token versions, the
  safety test cases, what is measured). No data mixes, no hyperparameters, no seeds while those factors are not fixed.
