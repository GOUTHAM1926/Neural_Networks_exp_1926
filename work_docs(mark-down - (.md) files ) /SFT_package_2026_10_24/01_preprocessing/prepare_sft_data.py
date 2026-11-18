#!/usr/bin/env python3
"""Turn a Hugging Face chat dataset into BluTrain SFT files (SFT report sections 3.2-3.10).

Output directory:
  inputs.bin        uint16 [N, T]  token ids the model reads, padded with the tokenizer's pad id (GPT-2: 50256
                                   <|endoftext|>; the 135M tokenizer: 0 <|endoftext|>; meta.json "pad_id")
  labels.bin        uint16 [N, T]  token to predict at each position; 65535 = ignore (prompt, padding)
  valid_counts.bin  uint32 [N]     number of non-ignored labels in each row (for the global denominator)
  seq_lengths.bin   int32  [sum]   --pack ffd only (v2 packed rows): the lengths of the chats of every row, row after
                                   row, in the order they sit in the row (TRL's seq_lengths column, trl/data_utils.py).
  seq_offsets.bin   int64  [N + 1] --pack ffd only: row r's chats are seq_lengths[seq_offsets[r] : seq_offsets[r+1]].
                                   The rest of a row (T - sum of its chats) is its pad tail. The trainer turns the list
                                   into the attention kernels' per-chat plan (each chat attends only to itself) and
                                   into RoPE positions that restart at 0 for every chat (as TRL's position_ids).
  meta.json         settings, dataset revision, tool versions, counts, drop reasons, length histogram

Template "tulu-plain-v1" (option A: role markers are plain text, each answer ends with the real end token):
  <|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...
Each piece is tokenized on its own, so a marker never merges with the text next to it (BPE).
Only answer tokens and the end token after each answer are targets; every assistant turn is trained.
Text that looks like a special token (e.g. "<|endoftext|>" inside a user message) is encoded as plain text.
Default data: HuggingFaceH4/ultrachat_200k (English-only, MIT), so no language filter is needed.

--tokenizer <tokenizer.json> (2026-10-27, the 135M base: the team BPE of 49,152 ids, model rows 49,280): template
"chatml-v1" with the tokenizer's own special ids (SmolLM2's set: 0 <|endoftext|>, 1 <|im_start|>, 2 <|im_end|>; the
tok50b_v2 README: "SFT and DPO: use ChatML"):
  <|endoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n<|im_start|>user\n ...
  - the chat starts with <|endoftext|> (id 0) because every pretraining document starts with it; an input only;
  - targets: the answer tokens and the <|im_end|> after each answer (the model learns to stop); the "\n" after
    <|im_end|> separates turns (ChatML), is an input only, and is not written after the last turn;
  - row padding (pad tails) is id 0; a chat whose text holds "<|endoftext|>", "<|im_start|>" or "<|im_end|>" is dropped
    (the HF tokenizer would turn that text into the structural id); ids 3-16 stay as the tokenizer makes them (as in
    pretraining); the 128 reserved rows 49,152-49,279 never occur.

--tokenizer with --template tulu-plain-v1 / tulu-special-v1 (2026-10-28, the special-token ablation on the 135M base;
our template, decided 2026-10-15: Tulu 3 without the "\n" after the end token, single- and multi-turn):
  <|endoftext|><|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n{answer}<|endoftext|><|user|>\n ...
  - A tulu-plain-v1:   the markers are plain text (<|assistant|>\n = <, |, ass, istant, |, >, \n); the BASE tokenizer;
  - B tulu-special-v1: <|system|> / <|user|> / <|assistant|> are single tokens (ids 49152-49154 from
    sft/hf_to_blutrain.py --new-tokens, which also writes the extended tokenizer.json to pass here), then "\n" as text;
  - both: the chat starts with <|endoftext|> (id 0: every pretraining document starts with it, and the HF tokenizer
    adds it as BOS), the end token after every answer is <|endoftext|> (id 0, a target), padding is id 0; a chat whose
    text holds <|endoftext|>, <|im_start|>, <|im_end|>, <|system|>, <|user|> or <|assistant|> is dropped in BOTH arms
    (so A and B train on exactly the same chats and the same answer tokens).

--mix <file.json> (2026-10-28): several datasets in one packed set, written as <out>/train and <out>/val in one go.
The file is a list of parts {"dataset", "revision", "split", "sources" (optional: keep only these 'source' values),
"drop_sources" (optional), "num" (train chats, 0 = all, -1 = none: a val-only part), "val_num" (held-out chats)}.
Each part is shuffled with
--seed; its first val_num usable chats go to val, the rest (up to num) to train, so train and val never share a chat.

Rows are labelled the usual way: inputs = ids[:-1], labels = ids[1:] with prompt positions set to 65535.
--pack none (default, v1): one chat per row, padded. --pack ffd (v2): every chat is labelled on its own first (so no
target ever crosses into the next chat), then whole chats are packed into rows with First-Fit-Decreasing (the rule
chosen in work_docs/sft_packing_algorithms_bfd_ffd.md section 13), and the rows are shuffled.

A chat longer than T is dropped (reason "too_long"), never cut: every kept chat is whole, with every answer and its end
token (as torchtitan; decided 2026-10-31). Longer chats need a longer T. Why, what was built and measured before the
decision, and the only split we would consider later (with examples): the comment at the length rule in _examine().
"""
import argparse
import collections
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

PAD_ID = 50256          # input padding = <|endoftext|>, a real GPT-2 id the embedding handles (report 3.8)
IGNORE_ID = 65535       # label marker: no loss, no gradient (report 3.5)
VOCAB_ROWS = 50304      # model vocab rows (50,257 GPT-2 ids padded to 50,304)
TEMPLATE = "tulu-plain-v1"
END_ID = PAD_ID         # the target after every answer: <|endoftext|> for tulu-plain-v1 (main() sets all four)
N_IDS = 50257           # ids the tokenizer can make (inputs and labels must stay below it)
ROLES = ("system", "user", "assistant")
CHATML_STRUCT = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
STRUCT_TEXT = CHATML_STRUCT + ("<|system|>", "<|user|>", "<|assistant|>")   # dropped from chat text (HF tokenizer)


class Gpt2Enc:
    """tiktoken gpt2, template tulu-plain-v1 (the 1.6B / 1.7B runs)."""
    template = "tulu-plain-v1"

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        assert self.enc.eot_token == 50256 and self.enc.n_vocab == 50257
        self.pad, self.end, self.bos, self.vocab_rows, self.n_ids = 50256, 50256, None, 50304, 50257
        self.desc, self.tools = f"tiktoken gpt2 ({tiktoken.__version__})", {"tiktoken": tiktoken.__version__}
        self.drop_struct = False        # the GPT-2 path encodes special-looking text as plain text (unchanged)
        self.strip = False              # and keeps message text exactly as it is (unchanged)

    def text(self, s):
        return self.enc.encode_ordinary(s)

    def marker(self, role):
        return self.enc.encode_ordinary(f"<|{role}|>\n")

    def decode(self, ids):
        return self.enc.decode(ids)


class HfEnc:
    """A Hugging Face tokenizer.json with SmolLM2's special ids (the 135M base). Templates: tulu-plain-v1 (arm A),
    tulu-special-v1 (arm B, needs the extended tokenizer), chatml-v1."""

    def __init__(self, path, vocab_rows, template):
        import tokenizers
        self.template = template
        self.tok = tokenizers.Tokenizer.from_file(path)
        got = {t: self.tok.token_to_id(t) for t in CHATML_STRUCT}
        assert got == {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}, f"not SmolLM2's special ids: {got}"
        self.pad, self.bos, self.start = 0, 0, 1
        self.end = 2 if template == "chatml-v1" else 0          # <|im_end|> for ChatML, else <|endoftext|>
        self.drop_struct = True
        # Strip whitespace at both ends of every message (as Llama 3's template does with | trim, and as
        # blutrain_to_hf.py's chat template does): otherwise text that starts or ends with newlines merges with the
        # template's "\n" when a tool tokenizes the whole chat (measured 2026-10-28 on 2,000 smol-smoltalk chats:
        # arm A 1997/2000 token-identical, arm B 1856/2000; the mismatches were all leading/trailing whitespace).
        self.strip = True
        self.role_id = {r: self.tok.token_to_id(f"<|{r}|>") for r in ROLES}
        if template == "tulu-special-v1":
            assert all(i is not None and i >= 49152 for i in self.role_id.values()), \
                f"tulu-special-v1 needs the extended tokenizer (hf_to_blutrain.py --new-tokens): {self.role_id}"
            # The length limit uses arm A's length (the plain-text markers are 5-6 pieces longer than one token), so
            # arm B drops exactly the chats arm A drops: both arms train on the same chats (2026-10-28: without this,
            # 53 smol-smoltalk chats just over T in arm A fitted in arm B).
            # Arm A's marker length needs the BASE tokenizer: in the extended one "<|user|>" is a single token.
            j = json.load(open(path))
            j["added_tokens"] = [t for t in j["added_tokens"] if t["content"] not in {f"<|{r}|>" for r in ROLES}]
            base = tokenizers.Tokenizer.from_str(json.dumps(j))
            self.filter_extra = {r: len(base.encode(f"<|{r}|>\n", add_special_tokens=False).ids) - len(self.marker(r))
                                 for r in ROLES}
            assert all(v > 0 for v in self.filter_extra.values()), self.filter_extra
        else:
            assert all(i is None for i in self.role_id.values()), \
                f"{template} needs the BASE tokenizer (the markers are tokens here: {self.role_id})"
        self.n_ids = self.tok.get_vocab_size(with_added_tokens=True)
        assert self.n_ids <= vocab_rows, f"tokenizer has {self.n_ids} ids, the model only {vocab_rows} rows"
        self.vocab_rows = vocab_rows
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        self.desc = f"{os.path.abspath(path)} (sha256 {sha}, {self.n_ids} ids)"
        self.tools = {"tokenizers": tokenizers.__version__}

    def text(self, s):
        return self.tok.encode(s, add_special_tokens=False).ids

    def marker(self, role):
        if self.template == "tulu-special-v1":
            return [self.role_id[role]] + self.text("\n")
        return self.text(f"<|{role}|>\n")

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)


def tokenize_conversation(messages, enc):
    """Return (ids, train_mask) for one conversation, or (None, reason) if the row is unusable."""
    if not messages:
        return None, "empty_row"
    if any(m.get("role") not in ROLES for m in messages):
        return None, "unknown_role"
    if not any(m["role"] == "assistant" for m in messages):
        return None, "no_assistant_turn"
    if messages[-1]["role"] != "assistant":
        return None, "last_turn_not_assistant"
    if enc.template == "chatml-v1":
        return tokenize_chatml(messages, enc)
    ids, mask = ([enc.bos], [False]) if enc.bos is not None else ([], [])
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        if enc.strip:
            content = content.strip()
        if role == "assistant" and not content.strip():
            return None, "empty_answer"
        if enc.drop_struct and any(t in content for t in STRUCT_TEXT):
            return None, "special_token_text"
        marker = enc.marker(role)
        ids += marker
        mask += [False] * len(marker)
        body = enc.text(content)
        trained = role == "assistant"
        ids += body
        mask += [trained] * len(body)
        if trained:
            ids.append(enc.end)         # the end token is a target: the model must learn to stop
            mask.append(True)
        else:
            nl = enc.text("\n")
            ids += nl
            mask += [False] * len(nl)
    return (ids, mask), None


def tokenize_chatml(messages, enc):
    """chatml-v1 (the 135M base): <|endoftext|> first (every pretraining document starts with it), then
    <|im_start|>{role}\n{content}<|im_end|> per turn with "\n" between turns. Targets: answer + its <|im_end|>."""
    ids, mask = [enc.bos], [False]
    for i, m in enumerate(messages):
        role, content = m["role"], m.get("content") or ""
        if enc.strip:
            content = content.strip()
        if role == "assistant" and not content.strip():
            return None, "empty_answer"
        if any(t in content for t in STRUCT_TEXT):
            return None, "special_token_text"
        head = [enc.start] + enc.text(f"{role}\n")
        ids += head
        mask += [False] * len(head)
        body = enc.text(content)
        trained = role == "assistant"
        ids += body
        mask += [trained] * len(body)
        ids.append(enc.end)             # <|im_end|>: a target after an answer (the model must learn to stop)
        mask.append(trained)
        if i < len(messages) - 1:       # ChatML's newline between turns: an input, never a target
            nl = enc.text("\n")
            ids += nl
            mask += [False] * len(nl)
    return (ids, mask), None


def build_row(ids, mask, T):
    """inputs = ids[:-1]; labels = ids[1:] where the next token is trained, else IGNORE. Padded to T."""
    n = len(ids) - 1
    inp = np.full(T, PAD_ID, dtype=np.uint16)
    lab = np.full(T, IGNORE_ID, dtype=np.uint16)
    inp[:n] = ids[:-1]
    nxt = np.asarray(ids[1:], dtype=np.int64)
    keep = np.asarray(mask[1:], dtype=bool)
    lab[:n] = np.where(keep, nxt, IGNORE_ID)
    return inp, lab, int(keep.sum())


def ffd_pack(lengths, T):
    """First-Fit-Decreasing: chats longest first; each goes into the FIRST row that still has room, else a new row.
    A max segment tree over the rows' free space finds that row in O(log rows). Returns a list of rows, each a list
    of chat indices. Same result as the plain loop (checked in self_test)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    size = 1
    while size < max(1, len(lengths)):
        size *= 2
    tree = [0] * (2 * size)          # leaf size + r = free slots of row r (0 = no row yet, or full)
    rows, used = [], 0
    for i in order:
        L = lengths[i]
        if tree[1] >= L:             # some open row has room: walk down to the leftmost one
            node = 1
            while node < size:
                node = 2 * node if tree[2 * node] >= L else 2 * node + 1
            r = node - size
        else:                        # no row has room: open a new one
            r = used; used += 1
            node = size + r
            tree[node] = T
            assert r < size, "more rows than chats (cannot happen)"
            rows.append([])
        rows[r].append(i)
        tree[node] -= L
        node //= 2
        while node:
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
            node //= 2
    return rows


def ffd_pack_plain(lengths, T):
    """The same rule written as the plain loop (used only by self_test to check ffd_pack)."""
    rows, free = [], []
    for i in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        for r in range(len(rows)):
            if free[r] >= lengths[i]:
                rows[r].append(i); free[r] -= lengths[i]
                break
        else:
            rows.append([i]); free.append(T - lengths[i])
    return rows


def build_packed_row(chats, T):
    """chats: list of (ids, mask) packed into one row. Each chat is labelled on its own (inputs = ids[:-1],
    labels = ids[1:] where trained), so the last target of a chat is its own end token and nothing points into the
    next chat. Returns inputs, labels, the chats' lengths in row order, trained-token count."""
    inp = np.full(T, PAD_ID, dtype=np.uint16)
    lab = np.full(T, IGNORE_ID, dtype=np.uint16)
    lens = []
    t, nvalid = 0, 0
    for ids, mask in chats:
        ci, cl, cv = build_row(ids, mask, len(ids) - 1)     # the chat alone, exactly its length (no padding)
        n = len(ids) - 1
        inp[t:t + n], lab[t:t + n] = ci, cl
        lens.append(n)
        t += n; nvalid += cv
    assert t <= T, "row overflow"
    return inp, lab, lens, nvalid                           # positions t.. T-1 (if any) are the row's pad tail


def show_row(enc, inp, lab):
    """Decode one row; the trained part (what the labels ask for) is wrapped in [[ ]].
    End tokens of earlier answers are also INPUTS, so padding is not "the first 50256": the row's real
    length is fixed by its last target (every row ends with the trained end token)."""
    n = int(np.nonzero(lab != IGNORE_ID)[0][-1]) + 1
    ids = [int(inp[0])] + [int(lab[t]) if lab[t] != IGNORE_ID else int(inp[t + 1]) for t in range(n)]
    trained = [False] + [bool(lab[t] != IGNORE_ID) for t in range(n)]
    out = []
    for is_trained, grp in itertools.groupby(zip(ids, trained), key=lambda x: x[1]):
        toks = [t for t, _ in grp]
        text, run = [], []
        for t in toks + [None]:                      # decode runs between end tokens in one go (keeps UTF-8 whole)
            if t is None or (t == PAD_ID and isinstance(enc, Gpt2Enc)):
                if run:
                    text.append(enc.decode(run)); run = []
                if t is not None:
                    text.append("<|endoftext|>")
            else:
                run.append(t)
        out.append("[[" + "".join(text) + "]]" if is_trained else "".join(text))
    return "".join(out)


def self_test(enc):
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Bye <|endoftext|>"}, {"role": "assistant", "content": "Bye!"}]
    if enc.drop_struct:                    # the HF tokenizer path drops a chat whose text holds a structural token
        assert tokenize_conversation(msgs, enc) == (None, "special_token_text")
        msgs[2]["content"] = "Bye"
    (ids, mask), err = tokenize_conversation(msgs, enc)
    assert err is None
    inp, lab, nvalid = build_row(ids, mask, 64)
    trained = [int(t) for t in lab if t != IGNORE_ID]
    if enc.template == "chatml-v1":        # <|endoftext|> first, 4 turns = 4 <|im_start|> and 4 <|im_end|>
        assert ids[0] == enc.bos and ids.count(enc.start) == 4 and ids.count(enc.end) == 4
    if isinstance(enc, HfEnc) and enc.template != "chatml-v1":   # <|endoftext|> first + one end token per answer
        assert ids[0] == enc.bos and ids.count(enc.end) == 3
        if enc.template == "tulu-special-v1":
            assert ids.count(enc.role_id["user"]) == 2 and ids.count(enc.role_id["assistant"]) == 2
        else:
            assert enc.marker("assistant") == enc.text("<|assistant|>\n") and len(enc.marker("assistant")) > 1
    expect = enc.text("Hello.") + [enc.end] + enc.text("Bye!") + [enc.end]
    assert trained == expect, (trained, expect)
    assert nvalid == len(expect)
    if isinstance(enc, Gpt2Enc):
        assert enc.decode(enc.text("<|endoftext|>")) == "<|endoftext|>"                  # stays plain text
        assert PAD_ID not in enc.text("Bye <|endoftext|>")                               # never the real id
    assert show_row(enc, inp, lab).count("[[") == 2
    print("self-test OK:", show_row(enc, inp, lab).replace("\n", "\\n"))
    # packing: the tree version gives the same rows as the plain First-Fit-Decreasing loop
    rng = np.random.default_rng(7)
    for trial in range(300):
        T_ = int(rng.choice([8, 20, 64, 4096, 131072]))
        lens = [int(x) for x in rng.integers(1, T_ + 1, size=int(rng.integers(1, 60)))]
        assert ffd_pack(lens, T_) == ffd_pack_plain(lens, T_), (T_, lens)
    # a packed row: chats back to back, each labelled alone, pad tail = its own segment
    (ids2, mask2), _ = tokenize_conversation(msgs[:2], enc)
    pin, plab, plens, pv = build_packed_row([(ids, mask), (ids2, mask2)], 128)
    n1, n2 = len(ids) - 1, len(ids2) - 1
    assert plens == [n1, n2] and (pin[n1 + n2:] == PAD_ID).all() and (plab[n1 + n2:] == IGNORE_ID).all()
    assert plab[n1 - 1] == enc.end and plab[n1 + n2 - 1] == enc.end, "each chat's last target is its own end token"
    assert pv == nvalid + int((np.asarray(mask2[1:])).sum())
    print(f"packing self-test OK: tree FFD == plain FFD on 300 random sets; packed row of chats {n1}+{n2} + pad")


def _examine(row, idx, name, enc, T, drop_sources, keep_sources, source_field):
    """One row of collect(): the source filter, the template and the length rule (a chat longer than T is dropped).
    Returns (src, reason, rid, ids, mask): reason = why the row is not used (None = used)."""
    src = row.get(source_field) or name
    if src in drop_sources or (keep_sources and src not in keep_sources):
        return src, ("dropped_source" if src in drop_sources else "not_selected_source"), None, None, None
    res, reason = tokenize_conversation(row["messages"], enc)
    # ---- The length rule: a chat longer than T is DROPPED, never cut (the user's decision, 2026-10-31) ----
    # T = the row length (--seq-len). A chat whose template text needs more than T positions is dropped (meta.json
    # drops "too_long"). It is never cut, split or truncated. This is the only method: no option, no second algorithm.
    #
    # Why (the user's reasoning):
    #  - SFT must teach the model to answer a question fully and to stop at the end token. So every training chat must
    #    be whole: every answer complete and ending with its end token. More tokens from broken chats do not help, and
    #    SFT uses little compute, so saving compute by cutting chats is not the goal.
    #  - Cutting a long answer breaks that. The first piece ends in the middle of the answer with no end token. The next
    #    piece starts in the middle of the answer, either without the question (TRL's bfd_split) or with the question
    #    repeated in front of an answer that starts halfway (our split_turns of 2026-10-30). Both confuse the model.
    #  - A multi-turn chat cannot be cut into single turns: every turn needs the turns before it. In "What is 2+2?" ->
    #    "4", then "Times 3?" -> "12", the question "Times 3?" means nothing alone.
    #  - Others: torchtitan drops the same way ("avoiding training on truncated responses"); Megatron-LM cuts the end of
    #    a row that overflows (the end token is lost: the worst case for SFT); TRL's bfd truncates, bfd_split splits.
    #  - The real fix for long chats is a longer T (after a long-context checkpoint), not cutting them.
    #
    # What we built and measured before deciding (the 10-28 mixes, T 4,096). Full record with every number:
    # sft/docs/sft_packing_algorithms_bfd_ffd.md section 7.8; SFT work doc 45.4, 45.108 parts 15-18, 45.109-45.112.
    #  - 23,253 chats are longer than 4,096 tokens (WildChat 18,653, LongAlign 3,559, smol-summarize 522, others 519).
    #    With a longer row 10,915 of them fit whole at 8,192, 19,538 at 16,384, 22,043 at 32,768. On the long-source mix,
    #    drop keeps 132,316 chats (90.7 M answer tokens) and drops 22,744 (about 102 M answer tokens).
    #  - 2026-10-30: --long-chats split_trl (TRL's bfd_split) and split_turns (ours: cut between turns, and repeat the
    #    question in front of a cut answer) were built, tested (5,000 random chats per template; on the GPU the trainer's
    #    step-0 loss equalled the numpy reference) and pushed in commit 334ecbb.
    #  - 2026-10-31: about 40 ways to keep the chats whose question is longer than half a row were studied, scored by
    #    "hidden copies": the answer writes an 8-token phrase that appears earlier in the chat but is not in its own
    #    piece, so the model learns to write text it cannot see. Per 1,000 trained tokens: TRL's bfd_split 45 (104 for
    #    long questions), split_turns 26, the best variant 12, a simpler split 32-41; a turn that starts a new piece with
    #    no earlier turns in front 92 (with them 55).
    #  - Decision 2026-10-31: drop only. The split code was removed; the data files are byte-identical to the pushed drop
    #    mode.
    #
    # The only split the user would consider later (2026-10-31), and only if needed: TRL's bfd_split exactly, with ONE
    # rule added, kept simple: if ANY prompt of the chat (the system message + the question of ANY turn, the first turn
    # or a later one) is as long as a whole row (T + 1 tokens) or longer, the WHOLE chat is dropped, single-turn and
    # multi-turn alike. Every other chat is split by bfd_split as it is: its token stream is cut into blocks of T + 1
    # tokens, nothing is added at a cut, every block is packed as a chat of its own. Examples at T = 4,096 (a row holds
    # 4,097 tokens; counts rounded, the role markers counted with the question):
    #  - Single turn: a 10,000-token question (a long document + "summarise it") and a 500-token answer: the prompt is
    #    longer than a row -> the WHOLE chat is dropped.
    #  - Multi-turn, long FIRST question: turn 1 = a 6,000-token pasted document + "what does it say?" and a 300-token
    #    answer; turn 2 = 100 + 400 tokens: turn 1's prompt is longer than a row -> the WHOLE chat is dropped, turn 2 too.
    #  - Multi-turn, long LATER question: turn 1 = 300 + 700 tokens; turn 2 = a 9,000-token pasted log + "what is
    #    wrong?" and a 400-token answer: turn 2's prompt is longer than a row -> the WHOLE chat is dropped, turn 1 too.
    #  - Multi-turn, long answer: turn 1 = a 200-token question + a 6,000-token answer; turn 2 = 300 + 500 tokens: no
    #    prompt is a whole row -> bfd_split as it is: block 1 = the question + about 3,900 answer tokens (no end token),
    #    block 2 = the other ~2,100 answer tokens (no question) + turn 2.
    # The 334ecbb split_trl (git show 334ecbb:sft/prepare_sft_data.py) is bfd_split without this rule (it dropped only
    # the blocks with no trained token and kept the rest). Even with the rule, a long answer is still cut (a block that
    # ends with no end token, a block without its question), which is why nothing is split now.
    extra = getattr(enc, "filter_extra", None)        # arm B: judge the length as arm A would
    if reason is None and len(res[0]) - 1 + (sum(extra[m["role"]] for m in row["messages"]) if extra else 0) > T:
        reason = "too_long"
    if reason:
        return src, reason, None, None, None
    rid = str(row.get("id") or row.get("prompt_id") or int(idx))
    return src, None, rid, np.asarray(res[0], dtype=np.uint16), np.asarray(res[1], dtype=bool)


_EXAMINE = {}   # the rows and settings the worker processes read (set before the pool forks)


def _examine_chunk(bounds):
    a = _EXAMINE
    return [_examine(a["ds"][int(i)], i, *a["args"]) for i in a["order"][bounds[0]:bounds[1]]]


def collect(ds, name, order, enc, T, num, skip, drop_sources, keep_sources, source_field="source", workers=1):
    """Tokenize rows in the shuffled order: leave out the first `skip` usable chats (the held-out split), keep the next
    `num` (0 = all). A chat longer than T is dropped. Returns the kept chats and the counts for meta.json.
    workers > 1: the rows are examined by that many processes, in chunks, and the results are taken back in the
    shuffled order, so skip, num and every count are applied exactly as in the serial loop (same output files)."""
    out = {"chats": [], "lengths": [], "kept_ids": [], "kept_src": collections.Counter(),
           "drops": collections.Counter(), "drop_src": collections.Counter(), "scanned": 0, "skipped": 0}
    args = (name, enc, T, drop_sources, keep_sources, source_field)
    pool = None
    if workers > 1:
        import multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"   # the Rust tokenizer must not hold threads across fork
        _EXAMINE.update(ds=ds, order=order, args=args)
        pool = multiprocessing.get_context("fork").Pool(workers)
        chunk = 2000
        records = (r for part in pool.imap(_examine_chunk, [(lo, min(lo + chunk, len(order)))
                                                            for lo in range(0, len(order), chunk)]) for r in part)
    else:
        records = (_examine(ds[int(idx)], idx, *args) for idx in order)
    kept = 0
    for src, reason, rid, ids, mask in records:
        if num > 0 and kept == num:
            break
        out["scanned"] += 1
        if reason:
            out["drops"][reason] += 1; out["drop_src"][(src, reason)] += 1
            continue
        if out["skipped"] < skip:               # held out: the validation split
            out["skipped"] += 1
            continue
        out["kept_src"][src] += 1
        kept += 1
        out["chats"].append((ids, mask))
        out["lengths"].append(len(ids) - 1)
        out["kept_ids"].append(rid)
    if pool is not None:
        pool.terminate()
        _EXAMINE.clear()
    return out


def write_split(out_dir, got, enc, T, pack, seed, show, source_meta):
    """Pack (or pad) the collected chats, check everything, write the files and meta.json, print a summary."""
    import datasets
    os.makedirs(out_dir, exist_ok=True)
    chats, lengths, kept_ids = got["chats"], got["lengths"], got["kept_ids"]
    C = len(kept_ids)                           # chats kept
    seq_lengths = None
    if pack == "none":                          # v1: one chat per row, in the sampled (random) order
        N = C
        inputs = np.empty((N, T), dtype=np.uint16)
        labels = np.empty((N, T), dtype=np.uint16)
        valid = np.empty(N, dtype=np.uint32)
        for k, (ids, mask) in enumerate(chats):
            inputs[k], labels[k], valid[k] = build_row(ids, mask, T)
        row_chats = [[k] for k in range(N)]
    else:                                       # v2: First-Fit-Decreasing, then the rows in random order
        packed = ffd_pack(lengths, T)
        perm = np.random.default_rng(seed + 2).permutation(len(packed))
        row_chats = [packed[r] for r in perm]
        N = len(row_chats)
        inputs = np.empty((N, T), dtype=np.uint16)
        labels = np.empty((N, T), dtype=np.uint16)
        valid = np.empty(N, dtype=np.uint32)
        row_lens = []
        for r, idxs in enumerate(row_chats):
            inputs[r], labels[r], lens_r, valid[r] = build_packed_row([chats[i] for i in idxs], T)
            row_lens.append(lens_r)
        seq_lengths = np.asarray([n for lr in row_lens for n in lr], dtype=np.int32)
        seq_offsets = np.zeros(N + 1, dtype=np.int64)
        seq_offsets[1:] = np.cumsum([len(lr) for lr in row_lens])
        # packing checks: every chat exactly once, rows never overflow, the lengths are the chats' own lengths in
        # row order, every chat's last position predicts its own end token (so the stored boundaries are the real
        # ones), and the rest of every row is padding that trains nothing
        flat = sorted(i for idxs in row_chats for i in idxs)
        assert flat == list(range(C)), "a chat is missing or packed twice"
        assert all(sum(lengths[i] for i in idxs) <= T for idxs in row_chats), "a row overflows"
        assert len(seq_lengths) == C and seq_offsets[-1] == C and (seq_lengths >= 1).all(), "seq_lengths count"
        for r, idxs in enumerate(row_chats):
            lr = seq_lengths[seq_offsets[r]:seq_offsets[r + 1]]
            assert list(lr) == [lengths[i] for i in idxs], f"row {r}: seq_lengths != the chats' lengths"
            ends = np.cumsum(lr)
            assert (labels[r, ends - 1] == END_ID).all(), f"row {r}: a chat's last target is not its end token"
            assert (inputs[r, ends[-1]:] == PAD_ID).all() and (labels[r, ends[-1]:] == IGNORE_ID).all(), \
                f"row {r}: the pad tail is not padding"

    # ---- checks (fail loudly: a bad file trains silently) ----
    assert inputs.max() < N_IDS <= VOCAB_ROWS, "input id >= the tokenizer's ids (or the model's rows)"
    lab_ok = (labels < N_IDS) | (labels == IGNORE_ID)
    assert lab_ok.all(), "label outside the tokenizer's ids and not the ignore marker"
    assert (valid > 0).all(), "a row with zero trained tokens (0/0 = NaN in the loss)"
    for k in range(N):  # the last trained target of every row is the end token
        last = labels[k][labels[k] != IGNORE_ID][-1]
        assert last == END_ID, f"row {k}: answer does not end with the end token"
    eos_targets = int((labels == END_ID).sum())

    inputs.tofile(os.path.join(out_dir, "inputs.bin"))
    labels.tofile(os.path.join(out_dir, "labels.bin"))
    valid.tofile(os.path.join(out_dir, "valid_counts.bin"))
    if seq_lengths is not None:
        seq_lengths.tofile(os.path.join(out_dir, "seq_lengths.bin"))
        seq_offsets.tofile(os.path.join(out_dir, "seq_offsets.bin"))
    chats_per_row = np.asarray([len(idxs) for idxs in row_chats])
    lens = np.asarray(lengths)
    hist_edges = [0, 256, 512, 1024, 2048, 3072, 4096]
    while hist_edges[-1] < T:                   # rows longer than 4,096: one more bin per doubling, up to T
        hist_edges.append(min(2 * hist_edges[-1], T))
    meta = {
        "format": {"inputs": "uint16 [N,T] little-endian, row-major", "labels": "uint16 [N,T], 65535 = ignore",
                   "valid_counts": "uint32 [N]", "label_rule": "labels = ids[1:] masked; inputs = ids[:-1]"},
        "N": N, "T": T, "pad_id": PAD_ID, "end_id": END_ID, "ignore_id": IGNORE_ID, "vocab_rows": VOCAB_ROWS,
        "tokenizer_ids": N_IDS, "bos_id": enc.bos,
        "packed": pack != "none",
        "packing": None if pack == "none" else {
            "rule": "first-fit-decreasing (max segment tree), rows shuffled with seed + 2",
            "seq_lengths": "int32: every row's chat lengths in row order (TRL's seq_lengths); row r = "
                           "seq_lengths[seq_offsets[r]:seq_offsets[r+1]]; the rest of a row is its pad tail",
            "seq_offsets": "int64 [N+1]",
            "positions": "RoPE position = index of the token inside its chat (restarts at 0 for every chat)",
            "chats": C, "rows": N, "min_rows_possible": int(-(-sum(lengths) // T)),
            "real_token_share": round(float(sum(lengths)) / (N * T), 4),
            "chats_per_row": {"mean": round(float(chats_per_row.mean()), 2), "min": int(chats_per_row.min()),
                              "max": int(chats_per_row.max())},
            "row_chats": [[kept_ids[i] for i in idxs] for idxs in row_chats]},
        "template": enc.template, "tokenizer": enc.desc,
        **source_meta,
        "rows_scanned": got["scanned"], "rows_kept": N, "chats_kept": int(sum(got["kept_src"].values())),
        "chats_left_out_for_validation": got["skipped"],
        "drops": dict(got["drops"]),
        "drops_by_source": {f"{s} | {r}": c for (s, r), c in sorted(got["drop_src"].items())},
        "kept_by_source": dict(got["kept_src"].most_common()),
        "trained_tokens": int(valid.sum()), "positions": int(N * T),
        "trained_share_of_positions": round(float(valid.sum()) / (N * T), 4),
        "end_token_targets": eos_targets,
        "length_tokens": {"mean": round(float(lens.mean()), 1), "median": int(np.median(lens)),
                          "p95": int(np.percentile(lens, 95)), "max": int(lens.max()),
                          "histogram": {f"{lo}-{hi}": int(((lens > lo) & (lens <= hi)).sum())
                                        for lo, hi in zip(hist_edges[:-1], hist_edges[1:])}},
        "kept_row_ids": kept_ids,
        "tools": {"python": sys.version.split()[0], "numpy": np.__version__, "datasets": datasets.__version__,
                  **enc.tools},
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    print(f"kept {C:,} chats of {got['scanned']:,} scanned | drops {dict(got['drops'])}")
    if seq_lengths is not None:
        print(f"packed (First-Fit-Decreasing): {C:,} chats -> {N:,} rows (minimum possible "
              f"{meta['packing']['min_rows_possible']:,}), real tokens {100 * meta['packing']['real_token_share']:.1f}% "
              f"of the slots, chats per row mean {meta['packing']['chats_per_row']['mean']} max "
              f"{meta['packing']['chats_per_row']['max']}; chat-length checks OK")
    print(f"trained tokens {meta['trained_tokens']:,} = {100 * meta['trained_share_of_positions']:.1f}% of "
          f"{N * T:,} positions | length mean {meta['length_tokens']['mean']} median "
          f"{meta['length_tokens']['median']} max {meta['length_tokens']['max']}")
    print(f"checks OK: ids < {N_IDS} (model rows {VOCAB_ROWS}), labels valid or {IGNORE_ID}, every row trains >= 1 token "
          f"and ends with the end token ({eos_targets:,} end-token targets)")
    rng = np.random.default_rng(seed + 1)
    for k in rng.choice(N, size=min(show, N), replace=False):
        tag = kept_ids[k] if seq_lengths is None else f"{len(row_chats[k])} chats"
        print(f"\n----- row {k} ({tag}): trained part in [[ ]] -----")
        print(show_row(enc, inputs[k], labels[k])[:1500])
    print(f"\nwrote {out_dir}: inputs.bin + labels.bin ({2 * N * T * 2 / 1e6:.1f} MB), valid_counts.bin, meta.json"
          + (", seq_lengths.bin + seq_offsets.bin" if seq_lengths is not None else ""))


def cached_collect(cache_dir, key_obj, fn):
    """Run fn() (a collect) once per key and keep the result in cache_dir (mix mode, --part-cache): a part shared by
    several mixes is tokenized once. The key holds the part spec, the tokenizer file hash, the template, T, the seed
    and this script's hash, so any change re-tokenizes."""
    if not cache_dir:
        return fn()
    key = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:24]
    base = os.path.join(cache_dir, key)
    if os.path.exists(base + ".json"):
        info = json.load(open(base + ".json"))
        z = np.load(base + ".npz")
        ids, msk, lens = z["ids"], z["mask"], z["lens"]
        ends = np.cumsum(lens + 1)
        starts = ends - (lens + 1)
        chats = [(ids[a:b], msk[a:b]) for a, b in zip(starts, ends)]
        print(f"  part cache hit {key}: {len(chats):,} chats")
        return {"chats": chats, "lengths": [int(x) for x in lens], "kept_ids": info["kept_ids"],
                "kept_src": collections.Counter(info["kept_src"]), "drops": collections.Counter(info["drops"]),
                "drop_src": collections.Counter({tuple(k.split(" | ", 1)): v for k, v in info["drop_src"].items()}),
                "scanned": info["scanned"], "skipped": info["skipped"]}
    got = fn()
    os.makedirs(cache_dir, exist_ok=True)
    ids = np.concatenate([c[0] for c in got["chats"]]) if got["chats"] else np.zeros(0, np.uint16)
    msk = np.concatenate([c[1] for c in got["chats"]]) if got["chats"] else np.zeros(0, bool)
    np.savez(base + ".tmp.npz", ids=ids, mask=msk, lens=np.asarray(got["lengths"], dtype=np.int64))
    os.replace(base + ".tmp.npz", base + ".npz")
    json.dump({"key": key_obj, "kept_ids": got["kept_ids"], "kept_src": dict(got["kept_src"]), "drops": dict(got["drops"]),
               "drop_src": {f"{s} | {r}": c for (s, r), c in got["drop_src"].items()}, "scanned": got["scanned"],
               "skipped": got["skipped"]}, open(base + ".json", "w"))
    return got


def merge(parts):
    """Concatenate collect() results (mix mode)."""
    out = {"chats": [], "lengths": [], "kept_ids": [], "kept_src": collections.Counter(),
           "drops": collections.Counter(), "drop_src": collections.Counter(), "scanned": 0, "skipped": 0}
    for g in parts:
        for k in ("chats", "lengths", "kept_ids"):
            out[k] += g[k]
        for k in ("kept_src", "drops", "drop_src"):
            out[k].update(g[k])
        out["scanned"] += g["scanned"]; out["skipped"] += g["skipped"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--revision", default="8049631c405ae6576f93f445c6b8166f76f5505a",
                    help="dataset commit, pinned so the data is reproducible")
    ap.add_argument("--split", default="train_sft", help="train_sft for training, test_sft for validation")
    ap.add_argument("--num", type=int, default=5000, help="rows to keep (random sample); 0 = all")
    ap.add_argument("--skip-kept", type=int, default=0,
                    help="leave out the first N usable chats of the shuffled order: the validation split made with "
                         "the same seed, filters and --num N, so train and validation never share a chat")
    ap.add_argument("--drop-source", action="append", default=[],
                    help="leave out rows whose 'source' is exactly this (repeatable), e.g. a multilingual subset")
    ap.add_argument("--mix", default=None, help="a JSON list of dataset parts; writes <out>/train and <out>/val")
    ap.add_argument("--part-cache", default=None, help="mix mode: keep each tokenized part here, reuse it across mixes")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--out", required=False)
    ap.add_argument("--show", type=int, default=2, help="decoded rows to print as a check")
    ap.add_argument("--pack", choices=["none", "ffd"], default="none",
                    help="none = v1 (one chat per row); ffd = v2 packed rows (First-Fit-Decreasing) + seq_lengths.bin")
    ap.add_argument("--tokenizer", default=None,
                    help="a Hugging Face tokenizer.json with SmolLM2's special ids (the 135M base: "
                         "/localstorage/dinesh/smol135/tokenizers/ours_49152/tokenizer.json); default: GPT-2")
    ap.add_argument("--template", default="tulu-plain-v1", choices=["tulu-plain-v1", "tulu-special-v1", "chatml-v1"],
                    help="with --tokenizer: tulu-plain-v1 (arm A), tulu-special-v1 (arm B, extended tokenizer), chatml-v1")
    ap.add_argument("--vocab-rows", type=int, default=49280,
                    help="with --tokenizer: the model's embedding rows (the 135M base: 49,280 = 49,152 ids + 128 reserved)")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes that tokenize the rows (1 = serial; any number gives the same files)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    global PAD_ID, END_ID, VOCAB_ROWS, N_IDS
    if a.tokenizer:
        enc = HfEnc(a.tokenizer, a.vocab_rows, a.template)
    else:
        assert a.template == "tulu-plain-v1", "the GPT-2 path only has tulu-plain-v1"
        enc = Gpt2Enc()
    PAD_ID, END_ID, VOCAB_ROWS, N_IDS = enc.pad, enc.end, enc.vocab_rows, enc.n_ids
    print(f"tokenizer: {enc.desc} | template {enc.template} | pad {PAD_ID} end {END_ID} | model rows {VOCAB_ROWS}")
    if a.self_test:
        self_test(enc)
        return
    T = a.seq_len
    if T % 256 or T > 131072:   # 2026-10-31: was <= 4096; the long-context tests (16,384-131,072) run on these files
        sys.exit("--seq-len must be a multiple of 256 (attention tiles) and <= 131072 (the longest row being tested)")
    if not a.out:
        sys.exit("--out is required")

    import datasets
    if not a.mix:                               # one dataset, one split (the train or the val folder)
        t0 = time.time()
        ds = datasets.load_dataset(a.dataset, split=a.split, revision=a.revision)
        order = np.random.default_rng(a.seed).permutation(len(ds))
        print(f"loaded {a.dataset}@{a.revision[:8]} {a.split}: {len(ds):,} rows in {time.time() - t0:.0f}s; sampling "
              f"{'all' if a.num <= 0 else f'{a.num:,}'}, leaving out the first {a.skip_kept:,} usable chats")
        got = collect(ds, a.dataset, order, enc, T, a.num, a.skip_kept, a.drop_source, None, workers=a.workers)
        if a.num > 0 and sum(got["kept_src"].values()) < a.num:
            print(f"WARNING: only {sum(got['kept_src'].values()):,} usable chats after scanning everything")
        write_split(a.out, got, enc, T, a.pack, a.seed, a.show,
                    {"dataset": a.dataset, "revision": a.revision, "split": a.split, "seed": a.seed,
                     "filters": {"max_tokens": T, "dropped_sources": a.drop_source, "skip_kept": a.skip_kept,
                                 "language": "none" if not a.drop_source else "by source: the dropped sources above"}})
        return

    parts = json.load(open(a.mix))
    trains, vals = [], []
    for k, part in enumerate(parts):
        t0 = time.time()
        ds = datasets.load_dataset(part["dataset"], split=part.get("split", "train"), revision=part["revision"])
        order = np.random.default_rng(a.seed).permutation(len(ds))
        keep = set(part.get("sources") or [])
        drop = list(part.get("drop_sources") or [])
        vn = int(part.get("val_num", 0))
        sf = part.get("source_field", "source")
        ckey = {"part": part, "tok": enc.desc, "template": enc.template, "T": T, "seed": a.seed,
                "script": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
        print(f"part {k}: {part['dataset']}@{part['revision'][:8]} {part.get('split', 'train')}: {len(ds):,} rows "
              f"({time.time() - t0:.0f}s) | keep {sorted(keep) or 'all sources'} | drop {drop} | val {vn:,} | "
              f"train {part.get('num', 0) or 'all'}")
        if vn:
            vals.append(cached_collect(a.part_cache, dict(ckey, role="val"),
                                       lambda: collect(ds, part["dataset"], order, enc, T, vn, 0, drop, keep, sf,
                                                       a.workers)))
        if int(part.get("num", 0)) >= 0:
            trains.append(cached_collect(a.part_cache, dict(ckey, role="train"),
                                         lambda: collect(ds, part["dataset"], order, enc, T, int(part.get("num", 0)), vn,
                                                         drop, keep, sf, a.workers)))
        print(f"  -> train {sum(trains[-1]['kept_src'].values()) if int(part.get('num', 0)) >= 0 else 0:,} chats, "
              f"val {sum(vals[-1]['kept_src'].values()) if vn else 0:,}")
    src_meta = {"mix": parts, "mix_file_sha256": hashlib.sha256(open(a.mix, "rb").read()).hexdigest(), "seed": a.seed,
                "filters": {"max_tokens": T, "rule": "per part: first val_num usable chats -> val, the rest -> train"}}
    print("\n===== train"); write_split(os.path.join(a.out, "train"), merge(trains), enc, T, a.pack, a.seed, a.show, src_meta)
    if vals:
        print("\n===== val"); write_split(os.path.join(a.out, "val"), merge(vals), enc, T, a.pack, a.seed, a.show, src_meta)


if __name__ == "__main__":
    main()
