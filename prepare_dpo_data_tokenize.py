#!/usr/bin/env python3
"""
prepare_dpo_data_tokenize.py — Convert DPO JSONL to DPO1 binary format.

Reads raw JSONL files with chosen/rejected message pairs and:
1. Applies tulu-plain-v1 chat template (same as SFT)
2. Tokenizes with tiktoken GPT-2 (same as SFT)
3. Creates mask arrays (True=response, False=prompt)
4. Pads every sequence to T+1 (the extra slot feeds the input/target shift,
   so a record yields exactly T training positions)
5. Writes directly to DPO1 binary format (no intermediate pickle)

Output: DPO1 binary files with placeholder reference values (0.0)
        These are filled in by compute_dpo_reference_logprobs.cpp

Template: "tulu-plain-v1" (same as SFT)
  <|user|>\n{content}\n<|assistant|>\n{content}

Tokenizer: tiktoken gpt2 (same as SFT)

--seq-len MUST match the SFT checkpoint's context_length (4096 for
stage0_adamw_full), because attention is context-dependent: references scored
at one length do not transfer to another.

Usage:
  python3 prepare_dpo_data_tokenize.py \
    --in-dir /path/to/dpo_v1 \
    --out-dir /path/to/bin_files \
    --seq-len 4096

  python3 prepare_dpo_data_tokenize.py --self-test   # tokenizer + bit-packing only
"""

import argparse
import collections
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import tiktoken

PAD_ID = 50256          # Pad token (same as SFT: <|endoftext|>)
VOCAB_ROWS = 50304      # Same vocab size as SFT
TEMPLATE = "tulu-plain-v1"
ROLES = ("system", "user", "assistant")


def get_tokenizer():
    """Return tiktoken GPT-2 encoder with custom special tokens (same as SFT)."""
    base_enc = tiktoken.get_encoding("gpt2")
    assert base_enc.eot_token == PAD_ID and base_enc.n_vocab == 50257

    # Add custom special tokens (same as sft_tokenizer.py)
    USER_ID = 50257
    ASSISTANT_ID = 50258
    SYSTEM_ID = 50259

    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    SYSTEM_TOKEN = "<|system|>"

    special_tokens = dict(base_enc._special_tokens)
    special_tokens[USER_TOKEN] = USER_ID
    special_tokens[ASSISTANT_TOKEN] = ASSISTANT_ID
    special_tokens[SYSTEM_TOKEN] = SYSTEM_ID

    enc = tiktoken.Encoding(
        name="gpt2_dpo",
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens=special_tokens,
    )

    return enc


def tokenize_messages(messages, enc):
    """
    Tokenize a message sequence (chosen or rejected response).

    Returns ((ids, mask), error) where:
      - ids: token sequence [prompt tokens, response tokens]
      - mask: bool array marking which tokens are "response" (True) vs prompt (False)
      - error: None on success, error string on failure
    """
    if not messages:
        return (None, None), "empty_messages"

    if any(m.get("role") not in ROLES for m in messages):
        return (None, None), "unknown_role"

    if not any(m["role"] == "assistant" for m in messages):
        return (None, None), "no_assistant_turn"

    if messages[-1]["role"] != "assistant":
        return (None, None), "last_turn_not_assistant"

    ids, mask = [], []

    # Allow special tokens in encoding
    allowed_special = {"<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"}

    for m in messages:
        role, content = m["role"], m.get("content") or ""

        if role == "assistant" and not content.strip():
            return (None, None), "empty_assistant_content"

        # Add role marker (same as SFT) - use encode() to support special tokens
        marker = enc.encode(f"<|{role}|>\n", allowed_special=allowed_special)
        ids += marker
        mask += [False] * len(marker)  # Role markers never trained

        # Add message content
        body = enc.encode(content, allowed_special=allowed_special)
        ids += body

        # Only assistant content counts as "trained" (response)
        trained = (role == "assistant")
        mask += [trained] * len(body)

        # If assistant, add end token and mark as trained (model learns to stop)
        if trained:
            ids.append(PAD_ID)
            mask.append(True)
        else:
            # Non-assistant turns get newline (same as SFT)
            nl = enc.encode("\n", allowed_special=allowed_special)
            ids += nl
            mask += [False] * len(nl)

    return (ids, mask), None


def bit_pack_mask(mask_array):
    """Convert bool array to bit-packed bytes."""
    mask_bytes = (len(mask_array) + 7) // 8
    packed = np.zeros(mask_bytes, dtype=np.uint8)
    for i, bit in enumerate(mask_array):
        if bit:
            packed[i >> 3] |= (1 << (i & 7))
    return packed.tobytes()


def write_dpo1_pair(f, c_ids, c_mask, r_ids, r_mask):
    """Write one DPO pair to DPO1 binary file.

    Field order must match dpo::fixed_detail::ShardView::at() in
    DPODataLoaderFixed.h, which reads BOTH lengths before either payload:
        chosen_n int32 | rejected_n int32
        chosen_ids   | chosen_mask_bits
        rejected_ids | rejected_mask_bits
        4 reference values
    Interleaving a length with its payload makes the loader read the first
    token id as rejected_n and index off the end of the mapping.
    """
    # Both lengths first
    f.write(struct.pack("<i", len(c_ids)))
    f.write(struct.pack("<i", len(r_ids)))

    # Chosen payload
    f.write(np.asarray(c_ids, dtype=np.uint16).tobytes())
    f.write(bit_pack_mask(c_mask))

    # Rejected payload
    f.write(np.asarray(r_ids, dtype=np.uint16).tobytes())
    f.write(bit_pack_mask(r_mask))

    # Reference log-probs (placeholder: all zeros)
    # These will be filled in by compute_dpo_reference_logprobs.cpp
    f.write(struct.pack("<f", 0.0))      # chosen_ref_logp
    f.write(struct.pack("<I", 0))        # chosen_ref_len
    f.write(struct.pack("<f", 0.0))      # rejected_ref_logp
    f.write(struct.pack("<I", 0))        # rejected_ref_len


def self_test(enc):
    """Verify tokenization matches SFT template."""
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
    (ids, mask), err = tokenize_messages(msgs, enc)
    assert err is None, f"Self-test failed: {err}"

    # Verify structure
    assert ids[0] == 50257, "First token should be <|user|> (50257)"
    assert PAD_ID in ids, "Should end with end token (50256)"
    assert any(mask), "Should have some trained tokens"
    assert not mask[0], "First token (role marker) should not be trained"

    # Test bit packing
    test_mask = [False, True, True, False, True]
    packed = bit_pack_mask(test_mask)
    assert len(packed) == 1, "5 bits should pack to 1 byte"
    assert packed[0] == 0b00010110, f"Bit packing incorrect: {bin(packed[0])}"

    print("✓ self-test OK: tokenization and bit-packing verified")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-dir", required=False, help="Input directory with dpo_train.jsonl, dpo_val.jsonl")
    ap.add_argument("--out-dir", required=False, help="Output directory for .bin files")
    ap.add_argument("--seq-len", type=int, default=4096,
                    help="T; must equal the SFT checkpoint's context_length (pairs longer than T+1 are dropped)")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--self-test", action="store_true", help="Run self-test and exit")
    args = ap.parse_args()

    enc = get_tokenizer()

    if args.self_test:
        self_test(enc)
        return

    # After self-test, require in-dir and out-dir
    if not args.in_dir or not args.out_dir:
        ap.error("--in-dir and --out-dir are required (unless using --self-test)")

    T = args.seq_len
    if T <= 0 or T > 4096:
        sys.exit(f"--seq-len must be in (0, 4096], got {T}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Process each split
    all_stats = {
        "train": {"pairs": 0, "dropped": collections.Counter(), "sources": collections.Counter()},
        "val": {"pairs": 0, "dropped": collections.Counter(), "sources": collections.Counter()}
    }

    for split in ("train", "val"):
        jsonl_path = os.path.join(args.in_dir, f"dpo_{split}.jsonl")

        if not os.path.exists(jsonl_path):
            print(f"[SKIP] {jsonl_path} not found")
            continue

        print(f"\n[{split}] Processing {jsonl_path}...")

        bin_path = os.path.join(args.out_dir, f"{split}.bin")
        pair_count = 0
        kept_count = 0

        with open(bin_path, "wb") as f_out:
            # Write magic header
            f_out.write(b"DPO1")

            # Write placeholder count (will update later)
            count_offset = f_out.tell()
            f_out.write(struct.pack("<I", 0))  # Placeholder

            with open(jsonl_path) as f_in:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  Skipping line {line_num}: JSON parse error")
                        all_stats[split]["dropped"]["json_parse_error"] += 1
                        continue

                    pair_count += 1
                    source = ex.get("source", "unknown")

                    # Extract and validate structure
                    chosen_msgs = ex.get("chosen")
                    rejected_msgs = ex.get("rejected")

                    if not chosen_msgs or not rejected_msgs:
                        all_stats[split]["dropped"]["missing_messages"] += 1
                        continue

                    # Tokenize chosen
                    (c_ids, c_mask), c_err = tokenize_messages(chosen_msgs, enc)
                    if c_err:
                        all_stats[split]["dropped"][f"chosen_{c_err}"] += 1
                        continue

                    # Tokenize rejected
                    (r_ids, r_mask), r_err = tokenize_messages(rejected_msgs, enc)
                    if r_err:
                        all_stats[split]["dropped"][f"rejected_{r_err}"] += 1
                        continue

                    # Check lengths
                    max_len = max(len(c_ids), len(r_ids))
                    if max_len > T + 1:
                        all_stats[split]["dropped"]["too_long"] += 1
                        continue

                    # Verify token IDs are within vocab
                    if max(c_ids + r_ids) >= VOCAB_ROWS:
                        all_stats[split]["dropped"]["vocab_out_of_range"] += 1
                        continue

                    # Pad sequences to T
                    c_ids_padded = c_ids + [PAD_ID] * (T + 1 - len(c_ids))
                    r_ids_padded = r_ids + [PAD_ID] * (T + 1 - len(r_ids))
                    c_mask_padded = c_mask + [False] * (T + 1 - len(c_mask))
                    r_mask_padded = r_mask + [False] * (T + 1 - len(r_mask))

                    # Write to binary
                    write_dpo1_pair(f_out, c_ids_padded, c_mask_padded,
                                    r_ids_padded, r_mask_padded)

                    all_stats[split]["sources"][source] += 1
                    kept_count += 1

                    if kept_count % 5000 == 0:
                        print(f"  Processed {kept_count} pairs...")

            # Update count field
            f_out.seek(count_offset)
            f_out.write(struct.pack("<I", kept_count))

        # Verify file
        file_size = os.path.getsize(bin_path)
        expected_size = 8 + kept_count * (8 + (T+1)*2 + (T+1+7)//8 + 8 + (T+1)*2 + (T+1+7)//8 + 16)

        all_stats[split]["pairs"] = kept_count

        print(f"[{split}] {kept_count:,} pairs kept, {pair_count - kept_count:,} dropped")
        print(f"       output: {bin_path} ({file_size / 1024 / 1024:.1f} MB)")

        dropped_counts = all_stats[split]["dropped"]
        if dropped_counts:
            print(f"       drop reasons: {dict(dropped_counts)}")

    # Write metadata
    meta = {
        "format": {
            "type": "DPO1 binary",
            "description": "Binary format with token IDs, masks, and placeholder reference values",
            "magic": "DPO1",
            "structure_per_pair": {
                "_order": "BOTH lengths first, then payloads — must match "
                          "dpo::fixed_detail::ShardView::at() in DPODataLoaderFixed.h",
                "chosen_n": "int32",
                "rejected_n": "int32",
                "chosen_ids": "uint16[chosen_n]",
                "chosen_mask_bits": "bit-packed bool[ceil(chosen_n/8)]",
                "rejected_ids": "uint16[rejected_n]",
                "rejected_mask_bits": "bit-packed bool[ceil(rejected_n/8)]",
                "chosen_ref_logp": "float32 (PLACEHOLDER: 0.0)",
                "chosen_ref_len": "int32 (PLACEHOLDER: 0)",
                "rejected_ref_logp": "float32 (PLACEHOLDER: 0.0)",
                "rejected_ref_len": "int32 (PLACEHOLDER: 0)"
            },
            "note": "Reference values are placeholders. They will be computed and filled in by compute_dpo_reference_logprobs.cpp"
        },
        "T": args.seq_len,
        "pad_id": PAD_ID,
        "vocab_rows": VOCAB_ROWS,
        "template": TEMPLATE,
        "tokenizer": f"tiktoken gpt2 ({tiktoken.__version__})",
        "splits": {}
    }

    for split in ("train", "val"):
        if all_stats[split]["pairs"] > 0:
            meta["splits"][split] = {
                "num_pairs": all_stats[split]["pairs"],
                "dropped": dict(all_stats[split]["dropped"]),
                "sources": dict(all_stats[split]["sources"])
            }

    meta["tools"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "tiktoken": tiktoken.__version__
    }
    meta["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    meta["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Wrote {args.out_dir}/meta.json")
    print(f"\nReady for next step: compute_dpo_reference_logprobs.cpp")
    print(f"  It will read: {os.path.join(args.out_dir, '*.bin')}")
    print(f"  It will fill in the reference values and output final .bin files")


if __name__ == "__main__":
    main()
