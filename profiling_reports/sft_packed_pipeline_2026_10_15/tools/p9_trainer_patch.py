#!/usr/bin/env python3
"""P9 (2026-10-15): bluscript_sft.cpp passes packed rows to attention as the per-chat plan (design A, built once per
micro-batch from the loader's segment lists) instead of design B's seg_start tensor. RoPE positions unchanged.
Every replacement must match exactly once. usage: p9_trainer_patch.py <BluTrain_sft dir>"""
import sys, os
p = os.path.join(sys.argv[1], "bluscript_sft.cpp")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

rep("""// bluscript_sft.cpp (2026-10-15): the SFT trainer. A copy of bluscript_zero_ckpt.cpp (pretraining, unchanged) with
// v2 PACKED rows added: several whole chats per row, each attending only to itself (document masking, design B) and
// with RoPE positions restarting at 0 for every chat. Packed data = BLU_SFT_DATA/<split>/seg_start.bin (written by
// sft/prepare_sft_data.py --pack ffd); without that file a split trains exactly as v1 (one chat per row).
//
// g_packed is set by the loop before every model.forward on an SFT batch (from SftDataLoader) and read by the
// attention block: seg_start [B,T] int32 = where each token's chat starts, positions [B,T] int32 = t - seg_start.
// Empty = dense, exactly as before. There is no activation recompute, so it only has to stay valid until the
// micro-step's backward, which runs before the next batch sets it again (the backward node keeps its own copy).
// =============================================================================
struct PackedRows { Tensor seg_start, positions; };
static PackedRows g_packed;
""", """// bluscript_sft.cpp (2026-10-15): the SFT trainer. A copy of bluscript_zero_ckpt.cpp (pretraining, unchanged) with
// v2 PACKED rows added: several whole chats per row, each attending only to itself and with RoPE positions restarting
// at 0 for every chat. Packed data = BLU_SFT_DATA/<split>/seq_lengths.bin + seq_offsets.bin (the chats of every row as
// a list of lengths, TRL's seq_lengths; written by sft/prepare_sft_data.py --pack ffd); without them a split trains
// exactly as v1 (one chat per row).
//
// g_packed is set by the loop before every model.forward on an SFT batch (from SftDataLoader) and read by the
// attention block: positions [B,T] int32 = index of each token inside its chat (RoPE); chats = the per-chat attention
// plan (design A, the cu_seqlens way of FlashAttention varlen: OwnTensor::cuda::GqaChatPlan), built once per
// micro-batch from the loader's segment lists and shared by every layer's forward and backward. Empty = dense, exactly
// as before. There is no activation recompute, so it only has to stay valid until the micro-step's backward, which
// runs before the next batch sets it again (each attention backward node holds its own reference to the plan).
// (Design B, one seg_start value per token, was used here on the morning of 2026-10-15 and removed the same day.)
// =============================================================================
struct PackedRows { Tensor positions; OwnTensor::cuda::GqaChatPlanPtr chats; };
static PackedRows g_packed;
// The packed-row inputs of the batch `l` just returned: its RoPE positions, and the attention plan of its B rows for
// Hq query heads (one asynchronous upload; the host does not wait for the GPU). Empty when `l` holds v1 rows.
static PackedRows packed_rows_of(const SftDataLoader& l, int B, int T, int Hq) {
    if (!l.packed()) return PackedRows{};
    const std::vector<int32_t>& rows = l.last_segment_rows();
    return PackedRows{l.last_positions(),
                      OwnTensor::cuda::gqa_chat_plan_make(rows.data(), l.last_segment_starts().data(),
                                                          l.last_segment_lengths().data(), static_cast<int>(rows.size()),
                                                          B, T, Hq)};
}
""")
rep("""        if (g_packed.seg_start.is_valid())
            throw std::runtime_error("packed SFT rows need USE_FUSED_RMS_ROPE=1 (the sequential RoPE takes no "
                                     "per-chat positions)");""",
    """        if (g_packed.chats)
            throw std::runtime_error("packed SFT rows need USE_FUSED_RMS_ROPE=1 (the sequential RoPE takes no "
                                     "per-chat positions)");""")
rep("""            g_packed.seg_start);   // packed rows: each chat attends only to itself; empty = plain causal""",
    """            g_packed.chats);   // packed rows: each chat attends only to itself; null = plain causal""")
rep("""                g_packed = sft_val->packed() ? PackedRows{sft_val->last_seg_start(), sft_val->last_positions()}
                                             : PackedRows{};""",
    """                g_packed = packed_rows_of(*sft_val, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                          static_cast<int>(cfg.q_heads));""")
rep("""            g_packed = (sft && sft_train->packed())
                           ? PackedRows{sft_train->last_seg_start(), sft_train->last_positions()} : PackedRows{};""",
    """            g_packed = sft ? packed_rows_of(*sft_train, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                            static_cast<int>(cfg.q_heads))
                           : PackedRows{};""")
left = [l.strip() for l in s.splitlines() if "seg_start" in l]
print(left)
assert len(left) == 1 and "removed the same day" in left[0], left
open(p, "w", encoding="utf-8").write(s)
print("patched bluscript_sft.cpp")
