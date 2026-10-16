#!/usr/bin/env python3
"""P5 (2026-10-15): the autograd attention op (unpacked Q/K/V, the trainer's path) takes the per-chat plan
(cuda::GqaChatPlanPtr, design A) instead of design B's seg_start tensor; the backward node keeps the plan until its
backward has run. Every replacement must match exactly once. usage: p5_autograd_patch.py <Tensor-Implementations dir>"""
import sys, os
TI = sys.argv[1]
files = {}

def load(rel):
    files[rel] = open(os.path.join(TI, rel), encoding="utf-8").read()

def rep(rel, old, new):
    n = files[rel].count(old)
    assert n == 1, (rel, n, old[:120])
    files[rel] = files[rel].replace(old, new)

A = "include/autograd/operations/AttentionOps.h"
load(A)
rep(A, """#pragma once

#include "core/Tensor.h"
""", """#pragma once

#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"   // cuda::GqaChatPlanPtr (v2 packed rows)
""")
rep(A, """  SDPBackend backend = SDPBackend::FlashAttention,
  // v2 packed rows, design B (2026-10-15): optional int32 CUDA [B, T], where the chat of each token starts
  // (contract in ops/helpers/AttentionKernels.h). Empty (default) = plain causal over the whole row, as before.
  // Kept for the backward, which must use the same array.
  const Tensor& seg_start = Tensor()
);""", """  SDPBackend backend = SDPBackend::FlashAttention,
  // v2 packed rows (2026-10-15): optional per-chat plan (design A, the cu_seqlens way; GqaChatPlan in
  // ops/helpers/AttentionKernels.h) built for this (B, T, Nq_heads). nullptr (default) = plain causal over the whole
  // row, as before. The backward node keeps the plan: the backward must use the same one.
  const cuda::GqaChatPlanPtr& chats = nullptr
);""")

O = "src/autograd/operations/AttentionOps.cpp"
load(O)
rep(O, """    SDPBackend /*backend*/,
    const Tensor& seg_start)   // v2 design B packed rows, int32 CUDA [B, T]; empty = dense
{""", """    SDPBackend /*backend*/,
    const cuda::GqaChatPlanPtr& chats)   // v2 packed rows: per-chat plan (design A); nullptr = dense
{""")
rep(O, """    // v2 packed rows, design B (2026-10-15): the same checks as fused_rms_rope's positions. The kernels refuse a
    // seg_start they cannot honour (non-Hopper, hd 64, T % 256 != 0 in the forward, Vj1 / v44 in the backward).
    const int32_t* seg_ptr = nullptr;
    if (seg_start.is_valid()) {
        const std::string w = "sdpa_flash_attention_qkv (GQA)";
        if (!is_causal)
            throw std::runtime_error(w + ": seg_start (packed rows) needs is_causal = true");
        if (!seg_start.device().is_cuda() || seg_start.device().index != query.device().index)
            throw std::runtime_error(w + ": seg_start must be a CUDA tensor on the query's device");
        if (seg_start.dtype() != Dtype::Int32)
            throw std::runtime_error(w + ": seg_start must be int32");
        if (seg_start.ndim() != 2 || seg_start.shape().dims[0] != B || seg_start.shape().dims[1] != T)
            throw std::runtime_error(w + ": seg_start must have shape [B, T] = [" + std::to_string(B) + ", " +
                                     std::to_string(T) + "]");
        if (!seg_start.is_contiguous())
            throw std::runtime_error(w + ": seg_start must be contiguous");
        seg_ptr = seg_start.data<int32_t>();
    }
""", """    // v2 packed rows (design A, 2026-10-15): the kernels check the plan's shape (B, T, Nq_heads) and refuse a plan
    // they cannot honour (non-Hopper, hd 64, T % 256 != 0 in the forward, Vj1 / v44 in the backward).
    if (chats && !is_causal)
        throw std::runtime_error("sdpa_flash_attention_qkv (GQA): a packed-rows plan needs is_causal = true");
""")
rep(O, """        static_cast<int>(T), static_cast<int>(hd),
        is_causal, seg_ptr);""", """        static_cast<int>(T), static_cast<int>(hd),
        is_causal, chats.get());""")
rep(O, """            B, Nq_heads, Nkv_heads, T, hd, is_causal, seg_start);""",
    """            B, Nq_heads, Nkv_heads, T, hd, is_causal, chats);""")
rep(O, """  SDPBackend backend,
  const Tensor& seg_start
){""", """  SDPBackend backend,
  const cuda::GqaChatPlanPtr& chats
){""")
rep(O, """      return sdpa_flash_attention_qkv(query, key, value, Nq_heads, Nkv_heads, is_causal, dropout_p, backend,
                                      seg_start);""", """      return sdpa_flash_attention_qkv(query, key, value, Nq_heads, Nkv_heads, is_causal, dropout_p, backend,
                                      chats);""")

BH = "include/autograd/backward/AttentionBackward.h"
load(BH)
rep(BH, """#include "autograd/Node.h"
#include "core/Tensor.h"
""", """#include "autograd/Node.h"
#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"   // cuda::GqaChatPlanPtr (v2 packed rows)
""")
rep(BH, """    Tensor seg_start_;   // v2 design B packed rows (int32 [B, T]); empty = dense. The SAME array as the forward.""",
    """    cuda::GqaChatPlanPtr chats_;   // v2 packed rows (design A); null = dense. The SAME plan as the forward.""")
rep(BH, """        bool is_causal, const Tensor& seg_start = Tensor());""",
    """        bool is_causal, cuda::GqaChatPlanPtr chats = nullptr);""")
rep(BH, """        seg_start_    = Tensor();""", """        chats_.reset();""")

BC = "src/autograd/backward/AttentionBackward.cpp"
load(BC)
rep(BC, """    bool is_causal, const Tensor& seg_start)""", """    bool is_causal, cuda::GqaChatPlanPtr chats)""")
rep(BC, """      is_causal_(is_causal), seg_start_(seg_start) {}""", """      is_causal_(is_causal), chats_(std::move(chats)) {}""")
rep(BC, """        seg_start_.is_valid() ? seg_start_.data<int32_t>() : nullptr);   // v2 design B packed rows, else dense""",
    """        chats_.get());   // v2 packed rows (design A), else dense""")

for rel, s in files.items():
    assert "seg_start" not in s, (rel, [l for l in s.splitlines() if "seg_start" in l])
    open(os.path.join(TI, rel), "w", encoding="utf-8").write(s)
    print("patched", rel)
