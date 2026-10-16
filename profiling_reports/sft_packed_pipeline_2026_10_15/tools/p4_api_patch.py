#!/usr/bin/env python3
"""P4 (2026-10-15): the attention API keeps only the per-chat plan (design A). AttentionKernels.h (plan struct +
shared handle, seg_start removed), GQA_fwd_generic.cu (dispatcher, plan builder with Vj2 lists), GQA_bwd_generic.cu
(dispatcher). Every replacement must match exactly once. usage: p4_api_patch.py <Tensor-Implementations dir>"""
import sys, os
TI = sys.argv[1]
files = {}

def load(rel):
    files[rel] = open(os.path.join(TI, rel), encoding="utf-8").read()

def rep(rel, old, new):
    n = files[rel].count(old)
    assert n == 1, (rel, n, old[:120])
    files[rel] = files[rel].replace(old, new)

def cut(rel, start, end, new):
    s = files[rel]
    assert s.count(start) == 1 and s.count(end) == 1, (rel, s.count(start), s.count(end), start[:80], end[:80])
    a = s.index(start); b = s.index(end)
    assert a < b
    files[rel] = s[:a] + new + s[b:]

# =====================================================================================================================
H = "include/ops/helpers/AttentionKernels.h"
load(H)
rep(H, """#include <cstdint>
#include "dtype/Types.h"   // bfloat16_t (host+device safe)""", """#include <cstdint>
#include <memory>
#include "dtype/Types.h"   // bfloat16_t (host+device safe)""")
rep(H, """/**
* v2 packed rows, design A (per-segment work lists, the cu_seqlens way of FlashAttention varlen,
* TE/Megatron THD and torchtitan VarlenInnerAttention; added 2026-10-10). The same packed
* [B, H, T, D] rows as design B, but the kernels cut their work per segment instead of per row.
* Build ONE plan per batch from the segments of every row -- chats AND each row's pad tail as its
* own segment; together they must cover every token of every row exactly once (checked, throws) --
* and pass the same plan to gqa_flash_attn_forward_qkv and gqa_flash_attn_backward_qkv.
* The plan owns device memory: free it with gqa_chat_plan_free.
*/
struct GqaChatPlan {
    int32_t* fwd_units = nullptr;   // device, 4 ints per forward unit: {b, t0, L, (cpl << 8) | hq}
    int      n_fwd_units = 0;
    int32_t* bwd_tiles = nullptr;   // device, 4 ints per backward key tile: {b, t0, L, kt}
    int      n_bwd_tiles = 0;
    int B = 0, S = 0, Hq = 0;       // the shape the plan was built for (checked at every call)
};
// seg_b/seg_t0/seg_len: HOST arrays of n_seg segments (row, start, length). Hq <= 255.
void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq);
void gqa_chat_plan_free(GqaChatPlan& plan);""", """/**
* v2 packed rows (added 2026-10-10): several whole chats per row of T tokens, each attending only to
* itself. Given as a per-chat list -- design A, the cu_seqlens way of FlashAttention varlen,
* TE/Megatron THD, PyTorch varlen_attn, TRL and torchtitan VarlenInnerAttention: the kernels cut
* their work per segment instead of per row. (Design B, one seg_start value per token, was removed
* on 2026-10-15: A is faster and one representation keeps one tested path.)
* Build ONE plan per batch from the segments of every row -- chats AND each row's pad tail as its
* own segment; together they must cover every token of every row exactly once (checked, throws) --
* and pass the same plan to gqa_flash_attn_forward_qkv and gqa_flash_attn_backward_qkv.
* The plan owns device memory: free it with gqa_chat_plan_free (or hold it as a GqaChatPlanPtr).
* RoPE positions are not part of the plan: the caller gives fused_rms_rope positions that restart
* at 0 for every segment (as TRL's collator builds position_ids from the packed seq_lengths).
*/
struct GqaChatPlan {
    int32_t* fwd_units = nullptr;   // device, 4 ints per forward unit (v62): {b, t0, L, (cpl << 8) | hq}
    int      n_fwd_units = 0;
    int32_t* bwd_tiles = nullptr;   // device, 4 ints per Vj1d backward key tile (64 keys): {b, t0, L, kt}
    int      n_bwd_tiles = 0;
    int32_t* vj2_blocks = nullptr;  // device, 8 ints per Vj2 backward block (128 keys x one query head):
                                    //   {b, t0, L, kt, hq, dqt0, 0, 0}, dqt0 = padded dQ tile of its first step
    int      n_vj2_blocks = 0;
    int32_t* dq_tiles = nullptr;    // device, 4 ints per padded 64-row dQ tile of ONE query head (the same list
    int      n_dq_tiles = 0;        //   for every head): {b, row0, nrows, 0} (GQA_bwd_vj2_sm90.cu, bwd_prep_vj2_chats)
    int B = 0, S = 0, Hq = 0;       // the shape the plan was built for (checked at every call)
    int32_t* storage = nullptr;     // the ONE device allocation the four lists point into
};
// seg_b/seg_t0/seg_len: HOST arrays of n_seg segments (row, start, length). Hq <= 255.
// The lists go up with one asynchronous copy on the current stream: the call does not wait for the GPU.
void gqa_chat_plan_build(GqaChatPlan& plan, const int32_t* seg_b, const int32_t* seg_t0,
                         const int32_t* seg_len, int n_seg, int B, int S, int Hq);
void gqa_chat_plan_free(GqaChatPlan& plan);
// The plan as a shared handle (2026-10-15): the autograd attention op keeps it until the backward of the
// micro-step it was built for has run; it is freed when the last holder lets go.
using GqaChatPlanPtr = std::shared_ptr<const GqaChatPlan>;
GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
                                  int n_seg, int B, int S, int Hq);""")
rep(H, """*
* seg_start (v2 packed rows, design B, added 2026-10-10): optional DEVICE int32
* [B, T]. Each row holds several chats; seg_start[b*T + t] is where the segment of
* token t starts inside row b (a chat, or the row's pad tail as its own segment).
* Token i then only attends keys j with seg_start[i] <= j <= i. Contract, checked by
* whoever builds the array: seg_start[t] <= t, and it equals t (a segment starts at t)
* or seg_start[t-1]. nullptr (default) = plain causal over the whole row, exactly as
* before. A non-null seg_start is only accepted on the Hopper v62 path (causal,
* hd 128, T % 256 == 0); anywhere else this throws instead of ignoring it.
* The backward takes the SAME seg_start (gqa_flash_attn_backward_qkv, added in
* Step 4 on 2026-10-10); pass it to both, or the gradients mix the chats again.
*/""", """*
* chats (v2 packed rows, design A): optional per-chat plan built for this (B, T, Nq_heads)
* (see GqaChatPlan). nullptr (default) = plain causal over the whole row, exactly as before.
* A plan is only accepted on the Hopper v62 path (causal, hd 128, T % 256 == 0); anywhere
* else this throws instead of ignoring it. The backward takes the SAME plan; pass it to both,
* or the gradients mix the chats again.
*/""")
rep(H, """   bool is_causal,
   const int32_t*    seg_start = nullptr,  // v2 design B: [B, T] device, see above
   const GqaChatPlan* chats = nullptr      // v2 design A: per-segment plan (never with seg_start)
);""", """   bool is_causal,
   const GqaChatPlan* chats = nullptr      // v2 packed rows (design A): per-chat plan, see above
);""")
rep(H, """*
* seg_start (v2 packed rows, design B, added 2026-10-10): optional DEVICE int32
* [B, T], the same array the forward got (see gqa_flash_attn_forward_qkv for the
* contract); output and lse must come from that packed forward. nullptr (default)
* = plain causal, exactly as before. A non-null seg_start is only accepted on the
* Hopper Vj2 (default since 2026-10-15) / Vj1d (USE_VJ2=0) paths (causal, hd 128,
* T % 64 == 0, neither USE_FLASH=1 nor USE_DETERMINISTIC_DQ=1); anywhere else this
* throws instead of ignoring it. A design-A plan (chats) always runs on Vj1d.
*/""", """*
* chats (v2 packed rows, design A): the SAME plan the forward got; output and lse must
* come from that packed forward. nullptr (default) = plain causal, exactly as before.
* A plan is only accepted on the Hopper Vj2 (default since 2026-10-15; T % 128 == 0, else
* Vj1d runs) / Vj1d (USE_VJ2=0) paths (causal, hd 128, T % 64 == 0, neither USE_FLASH=1
* nor USE_DETERMINISTIC_DQ=1); anywhere else this throws instead of ignoring it.
*/""")
rep(H, """   bool is_causal,
   const int32_t*    seg_start = nullptr,  // v2 design B: [B, T] device, see above
   const GqaChatPlan* chats = nullptr      // v2 design A: the SAME plan the forward got
);""", """   bool is_causal,
   const GqaChatPlan* chats = nullptr      // v2 packed rows (design A): the SAME plan the forward got
);""")

# =====================================================================================================================
F = "src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu"
load(F)
rep(F, """#include "device/CachingCudaAllocator.h"   // v2 design A: the plan's device lists""",
    """#include "device/CachingCudaAllocator.h"   // v2 design A: the plan's device lists
#include "device/DeviceCore.h"             // getCurrentStream(): the plan's upload""")
rep(F, """// v2 (2026-10-10): seg_start != nullptr selects the packed-rows variant (design B);
// chat_units != nullptr (n_units entries) the per-segment variant (design A).
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* seg_start, const int32_t* chat_units, int n_units);""",
    """// v2 (2026-10-10): chat_units != nullptr (n_units entries) selects the per-segment variant (design A).
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units);""")
rep(F, """                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   /*seg_start=*/nullptr, /*chat_units=*/nullptr, 0);""",
    """                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   /*chat_units=*/nullptr, 0);""")
rep(F, """    bool is_causal, const int32_t* seg_start, const GqaChatPlan* chats)
{""", """    bool is_causal, const GqaChatPlan* chats)
{""")
rep(F, """    // v2 packed rows (design B, 2026-10-10): only the Hopper v62 kernel knows seg_start. Every
    // other path below would IGNORE it and silently let the chats of a row see each other, so a
    // seg_start that cannot be honoured is refused here instead.
    if (seg_start != nullptr && !v62_ok) {
        throw std::runtime_error(
            "gqa_flash_attn_forward_qkv: seg_start (packed rows) needs the Hopper v62 path: "
            "causal, head_dim 128, T % 256 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset");
    }
    // v2 design A (per-segment plan): the same rule, and a plan is only valid for the shape it was
    // built for (its units hold row indices, head indices and segment bounds of that shape).
    if (chats != nullptr) {
        if (seg_start != nullptr)
            throw std::runtime_error("gqa_flash_attn_forward_qkv: pass seg_start (design B) OR chats "
                                     "(design A), not both");
        if (!v62_ok)""", """    // v2 packed rows (design A, a per-chat plan): only the Hopper v62 kernel knows the plan. Every
    // other path below would IGNORE it and silently let the chats of a row see each other, so a plan
    // that cannot be honoured is refused here instead. A plan is also only valid for the shape it was
    // built for (its units hold row indices, head indices and segment bounds of that shape).
    if (chats != nullptr) {
        if (!v62_ok)""")
rep(F, """                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0, seg_start,
                                   chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0);""",
    """                                   B, Nq_heads, Nkv_heads, T, scale, /*stream=*/0,
                                   chats ? chats->fwd_units : nullptr, chats ? chats->n_fwd_units : 0);""")
# ---- the plan builder: + Vj2 lists, one allocation, asynchronous upload, shared handle
rep(F, """// (consecutive units share K/V in L2). Backward tiles {b, t0, L, kt}: one per 64-key tile of each
// segment, most q-tiles to visit first. Both lists are uploaded once; the kernels only read them.
// ---------------------------------------------------------------------------
void gqa_chat_plan_build(""", """// (consecutive units share K/V in L2). Backward tiles {b, t0, L, kt}: one per 64-key tile of each
// segment, most q-tiles to visit first (Vj1d).
// Vj2 (added 2026-10-15): blocks {b, t0, L, kt, hq, dqt0, 0, 0}, one per (128-key tile of a segment, query head),
// in the dense Vj2 grid's order -- sections of gqa_vj2_section_pairs(S) (batch, head) pairs, inside a section the
// blocks with the most 64-row steps first (FA3 SingleTileBwdLPTScheduler); for rows of ONE segment each this is
// exactly the dense order. dq tiles {b, row0, nrows, 0}: the padded 64-row slots of Vj2's LSE / D / fp32 dQ,
// ceil(L/64) per segment in (row, start) order, the same list for every head (FA3 varlen's offset_padded: a chat
// starts at any token, but those buffers need 64-row alignment); dqt0 = hq * n_dq_tiles + the segment's first
// slot + 2 kt. All four lists go up in ONE allocation with one asynchronous copy on the current stream: the plan
// is built once per micro-batch, and a synchronous copy would first wait for the GPU to finish the previous one.
// The kernels only read the lists.
// ---------------------------------------------------------------------------
int gqa_vj2_section_pairs(int S);   // GQA_bwd_vj2_sm90.cu

void gqa_chat_plan_build(""")
rep(F, """    std::stable_sort(kts.begin(), kts.end(), [](const KT& a, const KT& c) { return a.work > c.work; });
    for (const KT& k : kts) { bt.push_back(k.b); bt.push_back(k.t0); bt.push_back(k.L); bt.push_back(k.kt); }
    gqa_chat_plan_free(plan);
    auto& alloc = CachingCUDAAllocator::instance();
    plan.fwd_units = static_cast<int32_t*>(alloc.allocate(fu.size() * sizeof(int32_t)));
    plan.bwd_tiles = static_cast<int32_t*>(alloc.allocate(bt.size() * sizeof(int32_t)));
    if (cudaMemcpy(plan.fwd_units, fu.data(), fu.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(plan.bwd_tiles, bt.data(), bt.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("gqa_chat_plan_build: upload failed");
    plan.n_fwd_units = (int)(fu.size() / 4);
    plan.n_bwd_tiles = (int)(bt.size() / 4);
    plan.B = B; plan.S = S; plan.Hq = Hq;
}

void gqa_chat_plan_free(GqaChatPlan& plan)
{
    auto& alloc = CachingCUDAAllocator::instance();
    if (plan.fwd_units) alloc.deallocate(plan.fwd_units);
    if (plan.bwd_tiles) alloc.deallocate(plan.bwd_tiles);
    plan = GqaChatPlan{};
}""", """    std::stable_sort(kts.begin(), kts.end(), [](const KT& a, const KT& c) { return a.work > c.work; });
    for (const KT& k : kts) { bt.push_back(k.b); bt.push_back(k.t0); bt.push_back(k.L); bt.push_back(k.kt); }

    // Vj2: the padded dQ slots (segments in (row, start) order) ...
    std::vector<int32_t> dq;
    std::vector<int> slot0(byrow.size()), rowfirst((size_t)B + 1, 0);
    int nd = 0;
    for (size_t j = 0; j < byrow.size(); ++j) {
        const Seg& g = byrow[j];
        slot0[j] = nd;
        for (int q = 0; 64 * q < g.L; ++q) {
            dq.push_back(g.b); dq.push_back(g.t0 + 64 * q); dq.push_back(std::min(64, g.L - 64 * q)); dq.push_back(0);
            ++nd;
        }
        rowfirst[(size_t)g.b + 1] = (int)j + 1;
    }
    for (int b = 0; b < B; ++b) rowfirst[(size_t)b + 1] = std::max(rowfirst[(size_t)b + 1], rowfirst[b]);
    // ... and the blocks, section by section
    struct VB { int b, t0, L, kt, hq, dqt0, work; };
    std::vector<int32_t> vb;
    std::vector<VB> sec;
    const int sw = gqa_vj2_section_pairs(S), HB = B * Hq;
    for (int h0 = 0; h0 < HB; h0 += sw) {
        sec.clear();
        for (int hb = h0; hb < std::min(HB, h0 + sw); ++hb) {
            const int b = hb / Hq, hq = hb % Hq;
            for (int j = rowfirst[b]; j < rowfirst[(size_t)b + 1]; ++j) {
                const Seg& g = byrow[j];
                const int nq = (g.L + 63) / 64;
                for (int kt = 0; 128 * kt < g.L; ++kt)
                    sec.push_back({g.b, g.t0, g.L, kt, hq, hq * nd + slot0[j] + 2 * kt, nq - 2 * kt});
            }
        }
        std::stable_sort(sec.begin(), sec.end(), [](const VB& a, const VB& c) { return a.work > c.work; });
        for (const VB& v : sec) for (int x : {v.b, v.t0, v.L, v.kt, v.hq, v.dqt0, 0, 0}) vb.push_back(x);
    }

    std::vector<int32_t> all;   // every list is a multiple of 4 ints, so each one starts 16-B aligned (int4 loads)
    all.reserve(fu.size() + bt.size() + vb.size() + dq.size());
    for (const std::vector<int32_t>* v : {&fu, &bt, &vb, &dq}) all.insert(all.end(), v->begin(), v->end());
    gqa_chat_plan_free(plan);
    auto& alloc = CachingCUDAAllocator::instance();
    int32_t* buf = static_cast<int32_t*>(alloc.allocate(all.size() * sizeof(int32_t)));
    // Pageable source: the copy has read it when the call returns, so `all` may go out of scope.
    if (cudaMemcpyAsync(buf, all.data(), all.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
                        getCurrentStream()) != cudaSuccess) {
        alloc.deallocate(buf);
        throw std::runtime_error("gqa_chat_plan_build: upload failed");
    }
    plan.storage      = buf;
    plan.fwd_units    = buf;
    plan.bwd_tiles    = plan.fwd_units + fu.size();
    plan.vj2_blocks   = plan.bwd_tiles + bt.size();
    plan.dq_tiles     = plan.vj2_blocks + vb.size();
    plan.n_fwd_units  = (int)(fu.size() / 4);
    plan.n_bwd_tiles  = (int)(bt.size() / 4);
    plan.n_vj2_blocks = (int)(vb.size() / 8);
    plan.n_dq_tiles   = nd;
    plan.B = B; plan.S = S; plan.Hq = Hq;
}

void gqa_chat_plan_free(GqaChatPlan& plan)
{
    if (plan.storage) CachingCUDAAllocator::instance().deallocate(plan.storage);
    plan = GqaChatPlan{};
}

GqaChatPlanPtr gqa_chat_plan_make(const int32_t* seg_b, const int32_t* seg_t0, const int32_t* seg_len,
                                  int n_seg, int B, int S, int Hq)
{
    std::unique_ptr<GqaChatPlan> p(new GqaChatPlan());
    gqa_chat_plan_build(*p, seg_b, seg_t0, seg_len, n_seg, B, S, Hq);
    return GqaChatPlanPtr(p.release(), [](const GqaChatPlan* q) {
        GqaChatPlan* m = const_cast<GqaChatPlan*>(q);
        gqa_chat_plan_free(*m);
        delete m;
    });
}""")

# =====================================================================================================================
G = "src/Kernels/cuda/attention/arch/GQA_bwd_generic.cu"
load(G)
rep(G, """// This is the DEFAULT Hopper backward. seg_start (v2 design B, added 2026-10-10):
// packed rows, or nullptr for plain causal; only this kernel accepts it.
void gqa_vj1d_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* seg_start, const int32_t* chat_tiles, int n_chat_tiles);""",
    """// The default Hopper backward until 2026-10-15 (now USE_VJ2=0). chat_tiles = a GqaChatPlan's
// bwd_tiles (packed rows, design A), or nullptr for plain causal.
void gqa_vj1d_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_tiles, int n_chat_tiles);""")
rep(G, """// The default Hopper backward since 2026-10-15 (USE_VJ2=0 selects Vj1d); dense rows and packed rows of design B
// (seg_start, nullptr = dense). Design A plans stay on Vj1d.
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale, const int32_t* seg_start);""",
    """// The default Hopper backward since 2026-10-15 (USE_VJ2=0 selects Vj1d); dense rows, and packed rows as a
// GqaChatPlan's Vj2 lists (design A; nullptr = dense).
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_blocks, int n_chat_blocks, const int32_t* dq_tiles, int n_dq_tiles);""")
rep(G, """//                             S % 128 == 0 (else Vj1d runs). Packed rows: design B
//                             (seg_start) on Vj2, design A (chat plans) on Vj1d.
//   USE_VJ2=0        Vj1d  -- the default until 2026-10-15; bit-identical dK/dV
//                             (cuDNN-matching); dense, design A and design B""",
    """//                             S % 128 == 0 (else Vj1d runs). Dense rows and packed
//                             rows (a per-chat plan, design A).
//   USE_VJ2=0        Vj1d  -- the default until 2026-10-15; bit-identical dK/dV
//                             (cuDNN-matching); dense rows and packed rows (design A)""")
rep(G, """// The modes that honour packed rows: Vj2 (design B itself, design A through Vj1d) and Vj1d (both designs).""",
    """// The modes that honour packed rows (a per-chat plan): Vj2 and Vj1d.""")
rep(G, """// seg_start / chats reach only the Vj2 and Vj1d cases: the callers refuse them unless
// gqa_bwd_mode_packs(), so Vj1 / v44 never see packed rows.""",
    """// chats reach only the Vj2 and Vj1d cases: the callers refuse a plan unless gqa_bwd_mode_packs(),
// so Vj1 / v44 never see packed rows.""")
rep(G, """    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* seg_start, const GqaChatPlan* chats) {
    switch (gqa_bwd_mode()) {""", """    int B, int Hq, int Hkv, int G, int S, float scale,
    const GqaChatPlan* chats) {
    switch (gqa_bwd_mode()) {""")
rep(G, """            // Vj2 takes 128 keys per block. S % 128 == 64 stays on Vj1d: the forward's fast path needs S % 256 == 0
            // (gqa_v62_shape_ok), so training never sends such an S, and the tests (forward + backward) cannot reach
            // Vj2's half-size last block -- it is not run until it can be tested. Design A (chats) also stays on
            // Vj1d until Vj2 has it. Both fallbacks pass the packing through (before 2026-10-15 the S % 128 one
            // dropped it, which was unreachable only because the caller refused packed rows in this mode).
            if (S % 128 == 0 && chats == nullptr) {
                gqa_vj2_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,seg_start);
                return;
            }
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,seg_start,
                                            chats ? chats->bwd_tiles : nullptr, chats ? chats->n_bwd_tiles : 0);
            return;
        case GqaBwdMode::Vj1d:
        default:
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,seg_start,
                                            chats ? chats->bwd_tiles : nullptr, chats ? chats->n_bwd_tiles : 0);
            return;""", """            // Vj2 takes 128 keys per block. S % 128 == 64 stays on Vj1d: the forward's fast path needs S % 256 == 0
            // (gqa_v62_shape_ok), so training never sends such an S, and the tests (forward + backward) cannot reach
            // Vj2's half-size last block -- it is not run until it can be tested. The fallback passes the plan
            // through (Vj1d has design A too).
            if (S % 128 == 0) {
                gqa_vj2_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,
                                               chats ? chats->vj2_blocks : nullptr, chats ? chats->n_vj2_blocks : 0,
                                               chats ? chats->dq_tiles : nullptr, chats ? chats->n_dq_tiles : 0);
                return;
            }
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,
                                            chats ? chats->bwd_tiles : nullptr, chats ? chats->n_bwd_tiles : 0);
            return;
        case GqaBwdMode::Vj1d:
        default:
            gqa_vj1d_causal_bwd_launch_d128(Q,K,V,O,dO,lse,dQ,dK,dV,B,Hq,Hkv,G,S,scale,
                                            chats ? chats->bwd_tiles : nullptr, chats ? chats->n_bwd_tiles : 0);
            return;""")
rep(G, """            B, Nq_heads, Nkv_heads, G, T, scale, /*seg_start=*/nullptr, /*chats=*/nullptr);""",
    """            B, Nq_heads, Nkv_heads, G, T, scale, /*chats=*/nullptr);""")
rep(G, """    bool is_causal, const int32_t* seg_start, const GqaChatPlan* chats)
{""", """    bool is_causal, const GqaChatPlan* chats)
{""")
rep(G, """    // v2 packed rows (design B, 2026-10-10): only the Hopper Vj2 (default since 2026-10-15) and Vj1d
    // kernels know seg_start. Vj1 (USE_FLASH=1), v44 (USE_DETERMINISTIC_DQ=1) and every non-Hopper path
    // below would IGNORE it and silently let the chats of a row see each other, so a seg_start that
    // cannot be honoured is refused here instead (same rule as gqa_flash_attn_forward_qkv).
    if (seg_start != nullptr && !(hopper_ok && gqa_bwd_mode_packs())) {
        throw std::runtime_error(
            "gqa_flash_attn_backward_qkv: seg_start (packed rows) needs the Hopper Vj2/Vj1d path: "
            "causal, head_dim 128, T % 64 == 0, an H100/H200, BLUTRAIN_NO_SM90[_GQA] unset, "
            "and neither USE_FLASH=1 nor USE_DETERMINISTIC_DQ=1");
    }
    // v2 design A: the same rule for a plan, which must be the one the forward got (same shape).
    if (chats != nullptr) {
        if (seg_start != nullptr)
            throw std::runtime_error("gqa_flash_attn_backward_qkv: pass seg_start (design B) OR chats "
                                     "(design A), not both");
        if (!(hopper_ok && gqa_bwd_mode_packs()))
            throw std::runtime_error(
                "gqa_flash_attn_backward_qkv: chats (packed rows, design A) needs the Hopper Vj1d path "
                "(reached by default or with USE_VJ2=0): causal, head_dim 128, T % 64 == 0, an H100/H200, "
                "BLUTRAIN_NO_SM90[_GQA] unset, and neither USE_FLASH=1 nor USE_DETERMINISTIC_DQ=1");
        if (chats->B != B || chats->S != T || chats->Hq != Nq_heads || chats->bwd_tiles == nullptr)""",
    """    // v2 packed rows (design A, a per-chat plan): only the Hopper Vj2 (default since 2026-10-15) and Vj1d
    // kernels know the plan. Vj1 (USE_FLASH=1), v44 (USE_DETERMINISTIC_DQ=1) and every non-Hopper path
    // below would IGNORE it and silently let the chats of a row see each other, so a plan that cannot be
    // honoured is refused here instead (same rule as gqa_flash_attn_forward_qkv); it must also be the one
    // the forward got (same shape).
    if (chats != nullptr) {
        if (!(hopper_ok && gqa_bwd_mode_packs()))
            throw std::runtime_error(
                "gqa_flash_attn_backward_qkv: chats (packed rows, design A) needs the Hopper Vj2 (default) or "
                "Vj1d (USE_VJ2=0) path: causal, head_dim 128, T % 64 == 0, an H100/H200, "
                "BLUTRAIN_NO_SM90[_GQA] unset, and neither USE_FLASH=1 nor USE_DETERMINISTIC_DQ=1");
        if (chats->B != B || chats->S != T || chats->Hq != Nq_heads || chats->bwd_tiles == nullptr ||
            chats->vj2_blocks == nullptr || chats->dq_tiles == nullptr)""")
rep(G, """            B, Nq_heads, Nkv_heads, G, T, scale, seg_start, chats);""",
    """            B, Nq_heads, Nkv_heads, G, T, scale, chats);""")

for rel, s in files.items():
    left = [l.strip()[:120] for l in s.splitlines() if "seg_start" in l]
    print(rel, "seg_start lines left:", left)
for rel, s in files.items():
    open(os.path.join(TI, rel), "w", encoding="utf-8").write(s)
    print("patched", rel)
