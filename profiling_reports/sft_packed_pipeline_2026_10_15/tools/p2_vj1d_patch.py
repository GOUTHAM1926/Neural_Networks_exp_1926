#!/usr/bin/env python3
"""P2 (2026-10-15): Vj1d backward keeps dense + the per-chat list (design A); design B (per-token seg_start) is
removed. Every replacement must match exactly once. usage: p2_vj1d_patch.py <Tensor-Implementations dir>"""
import sys, os
TI = sys.argv[1]
p = os.path.join(TI, "src/Kernels/cuda/attention/arch/GQA_bwd_vj1d_sm90.cu")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

def cut(start, end, new=""):
    """replace s[start_marker : end_marker) (both unique) with new"""
    global s
    assert s.count(start) == 1 and s.count(end) == 1, (s.count(start), s.count(end), start[:80], end[:80])
    a = s.index(start); b = s.index(end)
    assert a < b
    s = s[:a] + new + s[b:]

# ---- fused_p_stsm: the SEG variant goes
rep("""// SEG (v2 design B, packed rows; added 2026-10-10): besides the causal mask, also hide every key that sits
// BEFORE the query row's segment start (sSS[r] = seg_start of local row r), so a chat never sees an earlier
// chat of the same packed row. Only instantiated with MASK = true. With SEG = false (the default, and the
// only value the dense kernel uses) the function is exactly what it was before.
template<int Bc, bool MASK, bool SEG = false>
__device__ __forceinline__ void fused_p_stsm(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale2, const int32_t* sSS = nullptr)
{""", """template<int Bc, bool MASK>
__device__ __forceinline__ void fused_p_stsm(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale2)
{""")
rep("""    int ss0 = 0, ss1 = 0;
    if constexpr (SEG) { ss0 = sSS[r0]; ss1 = sSS[r1]; }
""", "")
rep("""        if constexpr (SEG) {                     // same select pattern: key before the row's segment -> P = 0
            const int gc0 = k_row0 + nt*8 + cc, gc1 = gc0 + 1;
            e00 = (gc0 < ss0) ? 0.f : e00;
            e01 = (gc1 < ss0) ? 0.f : e01;
            e10 = (gc0 < ss1) ? 0.f : e10;
            e11 = (gc1 < ss1) ? 0.f : e11;
        }
""", "")

# ---- kernel comment, modes, parameters
rep("""// PACKED (v2 design B, added 2026-10-10): each row of length S holds several chats; seg_start[b*S + t] is
// where the segment of token t starts (contract: seg_start[t] <= t, and it is t or seg_start[t-1]). Query i
// sees key j only when seg_start[i] <= j <= i. PACKED = false is the plain causal kernel, unchanged; its
// SASS is checked identical to the version before this change. seg_start is the LAST parameter so the
// offsets of all the older parameters stay where they were.
""", """// v2 design B (one seg_start value per token, added 2026-10-10) was REMOVED on 2026-10-15: design A below is
// faster (backward 3-5% on middle and long chats, 2026-10-10/12 measurements) and is the cu_seqlens way of
// FlashAttention, TE/Megatron and TRL. The dense kernel is unchanged by both (SASS checked).
""")
rep("""constexpr int MODE_DENSE = 0, MODE_SEG = 1, MODE_CHATS = 2;

template<int Br, int Bc, int D, int MODE>""", """constexpr int MODE_DENSE = 0, MODE_CHATS = 1;

template<int Br, int Bc, int D, int MODE>""")
rep("""    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* __restrict__ seg_start,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {   // A only
    constexpr bool PACKED = (MODE == MODE_SEG);
    constexpr bool CHATS  = (MODE == MODE_CHATS);""", """    int B, int Hq, int Hkv, int G, int S, float scale,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {   // A only
    constexpr bool CHATS  = (MODE == MODE_CHATS);""")
rep("""    const int k_row0 = kt * Bc;                     // relative to the row (dense/B) or segment (A)""",
    """    const int k_row0 = kt * Bc;                     // relative to the row (dense) or segment (A)""")
rep("""    // PACKED only: seg_start of the current q-tile's 64 rows, loaded next to sLSE/sD in issue(). Unused (and
    // so not allocated) in the dense kernel. +256 B keeps 2 CTAs/SM: 2 x (smem + 1 KB reserve) <= 228 KB.
    __shared__ int32_t sSS[Br];
""", "")
cut("    // PACKED: how many q-tiles (from q = kt on) this key tile has to visit.", "    mbar_wait_v4(&mbar_kv, 0);\n")
rep("""    const int nq_ = PACKED ? packed_nq : nQ - kt, nIter = nq_;   // Vj1 per-hq: ONE query head per CTA""",
    """    const int nq_ = nQ - kt, nIter = nq_;   // Vj1 per-hq: ONE query head per CTA""")
rep("""        if constexpr (PACKED) {   // PACKED: the q-tile's seg_start rides with sLSE/sD (same publish barriers)
            if (wtid < Br) sSS[wtid] = seg_start[(long)b * S + (long)q * Br + wtid];
        }
""", "")
rep("""    // Tile 0's sLSE/sD (and sSS when PACKED) are written by threads 0..63 in issue(0) and read by all 4 warps in""",
    """    // Tile 0's sLSE/sD are written by threads 0..63 in issue(0) and read by all 4 warps in""")
rep("""        if constexpr (PACKED) {
            // PACKED: the masked path is needed on the diagonal tile (causal) AND on any tile where some row's
            // segment starts after the tile's first key. Rows are sorted by seg_start, so the last row
            // (sSS[Br-1]) decides: if even it starts at or before k_row0, every key of the tile is visible to
            // every row of this q > kt tile and the plain unmasked path is exact. sSS[Br-1] is the same in
            // all threads, so the branch is uniform.
            if (q == kt || sSS[Br - 1] > k_row0)
                fused_p_stsm<Bc,true,true>(acc, sP, sLSE, wtid, q*Br, k_row0, scale2, sSS);
            else
                fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);
        } else {
        if (q == kt) fused_p_stsm<Bc,true >(acc, sP, sLSE, wtid, q*Br, k_row0, scale2);
        else         fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);  // softmax OVERLAPS dP GEMM
        }""", """        if (q == kt) fused_p_stsm<Bc,true >(acc, sP, sLSE, wtid, q*Br, k_row0, scale2);
        else         fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);  // softmax OVERLAPS dP GEMM""")

# ---- non-Hopper stub
rep("""constexpr int MODE_DENSE = 0, MODE_SEG = 1, MODE_CHATS = 2;
template<int Br, int Bc, int D, int MODE>
__global__ void gqa_bwd_vj1d(""", """constexpr int MODE_DENSE = 0, MODE_CHATS = 1;
template<int Br, int Bc, int D, int MODE>
__global__ void gqa_bwd_vj1d(""")
rep("""    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* __restrict__ seg_start,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {}""",
    """    int B, int Hq, int Hkv, int G, int S, float scale,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {}""")

# ---- launcher + entry point
rep("""    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t *d_seg_start, const int32_t *d_chat_tiles, int n_chat_tiles) {""",
    """    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t *d_chat_tiles, int n_chat_tiles) {""")
rep("""    // d_seg_start != nullptr -> packed rows (design B). Same grid: every key tile still gets its CTA (each key
    // is at least visible to its own query), only the q-tile trip count shrinks.
    // d_chat_tiles != nullptr -> design A: one CTA per (plan key tile, query head).
    if (d_chat_tiles) {
        dim3 GRID_A(n_chat_tiles, Hq, 1);
        gqa_bwd_vj1d<Br,Bc,D,MODE_CHATS><<<GRID_A,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,nullptr,
            reinterpret_cast<const int4*>(d_chat_tiles), d_dVp, d_dKp);
    } else if (d_seg_start)
        gqa_bwd_vj1d<Br,Bc,D,MODE_SEG><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,d_seg_start,nullptr,nullptr,nullptr);
    else
        gqa_bwd_vj1d<Br,Bc,D,MODE_DENSE><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,nullptr,nullptr,nullptr,nullptr);""",
    """    // d_chat_tiles != nullptr -> design A: one CTA per (plan key tile, query head).
    if (d_chat_tiles) {
        dim3 GRID_A(n_chat_tiles, Hq, 1);
        gqa_bwd_vj1d<Br,Bc,D,MODE_CHATS><<<GRID_A,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,
            reinterpret_cast<const int4*>(d_chat_tiles), d_dVp, d_dKp);
    } else
        gqa_bwd_vj1d<Br,Bc,D,MODE_DENSE><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,nullptr,nullptr,nullptr);""")
rep("""// seg_start (v2 design B, added 2026-10-10): device int32 [B, S] or nullptr
// (plain causal, as before). See gqa_flash_attn_backward_qkv for the contract.
// ---------------------------------------------------------------------------""",
    """// chat_tiles/n_chat_tiles: the bwd_tiles list of a GqaChatPlan (design A), or nullptr = dense.
// ---------------------------------------------------------------------------""")
rep("""    const int32_t* seg_start, const int32_t* chat_tiles, int n_chat_tiles)
{
    gqa_bwd_vj1d::launch_gqa_backward_vj1d<64, 64, 128>(
        Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale, seg_start, chat_tiles, n_chat_tiles);""",
    """    const int32_t* chat_tiles, int n_chat_tiles)
{
    gqa_bwd_vj1d::launch_gqa_backward_vj1d<64, 64, 128>(
        Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale, chat_tiles, n_chat_tiles);""")

print([l.strip()[:110] for l in s.splitlines() if "seg_start" in l or "PACKED" in l or "MODE_SEG" in l or "sSS" in l or "packed_nq" in l])
assert s.count("seg_start") == 1, s.count("seg_start")   # only the removal note
for bad in ("PACKED", "MODE_SEG", "sSS", "packed_nq", "SEG,", "ss0"):
    assert bad not in s, bad
open(p, "w", encoding="utf-8").write(s)
print("patched GQA_bwd_vj1d_sm90.cu")
