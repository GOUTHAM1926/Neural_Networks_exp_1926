#!/usr/bin/env python3
"""P1 (2026-10-15): v62 forward keeps ONLY the per-chat list (design A, the cu_seqlens way); design B (per-token
seg_start) is removed. Dense and list-mode code stay as they were. Every replacement must match exactly once.
usage: p1_fwd_patch.py <Tensor-Implementations dir>"""
import sys, os
TI = sys.argv[1]
p = os.path.join(TI, "src/Kernels/cuda/attention/arch/GQA_fwd_sm90.cu")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

# ---- kernel comment: the design-B paragraph goes, a one-paragraph history note replaces it
a = s.index("// v2 packing (design B, 2026-10-10): PACKED rows hold several chats.")
b = s.index("// v2 packing, design A (per-segment work list, the cu_seqlens way; added 2026-10-10), MODE_CHATS:")
s = s[:a] + (
"// v2 packing: design B (one seg_start value per token, the FlexAttention document-mask way; added 2026-10-10)\n"
"// was REMOVED on 2026-10-15. Design A below does the same job faster (forward 2-16% on the 2026-10-10/12\n"
"// measurements: B's tiles that hold a chat border mix two chats, A starts every chat on its own tiles) and is\n"
"// the cu_seqlens way of FlashAttention 2/3/4, TE/Megatron THD, PyTorch varlen_attn and TRL. One packing\n"
"// representation keeps one tested path.\n"
) + s[b:]

rep("""constexpr int MODE_DENSE = 0, MODE_SEG = 1, MODE_CHATS = 2;

template <int MODE>""", """constexpr int MODE_DENSE = 0, MODE_CHATS = 1;

template <int MODE>""")
rep("""    int B, int Hq, int Hkv, int S, float scale,
    const int32_t* __restrict__ seg_start,   // v2 B: [B, S], read only when PACKED
    const int4* __restrict__ units,          // v2 A: work list, read only when CHATS
    int n_units)
{
    constexpr bool PACKED = (MODE == MODE_SEG);
    constexpr bool CHATS  = (MODE == MODE_CHATS);""", """    int B, int Hq, int Hkv, int S, float scale,
    const int4* __restrict__ units,          // v2 A: work list, read only when CHATS
    int n_units)
{
    constexpr bool CHATS  = (MODE == MODE_CHATS);""")

# ---- producer: the pair's K/V loop always starts at tile 0 (the segment start in A)
rep("""            // v2: first K/V tile of the pair. Dense: 0. PACKED: the tile that holds the chat start of
            // the pair's first row (the smallest chat start of all its rows). The consumers compute
            // the same n0 from the same element, so all three warpgroups run the same trip count.
            int n0 = 0;
            if constexpr (PACKED) n0 = seg_start[(long long)b * S + pair * (2 * BLOCK_M)] / BLOCK_N;
            const int nkv = shared_nkv(pair) - n0;
            const int kv_row_base = (b * Hkv + hkv) * S + t0;   // t0: A only (else 0)
            for (int n = 0; n < nkv; ++n) {
                if (g >= NSTAGES) mbar_wait(&empty[stage], ph ^ 1);
                if (lane_wg == 0) {
                    int row = kv_row_base + (n0 + n) * BLOCK_N;""", """            const int nkv = shared_nkv(pair);
            const int kv_row_base = (b * Hkv + hkv) * S + t0;   // t0: A only (else 0)
            for (int n = 0; n < nkv; ++n) {
                if (g >= NSTAGES) mbar_wait(&empty[stage], ph ^ 1);
                if (lane_wg == 0) {
                    int row = kv_row_base + n * BLOCK_N;""")

# ---- consumer
rep("""            // gR0/gR1: rows of this thread, relative to the row start (dense/B) or the segment start (A)""",
    """            // gR0/gR1: rows of this thread, relative to the row start (dense) or the segment start (A)""")
rep("""            // v2 PACKED: n0 = the pair's first K/V tile (same element the producer reads);
            // sR0/sR1 = chat start of this thread's two rows; smax = chat start of the q-tile's last
            // row = the largest in the tile (non-decreasing), so a tile whose first key is below
            // smax holds a chat border for at least one row and must be masked.
            int n0 = 0, sR0 = 0, sR1 = 0, smax = 0;
            if constexpr (PACKED) {
                const int32_t* ss = seg_start + (long long)b * S;
                n0 = ss[pair * (2 * BLOCK_M)] / BLOCK_N;
                sR0 = ss[gR0];
                sR1 = ss[gR1];
                smax = ss[q_tile * BLOCK_M + BLOCK_M - 1];
            }
            const int nkv_shared = shared_nkv(pair) - n0;
            int my_nkv = (q_tile * BLOCK_M + BLOCK_M - 1) / BLOCK_N + 1 - n0;""",
    """            const int nkv_shared = shared_nkv(pair);
            int my_nkv = (q_tile * BLOCK_M + BLOCK_M - 1) / BLOCK_N + 1;""")
rep("""                    // v2: nt = absolute K/V tile (dense: n0 = 0, so nt = n as before). PACKED also
                    // masks a tile whose first key is below smax: it holds a chat border for a row.
                    const int nt = n0 + n;
                    const bool diag = (nt * BLOCK_N + BLOCK_N - 1 >= q_tile * BLOCK_M);
                    const bool border = PACKED && (nt * BLOCK_N < smax);
                    float mA[8], mB[8];
                    if (!(diag || border)) {""", """                    const int nt = n;   // K/V tile, relative to the row (dense) or the segment (A)
                    const bool diag = (nt * BLOCK_N + BLOCK_N - 1 >= q_tile * BLOCK_M);
                    float mA[8], mB[8];
                    if (!diag) {""")
rep("""                            if (gcol > grow) v = -INFINITY;
                            if constexpr (PACKED) {   // v2: keys before the row's chat start are hidden too
                                if (gcol < ((s < 2) ? sR0 : sR1)) v = -INFINITY;
                            }
                            S_acc[reg] = v;""", """                            if (gcol > grow) v = -INFINITY;
                            S_acc[reg] = v;""")
rep("""                    // v2 PACKED: um = the max used for the rescale and the exponent bias. A row of a
                    // later chat can see NO key in the pair's first tiles; its max is still -inf and
                    // ex2((-inf) - (-inf)) = NaN. With um = 0 there: cc = ex2(-inf) = 0 and p =
                    // ex2(-inf + 0) = 0, so the row's l and O stay exactly 0 until its first visible
                    // key (Flex common.py.jinja:115-123). Dense: um = nm, as before.
                    float um0 = nm0, um1 = nm1;
                    if constexpr (PACKED) {
                        if (um0 == -INFINITY) um0 = 0.0f;
                        if (um1 == -INFINITY) um1 = 0.0f;
                    }
                    float cc0 = ex2_approx((m0 - um0) * scale2), cc1 = ex2_approx((m1 - um1) * scale2);
                    const float b0 = -um0 * scale2, b1 = -um1 * scale2;""",
    """                    // No -inf guard is needed: every row (dense, or A's segment-relative rows) sees key 0
                    // of its own row or segment in the first tile, so nm0/nm1 are finite from tile 0 on.
                    float cc0 = ex2_approx((m0 - nm0) * scale2), cc1 = ex2_approx((m1 - nm1) * scale2);
                    const float b0 = -nm0 * scale2, b1 = -nm1 * scale2;""")

# ---- non-Hopper stub
rep("""constexpr int MODE_DENSE = 0, MODE_SEG = 1, MODE_CHATS = 2;
template <int MODE>
__global__ void gqa_v62_sm90(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
                        const __grid_constant__ CUtensorMap, const __nv_bfloat16* __restrict__,
                        __nv_bfloat16* __restrict__, float* __restrict__, int, int, int, int, float,
                        const int32_t* __restrict__, const int4* __restrict__, int) {}""",
    """constexpr int MODE_DENSE = 0, MODE_CHATS = 1;
template <int MODE>
__global__ void gqa_v62_sm90(const __grid_constant__ CUtensorMap, const __grid_constant__ CUtensorMap,
                        const __grid_constant__ CUtensorMap, const __nv_bfloat16* __restrict__,
                        __nv_bfloat16* __restrict__, float* __restrict__, int, int, int, int, float,
                        const int4* __restrict__, int) {}""")

# ---- launcher
rep("""// v2 (2026-10-10): seg_start (device, int32 [B, S]) selects the packed-rows variant (design B,
// see the kernel comment); chat_units/n_units (device work list, design A) select the per-segment
// variant; both nullptr keeps the dense kernel exactly as before. The caller never passes both.
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* seg_start, const int32_t* chat_units, int n_units)""",
    """// v2 (2026-10-10): chat_units/n_units (device work list of a GqaChatPlan, design A) select the
// per-segment variant; nullptr keeps the dense kernel exactly as before. (Design B's seg_start
// parameter was removed on 2026-10-15, see the kernel comment.)
void gqa_v62_causal_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int B, int Hq, int Hkv, int S, float scale, cudaStream_t stream,
    const int32_t* chat_units, int n_units)""")
rep("""    static bool cfg_dense = false, cfg_packed = false, cfg_chats = false;""",
    """    static bool cfg_dense = false, cfg_chats = false;""")
rep("""        gqa62::gqa_v62_sm90<gqa62::MODE_CHATS><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, units, n_units);
    } else if (seg_start == nullptr) {
        if (!cfg_dense) { cudaFuncSetAttribute(gqa62::gqa_v62_sm90<gqa62::MODE_DENSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg_dense = true; }
        gqa62::gqa_v62_sm90<gqa62::MODE_DENSE><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, nullptr, 0);
    } else {
        if (!cfg_packed) { cudaFuncSetAttribute(gqa62::gqa_v62_sm90<gqa62::MODE_SEG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg_packed = true; }
        gqa62::gqa_v62_sm90<gqa62::MODE_SEG><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, seg_start, nullptr, 0);
    }""", """        gqa62::gqa_v62_sm90<gqa62::MODE_CHATS><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, units, n_units);
    } else {
        if (!cfg_dense) { cudaFuncSetAttribute(gqa62::gqa_v62_sm90<gqa62::MODE_DENSE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes); cfg_dense = true; }
        gqa62::gqa_v62_sm90<gqa62::MODE_DENSE><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, 0);
    }""")

print([l for l in s.splitlines() if "seg_start" in l]); assert s.count("seg_start") == 2, s.count("seg_start")   # the kernel note + the launcher note
for bad in ("PACKED", "MODE_SEG", "sR0", "smax", "n0 "):
    assert bad not in s, ("left over", bad, s[max(0, s.find(bad) - 200):s.find(bad) + 100])
open(p, "w", encoding="utf-8").write(s)
print("patched GQA_fwd_sm90.cu")
