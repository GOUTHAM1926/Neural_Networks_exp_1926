#!/usr/bin/env python3
"""P3 (2026-10-15): Vj2 backward -- design B (per-token seg_start, ported this morning) is replaced by design A (the
per-chat list of a GqaChatPlan, the cu_seqlens way). The dense kernel must stay as it was (SASS checked after).
Every replacement must match exactly once. usage: p3_vj2_patch.py <Tensor-Implementations dir>"""
import sys, os
TI = sys.argv[1]
p = os.path.join(TI, "src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

def cut(start, end, new=""):
    global s
    assert s.count(start) == 1 and s.count(end) == 1, (s.count(start), s.count(end), start[:80], end[:80])
    a = s.index(start); b = s.index(end)
    assert a < b
    s = s[:a] + new + s[b:]

# ---- file header
rep("""//  SCOPE: causal; dense rows, and packed rows of design B (seg_start, added 2026-10-15, the same rule as Vj1d's
//  MODE_SEG). Design A (chat plans) still runs on Vj1d: GQA_bwd_generic.cu routes it there.""",
"""//  SCOPE: causal; dense rows, and packed rows given as a per-chat list (v2 design A, the cu_seqlens way of
//  FlashAttention varlen; added 2026-10-15, replacing the per-token seg_start version written earlier that day).
//  See the kernel comment for how a chat's work is cut.""")

# ---- constants: the design-B seg_start stage goes
rep("""// Packed rows (design B, 2026-10-15) only: each step's 64 seg_start values (2 stages x 64 int32) after the barriers.
// The dense layout above is unchanged, so the dense kernel keeps its code.
constexpr uint32_t OFF_SS            = SMEM_USED;                   // 197696
constexpr uint32_t SMEM_BYTES_PACKED = OFF_SS + 2 * BM * 4 + 1024;  // 199232
static_assert(SMEM_BYTES_PACKED <= 232448 && OFF_SS % 16 == 0, "packed seg_start stage: size and bulk-copy alignment");
""", "")
rep("""constexpr uint32_t QSTEP_TX_PACKED = QSTEP_TX + BM * 4;      // + the step's 64 seg_start (int32) = 33536 B
""", "")

# ---- design-A prep kernel, after the dense one
rep("""        if (lane == 0) { d_Drow[row] = partial; d_Lse2[row] = d_LSE[row] * LOG2E; }
    }
}
""", """        if (lane == 0) { d_Drow[row] = partial; d_Lse2[row] = d_LSE[row] * LOG2E; }
    }
}

// Design A (per-chat lists): the same per-row work, written to the plan's PADDED layout. A chat can start at any token,
// but the 256-B LSE/D bulk loads and the fragment-order dQ tiles need 64-row alignment, so every chat gets its own
// 64-row-aligned slot per query head -- FA3's varlen backward does the same (seqlen.h offset_padded: each sequence of
// dQaccum / LSE_log2 / dPsum starts on a kBlockM boundary). Each chat owns ceil(L/64) tiles per head: padded row
// p = (hq * nT + T) * 64 + r is token row0 + r of row b when r < nrows (dq_tiles[T] = {b, row0, nrows, 0}), else
// padding. Padding gets LSE2 = +inf and D = 0, so P = ex2(S * scale2 - inf) = 0 and dS = P * (dP - D) = 0 there:
// the rows past a chat's end that its last step reads add exact zeros (FlashAttention 2 sets LSE = +inf past the
// sequence end the same way, csrc/flash_attn/src/flash_bwd_kernel.h:411-413). The fp32 dQ accumulator uses the same
// padded rows (128 floats per row, zeroed here). D is computed exactly as above, so a chat's D is bit-identical to the
// dense run of the same chat.
__global__ void bwd_prep_vj2_chats(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                                   const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                                   float* __restrict__ d_Lse2, float* __restrict__ d_dqa,
                                   const int4* __restrict__ dq_tiles, int nT, int Hq, int S, long nRowsP) {
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    for (long row = warp0; row < nRowsP; row += stride) {
        *reinterpret_cast<float4*>(d_dqa + row * 128 + lane * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
        const long tp = row >> 6;                         // padded tile over all heads: hq * nT + T
        const int  r  = (int)(row & 63);
        const int  hq = (int)(tp / nT);
        const int4 dt = dq_tiles[tp - (long)hq * nT];     // {b, row0, nrows, 0}
        if (r < dt.z) {                                    // one row per warp: the same in every lane
            const long src  = ((long)dt.x * Hq + hq) * S + dt.y + r;
            const long base = src * 128 + lane * 4;
            const uint2 vdO = *reinterpret_cast<const uint2*>(d_dO + base);
            const uint2 vO  = *reinterpret_cast<const uint2*>(d_O  + base);
            const bf16* a = reinterpret_cast<const bf16*>(&vdO);
            const bf16* b = reinterpret_cast<const bf16*>(&vO);
            float partial = 0.f;
#pragma unroll
            for (int k = 0; k < 4; k++) partial += __bfloat162float(a[k]) * __bfloat162float(b[k]);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
            if (lane == 0) { d_Drow[row] = partial; d_Lse2[row] = d_LSE[src] * LOG2E; }
        } else if (lane == 0) {
            d_Drow[row] = 0.f; d_Lse2[row] = INFINITY;
        }
    }
}
""")

# ---- kernel comment + signature + block decode
rep("""// PACKED (v2 design B, 2026-10-15): each row holds several chats; seg_start[b*S + t] is where the segment of token t
// starts (contract: seg_start[t] <= t, and it is t or seg_start[t-1]); query i sees key j only when
// seg_start[i] <= j <= i. Same rule and the same trip-count proof as Vj1d's MODE_SEG (GQA_bwd_vj1d_sm90.cu:428-505).
// PACKED = false is the dense kernel, unchanged; seg_start is the LAST parameter so the older ones keep their offsets.
template<bool PACKED>""",
"""// CHATS (v2 design A, the cu_seqlens way; 2026-10-15): one block per (128-key tile of a chat, query head), read from
// the plan: blocks[2i] = {b, t0, L, kt}, blocks[2i+1] = {hq, dqt0, 0, 0} = key tile kt of segment [t0, t0+L) of row b
// (a chat, or a row's pad tail as its own segment). Every index is segment-relative, so the steps, the diagonal mask
// and the trip count are the dense ones with S = L (Vj1d's MODE_CHATS does the same with 64-key tiles). The block
// order is made by the plan builder: FA3's LPT order inside sections of `sw` (batch, head) pairs, as the dense grid.
// What differs from dense:
//   * Q/dO/K/V rows are offset by t0. A chat's last step can hold rows past L (the next segment's rows, or rows past
//     the tensor end that TMA fills with zeros): their LSE/D come from the padded slot as +inf / 0, so P = 0, dS = 0.
//   * LSE, D and the fp32 dQ accumulator are in the plan's PADDED layout (bwd_prep_vj2_chats): dqt0 = the slot tile
//     of this block's first step; step i reads and adds tile dqt0 + i.
//   * keys past L sit only in the chat's last key tile. Causal hides them: every query of the chat is before them,
//     and every step that holds such keys is a diagonal step. Their dK/dV rows belong to the next segment, so a
//     compute group whose 64 keys cross L stores only its rows < L, with plain stores, and a group entirely past L
//     stores nothing (FA3 stores all varlen dK/dV with plain stores, hopper/epilogue_bwd.hpp:31,263-268).
// CHATS = false is the dense kernel, unchanged (SASS checked identical to the first Vj2). The three A-only parameters
// are the LAST ones so the older ones keep their offsets.
template<bool CHATS>""")
rep("""    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int32_t* __restrict__ seg_start) {
    int kt, hq, b;
    {
        const int nkt = S / BN, HB = B * Hq;""", """    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int4* __restrict__ blocks, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {   // A only
    int kt, hq, b;
    int t0 = 0, L = 0, dqt0 = 0;                        // A only (dense: 0)
    if constexpr (CHATS) {
        const int4 w0 = blocks[2 * blockIdx.x], w1 = blocks[2 * blockIdx.x + 1];
        b = w0.x; t0 = w0.y; L = w0.z; kt = w0.w; hq = w1.x; dqt0 = w1.y;
    } else {
        const int nkt = S / BN, HB = B * Hq;""")
rep("""    const int k0 = kt * BN;
    const int q_first = 2 * kt;
    int nIter = S / BM - q_first;                       // >= 1: k0 < S and S % 64 == 0 (PACKED: set below)""",
"""    const int k0 = kt * BN;                             // relative to the row (dense) or the segment (A)
    const int q_first = 2 * kt;
    int nIter = S / BM - q_first;                       // >= 1: k0 < S and S % 64 == 0
    if constexpr (CHATS) nIter = (L + BM - 1) / BM - q_first;   // A: >= 1 because k0 < L""")
rep("""    int32_t* sSS = reinterpret_cast<int32_t*>(sm + OFF_SS); // [stage][BM] seg_start, PACKED only
""", "")
cut("    // PACKED: the query steps that see at least one key of this block.", "    if (wg == 0) {\n")

# ---- loader
rep("""                const int kvRow = (b * Hkv + hkv) * S + k0;""",
    """                const int kvRow = (b * Hkv + hkv) * S + t0 + k0;   // t0: A only (else 0)""")
rep("""                    const int qRow = (b * Hq + hq) * S + (q_first + i) * BM;
                    bf16* q_s  = sQ  + s * Q_TILE;
                    bf16* do_s = sdO + s * Q_TILE;
                    mbar_arrive_expect_tx(&q_full[s], PACKED ? QSTEP_TX_PACKED : QSTEP_TX);
                    tma_load_2d(q_s,           &tma_Q,  0,  qRow, &q_full[s]);
                    tma_load_2d(q_s + Q_HALF,  &tma_Q,  64, qRow, &q_full[s]);
                    tma_load_2d(do_s,          &tma_dO, 0,  qRow, &q_full[s]);
                    tma_load_2d(do_s + Q_HALF, &tma_dO, 64, qRow, &q_full[s]);
                    bulk_load_1d(sLSE + s * BM, Lse2 + qRow, BM * 4, &q_full[s]);
                    bulk_load_1d(sD   + s * BM, Drow + qRow, BM * 4, &q_full[s]);
                    if constexpr (PACKED)   // the step's 64 seg_start, rows of batch b (not per head)
                        bulk_load_1d(sSS + s * BM, seg_start + (long)b * S + (q_first + i) * BM, BM * 4, &q_full[s]);
""", """                    const int qRow = (b * Hq + hq) * S + t0 + (q_first + i) * BM;   // t0: A only (else 0)
                    bf16* q_s  = sQ  + s * Q_TILE;
                    bf16* do_s = sdO + s * Q_TILE;
                    mbar_arrive_expect_tx(&q_full[s], QSTEP_TX);
                    tma_load_2d(q_s,           &tma_Q,  0,  qRow, &q_full[s]);
                    tma_load_2d(q_s + Q_HALF,  &tma_Q,  64, qRow, &q_full[s]);
                    tma_load_2d(do_s,          &tma_dO, 0,  qRow, &q_full[s]);
                    tma_load_2d(do_s + Q_HALF, &tma_dO, 64, qRow, &q_full[s]);
                    const float* lse_src = Lse2 + qRow;
                    const float* d_src   = Drow + qRow;
                    if constexpr (CHATS) {   // A: the step's padded slot (64-row aligned, see bwd_prep_vj2_chats)
                        lse_src = Lse2 + (long)(dqt0 + i) * BM;
                        d_src   = Drow + (long)(dqt0 + i) * BM;
                    }
                    bulk_load_1d(sLSE + s * BM, lse_src, BM * 4, &q_full[s]);
                    bulk_load_1d(sD   + s * BM, d_src,   BM * 4, &q_full[s]);
""")
rep("""                float* tile = dQacc + ((long)(b * Hq + hq) * (S / BM) + (q_first + i)) * (BM * D);
""", """                float* tile = dQacc + ((long)(b * Hq + hq) * (S / BM) + (q_first + i)) * (BM * D);
                if constexpr (CHATS) tile = dQacc + (long)(dqt0 + i) * (BM * D);   // A: the step's padded slot
""")

# ---- compute: back to the original dense mask (A uses it unchanged: segment-relative indices, +inf LSE past L)
rep("""        // P = exp(S * scale - LSE); on the first two steps (the ones holding the diagonal) hide key > query.
        // PACKED also hides key < seg_start[query] (a key of an earlier chat) on any step where some row's segment
        // starts after this group's first key: the rows are sorted by seg_start, so the step's last row decides, and
        // the value is the same in every thread (uniform branch). On the other steps every key of the group is
        // visible to every row of the step, so the plain path is exact.
        bool diag = (q0 < k0 + BN);
        if constexpr (PACKED) diag = diag || (sSS[s * BM + BM - 1] > kbase);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float p = ex2_approx(__fmaf_rn(S_[4 * nt + e], scale2, -l2[2 * nt + (e & 1)]));
                if (diag) {
                    const int key = kbase + r0 + ((e & 2) ? 8 : 0);
                    const int qry = q0 + 8 * nt + 2 * t + (e & 1);
                    bool hide = key > qry;
                    if constexpr (PACKED) hide = hide || key < sSS[s * BM + 8 * nt + 2 * t + (e & 1)];
                    p = hide ? 0.f : p;
                }
                S_[4 * nt + e] = p;
            }
        }""", """        // P = exp(S * scale - LSE); on the first two steps (the ones holding the diagonal) hide key > query.
        // A: the same, in segment-relative indices; rows past L have LSE2 = +inf, so their P is 0 on every step.
        const bool diag = (q0 < k0 + BN);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float p = ex2_approx(__fmaf_rn(S_[4 * nt + e], scale2, -l2[2 * nt + (e & 1)]));
                if (diag) {
                    const int key = kbase + r0 + ((e & 2) ? 8 : 0);
                    const int qry = q0 + 8 * nt + 2 * t + (e & 1);
                    p = (key > qry) ? 0.f : p;
                }
                S_[4 * nt + e] = p;
            }
        }""")

# ---- epilogue
rep("""    bar_sync(BAR_EPI + c, WGT);
    if (tw == 0 && kbase < S) {                   // S % 128 == 64: group 1 of the last block has no keys
        const int hqRow = (b * Hq + hq) * S + kbase;
        tma_store_2d(&tma_dV_st, stg,              0,  hqRow);
        tma_store_2d(&tma_dV_st, stg + Q_HALF,     64, hqRow);
        tma_store_2d(&tma_dK_st, stg + 2 * Q_HALF, 0,  hqRow);
        tma_store_2d(&tma_dK_st, stg + 3 * Q_HALF, 64, hqRow);
        bulk_commit();
        bulk_wait_all();
    }
}""", """    bar_sync(BAR_EPI + c, WGT);
    if constexpr (CHATS) {
        // A: this group's keys are rows kbase .. kbase+63 of the segment; only the rows < L are the segment's.
        const int nrow = L - kbase;
        const int hqRow = (b * Hq + hq) * S + t0 + kbase;
        if (nrow >= 64) {
            if (tw == 0) {
                tma_store_2d(&tma_dV_st, stg,              0,  hqRow);
                tma_store_2d(&tma_dV_st, stg + Q_HALF,     64, hqRow);
                tma_store_2d(&tma_dK_st, stg + 2 * Q_HALF, 0,  hqRow);
                tma_store_2d(&tma_dK_st, stg + 3 * Q_HALF, 64, hqRow);
                bulk_commit();
                bulk_wait_all();
            }
        } else if (nrow > 0) {
            // Plain 16-B stores of rows < nrow from the swizzled staging (stage_acc_bf16_sw128: chunk k of row r sits
            // at chunk k ^ (r & 7)). Blocks: dV cols 0-63, dV cols 64-127, dK cols 0-63, dK cols 64-127.
            for (int e = tw; e < 4 * 64 * 8; e += WGT) {
                const int blk = e >> 9, r = (e >> 3) & 63, k = e & 7;
                if (r >= nrow) continue;
                const bf16* src = stg + blk * Q_HALF + r * 64 + ((k ^ (r & 7)) << 3);
                bf16* dst = ((blk < 2) ? dVp : dKp) + ((long)hqRow + r) * D + (blk & 1) * 64 + k * 8;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
            }
        }
    } else {
    if (tw == 0 && kbase < S) {                   // S % 128 == 64: group 1 of the last block has no keys
        const int hqRow = (b * Hq + hq) * S + kbase;
        tma_store_2d(&tma_dV_st, stg,              0,  hqRow);
        tma_store_2d(&tma_dV_st, stg + Q_HALF,     64, hqRow);
        tma_store_2d(&tma_dK_st, stg + 2 * Q_HALF, 0,  hqRow);
        tma_store_2d(&tma_dK_st, stg + 3 * Q_HALF, 64, hqRow);
        bulk_commit();
        bulk_wait_all();
    }
    }
}""")

# ---- dQ convert: A writes a padded tile back to its chat's token rows
rep("""// is applied here, once per value (see store_acc_frag_f32).
constexpr int CVT_THREADS = 256, CVT_LD = 136;
__global__ void __launch_bounds__(CVT_THREADS)
convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale) {""",
"""// is applied here, once per value (see store_acc_frag_f32).
// A (CHATS): tile = padded tile hq * nT + T of the plan; it holds token rows row0 .. row0 + nrows - 1 of row b
// (dq_tiles[T] = {b, row0, nrows, 0}); its padding rows (past the chat's end) are not written.
constexpr int CVT_THREADS = 256, CVT_LD = 136;
template<bool CHATS>
__global__ void __launch_bounds__(CVT_THREADS)
convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale,
                    const int4* __restrict__ dq_tiles, int nT, int Hq, int S) {   // the last four: A only""")
rep("""    __syncthreads();
    uint4* dst = reinterpret_cast<uint4*>(out + tile * (BM * D));
#pragma unroll
    for (int m = 0; m < (BM * D / 8) / CVT_THREADS; m++) {
        const int k = threadIdx.x + m * CVT_THREADS, row = k >> 4, ch = k & 15;
        dst[k] = *reinterpret_cast<const uint4*>(&st[row * CVT_LD + ch * 8]);
    }
}""", """    __syncthreads();
    uint4* dst = reinterpret_cast<uint4*>(out + tile * (BM * D));
    int nrows = BM;
    if constexpr (CHATS) {
        const int hq = (int)(tile / nT);
        const int4 dt = dq_tiles[tile - (long)hq * nT];
        nrows = dt.z;
        dst = reinterpret_cast<uint4*>(out + (((long)dt.x * Hq + hq) * S + dt.y) * D);
    }
#pragma unroll
    for (int m = 0; m < (BM * D / 8) / CVT_THREADS; m++) {
        const int k = threadIdx.x + m * CVT_THREADS, row = k >> 4, ch = k & 15;
        if (!CHATS || row < nrows) dst[k] = *reinterpret_cast<const uint4*>(&st[row * CVT_LD + ch * 8]);
    }
}""")

# ---- non-Hopper stubs
rep("""template<bool PACKED>
__global__ void gqa_bwd_vj2_kernel(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    float* __restrict__ dQacc,
    const float* __restrict__ Lse2, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int32_t* __restrict__ seg_start) {}
__global__ void gqa_dkdv_greduce_vj2(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                     int Hq, int Hkv, int G, int S) {}
constexpr int CVT_THREADS = 256;
__global__ void convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale) {}""",
"""__global__ void bwd_prep_vj2_chats(const bf16* __restrict__ d_dO, const bf16* __restrict__ d_O,
                                   const float* __restrict__ d_LSE, float* __restrict__ d_Drow,
                                   float* __restrict__ d_Lse2, float* __restrict__ d_dqa,
                                   const int4* __restrict__ dq_tiles, int nT, int Hq, int S, long nRowsP) {}
template<bool CHATS>
__global__ void gqa_bwd_vj2_kernel(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    float* __restrict__ dQacc,
    const float* __restrict__ Lse2, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale, int sw,
    const int4* __restrict__ blocks, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {}
__global__ void gqa_dkdv_greduce_vj2(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                     int Hq, int Hkv, int G, int S) {}
constexpr int CVT_THREADS = 256;
template<bool CHATS>
__global__ void convert_dq_frag_vj2(const float4* __restrict__ in, bf16* __restrict__ out, float scale,
                                    const int4* __restrict__ dq_tiles, int nT, int Hq, int S) {}""")

# ---- host: section size helper (shared with the plan builder) + launcher
rep("""// ---- host launcher -------------------------------------------------------------------------------------------
static void launch_gqa_backward_vj2(
    const bf16* d_Q, const bf16* d_K, const bf16* d_V, const bf16* d_O,
    const bf16* d_dO, const float* d_LSE,
    bf16* d_dQ, bf16* d_dK, bf16* d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* d_seg) {                             // design B seg_start [B, S] (device), or nullptr = dense""",
"""// ---- host launcher -------------------------------------------------------------------------------------------
// Block-order section size, as FA3 (tile_scheduler.hpp:389-398): as many (batch, head) pairs as fit their Q + dO (bf16)
// and dQ accumulator (fp32) in 40 MB of L2, rounded down to a power of 2, at least 1. The dense grid uses it below;
// gqa_chat_plan_build orders design A's blocks with the same sections (gqa_vj2_section_pairs).
static int section_pairs(int S) {
    const long one_head = (long)S * (2 * D) * 2 + (long)S * D * 4;
    int sw = 1;
    while (2L * sw * one_head <= 40L * 1024 * 1024) sw *= 2;
    return sw;
}

static void launch_gqa_backward_vj2(
    const bf16* d_Q, const bf16* d_K, const bf16* d_V, const bf16* d_O,
    const bf16* d_dO, const float* d_LSE,
    bf16* d_dQ, bf16* d_dK, bf16* d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* d_blocks, int n_blocks,              // design A: the plan's Vj2 block list, or nullptr = dense
    const int32_t* d_dqt, int n_dqt) {                  //   and its padded dQ tiles (n_dqt per query head)
    const bool chats = d_blocks != nullptr;""")
rep("""    const long rowsN = (long)B * Hq * S;
    static float *d_Drow = nullptr, *d_Lse2 = nullptr; static long rcap = 0;
    if (rowsN > rcap) {
        if (d_Drow) alloc.deallocate(d_Drow);
        if (d_Lse2) alloc.deallocate(d_Lse2);
        d_Drow = static_cast<float*>(alloc.allocate((rowsN + 64) * sizeof(float)));
        d_Lse2 = static_cast<float*>(alloc.allocate((rowsN + 64) * sizeof(float)));
        rcap = rowsN;
    }""", """    const long rowsN = (long)B * Hq * S;
    // LSE*log2e / D rows and the fp32 dQ accumulator: one per token row (dense), or one per PADDED row of the plan
    // (design A: every chat 64-row aligned per head, see bwd_prep_vj2_chats), which is never fewer.
    const long rowsP = chats ? (long)Hq * n_dqt * BM : rowsN;
    static float *d_Drow = nullptr, *d_Lse2 = nullptr; static long rcap = 0;
    if (rowsP > rcap) {
        if (d_Drow) alloc.deallocate(d_Drow);
        if (d_Lse2) alloc.deallocate(d_Lse2);
        d_Drow = static_cast<float*>(alloc.allocate((rowsP + 64) * sizeof(float)));
        d_Lse2 = static_cast<float*>(alloc.allocate((rowsP + 64) * sizeof(float)));
        rcap = rowsP;
    }""")
rep("""    const long dqN = (long)B * Hq * S * D;
    static float* d_dqa = nullptr; static long dqc = 0;""", """    const long dqN = rowsP * D;
    static float* d_dqa = nullptr; static long dqc = 0;""")
rep("""    // dQ accumulator: zeroed by bwd_prep_vj2 below (B-1), no separate cudaMemset (dqN == rowsN * 128).""",
    """    // dQ accumulator: zeroed by bwd_prep_vj2(_chats) below (B-1), no separate cudaMemset (dqN == rowsP * 128).""")
rep("""    const int pBlock = 256; const long pGrid = (rowsN + (pBlock / 32) - 1) / (pBlock / 32);
    bwd_prep_vj2<<<(unsigned)pGrid, pBlock>>>(d_dO, d_O, d_LSE, d_Drow, d_Lse2, d_dqa, rowsN);

    static bool attr_set = false, attr_set_packed = false;
    if (!d_seg && !attr_set) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set = true;
    }
    if (d_seg && !attr_set_packed) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES_PACKED));
        attr_set_packed = true;
    }
    // The step's seg_start comes in with a 256-B bulk copy: its source must be 16-B aligned (a row starts at b*S*4 B
    // and a step at q*256 B past it, both multiples of 16 because S % 128 == 0 here).
    if ((uintptr_t)d_seg & 15) { fprintf(stderr, "vj2: seg_start is not 16-B aligned (bulk copy needs it)\\n"); exit(1); }
    // Section size as FA3 (tile_scheduler.hpp:389-398): as many (batch, head) pairs as fit their Q + dO (bf16)
    // and dQ accumulator (fp32) in 40 MB of L2, rounded down to a power of 2, at least 1.
    const long one_head = (long)S * (2 * D) * 2 + (long)S * D * 4;
    int sw = 1;
    while (2L * sw * one_head <= 40L * 1024 * 1024) sw *= 2;
    const unsigned nBlocks = (unsigned)((S / BN) * B * Hq);  // S % 128 == 0 here (the router's condition)
    if (d_seg)
        gqa_bwd_vj2_kernel<true><<<nBlocks, NTHREADS, SMEM_BYTES_PACKED>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2,
                                                                          d_Drow, B, Hq, Hkv, G, S, scale, sw, d_seg);
    else
        gqa_bwd_vj2_kernel<false><<<nBlocks, NTHREADS, SMEM_BYTES>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2, d_Drow,
                                                                    B, Hq, Hkv, G, S, scale, sw, nullptr);
    dim3 grGrid(S / 8, Hkv, B);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dVp, d_dV, Hq, Hkv, G, S);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dKp, d_dK, Hq, Hkv, G, S);
    const long nTiles = (long)B * Hq * (S / BM);         // S % 128 == 0 here (the router's condition)
    convert_dq_frag_vj2<<<(unsigned)nTiles, CVT_THREADS>>>(reinterpret_cast<const float4*>(d_dqa), d_dQ, scale);
}""", """    const int pBlock = 256; const long pGrid = (rowsP + (pBlock / 32) - 1) / (pBlock / 32);
    const int4* dqt = reinterpret_cast<const int4*>(d_dqt);
    if (chats)
        bwd_prep_vj2_chats<<<(unsigned)pGrid, pBlock>>>(d_dO, d_O, d_LSE, d_Drow, d_Lse2, d_dqa, dqt, n_dqt, Hq, S, rowsP);
    else
        bwd_prep_vj2<<<(unsigned)pGrid, pBlock>>>(d_dO, d_O, d_LSE, d_Drow, d_Lse2, d_dqa, rowsN);

    static bool attr_set = false, attr_set_chats = false;
    if (!chats && !attr_set) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set = true;
    }
    if (chats && !attr_set_chats) {
        GQA_BWD_VJ2_CUDA_CHECK(cudaFuncSetAttribute(gqa_bwd_vj2_kernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES));
        attr_set_chats = true;
    }
    const int sw = section_pairs(S);
    if (chats) {   // design A: one block per plan entry, in the plan's order (LPT inside sections of sw pairs)
        gqa_bwd_vj2_kernel<true><<<(unsigned)n_blocks, NTHREADS, SMEM_BYTES>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2,
                                    d_Drow, B, Hq, Hkv, G, S, scale, sw, reinterpret_cast<const int4*>(d_blocks), d_dVp, d_dKp);
    } else {
        const unsigned nBlocks = (unsigned)((S / BN) * B * Hq);  // S % 128 == 0 here (the router's condition)
        gqa_bwd_vj2_kernel<false><<<nBlocks, NTHREADS, SMEM_BYTES>>>(tK, tV, tQ, tdO, tdV, tdK, d_dqa, d_Lse2, d_Drow,
                                                                    B, Hq, Hkv, G, S, scale, sw, nullptr, nullptr, nullptr);
    }
    dim3 grGrid(S / 8, Hkv, B);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dVp, d_dV, Hq, Hkv, G, S);
    gqa_dkdv_greduce_vj2<<<grGrid, 128>>>(d_dKp, d_dK, Hq, Hkv, G, S);
    if (chats) {
        const long nTiles = (long)Hq * n_dqt;               // every padded tile of every head
        convert_dq_frag_vj2<true><<<(unsigned)nTiles, CVT_THREADS>>>(reinterpret_cast<const float4*>(d_dqa), d_dQ,
                                                                     scale, dqt, n_dqt, Hq, S);
    } else {
        const long nTiles = (long)B * Hq * (S / BM);         // S % 128 == 0 here (the router's condition)
        convert_dq_frag_vj2<false><<<(unsigned)nTiles, CVT_THREADS>>>(reinterpret_cast<const float4*>(d_dqa), d_dQ,
                                                                      scale, nullptr, 0, 0, 0);
    }
}""")

# ---- library-facing entry points
rep("""// Library-facing entry point, called from GQA_bwd_generic.cu: the default Hopper backward since 2026-10-15 (causal,
// head_dim 128, S % 128 == 0, H100/H200; USE_VJ2=0 selects Vj1d instead). seg_start = packed rows of design B
// (nullptr = dense); design A plans never reach it (the caller routes them to Vj1d).
// ---------------------------------------------------------------------------
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale, const int32_t* seg_start) {
    gqa_bwd_vj2::launch_gqa_backward_vj2(Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale, seg_start);
}""", """// Library-facing entry point, called from GQA_bwd_generic.cu: the default Hopper backward since 2026-10-15 (causal,
// head_dim 128, S % 128 == 0, H100/H200; USE_VJ2=0 selects Vj1d instead). chat_blocks / dq_tiles = the Vj2 lists of
// a GqaChatPlan (packed rows, design A), or nullptr = dense.
// ---------------------------------------------------------------------------
void gqa_vj2_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_blocks, int n_chat_blocks, const int32_t* dq_tiles, int n_dq_tiles) {
    gqa_bwd_vj2::launch_gqa_backward_vj2(Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale,
                                         chat_blocks, n_chat_blocks, dq_tiles, n_dq_tiles);
}
// The section size of Vj2's block order for rows of S tokens (gqa_chat_plan_build orders design A's blocks with it).
int gqa_vj2_section_pairs(int S) { return gqa_bwd_vj2::section_pairs(S); }""")

left = [l.strip()[:110] for l in s.splitlines() if any(x in l for x in ("seg_start", "PACKED", "sSS", "OFF_SS", "d_seg"))]
print(left)
assert len(left) == 1 and "replacing the per-token seg_start" in left[0], left
open(p, "w", encoding="utf-8").write(s)
print("patched GQA_bwd_vj2_sm90.cu")
