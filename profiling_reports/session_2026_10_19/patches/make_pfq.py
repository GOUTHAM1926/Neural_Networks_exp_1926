# 2026-10-19 (work doc 45.58): the last-chance forward candidates, built from PF (cands/fwd_PF.cu 1888246a) by anchored edits
# (each anchor must occur exactly once, else nothing is written). Three switches, each one cause proven in 45.57:
#   Q_PREF     FA3's L2 prefetch of the tile's Q when the producer starts it (mainloop_fwd_sm90_tma_gmma_ws.hpp:681)
#   DEC_PUB    the producer publishes the decoded dense unit (b, hq, pair): no integer division in the consumers
#   DIAG_FIRST FA3's key order: the diagonal tile first (masked in the prologue, :1150), then the others downwards
# Writes cands/fwd_<name>.cu: QP (1 0 0), QPU (1 1 0), QPD (1 0 1), QPUD (1 1 1), QPoff (0 0 0: must be PF's SASS).
# DEC_PUB variant B (18:10 rewrite): the consumers read this unit's slot at its top (variant A carried next_* registers
# through the epilogue and moved the dense steady loop: 49 changed blocks after register renaming, deep/loop_order_diff.py).
# usage (from profiling_reports/session_2026_10_19/): python3 patches/make_pfq.py
import hashlib, sys

SRC = "scripts/cands/fwd_PF.cu"
src = open(SRC).read()
assert hashlib.md5(src.encode()).hexdigest()[:8] == "1888246a", "fwd_PF.cu is not PF (1888246a)"

EDITS = [
# 1. switch documentation, after KV0_FIRST's
("""#ifndef KV0_FIRST
#define KV0_FIRST 0
#endif
""", """#ifndef KV0_FIRST
#define KV0_FIRST 0
#endif
// Q_PREF (2026-10-19, work doc 45.57-45.58): FA3 prefetches a tile's Q into L2 when its producer STARTS the tile (flash-attention
// 94e22c9 hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:681 `prefetch(params.tma_load_Q, tQgQ)`, before load_K :769 and the QueryEmpty
// sync :776), so its Q load after QueryEmpty hits L2. Ours loaded Q only after qempty, from DRAM. Both release Q at the same point
// (FA3 :1237, ours after the last QK) and both producers already wait there, so only the load's source differed: the consumers'
// wait for Q was 286 cycles per unit vs FA3's 103 (ncu, same box, dense; packed 1.27% of its samples). Q_PREF: the same prefetch
// (cp.async.bulk.prefetch.tensor.2d.L2, as CUTLASS SM90_TMA_LOAD_2D::PREFETCH) of the pair's two Q tiles, both kernels.
#ifndef Q_PREF
#define Q_PREF 0
#endif
// DEC_PUB (2026-10-19, work doc 45.57-45.58): the dense consumers decoded the unit number themselves (decode_u: 7 divisions /
// remainders by run-time values, two I2F.RP / MUFU.RCP sequences, ~190 cycles per unit) right before their wait for Q; FA3
// decodes with precomputed constants. DEC_PUB: the producer, which decodes every unit anyway, publishes b, hq and the pair next to
// the unit number (same slot, same mbarrier, so the same slot-reuse rule); the dense consumers only read them. Packed (A) reads
// its unit table and is unchanged.
#ifndef DEC_PUB
#define DEC_PUB 0
#endif
// DIAG_FIRST (2026-10-19, work doc 45.57-45.58): FA3 walks the key tiles from the diagonal DOWN (mainloop :1150 masks the FIRST
// tile, then fwd_step for n_block_max - 2 .. 0 without a mask); ours walked up, so the masked diagonal was the LAST, pipelined
// step: +583 cycles per unit over a plain step, where FA3's masked first tile costs +285 over an unmasked one (ncu, same box).
// DIAG_FIRST: the n-th tile the producer loads is tile nkv - 1 - n (the stage / slot / barrier order is unchanged, only which
// rows go in), the prologue masks (its tile is the diagonal), no later tile is masked, and the last step keeps the K4 count (its
// mask_tag now means "last tile"). Both kernels, so packed == alone stays bit-identical; the order of the online-softmax
// sums changes, so results are not bit-identical to PF's.
#ifndef DIAG_FIRST
#define DIAG_FIRST 0
#endif
"""),
# 2. the prefetch helper
("""__device__ __forceinline__ void wgmma_fence()  { asm volatile""",
 """#if Q_PREF
__device__ __forceinline__ void tma_prefetch_2d(const void* tmap, int c, int r) {   // Q_PREF: CUTLASS SM90_TMA_LOAD_2D::PREFETCH
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" :: "l"(tmap), "r"(c), "r"(r) : "memory");
}
#endif

__device__ __forceinline__ void wgmma_fence()  { asm volatile"""),
# 3. shared memory: 6 more ints for the decoded units
("""                               + 2 * sizeof(int)                                        // sched_unit[2] (G5)
""", """                               + (DEC_PUB ? 8 : 2) * sizeof(int)                        // sched_unit[2] (G5) + DEC_PUB's sched_dec[2][3]
"""),
("""    uint64_t*      order_bar  = reinterpret_cast<uint64_t*>(sched_unit + 2);  // K1: [2] never waited on (see order_after)
""", """#if DEC_PUB
    int*           sched_dec  = sched_unit + 2;  // DEC_PUB: [2][3] the published unit decoded by the producer (b, hq, pair; dense)
    uint64_t*      order_bar  = reinterpret_cast<uint64_t*>(sched_unit + 8);  // K1: [2] never waited on (see order_after)
#else
    uint64_t*      order_bar  = reinterpret_cast<uint64_t*>(sched_unit + 2);  // K1: [2] never waited on (see order_after)
#endif
"""),
# 4. the producer publishes the decoded unit (first unit, then every claimed unit)
("""        if (lane == 0) { sched_unit[0] = (uidx < total_units) ? uidx : -1; mbar_arrive(&sched_full[0]); }
""", """        if (lane == 0) {
            sched_unit[0] = (uidx < total_units) ? uidx : -1;
#if DEC_PUB
            if constexpr (!CHATS) {
                if (uidx < total_units) {
                    int b_, hq_, hkv_, pr_; decode_u(uidx, b_, hq_, hkv_, pr_);
                    sched_dec[0] = b_; sched_dec[1] = hq_; sched_dec[2] = pr_;
                }
            }
#endif
            mbar_arrive(&sched_full[0]);
        }
"""),
("""            if (lane == 0) { sched_unit[ju & 1] = (uidx < total_units) ? uidx : -1; mbar_arrive(&sched_full[ju & 1]); }
""", """            if (lane == 0) {
                sched_unit[ju & 1] = (uidx < total_units) ? uidx : -1;
#if DEC_PUB
                if constexpr (!CHATS) {
                    if (uidx < total_units) {
                        int b_, hq_, hkv_, pr_; decode_u(uidx, b_, hq_, hkv_, pr_);
                        int* d = &sched_dec[3 * (ju & 1)];
                        d[0] = b_; d[1] = hq_; d[2] = pr_;
                    }
                }
#endif
                mbar_arrive(&sched_full[ju & 1]);
            }
"""),
# 5. the producer's key order
("""                    int row = kv_row_base + n * BLOCK_N;
""", """#if DIAG_FIRST
                    int row = kv_row_base + (nkv - 1 - n) * BLOCK_N;   // DIAG_FIRST: the n-th tile loaded is tile nkv - 1 - n
#else
                    int row = kv_row_base + n * BLOCK_N;
#endif
"""),
# 6. the producer's Q prefetch, at the start of the pair
("""            int n0 = 0;
#if KV0_FIRST
""", """#if Q_PREF
            if (lane_wg == 0) {                        // Q_PREF (FA3 :681): this pair's Q into L2 before its first K / V and qempty
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int q_row = (b * Hq + hq) * S + t0 + (2 * pair + c) * BLOCK_M;
                    tma_prefetch_2d(&tmapQsw, 0, q_row);
                    tma_prefetch_2d(&tmapQsw, 64, q_row);
                }
            }
#endif
            int n0 = 0;
#if KV0_FIRST
"""),
# 7. the consumers read the decoded unit with the unit number
("""                decode_u(uidx, b, hq, hkv, pair);
            }
            const int q_tile = 2 * pair + c;
""", """#if DEC_PUB
                {   // DEC_PUB: decoded by the producer into this unit's slot (ju & 1), which it rewrites only after both groups'
                    // last QK of this unit (the slot-reuse rule of G5); no division here, nothing carried through the epilogue
                    const volatile int* dec = reinterpret_cast<volatile int*>(&sched_dec[3 * (ju & 1)]);
                    b = __shfl_sync(0xffffffffu, dec[0], 0);
                    hq = __shfl_sync(0xffffffffu, dec[1], 0);
                    pair = __shfl_sync(0xffffffffu, dec[2], 0);
                    hkv = 0;                            // not used by the consumers
                }
#else
                decode_u(uidx, b, hq, hkv, pair);
#endif
            }
            const int q_tile = 2 * pair + c;
"""),
# 8. the consumers' mask: on the first tile (DIAG_FIRST), never on the later ones
("""                if (my_nkv == 1) softmax_tile(0, std::true_type{});   // tile 0 is the diagonal one only when it is the last
                else             softmax_tile(0, std::false_type{});
""", """#if DIAG_FIRST
                softmax_tile(my_nkv - 1, std::true_type{});   // DIAG_FIRST: the first tile IS the diagonal one (FA3 :1150)
#else
                if (my_nkv == 1) softmax_tile(0, std::true_type{});   // tile 0 is the diagonal one only when it is the last
                else             softmax_tile(0, std::false_type{});
#endif
"""),
("""                softmax_tile(n, mask_tag);              // ... while P@V(n-1) runs
""", """#if DIAG_FIRST
                softmax_tile(n, std::false_type{});     // DIAG_FIRST: no later tile is masked (mask_tag marks the LAST tile: K4)
#else
                softmax_tile(n, mask_tag);              // ... while P@V(n-1) runs
#endif
"""),
]

out = src
for old, new in EDITS:
    n = out.count(old)
    assert n == 1, f"anchor found {n} times: {old[:80]!r}"
    out = out.replace(old, new)

HEAD = "#define KV0_FIRST 1\n"
assert out.startswith(HEAD)
for name, (q, d, g) in {"QP": (1, 0, 0), "QPU": (1, 1, 0), "QPD": (1, 0, 1), "QPUD": (1, 1, 1), "QPoff": (0, 0, 0)}.items():
    top = (f"#define Q_PREF {q}   // 2026-10-19 candidate {name}: PF + Q_PREF {q} / DEC_PUB {d} / DIAG_FIRST {g} (work doc 45.58)\n"
           f"#define DEC_PUB {d}\n#define DIAG_FIRST {g}\n")
    text = top + out
    open(f"scripts/cands/fwd_{name}.cu", "w").write(text)
    print(f"fwd_{name}.cu {hashlib.md5(text.encode()).hexdigest()[:8]} ({len(text)} bytes)")
