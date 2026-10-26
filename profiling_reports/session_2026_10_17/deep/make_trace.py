# 2026-10-17: add the V62_TRACE measurement stamps to a forward source (F3c or K1 family). Output = same kernel + clock stamps:
#  per (CTA, consumer group, query pair) record: globaltimer at pair start / end, clock64 at the phase borders (pair start, Q
#  landed, tile 0 K/V landed, prologue done, loop done, last P.V + tail + fetch done, epilogue done), cycles in fetch_unit, and
#  per-tile accumulators over the loop's steps (32-bit clock): wait K/V + baton | issue + wait QK | softmax (+pack, K1) |
#  wait P.V | rescale (+pack, F3c) ; steps counted. The launcher dumps launch V62_TRACE_AT (default 900) to V62_TRACE_FILE.
# K1 family (2026-10-17, K1_SPLIT build): the step's 'softmax' = part A (+pack, order arrive), 'rescale' = part B + rescale
# + pack when K1_SPLIT < 64.
# usage: python3 make_trace.py <in.cu> <out.cu>       (asserts every pattern: a source that does not match fails loudly)
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
K1 = "auto step = [&](const int n, auto mask_tag, auto buf_tag)" in s
def rep(o, n, c=1):
    global s
    assert s.count(o) == c, (s.count(o), o[:110]); s = s.replace(o, n)
rep("#include <cassert>\n", "#include <cassert>\n#include <cstdlib>\n")
rep("__device__ int g_v62_next_unit = 0;\n", """__device__ int g_v62_next_unit = 0;
#ifdef V62_TRACE
constexpr int TR_MAXREC = 64, TR_W = 24;   // records per (CTA, group); u64 words per record
__device__ unsigned long long g_v62_trace[NUM_SMS * 2 * TR_MAXREC * TR_W];
__device__ unsigned long long g_v62_trace_cta[NUM_SMS * 2 * 4];   // per (CTA, group): start gt, end gt, records, 0
constexpr int TL_STEPS = 64, TL_W = 8;     // timeline: every loop step of each group's FIRST pair, 32-bit clocks s0..s5, n
__device__ unsigned int g_v62_tl[NUM_SMS * 2 * TL_STEPS * TL_W];
__device__ __forceinline__ unsigned long long tr_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned int tr_smid() { unsigned int r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned int tr_clk32() { unsigned int r; asm volatile("mov.u32 %0, %%clock;" : "=r"(r)); return r; }
#define TR_CLK(v) const unsigned long long v = clock64()
#define TR_S(v) const unsigned int v = tr_clk32()
#define TR_SET(v) v = tr_clk32()
#else
#define TR_SET(v)
#define TR_CLK(v)
#define TR_S(v)
#endif
""")
rep("""        auto fetch_unit = [&](int k) {
            mbar_wait(&sched_full[k & 1], (k >> 1) & 1);
            asm volatile("" ::: "memory");                   // (mbar_wait has no memory clobber: read after it)
            return __shfl_sync(0xffffffffu, *reinterpret_cast<volatile int*>(&sched_unit[k & 1]), 0);
        };""", """#ifdef V62_TRACE
        unsigned long long tr_fetch = 0;          // cycles spent in fetch_unit since the last record
        int tr_k = 0;
        if (lane_wg == 0) g_v62_trace_cta[(blockIdx.x * 2 + c) * 4 + 0] = tr_gt();
#endif
        auto fetch_unit = [&](int k) {
            TR_CLK(tf0);
            mbar_wait(&sched_full[k & 1], (k >> 1) & 1);
            asm volatile("" ::: "memory");                   // (mbar_wait has no memory clobber: read after it)
            const int r = __shfl_sync(0xffffffffu, *reinterpret_cast<volatile int*>(&sched_unit[k & 1]), 0);
#ifdef V62_TRACE
            tr_fetch += clock64() - tf0;
#endif
            return r;
        };""")
rep("""            if (uidx < 0) break;
            int4 un{};""", """            if (uidx < 0) break;
            int4 un{};
#ifdef V62_TRACE
            const unsigned long long tr_gt0 = tr_gt();
#endif""")
rep("""            const int q_tile = 2 * pair + c;
""", """            const int q_tile = 2 * pair + c;
            TR_CLK(t0c);                                  // V62_TRACE: pair start (decoded)
#ifdef V62_TRACE
            unsigned int ac_kv = 0, ac_qk = 0, ac_sm = 0, ac_pv = 0, ac_rs = 0, ac_n = 0;   // per-step accumulators
#endif
""")
rep("""            mbar_wait(&qfull[c], j & 1);
""", """            mbar_wait(&qfull[c], j & 1);
            TR_CLK(t1c);                                  // V62_TRACE: Q landed
""")
rep("""            if (nkv_shared > 0) mbar_wait(&full[stage], ph);   // tile 0's K/V (the inactive tail is paced by it too)
""", """            if (nkv_shared > 0) mbar_wait(&full[stage], ph);   // tile 0's K/V (the inactive tail is paced by it too)
            TR_CLK(t2c);                                  // V62_TRACE: tile 0's K/V landed
""")
if K1:
    rep("""                pack_p(std::integral_constant<int, 0>{});   // P(0) -> pa0
            }
""", """                pack_p(std::integral_constant<int, 0>{});   // P(0) -> pa0
            }
            TR_CLK(t3c);                                  // V62_TRACE: prologue done
""")
    rep("""                stage = __shfl_sync(0xffffffffu, stage, 0);   // F2: keep the stage (and the descriptors) uniform
                const int prev = stage;                 // stage of tile n-1: its V is read by P@V(n-1)""", """                TR_S(s0);
#ifdef V62_TRACE
                unsigned int s3 = 0, s4 = 0;            // set at whichever wait for P@V(n-1) this build has
#endif
                stage = __shfl_sync(0xffffffffu, stage, 0);   // F2: keep the stage (and the descriptors) uniform
                const int prev = stage;                 // stage of tile n-1: its V is read by P@V(n-1)""")
    rep("""                if (c == 0) mbar_wait(&full[nstage], nph);   // K/V of tile n have landed
                if (c == 0) pp_sync1(); else pp_sync2();
                issue_qk(&sK[nstage * TILE_K]);         // group: QK(n)""", """                if (c == 0) mbar_wait(&full[nstage], nph);   // K/V of tile n have landed
                if (c == 0) pp_sync1(); else pp_sync2();
                TR_S(s1);
                issue_qk(&sK[nstage * TILE_K]);         // group: QK(n)""")
    rep("""                wgmma_wait1();                          // QK(n) done, P@V(n-1) still running
""", """                wgmma_wait1();                          // QK(n) done, P@V(n-1) still running
                TR_S(s2);
""")
    # every wait for P@V(n-1) inside the step (3 branches): s3 before, s4 after
    import re
    # any indentation (the K8 refactor moved the single-set waits into if-constexpr blocks, 4 spaces deeper)
    pat = re.compile(r"^( +)(wgmma_wait0\(\);\s+// P@V\(n-1\) done[^\n]*\n)(\1if \(lane_wg == 0\) mbar_arrive\(&empty\[prev\]\);)", re.M)
    nwait = len(pat.findall(s))
    assert nwait in (3, 4), ("step waits", nwait)   # 3 before the K5 single-set split path existed, 4 after
    s = pat.sub(lambda m: m.group(1) + "TR_SET(s3);\n" + m.group(1) + m.group(2) + m.group(1) + "TR_SET(s4);\n" + m.group(3), s)
    rep("""                ++g;
                stage = nstage; ph = nph;
            };""", """                TR_S(s5);
#ifdef V62_TRACE
                ac_kv += s1 - s0; ac_qk += s2 - s1; ac_sm += s3 - s2; ac_pv += s4 - s3; ac_rs += s5 - s4; ++ac_n;
                if (tr_k == 0 && lane_wg == 0 && n < TL_STEPS) {   // timeline of the first pair
                    unsigned int* q = &g_v62_tl[((blockIdx.x * 2 + c) * TL_STEPS + n) * TL_W];
                    q[0] = s0; q[1] = s1; q[2] = s2; q[3] = s3; q[4] = s4; q[5] = s5; q[6] = (unsigned)n; q[7] = (unsigned)my_nkv;
                }
#endif
                ++g;
                stage = nstage; ph = nph;
            };""")
    if "            int s_last = 0;                             // K4" in s:
        rep("""            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            int s_last = 0;""", """            TR_CLK(t4c);                                  // V62_TRACE: main loop done
            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            int s_last = 0;""")
    else:
        rep("""            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            if (my_nkv > 0) {                           // the last tile's P@V""", """            TR_CLK(t4c);                                  // V62_TRACE: main loop done
            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            if (my_nkv > 0) {                           // the last tile's P@V""")
else:   # F3c
    rep("""                wgmma_wait0();
                softmax_tile(0);
                rescale_pack();
            }
""", """                wgmma_wait0();
                softmax_tile(0);
                rescale_pack();
            }
            TR_CLK(t3c);                                  // V62_TRACE: prologue done
""")
    rep("""            for (int n = 1; n < my_nkv; ++n) {
                stage = __shfl_sync(0xffffffffu, stage, 0);   // F2: keep the stage (and the descriptors) uniform""", """            for (int n = 1; n < my_nkv; ++n) {
                TR_S(s0);
                stage = __shfl_sync(0xffffffffu, stage, 0);   // F2: keep the stage (and the descriptors) uniform""")
    rep("""                mbar_wait(&full[nstage], nph);          // K/V of tile n have landed
                if (c == 0) pp_sync1(); else pp_sync2();
                issue_qk(&sK[nstage * TILE_K]);         // group: QK(n)""", """                mbar_wait(&full[nstage], nph);          // K/V of tile n have landed
                if (c == 0) pp_sync1(); else pp_sync2();
                TR_S(s1);
                issue_qk(&sK[nstage * TILE_K]);         // group: QK(n)""")
    rep("""                wgmma_wait1();                          // QK(n) done, P@V(n-1) still running
                softmax_tile(n);                        // ... while P@V(n-1) runs
                wgmma_wait0();                          // P@V(n-1) done
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                rescale_pack();""", """                wgmma_wait1();                          // QK(n) done, P@V(n-1) still running
                TR_S(s2);
                softmax_tile(n);                        // ... while P@V(n-1) runs
                TR_S(s3);
                wgmma_wait0();                          // P@V(n-1) done
                TR_S(s4);
                if (lane_wg == 0) mbar_arrive(&empty[prev]);
                rescale_pack();
                TR_S(s5);
#ifdef V62_TRACE
                ac_kv += s1 - s0; ac_qk += s2 - s1; ac_sm += s3 - s2; ac_pv += s4 - s3; ac_rs += s5 - s4; ++ac_n;
#endif""")
    rep("""            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            if (my_nkv > 0) {                           // the last tile's P@V""", """            TR_CLK(t4c);                                  // V62_TRACE: main loop done
            if (lane_wg == 0) mbar_arrive(&qempty[c]);
            if (my_nkv > 0) {                           // the last tile's P@V""")
rep("""
            l0 = quad_sum(l0); l1 = quad_sum(l1);""", """
            TR_CLK(t5c);                                  // V62_TRACE: last P@V + inactive tail + fetch done
            l0 = quad_sum(l0); l1 = quad_sum(l1);""")
# record write at the end of the pair (before ++j of the pair loop): the LSE block is the last code before it
marker = """                if (st1) LSE[(long long)(b * Hq + hq) * S + t0 + gR1] = (m1 * scale2) * LN2 + logf(l1);
            }
"""
rep(marker, marker + """#ifdef V62_TRACE
            if (lane_wg == 0 && tr_k < TR_MAXREC) {
                const unsigned long long t6c = clock64();
                unsigned long long* r = &g_v62_trace[((blockIdx.x * 2 + c) * TR_MAXREC + tr_k) * TR_W];
                r[0] = tr_gt0; r[1] = t0c; r[2] = t1c; r[3] = t2c; r[4] = t3c; r[5] = t4c; r[6] = t5c; r[7] = t6c;
                r[8] = tr_gt(); r[9] = tr_fetch; r[10] = (unsigned long long)uidx;
                r[11] = (unsigned long long)tr_smid() | ((unsigned long long)half << 8) | ((unsigned long long)(my_nkv & 0xffff) << 16)
                      | ((unsigned long long)(nkv_shared & 0xffff) << 32) | ((unsigned long long)(pair & 0xffff) << 48);
                r[12] = (unsigned long long)(CHATS ? L : S);
                r[13] = ac_kv; r[14] = ac_qk; r[15] = ac_sm; r[16] = ac_pv; r[17] = ac_rs; r[18] = ac_n;
            }
            ++tr_k; tr_fetch = 0;
#endif
""")
rep("""            ++j;
        }
        }
    }
}
""", """            ++j;
        }
        }
#ifdef V62_TRACE
        if (lane_wg == 0) {
            g_v62_trace_cta[(blockIdx.x * 2 + c) * 4 + 1] = tr_gt();
            g_v62_trace_cta[(blockIdx.x * 2 + c) * 4 + 2] = (unsigned long long)tr_k;
        }
#endif
    }
}
""")
rep("""    const int4* units = reinterpret_cast<const int4*>(chat_units);
""", """    const int4* units = reinterpret_cast<const int4*>(chat_units);
#ifdef V62_TRACE
    static int tr_calls = 0;
    const char* tr_file = std::getenv("V62_TRACE_FILE");
    const bool tr_this = tr_file && ++tr_calls == (std::getenv("V62_TRACE_AT") ? std::atoi(std::getenv("V62_TRACE_AT")) : 900);
    if (tr_this) cudaStreamSynchronize(stream);
#endif
""")
rep("""        gqa62::gqa_v62_sm90<gqa62::MODE_DENSE><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, 0);
    }
}
""", """        gqa62::gqa_v62_sm90<gqa62::MODE_DENSE><<<grid, block, smem_bytes, stream>>>(tmapKsw, tmapVsw, tmapQsw, Q, O, LSE, B, Hq, Hkv, S, scale, nullptr, 0);
    }
#ifdef V62_TRACE
    if (tr_this) {   // header: 16 int64 (magic, mode, B, Hq, Hkv, S, n_units, NUM_SMS, TR_MAXREC, TR_W), then cta[], then records
        cudaStreamSynchronize(stream);
        static unsigned long long rec[gqa62::NUM_SMS * 2 * gqa62::TR_MAXREC * gqa62::TR_W], cta[gqa62::NUM_SMS * 2 * 4];
        static unsigned int tl[gqa62::NUM_SMS * 2 * gqa62::TL_STEPS * gqa62::TL_W];
        cudaMemcpyFromSymbol(rec, gqa62::g_v62_trace, sizeof rec);
        cudaMemcpyFromSymbol(cta, gqa62::g_v62_trace_cta, sizeof cta);
        cudaMemcpyFromSymbol(tl, gqa62::g_v62_tl, sizeof tl);
        long long hdr[16] = {0x76363254, chat_units ? 1 : 0, B, Hq, Hkv, S, n_units, gqa62::NUM_SMS, gqa62::TR_MAXREC, gqa62::TR_W};
        if (FILE* f = std::fopen(tr_file, "wb")) {
            std::fwrite(hdr, sizeof hdr, 1, f); std::fwrite(cta, sizeof cta, 1, f); std::fwrite(rec, sizeof rec, 1, f);
            std::fwrite(tl, sizeof tl, 1, f); std::fclose(f);   // timeline after the records (older readers ignore it)
            std::fprintf(stderr, "V62_TRACE: launch %d written to %s\\n", tr_calls, tr_file);
        }
    }
#endif
}
""")
open(dst, "w").write("#define V62_TRACE 1   // 2026-10-17 measurement build (deep/make_trace.py): clock stamps, output unchanged\n" + s)
print("written", dst, "(K1 family)" if K1 else "(F3c)")
