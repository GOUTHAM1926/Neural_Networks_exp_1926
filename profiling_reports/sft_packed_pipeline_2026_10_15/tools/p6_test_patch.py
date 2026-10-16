#!/usr/bin/env python3
"""P6 (2026-10-15): test_varlen_attention.cu follows the API: design B (seg_start) is gone, so G8/G9 (the B groups)
are deleted; the helpers the A groups use stay; G11 now runs under Vj2 too (A runs on Vj2 there, as do the dense
"alone" runs it compares with); timings are dense vs A. Every replacement must match exactly once.
usage: p6_test_patch.py <Tensor-Implementations dir>"""
import sys, os, re
TI = sys.argv[1]
p = os.path.join(TI, "Tests/kernel_tests/test_varlen_attention.cu")
s = open(p, encoding="utf-8").read()

def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:120])
    s = s.replace(old, new)

def block(start_marker, end_marker):
    """(a, b): s[a:b] runs from the '// ====' line before start_marker (or start_marker itself if it is code) up to
    the '// ====' line before end_marker."""
    assert s.count(start_marker) == 1 and s.count(end_marker) == 1, (start_marker[:60], end_marker[:60])
    a = s.index(start_marker); b = s.index(end_marker)
    if start_marker.startswith("//"): a = s.rindex("// ====", 0, a)
    if end_marker.startswith("//"): b = s.rindex("// ====", 0, b)
    assert a < b
    return a, b

def sub_in(start_marker, end_marker, pairs):
    """replacements only inside [start_marker, end_marker)"""
    global s
    a = s.index(start_marker); b = s.index(end_marker, a)
    part = s[a:b]
    for old, new in pairs:
        n = part.count(old)
        assert n == 1, (start_marker[:40], n, old[:120])
        part = part.replace(old, new)
    s = s[:a] + part + s[b:]

# ---- header comment ------------------------------------------------------------------------------------------------
rep("""//     gqa_flash_attn_backward_qkv -> kernel Vj1d (GQA_bwd_vj1d_sm90.cu, default mode)
//   Step 2 runs ONLY today's kernels. It proves the harness itself (references, pass
//   rule, data loads) before any kernel changes. Steps 3-5 add the packed-row variants
//   (B: packed row + chat mask, A: per-chat offsets) to the same groups.""",
"""//     gqa_flash_attn_backward_qkv -> kernel Vj2 (GQA_bwd_vj2_sm90.cu, the default since 2026-10-15;
//                                    USE_VJ2=0 -> Vj1d, GQA_bwd_vj1d_sm90.cu)
//   Step 2 runs ONLY today's kernels. It proves the harness itself (references, pass
//   rule, data loads) before any kernel changes. Steps 3-5 added the packed-row variants
//   (B: packed row + chat mask, A: per-chat offsets) to the same groups. B was removed on
//   2026-10-15 (A is the cu_seqlens way and faster), so the packed groups test A only.""")
rep("""//   G8 B forward (Step 3)  : packed rows through gqa_flash_attn_forward_qkv(seg_start):
//                            O/LSE of every chat AND pad tail vs the per-segment reference;
//                            tile-aligned chats bitwise = alone; isolation (one chat changed,
//                            the others bitwise unchanged); offsets (D11); API refusals.
//   G9 B backward (Step 4) : packed rows through gqa_flash_attn_backward_qkv(seg_start), after
//                            the G8 forward: dQ/dK/dV (+O/LSE) of every chat AND pad tail (random
//                            dO everywhere) vs the per-segment reference; 1-token segments vs the
//                            derived bound; tile-aligned chats = alone (dK/dV bitwise, dQ atol);
//                            isolation incl. dO; refusals (needs the seg_start message); repeat
//                            x1000 on 8 real rows; tile-visit model + timing packed vs dense.
//   G10 A forward (Step 5) : packed rows through gqa_flash_attn_forward_qkv(GqaChatPlan): O/LSE of
//                            every segment vs the reference; EVERY segment bitwise = alone (A cuts
//                            tiles at the segment start like dense at row 0); isolation; refusals
//                            incl. the plan builder; forward time dense vs B vs A on real rows.
//   G11 A backward (Step 5): the same plan through gqa_flash_attn_backward_qkv, G9's inputs: dQ/dK/dV
//                            of every segment vs the reference; EVERY segment = alone (O/LSE/dK/dV
//                            bitwise, dQ atol); isolation incl. dO; refusals; repeat; bwd time.""",
"""//   G8, G9 (design B forward / backward, Steps 3-4): REMOVED on 2026-10-15 together with design B
//                            (one seg_start value per token). G10/G11 run the same checks on
//                            design A, and compare EVERY segment with its run alone (G8/G9 could
//                            only do that for tile-aligned chats).
//   G10 A forward (Step 5) : packed rows through gqa_flash_attn_forward_qkv(GqaChatPlan): O/LSE of
//                            every segment vs the reference; EVERY segment bitwise = alone (A cuts
//                            tiles at the segment start like dense at row 0); isolation; refusals
//                            incl. the plan builder; forward time dense vs A on real rows.
//   G11 A backward (Step 5): the same plan through gqa_flash_attn_backward_qkv (Vj2 by default, Vj1d with
//                            USE_VJ2=0; the alone runs use the same kernel), on the removed G9's inputs:
//                            dQ/dK/dV of every segment vs the reference; EVERY segment = alone
//                            (O/LSE/dK/dV bitwise, dQ atol); isolation incl. dO; refusals; repeat; bwd time.""")
rep("""//   VARLEN_ONLY=G8        run only G8 (design B forward); with VARLEN_QUICK=1 it is ~1-2 min
""", "")
rep("""//   VARLEN_ONLY=BENCH     Step 6 timing of our dense / B / A (VARLEN_BENCH_ITERS, VARLEN_BENCH_EXPORT=<file>)
//   VARLEN_ONLY=G9        run only G9 (design B backward); with USE_FLASH=1 or USE_DETERMINISTIC_DQ=1
//                         it runs only the mode refusal (seconds)""",
"""//   VARLEN_ONLY=BENCH     Step 6 timing of our dense / A (VARLEN_BENCH_ITERS, VARLEN_BENCH_EXPORT=<file>)
//   VARLEN_ONLY=G11       run only G11 (design A backward); with USE_FLASH=1 or USE_DETERMINISTIC_DQ=1
//                         it runs only the mode refusal (seconds)""")

# ---- G9-only globals ---------------------------------------------------------------------------------------------
rep("""// G9: VARLEN_DIAG_CASE = which layout the diagnostic row belongs to (substring of its name, default
// "D10"); VARLEN_CASE = run only the G9 layouts whose name contains it (no D10/repeat/timing/refusals).
static std::string g_diag_case = "D10", g_case;
// G9: VARLEN_DIAG_LIST="row:t0,row:t0,..." = segments to diagnose (packed vs alone for dQ/dK/dV + a dV
// arithmetic model). With a list, G9 runs only the 8-row batches that contain those rows (same seeds
// as the full run) and skips the repeat/timing/refusal parts.
static std::vector<std::pair<int, int>> g_diag_list;
""", """// VARLEN_CASE = run only the layouts whose name contains it (G10-G13, BENCH; no D10/repeat/timing/refusals).
static std::string g_case;
""")
rep("""  if (const char* e = std::getenv("VARLEN_DIAG_CASE")) g_diag_case = e;
  if (const char* e = std::getenv("VARLEN_CASE")) g_case = e;
  if (const char* e = std::getenv("VARLEN_DIAG_LIST")) {
    std::string t = e;
    for (size_t a = 0; a < t.size();) {
      size_t b = t.find(',', a);
      if (b == std::string::npos) b = t.size();
      const std::string it = t.substr(a, b - a);
      const size_t c = it.find(':');
      if (c != std::string::npos) g_diag_list.push_back({std::atoi(it.substr(0, c).c_str()), std::atoi(it.substr(c + 1).c_str())});
      a = b + 1;
    }
  }""", """  if (const char* e = std::getenv("VARLEN_CASE")) g_case = e;""")

# ---- known list: the G9 entries can no longer match ------------------------------------------------------------------
rep("""    // Same mechanism, found by G9 quick on 2026-10-10 (random inputs, seed 906). Proven NOT a packing""",
    """    // (2026-10-15) G9 (design B) is removed: the G9 entries below can no longer match. Kept as the record of
    // what B's backward on Vj1d showed.
    // Same mechanism, found by G9 quick on 2026-10-10 (random inputs, seed 906). Proven NOT a packing""")

# ---- G8 helpers that A uses stay; B's go -----------------------------------------------------------------------------
a, b = block("// G8 (Step 3, 2026-10-10): design B forward on PACKED rows", "// New random values for tokens [t0, t1) of row b")
s = s[:a] + """// ===========================================================================
// Helpers of the packed-row groups (G10-G13). G8/G9 (design B) were removed on 2026-10-15.
// ===========================================================================
""" + s[b:]
a, b = block("static void g8_layout(", "// G10 (Step 5, 2026-10-10): design A forward on PACKED rows")
s = s[:a] + s[b:]

# ---- G10 ---------------------------------------------------------------------------------------------------------
rep("""//   d) refusals, each with its own message: plan + seg_start; plan on paths v62 cannot take; plan
//      built for another shape; plan builder: gap, overlap, segment outside its row, row not covered.
//   e) time: dense vs B vs A forward on 8 real rows from three places of the packed set (first =
//      longest chats, middle, last = shortest), median of 20, interleaved (reported, not gated).""",
"""//   d) refusals, each with its own message: plan on paths v62 cannot take; plan built for another
//      shape; plan builder: gap, overlap, segment outside its row, row not covered.
//   e) time: dense vs A forward on 8 real rows from three places of the packed set (first =
//      longest chats, middle, last = shortest), median of 20, interleaved (reported, not gated).""")
rep("""                                    p.hd, /*is_causal=*/true, /*seg_start=*/nullptr, plan);""",
    """                                    p.hd, /*is_causal=*/true, plan);""")
sub_in("static void g10_refusals(", "static void g10_chats_forward(", [
    ("""  auto fwd = [&](int B, int Hq, int Hkv, int T, int hd, bool causal, const int32_t* seg, const otc::GqaChatPlan* pl) {""",
     """  auto fwd = [&](int B, int Hq, int Hkv, int T, int hd, bool causal, const otc::GqaChatPlan* pl) {"""),
    ("""B, Hq, Hkv, T, hd, causal, seg, pl);""", """B, Hq, Hkv, T, hd, causal, pl);"""),
    ("""    g10_refuse(sum, "this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", [&] { fwd(1, 2, 1, 256, 128, true, nullptr, &pl.p); });""",
     """    g10_refuse(sum, "this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", [&] { fwd(1, 2, 1, 256, 128, true, &pl.p); });"""),
    ("""    DBuf<int32_t> ds;
    std::vector<int32_t> hsg(256, 0);
    ds.ensure(hsg.size());
    CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    g10_refuse(sum, "plan AND seg_start", "not both", [&] { fwd(1, 2, 1, 256, 128, true, ds.p, &pl.p); });
    g10_refuse(sum, "is_causal = false", "chats", [&] { fwd(1, 2, 1, 256, 128, false, nullptr, &pl.p); });
    g10_refuse(sum, "head_dim 64", "chats", [&] { fwd(1, 2, 1, 256, 64, true, nullptr, &pl.p); });
    g10_refuse(sum, "plan for B=1 used with B=2", "another shape", [&] { fwd(2, 2, 1, 256, 128, true, nullptr, &pl.p); });
    g10_refuse(sum, "plan for 2 heads used with 4", "another shape", [&] { fwd(1, 4, 1, 256, 128, true, nullptr, &pl.p); });""",
     """    g10_refuse(sum, "is_causal = false", "chats", [&] { fwd(1, 2, 1, 256, 128, false, &pl.p); });
    g10_refuse(sum, "head_dim 64", "chats", [&] { fwd(1, 2, 1, 256, 64, true, &pl.p); });
    g10_refuse(sum, "plan for B=1 used with B=2", "another shape", [&] { fwd(2, 2, 1, 256, 128, true, &pl.p); });
    g10_refuse(sum, "plan for 2 heads used with 4", "another shape", [&] { fwd(1, 4, 1, 256, 128, true, &pl.p); });"""),
    ("""    g10_refuse(sum, "T = 320 (not a multiple of 256)", "chats", [&] { fwd(1, 2, 1, 320, 128, true, nullptr, &pl.p); });""",
     """    g10_refuse(sum, "T = 320 (not a multiple of 256)", "chats", [&] { fwd(1, 2, 1, 320, 128, true, &pl.p); });"""),
])
a = s.index("// e) forward time: dense vs B vs A on the same 8 rows (free clocks; reported, not gated)")
b = s.index("static void g10_chats_forward(")
s = s[:a] + """// e) forward time: dense vs A on the same 8 rows (free clocks; reported, not gated)
static void g10_time(const std::vector<std::vector<int>>& rows, int first, Summary& sum) {
  Layout lo{"D10 rows " + std::to_string(first) + "-" + std::to_string(first + 7), 4096, {}};
  lo.rows.assign(rows.begin() + first, rows.begin() + first + 8);
  const Problem p{8, 4096, 18, 6, 128};
  Inputs in;
  make_inputs(in, p, 1010 + first);
  PlanHolder pl;
  build_plan(lo, p.Hq, pl.p);
  // work in key-tile visits for one head of every row: per 64-row q-tile, the 128-key tiles it visits
  long long wa = 0, wd = 0;
  for (const Segment& g : packed_segments(lo)) {
    const int nq = 2 * ((g.L + 127) / 128);
    for (int qt = 0; qt < nq; ++qt) wa += (qt * 64 >= g.L) ? 0 : (qt * 64 + 63) / 128 + 1;
  }
  for (int qt = 0; qt < p.T / 64; ++qt) wd += (long long)p.B * ((qt * 64 + 63) / 128 + 1);
  KOut kd, ka;
  kd.ensure(p); ka.ensure(p);
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  auto timed = [&](const std::function<void()>& f) {
    float ms = 0;
    CUDA_CHECK(cudaEventRecord(a));
    f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  };
  auto f_dense = [&] { today_fwd(in, kd); };
  auto f_a = [&] {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(ka.O.p), ka.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, &pl.p);
  };
  std::vector<float> td, ta;
  try {
    for (int i = 0; i < 5; ++i) { f_dense(); f_a(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; ++i) { td.push_back(timed(f_dense)); ta.push_back(timed(f_a)); }
  } catch (const std::exception& e) {
    sum.record(false, std::string("G10 timing threw: ") + e.what());
    return;
  }
  auto med = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? (double)v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  };
  const double md = med(td), maa = med(ta);
  std::printf("  G10 %-16s work (key-tile visits vs dense): A %.1f%% | fwd median: dense %.4f ms, A %.4f ms (%.1f%%)\\n",
              lo.name.c_str(), 100.0 * wa / wd, md, maa, 100.0 * maa / md);
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
}

""" + s[b:]
rep("""  std::printf("\\n===== G10 design A forward: per-segment plan (chats + pad tails) vs reference, alone, B =====\\n");""",
    """  std::printf("\\n===== G10 design A forward: per-segment plan (chats + pad tails) vs reference, alone =====\\n");""")

# ---- G11 ---------------------------------------------------------------------------------------------------------
rep("""// G11 (Step 5, 2026-10-10): design A BACKWARD (gqa_flash_attn_backward_qkv with the same GqaChatPlan
// -> gqa_bwd_vj1d<MODE_CHATS>), after the A forward. Same inputs as G9 (same seeds).""",
    """// G11 (Step 5, 2026-10-10): design A BACKWARD (gqa_flash_attn_backward_qkv with the same GqaChatPlan
// -> gqa_bwd_vj2_kernel<CHATS> by default since 2026-10-15, gqa_bwd_vj1d<MODE_CHATS> with USE_VJ2=0), after the
// A forward. Same inputs as the removed G9 had (same seeds).""")
rep("""//   d) refusals (plan + seg_start, paths Vj1d cannot take, wrong shape; USE_FLASH / DET_DQ runs).
//   e) repeat x N on 8 real rows (bitwise O/LSE/dK/dV), f) backward time dense vs B vs A.""",
    """//   d) refusals (paths Vj2/Vj1d cannot take, wrong shape; USE_FLASH / DET_DQ runs).
//   e) repeat x N on 8 real rows (bitwise O/LSE/dK/dV), f) backward time dense vs A.""")
sub_in("static bool run_chats(", "static void g11_layout(", [
    ("""                                    p.hd, true, nullptr, plan);""", """                                    p.hd, true, plan);"""),
    ("""mb(o.dK.p), mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, nullptr, plan);""",
     """mb(o.dK.p), mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, plan);"""),
])
sub_in("static void g11_repeat_and_time(", "static void g11_chats_backward(", [
    ("""    const std::vector<int32_t> hs = build_seg_start(lo);
    DBuf<int32_t> d_seg;
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
""", ""),
    ("""    KOut kd, kb, ka;
    kd.ensure(p); kb.ensure(p); ka.ensure(p);
    today_fwd(in, kd);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(kb.O.p), kb.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, d_seg.p);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(ka.O.p), ka.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, nullptr, &pl.p);""",
     """    KOut kd, ka;
    kd.ensure(p); ka.ensure(p);
    today_fwd(in, kd);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(ka.O.p), ka.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, &pl.p);"""),
    ("""    auto bwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {""",
     """    auto bwd = [&](KOut& o, const otc::GqaChatPlan* pp) {"""),
    ("""mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);""", """mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, pp);"""),
    ("""    std::vector<float> td, tb, ta;
    for (int i = 0; i < 5; ++i) { bwd(kd, nullptr, nullptr); bwd(kb, d_seg.p, nullptr); bwd(ka, nullptr, &pl.p); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; ++i) {
      td.push_back(timed([&] { bwd(kd, nullptr, nullptr); }));
      tb.push_back(timed([&] { bwd(kb, d_seg.p, nullptr); }));
      ta.push_back(timed([&] { bwd(ka, nullptr, &pl.p); }));
    }""", """    std::vector<float> td, ta;
    for (int i = 0; i < 5; ++i) { bwd(kd, nullptr); bwd(ka, &pl.p); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; ++i) {
      td.push_back(timed([&] { bwd(kd, nullptr); }));
      ta.push_back(timed([&] { bwd(ka, &pl.p); }));
    }"""),
    ("""    const auto bw = bwd_tiles(hs, p.B, p.T);
    long long wa = 0;
    for (const Segment& g : packed_segments(lo)) { const int nk = (g.L + 63) / 64; wa += (long long)nk * (nk + 1) / 2; }
    const double md = med(td), mbb = med(tb), maa = med(ta);
    std::printf("  G11 %-16s work (q-tile visits vs dense): B %.1f%%, A %.1f%% | bwd median: dense %.4f ms, B %.4f ms (%.1f%%), A %.4f ms (%.1f%%)\\n",
                lo.name.c_str(), 100.0 * bw.first / bw.second, 100.0 * wa / bw.second, md, mbb, 100.0 * mbb / md, maa, 100.0 * maa / md);""",
     """    // work in q-tile visits of 64-key tiles for one head of every row: dense = B * n(n+1)/2, n = T/64
    const long long nk64 = p.T / 64, wd = (long long)p.B * nk64 * (nk64 + 1) / 2;
    long long wa = 0;
    for (const Segment& g : packed_segments(lo)) { const int nk = (g.L + 63) / 64; wa += (long long)nk * (nk + 1) / 2; }
    const double md = med(td), maa = med(ta);
    std::printf("  G11 %-16s work (q-tile visits vs dense): A %.1f%% | bwd median (%s): dense %.4f ms, A %.4f ms (%.1f%%)\\n",
                lo.name.c_str(), 100.0 * wa / wd, bwd_mode_name().c_str(), md, maa, 100.0 * maa / md);"""),
])
sub_in("static void g11_chats_backward(", "// BENCH (Step 6, 2026-10-12)", [
    ("""  auto refuse = [&](const char* name, const char* expect, int B, int Hq, int T, int hd, bool causal, bool with_seg, int planB) {""",
     """  auto refuse = [&](const char* name, const char* expect, int B, int Hq, int T, int hd, bool causal, int planB) {"""),
    ("""    DBuf<int32_t> ds;
    std::vector<int32_t> hsg((size_t)B * T, 0);
    ds.ensure(hsg.size());
    CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
""", ""),
    ("""mb(o.dV.p), B, Hq, 1, T, hd, causal, with_seg ? ds.p : nullptr, &pl.p);""", """mb(o.dV.p), B, Hq, 1, T, hd, causal, &pl.p);"""),
    ("""  if (!kernels_ok) { refuse("this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", 1, 2, 256, 128, true, false, 1); return; }
  const std::string mode = bwd_mode_name();
  if (mode.rfind("Vj2", 0) == 0) {
    // Since 2026-10-15 the default backward is Vj2, but design A (chat plans) still runs on Vj1d, while the "alone"
    // runs that G11 compares bit for bit run on Vj2: the comparison would only show Vj1d-vs-Vj2 rounding. G11 runs
    // with USE_VJ2=0 (Vj1d for both), as the PENDING list at the end says.
    std::printf("  backward mode is %s: G11 needs USE_VJ2=0 (design A runs on Vj1d) -- skipped\\n", mode.c_str());
    sum.skipped.push_back("G11 (design A backward): run it with USE_VJ2=0");
    return;
  }
  if (mode.rfind("Vj1d", 0) != 0) { refuse(mode.c_str(), "chats", 1, 2, 256, 128, true, false, 1); return; }
  refuse("plan AND seg_start", "not both", 1, 2, 256, 128, true, true, 1);
  refuse("is_causal = false", "chats", 1, 2, 256, 128, false, false, 1);
  refuse("head_dim 64", "chats", 1, 2, 256, 64, true, false, 1);
  refuse("T = 288 (not a multiple of 64)", "chats", 1, 2, 288, 128, true, false, 1);
  refuse("plan for B=1 used with B=2", "another shape", 2, 2, 256, 128, true, false, 1);""",
     """  if (!kernels_ok) { refuse("this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", 1, 2, 256, 128, true, 1); return; }
  // Vj2 (default since 2026-10-15) and Vj1d (USE_VJ2=0) both run design A; the dense "alone" runs that G11 compares
  // bit for bit use the same kernel, so the comparison holds in both modes.
  const std::string mode = bwd_mode_name();
  std::printf("  backward mode: %s\\n", mode.c_str());
  if (mode.rfind("Vj2", 0) != 0 && mode.rfind("Vj1d", 0) != 0) { refuse(mode.c_str(), "chats", 1, 2, 256, 128, true, 1); return; }
  refuse("is_causal = false", "chats", 1, 2, 256, 128, false, 1);
  refuse("head_dim 64", "chats", 1, 2, 256, 64, true, 1);
  refuse("T = 288 (not a multiple of 64)", "chats", 1, 2, 288, 128, true, 1);
  refuse("plan for B=1 used with B=2", "another shape", 2, 2, 256, 128, true, 1);"""),
])

# ---- BENCH ---------------------------------------------------------------------------------------------------------
sub_in("// BENCH (Step 6, 2026-10-12)", "// BENCHDENSE (Step 6a, 2026-10-12)", [
    ("""// BENCH (Step 6, 2026-10-12): our three modes timed on fixed layouts, for the cross-library comparison.""",
     """// BENCH (Step 6, 2026-10-12): our modes timed on fixed layouts, for the cross-library comparison."""),
    ("""//   per layout: dense (full-row causal), B (seg_start), A (plan); forward and backward timed separately,""",
     """//   per layout: dense (full-row causal), A (plan; B was removed 2026-10-15); fwd and bwd timed separately,"""),
    ("""  std::printf("\\n===== BENCH: ours (dense / B / A), fwd and bwd, interleaved =====\\n");""",
     """  std::printf("\\n===== BENCH: ours (dense / A), fwd and bwd, interleaved =====\\n");"""),
    ("""    const std::vector<int32_t> hs = build_seg_start(lo);
    DBuf<int32_t> d_seg;
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
""", ""),
    ("""    KOut kd, kb, ka;
    kd.ensure(p); kb.ensure(p); ka.ensure(p);
    auto fwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);
    };
    auto bwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p), mb(o.dK.p),
                                       mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);
    };
    struct Impl { const char* name; KOut* o; const int32_t* sg; const otc::GqaChatPlan* pp; std::vector<float> tf, tb; };
    std::vector<Impl> im = {{"ours_dense_fullrow", &kd, nullptr, nullptr, {}, {}},
                            {"ours_B_segstart", &kb, d_seg.p, nullptr, {}, {}},
                            {"ours_A_plan", &ka, nullptr, &pl.p, {}, {}}};""",
     """    KOut kd, ka;
    kd.ensure(p); ka.ensure(p);
    auto fwd = [&](KOut& o, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, pp);
    };
    auto bwd = [&](KOut& o, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p), mb(o.dK.p),
                                       mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, pp);
    };
    struct Impl { const char* name; KOut* o; const otc::GqaChatPlan* pp; std::vector<float> tf, tb; };
    std::vector<Impl> im = {{"ours_dense_fullrow", &kd, nullptr, {}, {}},
                            {"ours_A_plan", &ka, &pl.p, {}, {}}};"""),
    ("""      for (int w = 0; w < 10; ++w) for (Impl& x : im) { fwd(*x.o, x.sg, x.pp); bwd(*x.o, x.sg, x.pp); }""",
     """      for (int w = 0; w < 10; ++w) for (Impl& x : im) { fwd(*x.o, x.pp); bwd(*x.o, x.pp); }"""),
    ("""        for (Impl& x : im) x.tf.push_back(timed([&] { fwd(*x.o, x.sg, x.pp); }));
        for (Impl& x : im) x.tb.push_back(timed([&] { bwd(*x.o, x.sg, x.pp); }));""",
     """        for (Impl& x : im) x.tf.push_back(timed([&] { fwd(*x.o, x.pp); }));
        for (Impl& x : im) x.tb.push_back(timed([&] { bwd(*x.o, x.pp); }));"""),
])

# ---- G12 / G13: A only ---------------------------------------------------------------------------------------------
rep("""// G12 (Step 6, 2026-10-12): edge cases G8-G11 never covered, run through the SAME full checks
// (g9_layout = design B fwd+bwd, g11_layout = design A fwd+bwd: per-segment accuracy vs the reference,
// 1-token bound, A == alone bitwise, isolation incl. dO, outputs written + finite):""",
    """// G12 (Step 6, 2026-10-12): edge cases G8-G11 never covered, run through the SAME full checks
// (g11_layout = design A fwd+bwd: per-segment accuracy vs the reference, 1-token bound, A == alone bitwise,
// isolation incl. dO, outputs written + finite; design B's g9_layout ran here too until B was removed 2026-10-15):""")
rep("""//   - one chat of 16384 tokens (2/1 heads, to keep the fp64 reference small): B's backward then needs a
//     2nd pass of its __syncthreads_count prefix loop (more than 128 q-tiles after a key tile) -- the
//     only path of that loop no test reached before;""",
    """//   - one chat of 16384 tokens (2/1 heads, to keep the fp64 reference small): the longest segment, 256
//     q-steps for the first key tile (it was added for a loop of design B, kept as A's long-chat case);""")
rep("""// Tags start "G9 [G12]" / "G11 [G12]", so no known-list entry can match them.""",
    """// Tags start "G11 [G12]", so no known-list entry can match them.""")
rep("""  std::printf("\\n===== G12 edge cases: other row lengths, a 16384-token chat, 4096 one-token chats (B and A) =====\\n");""",
    """  std::printf("\\n===== G12 edge cases: other row lengths, a 16384-token chat, 4096 one-token chats (A) =====\\n");""")
rep("""    g9_layout(c.lo, c.h, sum, sd);    // design B
    g11_layout(c.lo, c.h, sum, sd);   // design A (same inputs)""",
    """    g11_layout(c.lo, c.h, sum, sd);   // design A""")
rep("""// G13 (Step 6, 2026-10-12): the last two gaps of the coverage audit, through the same full B and A checks.""",
    """// G13 (Step 6, 2026-10-12): the last two gaps of the coverage audit, through the same full A checks.""")
rep("""    g9_layout(lo, {18, 6}, sum, seed);
    g11_layout(lo, {18, 6}, sum, seed);""", """    g11_layout(lo, {18, 6}, sum, seed);""")
rep("""  g9_layout(m, {18, 6}, sum, 1399);
  g11_layout(m, {18, 6}, sum, 1399);""", """  g11_layout(m, {18, 6}, sum, 1399);""")

# ---- main ------------------------------------------------------------------------------------------------------------
rep("""  } else if (only == "G8") {
    std::printf("\\n===== VARLEN_ONLY=G8 (Step 3: design B forward only) =====\\n");
    if (kernels_ok) g8_packed_forward(have_real ? &Lreal : nullptr, sum);
    else sum.skip_("G8: kernels under test are Hopper-only");
  } else if (only == "G7") {""", """  } else if (only == "G7") {""")
rep("""  } else if (only == "G9") {
    std::printf("\\n===== VARLEN_ONLY=G9 (Step 4: design B backward) =====\\n");
    g9_packed_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
  } else if (!only.empty()) {
    sum.record(false, "unknown VARLEN_ONLY=" + only + " (supported: CLASSIC, D10, G7, G8, G9, G10, G11, G12, G13, BENCH, BENCHDENSE)");""",
    """  } else if (!only.empty()) {
    sum.record(false, "unknown VARLEN_ONLY=" + only + " (supported: CLASSIC, D10, G7, G10, G11, G12, G13, BENCH, BENCHDENSE)");""")
rep("""    for (const char* g : {"G2", "G3", "G4", "G5", "G6", "G7", "G8"}) sum.skip_(std::string(g) + ": " + r);
    g9_packed_backward(nullptr, sum, /*kernels_ok=*/false);   // only the refusals; they must hold here too
    g10_chats_forward(nullptr, sum, /*kernels_ok=*/false);""",
    """    for (const char* g : {"G2", "G3", "G4", "G5", "G6", "G7"}) sum.skip_(std::string(g) + ": " + r);
    g10_chats_forward(nullptr, sum, /*kernels_ok=*/false);   // only the refusals; they must hold here too""")
rep("""    g8_packed_forward(have_real ? &Lreal : nullptr, sum);
    g9_packed_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
    g10_chats_forward(have_real ? &Lreal : nullptr, sum, kernels_ok);""",
    """    g10_chats_forward(have_real ? &Lreal : nullptr, sum, kernels_ok);""")
rep("""  std::printf("  PENDING (Step 5; design B is covered by G8 (forward) and G9 (backward)):\\n"
              "    G9 mode refusals need their own runs: USE_FLASH=1 and USE_DETERMINISTIC_DQ=1 with VARLEN_ONLY=G9\\n"
              "    G11 mode refusals need their own runs: USE_FLASH=1 and USE_DETERMINISTIC_DQ=1 with VARLEN_ONLY=G11\\n"
              "    G11 (design A backward) and packed backward on Vj1d need their own run since 2026-10-15: USE_VJ2=0\\n\"""",
    """  std::printf("  PENDING (packed rows = design A only since 2026-10-15):\\n"
              "    G11 mode refusals need their own runs: USE_FLASH=1 and USE_DETERMINISTIC_DQ=1 with VARLEN_ONLY=G11\\n"
              "    G11 on Vj1d (A's 64-key tiles) needs its own run: USE_VJ2=0 VARLEN_ONLY=G11\\n\"""")

left = [l.strip()[:120] for l in s.splitlines()
        if re.search(r"seg_start|d_seg|g8_|g9_|build_seg_start|fwd_tiles|bwd_tiles\(|run_packed\b|run_packed_fwd|g_diag_case|g_diag_list", l)]
print("\n".join(left))
open(p, "w", encoding="utf-8").write(s)
print("patched test_varlen_attention.cu")
