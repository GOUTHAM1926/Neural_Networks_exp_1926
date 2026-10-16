#!/usr/bin/env python3
"""CPU-only test of the REAL gqa_chat_plan_build source (2026-10-15). Extracts the function (and Vj2's section_pairs)
from the .cu files, compiles them with a fake allocator / fake cudaMemcpyAsync, builds plans for many random packed
layouts and checks the Vj2 lists against the kernels' indexing rules. usage: plan_cpu_test.py <TI dir> <work dir>"""
import sys, os, re, subprocess
TI, W = sys.argv[1], sys.argv[2]
fwd = open(os.path.join(TI, "src/Kernels/cuda/attention/arch/GQA_fwd_generic.cu")).read()
vj2 = open(os.path.join(TI, "src/Kernels/cuda/attention/arch/GQA_bwd_vj2_sm90.cu")).read()

def func(src, head):
    a = src.index(head)
    i = src.index("{", a); d = 0
    for j in range(i, len(src)):
        d += src[j] == "{"; d -= src[j] == "}"
        if d == 0:
            return src[a:j + 1]
    raise SystemExit("unbalanced " + head)

build = func(fwd, "void gqa_chat_plan_build(")
free_ = func(fwd, "void gqa_chat_plan_free(")
sect = func(vj2, "static int section_pairs(int S)")
hdr = open(os.path.join(TI, "include/ops/helpers/AttentionKernels.h")).read()
plan_struct = hdr[hdr.index("struct GqaChatPlan {"):]
plan_struct = plan_struct[:plan_struct.index("};") + 2]

test = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <random>
typedef int cudaError_t; typedef void* cudaStream_t; const int cudaSuccess = 0; const int cudaMemcpyHostToDevice = 1;
static cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, int, cudaStream_t) { std::memcpy(d, s, n); return 0; }
static cudaStream_t getCurrentStream() { return nullptr; }
struct CachingCUDAAllocator {
  static CachingCUDAAllocator& instance() { static CachingCUDAAllocator a; return a; }
  void* allocate(size_t n) { return std::malloc(n ? n : 1); }
  void deallocate(void* p) { std::free(p); }
};
namespace vj2 { constexpr int D = 128; SECT }
int gqa_vj2_section_pairs(int S) { return vj2::section_pairs(S); }
PLAN
FREE
BUILD
struct Sg { int b, t0, L; };
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { if (fails++ < 20) { std::printf("FAIL %s:%d ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)
static void run(const std::vector<std::vector<int>>& rows, int S, int Hq, bool dense_order_check) {
  std::vector<int32_t> b, t0, L; std::vector<Sg> segs;
  for (int r = 0; r < (int)rows.size(); ++r) { int t = 0; for (int l : rows[r]) { b.push_back(r); t0.push_back(t); L.push_back(l); segs.push_back({r, t, l}); t += l; } }
  const int B = (int)rows.size();
  GqaChatPlan p;
  gqa_chat_plan_build(p, b.data(), t0.data(), L.data(), (int)b.size(), B, S, Hq);
  // dq tiles: per head, every token row exactly once, tiles of a segment consecutive, 64-row slots
  const int nd = p.n_dq_tiles;
  std::vector<int> seen((size_t)B * S, 0);
  for (int T = 0; T < nd; ++T) {
    const int32_t* d = p.dq_tiles + 4 * T;
    CHECK(d[2] >= 1 && d[2] <= 64 && d[3] == 0, "dq tile %d nrows %d", T, d[2]);
    for (int r = 0; r < d[2]; ++r) seen[(size_t)d[0] * S + d[1] + r]++;
  }
  for (size_t i = 0; i < seen.size(); ++i) CHECK(seen[i] == 1, "token row %zu covered %d times by dq tiles", i, seen[i]);
  // vj2 blocks: each (segment, head, kt) exactly once; steps map to the segment's slots in order
  std::vector<std::vector<int>> cover(segs.size() * Hq);
  long long prev_sec = -1; int prev_work = 1 << 30;
  const int sw = gqa_vj2_section_pairs(S);
  for (int i = 0; i < p.n_vj2_blocks; ++i) {
    const int32_t* v = p.vj2_blocks + 8 * i;
    const int vb = v[0], vt0 = v[1], vL = v[2], kt = v[3], hq = v[4], dqt0 = v[5];
    CHECK(v[6] == 0 && v[7] == 0, "pad");
    int si = -1;
    for (size_t j = 0; j < segs.size(); ++j) if (segs[j].b == vb && segs[j].t0 == vt0 && segs[j].L == vL) si = (int)j;
    CHECK(si >= 0, "block %d: segment (%d,%d,%d) not in the input", i, vb, vt0, vL);
    if (si < 0) continue;
    CHECK(hq >= 0 && hq < Hq && kt >= 0 && 128 * kt < vL, "block %d: hq %d kt %d L %d", i, hq, kt, vL);
    cover[(size_t)si * Hq + hq].push_back(kt);
    const int nIter = (vL + 63) / 64 - 2 * kt;
    CHECK(nIter >= 1, "block %d nIter %d", i, nIter);
    for (int s = 0; s < nIter; ++s) {   // step s = segment rows 64(2kt+s) .. +63 -> slot dqt0 + s of head hq
      const int tile = dqt0 + s;
      CHECK(tile / nd == hq, "block %d step %d: slot %d is not head %d's", i, s, tile, hq);
      const int32_t* d = p.dq_tiles + 4 * (tile % nd);
      const int qrow0 = vt0 + 64 * (2 * kt + s);
      CHECK(d[0] == vb && d[1] == qrow0 && d[2] == std::min(64, vL - 64 * (2 * kt + s)),
            "block %d step %d: slot {%d,%d,%d} != rows {%d,%d,%d}", i, s, d[0], d[1], d[2], vb, qrow0, std::min(64, vL - 64 * (2 * kt + s)));
    }
    // LPT order inside sections of sw (b, hq) pairs
    const long long sec = ((long long)vb * Hq + hq) / sw;
    const int work = nIter;
    CHECK(sec >= prev_sec, "block %d: section went back (%lld after %lld)", i, sec, prev_sec);
    if (sec == prev_sec) CHECK(work <= prev_work, "block %d: work %d after %d in one section", i, work, prev_work);
    prev_sec = sec; prev_work = work;
    if (dense_order_check) {   // one segment of length S per row: the plan must be the dense grid's order
      const int nkt = S / 128, HB = B * Hq;
      const int secd = i / (sw * nkt), in = i % (sw * nkt);
      const int n = (secd < HB / sw) ? sw : HB % sw;
      const int ktd = in / n, hb = secd * sw + in % n;
      CHECK(ktd == kt && hb / Hq == vb && hb % Hq == hq, "block %d: plan (b%d h%d kt%d) != dense (b%d h%d kt%d)", i, vb, hq, kt, hb / Hq, hb % Hq, ktd);
    }
  }
  for (size_t j = 0; j < segs.size(); ++j)
    for (int hq = 0; hq < Hq; ++hq) {
      std::vector<int> k = cover[j * Hq + hq]; std::sort(k.begin(), k.end());
      const int nk = (segs[j].L + 127) / 128;
      bool ok = (int)k.size() == nk;
      for (int x = 0; ok && x < nk; ++x) ok = k[x] == x;
      CHECK(ok, "segment %zu head %d: %zu key tiles, expected 0..%d once each", j, hq, k.size(), nk - 1);
    }
  // the older lists are still there and consistent
  CHECK(p.n_fwd_units > 0 && p.n_bwd_tiles > 0, "fwd/bwd lists empty");
  CHECK(p.bwd_tiles == p.fwd_units + 4 * p.n_fwd_units && p.vj2_blocks == p.bwd_tiles + 4 * p.n_bwd_tiles &&
        p.dq_tiles == p.vj2_blocks + 8 * p.n_vj2_blocks && p.storage == p.fwd_units, "list layout");
  CHECK(((uintptr_t)p.vj2_blocks % 16) == 0 && ((uintptr_t)p.dq_tiles % 16) == 0, "16-B alignment");
  gqa_chat_plan_free(p);
}
int main() {
  std::mt19937 rng(20261015);
  int cases = 0;
  for (int S : {256, 1024, 4096, 8192}) for (int B : {1, 2, 8}) for (int Hq : {2, 18, 32}) {
    std::vector<std::vector<int>> rows(B, std::vector<int>{S});
    run(rows, S, Hq, true); ++cases;                     // dense-equivalent: must be the dense order
    for (int rep = 0; rep < 6; ++rep) {                   // random chats + pad tails
      for (auto& r : rows) {
        r.clear(); int t = 0;
        std::uniform_int_distribution<int> U(1, rep < 3 ? 300 : S);
        while (t < S) { int l = std::min(U(rng), S - t); r.push_back(l); t += l; }
      }
      run(rows, S, Hq, false); ++cases;
    }
  }
  { std::vector<std::vector<int>> rows(1, std::vector<int>(4096, 1)); run(rows, 4096, 18, false); ++cases; }   // D6
  { std::vector<std::vector<int>> rows = {{1, 4095}, {4095, 1}, {63, 65, 3968}, {64, 64, 128, 3840}}; run(rows, 4096, 18, false); ++cases; }
  std::printf("%d layouts, %d failures -> %s\n", cases, fails, fails ? "FAIL" : "PASS");
  return fails != 0;
}
'''.replace("SECT", sect).replace("PLAN", plan_struct).replace("FREE", free_).replace("BUILD", build)
os.makedirs(W, exist_ok=True)
src = os.path.join(W, "plan_cpu_test.cpp")
open(src, "w").write(test)
r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-o", os.path.join(W, "plan_cpu_test"), src], capture_output=True, text=True)
print(r.stdout, r.stderr[-3000:])
if r.returncode: raise SystemExit("compile failed")
r = subprocess.run([os.path.join(W, "plan_cpu_test")], capture_output=True, text=True)
print(r.stdout[-4000:], r.stderr[-2000:])
raise SystemExit(r.returncode)
