#!/usr/bin/env python3
"""CPU-only test of the REAL new SftDataLoader code (2026-10-15): pastes its packed-row check block (check_files) and
its per-batch segment/positions builder (next_batch) into a harness, runs them over every row / micro-batch of the
packed data, and compares with the OLD per-token seg_start.bin of the same rows: positions must be t - seg_start[t]
and the segments must be the runs of seg_start. usage: loader_cpu_test.py <SftDataLoader.h> <data dir> <old dir> <work>"""
import sys, os, subprocess, json
H, DATA, OLD, W = sys.argv[1:5]
h = open(H).read()

def between(src, start, end):
    a = src.index(start); b = src.index(end, a)
    return src[a:b]

# check_files' packed block: from "if (packed_) {   // the chats fit the row" to the valid_counts check
chk = between(h, "            if (packed_) {   // the chats fit the row", "            if (valid != static_cast<int64_t>(cnt[r]))")
# next_batch's packed block: from "if (packed_) {\n            // The B rows' segments" up to the tensor upload
nb = between(h, "        if (packed_) {\n            // The B rows' segments", "            OwnTensor::TensorOptions o32")

src = r'''
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
static std::vector<char> slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
struct Map { void* p; };
static const uint16_t kIgnore = 65535, kEnd = 50256;
int main(int argc, char** argv) {
  const std::string d = argv[1], old = argv[2];
  const int64_t N_ = std::stoll(argv[3]); const int T_ = std::stoi(argv[4]), B_ = 8;
  std::vector<char> offs_b = slurp(d + "/seq_offsets.bin"), lens_b = slurp(d + "/seq_lengths.bin"), lab_b = slurp(d + "/labels.bin"),
                    seg_b = slurp(old + "/seg_start.bin");
  Map offs_{offs_b.data()}, lens_{lens_b.data()};
  const uint16_t* lab = reinterpret_cast<const uint16_t*>(lab_b.data());
  const uint16_t* oldseg = reinterpret_cast<const uint16_t*>(seg_b.data());
  bool packed_ = true;
  long bads = 0;
  auto bad = [&](int64_t r, const std::string& w) { if (bads++ < 10) std::printf("bad row %lld: %s\n", (long long)r, w.c_str()); };
  // ---- check_files' block, every row
  for (int64_t r = 0; r < N_; ++r) {
CHK
  }
  // ---- next_batch's block, every micro-batch of B_ rows (the tail rows that do not fill one are skipped, as the loader)
  std::vector<int32_t> pos32_, seg_b_, seg_t0_, seg_len_;
  long mism = 0, segs = 0, seg_mism = 0;
  for (int64_t row0 = 0; row0 + B_ <= N_; row0 += B_) {
NB
        }
    for (int i = 0; i < B_; ++i)
      for (int t = 0; t < T_; ++t)
        if (pos32_[(size_t)i * T_ + t] != t - (int)oldseg[(row0 + i) * T_ + t]) ++mism;
    for (size_t k = 0; k < seg_b_.size(); ++k) {   // each segment = a run of the old seg_start
      ++segs;
      const int i = seg_b_[k], t0 = seg_t0_[k], L = seg_len_[k];
      const uint16_t* os = oldseg + (row0 + i) * T_;
      bool ok = os[t0] == t0 && (t0 + L == T_ || os[t0 + L] == t0 + L);
      for (int t = t0; ok && t < t0 + L; ++t) ok = os[t] == t0;
      if (!ok) ++seg_mism;
    }
  }
  std::printf("%s: rows %lld, check failures %ld | positions != t - old seg_start: %ld | segments %ld, not an old run: %ld -> %s\n",
              d.c_str(), (long long)N_, bads, mism, segs, seg_mism, (bads || mism || seg_mism) ? "FAIL" : "PASS");
  return (bads || mism || seg_mism) ? 1 : 0;
}
'''.replace("CHK", chk.replace("bad(r,", "bad(r,")).replace("NB", nb.replace("        if (packed_) {", "        {", 1))
os.makedirs(W, exist_ok=True)
cpp = os.path.join(W, "loader_cpu_test.cpp")
open(cpp, "w").write(src)
r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-o", os.path.join(W, "loader_cpu_test"), cpp], capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-3000:]); raise SystemExit("compile failed")
rc = 0
for sp in ("train", "val"):
    m = json.load(open(os.path.join(DATA, sp, "meta.json")))
    r = subprocess.run([os.path.join(W, "loader_cpu_test"), os.path.join(DATA, sp), os.path.join(OLD, sp), str(m["N"]), str(m["T"])],
                       capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-500:])
    rc |= r.returncode
raise SystemExit(rc)
