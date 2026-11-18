// test_sft_dataloader.cpp -- checks sft/SftDataLoader.h against the files it reads.
//
//   test_sft_dataloader <train_dir> <val_dir>
//
// <train_dir> is only read (checks 1-4). Check 5 corrupts COPIES of <val_dir> in a temp directory
// (TMPDIR, else /tmp) and deletes them at the end; the originals are never written.
// Exit code 0 = every check passed.
//
//   1. data    every batch, copied back from the GPU, is byte-equal to its rows of inputs.bin / labels.bin;
//              every trained label is the next input token (the shift by one), and each row's last trained
//              label is the end token. Control: the pretraining target (the input shifted by one) is NOT
//              what the loader returns, so the comparison can tell the two apart.
//   2. counts  last_valid_tokens() = the non-65535 targets of the batch just returned; valid_tokens_ahead(n)
//              = what the next n batches really contain, also across the end of the epoch;
//              total_valid_tokens() = meta.json "trained_tokens".
//   3. ranks   W = 2: over one epoch no row goes to both ranks, and together they cover every row except
//              the N % (B*W) left over; the two ranks' look-ahead counts add up to the same global count
//              that one GPU sees for the same 64 rows (what the trainer will all-reduce).
//   4. epochs  after micro_batches_per_epoch() batches, epoch() = 1 and the rows start again at row 0;
//              reset() and skip_micro_batches() land on the right rows.
//   5. errors  a wrong T, a bad rank, too few rows, a missing file, a short file, an input id >= vocab,
//              a label >= vocab, a wrong count, a row with no trained token, and (2026-10-31) data whose meta.json
//              vocab_rows is not the model's each throw, with a message naming the problem (and the row).
//              Control: the unchanged copy opens.
// The end token and the model's vocab rows come from the data's own meta.json (end_id, vocab_rows; 2026-10-31),
// so the test runs on GPT-2 data (50256 / 50,304) and on the 135M base's data (0 / 49,280) alike.
//
// Build from the BluTrain root (the binary goes outside the repo):
//   g++ -std=c++2a -O2 -DWITH_CUDA -fopenmp -Isft -ITensor-Implementations/include -IProfiler/include \
//       -I/usr/local/cuda/include sft/test_sft_dataloader.cpp -o /tmp/test_sft_dataloader \
//       -LTensor-Implementations/lib -LProfiler/lib -L/usr/local/cuda/lib64 \
//       -Wl,-rpath,$PWD/Tensor-Implementations/lib -Wl,-rpath,$PWD/Profiler/lib -ltensor -lprofiler -lcudart -lcuda
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <cuda_runtime.h>
#include "SftDataLoader.h"

namespace fs = std::filesystem;

namespace {

constexpr uint16_t kIgnore = SftDataLoader::kIgnore;
uint16_t kEnd = 50256;     // the target after each answer: meta.json end_id when present, else GPT-2's <|endoftext|>
int64_t kVocab = 50304;    // the model's rows the loader checks against: meta.json vocab_rows when present (2026-10-31:
                           // the loader has no default any more), else the GPT-2 models' 50,304
constexpr int B = 8;                  // the trainer's micro-batch
constexpr int T = 4096;
constexpr int kDev = 0;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++g_fail; std::cout << "    FAIL: " << what << "\n"; }
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <typename V>
std::vector<V> read_all(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::vector<V> v(fs::file_size(p) / sizeof(V));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(V)));
    if (!f) throw std::runtime_error("short read " + p.string());
    return v;
}

int64_t meta_int(const fs::path& dir, const std::string& key) {
    std::ifstream f(dir / "meta.json");
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t at = s.find("\"" + key + "\":");
    if (at == std::string::npos) throw std::runtime_error("meta.json has no " + key);
    return std::stoll(s.substr(at + key.size() + 3));
}
bool meta_has(const fs::path& dir, const std::string& key) {
    std::ifstream f(dir / "meta.json");
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s.find("\"" + key + "\":") != std::string::npos;
}

// The files, read with plain ifstream: an independent second view of what the loader mmaps.
struct Files {
    int64_t N = 0;
    std::vector<uint16_t> in, lab;
    std::vector<uint32_t> cnt;
    const uint16_t* in_row(int64_t r) const { return in.data() + r * T; }
    const uint16_t* lab_row(int64_t r) const { return lab.data() + r * T; }
    int64_t count_rows(int64_t r0, int64_t n) const {
        int64_t s = 0;
        for (int64_t r = r0; r < r0 + n; ++r) s += cnt[r];
        return s;
    }
};
Files load(const fs::path& dir) {
    Files f;
    f.in = read_all<uint16_t>(dir / "inputs.bin");
    f.lab = read_all<uint16_t>(dir / "labels.bin");
    f.cnt = read_all<uint32_t>(dir / "valid_counts.bin");
    f.N = static_cast<int64_t>(f.cnt.size());
    if (f.in.size() != static_cast<size_t>(f.N) * T || f.lab.size() != f.in.size())
        throw std::runtime_error(dir.string() + ": file sizes do not match " + std::to_string(f.N) + " rows");
    return f;
}

struct Host { std::vector<uint16_t> in, tgt; };
Host to_host(Batch& b) {
    const size_t n = static_cast<size_t>(b.B) * static_cast<size_t>(b.T);
    Host h{std::vector<uint16_t>(n), std::vector<uint16_t>(n)};
    cudaStream_t s = OwnTensor::cuda::getCurrentStream();   // the stream the loader copied on
    ck(cudaMemcpyAsync(h.in.data(), b.input.data(), n * sizeof(uint16_t), cudaMemcpyDeviceToHost, s), "input back");
    ck(cudaMemcpyAsync(h.tgt.data(), b.target.data(), n * sizeof(uint16_t), cudaMemcpyDeviceToHost, s), "target back");
    ck(cudaStreamSynchronize(s), "sync");
    return h;
}

// Everything checked on one batch that should hold rows [row0, row0 + B) of the files.
struct BatchCheck {
    bool bytes_ok = false, form_ok = false;
    int64_t valid = 0;          // non-65535 targets in the batch, counted from the GPU copy
    int64_t shift_bad = 0;      // trained targets that are not the next input token
    int64_t end_bad = 0;        // rows whose last trained target is not the end token
    int64_t differs = 0;        // positions where the pretraining target (next input) != what we got
};
BatchCheck check_batch(Batch& b, const Files& f, int64_t row0) {
    BatchCheck c;
    const auto d = b.input.shape().dims;
    const auto e = b.target.shape().dims;
    c.form_ok = b.B == B && b.T == T && d.size() == 2 && d[0] == B && d[1] == T && e == d &&
                b.input.dtype() == OwnTensor::Dtype::UInt16 && b.target.dtype() == OwnTensor::Dtype::UInt16 &&
                !b.input.device().is_cpu() && b.input.device().index == kDev && !b.target.device().is_cpu();
    const Host h = to_host(b);
    const size_t bytes = static_cast<size_t>(B) * T * sizeof(uint16_t);
    c.bytes_ok = std::memcmp(h.in.data(), f.in_row(row0), bytes) == 0 &&
                 std::memcmp(h.tgt.data(), f.lab_row(row0), bytes) == 0;
    for (int i = 0; i < B; ++i) {
        const uint16_t* x = h.in.data() + static_cast<size_t>(i) * T;
        const uint16_t* y = h.tgt.data() + static_cast<size_t>(i) * T;
        int64_t last = -1;
        for (int64_t t = 0; t < T; ++t) {
            if (t + 1 < T && y[t] != x[t + 1]) ++c.differs;
            if (y[t] == kIgnore) continue;
            ++c.valid;
            last = t;
            if (t + 1 < T && y[t] != x[t + 1]) ++c.shift_bad;
        }
        if (last < 0 || y[last] != kEnd) ++c.end_bad;
    }
    return c;
}

void patch(const fs::path& p, size_t byte_off, const void* src, size_t n) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    if (!f) throw std::runtime_error("cannot patch " + p.string());
    f.seekp(static_cast<std::streamoff>(byte_off));
    f.write(static_cast<const char*>(src), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("patch failed " + p.string());
}

// Expects the constructor to throw std::runtime_error whose message contains every piece of `want`.
void expect_throw(const std::string& name, const std::function<void()>& make, const std::vector<std::string>& want) {
    try {
        make();
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        bool all = true;
        for (const auto& w : want) all = all && msg.find(w) != std::string::npos;
        std::cout << "    " << (all ? "ok  " : "BAD ") << name << " -> \"" << msg << "\"\n";
        check(all, name + ": the message does not name the problem");
        return;
    }
    std::cout << "    BAD " << name << " -> no error\n";
    check(false, name + ": opened without an error");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <train_dir> <val_dir>\n";
        return 2;
    }
    const fs::path train = argv[1], val = argv[2];
    ck(cudaSetDevice(kDev), "cudaSetDevice");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, kDev), "cudaGetDeviceProperties");
    const Files F = load(train);
    if (meta_has(train, "end_id")) kEnd = static_cast<uint16_t>(meta_int(train, "end_id"));
    if (meta_has(train, "vocab_rows")) kVocab = meta_int(train, "vocab_rows");
    std::cout << "train " << train << ": " << F.N << " rows x T=" << T << " | B=" << B << " | GPU " << kDev
              << " " << prop.name << " | end token " << kEnd << " | model rows " << kVocab << "\n";

    std::vector<std::pair<std::string, bool>> summary;
    auto run = [&](const std::string& name, const std::function<void()>& body) {
        std::cout << "\n== " << name << "\n";
        const int before = g_fail;
        try {
            body();
        } catch (const std::exception& e) {
            check(false, std::string("unexpected exception: ") + e.what());
        }
        summary.emplace_back(name, g_fail == before);
        std::cout << "  -> " << (g_fail == before ? "PASS" : "FAIL") << "\n";
    };

    run("1. data: GPU batches = file rows; target = next input; answers end with the end token", [&] {
        SftDataLoader L(train.string(), B, T, 0, 1, kDev, true, kVocab);
        const int64_t M = L.micro_batches_per_epoch();
        check(L.rows() == F.N && M == F.N / B, "rows / micro-batches per epoch");
        int64_t bad_bytes = 0, bad_form = 0, bad_where = 0, shift_bad = 0, end_bad = 0, differs = 0, ignored = 0;
        for (int64_t g = 0; g < M; ++g) {
            Batch b = L.next_batch();
            const int64_t row0 = g * B;
            if (b.token_offset != static_cast<size_t>(row0) * T || b.shard_index != 0) ++bad_where;
            const BatchCheck c = check_batch(b, F, row0);
            bad_bytes += !c.bytes_ok;
            bad_form += !c.form_ok;
            shift_bad += c.shift_bad;
            end_bad += c.end_bad;
            differs += c.differs;
            ignored += static_cast<int64_t>(B) * T - c.valid;
        }
        std::cout << "    " << M << " batches (" << M * B << " rows): byte mismatches " << bad_bytes
                  << ", wrong shape/dtype/device " << bad_form << ", wrong token_offset/shard " << bad_where << "\n"
                  << "    trained targets that are not the next input: " << shift_bad
                  << " | rows not ending with the end token: " << end_bad << "\n"
                  << "    control: the pretraining target (next input) differs from labels.bin at " << differs
                  << " positions (ignored positions: " << ignored << ")\n";
        check(bad_bytes == 0, "some batches are not byte-equal to the files");
        check(bad_form == 0, "some batches have the wrong shape, dtype or device");
        check(bad_where == 0, "some batches report the wrong token_offset / shard_index");
        check(shift_bad == 0, "some trained targets are not the next input token");
        check(end_bad == 0, "some rows do not end with the end token");
        check(differs > 0, "control: the targets look like the shifted input, the check has no teeth");
    });

    run("2. counts: per batch, look-ahead (also across the epoch end), total", [&] {
        SftDataLoader L(train.string(), B, T, 0, 1, kDev, false, kVocab);
        const int64_t M = L.micro_batches_per_epoch();
        constexpr int GA = 8;                                  // 64 rows per step on 1 GPU
        const int64_t file_total = F.count_rows(0, F.N);
        const int64_t meta_total = meta_int(train, "trained_tokens");
        std::cout << "    total_valid_tokens() " << L.total_valid_tokens() << " | valid_counts.bin " << file_total
                  << " | meta.json " << meta_total << "\n";
        check(L.total_valid_tokens() == file_total && file_total == meta_total, "totals disagree");
        std::vector<int64_t> ahead, actual;
        int64_t bad_last = 0;
        for (int64_t g = 0; g < M + GA; ++g) {                 // runs GA batches into epoch 1
            ahead.push_back(L.valid_tokens_ahead(GA));
            Batch b = L.next_batch();
            const Host h = to_host(b);
            const int64_t n = std::count_if(h.tgt.begin(), h.tgt.end(), [](uint16_t y) { return y != kIgnore; });
            actual.push_back(n);
            bad_last += L.last_valid_tokens() != n;
        }
        int64_t bad_ahead = 0;
        for (int64_t g = 0; g < M; ++g) {
            int64_t s = 0;
            for (int k = 0; k < GA; ++k) s += actual[g + k];
            bad_ahead += ahead[g] != s;
        }
        const auto [mn, mx] = std::minmax_element(actual.begin(), actual.begin() + M);
        int64_t sum = 0;
        for (int64_t g = 0; g < M; ++g) sum += actual[g];
        std::cout << "    per batch (from the GPU copy): min " << *mn << " max " << *mx << " mean " << sum / M
                  << " | last_valid_tokens() wrong in " << bad_last << " of " << M + GA << " batches\n"
                  << "    valid_tokens_ahead(" << GA << ") wrong in " << bad_ahead << " of " << M
                  << " positions (the last " << GA - 1 << " look across the epoch end)\n";
        check(sum == file_total, "one epoch of batches does not hold every trained token");
        check(bad_last == 0, "last_valid_tokens() disagrees with the batch");
        check(bad_ahead == 0, "valid_tokens_ahead() disagrees with the batches that follow");
    });

    run("3. ranks: W = 2 split, no overlap, global count", [&] {
        SftDataLoader L0(train.string(), B, T, 0, 2, kDev, false, kVocab);
        SftDataLoader L1(train.string(), B, T, 1, 2, kDev, false, kVocab);
        SftDataLoader one(train.string(), B, T, 0, 1, kDev, false, kVocab);
        const int64_t M = L0.micro_batches_per_epoch();
        check(M == F.N / (2 * B) && L1.micro_batches_per_epoch() == M, "micro-batches per epoch at W = 2");
        // The step's global count as the trainer will build it: each rank's look-ahead, all-reduced (SUM).
        const int64_t two = L0.valid_tokens_ahead(4) + L1.valid_tokens_ahead(4);    // 64 rows on 2 GPUs
        const int64_t uno = one.valid_tokens_ahead(8);                               // 64 rows on 1 GPU
        const int64_t file = F.count_rows(0, 64);
        std::cout << "    global count for step 0 (rows 0-63): 2 GPUs " << two << " | 1 GPU " << uno
                  << " | valid_counts.bin " << file << "\n";
        check(two == file && uno == file, "the global count is wrong");
        std::vector<int> owner(F.N, -1);
        int64_t dup = 0, bad_rows = 0, bad_bytes = 0;
        for (int64_t g = 0; g < M; ++g) {
            for (int r = 0; r < 2; ++r) {
                Batch b = (r == 0 ? L0 : L1).next_batch();
                const int64_t row0 = (g * 2 + r) * B;
                bad_rows += b.token_offset != static_cast<size_t>(row0) * T;
                bad_bytes += !check_batch(b, F, row0).bytes_ok;
                for (int64_t i = row0; i < row0 + B; ++i) {
                    dup += owner[i] != -1;
                    owner[i] = r;
                }
            }
        }
        const int64_t n0 = std::count(owner.begin(), owner.end(), 0);
        const int64_t n1 = std::count(owner.begin(), owner.end(), 1);
        const int64_t none = std::count(owner.begin(), owner.end(), -1);
        const bool tail = std::all_of(owner.begin() + M * 2 * B, owner.end(), [](int o) { return o == -1; });
        std::cout << "    one epoch: rank 0 " << n0 << " rows, rank 1 " << n1 << " rows, on both " << dup
                  << ", unused " << none << " (expected " << F.N - M * 2 * B << ", the last rows) | wrong rows "
                  << bad_rows << ", byte mismatches " << bad_bytes << "\n";
        check(dup == 0, "a row went to both ranks");
        check(n0 == M * B && n1 == M * B, "the ranks do not get M*B rows each");
        check(none == F.N - M * 2 * B && tail, "the unused rows are not exactly the N % (B*W) at the end");
        check(bad_rows == 0 && bad_bytes == 0, "a rank got the wrong rows");
    });

    run("4. epochs: wrap-around, reset, skip", [&] {
        SftDataLoader L(train.string(), B, T, 0, 1, kDev, false, kVocab);
        const int64_t M = L.micro_batches_per_epoch();
        for (int64_t g = 0; g < M; ++g) L.next_batch();
        std::cout << "    after " << M << " batches: epoch() = " << L.epoch() << "\n";
        check(L.epoch() == 1, "epoch() is not 1 after one epoch");
        Batch b = L.next_batch();
        std::cout << "    next batch: token_offset " << b.token_offset << ", shard_index " << b.shard_index << "\n";
        check(b.token_offset == 0 && b.shard_index == 1 && check_batch(b, F, 0).bytes_ok,
              "the batch after the epoch is not rows 0..7 of epoch 1");
        L.reset();
        b = L.next_batch();
        check(L.epoch() == 0 && b.token_offset == 0 && b.shard_index == 0 && check_batch(b, F, 0).bytes_ok,
              "reset() does not restart at row 0");
        L.skip_micro_batches(5);                               // batch 1 is next, skip 1..5 -> batch 6
        b = L.next_batch();
        check(b.token_offset == static_cast<size_t>(6) * B * T && check_batch(b, F, 6 * B).bytes_ok,
              "skip_micro_batches(5) does not land on batch 6");
        L.reset();
        L.skip_micro_batches(M + 3);                           // across the epoch end
        b = L.next_batch();
        check(b.token_offset == static_cast<size_t>(3) * B * T && b.shard_index == 1 && check_batch(b, F, 3 * B).bytes_ok,
              "skip_micro_batches(M + 3) does not land on batch 3 of epoch 1");
        std::cout << "    reset() -> row 0; skip 5 -> batch 6; skip M + 3 -> batch 3 of epoch 1\n";
    });

    run("5. bad files fail loudly", [&] {
        const Files V = load(val);
        const fs::path root = fs::temp_directory_path() / ("sft_loader_test_" + std::to_string(::getpid()));
        struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); } } cleanup{root};
        auto fresh = [&](const std::string& name) {
            const fs::path d = root / name;
            fs::create_directories(d);
            for (const char* f : {"inputs.bin", "labels.bin", "valid_counts.bin", "meta.json"})
                fs::copy_file(val / f, d / f);
            return d;
        };
        auto open = [&](const fs::path& d, int t, int rank, int world) {
            SftDataLoader L(d.string(), B, t, rank, world, kDev, false, kVocab);
        };
        {   // control: an unchanged copy opens
            SftDataLoader ok(fresh("ok").string(), B, T, 0, 1, kDev, false, kVocab);
            std::cout << "    ok   unchanged copy opens: " << ok.rows() << " rows\n";
            check(ok.rows() == V.N, "the unchanged copy does not open with every row");
        }
        int64_t p7 = 0;                                        // first trained position of row 7
        while (V.lab[7 * T + p7] == kIgnore) ++p7;
        expect_throw("wrong T", [&] { open(val, 2048, 0, 1); }, {"was made with T=4096", "uses T=2048"});
        expect_throw("rank >= world", [&] { open(val, T, 2, 2); }, {"bad B / T / rank / world"});
        expect_throw("too few rows", [&] { open(val, T, 0, 64); }, {"less than one micro-batch"});
        {
            const fs::path d = fresh("missing");
            fs::remove(d / "valid_counts.bin");
            expect_throw("missing file", [&] { open(d, T, 0, 1); }, {"cannot open", "valid_counts.bin"});
        }
        {
            const fs::path d = fresh("short");
            fs::resize_file(d / "labels.bin", fs::file_size(d / "labels.bin") - 2);
            expect_throw("short file", [&] { open(d, T, 0, 1); }, {"labels.bin has", "meta.json says"});
        }
        {
            const fs::path d = fresh("bad_input");
            const uint16_t v = static_cast<uint16_t>(kVocab);    // the first id past the model's rows (2026-10-31)
            patch(d / "inputs.bin", static_cast<size_t>(5) * T * 2, &v, 2);
            expect_throw("input id >= vocab", [&] { open(d, T, 0, 1); }, {"row 5:", "input id " + std::to_string(kVocab)});
        }
        {
            const fs::path d = fresh("bad_label");
            const uint16_t v = 60000;
            patch(d / "labels.bin", static_cast<size_t>(7 * T + p7) * 2, &v, 2);
            expect_throw("label >= vocab", [&] { open(d, T, 0, 1); }, {"row 7:", "label 60000"});
        }
        {
            const fs::path d = fresh("bad_count");
            const uint32_t v = V.cnt[3] + 1;
            patch(d / "valid_counts.bin", 3 * 4, &v, 4);
            expect_throw("wrong count", [&] { open(d, T, 0, 1); }, {"row 3:", "valid_counts.bin says"});
        }
        if (meta_has(val, "vocab_rows")) {   // 2026-10-31: data prepared for another model's vocabulary
            expect_throw("vocab_rows of another model", [&] { SftDataLoader L(val.string(), B, T, 0, 1, kDev, false, kVocab + 128); },
                         {"vocab_rows " + std::to_string(kVocab), "another model"});
        }
        {
            const fs::path d = fresh("no_trained");
            const std::vector<uint16_t> all_ignored(T, kIgnore);
            const uint32_t zero = 0;
            patch(d / "labels.bin", static_cast<size_t>(9) * T * 2, all_ignored.data(), T * 2);
            patch(d / "valid_counts.bin", 9 * 4, &zero, 4);
            expect_throw("no trained token", [&] { open(d, T, 0, 1); }, {"row 9:", "no trained token"});
        }
    });

    std::cout << "\n==================== SUMMARY ====================\n";
    for (const auto& [name, ok] : summary) std::cout << (ok ? "PASS  " : "FAIL  ") << name << "\n";
    const auto passed = std::count_if(summary.begin(), summary.end(), [](const auto& s) { return s.second; });
    std::cout << passed << " of " << summary.size() << " checks passed, " << g_fail << " failures\n";
    return g_fail == 0 ? 0 : 1;
}
