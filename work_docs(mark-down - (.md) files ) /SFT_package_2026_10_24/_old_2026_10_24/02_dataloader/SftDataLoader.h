#pragma once
// SftDataLoader -- reads the SFT files written by sft/prepare_sft_data.py (SFT report 3.9).
//
//   <dir>/inputs.bin        uint16 [N, T]  what the model reads (padded with 50256)
//   <dir>/labels.bin        uint16 [N, T]  the target at the SAME position (the shift by one was done when
//                                          the file was made); 65535 = ignore (prompt, padding)
//   <dir>/valid_counts.bin  uint32 [N]     number of non-65535 labels in each row
//   <dir>/meta.json         only "N" and "T" are read
//   <dir>/seq_lengths.bin   int32 [sum]    OPTIONAL, v2 packed rows (prepare_sft_data.py --pack ffd, 2026-10-15):
//   <dir>/seq_offsets.bin   int64 [N + 1]  the lengths of the chats of every row (TRL's seq_lengths column); row r's
//                                          chats are seq_lengths[seq_offsets[r] : seq_offsets[r+1]] in row order, and
//                                          the rest of the row is its pad tail. Present = packed rows, absent = v1
//                                          (one chat per row). For each micro-batch the loader uploads the RoPE
//                                          positions (index of each token inside its chat, int32 [B, T]) and keeps
//                                          the B rows' segments -- every chat, then each pad tail as a segment of its
//                                          own -- as host lists, the input of the attention plan (gqa_chat_plan_make).
//                                          The removed per-token format (seg_start.bin, the morning of 2026-10-15) is
//                                          refused, so old data cannot train with the chats of a row seeing each other.
//
// The differences from DataLoaderLite (pretraining), and the only ones:
//   * the target comes from labels.bin, not from the input shifted by one (that is the SFT masking);
//   * whole rows (one example per row) instead of a sliding window over one token stream;
//   * it reports the valid-token count of each micro-batch, for the global loss denominator.
// Rows are used in file order (prepare_sft_data.py already wrote them in random order). Micro-batch m of
// rank r is rows [(m*W + r)*B, +B), so ranks never share a row. After the last full micro-batch it wraps
// to row 0 and epoch() goes up by one; the N % (B*W) rows at the end are not used.
// The files are checked once when opened, and anything wrong throws: a bad file would train silently.
// Returns the same Batch struct as DataLoaderLite (UInt16 [B,T] on the GPU), so the trainer's forward and
// loss code does not change. The copy goes on the current stream, so it is ordered after the previous
// micro-step; the source is pageable (mmap), so cudaMemcpyAsync has read it before it returns.
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "DataLoader.h"          // Batch, the struct the trainer already uses
#include "device/DeviceCore.h"   // OwnTensor::cuda::getCurrentStream()

class SftDataLoader {
public:
    static constexpr uint16_t kIgnore = 65535;
    static constexpr uint16_t kEnd = 50256;   // <|endoftext|>: every chat's last target (prepare_sft_data.py PAD_ID)

    SftDataLoader(const std::string& dir, int B, int T, int rank, int world, int device_idx, bool master,
                  int64_t vocab_rows = 50304)
        : dir_(dir), B_(B), T_(T), rank_(rank), world_(world), device_idx_(device_idx) {
        if (B_ <= 0 || T_ <= 0 || world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("SftDataLoader: bad B / T / rank / world");
        const std::string meta = read_text(dir_ + "/meta.json");
        N_ = json_int(meta, "N");
        const int64_t fileT = json_int(meta, "T");
        if (N_ <= 0) throw std::runtime_error("SftDataLoader: " + dir_ + " has no rows");
        if (fileT != T_)
            throw std::runtime_error("SftDataLoader: " + dir_ + " was made with T=" + std::to_string(fileT) +
                                     " but the trainer uses T=" + std::to_string(T_));
        const size_t cells = static_cast<size_t>(N_) * static_cast<size_t>(T_);
        in_  = map_file(dir_ + "/inputs.bin", cells * sizeof(uint16_t));
        lab_ = map_file(dir_ + "/labels.bin", cells * sizeof(uint16_t));
        cnt_ = map_file(dir_ + "/valid_counts.bin", static_cast<size_t>(N_) * sizeof(uint32_t));
        {   // v2 packed rows: seq_lengths.bin + seq_offsets.bin (checked against the rows in check_files)
            auto exists = [&](const char* f) { struct stat st {}; return ::stat((dir_ + "/" + f).c_str(), &st) == 0; };
            const bool lens = exists("seq_lengths.bin"), offs = exists("seq_offsets.bin");
            if (exists("seg_start.bin") && !lens)
                throw std::runtime_error("SftDataLoader: " + dir_ + " holds packed rows in the removed per-token format "
                                         "(seg_start.bin); regenerate it with prepare_sft_data.py --pack ffd "
                                         "(writes seq_lengths.bin + seq_offsets.bin)");
            if (lens != offs)
                throw std::runtime_error("SftDataLoader: " + dir_ + " has only one of seq_lengths.bin / seq_offsets.bin");
            packed_ = lens;
        }
        if (packed_) {
            offs_ = map_file(dir_ + "/seq_offsets.bin", static_cast<size_t>(N_ + 1) * sizeof(int64_t));
            const int64_t n_chats = static_cast<const int64_t*>(offs_.p)[N_];
            if (n_chats <= 0) throw std::runtime_error("SftDataLoader: " + dir_ + "/seq_offsets.bin lists no chat");
            lens_ = map_file(dir_ + "/seq_lengths.bin", static_cast<size_t>(n_chats) * sizeof(int32_t));
        }
        check_files(vocab_rows);
        M_ = N_ / (static_cast<int64_t>(B_) * world_);
        if (M_ == 0)
            throw std::runtime_error("SftDataLoader: " + std::to_string(N_) + " rows is less than one micro-batch "
                                     "on every rank (B*W = " + std::to_string(static_cast<int64_t>(B_) * world_) + ")");
        if (master)
            std::cout << "[SftLoader] " << dir_ << ": " << N_ << " rows x T=" << T_ << ", " << total_valid_
                      << " trained tokens | B=" << B_ << " W=" << world_ << " -> " << M_
                      << " micro-batches per rank per epoch, " << (N_ - M_ * B_ * world_)
                      << " rows unused per epoch"
                      << (packed_ ? " | PACKED rows (seq_lengths.bin: " + std::to_string(static_cast<const int64_t*>(
                                        offs_.p)[N_]) + " chats, each attends only to itself)" : std::string())
                      << std::endl;
    }
    ~SftDataLoader() { unmap(in_); unmap(lab_); unmap(cnt_); unmap(offs_); unmap(lens_); }
    SftDataLoader(const SftDataLoader&) = delete;
    SftDataLoader& operator=(const SftDataLoader&) = delete;

    Batch next_batch() {
        const int64_t row0 = first_row(next_);                 // this rank's B rows are consecutive in the file
        const size_t off = static_cast<size_t>(row0) * static_cast<size_t>(T_);
        const size_t bytes = static_cast<size_t>(B_) * static_cast<size_t>(T_) * sizeof(uint16_t);
        OwnTensor::TensorOptions opts = OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16)
                                            .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, device_idx_));
        Batch b;
        b.B = B_; b.T = T_;
        b.input  = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, opts);
        b.target = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, opts);
        cudaStream_t s = OwnTensor::cuda::getCurrentStream();
        check_cuda(cudaMemcpyAsync(b.input.data(),  static_cast<const uint16_t*>(in_.p)  + off, bytes,
                                   cudaMemcpyHostToDevice, s), "inputs");
        check_cuda(cudaMemcpyAsync(b.target.data(), static_cast<const uint16_t*>(lab_.p) + off, bytes,
                                   cudaMemcpyHostToDevice, s), "labels");
        if (packed_) {
            // The B rows' segments (row in the batch, start, length: every chat in row order, then the row's pad tail
            // if any) and the RoPE positions, which restart at 0 for every segment. The positions are staged through a
            // host vector: a copy from pageable memory has read its source when it returns, so the vector can be
            // refilled by the next call. A new tensor every call: the trainer keeps it until the micro-step's
            // backward has run.
            const int64_t* offs = static_cast<const int64_t*>(offs_.p);
            const int32_t* lens = static_cast<const int32_t*>(lens_.p);
            const size_t n = static_cast<size_t>(B_) * static_cast<size_t>(T_);
            pos32_.resize(n);
            seg_b_.clear(); seg_t0_.clear(); seg_len_.clear();
            for (int i = 0; i < B_; ++i) {
                int32_t* pos = pos32_.data() + static_cast<size_t>(i) * static_cast<size_t>(T_);
                int t = 0;
                auto add = [&](int L) {
                    seg_b_.push_back(i); seg_t0_.push_back(t); seg_len_.push_back(L);
                    for (int k = 0; k < L; ++k) pos[t + k] = k;
                    t += L;
                };
                for (int64_t c = offs[row0 + i]; c < offs[row0 + i + 1]; ++c) add(lens[c]);
                if (t < T_) add(T_ - t);                        // the pad tail: a segment of its own
            }
            OwnTensor::TensorOptions o32 = OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::Int32)
                                               .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, device_idx_));
            last_pos_ = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, o32);
            check_cuda(cudaMemcpyAsync(last_pos_.data(), pos32_.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s),
                       "positions");
        }
        b.token_offset = off;                                   // provenance: first token of the batch in the file
        b.shard_index  = static_cast<int>(epoch());
        last_valid_ = valid_of(next_);
        ++next_;
        return b;
    }

    // Valid (non-65535) labels in this rank's next n micro-batches, without consuming them.
    int64_t valid_tokens_ahead(int64_t n) const {
        int64_t sum = 0;
        for (int64_t g = next_; g < next_ + n; ++g) sum += valid_of(g);
        return sum;
    }
    int64_t last_valid_tokens() const { return last_valid_; }      // of the batch next_batch() just returned
    // v2 packed rows: true when seq_lengths.bin is present. Then, for the batch next_batch() just returned: the RoPE
    // positions (int32 [B, T] on the GPU) and the segments of its B rows as HOST lists of equal length -- row in the
    // batch, start, length; every chat, then each pad tail -- the input of OwnTensor::cuda::gqa_chat_plan_make.
    bool packed() const { return packed_; }
    const OwnTensor::Tensor& last_positions() const { return last_pos_; }
    const std::vector<int32_t>& last_segment_rows() const { return seg_b_; }
    const std::vector<int32_t>& last_segment_starts() const { return seg_t0_; }
    const std::vector<int32_t>& last_segment_lengths() const { return seg_len_; }
    int64_t epoch() const { return next_ / M_; }                    // completed epochs
    int64_t micro_batches_per_epoch() const { return M_; }          // per rank
    int64_t rows() const { return N_; }
    int64_t total_valid_tokens() const { return total_valid_; }
    void reset() { next_ = 0; }
    void skip_micro_batches(int64_t n) { next_ += n; }              // resume

private:
    struct Map { void* p = nullptr; size_t bytes = 0; };

    int64_t first_row(int64_t g) const { return ((g % M_) * world_ + rank_) * B_; }
    int64_t valid_of(int64_t g) const {
        const uint32_t* cnt = static_cast<const uint32_t*>(cnt_.p);
        int64_t sum = 0;
        for (int64_t r = first_row(g); r < first_row(g) + B_; ++r) sum += cnt[r];
        return sum;
    }

    void check_files(int64_t vocab_rows) {
        const uint16_t* in = static_cast<const uint16_t*>(in_.p);
        const uint16_t* lab = static_cast<const uint16_t*>(lab_.p);
        const uint32_t* cnt = static_cast<const uint32_t*>(cnt_.p);
        for (int64_t r = 0; r < N_; ++r) {
            int64_t valid = 0;
            for (int64_t t = 0; t < T_; ++t) {
                const uint16_t x = in[r * T_ + t], y = lab[r * T_ + t];
                if (x >= vocab_rows) bad(r, "input id " + std::to_string(x) + " at position " + std::to_string(t) + " >= vocab rows");
                if (y == kIgnore) continue;
                if (y >= vocab_rows) bad(r, "label " + std::to_string(y) + " at position " + std::to_string(t) + " >= vocab rows and not 65535");
                ++valid;
            }
            if (packed_) {   // the chats fit the row, and the stored boundaries are the real ones: every chat's
                             // last position predicts the end token, and the pad tail trains nothing
                const int64_t* offs = static_cast<const int64_t*>(offs_.p);
                const int32_t* lens = static_cast<const int32_t*>(lens_.p);
                if (offs[r] < 0 || offs[r + 1] < offs[r] || offs[r + 1] > offs[N_])
                    bad(r, "seq_offsets.bin is not non-decreasing inside [0, " + std::to_string(offs[N_]) + "]");
                int64_t t = 0;
                for (int64_t c = offs[r]; c < offs[r + 1]; ++c) {
                    if (lens[c] < 1 || t + lens[c] > T_)
                        bad(r, "chat " + std::to_string(c - offs[r]) + " has length " + std::to_string(lens[c]) +
                               " at position " + std::to_string(t) + " (row length " + std::to_string(T_) + ")");
                    t += lens[c];
                    if (lab[r * T_ + t - 1] != kEnd)
                        bad(r, "the chat ending at position " + std::to_string(t) + " does not end with the end token");
                }
                for (int64_t u = t; u < T_; ++u)
                    if (lab[r * T_ + u] != kIgnore) bad(r, "a trained label in the pad tail at " + std::to_string(u));
            }
            if (valid != static_cast<int64_t>(cnt[r]))
                bad(r, "valid_counts.bin says " + std::to_string(cnt[r]) + ", labels.bin has " + std::to_string(valid));
            if (valid == 0) bad(r, "no trained token");
            total_valid_ += valid;
        }
    }
    [[noreturn]] void bad(int64_t r, const std::string& what) const {
        throw std::runtime_error("SftDataLoader: " + dir_ + " row " + std::to_string(r) + ": " + what);
    }
    static std::string read_text(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("SftDataLoader: cannot open " + path);
        std::stringstream ss; ss << f.rdbuf(); return ss.str();
    }
    static int64_t json_int(const std::string& s, const std::string& key) {   // top-level  "key": <integer>
        const size_t at = s.find("\"" + key + "\":");
        if (at == std::string::npos) throw std::runtime_error("SftDataLoader: meta.json has no \"" + key + "\"");
        return std::stoll(s.substr(at + key.size() + 3));
    }
    static Map map_file(const std::string& path, size_t expect_bytes) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("SftDataLoader: cannot open " + path);
        struct stat st {};
        if (fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) != expect_bytes) {
            ::close(fd);
            throw std::runtime_error("SftDataLoader: " + path + " has " + std::to_string(st.st_size) +
                                     " bytes, meta.json says it should have " + std::to_string(expect_bytes));
        }
        void* p = ::mmap(nullptr, expect_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw std::runtime_error("SftDataLoader: mmap failed: " + path);
        return Map{p, expect_bytes};
    }
    static void unmap(Map& m) { if (m.p) ::munmap(m.p, m.bytes); m.p = nullptr; }
    static void check_cuda(cudaError_t e, const char* what) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("SftDataLoader: copy of ") + what + " to the GPU failed: " +
                                     cudaGetErrorString(e));
    }

    std::string dir_;
    int B_, T_, rank_, world_, device_idx_;
    int64_t N_ = 0, M_ = 0, next_ = 0, last_valid_ = 0, total_valid_ = 0;
    Map in_, lab_, cnt_, offs_, lens_;
    bool packed_ = false;
    std::vector<int32_t> pos32_;                      // host staging of the positions
    std::vector<int32_t> seg_b_, seg_t0_, seg_len_;   // segments of the last batch (host)
    OwnTensor::Tensor last_pos_;                      // of the last batch (empty when not packed)
};
