#!/usr/bin/env python3
"""P8 (2026-10-15): SftDataLoader.h reads packed rows as per-row chat-length lists (seq_lengths.bin + seq_offsets.bin,
TRL's seq_lengths) instead of seg_start.bin. Per micro-batch it uploads the RoPE positions and keeps the rows'
segments as host lists for the attention plan. Every replacement must match exactly once.
usage: p8_loader_patch.py <BluTrain_sft dir>"""
import sys, os
p = os.path.join(sys.argv[1], "sft/SftDataLoader.h")
s = open(p, encoding="utf-8").read()

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, (n, old[:120])
    s = s.replace(old, new)

rep("""//   <dir>/seg_start.bin     uint16 [N, T]  OPTIONAL (v2 packed rows, prepare_sft_data.py --pack ffd, 2026-10-15):
//                                          where the chat of each token starts in its row; the row's pad tail is a
//                                          segment of its own. Present = packed rows, absent = v1 (one chat per row).
//                                          For each micro-batch the loader also uploads seg_start and the RoPE
//                                          positions t - seg_start as int32 [B, T] (last_seg_start / last_positions).""",
"""//   <dir>/seq_lengths.bin   int32 [sum]    OPTIONAL, v2 packed rows (prepare_sft_data.py --pack ffd, 2026-10-15):
//   <dir>/seq_offsets.bin   int64 [N + 1]  the lengths of the chats of every row (TRL's seq_lengths column); row r's
//                                          chats are seq_lengths[seq_offsets[r] : seq_offsets[r+1]] in row order, and
//                                          the rest of the row is its pad tail. Present = packed rows, absent = v1
//                                          (one chat per row). For each micro-batch the loader uploads the RoPE
//                                          positions (index of each token inside its chat, int32 [B, T]) and keeps
//                                          the B rows' segments -- every chat, then each pad tail as a segment of its
//                                          own -- as host lists, the input of the attention plan (gqa_chat_plan_make).
//                                          The removed per-token format (seg_start.bin, the morning of 2026-10-15) is
//                                          refused, so old data cannot train with the chats of a row seeing each other.""")
rep("""        {   // v2 packed rows: seg_start.bin present (it is checked against the contract in check_files)
            struct stat st {};
            packed_ = ::stat((dir_ + "/seg_start.bin").c_str(), &st) == 0;
        }
        if (packed_) seg_ = map_file(dir_ + "/seg_start.bin", cells * sizeof(uint16_t));
        check_files(vocab_rows);""",
"""        {   // v2 packed rows: seq_lengths.bin + seq_offsets.bin (checked against the rows in check_files)
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
        check_files(vocab_rows);""")
rep("""                      << (packed_ ? " | PACKED rows (seg_start.bin: several chats per row)" : "") << std::endl;
    }
    ~SftDataLoader() { unmap(in_); unmap(lab_); unmap(cnt_); unmap(seg_); }""",
"""                      << (packed_ ? " | PACKED rows (seq_lengths.bin: " + std::to_string(static_cast<const int64_t*>(
                                        offs_.p)[N_]) + " chats, each attends only to itself)" : std::string())
                      << std::endl;
    }
    ~SftDataLoader() { unmap(in_); unmap(lab_); unmap(cnt_); unmap(offs_); unmap(lens_); }""")
rep("""        if (packed_) {
            // seg_start of this rank's B rows as int32 [B, T], and the RoPE positions t - seg_start (they restart at 0
            // for every chat). Staged through host vectors: a copy from pageable memory has read its source when it
            // returns, so the vectors can be refilled by the next call. New tensors every call: the attention backward
            // node keeps the forward's seg_start until the micro-step's backward has run.
            const uint16_t* sg = static_cast<const uint16_t*>(seg_.p) + off;
            const size_t n = static_cast<size_t>(B_) * static_cast<size_t>(T_);
            seg32_.resize(n); pos32_.resize(n);
            for (size_t i = 0; i < n; ++i) {
                seg32_[i] = static_cast<int32_t>(sg[i]);
                pos32_[i] = static_cast<int32_t>(i % static_cast<size_t>(T_)) - static_cast<int32_t>(sg[i]);
            }
            OwnTensor::TensorOptions o32 = OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::Int32)
                                               .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, device_idx_));
            last_seg_ = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, o32);
            last_pos_ = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, o32);
            check_cuda(cudaMemcpyAsync(last_seg_.data(), seg32_.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s),
                       "seg_start");
            check_cuda(cudaMemcpyAsync(last_pos_.data(), pos32_.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s),
                       "positions");
        }""",
"""        if (packed_) {
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
        }""")
rep("""    // v2 packed rows: true when seg_start.bin is present; then, for the batch next_batch() just returned,
    // int32 [B, T] CUDA tensors: where each token's chat starts, and its RoPE position (t - seg_start).
    bool packed() const { return packed_; }
    const OwnTensor::Tensor& last_seg_start() const { return last_seg_; }
    const OwnTensor::Tensor& last_positions() const { return last_pos_; }""",
"""    // v2 packed rows: true when seq_lengths.bin is present. Then, for the batch next_batch() just returned: the RoPE
    // positions (int32 [B, T] on the GPU) and the segments of its B rows as HOST lists of equal length -- row in the
    // batch, start, length; every chat, then each pad tail -- the input of OwnTensor::cuda::gqa_chat_plan_make.
    bool packed() const { return packed_; }
    const OwnTensor::Tensor& last_positions() const { return last_pos_; }
    const std::vector<int32_t>& last_segment_rows() const { return seg_b_; }
    const std::vector<int32_t>& last_segment_starts() const { return seg_t0_; }
    const std::vector<int32_t>& last_segment_lengths() const { return seg_len_; }""")
rep("""    static constexpr uint16_t kIgnore = 65535;
""", """    static constexpr uint16_t kIgnore = 65535;
    static constexpr uint16_t kEnd = 50256;   // <|endoftext|>: every chat's last target (prepare_sft_data.py PAD_ID)
""")
rep("""            if (packed_) {   // the contract the attention kernels rely on (AttentionKernels.h): seg[0] = 0, and
                             // seg[t] is t (a segment starts here) or seg[t-1] (so seg[t] <= t as well)
                const uint16_t* sg = static_cast<const uint16_t*>(seg_.p) + r * T_;
                if (sg[0] != 0) bad(r, "seg_start[0] is " + std::to_string(sg[0]) + ", not 0");
                for (int64_t t = 1; t < T_; ++t)
                    if (sg[t] != t && sg[t] != sg[t - 1])
                        bad(r, "seg_start breaks the contract at position " + std::to_string(t));
            }""",
"""            if (packed_) {   // the chats fit the row, and the stored boundaries are the real ones: every chat's
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
            }""")
rep("""    Map in_, lab_, cnt_, seg_;
    bool packed_ = false;
    std::vector<int32_t> seg32_, pos32_;              // host staging for the packed-row tensors
    OwnTensor::Tensor last_seg_, last_pos_;           // of the last batch (empty when not packed)
};""", """    Map in_, lab_, cnt_, offs_, lens_;
    bool packed_ = false;
    std::vector<int32_t> pos32_;                      // host staging of the positions
    std::vector<int32_t> seg_b_, seg_t0_, seg_len_;   // segments of the last batch (host)
    OwnTensor::Tensor last_pos_;                      // of the last batch (empty when not packed)
};""")
print([l.strip()[:100] for l in s.splitlines() if "seg_start" in l]); assert s.count("seg_start") == 3, s.count("seg_start")   # the format note, the old-file test, its error
open(p, "w", encoding="utf-8").write(s)
print("patched SftDataLoader.h")
