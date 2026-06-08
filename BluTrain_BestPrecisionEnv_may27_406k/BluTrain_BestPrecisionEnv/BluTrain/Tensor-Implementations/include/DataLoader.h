// #pragma once

// #include <algorithm>
// #include <cstdint>
// #include <cstdlib>
// #include <cstring>
// #include <filesystem>
// #include <iostream>
// #include <stdexcept>

// #include <string>
// #include <vector>
// #include <cuda_runtime.h>

// #include <fcntl.h>
// #include <sys/mman.h>
// #include <sys/stat.h>
// #include <unistd.h>

// #include "TensorLib.h"

// namespace fs = std::filesystem;



// static int getenv_int(const char* key, int def) {
//     const char* v = std::getenv(key);
//     return v ? std::atoi(v) : def;
// }

// static std::vector<std::string> list_shards(const std::string& root,
//                                             const std::string& split,
//                                             const std::string& ext = ".bin") {
//     std::vector<std::string> shards;
//     for (const auto& e : fs::directory_iterator(root)) {
//         if (!e.is_regular_file()) continue;
//         auto p = e.path();
//         std::string name = p.filename().string();
//         if (p.extension() == ext && name.find(split) != std::string::npos) {
//             shards.push_back(p.string());
//         }
//     }
//     std::sort(shards.begin(), shards.end());
//     return shards;
// }

// // mmap shard view 

// class UInt16ShardView {
// public:
//     UInt16ShardView() = default;
//     ~UInt16ShardView() { close(); }

//     UInt16ShardView(const UInt16ShardView&) = delete;
//     UInt16ShardView& operator=(const UInt16ShardView&) = delete;

//     void open(const std::string& path, size_t max_tokens) {
//         close();
//         path_ = path;

//         fd_ = ::open(path.c_str(), O_RDONLY);
//         if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

//         struct stat st {};
//         if (fstat(fd_, &st) != 0) {
//             ::close(fd_); fd_ = -1;
//             throw std::runtime_error("failed to stat: " + path);
//         }

//         file_bytes_ = static_cast<size_t>(st.st_size);
//         if (file_bytes_ % sizeof(u_int16_t) != 0) {
//             ::close(fd_); fd_ = -1;
//             throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
//         }

//         size_t total_tokens = file_bytes_ /2;
//         tokens_ = std::min(total_tokens, max_tokens);

//         // Map the whole file. OS will page-in only what you touch.
//         data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
//         if (data_ == MAP_FAILED) {
//             ::close(fd_); fd_ = -1; data_ = nullptr;
//             throw std::runtime_error("mmap failed: " + path);
//         }

//         // Optimization #4: hint the kernel we'll read sequentially
//         ::madvise(data_, file_bytes_, MADV_SEQUENTIAL);
//     }

//     void close() {
//         if (data_) {
//             ::munmap(data_, file_bytes_);
//             data_ = nullptr;
//         }
//         if (fd_ >= 0) {
//             ::close(fd_);
//             fd_ = -1;
//         }
//         file_bytes_ = 0;
//         tokens_ = 0;
//         path_.clear();
//     }

//     size_t size_tokens() const { return tokens_; }
//     const std::string& path() const { return path_; }

//     // Optimization #2: direct pointer into the mmap'd region (zero-copy access)
//     const u_int16_t* data_ptr() const {
//         return reinterpret_cast<const u_int16_t*>(data_);
//     }

//     // Optimization #1: memcpy instead of element-by-element loop
//     void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
//         if (start + count > tokens_) throw std::out_of_range("read_block out of range");
//         out.resize(count);
//         const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
//         std::memcpy(out.data(), p + start, count * sizeof(u_int16_t));
//     }

// private:
//     std::string path_;
//     int fd_ = -1;
//     void* data_ = nullptr;
//     size_t file_bytes_ = 0;
//     size_t tokens_ = 0;
// };


// // Optimization #6: removed dead x/y vectors from Batch
// struct Batch {
//     int B = 0, T = 0;
//     OwnTensor::Tensor input;
//     OwnTensor::Tensor target;
// };

// class DataLoaderLite {
// public:
//     DataLoaderLite(int B, int T,
//                      int rank, int world_size,
//                      const std::string& split,
//                      const std::string& data_root,
//                      bool master_process = true,
//                      size_t max_tokens_per_shard = 100000000,
//                      int device_idx = 0)
//         : B_(B), T_(T),
//           rank_(rank), world_(world_size),
//           split_(split), root_(data_root),
//           master_(master_process),
//           max_tokens_(max_tokens_per_shard),
//           device_idx_(device_idx),
//           pinned_x_(nullptr), pinned_y_(nullptr) {

//         if (!(split_ == "train" || split_ == "val"))
//             throw std::runtime_error("split must be 'train' or 'val'");
//         if (B_ <= 0 || T_ <= 0)
//             throw std::runtime_error("B and T must be > 0");
//         if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
//             throw std::runtime_error("invalid rank/world_size");

//         shards_ = list_shards(root_, split_, ".bin");
//         if (shards_.empty())
//             throw std::runtime_error("no .bin shards found for split " + split_);

//         if (master_) {
//             std::cout << "found " << shards_.size() << " shards for split " << split_ << "\n";
//         }

//         // Optimization #3: allocate pinned (page-locked) staging buffers once
//         const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
//         pinned_x_ = nullptr;
//         pinned_y_ = nullptr;
//         cudaError_t err1 = cudaHostAlloc(&pinned_x_, BT * sizeof(u_int16_t), cudaHostAllocDefault);
//         cudaError_t err2 = cudaHostAlloc(&pinned_y_, BT * sizeof(u_int16_t), cudaHostAllocDefault);
//         if (err1 != cudaSuccess || err2 != cudaSuccess) {
//             std::cerr << "Warning: cudaHostAlloc failed, falling back to pageable memory\n";
//             if (pinned_x_) { cudaFreeHost(pinned_x_); pinned_x_ = nullptr; }
//             if (pinned_y_) { cudaFreeHost(pinned_y_); pinned_y_ = nullptr; }
//         }

//         // Optimization #5: pre-allocate tensors once, reuse across batches
//         OwnTensor::DeviceIndex dev(OwnTensor::Device::CUDA, device_idx_);
//         input_tensor_ = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
//             OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));
//         target_tensor_ = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
//             OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));

//         reset();
//     }

//     ~DataLoaderLite() {
//         if (pinned_x_) { cudaFreeHost(pinned_x_); pinned_x_ = nullptr; }
//         if (pinned_y_) { cudaFreeHost(pinned_y_); pinned_y_ = nullptr; }
//     }

//     // Non-copyable (pinned buffer ownership)
//     DataLoaderLite(const DataLoaderLite&) = delete;
//     DataLoaderLite& operator=(const DataLoaderLite&) = delete;

//     void reset() {
//         current_shard_ = 0;
//         shard_.open(shards_[current_shard_], max_tokens_);

//         // current_position = B*T*process_rank
//         pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
//     }

//     Batch next_batch() {
//         const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
//         const size_t need = BT + 1; // because y is shifted by 1

//         if (shard_.size_tokens() < need) {
//             throw std::runtime_error("shard too small for one batch: " + shard_.path());
//         }

//         if (pos_ + need > shard_.size_tokens()) {
//             advance_shard();
//         }

//         // Optimization #2: zero-copy pointer directly into mmap'd region
//         const u_int16_t* tokens = shard_.data_ptr() + pos_;

//         Batch b;
//         b.B = B_; b.T = T_;

//         if (pinned_x_) {
//             // Stage into pinned buffers, then cudaMemcpy directly from pinned → GPU
//             // This enables DMA transfer (bypasses CPU, ~2× faster than pageable)
//             std::memcpy(pinned_x_, tokens, BT * sizeof(u_int16_t));
//             std::memcpy(pinned_y_, tokens + 1, BT * sizeof(u_int16_t));

//             cudaMemcpy(input_tensor_.data(), pinned_x_, BT * sizeof(u_int16_t), cudaMemcpyHostToDevice);
//             cudaMemcpy(target_tensor_.data(), pinned_y_, BT * sizeof(u_int16_t), cudaMemcpyHostToDevice);
//         } else {
//             // Fallback: use set_data with pageable memory
//             x_buf_.resize(BT);
//             y_buf_.resize(BT);
//             std::memcpy(x_buf_.data(), tokens, BT * sizeof(u_int16_t));
//             std::memcpy(y_buf_.data(), tokens + 1, BT * sizeof(u_int16_t));
//             input_tensor_.set_data(x_buf_);
//             target_tensor_.set_data(y_buf_);
//         }

//         b.input = input_tensor_;
//         b.target = target_tensor_;

//         //current_position += B*T*num_processes
//         pos_ += BT * static_cast<size_t>(world_);

//         // if current_position + (B*T*num_processes + 1) > len(tokens):
//         //     load next shard; self.current_position = B*T*rank
//         if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
//             advance_shard();
//         }

//         return b;
//     }

//     void skip_batches(size_t n) {
//         const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
//         const size_t batch_stride = BT * static_cast<size_t>(world_);
//         const size_t need = BT + 1;

//         for (size_t i = 0; i < n; ++i) {
//             if (pos_ + need > shard_.size_tokens()) {
//                 advance_shard();
//             }
            
//             // Advance position
//             pos_ += batch_stride;

//             // Check if we need to advance shard for NEXT batch
//             if (pos_ + (batch_stride + 1) > shard_.size_tokens()) {
//                 advance_shard();
//             }
//         }
//     }

// private:
//     void advance_shard() {
//         current_shard_ = (current_shard_ + 1) % shards_.size();
//         std::cout << "Current shard: "<< shards_[current_shard_] << std::endl;
//         std::cout << max_tokens_ << std::endl;
//         shard_.open(shards_[current_shard_], max_tokens_);

//         const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
//         pos_ = BT * static_cast<size_t>(rank_);
//     }

//     int B_, T_;
//     int rank_, world_;
//     std::string split_, root_;
//     bool master_;
//     size_t max_tokens_;
//     int device_idx_;

//     std::vector<std::string> shards_;
//     size_t current_shard_ = 0;
//     size_t pos_ = 0;

//     UInt16ShardView shard_;

//     // Optimization #3: pinned staging buffers for direct H2D DMA transfer
//     u_int16_t* pinned_x_;
//     u_int16_t* pinned_y_;

//     // Fallback pageable buffers (only used if pinned alloc fails)
//     std::vector<u_int16_t> x_buf_;
//     std::vector<u_int16_t> y_buf_;

//     // Optimization #5: pre-allocated tensors reused across batches
//     OwnTensor::Tensor input_tensor_;
//     OwnTensor::Tensor target_tensor_;
// };

















#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "TensorLib.h"

namespace fs = std::filesystem;



static int getenv_int(const char* key, int def) {
    const char* v = std::getenv(key);
    return v ? std::atoi(v) : def;
}

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

// mmap shard view

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    UInt16ShardView(const UInt16ShardView&) = delete;
    UInt16ShardView& operator=(const UInt16ShardView&) = delete;

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(u_int16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ /2;
        tokens_ = std::min(total_tokens, max_tokens);

        // Map the whole file. OS will page-in only what you touch.
        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }

        // Optimization #4: hint the kernel we'll read sequentially
        ::madvise(data_, file_bytes_, MADV_SEQUENTIAL);
    }

    void close() {
        if (data_) {
            ::munmap(data_, file_bytes_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        file_bytes_ = 0;
        tokens_ = 0;
        path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    // Optimization #2: direct pointer into the mmap'd region (zero-copy access)
    const u_int16_t* data_ptr() const {
        return reinterpret_cast<const u_int16_t*>(data_);
    }

    // Optimization #1: memcpy instead of element-by-element loop
    void read_block(size_t start, size_t count, std::vector<u_int16_t>& out) const {
        if (start + count > tokens_) throw std::out_of_range("read_block out of range");
        out.resize(count);
        const u_int16_t* p = reinterpret_cast<const u_int16_t*>(data_);
        std::memcpy(out.data(), p + start, count * sizeof(u_int16_t));
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};


// Optimization #6: removed dead x/y vectors from Batch
struct Batch {
    int B = 0, T = 0;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

// =============================================================================
// DataLoaderLite — async + double-buffered GPU prefetch
// =============================================================================
//
// Two pinned-host staging buffers and two device tensors are kept in a ping-
// pong. A dedicated CUDA stream (`copy_stream_`) issues `cudaMemcpyAsync` for
// the NEXT batch while the consumer is still computing on the CURRENT one
// using the default stream. Stream coordination is via three events:
//   h2d_done_[0/1]    : copy_stream signals when buffer i's H2D is finished;
//                       default stream waits on this before kernels read it.
//   consumer_done_evt_: default stream signals "all queued reads of last
//                       returned buffer are committed"; copy_stream waits on
//                       this before overwriting that buffer with the next
//                       prefetch (otherwise the H2D would race the consumer's
//                       in-flight reads).
//
// Pipeline (steady state, after priming):
//
//   call N:
//     [host] record consumer_done on default, copy_stream waits on it
//     [host] CPU memcpy mmap → pinned[1-buf_idx]; cudaMemcpyAsync pinned →
//            tensor[1-buf_idx] on copy_stream; cudaEventRecord h2d_done[1-buf_idx]
//     [host] cudaStreamWaitEvent(default, h2d_done[buf_idx])  -- non-blocking
//     [host] return tensor[buf_idx]; toggle buf_idx
//     [device] consumer's kernels read tensor[buf_idx_old] on default stream
//
// Falls back to the original synchronous path if pinned-host allocation
// fails (cudaHostAlloc returns non-zero), so deployment without pinned-mem
// support still works (just without the prefetch overlap).
// =============================================================================

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T,
                     int rank, int world_size,
                     const std::string& split,
                     const std::string& data_root,
                     bool master_process = true,
                     size_t max_tokens_per_shard = 100000000,
                     int device_idx = 0)
        : B_(B), T_(T),
          rank_(rank), world_(world_size),
          split_(split), root_(data_root),
          master_(master_process),
          max_tokens_(max_tokens_per_shard),
          device_idx_(device_idx) {

        pinned_x_[0] = pinned_x_[1] = nullptr;
        pinned_y_[0] = pinned_y_[1] = nullptr;

        if (!(split_ == "train" || split_ == "val"))
            throw std::runtime_error("split must be 'train' or 'val'");
        if (B_ <= 0 || T_ <= 0)
            throw std::runtime_error("B and T must be > 0");
        if (world_ <= 0 || rank_ < 0 || rank_ >= world_)
            throw std::runtime_error("invalid rank/world_size");

        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty())
            throw std::runtime_error("no .bin shards found for split " + split_);

        if (master_) {
            std::cout << "found " << shards_.size() << " shards for split " << split_ << "\n";
        }

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);

        // Allocate TWO pinned (page-locked) staging buffers per channel.
        // Pinned host memory is required for cudaMemcpyAsync to actually be
        // asynchronous — pageable memory falls back to a sync copy and
        // defeats the prefetch.
        bool pinned_ok = true;
        for (int i = 0; i < 2 && pinned_ok; ++i) {
            cudaError_t e1 = cudaHostAlloc(&pinned_x_[i], BT * sizeof(u_int16_t), cudaHostAllocDefault);
            cudaError_t e2 = cudaHostAlloc(&pinned_y_[i], BT * sizeof(u_int16_t), cudaHostAllocDefault);
            if (e1 != cudaSuccess || e2 != cudaSuccess) pinned_ok = false;
        }
        if (!pinned_ok) {
            std::cerr << "Warning: cudaHostAlloc failed, falling back to pageable / sync H2D\n";
            for (int i = 0; i < 2; ++i) {
                if (pinned_x_[i]) { cudaFreeHost(pinned_x_[i]); pinned_x_[i] = nullptr; }
                if (pinned_y_[i]) { cudaFreeHost(pinned_y_[i]); pinned_y_[i] = nullptr; }
            }
        }

        // Pre-allocate TWO device tensors per channel for the ping-pong. With
        // distinct device buffers the consumer can keep reading tensor[A]
        // while copy_stream is writing tensor[B] without any aliasing.
        OwnTensor::DeviceIndex dev(OwnTensor::Device::CUDA, device_idx_);
        for (int i = 0; i < 2; ++i) {
            input_tensor_[i] = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
                OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));
            target_tensor_[i] = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}},
                OwnTensor::TensorOptions().with_dtype(OwnTensor::Dtype::UInt16).with_device(dev));
        }

        // Stream + events for the async path. cudaStreamNonBlocking lets
        // copy_stream run concurrently with the default stream rather than
        // implicitly serialising on stream-0 sync points. Events use
        // DisableTiming since we only need ordering, not timestamps.
        if (pinned_x_[0]) {
            cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&h2d_done_[0], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&h2d_done_[1], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&consumer_done_evt_, cudaEventDisableTiming);
        }

        reset();
    }

    ~DataLoaderLite() {
        // Drain any in-flight async copies before tearing down resources.
        if (copy_stream_) cudaStreamSynchronize(copy_stream_);

        for (int i = 0; i < 2; ++i) {
            if (pinned_x_[i]) { cudaFreeHost(pinned_x_[i]); pinned_x_[i] = nullptr; }
            if (pinned_y_[i]) { cudaFreeHost(pinned_y_[i]); pinned_y_[i] = nullptr; }
            if (h2d_done_[i]) { cudaEventDestroy(h2d_done_[i]); h2d_done_[i] = nullptr; }
        }
        if (consumer_done_evt_) { cudaEventDestroy(consumer_done_evt_); consumer_done_evt_ = nullptr; }
        if (copy_stream_) { cudaStreamDestroy(copy_stream_); copy_stream_ = nullptr; }
    }

    // Non-copyable (pinned buffer + stream + events ownership)
    DataLoaderLite(const DataLoaderLite&) = delete;
    DataLoaderLite& operator=(const DataLoaderLite&) = delete;

    void reset() {
        // Drain in-flight prefetches whose contents target the OLD position.
        if (copy_stream_) cudaStreamSynchronize(copy_stream_);

        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);

        // current_position = B*T*process_rank
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);

        // Force re-priming on the next next_batch() — any prefetched buffers
        // now hold data from before the reset and must be discarded.
        primed_ = false;
        buf_idx_ = 0;
    }

    Batch next_batch() {
        // Pageable fallback: keep the original synchronous semantics so the
        // loader still works on systems where cudaHostAlloc failed.
        if (!pinned_x_[0]) {
            return next_batch_sync_fallback();
        }

        if (!primed_) {
            // Prime both buffers in order: buffer 0 holds batch 0, buffer 1
            // holds batch 1. Priming is a one-time cost paid on the first
            // next_batch() after construction or reset(). Both prefetches
            // serialise on copy_stream (stream-ordered), which is fine — they
            // run while the host is doing other init work.
            issue_prefetch(0);
            issue_prefetch(1);
            primed_ = true;
            buf_idx_ = 0;
        } else {
            // Steady state: refill the buffer we returned LAST call (= 1 -
            // buf_idx_, since we toggled at the end of that call). The
            // consumer of that previous call has been queueing kernels on
            // the default stream that read from that buffer. Record an event
            // on the default stream NOW to capture all of those queued
            // reads, then make copy_stream wait on the event before
            // overwriting the buffer.
            cudaEventRecord(consumer_done_evt_, /*default stream*/ 0);
            cudaStreamWaitEvent(copy_stream_, consumer_done_evt_, 0);
            issue_prefetch(1 - buf_idx_);
        }

        // Make the default (compute) stream wait until THIS batch's H2D has
        // landed in device memory. Non-blocking on the host: the call
        // returns immediately and the wait is enqueued on the default
        // stream so the consumer's first read kernel will block until the
        // event signals.
        cudaStreamWaitEvent(/*default stream*/ 0, h2d_done_[buf_idx_], 0);

        Batch b;
        b.B = B_;
        b.T = T_;
        b.input  = input_tensor_[buf_idx_];
        b.target = target_tensor_[buf_idx_];

        buf_idx_ = (buf_idx_ + 1) % 2;
        return b;
    }

    void skip_batches(size_t n) {
        // Discard any prefetch in flight: those buffers contain pre-skip
        // data which is about to be wrong. The next next_batch() re-primes.
        if (copy_stream_) cudaStreamSynchronize(copy_stream_);
        primed_ = false;

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t batch_stride = BT * static_cast<size_t>(world_);
        const size_t need = BT + 1;

        for (size_t i = 0; i < n; ++i) {
            if (pos_ + need > shard_.size_tokens()) {
                advance_shard();
            }
            pos_ += batch_stride;
            if (pos_ + (batch_stride + 1) > shard_.size_tokens()) {
                advance_shard();
            }
        }
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        std::cout << "Current shard: "<< shards_[current_shard_] << std::endl;
        std::cout << max_tokens_ << std::endl;
        shard_.open(shards_[current_shard_], max_tokens_);

        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        pos_ = BT * static_cast<size_t>(rank_);
    }

    // Stage one batch: mmap → pinned[idx] (CPU memcpy, sync), pinned →
    // device tensor[idx] (cudaMemcpyAsync on copy_stream), record
    // h2d_done_[idx]. Advances pos_ for the next prefetch. Caller is
    // responsible for ensuring it's safe to overwrite tensor[idx]
    // (e.g. via cudaStreamWaitEvent on consumer_done_evt_).
    void issue_prefetch(int idx) {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (shard_.size_tokens() < need) {
            throw std::runtime_error("shard too small for one batch: " + shard_.path());
        }
        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        const u_int16_t* tokens = shard_.data_ptr() + pos_;

        // CPU side: mmap → pinned. This is a pure host memcpy; cheap
        // (~few µs for B*T*2 = 8 KiB at the current config) and serialises
        // only the host. If this ever shows up in a profile, kicking it to
        // a worker thread is a follow-up.
        std::memcpy(pinned_x_[idx], tokens,     BT * sizeof(u_int16_t));
        std::memcpy(pinned_y_[idx], tokens + 1, BT * sizeof(u_int16_t));

        // GPU side: pinned → device, asynchronous. Returns immediately on
        // host; the GPU does the DMA on copy_stream concurrently with
        // anything running on the default stream.
        cudaMemcpyAsync(input_tensor_[idx].data(),  pinned_x_[idx],
                        BT * sizeof(u_int16_t), cudaMemcpyHostToDevice, copy_stream_);
        cudaMemcpyAsync(target_tensor_[idx].data(), pinned_y_[idx],
                        BT * sizeof(u_int16_t), cudaMemcpyHostToDevice, copy_stream_);

        cudaEventRecord(h2d_done_[idx], copy_stream_);

        // current_position += B*T*num_processes
        pos_ += BT * static_cast<size_t>(world_);

        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }
    }

    // Original synchronous path, used only when pinned-host alloc failed.
    // Kept verbatim so behaviour is unchanged in that fallback regime.
    Batch next_batch_sync_fallback() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        const size_t need = BT + 1;

        if (shard_.size_tokens() < need) {
            throw std::runtime_error("shard too small for one batch: " + shard_.path());
        }
        if (pos_ + need > shard_.size_tokens()) {
            advance_shard();
        }

        const u_int16_t* tokens = shard_.data_ptr() + pos_;

        Batch b;
        b.B = B_; b.T = T_;

        x_buf_.resize(BT);
        y_buf_.resize(BT);
        std::memcpy(x_buf_.data(), tokens,     BT * sizeof(u_int16_t));
        std::memcpy(y_buf_.data(), tokens + 1, BT * sizeof(u_int16_t));
        input_tensor_[0].set_data(x_buf_);
        target_tensor_[0].set_data(y_buf_);

        b.input  = input_tensor_[0];
        b.target = target_tensor_[0];

        pos_ += BT * static_cast<size_t>(world_);
        if (pos_ + (BT * static_cast<size_t>(world_) + 1) > shard_.size_tokens()) {
            advance_shard();
        }
        return b;
    }

    int B_, T_;
    int rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_;
    int device_idx_;

    std::vector<std::string> shards_;
    size_t current_shard_ = 0;
    size_t pos_ = 0;

    UInt16ShardView shard_;

    // Double-buffered pinned staging (2× pinned host buffers per channel).
    u_int16_t* pinned_x_[2];
    u_int16_t* pinned_y_[2];

    // Pageable fallback buffers (only used if pinned alloc fails — async
    // path becomes the sync path in that regime).
    std::vector<u_int16_t> x_buf_;
    std::vector<u_int16_t> y_buf_;

    // Two device tensors per channel for the ping-pong (index 0 also serves
    // the pageable fallback path).
    OwnTensor::Tensor input_tensor_[2];
    OwnTensor::Tensor target_tensor_[2];

    // Async-prefetch state.
    cudaStream_t copy_stream_         = nullptr;
    cudaEvent_t  h2d_done_[2]         = {nullptr, nullptr};
    cudaEvent_t  consumer_done_evt_   = nullptr;
    bool         primed_              = false;
    int          buf_idx_             = 0;  // index of the buffer to RETURN next
};