#pragma once

#include "communication/include/ProcessGroupNCCL.h"
#include "core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// Base class for all ring rotators.
// A rotator shifts KV (or grad) buffers around a ring of ranks.
//
// Usage pattern (per ring iteration):
//   1. exchange_buffers(curr_buffer)   -- initiates async send of curr_buffer
//   2. next_buffer()                   -- blocks until the received buffer is ready
// ---------------------------------------------------------------------------
class RingRotatorBase {
public:
    RingRotatorBase(std::shared_ptr<ProcessGroupNCCL> pg)
        : pg_(pg),
          rank_(pg->get_rank()),
          world_size_(pg->get_worldsize()) {}

    virtual ~RingRotatorBase() = default;

    virtual void exchange_buffers(Tensor& curr_buffer) = 0;
    virtual Tensor next_buffer() = 0;

protected:
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int rank_;
    int world_size_;
};


// ---------------------------------------------------------------------------
// P2P Ring Rotator (optimized)
//
// Pre-allocates the receive buffer on first use to avoid per-step allocation.
// Uses ncclSend / ncclRecv point-to-point operations.
// Even ranks send first then receive; odd ranks receive first then send.
// ---------------------------------------------------------------------------
class P2PRingRotator : public RingRotatorBase {
public:
    P2PRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), recv_buffer_(), buffer_allocated_(false) {}

    void exchange_buffers(Tensor& curr_buffer) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t count = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        // Pre-allocate receive buffer once, reuse across ring steps
        if (!buffer_allocated_) {
            recv_buffer_ = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            buffer_allocated_ = true;
        }

        std::vector<std::shared_ptr<Work>> reqs;

        if (rank_ % 2 == 0) {
            auto send_req = pg_->send_async(
                curr_buffer.data<float>(), count, dtype, next_rank);
            reqs.push_back(send_req);

            auto recv_req = pg_->recieve_async(
                recv_buffer_.data<float>(), count, dtype, prev_rank);
            reqs.push_back(recv_req);
        } else {
            auto recv_req = pg_->recieve_async(
                recv_buffer_.data<float>(), count, dtype, prev_rank);
            reqs.push_back(recv_req);

            auto send_req = pg_->send_async(
                curr_buffer.data<float>(), count, dtype, next_rank);
            reqs.push_back(send_req);
        }

        pending_reqs_ = reqs;
    }

    Tensor next_buffer() override {
        for (auto& req : pending_reqs_) {
            if (req) {
                req->wait();
            }
        }
        pending_reqs_.clear();

        if (!recv_buffer_.is_valid()) {
            throw std::runtime_error("P2PRingRotator::next_buffer: no buffer available");
        }
        return recv_buffer_;
    }

private:
    Tensor recv_buffer_;
    bool buffer_allocated_;
    std::vector<std::shared_ptr<Work>> pending_reqs_;
};


// ---------------------------------------------------------------------------
// AlltoAll Ring Rotator
//
// Implements ring rotation via ncclAlltoAll using the shifted permutation
// pattern dsts = [1, 2, ..., n-1, 0].
//
// Each rank builds a send buffer with world_size chunks. Only the chunk
// destined for next_rank contains the actual KV data; all other chunks
// are zeroed. After the collective, we extract the data from the
// prev_rank slot of the receive buffer.
//
// This is deadlock-free and topology-aware (NCCL optimizes the routing).
// ---------------------------------------------------------------------------
class AlltoAllRingRotator : public RingRotatorBase {
public:
    AlltoAllRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), recv_buffer_(), buffer_allocated_(false) {}

    void exchange_buffers(Tensor& curr_buffer) override {
        int next_rank = (rank_ + 1) % world_size_;

        size_t per_rank_count = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();
        size_t elem_size = Tensor::dtype_size(dtype);

        // Pre-allocate the scatter/gather buffers once (world_size chunks each)
        if (!buffer_allocated_) {
            Shape agg_shape({{static_cast<int64_t>(per_rank_count * world_size_)}});
            send_buffer_ = Tensor::zeros(agg_shape, curr_buffer.opts());
            recv_agg_buffer_ = Tensor::zeros(agg_shape, curr_buffer.opts());
            recv_buffer_ = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            buffer_allocated_ = true;
            per_rank_numel_ = per_rank_count;
        }

        // Zero the send buffer, then place curr_buffer into the next_rank slot
        // dsts = [1, 2, ..., n-1, 0]:  rank i sends its data to rank (i+1)%N
        cudaMemsetAsync(send_buffer_.data(), 0,
                        per_rank_count * world_size_ * elem_size, 0);

        uint8_t* dst_slot = static_cast<uint8_t*>(send_buffer_.data())
                            + next_rank * per_rank_count * elem_size;
        cudaMemcpyAsync(dst_slot, curr_buffer.data(),
                        per_rank_count * elem_size,
                        cudaMemcpyDeviceToDevice, 0);

        // Launch ncclAlltoAll: rank j receives the j-th chunk from every rank
        pending_work_ = pg_->alltoall_async(
            send_buffer_.data(),
            recv_agg_buffer_.data(),
            per_rank_count,
            dtype);
    }

    Tensor next_buffer() override {
        if (pending_work_) {
            pending_work_->wait();
            pending_work_ = nullptr;
        }

        if (!recv_buffer_.is_valid()) {
            throw std::runtime_error("AlltoAllRingRotator::next_buffer: no buffer available");
        }

        // Extract the data from the prev_rank slot
        // After alltoall, slot j in recv_agg_buffer_ contains data sent by rank j
        // The rank that sent us data is prev_rank = (rank_ - 1 + N) % N
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;
        size_t elem_size = Tensor::dtype_size(recv_buffer_.dtype());

        const uint8_t* src_slot = static_cast<const uint8_t*>(recv_agg_buffer_.data())
                                  + prev_rank * per_rank_numel_ * elem_size;
        cudaMemcpyAsync(recv_buffer_.data(), src_slot,
                        per_rank_numel_ * elem_size,
                        cudaMemcpyDeviceToDevice, 0);
        cudaStreamSynchronize(0);

        return recv_buffer_;
    }

private:
    Tensor send_buffer_;       // world_size chunks for alltoall input
    Tensor recv_agg_buffer_;   // world_size chunks for alltoall output
    Tensor recv_buffer_;       // single-chunk output for the caller
    bool buffer_allocated_;
    size_t per_rank_numel_ = 0;
    std::shared_ptr<Work> pending_work_;
};


// ---------------------------------------------------------------------------
// AllGather Ring Rotator
//
// Gathers all buffers from all ranks in a single all_gather call on the
// first exchange. Subsequent calls just index into the gathered buffer.
// ---------------------------------------------------------------------------
class AllGatherRingRotator : public RingRotatorBase {
public:
    AllGatherRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), idx_(0), aggregated_buffer_() {}

    void exchange_buffers(Tensor& curr_buffer) override {
        idx_ += 1;

        if (!aggregated_buffer_.is_valid()) {
            size_t per_rank_count = static_cast<size_t>(curr_buffer.numel());
            size_t total_count = per_rank_count * static_cast<size_t>(world_size_);
            Dtype dtype = curr_buffer.dtype();

            Shape agg_shape({{static_cast<int64_t>(total_count)}});
            aggregated_buffer_ = Tensor::empty(agg_shape, curr_buffer.opts());

            Tensor flat_input = curr_buffer.flatten();

            pg_->all_gather(
                flat_input.data<float>(),
                aggregated_buffer_.data<float>(),
                per_rank_count,
                dtype,
                true);

            per_rank_numel_ = per_rank_count;
        }
    }

    Tensor next_buffer() override {
        if (!aggregated_buffer_.is_valid()) {
            throw std::runtime_error("AllGatherRingRotator::next_buffer: exchange_buffers not called");
        }

        int source_rank = ((rank_ - idx_) % world_size_ + world_size_) % world_size_;
        int64_t offset = static_cast<int64_t>(source_rank) * static_cast<int64_t>(per_rank_numel_);

        float* base_ptr = aggregated_buffer_.data<float>() + offset;

        Shape chunk_shape({{static_cast<int64_t>(per_rank_numel_)}});
        Tensor chunk = Tensor::empty(chunk_shape, aggregated_buffer_.opts());

        cudaMemcpyAsync(
            chunk.data<float>(),
            base_ptr,
            per_rank_numel_ * sizeof(float),
            cudaMemcpyDeviceToDevice,
            0);

        return chunk;
    }

private:
    int idx_;
    Tensor aggregated_buffer_;
    size_t per_rank_numel_ = 0;
};