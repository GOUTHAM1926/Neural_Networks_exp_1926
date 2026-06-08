#pragma once
// =============================================================================
// Embedding.h — Token and Position embeddings
//   - Embedding:    Standard embedding (single GPU / DDP)
//   - VPEmbedding:  Vocab-parallel embedding (TP / 2D parallel)
//   - PosEmbedding: Position embedding (replicated, used in 2D parallel)
// =============================================================================

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"

using namespace OwnTensor;

// Forward declare for VPEmbedding (only needed when TP is compiled in)
class ProcessGroupNCCL;
class Work;

namespace gpt2 {

// =============================================================================
// Standard Embedding (single GPU / DDP)
// =============================================================================
class Embedding : public nn::Module {
public:
    Tensor weight;  // [vocab_size, n_embd]

    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device,
              uint64_t seed = 1234)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
        register_parameter(weight);
    }

    Tensor forward(const Tensor& indices) override {
        return autograd::embedding(weight, indices);
    }

private:
    int64_t vocab_size_ = 0;
    int64_t embed_dim_ = 0;
};

// =============================================================================
// Vocab-Parallel Embedding (sharded along vocab dimension, for TP)
// =============================================================================
class VPEmbedding {
public:
    Tensor weight;  // [local_vocab, n_embd]

    VPEmbedding() = default;

    VPEmbedding(int64_t vocab_size, int64_t n_embd,
                std::shared_ptr<ProcessGroupNCCL>& tp_pg,
                DeviceIndex device, uint64_t seed = 1234);

    // Synchronous forward
    Tensor forward(const Tensor& indices);

    // Async forward: returns (partial_result, work_handle)
    // Caller must wait on work before using the tensor
    std::pair<Tensor, std::shared_ptr<Work>> forward_async(const Tensor& indices);

private:
    int64_t n_embd_ = 0, local_v_ = 0, vocab_start_ = 0, vocab_end_ = 0;
    std::shared_ptr<ProcessGroupNCCL> tp_pg_;
};

// =============================================================================
// Position Embedding (replicated, for 2D parallel model)
// =============================================================================
class PosEmbedding : public nn::Module {
public:
    Tensor weight;

    PosEmbedding() = default;
    PosEmbedding(int64_t context_length, int64_t n_embd, DeviceIndex device,
                 uint64_t seed = 1234)
    {
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(device)
                                            .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{context_length, n_embd}}, opts, seed, 0.02f);
        register_parameter(weight);
    }

    Tensor forward(const Tensor& indices) override {
        return autograd::embedding(weight, indices);
    }
};

}  // namespace gpt2
