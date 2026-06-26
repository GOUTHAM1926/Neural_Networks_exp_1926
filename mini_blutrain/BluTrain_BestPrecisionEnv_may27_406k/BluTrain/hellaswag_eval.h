// HellaSwag evaluation helper for the C++ trainer (mirrors gpt2.py:507-538).
//
// Reads a pre-tokenized HellaSwag val binary produced by prep_hellaswag.py
// (the C++ side has no GPT-2 tokenizer) and runs the same completion-style
// scoring used in hellaswag.py: for each example, build a [4, T] token tensor,
// forward through the model, then for each of the 4 candidate endings compute
// the average cross-entropy over the completion span and pick the row with the
// lowest loss (acc_norm). Counts are summed across ranks via MPI.
//
// Cost: forward of [4, ~30] per example × (10042 / world_size) examples per
// eval. On 8x B200 each rank sees ~1255 examples, roughly 1-2 minutes per
// eval, comparable to gpt2.py's hellaswag block.
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include "TensorLib.h"
#include "autograd/operations/LossOps.h"
#include "checkpointing/GradMode.h"

namespace hellaswag {

struct Example {
    int max_len;        // ctx_len + max(end_lens), columns of the [4, T] tensor
    int ctx_len;        // shared-context length (same for all 4 rows)
    int label;          // ground-truth ending index, 0..3
    int end_lens[4];    // ending length per row (rows are padded to max_len)
    std::vector<int32_t> tokens;  // [4 * max_len], row-major
};

inline std::vector<Example> load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != "HSW1")
        throw std::runtime_error("bad magic in " + path
                                 + " (expected HSW1; run prep_hellaswag.py)");
    int32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), 4);

    std::vector<Example> exs;
    exs.reserve(n);
    for (int i = 0; i < n; ++i) {
        Example ex{};
        in.read(reinterpret_cast<char*>(&ex.max_len), 4);
        in.read(reinterpret_cast<char*>(&ex.ctx_len), 4);
        in.read(reinterpret_cast<char*>(&ex.label),   4);
        in.read(reinterpret_cast<char*>(ex.end_lens), 16);
        ex.tokens.resize(static_cast<size_t>(4) * ex.max_len);
        in.read(reinterpret_cast<char*>(ex.tokens.data()),
                static_cast<std::streamsize>(ex.tokens.size()) * 4);
        if (!in)
            throw std::runtime_error("short read at example " + std::to_string(i));
        exs.push_back(std::move(ex));
    }
    return exs;
}

// Returns acc_norm in [0, 1]. Master rank also prints it.
template <typename Model>
inline double evaluate(Model& model,
                       OwnTensor::DeviceIndex device,
                       int rank, int world_size,
                       const std::vector<Example>& exs,
                       bool is_master,
                       int step = -1) {
    using namespace OwnTensor;
    autograd::NoGradGuard no_grad;

    int64_t local_correct = 0;
    int64_t local_total   = 0;

    for (size_t i = 0; i < exs.size(); ++i) {
        // Same sharding strategy as gpt2.py:513 ("if i % ddp_world_size != ddp_rank: continue").
        if (static_cast<int>(i % static_cast<size_t>(world_size)) != rank) continue;
        const auto& ex = exs[i];
        const int T = ex.max_len;

        // Build [4, T] int64 tokens on host, then move to device.
        Tensor toks_cpu(Shape{{4, T}}, TensorOptions().with_dtype(Dtype::Int64));
        int64_t* td = toks_cpu.data<int64_t>();
        for (int j = 0; j < 4 * T; ++j)
            td[j] = static_cast<int64_t>(ex.tokens[j]);
        Tensor toks_dev = toks_cpu.to(device);

        // Forward: logits = [4, T, V].
        Tensor logits = model.forward(toks_dev);

        // For each of the 4 candidate endings, slice the completion span and
        // compute its mean cross-entropy. Predicting positions [ctx_len-1
        // .. ctx_len-1+end_len) against targets [ctx_len .. ctx_len+end_len)
        // matches the (logits[..., :-1, :], tokens[..., 1:]) shift in
        // hellaswag.get_most_likely_row.
        float losses[4];
        for (int r = 0; r < 4; ++r) {
            const int end_len = ex.end_lens[r];
            Tensor lg = logits
                .narrow_view(0, r, 1)
                .narrow_view(1, ex.ctx_len - 1, end_len)
                .contiguous();                                 // [1, end_len, V]
            Tensor tg = toks_dev
                .narrow_view(0, r, 1)
                .narrow_view(1, ex.ctx_len, end_len)
                .contiguous();                                 // [1, end_len]
            Tensor loss = autograd::sparse_cross_entropy_loss(lg, tg);
            losses[r] = loss.to_cpu().data<float>()[0];
        }

        int pred_norm = 0;
        for (int r = 1; r < 4; ++r)
            if (losses[r] < losses[pred_norm]) pred_norm = r;

        if (pred_norm == ex.label) ++local_correct;
        ++local_total;
    }

    int64_t global_correct = local_correct;
    int64_t global_total   = local_total;
    if (world_size > 1) {
        MPI_Allreduce(&local_correct, &global_correct, 1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_total,   &global_total,   1, MPI_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD);
    }

    const double acc = global_total > 0
        ? static_cast<double>(global_correct) / static_cast<double>(global_total)
        : 0.0;
    if (is_master) {
        std::cout << "HellaSwag accuracy";
        if (step >= 0) std::cout << " (step " << step << ")";
        std::cout << ": " << global_correct << "/" << global_total
                  << " = " << std::fixed << std::setprecision(4) << acc
                  << std::endl;
    }
    return acc;
}

} // namespace hellaswag