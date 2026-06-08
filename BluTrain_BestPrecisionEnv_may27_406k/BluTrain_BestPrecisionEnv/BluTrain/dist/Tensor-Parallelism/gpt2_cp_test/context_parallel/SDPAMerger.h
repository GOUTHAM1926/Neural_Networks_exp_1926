#pragma once

#include "core/Tensor.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ExponentsOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/TensorOps.h"
#include <stdexcept>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// SDPAMerger
//
// Online softmax merger for context parallel ring attention.
//
// When attention is split across multiple chunks of K,V (across ranks),
// each chunk produces a partial attention output and its log-sum-exp (LSE).
// The merger combines these partial results into the correct global output
// using the numerically stable online softmax correction formula:
//
//   out = out - sigmoid(block_lse - lse) * (out - block_out)
//   lse = lse - log(sigmoid(lse - block_lse))
//
// This is equivalent to the standard "log-sum-exp trick" for combining
// softmax outputs computed over different slices of the key dimension.
//
// Reference: FlashAttention-2, Ring Attention (Liu et al. 2023)
// ---------------------------------------------------------------------------
class SDPAMerger {
public:
    SDPAMerger(bool convert_to_f32 = true)
        : convert_to_f32_(convert_to_f32),
          out_(),
          lse_(),
          out_dtype_(Dtype::Float32),
          lse_dtype_(Dtype::Float32),
          initialized_(false) {}

    // -----------------------------------------------------------------------
    // step
    //
    // Incorporate a new partial attention result (block_out, block_lse)
    // into the running accumulator.
    //
    // Parameters:
    //   block_out: partial attention output [B, H, T_q, D]
    //   block_lse: partial log-sum-exp      [B, H, T_q, 1]
    // -----------------------------------------------------------------------
    void step(Tensor block_out, Tensor block_lse) {
        // Save original dtypes for final cast
        out_dtype_ = block_out.dtype();
        lse_dtype_ = block_lse.dtype();

        // Optionally convert to float32 for numerical stability
        if (convert_to_f32_ && block_out.dtype() != Dtype::Float32) {
            block_out = block_out.as_type(Dtype::Float32);
        }
        if (convert_to_f32_ && block_lse.dtype() != Dtype::Float32) {
            block_lse = block_lse.as_type(Dtype::Float32);
        }

        if (!initialized_) {
            // First chunk: just store directly
            out_ = block_out;
            lse_ = block_lse;
            initialized_ = true;
            return;
        }

;
        //
        // Using raw tensor ops (not autograd) since the merger is a numerical
        // correction that does not need to be differentiated through directly.
        // The autograd graph flows through the SDPA ops themselves.

        // sigmoid(block_lse - lse)
        Tensor lse_diff = block_lse - lse_;
        Tensor sig = autograd::sigmoid(lse_diff);

        // out = out - sig * (out - block_out)
        Tensor out_diff = out_ - block_out;
        Tensor correction = sig * out_diff;
        out_ = out_ - correction;

        // lse = lse - log(sigmoid(lse - block_lse))
        // Note: sigmoid(lse - block_lse) = sigmoid(-lse_diff) = 1 - sigmoid(lse_diff)
        // log(1 - sigmoid(lse_diff)) is more stable than log(sigmoid(-lse_diff))
        // But for functional correctness, direct computation works:
        Tensor neg_lse_diff = lse_ - block_lse;
        Tensor sig_neg = autograd::sigmoid(neg_lse_diff);
        Tensor log_sig = autograd::log(sig_neg);
        lse_ = lse_ - log_sig;
    }

    // -----------------------------------------------------------------------
    // results
    //
    // Returns the final merged (out, lse) pair, cast back to original dtypes.
    // -----------------------------------------------------------------------
    std::pair<Tensor, Tensor> results() const {
        if (!initialized_) {
            throw std::runtime_error("SDPAMerger::results: no steps were taken");
        }

        Tensor final_out = out_;
        Tensor final_lse = lse_;

        // Cast back to original dtypes if needed
        if (final_out.dtype() != out_dtype_) {
            final_out = final_out.as_type(out_dtype_);
        }
        if (final_lse.dtype() != lse_dtype_) {
            final_lse = final_lse.as_type(lse_dtype_);
        }

        return {final_out, final_lse};
    }

    // -----------------------------------------------------------------------
    // reset
    //
    // Reset the merger state for reuse in a new forward/backward pass.
    // -----------------------------------------------------------------------
    void reset() {
        out_ = Tensor();
        lse_ = Tensor();
        initialized_ = false;
    }

private:
    bool convert_to_f32_;
    Tensor out_;
    Tensor lse_;
    Dtype out_dtype_;
    Dtype lse_dtype_;
    bool initialized_;
};
