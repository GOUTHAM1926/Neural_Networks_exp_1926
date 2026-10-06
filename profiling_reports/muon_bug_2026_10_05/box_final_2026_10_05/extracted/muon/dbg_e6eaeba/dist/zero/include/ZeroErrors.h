#pragma once
//======================================================================
// ZeroErrors.h — typed exceptions for the ZeRO sharded optimizer.
//
// Every failure mode the optimizer can hit gets a distinct type so callers
// (and tests) can react precisely instead of parsing strings. All derive from
// std::runtime_error so a generic catch still works.
//======================================================================

#include <stdexcept>
#include <string>

namespace owt {
namespace zero {

// Base for every ZeRO error.
struct ZeroError : std::runtime_error {
    explicit ZeroError(const std::string& what) : std::runtime_error("[ZeRO] " + what) {}
};

// Ranks disagree on something that must be identical (param count / order /
// shapes / hyperparameters). Raised at construction (F2).
struct ZeroConsistencyError : ZeroError {
    explicit ZeroConsistencyError(const std::string& what)
        : ZeroError("consistency: " + what) {}
};

// A collective did not complete within the configured timeout, or the process
// group reported a NCCL/CUDA transport failure (F1).
struct ZeroCollectiveError : ZeroError {
    explicit ZeroCollectiveError(const std::string& what)
        : ZeroError("collective: " + what) {}
};

// F1 (new error type).
// The gradient norm came out NaN/Inf and no LossScaler is installed to skip the
// step. Raised by step() BEFORE any weight/moment update, so the master weights
// and any later checkpoint stay clean; the caller is expected to stop. Every
// rank sees the same all-reduced norm, so every rank raises it together.
struct ZeroNonFiniteGradError : ZeroError {
    explicit ZeroNonFiniteGradError(const std::string& what)
        : ZeroError("non-finite gradient: " + what) {}
};

// The optimizer was configured or used in a way that is not supported
// (e.g. an unsupported optimizer variant, mixed param dtypes, etc.).
struct ZeroConfigError : ZeroError {
    explicit ZeroConfigError(const std::string& what)
        : ZeroError("config: " + what) {}
};

// A checkpoint could not be loaded because its layout descriptor does not match
// the current run and no resharding path is available.
struct ZeroCheckpointError : ZeroError {
    explicit ZeroCheckpointError(const std::string& what)
        : ZeroError("checkpoint: " + what) {}
};

} // namespace zero
} // namespace owt
