#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "core/Shape.h"
#include "dtype/Dtype.h"

namespace OwnTensor {
namespace autograd {

// Flag: set true to attach shape/dtype annotations to tape entries.
// Lives in .bss — no heap allocation.
inline bool g_shape_debug = false;

// ---------------------------------------------------------------------------
// TapeEntry — one recorded operation with optional inline shape annotation.
// shape_info is empty when g_shape_debug is false.
// ---------------------------------------------------------------------------
struct TapeEntry {
    std::string name;
    std::string shape_info;  // "shape=[N,M] dtype=float32", or "" if unavailable
};

/// Records the forward and backward operation sequences of a computational graph.
//
// Stores two tapes:
// - forward_tape_:  operations in the order they were created during the forward pass.
// - backward_tape_: operations in the order they were executed during the backward pass.
//
// When no recorder is active (the default), zero memory is allocated.
// The only overhead is a single nullptr check per operation.

class GraphRecorder {
public:
    GraphRecorder() : out_(&std::cout) {}

    // ========================================================================
    // Output destination
    // ========================================================================

    // Redirect output to a file. Silent fail (no console message) if the file
    // cannot be opened, matching the AllocationTracker convention.
    void open_output_file(const std::string& path) {
        file_stream_.open(path, std::ios::out | std::ios::trunc);
        if (file_stream_.is_open())
            out_ = &file_stream_;
        // else: silently fall back to stdout
    }

    void close_output_file() {
        if (file_stream_.is_open()) {
            file_stream_.flush();
            file_stream_.close();
        }
        out_ = &std::cout;
    }

    // ========================================================================
    // Tape access
    // ========================================================================

    void add_forward(const std::string& name) {
        forward_tape_.push_back({name, ""});
    }

    // Attach shape_info to the most recently added forward entry.
    void set_last_forward_shape(std::string info) {
        if (!forward_tape_.empty())
            forward_tape_.back().shape_info = std::move(info);
    }

    void add_backward(const std::string& name, std::string info = "") {
        backward_tape_.push_back({name, std::move(info)});
    }

    const std::vector<TapeEntry>& forward_tape()  const { return forward_tape_;  }
    const std::vector<TapeEntry>& backward_tape() const { return backward_tape_; }

    // ========================================================================
    // Printing — writes to the configured output stream (file or stdout)
    // ========================================================================

    void print_forward_sequence() const {
        *out_ << "\n=== Forward Sequence (" << forward_tape_.size() << " ops) ===\n";
        for (size_t i = 0; i < forward_tape_.size(); ++i) {
            *out_ << "  [" << std::setw(3) << i << "] " << forward_tape_[i].name;
            if (!forward_tape_[i].shape_info.empty())
                *out_ << "  " << forward_tape_[i].shape_info;
            *out_ << '\n';
        }
        out_->flush();
    }

    void print_backward_sequence() const {
        *out_ << "\n=== Backward Sequence (" << backward_tape_.size() << " ops) ===\n";
        for (size_t i = 0; i < backward_tape_.size(); ++i) {
            *out_ << "  [" << std::setw(3) << i << "] " << backward_tape_[i].name;
            if (!backward_tape_[i].shape_info.empty())
                *out_ << "  " << backward_tape_[i].shape_info;
            *out_ << '\n';
        }
        out_->flush();
    }

    void print_all() const {
        print_forward_sequence();
        if (!backward_tape_.empty())
            print_backward_sequence();
    }

    void clear() {
        forward_tape_.clear();
        backward_tape_.clear();
    }

private:
    std::vector<TapeEntry> forward_tape_;
    std::vector<TapeEntry> backward_tape_;
    std::ostream*          out_;          // points to file_stream_ or std::cout
    std::ofstream          file_stream_;  // only open when writing to a file
};

// ---------------------------------------------------------------------------
// Global state for graph recording, following the GradMode pattern.
//
// When no recorder is active, the thread_local pointer is nullptr.
// All recording functions check this pointer first — if null, they return
// immediately with zero allocations.
// ---------------------------------------------------------------------------
class GraphRecordMode {
public:
    static GraphRecorder* get_active();
    static void set_active(GraphRecorder* recorder);

    // Redirect the active recorder's output to a file.
    // Must be called after a recorder is active (e.g. inside a GraphRecordGuard scope).
    static void set_output_file(const std::string& path) {
        GraphRecorder* rec = get_active();
        if (rec) rec->open_output_file(path);
    }

    // -----------------------------------------------------------------------
    // Build a "shape=[...] dtype=xxx" annotation string.
    // Only called when g_shape_debug is true and a recorder is active,
    // so the string allocation only happens on the recording hot-path.
    // -----------------------------------------------------------------------
    static std::string make_shape_info(const Shape& shape, Dtype dtype) {
        std::string s;
        s.reserve(32);
        s += "shape=[";
        for (size_t i = 0; i < shape.dims.size(); ++i) {
            if (i) s += ',';
            s += std::to_string(shape.dims[i]);
        }
        s += "] dtype=";
        switch (dtype) {
            case Dtype::Float32:        s += "float32";        break;
            case Dtype::Float64:        s += "float64";        break;
            case Dtype::Float16:        s += "float16";        break;
            case Dtype::Bfloat16:       s += "bfloat16";       break;
            case Dtype::Int8:           s += "int8";           break;
            case Dtype::Int16:          s += "int16";          break;
            case Dtype::Int32:          s += "int32";          break;
            case Dtype::Int64:          s += "int64";          break;
            case Dtype::UInt8:          s += "uint8";          break;
            case Dtype::UInt16:         s += "uint16";         break;
            case Dtype::UInt32:         s += "uint32";         break;
            case Dtype::UInt64:         s += "uint64";         break;
            case Dtype::Bool:           s += "bool";           break;
            case Dtype::Complex32:      s += "complex32";      break;
            case Dtype::Complex64:      s += "complex64";      break;
            case Dtype::Complex128:     s += "complex128";     break;
            case Dtype::Float4_e2m1:    s += "float4_e2m1";    break;
            case Dtype::Float4_e2m1_2x: s += "float4_e2m1_2x"; break;
            default:                    s += "unknown";        break;
        }
        return s;
    }

    // -----------------------------------------------------------------------
    // Forward recording
    // -----------------------------------------------------------------------

    static inline void record_forward(const std::string& name) {
        GraphRecorder* rec = get_active();
        if (rec) rec->add_forward(name);
    }

    // Attach output shape/dtype to the last recorded forward entry.
    // Gated by g_shape_debug at the call site to avoid any work when disabled.
    static inline void attach_forward_shape(const Shape& shape, Dtype dtype) {
        GraphRecorder* rec = get_active();
        if (rec) rec->set_last_forward_shape(make_shape_info(shape, dtype));
    }

    // -----------------------------------------------------------------------
    // Backward recording
    // -----------------------------------------------------------------------

    // shape_info must be built from input grads BEFORE they are moved into apply().
    static inline void record_backward(const std::string& name,
                                       std::string shape_info = "") {
        GraphRecorder* rec = get_active();
        if (rec) rec->add_backward(name, std::move(shape_info));
    }

private:
    static thread_local GraphRecorder* active_recorder_;
};


// ---------------------------------------------------------------------------
// GraphRecordGuard — RAII activation of a recorder.
//
// Usage (stdout):
//   GraphRecordGuard guard(true);
//
// Usage (file):
//   GraphRecordGuard guard(true, "graph_trace.txt");
//
// On destruction: flushes and prints (or writes) the tape, closes the file.
// ---------------------------------------------------------------------------
class GraphRecordGuard {
public:
    explicit GraphRecordGuard(bool auto_print = true,
                              const std::string& output_file = "")
        : prev_recorder_(GraphRecordMode::get_active()),
          auto_print_(auto_print)
    {
        if (!output_file.empty())
            recorder_.open_output_file(output_file);
        GraphRecordMode::set_active(&recorder_);
    }

    ~GraphRecordGuard() {
        if (auto_print_) {
            recorder_.print_all();
            recorder_.clear();
        }
        recorder_.close_output_file();
        GraphRecordMode::set_active(prev_recorder_);
    }

    GraphRecorder& recorder()             { return recorder_; }
    const GraphRecorder& recorder() const { return recorder_; }

    GraphRecordGuard(const GraphRecordGuard&)            = delete;
    GraphRecordGuard& operator=(const GraphRecordGuard&) = delete;

private:
    GraphRecorder* prev_recorder_;
    GraphRecorder  recorder_;
    bool           auto_print_;
};

} // namespace autograd
} // namespace OwnTensor
