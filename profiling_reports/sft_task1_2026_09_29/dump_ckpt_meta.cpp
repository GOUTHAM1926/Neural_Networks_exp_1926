// dump_ckpt_meta.cpp -- print what a BluTrain DCP checkpoint contains, using the library's own reader
// (CheckpointMetadata::deserialize in dist/Distributed_Checkpointing/include/Metadata.h), not a re-implementation.
//
// Usage: dump_ckpt_meta <checkpoint dir | metadata.bin> [data-file sizes to cross-check, e.g. rank0=6471...]
//
// Prints:
//   1. whether the reader consumed the file to the last byte (proves the on-disk format matches this reader),
//   2. the header: version, epoch, step, loss, optimizer config, RNG ranks,
//   3. every model weight (the "param_NNNNNN" entries): index, dtype, shape, element count, replicated or sharded,
//   4. every other entry, grouped by name prefix (optimizer state and extras),
//   5. the bytes each data file should hold, from the shard records (to compare with the real file sizes).
#include "Metadata.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using OwnTensor::CheckpointMetadata;
using OwnTensor::Dtype;

static const char* dtype_name(Dtype d) {
    switch (d) {
        case Dtype::Int8: return "int8";       case Dtype::Int16: return "int16";
        case Dtype::Int32: return "int32";     case Dtype::Int64: return "int64";
        case Dtype::UInt8: return "uint8";     case Dtype::UInt16: return "uint16";
        case Dtype::UInt32: return "uint32";   case Dtype::UInt64: return "uint64";
        case Dtype::Bfloat16: return "bf16";   case Dtype::Float16: return "fp16";
        case Dtype::Float32: return "fp32";    case Dtype::Float64: return "fp64";
        case Dtype::Bool: return "bool";       case Dtype::Complex32: return "c32";
        case Dtype::Complex64: return "c64";   case Dtype::Complex128: return "c128";
        default: return "other";
    }
}
static uint64_t dtype_bytes(Dtype d) {
    switch (d) {
        case Dtype::Int8: case Dtype::UInt8: case Dtype::Bool: return 1;
        case Dtype::Int16: case Dtype::UInt16: case Dtype::Bfloat16: case Dtype::Float16: return 2;
        case Dtype::Int32: case Dtype::UInt32: case Dtype::Float32: case Dtype::Complex32: return 4;
        case Dtype::Int64: case Dtype::UInt64: case Dtype::Float64: case Dtype::Complex64: return 8;
        case Dtype::Complex128: return 16;
        default: return 0;
    }
}
static int64_t numel(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t v : s) n *= v;
    return n;
}
static std::string shape_str(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? " x " : "") + std::to_string(s[i]);
    return r + "]";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dump_ckpt_meta <checkpoint dir | metadata.bin>\n";
        return 2;
    }
    std::filesystem::path p(argv[1]);
    if (std::filesystem::is_directory(p)) p /= "metadata.bin";
    std::ifstream is(p, std::ios::binary);
    if (!is) {
        std::cerr << "cannot open " << p << "\n";
        return 1;
    }
    CheckpointMetadata m = CheckpointMetadata::deserialize(is);
    const std::streampos stop = is.tellg();
    is.seekg(0, std::ios::end);
    const std::streampos end = is.tellg();

    std::cout << "=== 1. format check\n"
              << "file " << p << ": " << static_cast<long long>(end) << " bytes; reader stopped at byte "
              << static_cast<long long>(stop)
              << (stop == end ? "  -> OK, the whole file was read\n" : "  -> MISMATCH, format differs\n");

    std::cout << "\n=== 2. header\n"
              << "version " << m.version << " | epoch " << m.epoch << " | step " << m.step << " | loss "
              << std::setprecision(7) << m.loss << " | RNG states for " << m.rank_rng_states.size()
              << " rank(s)\n"
              << "optimizer_config (" << m.optimizer_config.size() << " entries):\n";
    for (const auto& [k, v] : m.optimizer_config) std::cout << "  " << k << " = " << v << "\n";

    // 3. model weights, in index order (std::map already sorts "param_000000" < "param_000001" < ...)
    std::cout << "\n=== 3. model weights (param_*)\n";
    std::map<std::string, std::pair<int64_t, int64_t>> by_dtype;  // dtype -> (count, elements)
    int64_t n_params = 0, total_elems = 0;
    for (const auto& [name, t] : m.tensors) {
        if (name.rfind("param_", 0) != 0) continue;
        const int64_t n = numel(t.global_shape);
        std::cout << "  " << name << "  " << std::setw(5) << dtype_name(t.dtype) << "  " << std::setw(20)
                  << shape_str(t.global_shape) << "  " << std::setw(12) << n << "  "
                  << (t.is_replicated ? "replicated" : "sharded") << ", " << t.shards.size() << " shard(s)\n";
        auto& b = by_dtype[dtype_name(t.dtype)];
        b.first += 1;
        b.second += n;
        ++n_params;
        total_elems += n;
    }
    std::cout << "  -> " << n_params << " weight tensors, " << total_elems << " elements in total\n";
    for (const auto& [d, b] : by_dtype)
        std::cout << "     " << d << ": " << b.first << " tensors, " << b.second << " elements\n";

    // 4. everything else, grouped by the name up to the first '.' or '/'
    std::cout << "\n=== 4. other entries (optimizer state and extras), grouped by prefix\n";
    struct Group { int64_t count = 0, elems = 0; std::string example; std::map<std::string, int> dtypes; };
    std::map<std::string, Group> groups;
    for (const auto& [name, t] : m.tensors) {
        if (name.rfind("param_", 0) == 0) continue;
        const size_t cut = name.find_first_of("./");
        auto& g = groups[cut == std::string::npos ? name : name.substr(0, cut)];
        g.count += 1;
        g.elems += numel(t.global_shape);
        g.dtypes[dtype_name(t.dtype)] += 1;
        if (g.example.empty()) g.example = name + " " + shape_str(t.global_shape);
    }
    for (const auto& [k, g] : groups) {
        std::cout << "  " << k << ": " << g.count << " entries, " << g.elems << " elements, dtypes {";
        for (const auto& [d, c] : g.dtypes) std::cout << " " << d << ":" << c;
        std::cout << " }  e.g. " << g.example << "\n";
    }

    // 5. bytes per data file, from the shard records
    std::cout << "\n=== 5. bytes each data file should hold (from the shard records)\n";
    std::map<std::string, uint64_t> file_bytes;
    for (const auto& [name, t] : m.tensors)
        for (const auto& s : t.shards)
            file_bytes[s.file_path] += s.data_length ? s.data_length
                                                     : static_cast<uint64_t>(numel(s.shape)) * dtype_bytes(t.dtype);
    for (const auto& [f, b] : file_bytes)
        std::cout << "  " << f << ": " << b << " bytes (" << std::fixed << std::setprecision(2) << b / 1e9
                  << " GB)\n";
    return stop == end ? 0 : 3;
}
