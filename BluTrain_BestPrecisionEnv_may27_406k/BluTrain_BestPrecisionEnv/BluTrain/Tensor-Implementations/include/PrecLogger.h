#pragma once
// Header-only precision/profiling logger. Dumps per-module forward I/O and
// per-parameter values + gradients to .bin files for cross-framework precision
// diffing against PyTorch. See BluTrain/tools/compare_dumps.py.
//
// Activated only when PREC_DUMP_DIR is set. Env knobs (all optional):
//   PREC_DUMP_DIR=dumps/cpp              root output dir (required to enable)
//   PREC_DUMP_STEPS=0,500,1000           explicit step list (union with EVERY)
//   PREC_DUMP_EVERY=500                  every-N steps (union with STEPS)
//   PREC_DUMP_KINDS=fwd_in,fwd_out,param,grad   default: all four
//   PREC_DUMP_PRINT=1                    also print per-tensor summary to cout

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_set>
#include <vector>

#include "TensorLib.h"

namespace prec {

inline void mkdir_p(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty()) ::mkdir(cur.c_str(), 0755);
            if (i < path.size()) cur.push_back('/');
        } else {
            cur.push_back(path[i]);
        }
    }
}

class Logger {
public:
    static Logger& instance() { static Logger L; return L; }

    // Call once per training step BEFORE any dumps for that step.
    void begin_step(int step) {
        cur_step_ = step;
        active_step_ = enabled_ && step_selected(step);
        if (!active_step_) return;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "step_%06d", step);
        step_dir_ = root_ + "/" + buf;
        mkdir_p(step_dir_);
        mkdir_p(step_dir_ + "/params");
        mkdir_p(step_dir_ + "/grads");
        summary_.open(step_dir_ + "/summary.csv");
        summary_ << "name,kind,dtype,shape,numel,mean,min,max,l2,inf\n";
        summary_ << std::scientific << std::setprecision(9);
    }

    void end_step() {
        if (active_step_) {
            summary_.flush();
            summary_.close();
        }
        active_step_ = false;
    }

    bool active() const { return active_step_; }
    bool active_for(const std::string& kind) const {
        return active_step_ && (kinds_.empty() || kinds_.count(kind));
    }

    // Module forward I/O — `kind` is "fwd_in" or "fwd_out".
    void dump_tensor(const std::string& name,
                     const std::string& kind,
                     const OwnTensor::Tensor& t) {
        if (!active_for(kind)) return;
        write_(name, kind, t, /*subdir=*/"");
    }

    // Parameter values (one per dump-step). subdir=params/
    void dump_param(const std::string& name, const OwnTensor::Tensor& t) {
        if (!active_for("param")) return;
        write_(name, "param", t, "params/");
    }

    // Parameter gradients (one per dump-step). subdir=grads/
    void dump_grad(const std::string& name, const OwnTensor::Tensor& t) {
        if (!active_for("grad")) return;
        write_(name, "grad", t, "grads/");
    }

private:
    Logger() {
        const char* d = std::getenv("PREC_DUMP_DIR");
        if (!d || !d[0]) { enabled_ = false; return; }
        root_ = d;
        mkdir_p(root_);
        enabled_ = true;

        if (const char* s = std::getenv("PREC_DUMP_STEPS")) parse_csv_ints_(s, steps_);
        if (const char* e = std::getenv("PREC_DUMP_EVERY")) every_ = std::atoi(e);
        if (const char* k = std::getenv("PREC_DUMP_KINDS")) {
            parse_csv_strs_(k, kinds_);
        } else {
            // Safe default: dump only params + grads. Forward activations are
            // huge (val loop alone writes tens of GB at default config), so
            // they must be opted into via PREC_DUMP_KINDS explicitly.
            kinds_.insert("param");
            kinds_.insert("grad");
        }
        if (const char* p = std::getenv("PREC_DUMP_PRINT")) print_ = (p[0] == '1');

        std::cout << "[PrecLogger] enabled root=" << root_
                  << " steps={"; for (int s : steps_) std::cout << s << ","; std::cout << "}"
                  << " every=" << every_
                  << " kinds={"; for (auto& k : kinds_) std::cout << k << ","; std::cout << "}"
                  << " print=" << print_ << std::endl;
    }

    bool step_selected(int step) const {
        if (!enabled_) return false;
        if (steps_.count(step)) return true;
        if (every_ > 0 && (step % every_) == 0) return true;
        return false;
    }

    static void parse_csv_ints_(const char* s, std::set<int>& out) {
        std::string cur;
        for (const char* p = s; ; ++p) {
            if (*p == ',' || *p == '\0') {
                if (!cur.empty()) out.insert(std::atoi(cur.c_str()));
                cur.clear();
                if (*p == '\0') break;
            } else cur.push_back(*p);
        }
    }
    static void parse_csv_strs_(const char* s, std::unordered_set<std::string>& out) {
        std::string cur;
        for (const char* p = s; ; ++p) {
            if (*p == ',' || *p == '\0') {
                if (!cur.empty()) out.insert(cur);
                cur.clear();
                if (*p == '\0') break;
            } else cur.push_back(*p);
        }
    }

    // Materialize the tensor on CPU (Float32 only — only path that's
    // meaningful for cross-framework precision comparison), write its raw
    // bytes to <subdir><name>.<kind>.bin, write a sidecar JSON with
    // shape/dtype, and append a row to summary.csv with mean/min/max/l2/inf.
    void write_(const std::string& name,
                const std::string& kind,
                const OwnTensor::Tensor& t,
                const std::string& subdir) {
        if (!t.is_valid()) return;
        if (t.dtype() != OwnTensor::Dtype::Float32) return;  // skip int idx tensors etc.

        OwnTensor::Tensor cpu = t.to_cpu();
        const float* data = cpu.data<float>();
        size_t n = cpu.numel();

        std::string file_stem = step_dir_ + "/" + subdir + name + "." + kind;
        std::ofstream bin(file_stem + ".bin", std::ios::binary);
        bin.write(reinterpret_cast<const char*>(data), n * sizeof(float));
        bin.close();

        // Use 'x' (not ',') so the shape stays a single CSV field — e.g.
        // "50304x384" instead of "[50304,384]". The sidecar JSON still
        // carries the real shape as an array.
        std::string shape_str;
        for (size_t i = 0; i < cpu.shape().dims.size(); ++i) {
            shape_str += std::to_string(cpu.shape().dims[i]);
            if (i + 1 < cpu.shape().dims.size()) shape_str += "x";
        }

        std::string shape_json = "[";
        for (size_t i = 0; i < cpu.shape().dims.size(); ++i) {
            shape_json += std::to_string(cpu.shape().dims[i]);
            if (i + 1 < cpu.shape().dims.size()) shape_json += ",";
        }
        shape_json += "]";

        std::ofstream js(file_stem + ".json");
        js << "{\"name\":\"" << name << "\",\"kind\":\"" << kind
           << "\",\"dtype\":\"float32\",\"shape\":" << shape_json
           << ",\"numel\":" << n << "}\n";
        js.close();

        double sum = 0.0, sumsq = 0.0;
        float mn = n ? data[0] : 0.0f, mx = n ? data[0] : 0.0f, inf = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float v = data[i];
            sum += v; sumsq += (double)v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            float a = v < 0 ? -v : v;
            if (a > inf) inf = a;
        }
        double mean = n ? sum / (double)n : 0.0;
        double l2 = std::sqrt(sumsq);

        summary_ << name << "," << kind << ",float32," << shape_str << ","
                 << n << "," << mean << "," << mn << "," << mx << ","
                 << l2 << "," << inf << "\n";

        if (print_) {
            std::cout << "[PrecLogger] step=" << cur_step_
                      << " " << name << "." << kind
                      << " shape=" << shape_str
                      << " mean=" << mean << " min=" << mn << " max=" << mx
                      << " l2=" << l2 << " inf=" << inf << std::endl;
        }
    }

    bool enabled_ = false;
    bool active_step_ = false;
    bool print_ = false;
    int cur_step_ = -1;
    int every_ = 0;
    std::string root_;
    std::string step_dir_;
    std::set<int> steps_;
    std::unordered_set<std::string> kinds_;
    std::ofstream summary_;
};

}  // namespace prec

// Convenience macros — no-op on non-dump steps thanks to the `active()` gate.
#define PREC_DUMP(name, kind, t) ::prec::Logger::instance().dump_tensor((name), (kind), (t))
#define PREC_DUMP_IO(name, in, out) do { \
    PREC_DUMP((name), "fwd_in", (in)); \
    PREC_DUMP((name), "fwd_out", (out)); \
} while (0)
