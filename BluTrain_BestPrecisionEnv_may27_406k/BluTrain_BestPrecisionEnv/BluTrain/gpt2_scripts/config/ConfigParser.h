#pragma once
// =============================================================================
// ConfigParser.h — Simple INI-style config parser for BluTrain
//
// Format:
//   # comment
//   [section]
//   key = value
//
// Supported value types: string, int, float, bool, string list (comma-separated)
// =============================================================================

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <iostream>

namespace config {

// Trim whitespace from both ends
inline std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// ---- Parallelism types ----
enum class ParallelismType {
    NONE,   // Single GPU
    DDP,    // Data Parallel
    TP,     // Tensor Parallel
};

// ---- Attention types ----
enum class AttentionType {
    NORMAL,              // Standard scaled dot-product
    FMHA,                // Memory-efficient (flash-style)
    FUSED_TRIL_SOFTMAX,  // Fused causal tril + softmax
    FLASH,               // True Flash Attention (placeholder)
};

// ---- Optimizer types ----
enum class OptimizerType {
    ADAMW,
    ADAM,
    SGD,
};

// ---- BLAS backend ----
enum class BlasBackend {
    CUBLAS,
    BLU_BLAS,
};

// =============================================================================
// TrainConfig: all training configuration in one struct
// =============================================================================
struct TrainConfig {
    // --- Model ---
    // context_length == sequence_length (token_length drives both).
    // Set via [training] token_length in the config file.
    int64_t vocab_size     = 50304;
    int64_t n_embd         = 384;
    int64_t n_layers       = 3;
    int64_t n_heads        = 6;
    bool    weight_tying   = false;

    // --- Training hyperparameters ---
    int     batch_size        = 8;
    int     token_length      = 1024;   // sequence length T AND model context length
    int     global_batch_size = 65536;
    float   max_lr            = 0.0028f;
    float   min_lr            = 0.00028f;
    int     warmup_steps      = 10;
    int     max_steps         = 1000;
    float   grad_clip_norm    = 1.0f;
    uint64_t seed             = 1337;   // weight init seed (same across all modes)

    // Derived: context_length = token_length (used by model constructors)
    int64_t context_length() const { return static_cast<int64_t>(token_length); }

    // --- Optimizer ---
    OptimizerType optimizer   = OptimizerType::ADAMW;
    float   beta1             = 0.9f;
    float   beta2             = 0.95f;
    float   eps               = 1e-8f;
    float   weight_decay      = 0.1f;
    float   sgd_momentum      = 0.9f;

    // --- Parallelism ---
    // Stored as a vector so you can combine: [DDP, TP] = 2D parallel
    std::vector<ParallelismType> parallelism = {};  // empty = single GPU
    int     tp_size = 2;  // tensor parallel degree (used when TP is in the list)

    // --- Attention ---
    AttentionType attention = AttentionType::FMHA;

    // --- Logging ---
    bool        logging_enabled  = true;
    bool        log_csv          = true;
    std::string log_csv_path     = "Training_logs/training.csv";
    int         val_freq         = 500;
    int         val_steps        = 20;

    // --- Checkpointing ---
    bool        activation_checkpointing = false;
    bool        model_checkpointing      = false;
    int         checkpoint_freq          = 5000;
    std::string checkpoint_dir           = "checkpoints";
    int         max_checkpoints          = 5;

    // --- Target loss tracking ---
    float   target_train_loss = 4.00f;
    float   target_val_loss   = 4.00f;

    // --- BLAS ---
    BlasBackend blas_backend = BlasBackend::CUBLAS;

    // --- Data ---
    std::string data_root = "/home/blu-bridge015/Desktop/edufineweb";
    int64_t     max_seq_len_data = 100000000;

    // --- Device ---
    int gpu_device = 0;

    // ---- Derived helpers ----

    bool has_parallelism(ParallelismType p) const {
        for (auto& pt : parallelism) {
            if (pt == p) return true;
        }
        return false;
    }

    bool is_single_gpu() const { return parallelism.empty(); }
    bool has_ddp() const { return has_parallelism(ParallelismType::DDP); }
    bool has_tp() const { return has_parallelism(ParallelismType::TP); }
    bool is_2d_parallel() const { return has_ddp() && has_tp(); }

    // grad_accum steps for a given number of data-parallel replicas
    int grad_accum_steps(int dp_replicas) const {
        int steps = global_batch_size / (batch_size * token_length * dp_replicas);
        return steps < 1 ? 1 : steps;
    }

    // Effective grad scale: B*T / GLOBAL_BATCH (same for all modes)
    float effective_grad_scale() const {
        return static_cast<float>(batch_size * token_length) /
               static_cast<float>(global_batch_size);
    }

    // Print config summary
    void print(int rank = 0, int dp_replicas = 1) const {
        if (rank != 0) return;
        std::cout << "=== BluTrain Configuration ===" << std::endl;
        std::cout << "  [Model]" << std::endl;
        std::cout << "    vocab_size:     " << vocab_size << std::endl;
        std::cout << "    token_length:   " << token_length << std::endl;
        std::cout << "    n_embd:         " << n_embd << std::endl;
        std::cout << "    n_heads:        " << n_heads << std::endl;
        std::cout << "    n_layers:       " << n_layers << std::endl;
        std::cout << "    head_dim:       " << (n_embd / n_heads) << std::endl;
        std::cout << "    weight_tying:   " << (weight_tying ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "  [Training]" << std::endl;
        std::cout << "    batch_size:        " << batch_size << std::endl;
        std::cout << "    global_batch_size: " << global_batch_size << std::endl;
        std::cout << "    grad_accum_steps:  " << grad_accum_steps(dp_replicas)
                  << "  (dp_replicas=" << dp_replicas << ")" << std::endl;
        std::cout << "    eff_grad_scale:    " << effective_grad_scale() << std::endl;
        std::cout << "    seed:              " << seed << std::endl;
        std::cout << "    max_lr:            " << max_lr << std::endl;
        std::cout << "    min_lr:            " << min_lr << std::endl;
        std::cout << "    warmup_steps:      " << warmup_steps << std::endl;
        std::cout << "    max_steps:         " << max_steps << std::endl;
        std::cout << "    grad_clip_norm:    " << grad_clip_norm << std::endl;
        std::cout << "  [Optimizer]" << std::endl;
        std::cout << "    type: ";
        switch (optimizer) {
            case OptimizerType::ADAMW: std::cout << "AdamW"; break;
            case OptimizerType::ADAM:  std::cout << "Adam"; break;
            case OptimizerType::SGD:   std::cout << "SGD"; break;
        }
        std::cout << std::endl;
        std::cout << "  [Parallelism]" << std::endl;
        if (parallelism.empty()) {
            std::cout << "    mode: single_gpu" << std::endl;
        } else {
            std::cout << "    mode: [";
            for (size_t i = 0; i < parallelism.size(); ++i) {
                if (i > 0) std::cout << ", ";
                switch (parallelism[i]) {
                    case ParallelismType::DDP: std::cout << "ddp"; break;
                    case ParallelismType::TP:  std::cout << "tp"; break;
                    default: std::cout << "none"; break;
                }
            }
            std::cout << "]" << std::endl;
            if (has_tp()) std::cout << "    tp_size: " << tp_size << std::endl;
        }
        std::cout << "  [Attention]" << std::endl;
        std::cout << "    type: ";
        switch (attention) {
            case AttentionType::NORMAL:             std::cout << "normal"; break;
            case AttentionType::FMHA:               std::cout << "fmha"; break;
            case AttentionType::FUSED_TRIL_SOFTMAX: std::cout << "fused_tril_softmax"; break;
            case AttentionType::FLASH:              std::cout << "flash (placeholder)"; break;
        }
        std::cout << std::endl;
        std::cout << "  [Logging]" << std::endl;
        std::cout << "    enabled:  " << (logging_enabled ? "true" : "false") << std::endl;
        std::cout << "    csv:      " << (log_csv ? "true" : "false") << std::endl;
        std::cout << "    csv_path: " << log_csv_path << std::endl;
        std::cout << "  [Checkpointing]" << std::endl;
        std::cout << "    activation_checkpointing: " << (activation_checkpointing ? "true" : "false") << std::endl;
        std::cout << "    model_checkpointing:      " << (model_checkpointing ? "true" : "false") << std::endl;
        std::cout << "    checkpoint_freq:          " << checkpoint_freq << std::endl;
        std::cout << "    checkpoint_dir:           " << checkpoint_dir << std::endl;
        std::cout << "  [BLAS]" << std::endl;
        std::cout << "    backend: " << (blas_backend == BlasBackend::CUBLAS ? "CUBLAS" : "BLU_BLAS") << std::endl;
        std::cout << "  [Data]" << std::endl;
        std::cout << "    data_root: " << data_root << std::endl;
        std::cout << "===============================" << std::endl;
    }
};

// =============================================================================
// ConfigParser: reads a .cfg file and populates TrainConfig
// =============================================================================
class ConfigParser {
public:
    static TrainConfig parse(const std::string& filepath) {
        TrainConfig cfg;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + filepath);
        }

        std::unordered_map<std::string, std::string> kv;
        std::string line;
        std::string current_section;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            // Section header
            if (line[0] == '[' && line.back() == ']') {
                current_section = to_lower(trim(line.substr(1, line.size() - 2)));
                continue;
            }

            // Key = value
            auto eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = to_lower(trim(line.substr(0, eq_pos)));
            std::string val = trim(line.substr(eq_pos + 1));

            // Strip inline comments
            auto comment_pos = val.find('#');
            if (comment_pos != std::string::npos) {
                val = trim(val.substr(0, comment_pos));
            }

            // Prefix with section for namespacing
            std::string full_key = current_section.empty() ? key : current_section + "." + key;
            kv[full_key] = val;
        }

        // --- Populate config from key-value map ---

        // Model
        set_if(kv, "model.vocab_size",   cfg.vocab_size);
        set_if(kv, "model.n_embd",       cfg.n_embd);
        set_if(kv, "model.n_layers",     cfg.n_layers);
        set_if(kv, "model.n_heads",      cfg.n_heads);
        set_if(kv, "model.weight_tying", cfg.weight_tying);

        // Training
        set_if(kv, "training.batch_size",        cfg.batch_size);
        set_if(kv, "training.token_length",       cfg.token_length);
        set_if(kv, "training.global_batch_size",  cfg.global_batch_size);
        set_if(kv, "training.max_lr",             cfg.max_lr);
        set_if(kv, "training.min_lr",             cfg.min_lr);
        set_if(kv, "training.warmup_steps",       cfg.warmup_steps);
        set_if(kv, "training.max_steps",          cfg.max_steps);
        set_if(kv, "training.grad_clip_norm",     cfg.grad_clip_norm);
        set_if(kv, "training.seed",               cfg.seed);

        // Optimizer
        if (kv.count("optimizer.type")) {
            std::string opt = to_lower(kv["optimizer.type"]);
            if (opt == "adamw")     cfg.optimizer = OptimizerType::ADAMW;
            else if (opt == "adam") cfg.optimizer = OptimizerType::ADAM;
            else if (opt == "sgd")  cfg.optimizer = OptimizerType::SGD;
            else throw std::runtime_error("Unknown optimizer: " + opt);
        }
        set_if(kv, "optimizer.beta1",        cfg.beta1);
        set_if(kv, "optimizer.beta2",        cfg.beta2);
        set_if(kv, "optimizer.eps",          cfg.eps);
        set_if(kv, "optimizer.weight_decay", cfg.weight_decay);
        set_if(kv, "optimizer.momentum",     cfg.sgd_momentum);

        // Parallelism
        if (kv.count("parallelism.mode")) {
            cfg.parallelism = parse_parallelism(kv["parallelism.mode"]);
        }
        set_if(kv, "parallelism.tp_size", cfg.tp_size);

        // Attention
        if (kv.count("attention.type")) {
            std::string att = to_lower(kv["attention.type"]);
            if (att == "normal")                  cfg.attention = AttentionType::NORMAL;
            else if (att == "fmha")               cfg.attention = AttentionType::FMHA;
            else if (att == "fused_tril_softmax")  cfg.attention = AttentionType::FUSED_TRIL_SOFTMAX;
            else if (att == "flash")               cfg.attention = AttentionType::FLASH;
            else throw std::runtime_error("Unknown attention type: " + att);
        }

        // Logging
        set_if(kv, "logging.enabled",   cfg.logging_enabled);
        set_if(kv, "logging.csv",       cfg.log_csv);
        set_if_str(kv, "logging.csv_path", cfg.log_csv_path);
        set_if(kv, "logging.val_freq",  cfg.val_freq);
        set_if(kv, "logging.val_steps", cfg.val_steps);

        // Checkpointing
        set_if(kv, "checkpointing.activation", cfg.activation_checkpointing);
        set_if(kv, "checkpointing.model",      cfg.model_checkpointing);
        set_if(kv, "checkpointing.frequency",  cfg.checkpoint_freq);
        set_if_str(kv, "checkpointing.directory", cfg.checkpoint_dir);
        set_if(kv, "checkpointing.max_kept",   cfg.max_checkpoints);

        // Target loss
        set_if(kv, "training.target_train_loss", cfg.target_train_loss);
        set_if(kv, "training.target_val_loss",   cfg.target_val_loss);

        // BLAS
        if (kv.count("blas.use_cublas")) {
            bool use_cublas = parse_bool(kv["blas.use_cublas"]);
            cfg.blas_backend = use_cublas ? BlasBackend::CUBLAS : BlasBackend::BLU_BLAS;
        }

        // Data
        set_if_str(kv, "data.root",        cfg.data_root);
        set_if(kv, "data.max_seq_len",     cfg.max_seq_len_data);

        // Device
        set_if(kv, "device.gpu", cfg.gpu_device);

        return cfg;
    }

private:
    // Parse comma-separated parallelism list: "ddp, tp" -> [DDP, TP]
    static std::vector<ParallelismType> parse_parallelism(const std::string& val) {
        std::vector<ParallelismType> result;
        std::string lower_val = to_lower(val);

        if (lower_val == "none" || lower_val == "single" || lower_val.empty()) {
            return result;
        }

        std::stringstream ss(lower_val);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (token == "ddp")      result.push_back(ParallelismType::DDP);
            else if (token == "tp")  result.push_back(ParallelismType::TP);
            else if (token == "none") { /* skip */ }
            else throw std::runtime_error("Unknown parallelism type: " + token);
        }
        return result;
    }

    static bool parse_bool(const std::string& val) {
        std::string v = to_lower(val);
        return (v == "true" || v == "1" || v == "yes");
    }

    // Set helpers
    static void set_if(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, int& out) {
        if (kv.count(key)) out = std::stoi(kv.at(key));
    }
    static void set_if(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, int64_t& out) {
        if (kv.count(key)) out = std::stoll(kv.at(key));
    }
    static void set_if(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, uint64_t& out) {
        if (kv.count(key)) out = static_cast<uint64_t>(std::stoull(kv.at(key)));
    }
    static void set_if(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, float& out) {
        if (kv.count(key)) out = std::stof(kv.at(key));
    }
    static void set_if(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, bool& out) {
        if (kv.count(key)) out = parse_bool(kv.at(key));
    }
    static void set_if_str(const std::unordered_map<std::string, std::string>& kv,
                            const std::string& key, std::string& out) {
        if (kv.count(key)) out = kv.at(key);
    }
};

}  // namespace config
