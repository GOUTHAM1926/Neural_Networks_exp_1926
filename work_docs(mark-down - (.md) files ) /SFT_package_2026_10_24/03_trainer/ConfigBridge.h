#pragma once
// YAML -> environment for train/production_zero/bluscript_zero_ckpt.cpp: the trainer and libdist
// read env vars, so the YAML is exported as those vars before anything reads them. Precedence:
// an already-set env var > YAML > built-in default; unknown keys are errors.

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace zcfg {

enum class Kind { Bool, Int, Float, String };

struct EnvKey {
    const char* section;
    const char* key;
    const char* env;
    Kind        kind;
    bool        inverted = false;
    const char* choices  = nullptr;
};

inline const std::vector<EnvKey>& env_table() {
    static const std::vector<EnvKey> t = {
        {"training", "step_watchdog",        "BLU_STEP_WATCHDOG",      Kind::Bool},
        {"training", "step_timeout_s",       "BLU_STEP_TIMEOUT_S",     Kind::Int},

        {"data", "data_dir",                 "BLU_MIX_ROOT",           Kind::String},
        {"data", "mix_preset",               "BLU_MIX_PRESET",         Kind::String, false, "c0=c0,c1=c1,c2=c2"},

        {"optimizer", "type",                "BLU_OPTIMIZER",          Kind::String, false, "adamW=adamw,adamw=adamw,muon=muon"},

        {"checkpointing", "enabled",         "BLU_CKPT",               Kind::Bool},
        {"checkpointing", "freq",            "BLU_CKPT_FREQ",          Kind::Int},
        {"checkpointing", "ckpt_keep",       "BLU_CKPT_KEEP",          Kind::Int},
        {"checkpointing", "load",            "BLU_CKPT_LOAD",          Kind::Bool},
        {"checkpointing", "dir",             "BLU_CKPT_DIR",           Kind::String},

        {"logs", "enabled",                  "BLU_LOG_ENABLED",        Kind::Bool},
        {"logs", "log_dir",                  "BLU_LOG_DIR",            Kind::String},
        {"logs", "log_file",                 "BLU_LOG_CSV",            Kind::String},
        {"logs", "overwrite",                "BLU_LOG_OVERWRITE",      Kind::Bool},
        {"logs", "append",                   "BLU_LOG_APPEND",         Kind::Bool},

        {"run", "name",                      "BLU_RUN_NAME",           Kind::String},

        {"monitoring", "enabled",            "BLU_MONITOR",            Kind::Bool},
        {"monitoring", "logs_root",          "BLU_LOGS_ROOT",          Kind::String},
        {"monitoring", "port_range",         "BLU_MONITOR_PORTS",      Kind::String},
        {"monitoring", "targets_dir",        "BLU_MONITOR_TARGETS_DIR", Kind::String},
        {"monitoring", "grafana_url",        "BLU_GRAFANA_URL",        Kind::String},

        {"sizeclass", "mode",                "BLU_SIZE_CLASS",         Kind::String, false, "default=default,exact=exact"},

        {"embedding", "backward",            "BLU_EMB_BWD",            Kind::String, false, "atomic=atomic,cooperative=cooperative,segmented=segmented"},

        {"mixed_precision", "enabled",       "BLU_ALL_BF16",           Kind::Bool},
        {"mixed_precision", "precision",     nullptr,                  Kind::String, false, "bf16="},
        {"mixed_precision", "bf16_logits",   "BLU_BF16_LOGITS",        Kind::Bool},

        {"parallelism", "enabled",           "BLU_PARALLELISM",        Kind::Bool},
        {"parallelism", "config",            nullptr,                  Kind::String},
    };
    return t;
}

inline const std::vector<EnvKey>& parallel_table() {
    static const std::vector<EnvKey> t = {
        {"", "type",                         nullptr,                  Kind::String, false, "zero1="},
        {"", "backend",                      "BLU_BACKEND",            Kind::String, false, "nccl=nccl,bluccl=bluccl"},

        {"zero1", "variant",                 "BLU_ZERO_V2",            Kind::String, false, "v1=0,v2=1"},
        {"zero1", "symmetric",               "BLU_SYMMETRIC",          Kind::Bool},
        {"zero1", "global_clip",             "BLU_GLOBAL_CLIP",        Kind::Bool},
        {"zero1", "grad_view",               "BLU_GRAD_VIEW",          Kind::Bool},
        {"zero1", "fp32_reduce",             "BLU_FP32_REDUCE",        Kind::Bool},
        {"zero1", "bf16_reduce",             "BLU_BF16_REDUCE",        Kind::Bool},
        {"zero1", "no_overlap",              "BLU_NO_OVERLAP",         Kind::Bool},
        {"zero1", "no_overlap_bwd",          "BLU_NO_OVERLAP_BWD",     Kind::Bool},
        {"zero1", "overlap_refresh",         "BLU_OVERLAP_REFRESH",    Kind::Bool},
        {"zero1", "global_clip_strict",      "BLU_GLOBAL_CLIP_STRICT", Kind::Bool},
        {"zero1", "no_inplace",              "BLU_ZERO_INPLACE",       Kind::Bool,   true},
        {"zero1", "muon_reduce_broadcast",   "BLU_MUON_RB",            Kind::Bool},
        {"zero1", "no_muon_pipeline",        "BLU_MUON_PIPELINE",      Kind::Bool,   true},
        {"zero1", "bucket_mb",               "BLU_COMM_BUCKET_MB",     Kind::Int},
        {"zero1", "muon_buckets",            "BLU_MUON_BUCKETS",       Kind::Int},
        {"zero1", "muon_adjust_lr",          "BLU_MUON_ADJUST_LR",     Kind::String, false, "match_rms_adamw=match_rms_adamw,original=original"},
        {"zero1", "ckpt_align",              "BLU_ZERO_CKPT_ALIGN",    Kind::Int},

        {"bluccl", "fused_embedding_backward", "BLU_BLUCCL_FUSED_EMB", Kind::Bool},
        {"bluccl", "fused_adamw_allgather",  "BLU_FUSED_ADAMW",        Kind::Bool},
        {"bluccl", "fused_clip_adamw",       "BLU_FUSED_CLIP_ADAMW",   Kind::Bool},
        {"bluccl", "fused_muon_publish",     "BLU_FUSED_MUON_PUBLISH", Kind::Bool},
        // multi-node (backend bluccl over more than one node)
        {"bluccl", "internode_reduce_scatter", "BLU_IB_RS",            Kind::Bool},
        {"bluccl", "rs_flat_order",          "BLU_IB_RS_FLAT",         Kind::Bool},
        {"bluccl", "rs_tail",                "BLU_RS_TAIL",            Kind::Int},
        {"bluccl", "rs_overlap_blocks",      "BLU_IB_RS_OVL_BLOCKS",   Kind::Int},
        {"bluccl", "ag_overlap",             "BLU_IB_AG_OVERLAP",      Kind::Bool},
        {"bluccl", "symmetric_ib",           "BLU_SYMM_IB",            Kind::Bool},
        {"bluccl", "reduce_scatter_via",     "BLU_COMPOSITE_RS",       Kind::String, false, "bluccl=bluccl,nccl=nccl"},
        {"bluccl", "all_gather_via",         "BLU_COMPOSITE_AG",       Kind::String, false, "bluccl=bluccl,nccl=nccl"},
        // collective profiler and the reduce-scatter schedule autotuner (tools/rs_autotune.py)
        {"bluccl", "comm_profile",           "BLU_COMM_PROFILE",       Kind::Bool},
        {"bluccl", "comm_profile_dir",       "BLU_COMM_PROFILE_DIR",   Kind::String},
        {"bluccl", "rs_schedule",            "BLU_RS_SCHEDULE",        Kind::String},
    };
    return t;
}

inline const std::set<std::pair<std::string, std::string>>& field_keys() {
    static const std::set<std::pair<std::string, std::string>> s = {
        {"hyper-parameters", "context"}, {"hyper-parameters", "vocab"},
        {"hyper-parameters", "n_embd"}, {"hyper-parameters", "n_layers"},
        {"hyper-parameters", "n_heads"}, {"hyper-parameters", "kv_heads"},
        {"hyper-parameters", "head_dim"}, {"hyper-parameters", "ffn_hidden"},
        {"hyper-parameters", "rope_theta"}, {"hyper-parameters", "qk_eps"},
        {"hyper-parameters", "rms_eps"}, {"hyper-parameters", "rope_interleaved"},
        {"hyper-parameters", "weight_tying"}, {"hyper-parameters", "global_batch"},
        {"hyper-parameters", "mini_batch"}, {"hyper-parameters", "seed"},
        {"training", "total_steps"}, {"training", "warmup_steps"}, {"training", "stable_steps"},
        {"training", "validation_freq"}, {"training", "val_loss_steps"}, {"training", "grad_clip"},
        {"optimizer", "max_lr"}, {"optimizer", "min_lr"}, {"optimizer", "beta1"},
        {"optimizer", "beta2"}, {"optimizer", "eps"}, {"optimizer", "weight_decay"},
        {"optimizer", "muon_lr"}, {"optimizer", "muon_min_lr"}, {"optimizer", "muon_momentum"},
        {"optimizer", "muon_weight_decay"}, {"optimizer", "muon_ns_steps"},
        {"checkpointing", "prefix"},
    };
    return s;
}

using FieldPtr = std::variant<int64_t*, int*, float*, bool*, std::string*>;
struct Field {
    const char* section;
    const char* key;
    FieldPtr    ptr;
};

struct Report {
    std::string path;
    std::string parallel_path;
    std::vector<std::string> applied;
    std::vector<std::string> kept_env;
    std::vector<std::string> fields;
};

inline std::string to_env_value(const YAML::Node& v, const EnvKey& k) {
    if (k.choices) {
        const std::string in = v.as<std::string>();
        std::string list = k.choices;
        size_t pos = 0;
        while (pos <= list.size()) {
            const size_t comma = list.find(',', pos);
            const std::string item = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            const size_t eq = item.find('=');
            if (item.substr(0, eq) == in) return item.substr(eq + 1);
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        std::string allowed;
        for (size_t a = 0; a <= list.size();) {
            const size_t c = list.find(',', a), e = list.find('=', a);
            allowed += (allowed.empty() ? "" : ", ") + list.substr(a, e - a);
            if (c == std::string::npos) break;
            a = c + 1;
        }
        throw std::runtime_error("expected one of: " + allowed);
    }
    switch (k.kind) {
        case Kind::Bool:   return (v.as<bool>() != k.inverted) ? "1" : "0";
        case Kind::Int:    return std::to_string(v.as<long long>());
        case Kind::Float:  { const std::string s = v.as<std::string>(); (void)std::stod(s); return s; }
        case Kind::String: return v.as<std::string>();
    }
    return {};
}

inline void set_env(const std::string& name, const std::string& value,
                    const std::string& origin, Report& rep) {
    const char* cur = std::getenv(name.c_str());
    if (cur != nullptr) {
        rep.kept_env.push_back(name + "=" + cur + " (env wins over " + origin + ")");
        return;
    }
    ::setenv(name.c_str(), value.c_str(), 1);
    rep.applied.push_back(name + "=" + value + " (" + origin + ")");
}

inline bool take_flag(int* argc, char*** argv, const char* flag) {
    bool found = false;
    int out = 1;
    for (int i = 1; i < *argc; ++i) {
        if (std::strcmp((*argv)[i], flag) == 0) { found = true; continue; }
        (*argv)[out++] = (*argv)[i];
    }
    *argc = out;
    return found;
}

inline std::string config_path(int* argc, char*** argv) {
    std::string path;
    int out = 1;
    for (int i = 1; i < *argc; ++i) {
        const char* a = (*argv)[i];
        if (std::strncmp(a, "--config=", 9) == 0) { path = a + 9; continue; }
        (*argv)[out++] = (*argv)[i];
    }
    *argc = out;
    if (path.empty()) {
        const char* e = std::getenv("BLU_CONFIG");
        if (e && *e) path = e;
    }
    return path;
}

inline void export_map(const YAML::Node& root, const std::string& file,
                       const std::vector<EnvKey>& table, bool allow_fields, Report& rep) {
    if (!root.IsMap()) throw std::runtime_error("config " + file + ": top level must be a map");
    std::map<std::pair<std::string, std::string>, const EnvKey*> by_key;
    std::set<std::string> sections;
    for (const auto& k : table) {
        by_key[{k.section, k.key}] = &k;
        if (*k.section) sections.insert(k.section);
    }
    if (allow_fields)
        for (const auto& f : field_keys()) sections.insert(f.first);

    auto one = [&](const std::string& sname, const std::string& key, const YAML::Node& v) {
        const std::string where = sname.empty() ? key : sname + "." + key;
        auto it = by_key.find({sname, key});
        const bool is_field = allow_fields && field_keys().count({sname, key}) != 0;
        if (it == by_key.end() && !is_field)
            throw std::runtime_error("config " + file + ": unknown key '" + where + "'");
        if (it == by_key.end() || v.IsNull()) return;
        std::string val;
        try {
            val = to_env_value(v, *it->second);
        } catch (const std::exception& e) {
            throw std::runtime_error("config " + file + ": bad value for '" + where + "': " + e.what());
        }
        if (it->second->env) set_env(it->second->env, val, where + "=" + v.as<std::string>(), rep);
    };

    for (const auto& sec : root) {
        const std::string sname = sec.first.as<std::string>();
        if (!sec.second.IsMap() && !sec.second.IsNull()) {
            one("", sname, sec.second);
            continue;
        }
        if (!sections.count(sname))
            throw std::runtime_error("config " + file + ": unknown section '" + sname + "'");
        if (sec.second.IsNull()) continue;
        for (const auto& kv : sec.second) one(sname, kv.first.as<std::string>(), kv.second);
    }
}

inline YAML::Node load_and_export(int* argc, char*** argv, Report& rep) {
    rep.path = config_path(argc, argv);
    if (rep.path.empty()) return YAML::Node();

    YAML::Node root = YAML::LoadFile(rep.path);
    export_map(root, rep.path, env_table(), true, rep);

    const YAML::Node par = root["parallelism"];
    const bool parallel_on = par && par["enabled"] && !par["enabled"].IsNull() && par["enabled"].as<bool>();
    if (parallel_on) {
        if (!par["config"] || par["config"].IsNull())
            throw std::runtime_error("config " + rep.path + ": parallelism.enabled is true but parallelism.config is not set");
        std::string p = par["config"].as<std::string>();
        if (!p.empty() && p[0] != '/') {
            const auto slash = rep.path.find_last_of('/');
            if (slash != std::string::npos) p = rep.path.substr(0, slash + 1) + p;
        }
        rep.parallel_path = p;
        export_map(YAML::LoadFile(p), p, parallel_table(), false, rep);
    }
    return root;
}

inline void bind_fields(const YAML::Node& root, const std::vector<Field>& fields, Report& rep) {
    if (!root) return;
    for (const auto& f : fields) {
        if (!field_keys().count({f.section, f.key}))
            throw std::logic_error(std::string("ConfigBridge: field ") + f.section + "." + f.key +
                                   " is bound but missing from field_keys()");
        const YAML::Node sec = root[f.section];
        if (!sec || !sec.IsMap()) continue;
        const YAML::Node v = sec[f.key];
        if (!v || v.IsNull()) continue;
        const std::string where = std::string(f.section) + "." + f.key;
        try {
            std::visit([&](auto* p) {
                using T = std::remove_pointer_t<decltype(p)>;
                *p = v.as<T>();
            }, f.ptr);
        } catch (const std::exception&) {
            throw std::runtime_error("config " + rep.path + ": bad value for '" + where + "'");
        }
        const std::string line = where + " = " + v.as<std::string>();
        if (rep.fields.empty() || rep.fields.back() != line) rep.fields.push_back(line);
    }
}

}  // namespace zcfg
