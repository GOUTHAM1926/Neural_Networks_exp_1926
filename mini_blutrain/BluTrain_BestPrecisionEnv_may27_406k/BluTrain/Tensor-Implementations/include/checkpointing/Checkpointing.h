#pragma once

#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "core/Serialization.h"
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <regex>
#include <chrono>
#include <csignal>
#include "checkpointing/RNG.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
// Fix 1/3: CUDA stream API for D→H synchronization and dedicated checkpoint stream
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

namespace OwnTensor {

namespace fs = std::filesystem;

static constexpr int    kCheckpointVersion    = 1;
// Write-buffer size for both sync and async save paths.
// 16 MB coalesces per-tensor metadata writes (magic + dtype + rank + dims)
// into large sequential flushes, cutting system-call overhead.
// Tune upward (e.g. 32 MB) on NVMe/RAID, or downward on memory-constrained hosts.
static constexpr size_t kCheckpointWriteBufBytes = 16ULL * 1024 * 1024;

/**
 * @brief Represents a snapshot of the training state, residing on CPU.
 */
struct CheckpointState {
    int epoch;
    float loss;
    std::vector<Tensor> model_params;
    int64_t opt_step_count;
    std::vector<Tensor> opt_m;
    std::vector<Tensor> opt_v;
    RNGState rng_state;

    // Parameters for background cleanup
    std::string base_dir;
    std::string prefix;
    int max_to_keep = 0;
    
    /**
     * @brief Creates a pinned CPU clone of a tensor for fast staging.
     */
    static Tensor staged_clone(const Tensor& src) {
        if (!src.is_valid()) return Tensor();
        
        // If already on CPU, just clone normally
        if (src.device().is_cpu()) return src.to_cpu();
        
        // Create a pinned CPU tensor of the same shape/dtype
        TensorOptions opts;
        opts.device = Device::CPU;
        opts.dtype = src.dtype();
        opts.pinten = Pinned_Flag::Default;
        
        Tensor dst(src.shape(), opts);
        dst.copy_(src); // Synchronous but high-speed DMA transfer
        return dst;
    }

    static CheckpointState capture(::OwnTensor::nn::Module& model, ::OwnTensor::nn::Optimizer& optimizer, int epoch, float loss) {
        // std::cout << "[Capture] Starting state capture..." << std::endl;
        CheckpointState state;
        state.epoch = epoch;
        state.loss = loss;
        
        auto params = model.parameters();
        // std::cout << "[Capture] Model params count: " << params.size() << std::endl;
        state.model_params.reserve(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            state.model_params.push_back(staged_clone(params[i]));
        }

        // std::cout << "[Capture] Capturing optimizer state..." << std::endl;
        state.opt_step_count = optimizer.step_count();
        auto m = optimizer.get_m_buffers();
        auto v = optimizer.get_v_buffers();
        // std::cout << "[Capture] Optimizer buffers: m=" << m.size() << ", v=" << v.size() << std::endl;
        
        state.opt_m.reserve(m.size());
        state.opt_v.reserve(v.size());
        
        for (size_t i = 0; i < m.size(); ++i) {
            state.opt_m.push_back(staged_clone(m[i]));
        }
        for (size_t i = 0; i < v.size(); ++i) {
            state.opt_v.push_back(staged_clone(v[i]));
        }

#ifdef WITH_CUDA
        // Fix 1 — Correctness: staged_clone() issues cudaMemcpyAsync (truly async
        // despite the comment saying "synchronous"). Without this sync, the
        // background I/O thread reads pinned-CPU buffers that the DMA engine may
        // still be writing into — a silent data race.
        cudaStreamSynchronize(OwnTensor::cuda::getCurrentStream());
#endif

        try {
            // std::cout << "[Capture] Capturing RNG state..." << std::endl;
            state.rng_state = ::OwnTensor::RNG::get_state();
            // std::cout << "[Capture] RNG state captured." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CheckpointState::capture] Warning: Failed to capture RNG state: " << e.what() << std::endl;
        }

        // std::cout << "[Capture] Capture complete." << std::endl;
        return state;
    }
};

/**
 * @brief Background worker for asynchronous checkpointing.
 */
class AsyncCheckpointWorker {
public:
    AsyncCheckpointWorker() : stop_(false), busy_(false), has_error_(false) {
        worker_thread_ = std::thread(&AsyncCheckpointWorker::run, this);
    }

    ~AsyncCheckpointWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
    }

    void save_async(const std::string& path, CheckpointState&& state) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (busy_) {
                throw std::runtime_error("[AsyncCheckpointWorker] Cannot start new save: worker is still busy.");
            }
            pending_path_ = path;
            pending_state_ = std::move(state);
            busy_ = true;
        }
        cv_.notify_one();
    }

    bool is_busy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return busy_;
    }
    bool has_error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return has_error_;
    }
    std::string get_last_error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_error_;
    }
    void clear_error() {
        std::lock_guard<std::mutex> lock(mutex_);
        has_error_ = false;
        last_error_ = "";
    }
    
    void wait_for_completion() {
        std::unique_lock<std::mutex> lock(mutex_);
        completion_cv_.wait(lock, [this] { return !busy_; });
    }

private:
    void run() {
        while (true) {
            std::string path;
            CheckpointState state;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || busy_; });
                if (stop_ && !busy_) break;
                
                path = pending_path_;
                state = std::move(pending_state_);
            }
            
            process_save(path, state);
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                busy_ = false;
            }
            completion_cv_.notify_all();
        }
    }

    void process_save(const std::string& path, const CheckpointState& state) {
        std::string tmp_path = path + ".tmp";
        // Fix 4: 16 MB write buffer — coalesces per-tensor metadata writes into
        // large sequential chunks, reducing system-call overhead. Must be declared
        // before os so it outlives the stream (destroyed in reverse order).
        std::vector<char> write_buf(kCheckpointWriteBufBytes);
        std::ofstream os(tmp_path, std::ios::binary);
        if (!os.is_open()) {
            std::string err = "[AsyncCheckpointWorker] Failed to open: " + tmp_path;
            std::cerr << err << std::endl;
            record_error(err);
            return;
        }
        os.rdbuf()->pubsetbuf(write_buf.data(), write_buf.size());

        os.write("CKPT", 4);
        os.write(reinterpret_cast<const char*>(&kCheckpointVersion), sizeof(int));
        os.write(reinterpret_cast<const char*>(&state.epoch), sizeof(int));
        os.write(reinterpret_cast<const char*>(&state.loss), sizeof(float));

        int count = static_cast<int>(state.model_params.size());
        os.write(reinterpret_cast<const char*>(&count), sizeof(int));
        for (const auto& p : state.model_params) {
            if (p.is_valid()) {
                save_tensor(p, os);
            } else {
                os.write("TNS0", 4);
            }
        }

        os.write(reinterpret_cast<const char*>(&state.opt_step_count), sizeof(int64_t));
        int opt_count = static_cast<int>(state.opt_m.size());
        os.write(reinterpret_cast<const char*>(&opt_count), sizeof(int));
        for (const auto& t : state.opt_m) {
            if (t.is_valid()) {
                save_tensor(t, os);
            } else {
                os.write("TNS0", 4);
            }
        }
        for (const auto& t : state.opt_v) {
            if (t.is_valid()) {
                save_tensor(t, os);
            } else {
                os.write("TNS0", 4);
            }
        }

        uint32_t cpu_state_len = static_cast<uint32_t>(state.rng_state.cpu_state.size());
        os.write(reinterpret_cast<const char*>(&cpu_state_len), sizeof(uint32_t));
        os.write(state.rng_state.cpu_state.data(), cpu_state_len);
#ifdef WITH_CUDA
        os.write(reinterpret_cast<const char*>(&state.rng_state.gpu_seed), sizeof(unsigned long long));
        os.write(reinterpret_cast<const char*>(&state.rng_state.gpu_offset), sizeof(unsigned long long));
#endif
        os.flush();
        if (os.fail()) {
            std::string err = "[AsyncCheckpointWorker] Error occurred during flushing to: " + tmp_path + " (possibly disk full)";
            std::cerr << err << std::endl;
            os.close();
            if (fs::exists(tmp_path)) fs::remove(tmp_path);
            record_error(err);
            return;
        }

        os.close();
        if (os.fail()) {
            std::string err = "[AsyncCheckpointWorker] Error occurred during closing: " + tmp_path;
            std::cerr << err << std::endl;
            if (fs::exists(tmp_path)) fs::remove(tmp_path);
            record_error(err);
            return;
        }

        try {
            fs::rename(tmp_path, path);
        } catch (const fs::filesystem_error& e) {
            std::string err = "[AsyncCheckpointWorker] Rename failed: " + std::string(e.what());
            std::cerr << err << std::endl;
            if (fs::exists(tmp_path)) fs::remove(tmp_path);
            record_error(err);
        }

        // Cleanup after successful save
        if (state.max_to_keep > 0) {
            std::vector<std::string> checkpoints;
            for (const auto& entry : fs::directory_iterator(state.base_dir)) {
                std::string filename = entry.path().filename().string();
                if (filename.starts_with(state.prefix + "_step_") && filename.ends_with(".ckpt")) {
                    checkpoints.push_back(entry.path().string());
                }
            }
            std::sort(checkpoints.begin(), checkpoints.end());
            if (checkpoints.size() > (size_t)state.max_to_keep) {
                for (size_t i = 0; i < checkpoints.size() - state.max_to_keep; ++i) {
                    fs::remove(checkpoints[i]);
                    std::cout << "[AsyncCheckpointWorker] Deleted old checkpoint: " << checkpoints[i] << std::endl;
                }
            }
        }
    }

    void record_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = err;
        has_error_ = true;
    }

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable completion_cv_;
    std::thread worker_thread_;
    std::string pending_path_;
    CheckpointState pending_state_;
    std::string last_error_;
    bool stop_;
    bool busy_;
    bool has_error_;
};

/**
 * Saves a full training checkpoint.
 */
inline void save_checkpoint(const std::string& path,
                           nn::Module& model,
                           nn::Optimizer& optimizer,
                           int epoch,
                           float loss) {
    // Fix 4: 16 MB write buffer — must outlive os (declared first, destroyed last)
    std::vector<char> write_buf(kCheckpointWriteBufBytes);
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for writing: " + path);
    }
    os.rdbuf()->pubsetbuf(write_buf.data(), write_buf.size());

    // 1. Header
    os.write("CKPT", 4);
    os.write(reinterpret_cast<const char*>(&kCheckpointVersion), sizeof(int));

    // 2. Training state
    os.write(reinterpret_cast<const char*>(&epoch), sizeof(int));
    os.write(reinterpret_cast<const char*>(&loss), sizeof(float));

    // 3. Model state
    auto params = model.parameters();
    int count = static_cast<int>(params.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(int));

    // Fix 5: Stage GPU params into pinned CPU memory before writing.
    // save_tensor() calls to_cpu() internally which uses paged (non-pinned) memory,
    // forcing the CUDA driver to shadow-pin each page during DMA — ~10-25% overhead.
    // By pre-staging into pinned buffers and syncing once, we eliminate that overhead.
    {
        std::vector<Tensor> staged;
        staged.reserve(params.size());
        bool has_gpu = false;
        for (const auto& p : params) {
            if (p.is_valid() && !p.is_cpu()) {
                staged.push_back(CheckpointState::staged_clone(p));
                has_gpu = true;
            } else {
                staged.push_back(p);
            }
        }
#ifdef WITH_CUDA
        if (has_gpu) {
            // staged_clone() uses cudaMemcpyAsync — sync once for all transfers
            cudaStreamSynchronize(OwnTensor::cuda::getCurrentStream());
        }
#endif
        for (const auto& p : staged) {
            if (p.is_valid()) {
                save_tensor(p, os);
            } else {
                os.write("TNS0", 4);
            }
        }
    }

    // 4. Optimizer state
    optimizer.save_state(os);

    // 5. RNG state — serialize std::string cpu_state by length-prefixed bytes
    RNGState rng_state = RNG::get_state();
    uint32_t cpu_state_len = static_cast<uint32_t>(rng_state.cpu_state.size());
    os.write(reinterpret_cast<const char*>(&cpu_state_len), sizeof(uint32_t));
    os.write(rng_state.cpu_state.data(), cpu_state_len);
#ifdef WITH_CUDA
    os.write(reinterpret_cast<const char*>(&rng_state.gpu_seed), sizeof(unsigned long long));
    os.write(reinterpret_cast<const char*>(&rng_state.gpu_offset), sizeof(unsigned long long));
#endif

    os.flush();
    if (os.fail()) {
        os.close();
        throw std::runtime_error("Failed to flush checkpoint to: " + path);
    }
    os.close();
    if (os.fail()) {
        throw std::runtime_error("Failed to close checkpoint file: " + path);
    }
}

/**
 * Loads a full training checkpoint.
 */
inline void load_checkpoint(const std::string& path, 
                           nn::Module& model, 
                           nn::Optimizer& optimizer, 
                           int& epoch, 
                           float& loss) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open checkpoint file for reading: " + path);
    }

    // 1. Header
    char magic[4];
    is.read(magic, 4);
    if (std::string(magic, 4) != "CKPT") {
        throw std::runtime_error("Invalid checkpoint format: " + path);
    }
    int version;
    is.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != kCheckpointVersion) {
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(version) + 
                                 " (expected " + std::to_string(kCheckpointVersion) + ")");
    }

    // 2. Training state
    is.read(reinterpret_cast<char*>(&epoch), sizeof(int));
    is.read(reinterpret_cast<char*>(&loss), sizeof(float));

    // 3. Model state
    auto params = model.parameters();
    int count;
    is.read(reinterpret_cast<char*>(&count), sizeof(int));
    if (count != static_cast<int>(params.size())) {
        throw std::runtime_error("Checkpoint model parameter count mismatch");
    }
    for (auto& p : params) {
        Tensor loaded = load_tensor(is);
        p.copy_(loaded);
    }

    // 4. Optimizer state
    optimizer.load_state(is);

    // 5. RNG state — deserialize length-prefixed cpu_state string
    RNGState rng_state;
    uint32_t cpu_state_len = 0;
    is.read(reinterpret_cast<char*>(&cpu_state_len), sizeof(uint32_t));
    rng_state.cpu_state.resize(cpu_state_len);
    is.read(rng_state.cpu_state.data(), cpu_state_len);
#ifdef WITH_CUDA
    is.read(reinterpret_cast<char*>(&rng_state.gpu_seed), sizeof(unsigned long long));
    is.read(reinterpret_cast<char*>(&rng_state.gpu_offset), sizeof(unsigned long long));
#endif
    RNG::set_state(rng_state);
    
    is.close();
}

/**
 * @brief Manages model checkpoints with directory support, auto-discovery,
 * and periodic saving (heartbeats).
 */
class CheckpointManager {
public:
    CheckpointManager(std::string base_dir, std::string prefix = "model", int max_to_keep = 5, int rank = 0, bool shard_dir = false, bool use_async = false)
        : base_dir_(base_dir), prefix_(prefix), max_to_keep_(max_to_keep), rank_(rank), use_async_(use_async) {

        // If sharding is requested, append rank subdirectory
        if (shard_dir && rank_ >= 0) {
            base_dir_ = (fs::path(base_dir_) / ("rank_" + std::to_string(rank_))).string();
        }

        if (!fs::exists(base_dir_)) {
            fs::create_directories(base_dir_);
        }
        last_save_time_ = std::chrono::steady_clock::now();

#ifdef WITH_CUDA
        // Fix 3: Dedicated non-blocking checkpoint stream so D→H transfers run
        // on a separate DMA channel, fully concurrent with training-stream compute.
        cudaStreamCreateWithFlags(&checkpoint_stream_, cudaStreamNonBlocking);
#endif
    }

    ~CheckpointManager() {
        // Wait for any in-flight async save before destroying staging buffers
        if (async_worker_) {
            async_worker_->wait_for_completion();
        }
#ifdef WITH_CUDA
        if (checkpoint_stream_) {
            cudaStreamDestroy(checkpoint_stream_);
            checkpoint_stream_ = nullptr;
        }
#endif
    }

    /**
     * @brief Static flag set by signal handlers.
     */
    static inline volatile sig_atomic_t stop_requested = 0;

    /**
     * @brief Signal handler for Ctrl+C and termination.
     */
    static void signal_handler(int sig) {
        stop_requested = 1;
    }

    /**
     * @brief Registers the static signal handler.
     */
    void register_signal_handler() {
        std::signal(SIGINT, CheckpointManager::signal_handler);
        std::signal(SIGTERM, CheckpointManager::signal_handler);
    }

    /**
     * @brief Set intervals for automatic saving.
     * @param step_interval Save every N steps (e.g., 100). -1 to disable.
     * @param seconds_interval Save every N seconds (e.g., 1800 for 30min). -1 to disable.
     */
    void set_save_intervals(int step_interval, int seconds_interval = -1) {
        step_interval_ = step_interval;
        seconds_interval_ = seconds_interval;
    }

    /**
     * @brief Smart step function. Saves if either step or time interval is reached.
     * @return true if a checkpoint was saved.
     */
    bool step(int current_step, nn::Module& model, nn::Optimizer& optimizer, float loss) {
        bool should_save = false;

        // Check step-based interval
        if (step_interval_ > 0 && current_step > 0 && current_step % step_interval_ == 0) {
            should_save = true;
        }

        // Check time-based interval (Heartbeat)
        if (seconds_interval_ > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_save_time_).count();
            if (elapsed >= seconds_interval_) {
                std::cout << "[CheckpointManager] Time-based heartbeat triggered after " << elapsed << "s" << std::endl;
                should_save = true;
            }
        }

        if (should_save) {
            save(current_step, model, optimizer, loss);
            return true;
        }
        return false;
    }

    void check_async_errors() {
        if (async_worker_ && async_worker_->has_error()) {
            std::string err = async_worker_->get_last_error();
            async_worker_->clear_error();
            throw std::runtime_error("Async checkpoint failed: " + err);
        }
    }

    void wait_for_completion() {
        if (async_worker_) {
            async_worker_->wait_for_completion();
        }
    }

    /**
     * @brief Manually save a checkpoint for a specific step.
     */
    void save(int step, nn::Module& model, nn::Optimizer& optimizer, float loss) {
        check_async_errors();
        fs::path p = fs::path(base_dir_) / (prefix_ + "_step_" + std::to_string(step) + ".ckpt");
        
        if (use_async_) {
            if (!async_worker_) async_worker_ = std::make_unique<AsyncCheckpointWorker>();

            // Wait if last save is still running — staging buffers are shared with
            // the background worker and must not be overwritten until it is done.
            if (async_worker_->is_busy()) {
                std::cout << "[CheckpointManager] Waiting for previous async save to finish..." << std::endl;
                async_worker_->wait_for_completion();
            }

            // Fix 2+3: Use optimized capture that reuses pre-allocated pinned staging
            // buffers and issues D→H transfers on the dedicated checkpoint stream.
            CheckpointState state = capture_with_staging(model, optimizer, step, loss);
            state.base_dir = base_dir_;
            state.prefix = prefix_;
            state.max_to_keep = max_to_keep_;
            async_worker_->save_async(p.string(), std::move(state));
            
            std::cout << "[CheckpointManager] Async save started for: " << p.string() << std::endl;
        } else {
            fs::path p_tmp = p;
            p_tmp.replace_extension(".tmp");
            
            std::string path_str = p.string();
            std::string tmp_path_str = p_tmp.string();
            
            save_checkpoint(tmp_path_str, model, optimizer, step, loss);
            fs::rename(p_tmp, p);
            
            std::cout << "[CheckpointManager] Saved: " << path_str << " (safely)" << std::endl;
        }
        
        last_save_time_ = std::chrono::steady_clock::now();
        if (!use_async_) {
            cleanup_old_checkpoints();
        }
    }

    /**
     * @brief Automatically find and load the latest checkpoint in base_dir.
     * @return true if a checkpoint was found and loaded.
     */
    bool load_latest(nn::Module& model, nn::Optimizer& optimizer, int& step, float& loss) {
        std::string latest_path = find_latest_checkpoint_path();
        if (latest_path.empty()) {
            return false;
        }

        std::cout << "[CheckpointManager] Auto-loading latest: " << latest_path << std::endl;
        load_checkpoint(latest_path, model, optimizer, step, loss);
        return true;
    }

private:
    /**
     * @brief Fix 2+3: Optimized capture that reuses pre-allocated pinned staging
     * buffers and issues all D→H transfers on the dedicated checkpoint stream.
     *
     * On the first call: allocates pinned CPU tensors once (cudaHostAlloc × N).
     * On subsequent calls: overwrites the same buffers (zero allocation cost).
     * All GPU→CPU copies go on checkpoint_stream_ (non-blocking wrt training).
     * One cudaStreamSynchronize at the end is the only blocking point.
     */
    CheckpointState capture_with_staging(nn::Module& model, nn::Optimizer& optimizer, int epoch, float loss) {
        CheckpointState state;
        state.epoch = epoch;
        state.loss = loss;
        state.opt_step_count = optimizer.step_count();

        auto params = model.parameters();
        auto m = optimizer.get_m_buffers();
        auto v = optimizer.get_v_buffers();

        // Fix 2: One-time allocation of pinned staging tensors
        if (!staging_initialized_) {
            auto alloc_staging = [](const std::vector<Tensor>& src) {
                std::vector<Tensor> out;
                out.reserve(src.size());
                for (const auto& t : src) {
                    if (!t.is_valid()) {
                        out.push_back(Tensor());
                        continue;
                    }
                    TensorOptions opts;
                    opts.device = Device::CPU;
                    opts.dtype = t.dtype();
                    // Pinned only for GPU tensors (DMA requires page-locked memory)
                    opts.pinten = t.is_cpu() ? Pinned_Flag::None : Pinned_Flag::Default;
                    out.emplace_back(t.shape(), opts);
                }
                return out;
            };
            staging_params_ = alloc_staging(params);
            staging_m_      = alloc_staging(m);
            staging_v_      = alloc_staging(v);
            staging_initialized_ = true;
        }

        // Fix 3: Issue all D→H copies on the dedicated checkpoint stream.
        // cudaStreamNonBlocking means this stream never synchronizes with the
        // default (training) stream — both run concurrently on separate DMA engines.
        auto issue_copies = [&](const std::vector<Tensor>& src, std::vector<Tensor>& dst) {
            for (size_t i = 0; i < src.size(); ++i) {
                if (!src[i].is_valid()) continue;
#ifdef WITH_CUDA
                if (!src[i].is_cpu()) {
                    cudaMemcpyAsync(dst[i].data(), src[i].data(),
                                   src[i].nbytes(),
                                   cudaMemcpyDeviceToHost,
                                   checkpoint_stream_);
                } else
#endif
                {
                    dst[i].copy_(src[i]);  // CPU→CPU: plain memcpy
                }
            }
        };

        issue_copies(params, staging_params_);
        issue_copies(m,      staging_m_);
        issue_copies(v,      staging_v_);

#ifdef WITH_CUDA
        // Fix 1+3: Single sync point — all DMA transfers on checkpoint_stream_ are
        // guaranteed complete here. Training continues on the main stream in parallel.
        cudaStreamSynchronize(checkpoint_stream_);
#endif

        // Shallow tensor copies: staging_[i] and state.*[i] share the same
        // pinned storage. Safe because we wait_for_completion() above before
        // every new capture, ensuring the worker is done reading before we
        // overwrite the buffers on the next checkpoint.
        state.model_params = staging_params_;
        state.opt_m        = staging_m_;
        state.opt_v        = staging_v_;

        try {
            state.rng_state = ::OwnTensor::RNG::get_state();
        } catch (const std::exception& e) {
            std::cerr << "[capture_with_staging] Warning: Failed to capture RNG state: " << e.what() << std::endl;
        }

        return state;
    }

    std::string find_latest_checkpoint_path() {
        std::vector<std::pair<int, std::string>> checkpoints = list_checkpoints();
        if (checkpoints.empty()) return "";

        // Sort by step number ascending
        std::sort(checkpoints.begin(), checkpoints.end());
        return checkpoints.back().second;
    }

    void cleanup_old_checkpoints() {
        if (max_to_keep_ <= 0) return;

        std::vector<std::pair<int, std::string>> checkpoints = list_checkpoints();
        if (checkpoints.size() <= (size_t)max_to_keep_) return;

        // Sort by step number ascending (oldest first)
        std::sort(checkpoints.begin(), checkpoints.end());

        int to_delete = checkpoints.size() - max_to_keep_;
        for (int i = 0; i < to_delete; ++i) {
            fs::remove(checkpoints[i].second);
            std::cout << "[CheckpointManager] Deleted old checkpoint: " << checkpoints[i].second << std::endl;
        }
    }

    std::vector<std::pair<int, std::string>> list_checkpoints() {
        std::vector<std::pair<int, std::string>> results;
        std::regex re(prefix_ + "_step_(\\d+)\\.ckpt");

        if (!fs::exists(base_dir_)) return {};

        for (const auto& entry : fs::directory_iterator(base_dir_)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, re)) {
                int step = std::stoi(match[1].str());
                results.push_back({step, entry.path().string()});
            }
        }
        return results;
    }

    std::string base_dir_;
    std::string prefix_;
    int max_to_keep_;
    int rank_;
    int step_interval_ = -1;
    int seconds_interval_ = -1;
    std::chrono::steady_clock::time_point last_save_time_;
    bool use_async_;
    std::unique_ptr<AsyncCheckpointWorker> async_worker_;

    // Fix 2: Pre-allocated pinned staging buffers — reused across checkpoints to
    // avoid per-checkpoint cudaHostAlloc (which takes a global CUDA driver lock).
    std::vector<Tensor> staging_params_;
    std::vector<Tensor> staging_m_;
    std::vector<Tensor> staging_v_;
    bool staging_initialized_ = false;

    // Fix 3: Dedicated non-blocking CUDA stream for checkpoint D→H transfers.
    // Transfers on this stream run concurrently with training-stream compute kernels.
#ifdef WITH_CUDA
    cudaStream_t checkpoint_stream_ = nullptr;
#endif
};

} // namespace OwnTensor