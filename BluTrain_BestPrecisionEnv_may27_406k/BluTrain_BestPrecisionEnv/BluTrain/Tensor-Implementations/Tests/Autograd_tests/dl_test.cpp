#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "TensorLib.h"

namespace fs = std::filesystem;

// static int getenv_int(const char* key, int def) {
//     const char* v = std::getenv(key);
//     return v ? std::atoi(v) : def;
// }

static std::vector<std::string> list_shards(const std::string& root,
                                            const std::string& split,
                                            const std::string& ext = ".bin") {
    std::vector<std::string> shards;
    if (!fs::exists(root)) return shards;
    for (const auto& e : fs::directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        auto p = e.path();
        std::string name = p.filename().string();
        if (p.extension() == ext && name.find(split) != std::string::npos) {
            shards.push_back(p.string());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

class UInt16ShardView {
public:
    UInt16ShardView() = default;
    ~UInt16ShardView() { close(); }

    void open(const std::string& path, size_t max_tokens) {
        close();
        path_ = path;

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open: " + path);

        struct stat st {};
        if (fstat(fd_, &st) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }

        file_bytes_ = static_cast<size_t>(st.st_size);
        if (file_bytes_ % sizeof(uint16_t) != 0) {
            ::close(fd_); fd_ = -1;
            throw std::runtime_error("file size not divisible by 2 (uint16): " + path);
        }

        size_t total_tokens = file_bytes_ / 2;
        tokens_ = std::min(total_tokens, max_tokens);

        data_ = ::mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            ::close(fd_); fd_ = -1; data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    void close() {
        if (data_) { ::munmap(data_, file_bytes_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        file_bytes_ = 0; tokens_ = 0; path_.clear();
    }

    size_t size_tokens() const { return tokens_; }
    const std::string& path() const { return path_; }

    void read_block(size_t start, size_t count, std::vector<uint16_t>& out) const {
        out.resize(count);
        const uint16_t* p = reinterpret_cast<const uint16_t*>(data_);
        for (size_t i = 0; i < count; ++i) {
            // Wrap around the available tokens using modulo
            out[i] = p[(start + i) % tokens_];
        }
    }

private:
    std::string path_;
    int fd_ = -1;
    void* data_ = nullptr;
    size_t file_bytes_ = 0;
    size_t tokens_ = 0;
};

struct Batch {
    int B = 0, T = 0;
    OwnTensor::Tensor input;
    OwnTensor::Tensor target;
};

class DataLoaderLite {
public:
    DataLoaderLite(int B, int T, int rank, int world_size, const std::string& split, const std::string& data_root, bool master_process = true, size_t max_tokens_per_shard = 400000000)
        : B_(B), T_(T), rank_(rank), world_(world_size), split_(split), root_(data_root), master_(master_process), max_tokens_(max_tokens_per_shard) {
        shards_ = list_shards(root_, split_, ".bin");
        if (shards_.empty()) throw std::runtime_error("no shards found");
        reset();
    }

    void reset() {
        current_shard_ = 0;
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    Batch next_batch() {
        const size_t BT = static_cast<size_t>(B_) * static_cast<size_t>(T_);
        if (pos_ + BT + 1 > shard_.size_tokens()) advance_shard();

        std::vector<uint16_t> buf;
        shard_.read_block(pos_, BT + 1, buf);

        Batch b;
        b.B = B_; b.T = T_;
        
        std::vector<uint16_t> x(BT), y(BT);
        for (size_t i = 0; i < BT; ++i) { x[i] = buf[i]; y[i] = buf[i + 1]; }

        OwnTensor::Device dev = OwnTensor::device::cuda_available() ? OwnTensor::Device::CUDA : OwnTensor::Device::CPU;
        
        b.input = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, 0)});
        b.input.set_data(x);
        
        b.target = OwnTensor::Tensor(OwnTensor::Shape{{B_, T_}}, {OwnTensor::Dtype::UInt16, OwnTensor::DeviceIndex(dev, 0)});
        b.target.set_data(y);

        pos_ += BT * static_cast<size_t>(world_);
        return b;
    }

private:
    void advance_shard() {
        current_shard_ = (current_shard_ + 1) % shards_.size();
        shard_.open(shards_[current_shard_], max_tokens_);
        pos_ = static_cast<size_t>(B_) * static_cast<size_t>(T_) * static_cast<size_t>(rank_);
    }

    int B_, T_, rank_, world_;
    std::string split_, root_;
    bool master_;
    size_t max_tokens_, current_shard_ = 0, pos_ = 0;
    std::vector<std::string> shards_;
    UInt16ShardView shard_;
};

int main() {
    try {
        const int B = 8;
        const int T = 1024;
        const int V = 50304; // Vocabulary size

        DataLoaderLite loader(B, T, 0, 1, "train", "./dummy_data");
        Batch batch = loader.next_batch();

        std::cout << "Successfully loaded batch. Input shape: " << batch.input.shape().dims[0] << "x" << batch.input.shape().dims[1] << std::endl;

        // Create a dummy model (Embedding layer)
        [[maybe_unused]] OwnTensor::Device dev = OwnTensor::device::cuda_available() ? OwnTensor::Device::CUDA : OwnTensor::Device::CPU;
        OwnTensor::DeviceIndex dev_idx = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0);

        // Create a dummy model (Embedding layer)
        OwnTensor::nn::Embedding embed(V, 32);
        // OwnTensor::nn::Linear linear(32, V);
        OwnTensor::nn::Sequential model({
            new OwnTensor::nn::Linear(32, 4*32),
            new OwnTensor::nn::ReLU(),
            new OwnTensor::nn::Linear(4*32, V)
        });
        embed.to(dev_idx);
        model.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, 0));

        for(auto& params: model.parameters()){
            params.to(dev_idx);
        }
        std::cout << "Completed";
        // if (dev == OwnTensor::Device::CUDA) {
            
        //     linear.to(dev_idx);
        // }
        
        // Forward pass
        OwnTensor::Tensor x = embed.forward(batch.input);
        std::cout << "Started1";
        std::cout << "Embed output requires_grad: " << x.requires_grad() << std::endl;
        OwnTensor::Tensor logits = model.forward(x);
        std::cout << "Started2";
        std::cout << "Logits requires_grad: " << logits.requires_grad() << std::endl;
        
        std::cout << "Logits shape: " << logits.shape().dims[0] << "x" << logits.shape().dims[1] << "x" << logits.shape().dims[2] << std::endl;

        // Compute Sparse Cross Entropy Loss
        OwnTensor::Tensor flattened_targets = batch.target.view(OwnTensor::Shape{{B * T}});
        // USE autograd::view to maintain the graph!
        OwnTensor::Tensor flattened_logits = OwnTensor::autograd::view(logits, OwnTensor::Shape{{B * T, V}});

        OwnTensor::Tensor loss = OwnTensor::autograd::sparse_cross_entropy_loss(flattened_logits, flattened_targets);
        std::cout << "Loss requires_grad: " << loss.requires_grad() << std::endl;
        
        std::cout << "Loss: ";
        loss.display();

        // Backward pass
        loss.backward();
        
        std::cout << "Backward pass successful." << std::endl;

        // Create a copy of weights before step to verify update
        OwnTensor::Tensor embed_w_old = embed.weight.clone();

        // Helper to stringify shape
        auto shape_to_str = [](const OwnTensor::Shape& s) {
            std::string res = "(";
            for (size_t i = 0; i < s.dims.size(); ++i) {
                res += std::to_string(s.dims[i]);
                if (i < s.dims.size() - 1) res += ", ";
            }
            res += ")";
            return res;
        };

        // Debug gradients
        auto check_grad = [&](const std::string& name, const OwnTensor::Tensor& t) {
            std::cout << "Parameter: " << name << " requires_grad: " << t.requires_grad();
            try {
                auto g = t.grad_view();
                std::cout << " | Grad allocated! | Shape: " << shape_to_str(g.shape());
            } catch (const std::exception& e) {
                std::cout << " | ERROR: Grad NOT allocated! (" << e.what() << ")";
            }
            std::cout << std::endl;
        };

        check_grad("embed.weight", embed.weight);
        // check_grad("linear.weight", linear.weight);
        int ind = 0;
        for(auto& params: model.parameters()){
            check_grad("linear" + std::to_string(ind), params);
            ind++;
        }
        // if (linear.bias.is_valid()) check_grad("linear.bias", linear.bias);
        
        // Test Optimizer
        std::vector<OwnTensor::Tensor> params = embed.parameters();
        auto lin_params = model.parameters();
        params.insert(params.end(), lin_params.begin(), lin_params.end());
        
        OwnTensor::nn::AdamW opt(params, 1e-3);
        opt.step();
        
        std::cout << "Optimizer step successful." << std::endl;

        // Verify weights changed
        OwnTensor::Tensor embed_w_new = embed.weight;
        // Check if they are different (a simple way is to check if norm of difference is > 0)
        // Since we might not have a simple norm, let's just print a few values
        // We'll just display a bit of the weight
        std::cout << "Embed weight sample (before opt.step): ";
        // Displaying a small part is hard without slice, so we'll just display the whole thing if it's not too big
        // Or we'll just assume display() is smart enough or we'll just see the first few lines
        embed_w_old.display();
        std::cout << "Embed weight sample (after opt.step): ";
        embed_w_new.display();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
