#pragma once

#include "autograd/AutogradOps.h"
#include "autograd/ops_template.h"
#include "../tensor/dtensor.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <optional>
#include <sstream>
#include <vector>

#include "EntropyKernels.h"
#include "dist_grad_norm_kernels.h"
#include "mlp/WeightInit.h"
#include "ops/helpers/GradNormKernels.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include <cmath>
#include <cuda_runtime.h>

#include "autograd/GraphRecorder.h"

namespace OwnTensor {
namespace dnn {

/**
 * @brief Distributed-aware gradient clipping
 * Based on the core clip_grad_norm_ but with all-reduce support.
 */
inline float clip_grad_norm_dtensor_nccl(std::vector<DTensor *> &params,
                                  float max_norm,
                                  std::shared_ptr<ProcessGroupNCCL> pg,
                                  float norm_type = 2.0f) {
  if (params.empty())
    return 0.0f;
  bool is_inf_norm = std::isinf(norm_type);
  int rank = pg->get_rank();

  // Setup GPU buffers
  static float *d_layer_norms = nullptr;
  static float *d_global_stat = nullptr;
  static size_t current_capacity = 0;

  if (params.size() > current_capacity) {
    if (d_layer_norms) {
      cudaFree(d_layer_norms);
      cudaFree(d_global_stat);
    }
    cudaMalloc(&d_layer_norms, params.size() * sizeof(float));
    cudaMalloc(&d_global_stat, sizeof(float));
    current_capacity = params.size();
  }

  cudaMemsetAsync(d_layer_norms, 0, current_capacity * sizeof(float));
  cudaMemsetAsync(d_global_stat, 0, sizeof(float));

  // 1. Compute Local Norms (Async)
  int valid_params = 0;
  for (auto *p : params) {
    auto &t = p->mutable_tensor();
    if (!t.has_grad()) {
      if (rank == 0)
        std::cout << "WARNING: " << p->name() << " has no gradient!"
                  << std::endl;
      continue;
    }
    Tensor g = t.grad_view();
    if (!g.is_valid() || g.numel() == 0) {
      if (rank == 0)
        std::cout << "WARNING: " << p->name()
                  << " gradient is invalid or empty!" << std::endl;
      continue;
    }

    if (!p->get_layout().is_replicated() || rank == 0) {
      if (is_inf_norm)
        launch_grad_norm_inf(g.data<float>(), &d_layer_norms[valid_params],
                             g.numel());
      else
        launch_grad_norm_sq(g.data<float>(), &d_layer_norms[valid_params],
                            g.numel());
      valid_params++;
    }
  }

  // 2. Reduce buffer
  launch_buffer_reduce(d_layer_norms, d_global_stat, valid_params, is_inf_norm);

  // 3. Global All-Reduce
  pg->all_reduce(d_global_stat, d_global_stat, 1, Dtype::Float32,
                 is_inf_norm ? op_t::max : op_t::sum);

  // 4. Apply Scaling
  for (auto *p : params) {
    auto t = p->mutable_tensor();

    if (!t.has_grad())
      continue;

    // Use grad_view() to get the gradient Tensor
    auto g = t.grad_view();

    if (!g.is_valid() || g.numel() == 0)
      continue;

    launch_apply_clip(g.data<float>(), d_global_stat, max_norm, g.numel(),
                      is_inf_norm);
  }

  // 5. Final sync
  float h_final_stat = 0.0f;
  cudaMemcpy(&h_final_stat, d_global_stat, sizeof(float),
             cudaMemcpyDeviceToHost);

  // Safety check: if the world exploded (NaN), don't sqrt a negative or NaN
  if (std::isnan(h_final_stat) || std::isinf(h_final_stat)) {
    return h_final_stat;
  }

  return is_inf_norm ? h_final_stat : std::sqrt(h_final_stat);
}

// inline float dist_clip_grad_norm(const std::vector<Tensor*>& params, float
// max_norm, ProcessGroupNCCL* pg, float norm_type = 2.0f) {
//     if (params.empty()) return 0.0f;

//     static float* s_d_norm = nullptr;
//     static float* s_d_clip_coef = nullptr;

//     bool is_cuda = false;
//     for (auto* p : params) {
//         if (p->has_grad() && p->device().is_cuda()) {
//             is_cuda = true;
//             break;
//         }
//     }

//     bool is_inf_norm = std::isinf(norm_type);

//     if (is_cuda) {
//         if (!s_d_norm) {
//             cudaMalloc(&s_d_norm, sizeof(float));
//             cudaMalloc(&s_d_clip_coef, sizeof(float));
//         }
//         cudaMemset(s_d_norm, 0, sizeof(float));

//         int grad_count = 0;
//         for (auto* p : params) {
//             if (!p->has_grad()) continue;
//             grad_count++;
//             Tensor grad = p->grad_view();

//             // Peak at the first valid gradient
//             if (pg && pg->get_rank() == 0 && grad_count == 1) {
//                 float g0 = grad.to_cpu().data<float>()[0];
//                 std::cout << "    [DEBUG] First valid grad peek: " <<
//                 std::scientific << g0 << std::endl;
//             }

//             if (is_inf_norm) {
//                 cuda::grad_norm_inf_cuda(grad.data<float>(), s_d_norm,
//                 grad.numel());
//             } else {
//                 cuda::grad_norm_squared_cuda(grad.data<float>(), s_d_norm,
//                 grad.numel());
//             }
//         }
//         if (pg && pg->get_rank() == 0) {
//             std::cout << "  [DEBUG] Params with grad: " << grad_count << " /
//             " << params.size() << std::endl;
//         }

//         float local_norm;
//         cudaMemcpy(&local_norm, s_d_norm, sizeof(float),
//         cudaMemcpyDeviceToHost);

//         float global_norm_val = local_norm;
//         if (pg) {
//             if (is_inf_norm) {
//                 MPI_Allreduce(&local_norm, &global_norm_val, 1, MPI_FLOAT,
//                 MPI_MAX, MPI_COMM_WORLD);
//             } else {
//                 MPI_Allreduce(&local_norm, &global_norm_val, 1, MPI_FLOAT,
//                 MPI_SUM, MPI_COMM_WORLD);
//             }
//             cudaMemcpy(s_d_norm, &global_norm_val, sizeof(float),
//             cudaMemcpyHostToDevice);
//         }

//         cuda::compute_clip_coef_cuda(s_d_norm, s_d_clip_coef, max_norm,
//         is_inf_norm);

//         for (auto* p : params) {
//             if (!p->has_grad()) continue;
//             Tensor grad = p->grad_view();
//             cuda::scale_gradients_with_gpu_coef_cuda(grad.data<float>(),
//             s_d_clip_coef, grad.numel());
//         }

//         float final_norm;
//         cudaMemcpy(&final_norm, s_d_norm, sizeof(float),
//         cudaMemcpyDeviceToHost); return final_norm;
//     } else {
//         // CPU implementation (briefly)
//         float total_norm = 0.0f;
//         // ... (can add if needed, but test is GPU)
//         return 0.0f;
//     }
// }

inline std::vector<float> load_csv(const std::string &filename) {
  std::vector<float> data;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open " << filename
              << ", using empty data\n";
    return data;
  }
  std::string line, cell;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
      if (!cell.empty()) {
        data.push_back(std::stof(cell));
      }
    }
  }
  return data;
}

inline std::vector<float> make_fixed_data(int64_t size, float base = 1.0f) {
  std::vector<float> data(size);
  for (int64_t i = 0; i < size; i++) {
    data[i] = base * (i + 1);
  }
  return data;
}

class DModule {
public:
  virtual ~DModule() = default;
  DModule() = default;
  DModule(const DModule &) = default;
  DModule &operator=(const DModule &) = default;
  DModule(DModule &&) = default;
  DModule &operator=(DModule &&) = default;

  virtual DTensor forward(DTensor &input) {
    throw std::runtime_error("DModule::forward not implemented");
  }

  virtual Tensor forward(Tensor &input) {
    throw std::runtime_error("DModule::forward not implemented");
  }

  virtual std::vector<DTensor *> parameters() {
    std::vector<DTensor *> all_params = params_;
    for (auto *child : children_) {
      auto child_params = child->parameters();
      all_params.insert(all_params.end(), child_params.begin(),
                        child_params.end());
    }
    return all_params;
  }

  virtual void to(DeviceIndex dev) {
    for (auto *child : children_) {
      child->to(dev);
    }
  }
  virtual void all_reduce_gradients(ProcessGroupNCCL *pg) {
    for (auto *child : children_) {
      child->all_reduce_gradients(pg);
    }
  }

  void zero_grad() {
    for (DTensor *p : parameters()) {
      if (p && p->mutable_tensor().requires_grad() &&
          p->mutable_tensor().has_grad()) {
        p->mutable_tensor().zero_grad();
      }
    }
  }

public:
  std::vector<DTensor *> params_;

  void register_parameter(Tensor *p) {
    DTensor *tensor = new DTensor();
    tensor->set_tensor(*p);
    params_.push_back(tensor);
  }

  void register_dmodule(DModule &m) { children_.push_back(&m); }

  void register_dmodule(DModule *m) { children_.push_back(m); }

  void register_module(DModule &m) { children_.push_back(&m); }

  void register_module(DModule *m) { children_.push_back(m); }

  std::vector<DModule *> children_;
};

class DLayerNorm : public DModule {
public:
  nn::LayerNorm ln;
  bool has_bias_;
  DLayerNorm() = default;
  DLayerNorm(int64_t dim, bool bias = true) : ln(dim), has_bias_(bias) {
    register_parameter(&ln.weight);
    if (has_bias_ && ln.bias.is_valid()) {
      register_parameter(&ln.bias);
    }
  }

  void to(DeviceIndex dev) { ln.to(dev); }

  DTensor forward(DTensor &input) {
    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DLayerNorm_forward");
    Tensor in_t = input.mutable_tensor();
    Tensor out_t = ln.forward(in_t);
    DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout(), "DLayerNorm_output");
    output.mutable_tensor() = out_t;
    return output;
  }

  void all_reduce_gradients(ProcessGroupNCCL *pg) {
    if (!pg)
      return;
    if (ln.weight.has_grad()) {
      pg->all_reduce_async(ln.weight.grad(), ln.weight.grad(),
                           ln.weight.numel(), Dtype::Float32, sum, false);
    }
    if (has_bias_ && ln.bias.is_valid() && ln.bias.has_grad()) {
      pg->all_reduce_async(ln.bias.grad(), ln.bias.grad(), ln.bias.numel(),
                           Dtype::Float32, sum, false);
    }
    DModule::all_reduce_gradients(pg);
  }
};

class DSequential : public DModule {
public:
  void add(std::shared_ptr<DModule> module) {
    modules_.push_back(module);
    register_module(*module);
  }

  DTensor forward(DTensor &input) {
    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DSequential_forward");
    DTensor x = input;
    for (auto &m : modules_) {
      x = m->forward(x);
    }
    return x;
  }

  void all_reduce_gradients(ProcessGroupNCCL *pg) {
    for (auto &m : modules_) {
      m->all_reduce_gradients(pg);
    }
  }

  auto &operator[](size_t idx) { return *modules_[idx]; }
  size_t size() const { return modules_.size(); }

private:
  std::vector<std::shared_ptr<DModule>> modules_;
};

class DColumnLinear : public DModule {
public:
  std::unique_ptr<DTensor> weight;
  std::unique_ptr<DTensor> bias = nullptr;
  DColumnLinear() = default;
  DColumnLinear(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
                int64_t batch_size, int64_t seq_len, int64_t in_features,
                int64_t out_features, std::vector<float> weight_data = {},
                bool use_bias = true, float sd = 0.02f, int seed = 42,
                bool sync_input = false, bool use_backward_hook = false)
      : mesh_(&mesh), pg_(pg), in_features_(in_features),
        out_features_(out_features), batch_size_(batch_size), seq_len_(seq_len),
        use_bias_(use_bias), sync_input_(sync_input),
        use_backward_hook_(use_backward_hook) {
    int world_size = pg->get_worldsize();
    int rank = pg->get_rank();
    out_local_ = out_features / world_size;

    Layout full_layout(mesh, {in_features_, out_features_});
    DTensor full_weight(mesh, pg, full_layout, "full_weight_init", sd, seed);

    if (rank == 0 && !weight_data.empty()) {

      full_weight.setData(weight_data);
      std::cout << " \n\n Set Data In FC1 done\n\n" << std::endl;
    }

    // } else {
    //     // Standard initialization on GPU (same seed on all ranks)
    //     Tensor w_full = mlp_forward::norm_rand_weight(
    //         Shape{{batch_size, in_features, out_features}},
    //         Dtype::Float32,
    //         Device::CPU,
    //         false,
    //         0.02f
    //     );
    //     full_weight.mutable_tensor().copy_(w_full);
    // }

    Layout weight_layout(mesh, {in_features_, out_local_}); // Sharded on dim 1
    weight = std::make_unique<DTensor>(mesh, pg, weight_layout,
                                       "DColumnLinear_weight");
    weight->shard_fused_transpose(1, 0, full_weight);

    weight->mutable_tensor().set_requires_grad(true);
    register_parameter(&weight->mutable_tensor());

    if (use_bias) {
      Layout bias_layout(mesh, {out_local_});
      bias = std::make_unique<DTensor>(mesh, pg, bias_layout,
                                       "DColumnLinear_bias");
      bias->mutable_tensor().fill(0.0f);
      bias->mutable_tensor().set_requires_grad(true);
      register_parameter(&bias->mutable_tensor());
    }
  }

  DTensor forward(DTensor &input) {
    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DColumnLinear_forward");
    if (sync_input_ && !use_backward_hook_) {
      input.sync_w_autograd(op_t::sum);
    } else if (use_backward_hook_) {
      input.register_backward_node();
    }

    auto in_shape = input.get_layout().get_global_shape();
    std::vector<int64_t> out_shape = in_shape;
    out_shape.back() = out_local_;

    Layout out_layout(*mesh_, out_shape);
    DTensor output(*mesh_, pg_, out_layout, "DColumnLinear_output");
    if (use_bias_ && bias) {
      output.linear_w_autograd(input, *weight, *bias);
    } else {
      output.linear_w_autograd(input, *weight);
    }
    return output;
  }

  bool use_bias() { return use_bias_; }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  int64_t in_features_;
  int64_t out_features_;
  int64_t out_local_;
  int64_t batch_size_;
  int64_t seq_len_;
  bool use_bias_;
  bool sync_input_;
  bool use_backward_hook_;
};

class DRowLinear : public DModule {
public:
  std::unique_ptr<DTensor> weight;
  std::unique_ptr<DTensor> bias;
  DRowLinear() = default;
  DRowLinear(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
             int64_t batch_size, int64_t seq_len, int64_t in_features,
             int64_t out_features, std::vector<float> weight_data = {},
             bool use_bias = true, float sd = 0.02f, int seed = 42,
             bool sync_output = true, bool with_autograd = false)
      : mesh_(&mesh), pg_(pg), in_features_(in_features),
        out_features_(out_features), batch_size_(batch_size), seq_len_(seq_len),
        use_bias_(use_bias), sync_output_(sync_output),
        with_autograd_(with_autograd) {
    int world_size = pg->get_worldsize();
    int rank = pg->get_rank();
    int64_t in_local_ = in_features / world_size;

    Layout full_layout(mesh, {in_features_, out_features_}, 0);
    DTensor full_weight(mesh, pg, full_layout, "full_weight_init", sd, seed);

    if (rank == 0 && !weight_data.empty()) {
      full_weight.setData(weight_data);
      std::cout << " \n\n Set Data In FC4 done\n\n" << std::endl;
    }
    // } else {
    //     // Standard initialization on GPU (same seed on all ranks)
    //     Tensor w_full = mlp_forward::norm_rand_weight(
    //         Shape{{batch_size, in_features, out_features}},
    //         Dtype::Float32,
    //         Device::CPU,
    //         false,
    //         0.02f
    //     );
    //     full_weight.mutable_tensor().copy_(w_full);
    // }

    Layout weight_layout(
        mesh, {in_features / world_size, out_features}); // Sharded on dim 0
    weight =
        std::make_unique<DTensor>(mesh, pg, weight_layout, "DRowLinear_weight");
    weight->shard_fused_transpose(0, 0, full_weight);
    weight->mutable_tensor().set_requires_grad(true);
    register_parameter(&weight->mutable_tensor());

    if (use_bias) {
      Layout bias_layout(mesh, {out_features_}); // Replicated
      bias =
          std::make_unique<DTensor>(mesh, pg, bias_layout, "DRowLinear_bias");
      bias->mutable_tensor().fill(0.0f);
      bias->mutable_tensor().set_requires_grad(true);
      register_parameter(&bias->mutable_tensor());
    }
  }

  DTensor forward(DTensor &input) {

    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DRowLinear_forward");
    auto in_shape = input.get_layout().get_global_shape();
    std::vector<int64_t> out_shape = in_shape;
    out_shape.back() = out_features_;

    Layout out_layout(*mesh_, out_shape);
    DTensor output(*mesh_, pg_, out_layout, "DRowLinear_output");
    if (use_bias_ && bias) {
      output.linear_w_autograd(input, *weight, *bias);
    } else {
      output.linear_w_autograd(input, *weight);
    }

    if (sync_output_ && with_autograd_) {
      output.sync_w_autograd(op_t::sum);
      output.wait();
    } else if (sync_output_ && !with_autograd_) {
      output.sync();
      output.wait();
    }
    return output;
  }

  void all_reduce_gradients(ProcessGroupNCCL *pg) {
    if (!pg)
      return;
    if (use_bias_ && bias && bias->mutable_tensor().has_grad()) {
      pg->all_reduce_async(
            bias->mutable_tensor().grad(), bias->mutable_tensor().grad(),
            bias->mutable_tensor().numel(), Dtype::Float32, sum, false);
          // ->wait();
    }
    DModule::all_reduce_gradients(pg);
  }

  void set_sync_output(bool sync) { sync_output_ = sync; }

  bool use_bias() { return use_bias_; }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  int64_t in_features_;
  int64_t in_local_;
  int64_t out_features_;
  int64_t batch_size_;
  int64_t seq_len_;
  bool use_bias_;
  bool sync_output_;
  bool with_autograd_;
};

// ... (DRowLinear ends)

class DEmbedding : public DModule {
public:
  std::unique_ptr<DTensor> weight;

  DEmbedding(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
             int64_t vocab_size, int64_t embedding_dim, int padding_idx = -1)
      : mesh_(&mesh), pg_(pg), num_embeddings_(vocab_size),
        embedding_dim_(embedding_dim), padding_idx_(padding_idx) {
    int world_size = pg->get_worldsize();
    int rank = pg->get_rank();

    // Get device ID for this rank
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int device_id = rank % num_devices;

    // REPLICATED STRATEGY: Each rank has the FULL embedding table
    // Simpler, no sharding, no masking needed
    vocab_start_ = 0;
    vocab_end_ = vocab_size;

    // Create full embedding table on each rank
    Layout weight_layout(mesh, {vocab_size, embedding_dim});
    weight =
        std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbedding_weight");
    weight->mutable_tensor().set_requires_grad(true);

    // Handle padding index if specified
    if (padding_idx >= 0 && padding_idx < vocab_size) {
      Tensor cpu_w = weight->mutable_tensor().to_cpu();
      float *data = cpu_w.data<float>();
      std::fill(data + padding_idx * embedding_dim,
                data + (padding_idx + 1) * embedding_dim, 0.0f);
      int device_idx = weight->mutable_tensor().device().index;
      weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
      weight->mutable_tensor().set_requires_grad(true);
    }

    register_parameter(&weight->mutable_tensor());
  }

  DTensor forward(DTensor &input) {
    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DEmbedding_forward");
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int rank = pg_->get_rank();
    int device_id = rank % num_devices;

    // REPLICATED STRATEGY: Simple embedding lookup, no masking needed
    // All ranks have full embedding table and compute same result

    Tensor input_tensor = input.mutable_tensor().to_cuda(device_id);
    Tensor local_out =
        autograd::embedding(weight->mutable_tensor(), input_tensor);

    // Create output DTensor
    std::vector<int64_t> input_shape = input.get_layout().get_global_shape();
    Layout out_layout(
        *mesh_,
        std::vector<int64_t>{input_shape[0], input_shape[1], embedding_dim_});
    DTensor output(*mesh_, pg_, out_layout, "DEmbedding_output");

    // output.assemble(2,0, *weight);

    // Assign result
    output.mutable_tensor() = local_out;

    // No all-reduce needed - all ranks have identical results
    // But we keep sync for gradient synchronization in backward
    // output.sync_w_autograd();
    output.wait();

    return output;
  }

  void all_reduce_gradients(ProcessGroupNCCL *pg) {
    if (!pg)
      return;
    if (weight->mutable_tensor().has_grad()) {
      pg->all_reduce_async(
            weight->mutable_tensor().grad(), weight->mutable_tensor().grad(),
            weight->mutable_tensor().numel(), Dtype::Float32, sum, false);
          // ->wait();
    }
    DModule::all_reduce_gradients(pg);
  }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  int64_t num_embeddings_;
  int64_t embedding_dim_;
  int padding_idx_;
  int64_t vocab_start_;
  int64_t vocab_end_;
  DTensor output;
};

// Custom backward node for fused vocab-parallel embedding
// class VocabParallelEmbeddingBackward : public Node {
//   Tensor saved_input_; // [B, T] int64 token indices
//   Tensor weight_ref_;  // reference to weight tensor (for grad accumulation)
//   int64_t B_, T_, C_;
//   int64_t start_v_, end_v_;

// public:
//   VocabParallelEmbeddingBackward(const Tensor &input, Tensor &weight, int64_t B,
//                                  int64_t T, int64_t C, int64_t start_v,
//                                  int64_t end_v)
//       : Node(1), saved_input_(input), weight_ref_(weight), B_(B), T_(T), C_(C),
//         start_v_(start_v), end_v_(end_v) {}

//   const char *name() const override { return "VocabParallelEmbeddingBackward"; }

//   variable_list apply(variable_list &&grads) override {
//     Tensor grad_output = grads[0]; // [B, T, D]

//     // Allocate grad_weight (same shape as weight) — zeros for atomic
//     // accumulation
//     Tensor grad_weight = Tensor::zeros(weight_ref_.shape(), weight_ref_.opts());

//     // Single fused kernel: scatter-add upstream grads into weight grad
//     OwnTensor::cuda::launch_vocab_parallel_embedding_bwd(
//         saved_input_.data<int64_t>(), grad_output.data<float>(),
//         grad_weight.data<float>(), B_, T_, C_, start_v_, end_v_, 0);

//     return {grad_weight};
//   }

//   void release_saved_variables() override { saved_input_ = Tensor(); }
// };

class DEmbeddingVParallel : public DModule {
public:
  int64_t vocab_size_;
  int64_t embedding_dim_;
  std::unique_ptr<DTensor> weight;

  int64_t local_v_;
  int64_t vocab_start_;
  int64_t vocab_end_;

  DEmbeddingVParallel(const DeviceMesh &mesh,
                      std::shared_ptr<ProcessGroupNCCL> pg, int64_t vocab_size,
                      int64_t embedding_dim, int64_t sd = 0.02f,
                      int64_t seed = 42)
      : mesh_(&mesh), pg_(pg), vocab_size_(vocab_size),
        embedding_dim_(embedding_dim) {

    int rank = pg->get_rank();
    int world_size = pg->get_worldsize();

    local_v_ = vocab_size / world_size;
    vocab_start_ = rank * local_v_;
    vocab_end_ = (rank + 1) * local_v_;

    if (rank == world_size - 1) {
      vocab_end_ = vocab_size;
      local_v_ = vocab_end_ - vocab_start_;
    }

    Layout weight_layout(mesh, {local_v_, embedding_dim});
    weight = std::make_unique<DTensor>(mesh, pg, weight_layout,
                                       "DEmbeddingVParallel_weight", sd, seed);

    weight->mutable_tensor().set_requires_grad(true);

    int64_t actual_padding_idx = vocab_size - 1;

    if (actual_padding_idx >= vocab_start_ && actual_padding_idx < vocab_end_) {
      int64_t local_pad_idx = actual_padding_idx - vocab_start_;
      Tensor cpu_w = weight->mutable_tensor().to_cpu();
      float *data = cpu_w.data<float>();
      std::fill(data + local_pad_idx * embedding_dim,
                data + (local_pad_idx + 1) * embedding_dim, 0.0f);
      int device_idx = weight->mutable_tensor().device().index;
      weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
      weight->mutable_tensor().set_requires_grad(true);
    }

    register_parameter(&weight->mutable_tensor());
  }

  Tensor forward(Tensor &input) {

    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DEmbeddingVParallel_forward");

    // 1. Convert to signed type for safe math
    Tensor input_i32 = input.as_type(Dtype::Int32);

    // 2. Identify input in this shard manually to avoid logical_AND CUDA issues
    Tensor mask_ge =
        (input_i32 >= (int32_t)vocab_start_).as_type(Dtype::Float32);
    Tensor mask_lt = (input_i32 < (int32_t)vocab_end_).as_type(Dtype::Float32);
    Tensor mask_f = autograd::mul(mask_ge, mask_lt);

    // 3. Map to local input [0, local_v)
    // We use the float mask to zero out out-of-shard input before converting to
    // UInt16
    Tensor local_input_i32 = (input_i32 - (int32_t)vocab_start_);
    Tensor local_input = (local_input_i32.as_type(Dtype::Float32) * mask_f)
                             .as_type(Dtype::UInt16);

    // 4. Local lookup
    Tensor local_embeds =
        autograd::embedding(weight->mutable_tensor(), local_input);

    // 5. Zero out embeddings for input not in this shard
    std::vector<int64_t> mask_dims = mask_f.shape().dims;
    mask_dims.push_back(1);
    Tensor mask_reshaped = autograd::reshape(mask_f, Shape{mask_dims});

    Tensor partial_embeds = autograd::mul(local_embeds, mask_reshaped);

    // 6. Aggregate from all shards using raw all-reduce (forward-only).
    // Backward gradient sync is handled by register_backward_all_reduce_hook in
    // GPT::forward().
    pg_->all_reduce(partial_embeds.data<float>(), partial_embeds.data<float>(),
                    partial_embeds.numel(), Dtype::Float32,
                    (op_t)0); // SUM, in-place

    return partial_embeds;
  }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
};

/**
 * Fused version of DEmbeddingVParallel:
 *  - Single CUDA kernel for forward (replaces 9 eager ops)
 *  - Single CUDA kernel for backward (fused scatter-add)
 *  - Accepts int64 input directly (no type casting chain)
 */
// class DEmbeddingVParallelFused : public DModule {
// public:
//   int64_t vocab_size_;
//   int64_t embedding_dim_;
//   std::unique_ptr<DTensor> weight;

//   int64_t local_v_;
//   int64_t vocab_start_;
//   int64_t vocab_end_;

//   DEmbeddingVParallelFused(const DeviceMesh &mesh,
//                            std::shared_ptr<ProcessGroupNCCL> pg,
//                            int64_t vocab_size, int64_t embedding_dim,
//                             std::vector<float> weight_data = {},
//                            int64_t sd = 0.02f, int64_t seed = 42)
//       : mesh_(&mesh), pg_(pg), vocab_size_(vocab_size),
//         embedding_dim_(embedding_dim) {

//     int rank = pg->get_rank();
//     int world_size = pg->get_worldsize();

//     local_v_ = vocab_size / world_size;
//     vocab_start_ = rank * local_v_;
//     vocab_end_ = (rank + 1) * local_v_;

//     if (rank == world_size - 1) {
//       vocab_end_ = vocab_size;
//       local_v_ = vocab_end_ - vocab_start_;
//     }

//     Layout weight_layout(mesh, {local_v_, embedding_dim});
//     weight = std::make_unique<DTensor>(
//         mesh, pg, weight_layout, "DEmbeddingVParallelFused_weight", sd, seed);

//     if(weight_data.empty()){
//         weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight", sd, seed );
//     }
//     else{
//         DTensor full_weight(mesh, pg, Layout(mesh, {vocab_size_, embedding_dim_}), "full_embedding_weight", sd, seed);
//         full_weight.setData(weight_data);
//         weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight");
//         weight->shard_fused_transpose(0, 0, full_weight);
//         // register_backward_node("DEmbeddingVParallel_weight", weight.get());
//     }

//     weight->mutable_tensor().set_requires_grad(true);

//     int64_t actual_padding_idx = vocab_size - 1;

//     if (actual_padding_idx >= vocab_start_ && actual_padding_idx < vocab_end_) {
//       int64_t local_pad_idx = actual_padding_idx - vocab_start_;
//       Tensor cpu_w = weight->mutable_tensor().to_cpu();
//       float *data = cpu_w.data<float>();
//       std::fill(data + local_pad_idx * embedding_dim,
//                 data + (local_pad_idx + 1) * embedding_dim, 0.0f);
//       int device_idx = weight->mutable_tensor().device().index;
//       weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
//       weight->mutable_tensor().set_requires_grad(true);
//     }

//     register_parameter(&weight->mutable_tensor());

//   }

//   Tensor forward(Tensor &input) {
//     int64_t B = input.shape().dims[0];
//     int64_t T = input.shape().dims[1];

//     // Ensure input is int64 for the fused kernel
//     Tensor input_i64 = input;
//     if (input.dtype() != Dtype::Int64) {
//       input_i64 = input.as_type(Dtype::Int64);
//     }

//     // Allocate output: [B, T, D]
//     Tensor output = Tensor::zeros(
//         Shape{{B, T, embedding_dim_}},
//         TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));

//     // Single fused kernel: shard-check + lookup + zero-fill
//     OwnTensor::cuda::launch_vocab_parallel_embedding_fwd(
//         input_i64.data<int64_t>(), weight->mutable_tensor().data<float>(),
//         output.data<float>(), B, T, embedding_dim_, vocab_start_, vocab_end_,
//         0);

//     // All-reduce partial embeddings across ranks (SUM)
//     pg_->all_reduce_async(output.data<float>(), output.data<float>(),
//                           output.numel(), OwnTensor::Dtype::Float32, (op_t)0,
//                           false)
//         ->wait();

//     // Custom autograd: connect output → weight gradient
//     if (weight->mutable_tensor().requires_grad()) {
//       auto grad_fn = std::make_shared<VocabParallelEmbeddingBackward>(
//           input_i64, weight->mutable_tensor(), B, T, embedding_dim_,
//           vocab_start_, vocab_end_);
//       grad_fn->set_next_edge(0,
//                              autograd::get_grad_edge(weight->mutable_tensor()));
//       output.set_grad_fn(grad_fn);
//       output.set_requires_grad(true);
//     }

//     return output;
//   }

// private:
//   const DeviceMesh *mesh_;
//   std::shared_ptr<ProcessGroupNCCL> pg_;
// };

class DGeLU : public DModule {
public:
  DTensor forward(DTensor &input) {
    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DGeLU_forward");
    Tensor in_tensor = input.mutable_tensor();
    Tensor out_tensor = autograd::gelu(in_tensor);

    // Create output with same layout as input
    DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
    output.mutable_tensor() = out_tensor;
    return output;
  }
};

// DLMHead: Replicated Language Model Head for next-token prediction
// Supports weight tying: can share weights with embedding layer (transposed)
class DLMHead : public DModule {
public:
  int64_t batch_size_;
  int64_t seq_len_;
  int64_t in_features_;
  int64_t vocab_size_;
  std::unique_ptr<DTensor> weight;
  bool use_tied_weights;
  DTensor *tied_weight_; // Pointer to embedding.weight

  DLMHead() = default;
  // Constructor with weight tying (shares embedding weight)
  DLMHead(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
          int64_t batch_size, int64_t seq_len, int64_t in_features,
          int64_t vocab_size, bool use_tied_weights,
          DTensor *embedding_weight) // Pointer to embedding.weight
      : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
        in_features_(in_features), vocab_size_(vocab_size),
        tied_weight_(embedding_weight), use_tied_weights(use_tied_weights) {}

  // Constructor without weight tying (separate weight matrix)
  DLMHead(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
          int64_t batch_size, int64_t seq_len, int64_t in_features,
          int64_t vocab_size, bool use_tied_weights,
          const std::vector<float> &weight_data = {})
      : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
        in_features_(in_features), vocab_size_(vocab_size),
        tied_weight_(nullptr), use_tied_weights(use_tied_weights) {
    // Shard along vocabulary dimension (dim 0)
    // std::cout << "Here:" << std::endl;

    Layout weight_layout(mesh, {vocab_size, in_features}, 0);
    weight =
        std::make_unique<DTensor>(mesh, pg, weight_layout, "lm_head_weight");

    if (!weight_data.empty() && pg->get_rank() == 0) {
      weight->setData(weight_data);
      std::cout << " \n\n Set Data In LM Head done\n\n" << std::endl;
    }

    weight->mutable_tensor().set_requires_grad(true);
  }

  DLMHead(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
          int64_t batch_size, int64_t seq_len, int64_t in_features,
          int64_t vocab_size, bool use_tied_weights, OwnTensor::Tensor tensor)
      : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
        in_features_(in_features), vocab_size_(vocab_size),
        tied_weight_(nullptr), use_tied_weights(use_tied_weights) {
    // Shard along vocabulary dimension (dim 0)
    // std::cout << "Here:" << std::endl;

    Layout weight_layout(mesh, {vocab_size, in_features}, 0);
    weight =
        std::make_unique<DTensor>(mesh, pg, weight_layout, "lm_head_weight");

    if (!tensor.data() && pg->get_rank() == 0) {
      weight->setData(tensor);
      std::cout << " \n\n Set Data In LM Head done\n\n" << std::endl;
    }

    weight->mutable_tensor().set_requires_grad(true);
  }

  std::vector<DTensor *> parameters() {
    if (!use_tied_weights && weight) {
      return {weight.get()};
    }
    return {};
  }

  DTensor forward(DTensor &input) {

    // autograd::GraphRecordMode::record_forward("ARITHMETIC: DLMHead_forward");
    Tensor weight_tensor = use_tied_weights ? tied_weight_->mutable_tensor()
                                            : weight->mutable_tensor();
    Tensor weight_t = weight_tensor.t();
    int64_t v_local = weight_t.shape().dims[1];

    // Compute local logits: [B, T, C] @ [C, V_local] -> [B, T, V_local]
    Tensor out_local =
        OwnTensor::autograd::matmul(input.mutable_tensor(), weight_t);

    if (v_local < vocab_size_ || mesh_->shape()[0] > 1) {
      // Sharded output: return [B, T, V_local]
      // We shard on dim 2 (the V dimension)
      Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_}, 2);
      DTensor output(*mesh_, pg_, out_layout, "lm_head_sharded_output");
      output.mutable_tensor() = out_local;
      return output;
    } else {
      // Replicated output: return [B, T, V]
      Layout out_layout(*mesh_, {batch_size_, seq_len_, vocab_size_});
      DTensor output(*mesh_, pg_, out_layout, "lm_head_replicated_output");
      output.mutable_tensor() = out_local;
      return output;
    }
  }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
};

class AdamW : public DModule {
public:
  /**
   * @brief Construct AdamW optimizer
   * @param lr Learning rate
   * @param beta1 First moment decay
   * @param beta2 Second moment decay
   * @param eps Epsilon for numerical stability
   * @param weight_decay Weight decay (L2 penalty)
   */
  explicit AdamW(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                 float eps = 1e-8f, float weight_decay = 0.01f)
      : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
        weight_decay_(weight_decay), t_(0) {}

  /**
   * @brief Perform optimization step with AdamW logic
   * @param params Vector of parameter tensors
   */

  void set_lr(float lr) { lr_ = lr; }
  float get_lr() const { return lr_; }

  void step(std::vector<DTensor *> params) {
    if (params.empty())
      return;
    t_++;

    // 1. Group Definitions
    struct TensorGroup {
      std::vector<OwnTensor::cuda::TensorInfo> gpu_params;
      std::vector<OwnTensor::cuda::TensorInfo> gpu_grads;
      std::vector<OwnTensor::cuda::TensorInfo> gpu_m;
      std::vector<OwnTensor::cuda::TensorInfo> gpu_v;
      float wd;
    };
    TensorGroup with_wd{{}, {}, {}, {}, weight_decay_};
    TensorGroup no_wd{{}, {}, {}, {}, 0.0f};

    static std::unordered_map<DTensor *, bool> wd_cache;

    for (DTensor *param : params) {
      if (!param->mutable_tensor().requires_grad())
        continue;

      // Safety check - gradients must be valid and already reduced/clipped
      if (!param->mutable_tensor().has_grad())
        continue;
      OwnTensor::Tensor grad_tensor = param->mutable_tensor().grad_view();
      if (!grad_tensor.is_valid())
        continue;

      // 2. State Management (Momentum Initialization)
      if (m_.find(param) == m_.end()) {
        TensorOptions opts_f32 =
            param->mutable_tensor().opts().with_dtype(Dtype::Float32);
        m_[param] =
            OwnTensor::Tensor::zeros(param->mutable_tensor().shape(), opts_f32);
        v_[param] =
            OwnTensor::Tensor::zeros(param->mutable_tensor().shape(), opts_f32);
      }

      // 3. Cached Weight Decay Filtering
      if (wd_cache.find(param) == wd_cache.end()) {
        std::string p_name = param->name();
        wd_cache[param] = (p_name.find("bias") == std::string::npos &&
                           p_name.find("norm") == std::string::npos &&
                           p_name.find("ln") == std::string::npos);
      }

      // Skip if gradient is not Float32 (kernel expects float*)
      if (grad_tensor.dtype() != Dtype::Float32) {
        std::cerr << "[AdamW] Warning: Skipping non-Float32 gradient for "
                  << param->name() << std::endl;
        continue;
      }

      TensorGroup &target = wd_cache[param] ? with_wd : no_wd;

      // 4. Batch Metadata for GPU
      target.gpu_params.push_back(
          {param->mutable_tensor().data<float>(),
           static_cast<int64_t>(param->mutable_tensor().numel())});
      target.gpu_grads.push_back({grad_tensor.data<float>(),
                                  static_cast<int64_t>(grad_tensor.numel())});
      target.gpu_m.push_back(
          {m_[param].data<float>(), static_cast<int64_t>(m_[param].numel())});
      target.gpu_v.push_back(
          {v_[param].data<float>(), static_cast<int64_t>(v_[param].numel())});
    }

    // 5. Launch Fused Kernels
    float bias_corr1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bias_corr2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    auto launch_group = [&](TensorGroup &g) {
      if (g.gpu_params.empty())
        return;
      OwnTensor::cuda::multi_tensor_adam_cuda(
          g.gpu_params, g.gpu_grads, g.gpu_m, g.gpu_v, lr_, beta1_, beta2_,
          eps_, g.wd, bias_corr1, bias_corr2, true);
    };

    launch_group(with_wd);
    launch_group(no_wd);
  }

private:
  float lr_, beta1_, beta2_, eps_, weight_decay_;
  int t_;
  // State: first and second moments for each parameter
  // Keys are DTensor pointers, values are local Tensor shards on GPU
  std::unordered_map<DTensor *, OwnTensor::Tensor> m_;
  std::unordered_map<DTensor *, OwnTensor::Tensor> v_;
};

// class VocabParallelCrossEntropyNode : public Node {
// private:
//   Tensor logits_;     // [B, T, local_V]
//   Tensor targets_;    // [B, T]
//   Tensor sum_exp_;    // [B, T, 1] (Global)
//   Tensor max_logits_; // [B, T, 1] (Global)
//   int rank_;
//   int64_t start_v_;
//   int64_t B_, T_, V_;

// public:
//   VocabParallelCrossEntropyNode(const Tensor &logits, const Tensor &targets,
//                                 const Tensor &sum_exp, const Tensor &max_logits,
//                                 int64_t start_v, int rank)
//       : Node(1), // One input: logits
//         logits_(logits.detach()), targets_(targets.detach()),
//         sum_exp_(sum_exp.detach()), max_logits_(max_logits.detach()),
//         start_v_(start_v), rank_(rank) {
//     B_ = logits.shape().dims[0];
//     T_ = logits.shape().dims[1];
//     V_ = logits.shape().dims[2];
//   }
//   const char *name() const override { return "VocabParallelCrossEntropyNode"; }
//   variable_list apply(variable_list &&grads) override {
//     Tensor grad_output = grads[0]; // [1]
//     float scale = 1.0f / (B_ * T_);

//     // Keep g_out on GPU to avoid CPU sync stall
//     Tensor g_out_tensor =
//         grad_output * scale; // [1] scalar tensor, stays on GPU

//     // Backward: dL/dx = (P - delta) * grad_output
//     // P = exp(logits - max) / sum_exp
//     // Use raw ops to avoid tracking
//     Tensor P;
//     {
//       // Use scope to release intermediates early
//       Tensor logits_minus_max = logits_ - max_logits_;
//       Tensor exp_logits = OwnTensor::exp(logits_minus_max);
//       P = exp_logits / sum_exp_; // Softmax probs
//     }
//     // grad = P * g_out (broadcasts scalar tensor)
//     Tensor grad_logits = P * g_out_tensor;
//     P.release();

//     // Sparse subtract needs host float — single D2H copy for the scalar
//     float g_out;
//     cudaMemcpy(&g_out, g_out_tensor.data<float>(), sizeof(float),
//                cudaMemcpyDeviceToHost);

//     OwnTensor::cuda::launch_sparse_subtract(grad_logits.data<float>(),
//                                             targets_.data<float>(), g_out, B_,
//                                             T_, V_, start_v_, 0);
//     return {grad_logits};
//   }
//   void release_saved_variables() override {
//     logits_ = Tensor();
//     targets_ = Tensor();
//     sum_exp_ = Tensor();
//     max_logits_ = Tensor();
//   }
// };

// Tensor vocab_parallel_cross_entropy(DTensor &logits_dt, Tensor &targets) {
//   Tensor local_logits = logits_dt.mutable_tensor();
//   int64_t rank = logits_dt.get_pg()->get_rank();
//   int64_t local_v = local_logits.shape().dims[2];
//   int64_t start_v = rank * local_v;
//   DeviceIndex device = local_logits.device();

//   int64_t B = logits_dt.mutable_tensor().shape().dims[0];
//   int64_t T = logits_dt.mutable_tensor().shape().dims[1];

//   // Local DTensors to avoid data races between micro-batches
//   std::unique_ptr<DTensor> s_global_max_dt;
//   std::unique_ptr<DTensor> s_global_sum_exp_dt;
//   std::unique_ptr<DTensor> s_global_target_dt;

//   Layout reduce_layout(logits_dt.get_device_mesh(), {B, T, 1});

//   s_global_max_dt =
//       std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(),
//                                 reduce_layout, "global_max");
//   s_global_sum_exp_dt =
//       std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(),
//                                 reduce_layout, "global_sum_exp");
//   s_global_target_dt =
//       std::make_unique<DTensor>(logits_dt.get_device_mesh(), logits_dt.get_pg(),
//                                 reduce_layout, "global_target_logit");

//   // 1. Compute Global Max (Raw)
//   Tensor local_max = OwnTensor::reduce_max(local_logits.detach(), {2}, true);
//   s_global_max_dt->mutable_tensor() = local_max;
//   s_global_max_dt->sync_w_autograd((op_t)1); // MAX
//   Tensor global_max = s_global_max_dt->mutable_tensor().detach();

//   // 2. Global SumExp (Raw)
//   Tensor logits_minus_max = (local_logits - global_max).detach();
//   Tensor exp_logits = OwnTensor::exp(logits_minus_max).detach();
//   Tensor local_sum_exp = OwnTensor::reduce_sum(exp_logits, {2}, true).detach();

//   s_global_sum_exp_dt->mutable_tensor() = local_sum_exp;
//   s_global_sum_exp_dt->sync_w_autograd((op_t)0); // SUM
//   Tensor global_sum_exp = s_global_sum_exp_dt->mutable_tensor().detach();
//   Tensor log_sum_exp = OwnTensor::log(global_sum_exp).detach();

//   // 3. Extract Target Logits (Raw)
//   Tensor targets_device = targets;
//   if (targets.device() != device)
//     targets_device = targets.to(device);
//   Tensor targets_float = targets_device.as_type(Dtype::Float32);

//   Tensor local_target_logits = Tensor::zeros(
//       Shape{{targets.shape().dims[0], targets.shape().dims[1], 1}},
//       targets_float.opts());
//   OwnTensor::cuda::launch_extract_target_logits(
//       local_logits.data<float>(), targets_float.data<float>(),
//       local_target_logits.data<float>(), targets.shape().dims[0],
//       targets.shape().dims[1], local_v, start_v, 0);

//   s_global_target_dt->mutable_tensor() = local_target_logits;
//   s_global_target_dt->sync_w_autograd((op_t)0); // SUM
//   Tensor global_target_val = s_global_target_dt->mutable_tensor().detach();

//   // 4. Final Loss (Raw calculation, then wrap in autograd if needed)
//   Tensor loss_per_token = log_sum_exp - (global_target_val - global_max);
//   Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0, 1, 2}, false);

//   // 5. Custom Autograd Setup
//   auto grad_fn = std::make_shared<VocabParallelCrossEntropyNode>(
//       local_logits, targets_float, global_sum_exp, global_max, start_v, rank);
//   grad_fn->set_next_edge(0, autograd::get_grad_edge(local_logits));
//   loss.set_grad_fn(grad_fn);
//   loss.set_requires_grad(true);

//   return loss;
// }

/**
 * V2 Backward Node: Fully fused backward using single CUDA kernel.
 * - Reads grad_output directly from GPU (no cudaMemcpy D2H)
 * - Computes softmax + gradient scaling + sparse subtraction in one pass
 * - sum_exp and max_logits stored as flat [BT] tensors (no reshape overhead)
//  */
// class VocabParallelCrossEntropyNodeV2 : public Node {
// private:
//   Tensor logits_;     // [B, T, V_local]
//   Tensor targets_;    // [BT] as float
//   Tensor sum_exp_;    // [BT] (global)
//   Tensor max_logits_; // [BT] (global)
//   int64_t start_v_;
//   int64_t B_, T_, V_;

// public:
//   VocabParallelCrossEntropyNodeV2(const Tensor &logits, const Tensor &targets,
//                                   const Tensor &sum_exp,
//                                   const Tensor &max_logits, int64_t start_v)
//       : Node(1), logits_(logits.detach()), targets_(targets.detach()),
//         sum_exp_(sum_exp.detach()), max_logits_(max_logits.detach()),
//         start_v_(start_v) {
//     B_ = logits.shape().dims[0];
//     T_ = logits.shape().dims[1];
//     V_ = logits.shape().dims[2];
//   }

//   const char *name() const override {
//     return "VocabParallelCrossEntropyNodeV2";
//   }

//   variable_list apply(variable_list &&grads) override {
//     Tensor grad_output = grads[0]; // [1] scalar, stays on GPU

//     float scale = 1.0f / (B_ * T_);

//     // Allocate output gradient
//     Tensor grad_logits = Tensor::zeros(logits_.shape(), logits_.opts());

//     // Single fused kernel: softmax + scale + sparse_subtract
//     // Reads grad_output[0] from device — NO cudaMemcpy needed
//     OwnTensor::cuda::launch_vocab_parallel_ce_backward(
//         logits_.data<float>(), targets_.data<float>(), sum_exp_.data<float>(),
//         max_logits_.data<float>(),
//         grad_output.data<float>(), // Device pointer!
//         grad_logits.data<float>(), B_, T_, V_, start_v_, scale, 0);

//     return {grad_logits};
//   }

//   void release_saved_variables() override {
//     logits_ = Tensor();
//     targets_ = Tensor();
//     sum_exp_ = Tensor();
//     max_logits_ = Tensor();
//   }
// };

// /**
//  * Fully optimized vocab-parallel cross-entropy (v2):
//  *
//  * vs. original:
//  *   Forward:  3 data passes over [BT*V_local] → 1 (fused kernel)
//  *   Backward: 5 kernel launches + cudaMemcpy  → 1 fused kernel
//  *   Comms:    3 all-reduces                   → 2 (packed SUM)
//  *   Overhead: 3 DTensor constructions/step     → 0 (raw all-reduce)
//  *   Memory:   [BT*V_local] exp intermediate    → 0 (computed inline)
//  */
// Tensor vocab_parallel_cross_entropy_v2(DTensor &logits_dt, Tensor &targets) {
//   Tensor local_logits = logits_dt.mutable_tensor();
//   auto pg = logits_dt.get_pg();
//   int64_t rank = pg->get_rank();
//   int64_t V_local = local_logits.shape().dims[2];
//   int64_t start_v = rank * V_local;
//   DeviceIndex device = local_logits.device();

//   int64_t B = local_logits.shape().dims[0];
//   int64_t T = local_logits.shape().dims[1];
//   int64_t BT = B * T;

//   // Prepare targets as float on device
//   Tensor targets_device = targets;
//   if (targets.device() != device)
//     targets_device = targets.to(device);
//   Tensor targets_float = targets_device.as_type(Dtype::Float32);

//   // Flatten targets from [B, T] to [BT] for fused kernels
//   Tensor targets_flat = targets_float;
//   if (targets_float.shape().dims.size() > 1) {
//     targets_flat = autograd::reshape(targets_float, Shape{{BT}});
//   }

//   // === Step 1: Local max (reduce_max over vocab dim) ===
//   // Result is [B, T, 1], flatten to [BT] for raw all-reduce
//   Tensor local_max =
//       OwnTensor::reduce_max(local_logits.detach(), {2}, true); // [B, T, 1]
//   Tensor local_max_flat = autograd::reshape(local_max, Shape{{BT}});

//   // === Step 2: All-reduce MAX (1 of 2 total all-reduces) ===
//   pg->all_reduce(local_max_flat.data<float>(), local_max_flat.data<float>(), BT,
//                  Dtype::Float32, (op_t)1); // MAX
//   Tensor global_max_flat = local_max_flat;

//   // === Step 3: Fused sum_exp + target_logit extraction (SINGLE pass over
//   // logits) === Packed output: first BT floats = sum_exp, next BT floats =
//   // target_logit
//   Tensor packed = Tensor::zeros(
//       Shape{{BT * 2}},
//       TensorOptions().with_dtype(Dtype::Float32).with_device(device));

//   OwnTensor::cuda::launch_fused_sum_exp_target(
//       local_logits.data<float>(), global_max_flat.data<float>(),
//       targets_flat.data<float>(), packed.data<float>(), B, T, V_local, start_v,
//       0);

//   // === Step 4: All-reduce SUM for packed [sum_exp | target_logit] (2 of 2) ===
//   pg->all_reduce(packed.data<float>(), packed.data<float>(), BT * 2,
//                  Dtype::Float32, (op_t)0); // SUM

//   // Unpack: create views without copying
//   // sum_exp = packed[0:BT], target_logit = packed[BT:2*BT]
//   Tensor global_sum_exp = Tensor(Shape{{BT}}, packed.opts());
//   Tensor global_target_logit = Tensor(Shape{{BT}}, packed.opts());
//   cudaMemcpyAsync(global_sum_exp.data<float>(), packed.data<float>(),
//                   BT * sizeof(float), cudaMemcpyDeviceToDevice);
//   cudaMemcpyAsync(global_target_logit.data<float>(), packed.data<float>() + BT,
//                   BT * sizeof(float), cudaMemcpyDeviceToDevice);

//   // === Step 5: Final loss = mean(log(sum_exp) - (target_logit - max)) ===
//   Tensor log_sum_exp = OwnTensor::log(global_sum_exp);
//   Tensor loss_per_token = log_sum_exp - (global_target_logit - global_max_flat);
//   Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0}, false);

//   // === Step 6: Custom autograd with V2 fused backward node ===
//   auto grad_fn = std::make_shared<VocabParallelCrossEntropyNodeV2>(
//       local_logits, targets_flat, global_sum_exp, global_max_flat, start_v);
//   grad_fn->set_next_edge(0, autograd::get_grad_edge(local_logits));
//   loss.set_grad_fn(grad_fn);
//   loss.set_requires_grad(true);

//   return loss;
// }

inline DTensor dmse_loss(DTensor &pred, DTensor &target) {
  // Tensor& pred_t = pred.mutable_tensor();
  // Tensor& target_t = target.mutable_tensor();

  // Tensor neg_target = target.mutable_tensor() * -1.0f;
  Tensor diff =
      autograd::add(pred.mutable_tensor(), target.mutable_tensor() * -1.0f);
  Tensor sq_diff = autograd::mul(diff, diff);
  Tensor local_loss = autograd::mean(sq_diff);

  Layout loss_layout(pred.get_device_mesh(), {1});
  DTensor loss(pred.get_device_mesh(), pred.get_pg(), loss_layout, "loss");
  loss.mutable_tensor() = local_loss;
  return loss;
}

// class ContextParallel : public DModule {
// public:

//   ContextParallel(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL>
//   pg, int64_t batch_size, int64_t seq_len, int64_t hidden_size)
//       : mesh_(&mesh), pg_(pg), batch_size_(batch_size), seq_len_(seq_len),
//         hidden_size_(hidden_size) {}
//   };

}
 // namespace dnn

namespace autograd{



class VocabParallelEmbeddingBackward : public Node {
  Tensor saved_input_; // [B, T] int64 token indices
  Tensor weight_ref_;  // reference to weight tensor (for grad accumulation)
  int64_t B_, T_, C_;
  int64_t start_v_, end_v_;

public:
  VocabParallelEmbeddingBackward(const Tensor &input, Tensor &weight, int64_t B,
                                 int64_t T, int64_t C, int64_t start_v,
                                 int64_t end_v)
      : Node(1), saved_input_(input), weight_ref_(weight), B_(B), T_(T), C_(C),
        start_v_(start_v), end_v_(end_v) {}

  const char *name() const override { return "VocabParallelEmbeddingBackward"; }

    variable_list apply(variable_list&& grads) override {
    if (grads.empty()) {
        throw std::runtime_error("ReluBackward: no gradients provided");
    }
    Tensor grad_output = grads[0]; // [B, T, D]

    // Allocate grad_weight (same shape as weight) — zeros for atomic
    // accumulation
    Tensor grad_weight = Tensor::zeros(weight_ref_.shape(), weight_ref_.opts());

    // Single fused kernel: scatter-add upstream grads into weight grad
    OwnTensor::cuda::launch_vocab_parallel_embedding_bwd_float4(
        saved_input_.data<int64_t>(), grad_output.data<float>(),
        grad_weight.data<float>(), B_, T_, C_, start_v_, end_v_, 0);

    return {grad_weight};
  }

  void release_saved_variables() override { saved_input_ = Tensor(); }
};


class DEmbeddingVParallelFused : public dnn::DModule {
public:
  int64_t vocab_size_;
  int64_t embedding_dim_;
  std::unique_ptr<DTensor> weight;

  int64_t local_v_;
  int64_t vocab_start_;
  int64_t vocab_end_;

  DEmbeddingVParallelFused(const DeviceMesh &mesh,
                           std::shared_ptr<ProcessGroupNCCL> pg,
                           int64_t vocab_size, int64_t embedding_dim,
                           float sd = 0.02f, int64_t seed = 42,
                           bool common_init = true)
      : mesh_(&mesh), pg_(pg), vocab_size_(vocab_size),
        embedding_dim_(embedding_dim) {

    int rank = pg->get_rank();
    int world_size = pg->get_worldsize();

    local_v_ = vocab_size / world_size;
    vocab_start_ = rank * local_v_;
    vocab_end_ = (rank + 1) * local_v_;

    if (rank == world_size - 1) {
      vocab_end_ = vocab_size;
      local_v_ = vocab_end_ - vocab_start_;
    }

    Layout weight_layout(mesh, {local_v_, embedding_dim_});
    weight = std::make_unique<DTensor>(
        mesh, pg, weight_layout, "DEmbeddingVParallelFused_weight", sd, seed);

        if(!common_init){
          weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight", sd, seed );
        }
        else{
          {

          // OwnTensor::TensorOptions opts;
          // opts.dtype = OwnTensor::Dtype::Float32;
          // OwnTensor::Shape shape{{vocab_size_, embedding_dim_}};
          // opts.device = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank);
          // opts.pinten = OwnTensor::Pinned_Flag(Pinned_Flag::Portable);
          DTensor full_weight(mesh, pg, Layout(mesh, {vocab_size_, embedding_dim_}), "full_embedding_weight", sd, seed);
          // full_weight.mutable_tensor() = OwnTensor::Tensor::randn<float>(shape, opts, seed);
          // full_weight.mutable_tensor().display();
          weight = std::make_unique<DTensor>(mesh, pg, weight_layout, "DEmbeddingVParallel_weight");
          weight->shard_fused_transpose(0, 0, full_weight);
        }
    }

    weight->mutable_tensor().set_requires_grad(true);

    int64_t actual_padding_idx = vocab_size - 1;

    if (actual_padding_idx >= vocab_start_ && actual_padding_idx < vocab_end_) {
      int64_t local_pad_idx = actual_padding_idx - vocab_start_;
      Tensor cpu_w = weight->mutable_tensor().to_cpu();
      float *data = cpu_w.data<float>();
      std::fill(data + local_pad_idx * embedding_dim,
                data + (local_pad_idx + 1) * embedding_dim, 0.0f);
      int device_idx = weight->mutable_tensor().device().index;
      weight->mutable_tensor() = cpu_w.to_cuda(device_idx);
      weight->mutable_tensor().set_requires_grad(true);
    }

    register_parameter(&weight->mutable_tensor());

  }

  Tensor forward(Tensor &input) {

    autograd::GraphRecordMode::record_forward("ARITHMETIC: DEmbeddingVParallelFused_forward");

    int64_t B = input.shape().dims[0];
    int64_t T = input.shape().dims[1];

    // Ensure input is int64 for the fused kernel
    Tensor input_i64 = input;
    if (input.dtype() != Dtype::Int64) {
      input_i64 = input.as_type(Dtype::Int64);
    }

    // Allocate output: [B, T, D]
    Tensor output = Tensor::zeros(
        Shape{{B, T, embedding_dim_}},
        TensorOptions().with_dtype(Dtype::Float32).with_device(input.device()));

    // Single fused kernel: shard-check + lookup + zero-fill
    OwnTensor::cuda::launch_vocab_parallel_embedding_fwd_float4(
        input_i64.data<int64_t>(), weight->mutable_tensor().data<float>(),
        output.data<float>(), B, T, embedding_dim_, vocab_start_, vocab_end_,
        0);

    // All-reduce partial embeddings across ranks (SUM)
    // GPU-side sync only (non-blocking to CPU) for pipeline efficiency
    auto work = pg_->all_reduce_async(output.data<float>(), output.data<float>(),
                          output.numel(), OwnTensor::Dtype::Float32, (op_t)0,
                          false);
    work->event_record();
    work->streamWait(OwnTensor::cuda::getCurrentStream());

    // Custom autograd: connect output → weight gradient
    if (weight->mutable_tensor().requires_grad()) {
      auto grad_fn = std::make_shared<VocabParallelEmbeddingBackward>(
          input_i64, weight->mutable_tensor(), B, T, embedding_dim_,
          vocab_start_, vocab_end_);
      grad_fn->set_next_edge(0, get_grad_edge(weight->mutable_tensor()));
      output.set_grad_fn(grad_fn);
      output.set_requires_grad(true);
    }

    return output;
  }

private:
  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
};



class VocabParallelCrossEntropyNodeV2 : public Node {
private:
  Tensor softmax_probs_; // [B, T, V_local] (Cached from forward pass)
  Tensor targets_;       // [BT] as float
  int64_t start_v_;
  int64_t B_, T_, V_;

public:
  VocabParallelCrossEntropyNodeV2(const Tensor &softmax_probs, const Tensor &targets,
                                  int64_t start_v)
      : Node(1), softmax_probs_(softmax_probs.detach()), targets_(targets.detach()),
        start_v_(start_v) {
    B_ = softmax_probs.shape().dims[0];
    T_ = softmax_probs.shape().dims[1];
    V_ = softmax_probs.shape().dims[2];
  }

  const char *name() const override {
    return "VocabParallelCrossEntropyNodeV2";
  }

  variable_list apply(variable_list &&grads) override {
    Tensor grad_output = grads[0]; // [1] scalar, stays on GPU

    float scale = 1.0f / (B_ * T_);

    // Allocate output gradient
    Tensor grad_logits = Tensor::zeros(softmax_probs_.shape(), softmax_probs_.opts());

    bool is_aligned = (reinterpret_cast<uintptr_t>(softmax_probs_.data<float>()) % 16 == 0) &&
                      (reinterpret_cast<uintptr_t>(grad_logits.data<float>()) % 16 == 0);

    // Fused float4 kernel: prob * scale - (is_target ? scale : 0)
    if (V_ % 4 == 0 && is_aligned) {
        OwnTensor::cuda::launch_vocab_parallel_ce_backward_float4(
            softmax_probs_.data<float>(), targets_.data<float>(),
            grad_output.data<float>(), // Device pointer!
            grad_logits.data<float>(), B_, T_, V_, start_v_, scale, 0);
    } else {
        OwnTensor::cuda::launch_vocab_parallel_ce_backward(
            softmax_probs_.data<float>(), targets_.data<float>(),
            grad_output.data<float>(), // Device pointer!
            grad_logits.data<float>(), B_, T_, V_, start_v_, scale, 0);
    }

    return {grad_logits};
  }

  void release_saved_variables() override {
    softmax_probs_ = Tensor();
    targets_ = Tensor();
  }
};

/**
 * Fully optimized vocab-parallel cross-entropy (v2):
 *
 * vs. original:
 *   Forward:  3 data passes over [BT*V_local] → 1 (fused kernel)
 *   Backward: 5 kernel launches + cudaMemcpy  → 1 fused kernel
 *   Comms:    3 all-reduces                   → 2 (packed SUM)
 *   Overhead: 3 DTensor constructions/step     → 0 (raw all-reduce)
 *   Memory:   [BT*V_local] exp intermediate    → 0 (computed inline)
 */
inline Tensor vocab_parallel_cross_entropy_v2(DTensor &logits_dt, Tensor &targets) {
  autograd::GraphRecordMode::record_forward("ARITHMETIC: vocab_parallel_cross_entropy_v2");
  Tensor local_logits = logits_dt.mutable_tensor();
  auto pg = logits_dt.get_pg();
  int64_t rank = pg->get_rank();
  int64_t V_local = local_logits.shape().dims[2];
  int64_t start_v = rank * V_local;
  DeviceIndex device = local_logits.device();

  int64_t B = local_logits.shape().dims[0];
  int64_t T = local_logits.shape().dims[1];
  int64_t BT = B * T;

  // Prepare targets as float on device
  Tensor targets_device = targets;
  if (targets.device() != device)
    targets_device = targets.to(device);
  Tensor targets_float = targets_device.as_type(Dtype::Float32);

  // Flatten targets from [B, T] to [BT] for fused kernels
  Tensor targets_flat = targets_float;
  if (targets_float.shape().dims.size() > 1) {
    targets_flat = autograd::reshape(targets_float, Shape{{BT}});
  }

  // === Step 1: Local max (reduce_max over vocab dim) ===
  // Result is [B, T, 1], flatten to [BT] for raw all-reduce
  Tensor local_max =
      OwnTensor::reduce_max(local_logits.detach(), {2}, true); // [B, T, 1]
  Tensor local_max_flat = autograd::reshape(local_max, Shape{{BT}});

  // === Step 2: All-reduce MAX (1 of 2 total all-reduces) ===
  pg->all_reduce(local_max_flat.data<float>(), local_max_flat.data<float>(), BT,
                 Dtype::Float32, (op_t)1); // MAX
  Tensor global_max_flat = local_max_flat;

  // === Step 3: Fused sum_exp + target_logit extraction (SINGLE pass over
  // logits) === Packed output: first BT floats = sum_exp, next BT floats =
  // target_logit
  Tensor packed = Tensor::zeros(
      Shape{{BT * 2}},
      TensorOptions().with_dtype(Dtype::Float32).with_device(device));

  // Allocate softmax_probs buffer early so the first kernel can write exponentiated logits into it
  Tensor softmax_probs = Tensor::empty(local_logits.shape(), local_logits.opts());

  // Check memory alignment for float4 optimizations (must be 16-byte aligned)
  bool is_aligned = (reinterpret_cast<uintptr_t>(local_logits.data<float>()) % 16 == 0) &&
                    (reinterpret_cast<uintptr_t>(softmax_probs.data<float>()) % 16 == 0);

  if (V_local % 4 == 0 && is_aligned) {
      OwnTensor::cuda::launch_fused_sum_exp_target_float4(
          local_logits.data<float>(), global_max_flat.data<float>(),
          targets_flat.data<float>(), packed.data<float>(),
          softmax_probs.data<float>(), B, T, V_local, start_v, 0);
  } else {
      OwnTensor::cuda::launch_fused_sum_exp_target(
          local_logits.data<float>(), global_max_flat.data<float>(),
          targets_flat.data<float>(), packed.data<float>(),
          softmax_probs.data<float>(), B, T, V_local, start_v, 0);
  }

  // === Step 4: All-reduce SUM for packed [sum_exp | target_logit] (2 of 2) ===
  pg->all_reduce(packed.data<float>(), packed.data<float>(), BT * 2,
                 Dtype::Float32, (op_t)0); // SUM

  // Unpack: create views without copying
  // sum_exp = packed[0:BT], target_logit = packed[BT:2*BT]
  Tensor global_sum_exp = Tensor(Shape{{BT}}, packed.opts());
  Tensor global_target_logit = Tensor(Shape{{BT}}, packed.opts());
  cudaMemcpyAsync(global_sum_exp.data<float>(), packed.data<float>(),
                  BT * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(global_target_logit.data<float>(), packed.data<float>() + BT,
                  BT * sizeof(float), cudaMemcpyDeviceToDevice);

  // === Step 5: Final loss = mean(log(sum_exp) - (target_logit - max)) ===
  Tensor log_sum_exp = OwnTensor::log(global_sum_exp);
  Tensor loss_per_token = log_sum_exp - (global_target_logit - global_max_flat);
  Tensor loss = OwnTensor::reduce_mean(loss_per_token, {0}, false);

  // === Step 6: Compute Softmax Probabilities for Backward Pass ===
  // probs = exp_logits / global_sum_exp (done in-place heavily utilizing the precomputed exps)
  if (V_local % 4 == 0 && is_aligned) {
      OwnTensor::cuda::launch_compute_softmax_probs_float4(
          softmax_probs.data<float>(), global_sum_exp.data<float>(),
          B, T, V_local, 0);
  } else {
      OwnTensor::cuda::launch_compute_softmax_probs(
          softmax_probs.data<float>(), global_sum_exp.data<float>(),
          B, T, V_local, 0);
  }

  // === Step 7: Custom autograd with V2 fused backward node ===
  auto grad_fn = std::make_shared<VocabParallelCrossEntropyNodeV2>(
      softmax_probs, targets_flat, start_v);
  grad_fn->set_next_edge(0, autograd::get_grad_edge(local_logits));
  loss.set_grad_fn(grad_fn);
  loss.set_requires_grad(true);

  return loss;
}



}
}// namespace OwnTensor