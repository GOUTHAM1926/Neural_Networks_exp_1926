#pragma once
#include "../../communication/include/ProcessGroupNCCL.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
// #include "memory/cachingAllocator.hpp"

#include "../../Tensor-Implementations/include/core/Tensor.h"
// #include "bridge/tensor_ops_bridge.h"
#include "../../Tensor-Implementations/include/device/Device.h"
#include "../../Tensor-Implementations/include/dtype/Dtype.h"

#include "../../Tensor-Implementations/include/autograd/AutogradOps.h"
#include "../../Tensor-Implementations/include/autograd/Engine.h"
#include "device_mesh.h"
#include "layout.h"
#include "placement.h"
// #include "reverse.cuh"

// #include "ad/ag_all.hpp"

using namespace OwnTensor;

// Forward declaration: LoadBalancer is defined later in this file
class LoadBalancer;

class DTensor {
public:
  DTensor(); // Default constructor for member initialization
  DTensor(const DeviceMesh &device_mesh, std::shared_ptr<ProcessGroupNCCL> pg,
          Layout layout, std::string name = "", float sd = 0.02f,
          int seed = 42);
  ~DTensor();

  void setData(const std::vector<float> &host_data);
  void setData(const std::vector<int64_t> &host_data);
  void setData(const OwnTensor::Tensor& tensor_data){ tensor_ = tensor_data; };
  std::vector<float> getData() const;

  // DTensor add(const DTensor& other) const;
  // DTensor sub(const DTensor& other) const;
  // DTensor mul(const DTensor& other) const;
  // DTensor div(const DTensor& other) const;

  void matmul(DTensor &A, DTensor &B);

  void Linear(DTensor &Input, DTensor &Weights, DTensor &Bias);

  // Autograd-enabled linear layer for gradient tracking
  void linear_w_autograd(DTensor &Input, DTensor &Weights, DTensor &Bias);
  void linear_w_autograd(DTensor &Input, DTensor &Weights); // No-bias overload

  // Backward pass - computes gradients for tensors with requires_grad=true
  void backward();
  // DTensor matmul(const DTensor& other) const;
  // DTensor  _column_parallel_matmul(const DTensor& other) const;
  // DTensor _row_parallel_matmul(const DTensor& other) const;
  // DTensor reshape(const std::vector<int64_t>& new_global_shape) const;

  void replicate(int root = 0);

  // void rotate3D( int dim, bool direction);
  // void rotate3D_mem( int dim, bool direction);  // Memory-optimized version

  void shard(int dim, int root, DTensor &parent_tensor);

  void shard_transpose(int dim, int root, DTensor &parent_tensor);
  // void shard_transpose_fused(int dim, int root, DTensor &parent_tensor);  //
  // Fused transpose+contiguous

  void shard_default(int dim, int root, DTensor &parent_tensor);
  void shard_fused_transpose(
      int dim, int root,
      DTensor &parent_tensor); // Uses custom kernel for reordering
  // void shard_own_transpose(int dim, int root, DTensor &parent_tensor);    //
  // Uses OwnTensor transpose



  void context_parallel_shard(std::vector<Tensor> &chunks, LoadBalancer &load_balancer); ;
  void context_parallel_unshard(std::vector<Tensor> &chunks, LoadBalancer &load_balancer) ;


  // void assemble(int dim, int root, DTensor &sharded_tensor);
  void sync();       // All-reduce with wait (blocking)
  void sync_async(); // All-reduce without wait (async)
  // void sync_async_backward_hook();
  void sync_w_autograd(op_t op = sum); // Autograd-aware sync (registers
                                       // backward for deep gradient all-reduce)
  void register_backward_all_reduce_hook(
      op_t op = sum); // Backward-only sync via hook
  void register_backward_node();
  void backward_allreduce();  // Identity forward, AllReduce(SUM) backward via autograd node
  void wait(); // Wait for pending async collective
  void wait_on_stream(cudaStream_t stream = 0); // GPU-side wait: no CPU block, inserts stream dependency
  // void wait_backward_hook();
  bool has_pending_collective() const; // Check if async collective pending

  // void assemble(int dim, int root, DTensor &sharded_tensor);

  // void permute_striped(int dim = 0);

  // // void DTensor::qkvsplit( DTensor &q, DTensor &k,DTensor &v);
  // void unpermute_striped(int dim = 0);

  // void saveCheckpoint(const std::string& path) const;
  // void loadCheckpoint(const std::string& path);
  void display();

  // void rand() ;

  void print() const;

  // ag::Value& get_value();
  // void enable_grad();
  // void get_tensor(){ return tensor_; }

  void setShape(std::vector<int64_t> newShape) { shape_ = newShape; }
  const Layout &get_layout() { return layout_; }
  Layout& mutable_layout() { return layout_; }
  const OwnTensor::Tensor &local_tensor() const { return tensor_; }
  OwnTensor::Tensor &mutable_tensor() { return tensor_; }
  void set_tensor(OwnTensor::Tensor &tensor) { tensor_ = tensor; }
  std::shared_ptr<ProcessGroupNCCL> get_pg() const { return pg_; }
  const DeviceMesh &get_device_mesh() const { return *device_mesh_; }
  int rank() const { return rank_; }
  int getSize() const { return size_; }
  std::string name() { return name_; }
  void set_name(const std::string& n) { name_ = n; }
  // void register_backward_node();

private:
  // DTensor(DeviceMesh device_mesh,
  //         std::shared_ptr<ProcessGroup> pg,
  //         const Layout& layout);

  // DTensor _column_parallel_matmul(const DTensor& other) const;
  // DTensor _row_parallel_matmul(const DTensor& other) const;

  int rank_;
  int world_size_;
  const DeviceMesh *device_mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  cudaStream_t stream_;
  // ag::Value value_ ;

  Layout layout_;

  OwnTensor::Tensor tensor_;

  // OwnTensor::Tensor temp_tensor_;

  int size_;
  std::vector<int64_t> shape_;
  std::string dtype_ = "float32";

  std::string name_;

  // Async collective tracking
  bool has_pending_collective_ = false;
  std::shared_ptr<Work> pending_work_ = nullptr;

  void printRecursive(const std::vector<float> &data,
                      const std::vector<int64_t> &dims, int dim,
                      int offset) const;
};


  class LoadBalancer  {
public:

  bool loadbalancing_enabled;
  int64_t world_size;
  int64_t chunkdim;
  LoadBalancer() = default;
  LoadBalancer(int64_t chunkdim_) { chunkdim = chunkdim_; }
  virtual void loadbalance(Tensor &tensor) = 0;
  virtual void unloadbalance(Tensor &tensor) = 0;
  void set_world_size(int64_t world_size_) { world_size = world_size_; }
  void set_load_balancing(bool enable) { loadbalancing_enabled = enable; }
  void set_chunk_dim(int64_t dim) { chunkdim = dim; }
  bool is_load_balancing_enabled() { return loadbalancing_enabled; }
};

class HeadTail : public LoadBalancer {
  private:
  cudaStream_t stream_ = nullptr;

  public:
  HeadTail() = default;
  HeadTail(int64_t world_size_) { world_size = world_size_; };
  void set_stream(cudaStream_t stream) { stream_ = stream; };
  void loadbalance(Tensor &tensor) override;
  void unloadbalance(Tensor &tensor) override;

  };
// void launch_reverse_kernel(float* d_src, float* d_dst, int nx, int ny, int
// nz, int dim, cudaStream_t stream);