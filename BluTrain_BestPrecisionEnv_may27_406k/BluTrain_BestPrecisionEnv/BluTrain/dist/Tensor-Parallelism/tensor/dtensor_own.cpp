#include "tensor/dtensor.h"
#include "bridge/tensor_ops_bridge_own.h"  
#include <cuda_runtime.h>
#include <nccl.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <numeric> 
#include <stdexcept> 
#include <sstream>


CachingAllocator gAllocator;


DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh, std::shared_ptr<ProcessGroup> pg)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(Layout::replicated(device_mesh, {})), 
      size_(0),
      data_block_(nullptr),
      temp_block_(nullptr),
      shape_({0}),
      tensor_(OwnTensor::Shape{{1}}, 
              OwnTensor::TensorOptions()
                  .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
                  .with_dtype(OwnTensor::Dtype::Float32)),
      temp_tensor_(tensor_) { 
    cudaSetDevice(rank_);
}


DTensor::DTensor(std::shared_ptr<DeviceMesh> device_mesh,
                 std::shared_ptr<ProcessGroup> pg,
                 const OwnTensor::Tensor& local_tensor,
                 const Layout& layout)
    : rank_(pg->getRank()),
      world_size_(pg->getWorldSize()),
      device_mesh_(device_mesh),
      pg_(pg),
      stream_(pg->getStream()),
      layout_(layout),
      tensor_(local_tensor),           
      temp_tensor_(local_tensor)      
{
    cudaSetDevice(rank_);
    
   
    shape_ = layout_.get_local_shape(rank_); 
    size_ = 1;
    for (int d : shape_) size_ *= d;

    
    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    

    temp_block_ = gAllocator.allocateMemory(layout.global_numel() * sizeof(float), stream_);
}

DTensor::~DTensor() {
    
    cudaStreamSynchronize(stream_); 
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
}


void DTensor::setData(const std::vector<float>& host_data, const Layout& layout) {
    layout_ = layout;
    
    
    std::vector<int> local_shape = layout_.get_local_shape(rank_);
    shape_ = local_shape;
    
    
    size_ = 1;
    for (int d : local_shape) size_ *= d;

    if (host_data.size() != (size_t)size_) {
        std::ostringstream oss;
        oss << "DTensor::setData: host_data size (" << host_data.size() 
            << ") does not match calculated local shard size (" << size_ << ")."
            << " Rank: " << rank_ << ", " << layout_.describe(rank_);
        throw std::runtime_error(oss.str());
    }

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(local_shape.begin(), local_shape.end());

    OwnTensor::TensorOptions opts;
    opts = opts.with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank_))
               .with_dtype(OwnTensor::Dtype::Float32);

    tensor_ = OwnTensor::Tensor(shape_obj, opts);
    tensor_.set_data(host_data);


    temp_tensor_ = OwnTensor::Tensor(shape_obj, opts);

    
    if (data_block_) gAllocator.freeMemory(data_block_);
    if (temp_block_) gAllocator.freeMemory(temp_block_);
    data_block_ = gAllocator.allocateMemory(size_ * sizeof(float), stream_);
    temp_block_ = gAllocator.allocateMemory(layout.global_numel() * sizeof(float), stream_);
}

std::vector<float> DTensor::getData() const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return host_data;
}


void DTensor::allReduce() {

    auto work = pg_->allReduce<float>(tensor_.data<float>(), size_, ncclFloat);
    work->wait();
}

void DTensor::reduceScatter() {
 
    auto work = pg_->reduceScatter<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_ / world_size_, ncclFloat);
    work->wait();
    
   
    std::swap(tensor_, temp_tensor_);
    shape_ = layout_.get_local_shape(rank_);
    size_ = 1; for(int d : shape_) size_ *= d;
}

void DTensor::allGather() {

    auto work = pg_->allGather<float>(
        temp_tensor_.data<float>(), tensor_.data<float>(), size_, ncclFloat);
    work->wait();
    
    
    std::swap(tensor_, temp_tensor_);
    shape_ = layout_.get_global_shape();
    size_ = 1; for(int d : shape_) size_ *= d;
}

void DTensor::broadcast(int root) {
    auto work = pg_->broadcast<float>(tensor_.data<float>(), size_, root, ncclFloat);
    work->wait();
}


// TensorOps (Bridge Integration)

#define DEFINE_TENSOR_OP(func, op_name) \
DTensor DTensor::func(const DTensor& other) const { \
    if (!layout_.is_compatible(other.get_layout())) { \
        throw std::runtime_error("Incompatible layouts for operation " #op_name); \
    } \
    OwnTensor::Tensor result = TensorOpsBridgeCustom::op_name(tensor_, other.tensor_); \
    return DTensor(device_mesh_, pg_, result, layout_); \
}

DEFINE_TENSOR_OP(add, add)
DEFINE_TENSOR_OP(sub, sub)
DEFINE_TENSOR_OP(mul, mul)
DEFINE_TENSOR_OP(div, div)


DTensor DTensor::matmul(const DTensor& other) const {
    const Layout& a_layout = this->layout_;
    const Layout& b_layout = other.get_layout();


    auto a_placement = a_layout.get_placement(0);  
    auto b_placement = b_layout.get_placement(0);

    // Column-Parallel Matmul 
    if (a_placement->type() == PlacementType::REPLICATE &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 1) {
        return _column_parallel_matmul(other);
    }
    
    // Row-Parallel Matmul (Split-K)

    if (a_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(a_placement.get())->dim() == 1 &&
        b_placement->type() == PlacementType::SHARD &&
        static_cast<const Shard*>(b_placement.get())->dim() == 0) {
         return _row_parallel_matmul(other);
    }


    std::ostringstream oss;
    oss << "DTensor::matmul: This sharding combination is not implemented!\n"
        << "  Layout A: " << a_layout.describe(rank_) << "\n"
        << "  Layout B: " << b_layout.describe(rank_);
    throw std::runtime_error(oss.str());
}

DTensor DTensor::_column_parallel_matmul(const DTensor& other) const {
    //    A = [M, K], B = [K, N_shard] -> Y_shard = [M, N_shard]
    OwnTensor::Tensor Y_shard = TensorOpsBridgeCustom::matmul(this->tensor_, other.local_tensor());

    
    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    
    std::vector<std::shared_ptr<Placement>> placements = {
        std::make_shared<Shard>(1)  
    };
    Layout Y_layout(device_mesh_, Y_global_shape, placements);

  
    return DTensor(device_mesh_, pg_, Y_shard, Y_layout);
}


DTensor DTensor::_row_parallel_matmul(const DTensor& other) const {
    //    A = [M, K_shard], B = [K_shard, N] -> Y_partial = [M, N]
    OwnTensor::Tensor Y_partial = TensorOpsBridgeCustom::matmul(this->tensor_, other.local_tensor());

    std::vector<int> Y_global_shape = {
        this->layout_.get_global_shape()[0],
        other.get_layout().get_global_shape()[1]
    };
    

    Layout Y_layout = Layout::replicated(device_mesh_, Y_global_shape);
    
.
    DTensor Y_out(device_mesh_, pg_, Y_partial, Y_layout);

    Y_out.allReduce(); 
    
    return Y_out;
}

void DTensor::matmulcustom( DTensor& A,  DTensor& B)  {

    if( ( shape_[1] != A.layout_.get_global_shape()[1] ) || ( shape_[2] != B.layout_.get_global_shape()[2] )){
        throw std::runtime_error("DTensor shape doesnt match matmul output shape ");
    }

    tensor_ = TensorOpsBridgeCustom::matmul(A.tensor_, B.tensor_);
}

DTensor DTensor::reshape(const std::vector<int>& new_global_shape) const {

    if (layout_.has_sharding()) {
        throw std::runtime_error("DTensor::reshape: Reshaping sharded tensors not yet implemented");
    }


    int new_local_size = 1;
    for (int d : new_global_shape) new_local_size *= d;
    if (new_local_size != size_) {
        throw std::runtime_error("DTensor::reshape: local element count mismatch.");
    }

    OwnTensor::Shape shape_obj;
    shape_obj.dims.assign(new_global_shape.begin(), new_global_shape.end());

    OwnTensor::Tensor reshaped_tensor = tensor_.reshape(shape_obj);


    Layout new_layout = Layout::replicated(device_mesh_, new_global_shape);


    return DTensor(device_mesh_, pg_, reshaped_tensor, new_layout);
}



void DTensor::saveCheckpoint(const std::string& path) const {
    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(), tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint file for writing: " << path << std::endl;
        return;
    }

    int ndim = static_cast<int>(shape_.size()); 
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(int));
    file.write(reinterpret_cast<const char*>(shape_.data()), ndim * sizeof(int));
    file.write(dtype_.c_str(), dtype_.size() + 1);
    file.write(reinterpret_cast<const char*>(host_data.data()), size_ * sizeof(float));
    file.close();

    std::cout << "[Rank " << rank_ << "] Checkpoint saved: " << path << " (" << ndim << "D, " << size_ << " elements)\n";
}

void DTensor::loadCheckpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Rank " << rank_ << "] Failed to open checkpoint: " << path << std::endl;
        return;
    }

    int ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
    
    std::vector<int> loaded_shape(ndim);
    file.read(reinterpret_cast<char*>(loaded_shape.data()), ndim * sizeof(int));
    char dtype_buf[32];
    file.read(dtype_buf, sizeof(dtype_buf));
    dtype_ = std::string(dtype_buf);
    
    int loaded_size = 1;
    for (int d : loaded_shape) loaded_size *= d;

    std::vector<float> host_data(loaded_size);
    file.read(reinterpret_cast<char*>(host_data.data()), loaded_size * sizeof(float));
    file.close();


    Layout loaded_layout = Layout::replicated(device_mesh_, loaded_shape);
    
    setData(host_data, loaded_layout);

    std::cout << "[Rank " << rank_ << "] Checkpoint loaded: " << path
              << " (" << ndim << "D, " << size_ << " elements)\n";
}


void DTensor::printRecursive(const std::vector<float>& data,
                             const std::vector<int>& dims,
                             int dim,
                             int offset) const {
    if (dims.empty() || dim < 0) return;

    if ((size_t)dim == dims.size() - 1) {
        std::cout << "[";
        for (int i = 0; i < dims[dim]; ++i) {
            if (i > 0) std::cout << ", ";
            if ((size_t)(offset + i) < data.size()) {
                std::cout << data[offset + i];
            }
        }
        std::cout << "]";
        return;
    }

    std::cout << "[";
    int stride = 1;
    for (size_t i = dim + 1; i < dims.size(); ++i) stride *= dims[i];
    for (int i = 0; i < dims[dim]; ++i) {
        if (i > 0) std::cout << ", ";
        printRecursive(data, dims, dim + 1, offset + i * stride);
    }
    std::cout << "]";
}

void DTensor::print() const {
    if (size_ <= 0 || tensor_.data<float>() == nullptr) {
        std::cerr << "[Rank " << rank_ << "] Tensor is empty or uninitialized!\n";
        return;
    }


    std::cout << layout_.describe(rank_) << std::endl;

    std::vector<float> host_data(size_);
    cudaMemcpyAsync(host_data.data(),
                    tensor_.data<float>(),
                    size_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_);
    cudaStreamSynchronize(stream_);

    std::cout << "[Rank " << rank_ << " Data] ";
    printRecursive(host_data, shape_, 0, 0);
    std::cout << "\n";
}