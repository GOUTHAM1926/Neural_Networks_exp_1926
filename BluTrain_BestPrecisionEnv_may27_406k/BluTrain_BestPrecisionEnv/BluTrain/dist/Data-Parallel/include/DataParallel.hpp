// #pragma once


// #include "Error_logs.hpp"
// #include "TensorLib.h"
// #include "autograd/Variable.h"
// #include "../../communication/include/ProcessGroupNCCL.h"
// #include <iostream>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <mpi.h>
// #include <thread>
// #include <future>
// #include <cstdlib>
// #include <memory>
// #include <vector>
// #include <unordered_map>
// #include "device/DeviceTransfer.h"
// #include "core/TensorDispatch.h"
// #include "profiler.hpp"
// #include <atomic>


// inline bool DEBUG_MODE_ = false;
// inline bool debug_log = false;

// struct DDP_Options{
//     std::shared_ptr<ProcessGroupNCCL> process_group_;
//     int64_t world_size_ = 1; // total no of ranks
//     bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
//     bool bucket_ = false;      //true if bucket logic to be enabled
//     int64_t bucket_size_ = 25 * 1024 * 1024;  //mb (only if bucket logic is true)
//     size_t rank_ = 0;
//     size_t local_rank_ = 0;
//     bool static_graph = false;
//     bool profiler_ = false;
//     size_t grad_accum_steps = 0;
//     bool is_accum_sync = false;
//     bool verify_params = false;
//     bool grad_as_view = false;
    

//     DDP_Options with_process_group(std::shared_ptr<ProcessGroupNCCL> process_group){
//         DDP_Options opts = *this;
//         opts.process_group_ = process_group;
//         opts.rank_ = opts.process_group_->get_rank();
//         opts.local_rank_ = opts.process_group_->get_local_rank();
//         return opts;
//     }

//     DDP_Options with_world_size(int world_size){
//         DDP_Options opts = *this;
//         opts.world_size_ = world_size;
//         return opts;
//     }

//     DDP_Options with_broadcast_buffer(bool broadcast_buffer){
//         DDP_Options opts = *this;
//         opts.broadcast_buffer_ = broadcast_buffer;
//         return opts;
//     }

//     DDP_Options with_rank(size_t rank = 0, size_t local_rank = 0){
//         DDP_Options opts = *this;
//         opts.rank_ = rank;
//         opts.local_rank_ = local_rank;
//         return opts;
//     }

//     DDP_Options with_bucket_data(bool bucket = false, int64_t bucket_size = 0){
//         DDP_Options opts = *this;
//         if(bucket && bucket_size == 0){
//             throw std::runtime_error(
//                 "Bucket size cannot be zero if bucketing is set to true..."
//             );
//         }
//         opts.bucket_ = bucket;
//         opts.bucket_size_ = bucket_size;
//         return opts;
//     }
//     DDP_Options with_static_graph(bool static_graph = false){
//         DDP_Options opts = *this;
//         opts.static_graph = static_graph;
//         return opts;
//     }

//     DDP_Options with_grad_accum(bool is_accum_sync = false, size_t steps = 0){
//         DDP_Options opts = *this;
//         opts.is_accum_sync = is_accum_sync;
//         opts.grad_accum_steps = steps;
//         return opts;
//     }

//     DDP_Options with_grad_view(bool grad_view_status){
//         DDP_Options opts = *this;
//         opts.grad_as_view = grad_view_status;
//         return opts;
//     }


// };

// /*
//     * Also contains components that are required.
//     * Components:
//         - initialization of the parameters through broadcast
//         - 
//         - hooks:
//             -Hooks are the components that can be hooked to any part of the backward or forward pass.
//             - This is mainly used to look at the intermediate results without disturbing the code
//             - This will be hooked inside the computation graph.
//             - Also hooks will not disturb the model flow at any time step.
//             - pre forward hooks:    Hooks that will be called before the model's forward pass.
//                                     This will work before forward to initialize the model
//                                     This is not be called by the user. works automatically.
//             - post backward hooks:  Hooks that will be called after the model's backward pass.
//                                     This iwll work after the backward pass for the gradient syncing.
//                                     This is either can be managed by the coder (without hooks) or can
//                                     use the hooks. (if needed)

//             - The hooks will take in functions as the parameter which will point to the function to be 
//             called has to happen.
//             - T         



// */



// struct BucketInit{
//     size_t bucket_size;
//     size_t bucket_max_size = 0;
//     std::vector<size_t> tensor_indices;
//     BucketInit() : bucket_max_size(0) {} 
//     BucketInit(size_t bucket_max_size):bucket_max_size(bucket_max_size){}
// };

// struct Key{
//     Key(OwnTensor::Dtype dtype, OwnTensor::DeviceIndex device_index, int ind = -1)
//         : dtype(dtype), device_index(device_index), ind(ind){}
//     OwnTensor::Dtype dtype;
//     OwnTensor::DeviceIndex device_index;
//     int ind = -1;
//     bool operator==(const Key& b) const {
//         return dtype == b.dtype &&
//             device_index.device == b.device_index.device &&
//             device_index.index == b.device_index.index &&
//             ind == b.ind;
//     }
    
// };


// struct TensorKey{
//     OwnTensor::Tensor tensor;
//     TensorKey(OwnTensor::Tensor tensor)
//         : tensor(tensor){}
//     int ind = -1;
//     bool operator==(const TensorKey& b) const {
//         auto bool_tensor = (tensor == b.tensor);
//         auto status_tensor = OwnTensor::reduce_all(bool_tensor);
//         bool status = (static_cast<bool*>(status_tensor.data())[0] == 1);
//         return status;
//     }
    
// };
// //this will be reducing the performance as we send the tensor to the cpu everytime and we can't afford that.
// //so this is temporary and the changes should be done to a optimal solution

// struct KeyHash {
//     size_t operator()(const Key& k) const {
//         return std::hash<int>()(static_cast<int>(k.dtype)) ^
//                std::hash<int>()(k.device_index.index) ^
//                std::hash<int>()(static_cast<int>(k.device_index.device)) ^
//                std::hash<int>()(k.ind);
//     }
// };
// struct simple_hash{
//     size_t operator()(const TensorKey& tensor) const{

//          return std::hash<const void*>{}(tensor.tensor.data()) ^
//            std::hash<float>{}(static_cast<float>(tensor.tensor.dtype())) ^
//            std::hash<float>{}(static_cast<float>(tensor.tensor.device().device));

//     }
// };



// // using namespace ag;

// class DataParallel : public OwnTensor::nn::Module{
// protected:
//     struct Bucket; 
// public:
//     DataParallel() = default;

//     DataParallel(OwnTensor::nn::Module* module, DDP_Options opts, bool init_sync = true );

//     ~DataParallel() {
//         if(compute_sync_event_) cudaEventDestroy(compute_sync_event_);
//     }

//     DataParallel& operator=(const DataParallel& other);
    
//     // DataParallel(ag::nn::Module* module,
//     //              int64_t world_size = 1,
//     //              int64_t rank = 0,
//     //              int64_t local_rank = 0,
//     //              bool init_sync = true,
//     //              bool bucket = false,
//     //              int64_t bucket_size = 25,
//     //              bool broadcast_buffer = false,
//     //              bool static_graph = false
//     // );

                 

//     void sync_model_parameter(std::vector<std::vector<size_t>>& bucket_indices);

//     //the pre and post functions are mainly for profiling and other stuffs,
//     // that will be used to manage and maintain the flow of the model and ddp.
//     void pre_forward();
//     void post_forward(OwnTensor::Tensor& output);
//     void pre_backward(const std::vector<OwnTensor::Tensor>& outputs);
//     void post_backward();


//     /*
//      * Below function will try to directly map the gradients to the flatten gradient without using the intermediat bucket views.
//      * Here, the gradients will act as a view not the bucket views.
//      * So, changes made in the gradient will reflect inside the flatten gradient and changes made in the flatten gradient will reflect in the gradient too.
//      * Can be optional, to activate this option, set ddp_opts.grad_as_view = true;
//      * */
//     void grad_map_flatten_grad(Bucket& bucket);

//     OwnTensor::Tensor forward(const OwnTensor::Tensor& input) override;

//     void naive_grad_sync();

//     void create_buckets(std::vector<std::vector<size_t>>& bucket_indices);
//     void bucket_data();
//     void create_bucket_gradient_views(Bucket& bucket);

//     void mark_param_ready(size_t variable_index);
//     void mark_bucket_ready(size_t bucket_index);

//     void accum_sync(size_t variable_index);

//     void finalize_backward();

//     void set_param_sync(bool status){ has_synced_params_ = status; }

//     bool is_param_sync(){ return has_synced_params_ == true; }

//     DDP_Options ddp_opts() { return opts_; } 

//     void all_reduce_grad_sync(Bucket& bucket);

//     void variable_locator_populator();

//     void mediator_hook(size_t variable_index);

//     void rebuild_buckets();

//     void search_for_unused(const std::vector<OwnTensor::Tensor>& outputs);

//     void verify_params();
    
//     void no_sync();

//     bool sync();

//     void set_stream(std::ostream& stream);
//     // void profiler(bool status){ DEBUG_MODE_ = status; }
    

// protected:

//     struct Bucket{
//         // the flattened gradient for the entire bucket's gradient.
//         // the gradients for each tensor within this bucket will be stored as a 1D flatten gradient tensor and
//         // can be seperated using offset.
//         OwnTensor::Tensor flatten_gradient;
        

//         //grad_incoming and grad_outgoing vectors contains the incoming and outgoing gradients at each step.
//         //the size is equal to the variables vector.
//         //This will help in an easy retrival and updation of the respective gradients in the buffer.
//         std::vector<OwnTensor::Tensor> grad_incoming;
//         std::vector<OwnTensor::Tensor> grad_outgoing;

//         //tensor variables. all the actaul tensors whose gradients are stored in the bucket.(this)
//         //will be in the order defined (not the graph order, but the deterministic order. [collect params]).
//         //will only contain tensors whose collective size is less than or equal to the bucket size.
//         //default bucket size = 25 MB.
//         std::vector<OwnTensor::Tensor> param_variables;
//         std::vector<size_t> param_indices;

//         //metrics that store the param_variable details.(can be used collected from the tensor meta data [from params]).
//         std::vector<size_t> offset;
//         std::vector<size_t> length;
//         std::vector<size_t> param_size;

//         //offset will store the initial index, and length will store the actual length.

//         //will be stored in bytes (not MB or not GB)
//         size_t bucket_size = 25 * 1024 * 1024;
//         std::shared_ptr<Work> work_obj_;
//         size_t bucket_pending = 0;
        
//     };

//     struct param_bucket_locator{
//         size_t bucket_index; // to store the bucket index it resides in
//         size_t inter_variable_index;
//         OwnTensor::Shape shape;
//         size_t numel;

//         param_bucket_locator() = default;

//         param_bucket_locator(
//             size_t bucket_index,
//             size_t variable_index
//         ): bucket_index(bucket_index), inter_variable_index(variable_index) {}
//     };
//     std::unordered_map<size_t, param_bucket_locator> variable_locator_;
//     std::vector<Bucket> buckets_;
//     std::vector<bool> completed_buckets_;
//     std::vector<OwnTensor::Tensor> rebuilt_params_order_;
//     std::vector<size_t> rebuilt_indices_order_;
//     // std::vector<std::shared_ptr<Work>> work_obj_;
    



// private:
//     DDP_Options opts_;
//     OwnTensor::nn::Module* module_;
//     bool init_sync_;
//     std::vector<OwnTensor::Tensor> parameters_;
//     std::shared_ptr<ProcessGroupNCCL> process_group_;
//     bool has_synced_params_ = false;
//     bool grad_sync_ = true;
//     LOG_CSV log_csv;
//     // int64_t bucket_size_ = 0;  //mb (only if bucket logic is true)
//     // int64_t world_size_ = 1; // total no of ranks
//     // bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
//     // bool bucket_ = false;      //true if bucket logic to be enabled
//     // size_t rank_ = 0;
//     // size_t local_rank_ = 0;
//     // bool static_graph_ = false;
//     int64_t number_of_iterations_ = 0;
//     bool first_forward = false; 
//     bool first_backward = false; // if the first backward is done, rebuild the buckets to make sure the only used parameters will be in the buckets.
//     bool is_accum_sync = false;
//     size_t no_of_backward = 0;
//     bool rebuild_bucket{true};

//     std::vector<size_t> unused_parameters_indices_;
//     std::vector<size_t> cached_unused_parameters_;  // cached after first backward
//     bool unused_params_cached_ = false;              // true after first BFS completes
//     std::unordered_map<std::shared_ptr<OwnTensor::Node>, size_t> gradAcc_variable_map_;
//     bool mark_unused = false;
//     size_t next_bucket = 0;
//     bool finalize_called = false;
//     std::mutex mutex_;
//     std::shared_ptr<Timer> timer_;
//     std::ostream* stream_ = &std::cout;
//     Profiler profiler;
//     cudaEvent_t compute_sync_event_ = nullptr;  // pre-allocated event for compute→comm syncing


// };

// std::vector<std::vector<size_t>> bucket_order_decide(
//     const std::vector<OwnTensor::Tensor>& params_, 
//     const std::optional<std::vector<size_t>>& param_backward_order = std::nullopt, 
//     size_t bucket_size_ = 25 * 1024 * 1024 
// );






// #pragma once


// #include "Error_logs.hpp"
// #include "TensorLib.h"
// #include "autograd/Variable.h"
// #include "../../communication/include/ProcessGroupNCCL.h"
// #include <iostream>
// #include <nccl.h>
// #include <cuda_runtime.h>
// #include <mpi.h>
// #include <thread>
// #include <future>
// #include <cstdlib>
// #include <memory>
// #include <vector>
// #include <unordered_map>
// #include "device/DeviceTransfer.h"
// #include "core/TensorDispatch.h"
// #include "profiler.hpp"
// #include <atomic>


// inline bool DEBUG_MODE_ = false;
// inline bool debug_log = false;

// struct DDP_Options{
//     std::shared_ptr<ProcessGroupNCCL> process_group_;
//     int64_t world_size_ = 1; // total no of ranks
//     bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
//     bool bucket_ = false;      //true if bucket logic to be enabled
//     int64_t bucket_size_ = 25 * 1024 * 1024;  //mb (only if bucket logic is true)
//     // Size (bytes) of the first bucket reduced in backward. Kept small (PyTorch
//     // uses 1 MB) so the first all-reduce fires as soon as the last layers' grads
//     // land, filling the comm pipeline early instead of waiting for a full
//     // bucket_size_ worth of gradients. 0 disables the small-first-bucket split.
//     int64_t first_bucket_size_ = 1 * 1024 * 1024;
//     size_t rank_ = 0;
//     size_t local_rank_ = 0;
//     bool static_graph = false;
//     bool profiler_ = false;
//     size_t grad_accum_steps = 0;
//     bool is_accum_sync = false;
//     bool verify_params = false;
//     bool grad_as_view = false;
    

//     DDP_Options with_process_group(std::shared_ptr<ProcessGroupNCCL> process_group){
//         DDP_Options opts = *this;
//         opts.process_group_ = process_group;
//         opts.rank_ = opts.process_group_->get_rank();
//         opts.local_rank_ = opts.process_group_->get_local_rank();
//         return opts;
//     }

//     DDP_Options with_world_size(int world_size){
//         DDP_Options opts = *this;
//         opts.world_size_ = world_size;
//         return opts;
//     }

//     DDP_Options with_broadcast_buffer(bool broadcast_buffer){
//         DDP_Options opts = *this;
//         opts.broadcast_buffer_ = broadcast_buffer;
//         return opts;
//     }

//     DDP_Options with_rank(size_t rank = 0, size_t local_rank = 0){
//         DDP_Options opts = *this;
//         opts.rank_ = rank;
//         opts.local_rank_ = local_rank;
//         return opts;
//     }

//     DDP_Options with_bucket_data(bool bucket = false, int64_t bucket_size = 0){
//         DDP_Options opts = *this;
//         if(bucket && bucket_size == 0){
//             throw std::runtime_error(
//                 "Bucket size cannot be zero if bucketing is set to true..."
//             );
//         }
//         opts.bucket_ = bucket;
//         opts.bucket_size_ = bucket_size;
//         return opts;
//     }
//     DDP_Options with_static_graph(bool static_graph = false){
//         DDP_Options opts = *this;
//         opts.static_graph = static_graph;
//         return opts;
//     }

//     DDP_Options with_grad_accum(bool is_accum_sync = false, size_t steps = 0){
//         DDP_Options opts = *this;
//         opts.is_accum_sync = is_accum_sync;
//         opts.grad_accum_steps = steps;
//         return opts;
//     }

//     DDP_Options with_grad_view(bool grad_view_status){
//         DDP_Options opts = *this;
//         opts.grad_as_view = grad_view_status;
//         return opts;
//     }

//     DDP_Options with_first_bucket(int64_t first_bucket_size_bytes){
//         DDP_Options opts = *this;
//         opts.first_bucket_size_ = first_bucket_size_bytes;
//         return opts;
//     }


// };

// /*
//     * Also contains components that are required.
//     * Components:
//         - initialization of the parameters through broadcast
//         - 
//         - hooks:
//             -Hooks are the components that can be hooked to any part of the backward or forward pass.
//             - This is mainly used to look at the intermediate results without disturbing the code
//             - This will be hooked inside the computation graph.
//             - Also hooks will not disturb the model flow at any time step.
//             - pre forward hooks:    Hooks that will be called before the model's forward pass.
//                                     This will work before forward to initialize the model
//                                     This is not be called by the user. works automatically.
//             - post backward hooks:  Hooks that will be called after the model's backward pass.
//                                     This iwll work after the backward pass for the gradient syncing.
//                                     This is either can be managed by the coder (without hooks) or can
//                                     use the hooks. (if needed)

//             - The hooks will take in functions as the parameter which will point to the function to be 
//             called has to happen.
//             - T         



// */



// struct BucketInit{
//     size_t bucket_size;
//     size_t bucket_max_size = 0;
//     std::vector<size_t> tensor_indices;
//     BucketInit() : bucket_max_size(0) {} 
//     BucketInit(size_t bucket_max_size):bucket_max_size(bucket_max_size){}
// };

// struct Key{
//     Key(OwnTensor::Dtype dtype, OwnTensor::DeviceIndex device_index, int ind = -1)
//         : dtype(dtype), device_index(device_index), ind(ind){}
//     OwnTensor::Dtype dtype;
//     OwnTensor::DeviceIndex device_index;
//     int ind = -1;
//     bool operator==(const Key& b) const {
//         return dtype == b.dtype &&
//             device_index.device == b.device_index.device &&
//             device_index.index == b.device_index.index &&
//             ind == b.ind;
//     }
    
// };


// struct TensorKey{
//     OwnTensor::Tensor tensor;
//     TensorKey(OwnTensor::Tensor tensor)
//         : tensor(tensor){}
//     int ind = -1;
//     bool operator==(const TensorKey& b) const {
//         auto bool_tensor = (tensor == b.tensor);
//         auto status_tensor = OwnTensor::reduce_all(bool_tensor);
//         bool status = (static_cast<bool*>(status_tensor.data())[0] == 1);
//         return status;
//     }
    
// };
// //this will be reducing the performance as we send the tensor to the cpu everytime and we can't afford that.
// //so this is temporary and the changes should be done to a optimal solution

// struct KeyHash {
//     size_t operator()(const Key& k) const {
//         return std::hash<int>()(static_cast<int>(k.dtype)) ^
//                std::hash<int>()(k.device_index.index) ^
//                std::hash<int>()(static_cast<int>(k.device_index.device)) ^
//                std::hash<int>()(k.ind);
//     }
// };
// struct simple_hash{
//     size_t operator()(const TensorKey& tensor) const{

//          return std::hash<const void*>{}(tensor.tensor.data()) ^
//            std::hash<float>{}(static_cast<float>(tensor.tensor.dtype())) ^
//            std::hash<float>{}(static_cast<float>(tensor.tensor.device().device));

//     }
// };



// // using namespace ag;

// class DataParallel : public OwnTensor::nn::Module{
// protected:
//     struct Bucket; 
// public:
//     DataParallel() = default;

//     DataParallel(OwnTensor::nn::Module* module, DDP_Options opts, bool init_sync = true );

//     ~DataParallel() {
//         if(compute_sync_event_) cudaEventDestroy(compute_sync_event_);
//     }

//     DataParallel& operator=(const DataParallel& other);
    
//     // DataParallel(ag::nn::Module* module,
//     //              int64_t world_size = 1,
//     //              int64_t rank = 0,
//     //              int64_t local_rank = 0,
//     //              bool init_sync = true,
//     //              bool bucket = false,
//     //              int64_t bucket_size = 25,
//     //              bool broadcast_buffer = false,
//     //              bool static_graph = false
//     // );

                 

//     void sync_model_parameter(std::vector<std::vector<size_t>>& bucket_indices);

//     //the pre and post functions are mainly for profiling and other stuffs,
//     // that will be used to manage and maintain the flow of the model and ddp.
//     void pre_forward();
//     void post_forward(OwnTensor::Tensor& output);
//     void pre_backward(const std::vector<OwnTensor::Tensor>& outputs);
//     void post_backward();


//     /*
//      * Below function will try to directly map the gradients to the flatten gradient without using the intermediat bucket views.
//      * Here, the gradients will act as a view not the bucket views.
//      * So, changes made in the gradient will reflect inside the flatten gradient and changes made in the flatten gradient will reflect in the gradient too.
//      * Can be optional, to activate this option, set ddp_opts.grad_as_view = true;
//      * */
//     void grad_map_flatten_grad(Bucket& bucket);

//     OwnTensor::Tensor forward(const OwnTensor::Tensor& input) override;

//     void naive_grad_sync();

//     void create_buckets(std::vector<std::vector<size_t>>& bucket_indices);
//     // Peels a small leading bucket off the registration-order bucket list
//     // (called BEFORE the std::reverse in the ctor) so the first bucket reduced
//     // in backward is small — see first_bucket_size_ in DDP_Options.
//     void apply_first_bucket_cap(std::vector<std::vector<size_t>>& bucket_indices);
//     void bucket_data();
//     void create_bucket_gradient_views(Bucket& bucket);

//     void mark_param_ready(size_t variable_index);
//     void mark_bucket_ready(size_t bucket_index);

//     void accum_sync(size_t variable_index);

//     void finalize_backward();

//     void set_param_sync(bool status){ has_synced_params_ = status; }

//     bool is_param_sync(){ return has_synced_params_ == true; }

//     DDP_Options ddp_opts() { return opts_; } 

//     void all_reduce_grad_sync(Bucket& bucket);

//     void variable_locator_populator();

//     void mediator_hook(size_t variable_index);

//     void rebuild_buckets();

//     void search_for_unused(const std::vector<OwnTensor::Tensor>& outputs);

//     void verify_params();
    
//     void no_sync();

//     bool sync();

//     void set_stream(std::ostream& stream);
//     // void profiler(bool status){ DEBUG_MODE_ = status; }
    

// protected:

//     struct Bucket{
//         // the flattened gradient for the entire bucket's gradient.
//         // the gradients for each tensor within this bucket will be stored as a 1D flatten gradient tensor and
//         // can be seperated using offset.
//         OwnTensor::Tensor flatten_gradient;
        

//         //grad_incoming and grad_outgoing vectors contains the incoming and outgoing gradients at each step.
//         //the size is equal to the variables vector.
//         //This will help in an easy retrival and updation of the respective gradients in the buffer.
//         std::vector<OwnTensor::Tensor> grad_incoming;
//         std::vector<OwnTensor::Tensor> grad_outgoing;

//         //tensor variables. all the actaul tensors whose gradients are stored in the bucket.(this)
//         //will be in the order defined (not the graph order, but the deterministic order. [collect params]).
//         //will only contain tensors whose collective size is less than or equal to the bucket size.
//         //default bucket size = 25 MB.
//         std::vector<OwnTensor::Tensor> param_variables;
//         std::vector<size_t> param_indices;

//         //metrics that store the param_variable details.(can be used collected from the tensor meta data [from params]).
//         std::vector<size_t> offset;
//         std::vector<size_t> length;
//         std::vector<size_t> param_size;

//         //offset will store the initial index, and length will store the actual length.

//         //will be stored in bytes (not MB or not GB)
//         size_t bucket_size = 25 * 1024 * 1024;
//         std::shared_ptr<Work> work_obj_;
//         size_t bucket_pending = 0;
        
//     };

//     struct param_bucket_locator{
//         size_t bucket_index; // to store the bucket index it resides in
//         size_t inter_variable_index;
//         OwnTensor::Shape shape;
//         size_t numel;

//         param_bucket_locator() = default;

//         param_bucket_locator(
//             size_t bucket_index,
//             size_t variable_index
//         ): bucket_index(bucket_index), inter_variable_index(variable_index) {}
//     };
//     std::unordered_map<size_t, param_bucket_locator> variable_locator_;
//     std::vector<Bucket> buckets_;
//     std::vector<bool> completed_buckets_;
//     std::vector<OwnTensor::Tensor> rebuilt_params_order_;
//     std::vector<size_t> rebuilt_indices_order_;
//     // std::vector<std::shared_ptr<Work>> work_obj_;
    



// private:
//     DDP_Options opts_;
//     OwnTensor::nn::Module* module_;
//     bool init_sync_;
//     std::vector<OwnTensor::Tensor> parameters_;
//     std::shared_ptr<ProcessGroupNCCL> process_group_;
//     bool has_synced_params_ = false;
//     bool grad_sync_ = true;
//     LOG_CSV log_csv;
//     // int64_t bucket_size_ = 0;  //mb (only if bucket logic is true)
//     // int64_t world_size_ = 1; // total no of ranks
//     // bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
//     // bool bucket_ = false;      //true if bucket logic to be enabled
//     // size_t rank_ = 0;
//     // size_t local_rank_ = 0;
//     // bool static_graph_ = false;
//     int64_t number_of_iterations_ = 0;
//     bool first_forward = false; 
//     bool first_backward = false; // if the first backward is done, rebuild the buckets to make sure the only used parameters will be in the buckets.
//     bool is_accum_sync = false;
//     size_t no_of_backward = 0;
//     bool rebuild_bucket{true};

//     std::vector<size_t> unused_parameters_indices_;
//     std::vector<size_t> cached_unused_parameters_;  // cached after first backward
//     bool unused_params_cached_ = false;              // true after first BFS completes
//     std::unordered_map<std::shared_ptr<OwnTensor::Node>, size_t> gradAcc_variable_map_;
//     bool mark_unused = false;
//     size_t next_bucket = 0;
//     bool finalize_called = false;
//     std::mutex mutex_;
//     std::shared_ptr<Timer> timer_;
//     std::ostream* stream_ = &std::cout;
//     Profiler profiler;
//     cudaEvent_t compute_sync_event_ = nullptr;  // pre-allocated event for compute→comm syncing


// };

// std::vector<std::vector<size_t>> bucket_order_decide(
//     const std::vector<OwnTensor::Tensor>& params_, 
//     const std::optional<std::vector<size_t>>& param_backward_order = std::nullopt, 
//     size_t bucket_size_ = 25 * 1024 * 1024 
// );


#pragma once


#include "Error_logs.hpp"
#include "TensorLib.h"
#include "autograd/Variable.h"
#include "../../communication/include/ProcessGroupNCCL.h"
#include <iostream>
#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <thread>
#include <future>
#include <cstdlib>
#include <memory>
#include <vector>
#include <unordered_map>
#include "device/DeviceTransfer.h"
#include "core/TensorDispatch.h"
#include "profiler.hpp"
#include <atomic>


inline bool DEBUG_MODE_ = false;
inline bool debug_log = false;

struct DDP_Options{
    std::shared_ptr<ProcessGroupNCCL> process_group_;
    int64_t world_size_ = 1; // total no of ranks
    bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
    bool bucket_ = false;      //true if bucket logic to be enabled
    int64_t bucket_size_ = 25 * 1024 * 1024;  //mb (only if bucket logic is true)
    // Size (bytes) of the first bucket reduced in backward. Kept small (PyTorch
    // uses 1 MB) so the first all-reduce fires as soon as the last layers' grads
    // land, filling the comm pipeline early instead of waiting for a full
    // bucket_size_ worth of gradients. 0 disables the small-first-bucket split.
    int64_t first_bucket_size_ = 1 * 1024 * 1024;
    size_t rank_ = 0;
    size_t local_rank_ = 0;
    bool static_graph = false;
    bool profiler_ = false;
    size_t grad_accum_steps = 0;
    bool is_accum_sync = false;
    bool verify_params = false;
    bool grad_as_view = false;
    

    DDP_Options with_process_group(std::shared_ptr<ProcessGroupNCCL> process_group){
        DDP_Options opts = *this;
        opts.process_group_ = process_group;
        opts.rank_ = opts.process_group_->get_rank();
        opts.local_rank_ = opts.process_group_->get_local_rank();
        return opts;
    }

    DDP_Options with_world_size(int world_size){
        DDP_Options opts = *this;
        opts.world_size_ = world_size;
        return opts;
    }

    DDP_Options with_broadcast_buffer(bool broadcast_buffer){
        DDP_Options opts = *this;
        opts.broadcast_buffer_ = broadcast_buffer;
        return opts;
    }

    DDP_Options with_rank(size_t rank = 0, size_t local_rank = 0){
        DDP_Options opts = *this;
        opts.rank_ = rank;
        opts.local_rank_ = local_rank;
        return opts;
    }

    DDP_Options with_bucket_data(bool bucket = false, int64_t bucket_size = 0){
        DDP_Options opts = *this;
        if(bucket && bucket_size == 0){
            throw std::runtime_error(
                "Bucket size cannot be zero if bucketing is set to true..."
            );
        }
        opts.bucket_ = bucket;
        opts.bucket_size_ = bucket_size;
        return opts;
    }
    DDP_Options with_static_graph(bool static_graph = false){
        DDP_Options opts = *this;
        opts.static_graph = static_graph;
        return opts;
    }

    DDP_Options with_grad_accum(bool is_accum_sync = false, size_t steps = 0){
        DDP_Options opts = *this;
        opts.is_accum_sync = is_accum_sync;
        opts.grad_accum_steps = steps;
        return opts;
    }

    DDP_Options with_grad_view(bool grad_view_status){
        DDP_Options opts = *this;
        opts.grad_as_view = grad_view_status;
        return opts;
    }

    DDP_Options with_first_bucket(int64_t first_bucket_size_bytes){
        DDP_Options opts = *this;
        opts.first_bucket_size_ = first_bucket_size_bytes;
        return opts;
    }


};

/*
    * Also contains components that are required.
    * Components:
        - initialization of the parameters through broadcast
        - 
        - hooks:
            -Hooks are the components that can be hooked to any part of the backward or forward pass.
            - This is mainly used to look at the intermediate results without disturbing the code
            - This will be hooked inside the computation graph.
            - Also hooks will not disturb the model flow at any time step.
            - pre forward hooks:    Hooks that will be called before the model's forward pass.
                                    This will work before forward to initialize the model
                                    This is not be called by the user. works automatically.
            - post backward hooks:  Hooks that will be called after the model's backward pass.
                                    This iwll work after the backward pass for the gradient syncing.
                                    This is either can be managed by the coder (without hooks) or can
                                    use the hooks. (if needed)

            - The hooks will take in functions as the parameter which will point to the function to be 
            called has to happen.
            - T         



*/



struct BucketInit{
    // bucket_size MUST start at 0 — bucket_order_decide reads
    // (bucket_size + tensor.nbytes() <= bucket_max) to decide coalescing on a
    // freshly-constructed BucketInit. Left uninitialized it holds garbage, the
    // test fails for every parameter, and each param lands in its own bucket →
    // one tiny NCCL all-reduce per parameter (~148) instead of ~20 grouped
    // 25 MB buckets, wrecking comm efficiency and backward/comm overlap.
    size_t bucket_size = 0;
    size_t bucket_max_size = 0;
    std::vector<size_t> tensor_indices;
    BucketInit() = default;
    BucketInit(size_t bucket_max_size):bucket_max_size(bucket_max_size){}
};

struct Key{
    Key(OwnTensor::Dtype dtype, OwnTensor::DeviceIndex device_index, int ind = -1)
        : dtype(dtype), device_index(device_index), ind(ind){}
    OwnTensor::Dtype dtype;
    OwnTensor::DeviceIndex device_index;
    int ind = -1;
    bool operator==(const Key& b) const {
        return dtype == b.dtype &&
            device_index.device == b.device_index.device &&
            device_index.index == b.device_index.index &&
            ind == b.ind;
    }
    
};


struct TensorKey{
    OwnTensor::Tensor tensor;
    TensorKey(OwnTensor::Tensor tensor)
        : tensor(tensor){}
    int ind = -1;
    bool operator==(const TensorKey& b) const {
        auto bool_tensor = (tensor == b.tensor);
        auto status_tensor = OwnTensor::reduce_all(bool_tensor);
        bool status = (static_cast<bool*>(status_tensor.data())[0] == 1);
        return status;
    }
    
};
//this will be reducing the performance as we send the tensor to the cpu everytime and we can't afford that.
//so this is temporary and the changes should be done to a optimal solution

struct KeyHash {
    size_t operator()(const Key& k) const {
        return std::hash<int>()(static_cast<int>(k.dtype)) ^
               std::hash<int>()(k.device_index.index) ^
               std::hash<int>()(static_cast<int>(k.device_index.device)) ^
               std::hash<int>()(k.ind);
    }
};
struct simple_hash{
    size_t operator()(const TensorKey& tensor) const{

         return std::hash<const void*>{}(tensor.tensor.data()) ^
           std::hash<float>{}(static_cast<float>(tensor.tensor.dtype())) ^
           std::hash<float>{}(static_cast<float>(tensor.tensor.device().device));

    }
};



// using namespace ag;

class DataParallel : public OwnTensor::nn::Module{
protected:
    struct Bucket; 
public:
    DataParallel() = default;

    DataParallel(OwnTensor::nn::Module* module, DDP_Options opts, bool init_sync = true );

    ~DataParallel() {
        if(compute_sync_event_) cudaEventDestroy(compute_sync_event_);
    }

    DataParallel& operator=(const DataParallel& other);
    
    // DataParallel(ag::nn::Module* module,
    //              int64_t world_size = 1,
    //              int64_t rank = 0,
    //              int64_t local_rank = 0,
    //              bool init_sync = true,
    //              bool bucket = false,
    //              int64_t bucket_size = 25,
    //              bool broadcast_buffer = false,
    //              bool static_graph = false
    // );

                 

    void sync_model_parameter(std::vector<std::vector<size_t>>& bucket_indices);

    //the pre and post functions are mainly for profiling and other stuffs,
    // that will be used to manage and maintain the flow of the model and ddp.
    void pre_forward();
    void post_forward(OwnTensor::Tensor& output);
    void pre_backward(const std::vector<OwnTensor::Tensor>& outputs);
    void post_backward();


    /*
     * Below function will try to directly map the gradients to the flatten gradient without using the intermediat bucket views.
     * Here, the gradients will act as a view not the bucket views.
     * So, changes made in the gradient will reflect inside the flatten gradient and changes made in the flatten gradient will reflect in the gradient too.
     * Can be optional, to activate this option, set ddp_opts.grad_as_view = true;
     * */
    void grad_map_flatten_grad(Bucket& bucket);

    OwnTensor::Tensor forward(const OwnTensor::Tensor& input) override;

    void naive_grad_sync();

    void create_buckets(std::vector<std::vector<size_t>>& bucket_indices);
    // Peels a small leading bucket off the registration-order bucket list
    // (called BEFORE the std::reverse in the ctor) so the first bucket reduced
    // in backward is small — see first_bucket_size_ in DDP_Options.
    void apply_first_bucket_cap(std::vector<std::vector<size_t>>& bucket_indices);
    void bucket_data();
    void create_bucket_gradient_views(Bucket& bucket);

    void mark_param_ready(size_t variable_index);
    void mark_bucket_ready(size_t bucket_index);

    void accum_sync(size_t variable_index);

    void finalize_backward();

    void set_param_sync(bool status){ has_synced_params_ = status; }

    bool is_param_sync(){ return has_synced_params_ == true; }

    DDP_Options ddp_opts() { return opts_; } 

    void all_reduce_grad_sync(Bucket& bucket);

    void variable_locator_populator();

    void mediator_hook(size_t variable_index);

    void rebuild_buckets();

    void search_for_unused(const std::vector<OwnTensor::Tensor>& outputs);

    void verify_params();
    
    void no_sync();

    bool sync();

    void set_stream(std::ostream& stream);
    // void profiler(bool status){ DEBUG_MODE_ = status; }
    

protected:

    struct Bucket{
        // the flattened gradient for the entire bucket's gradient.
        // the gradients for each tensor within this bucket will be stored as a 1D flatten gradient tensor and
        // can be seperated using offset.
        OwnTensor::Tensor flatten_gradient;
        

        //grad_incoming and grad_outgoing vectors contains the incoming and outgoing gradients at each step.
        //the size is equal to the variables vector.
        //This will help in an easy retrival and updation of the respective gradients in the buffer.
        std::vector<OwnTensor::Tensor> grad_incoming;
        std::vector<OwnTensor::Tensor> grad_outgoing;

        //tensor variables. all the actaul tensors whose gradients are stored in the bucket.(this)
        //will be in the order defined (not the graph order, but the deterministic order. [collect params]).
        //will only contain tensors whose collective size is less than or equal to the bucket size.
        //default bucket size = 25 MB.
        std::vector<OwnTensor::Tensor> param_variables;
        std::vector<size_t> param_indices;

        //metrics that store the param_variable details.(can be used collected from the tensor meta data [from params]).
        std::vector<size_t> offset;
        std::vector<size_t> length;
        std::vector<size_t> param_size;

        //offset will store the initial index, and length will store the actual length.

        //will be stored in bytes (not MB or not GB)
        size_t bucket_size = 25 * 1024 * 1024;
        std::shared_ptr<Work> work_obj_;
        size_t bucket_pending = 0;
        
    };

    struct param_bucket_locator{
        size_t bucket_index; // to store the bucket index it resides in
        size_t inter_variable_index;
        OwnTensor::Shape shape;
        size_t numel;

        param_bucket_locator() = default;

        param_bucket_locator(
            size_t bucket_index,
            size_t variable_index
        ): bucket_index(bucket_index), inter_variable_index(variable_index) {}
    };
    std::unordered_map<size_t, param_bucket_locator> variable_locator_;
    std::vector<Bucket> buckets_;
    std::vector<bool> completed_buckets_;
    std::vector<OwnTensor::Tensor> rebuilt_params_order_;
    std::vector<size_t> rebuilt_indices_order_;
    // std::vector<std::shared_ptr<Work>> work_obj_;
    



private:
    DDP_Options opts_;
    OwnTensor::nn::Module* module_;
    bool init_sync_;
    std::vector<OwnTensor::Tensor> parameters_;
    std::shared_ptr<ProcessGroupNCCL> process_group_;
    bool has_synced_params_ = false;
    bool grad_sync_ = true;
    LOG_CSV log_csv;
    // int64_t bucket_size_ = 0;  //mb (only if bucket logic is true)
    // int64_t world_size_ = 1; // total no of ranks
    // bool broadcast_buffer_ = false;    //broadcast buffers such as running average , mean, etc
    // bool bucket_ = false;      //true if bucket logic to be enabled
    // size_t rank_ = 0;
    // size_t local_rank_ = 0;
    // bool static_graph_ = false;
    int64_t number_of_iterations_ = 0;
    bool first_forward = false; 
    bool first_backward = false; // if the first backward is done, rebuild the buckets to make sure the only used parameters will be in the buckets.
    bool is_accum_sync = false;
    size_t no_of_backward = 0;
    bool rebuild_bucket{true};

    std::vector<size_t> unused_parameters_indices_;
    std::vector<size_t> cached_unused_parameters_;  // cached after first backward
    bool unused_params_cached_ = false;              // true after first BFS completes
    std::unordered_map<std::shared_ptr<OwnTensor::Node>, size_t> gradAcc_variable_map_;
    bool mark_unused = false;
    size_t next_bucket = 0;
    bool finalize_called = false;
    std::mutex mutex_;
    std::shared_ptr<Timer> timer_;
    std::ostream* stream_ = &std::cout;
    Profiler profiler;
    cudaEvent_t compute_sync_event_ = nullptr;  // pre-allocated event for compute→comm syncing


};

std::vector<std::vector<size_t>> bucket_order_decide(
    const std::vector<OwnTensor::Tensor>& params_, 
    const std::optional<std::vector<size_t>>& param_backward_order = std::nullopt, 
    size_t bucket_size_ = 25 * 1024 * 1024 
);













