
#include <cmath>
#include <iostream>
#include <thread>
#include <mpi.h>
#include "DataParallel.hpp"
#include "core/Shape.h"
#include <memory>
#include <algorithm>
#include <cuda_runtime.h>
#include <queue>
#include <unordered_set>
#include "Error_logs.hpp"
#include <cstdlib>
#include "device/DeviceSet.h"


#if __cplusplus < 201703L
    #warning "Requires version greater than c++17"
#endif

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 8, 3)
    #warning "Requires NCCL version greater than 2.8.3 to support some operations"
#endif





void DataParallel::no_sync() {
    grad_sync_ = false;
}

bool DataParallel::sync(){
    return grad_sync_ = true;
}

DataParallel::DataParallel(OwnTensor::nn::Module* module, DDP_Options opts, bool init_sync):
    module_(std::move(module)),
    opts_(std::move(opts)),
    process_group_(opts_.process_group_),
    init_sync_(init_sync),
    parameters_(module_->parameters())

{    
    
    const char* env = std::getenv("DEBUG_MODE_");

    if(env != nullptr){
        if(std::string(env) == "true") DEBUG_MODE_ = true;
	    else if(std::string(env) == "LOG") {
            debug_log = true;
            int r = static_cast<int>(process_group_->get_rank());
            if(r == 0) log_csv = LOG_CSV("logs/performance_log.csv", "logs/trace.json", r);
        }
    }
    if(env != nullptr && std::string(env) == "false"){
        DEBUG_MODE_ = false;
    }

    if(DEBUG_MODE_) timer_ = std::make_shared<Timer>();
    
    CONDITION_ASSERT_TRUE(parameters_.size() == 0,
            "Model parameters cannot be empty"
            );

    CONDITION_ASSERT_TRUE(opts.bucket_ && opts.bucket_size_ == 0,
        "Bucket Size should not be zero if bucket_ is set to true"
    );

    

    bool grad_tensor_exist = std::any_of(
        parameters_.begin(),
        parameters_.end(),
        [&](OwnTensor::Tensor tensor){
            if(tensor.requires_grad()) return true;
            return false;
        }    
    );

    CONDITION_ASSERT(grad_tensor_exist,
        "Module has no tensor that requires gradient"
    );

    //synchronize only if the user wants to....
    //else it is the responsibility of the user,
    // to make sure that all the rank has the,
    // same parameters at the start of forward.
    std::vector<std::vector<size_t>> bucket_indices;
    bucket_indices = bucket_order_decide(parameters_, {} , opts_.bucket_size_);
    // Carve a small leading bucket (in reduction order) so the first all-reduce
    // can fire early. Done before the reverse: it appends the small bucket to
    // the registration-order list, which the reverse then moves to position 0.
    apply_first_bucket_cap(bucket_indices);
    std::reverse(bucket_indices.begin(), bucket_indices.end());
    if(init_sync){
        if(DEBUG_MODE_){
            profiler.nvtx_scoped_range(nvtx3::registered_string("Model Parameters broadcast"), [&](){
                sync_model_parameter(bucket_indices);
                std::cout << "Called Hey_" << std::endl; 
            });
            
        }else{
            sync_model_parameter(bucket_indices);
        }
        
        opts_.process_group_->blockStream();
    }else{
        WARN("Parameter syncing should be done manually");

    }

    // Restore device context after sync operations
    cudaSetDevice(opts_.local_rank_);
    cudaEventCreateWithFlags(&compute_sync_event_, cudaEventDisableTiming);
    //build the preliminary buckets using the model.parameters().
    //This will be used in the first backward. this is not the permanent order,
    //and will be changed after the first backward, using rebuild_buckets()

    //create the buckets using the order of parameters.
    if(!first_forward){
        create_buckets(bucket_indices);
        first_forward = true;
    }  



    //hooks registration.
    //All the parameters present module will be hooked using the register_hook function...
    //These hooks will be fired later, once the backward is finished.
    for(size_t param_index = 0; param_index < parameters_.size(); param_index++){
        auto& parameter = parameters_[param_index];
        auto grad_accumalator = OwnTensor::impl::grad_accumulator(parameter);
        grad_accumalator->register_post_hook(
            [this, param_index](const std::vector<OwnTensor::Tensor>& inputs, const std::vector<OwnTensor::Tensor>& outputs) {
                const auto& tensor = outputs[0]; 
                // parameters_[param_index].grad_view().display();
                if (opts_.is_accum_sync) {
                    this->accum_sync(param_index);
                } else {
                    if(grad_sync_){
                        this->mediator_hook(param_index);
                    }else{}
                }
            }
        );
        gradAcc_variable_map_[grad_accumalator] = param_index;

    }
   
}


void DataParallel::grad_map_flatten_grad(Bucket& bucket){
	

	for(int i = 0; i < bucket.param_variables.size(); i++){
		auto& curr_param = bucket.param_variables[i];
		auto& curr_offset = bucket.offset[i];
		auto& curr_length = bucket.length[i];

		curr_param.unsafeGetTensorImpl()->mutable_grad() = bucket.flatten_gradient.narrow_view(0, curr_offset, curr_length).view(curr_param.shape());
	}
	return;
}

DataParallel& DataParallel::operator=(const DataParallel& other){
    if(this == &other){
        return *this;
    }

    this->module_ = other.module_;
    this->opts_ = other.opts_;
    this->init_sync_ = other.init_sync_;

    return *this;
}


void DataParallel::set_stream(std::ostream& stream){
    stream_ = &stream;
}

void DataParallel::create_buckets(std::vector<std::vector<size_t>>& bucket_indices){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    buckets_.clear();
    for(auto bucket_index = 0; bucket_index < bucket_indices.size(); bucket_index++ ){

        Bucket bucket; 
        auto& curr_bucket = bucket_indices[bucket_index];
        
        CONDITION_ASSERT_TRUE(curr_bucket.empty(),
                         "Bucket inside the specified index is empty to be used"
                        );
        //still have to consider the sparse_gradient. for now not considered (as no compatibility)
        //redeclaration

        auto variables_count = curr_bucket.size();

        bucket.param_variables.resize(variables_count);
        bucket.param_indices.resize(variables_count);
        bucket.length.resize(variables_count);
        bucket.offset.resize(variables_count);
        // bucket.grad_incoming.resize(variables_count); //not sure on the size
        // bucket.grad_outgoing.resize(variables_count); //not sure on the size
        bucket.param_size.resize(variables_count);

        uint64_t offset = 0;
        OwnTensor::TensorOptions opts;
        bool device_set = false;
        bool dtype_set = false;
        
        for(int index = 0; index < curr_bucket.size(); index++){

            auto variable_index = curr_bucket[index];

            if(variable_index < 0){
                //might be due to overflow
                throw std::runtime_error(
                    "Variable index cannot be negative!!"
                );
            }

            if(variable_index >= parameters_.size()){
                throw std::runtime_error(
                    "variable index is greater than the total parameter variables..."
                );
            }


            auto& curr_param = parameters_[variable_index];

            //the above to work will have to make changes in the tensor class.
            //for now brute.

            if(!device_set){
                opts.device = curr_param.device();
                device_set = true;
            }else{
                CONDITION_ASSERT_TRUE(opts.device.device != curr_param.device().device,
                         "Device of every parameter should be the same in a bucket"
                        );
            }

            if(!dtype_set){
                opts.dtype = curr_param.dtype();
                dtype_set = true;
            }else{
                CONDITION_ASSERT_TRUE(opts.dtype != curr_param.dtype(),
                         "Dtype of every parameter should be the same in a bucket"
                        );
            }

            auto length = curr_param.numel();
            //assign 
            bucket.length[index] = length;
            bucket.param_indices[index] = variable_index;
            bucket.param_variables[index] = curr_param;
            bucket.offset[index] = offset;
            bucket.param_size[index] = curr_param.nbytes();


            offset += length;

        }     
        
        bucket.flatten_gradient = OwnTensor::Tensor::empty(OwnTensor::Shape{{static_cast<int64_t>(offset)}}, opts);
    if(!opts_.grad_as_view) create_bucket_gradient_views(bucket);
	else{
		// Call the grad to flatten gradient mapping function

		grad_map_flatten_grad(bucket);
	} 
        bucket.param_indices = std::move(bucket_indices[bucket_index]);
        bucket.bucket_pending = bucket.param_indices.size();
        buckets_.push_back(std::move(bucket));

        
    }
    variable_locator_populator();

    completed_buckets_.resize(buckets_.size());
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
    
}

void DataParallel::apply_first_bucket_cap(std::vector<std::vector<size_t>>& bucket_indices){
    // bucket_indices is in registration order here (the ctor reverses it right
    // after this call). The LAST registration bucket holds the highest-index
    // params (final layers, e.g. ln_f) — those whose grads are ready EARLIEST
    // in backward. Peel a ~first_bucket_size_ tail of them into a standalone
    // bucket appended at the end; after the reverse it becomes reduction-order
    // bucket 0, so its all-reduce launches as soon as the last layers finish
    // instead of waiting for a full bucket_size_ to accumulate.
    if(opts_.first_bucket_size_ <= 0 || bucket_indices.empty()) return;

    std::vector<size_t>& last = bucket_indices.back();
    if(last.size() < 2) return;  // nothing meaningful to split off

    // Walk from the back (highest reg index = earliest ready) until the peeled
    // tail reaches the target size. `split` is where the small bucket starts.
    size_t accum = 0;
    size_t split = last.size();
    for(int i = static_cast<int>(last.size()) - 1; i >= 0; --i){
        accum += parameters_[last[i]].nbytes();
        split = static_cast<size_t>(i);
        if(accum >= static_cast<size_t>(opts_.first_bucket_size_)) break;
    }

    // Require a real split that leaves at least one param in the original bucket.
    if(split == 0 || split >= last.size()) return;

    std::vector<size_t> small(last.begin() + split, last.end());
    last.erase(last.begin() + split, last.end());
    bucket_indices.push_back(std::move(small));
}

void DataParallel::mark_param_ready(size_t variable_index){
    // TensorKey key(parameter);
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    //get the bucket locator for the parameter
    auto& bucket_locator = variable_locator_[variable_index];

    auto parameter = parameters_[variable_index];

    auto& bucket_index = bucket_locator.bucket_index;

    //the actual bucket which contains the parameter
    auto& bucket = buckets_[bucket_index];

    auto& bucket_param_index = bucket_locator.inter_variable_index;


    {
        if(parameter.requires_grad()){
            if(!opts_.grad_as_view) {
                // Non-view mode: copy gradient from parameter into bucket view.
                auto inter_variable_index = bucket_locator.inter_variable_index;
                auto& bucket_in_view = bucket.grad_incoming;
                auto& bucket_in_view_pos = bucket_in_view[inter_variable_index];

                // After the first step, finalize_backward's set_grad() points the
                // param's grad AT this very bucket view (set_grad shares storage),
                // zero_grad fills it in place, and AccumulateGrad does *grad += ...
                // in place — so from step 2 on the param grad and the bucket view
                // alias the SAME memory and this becomes a copy onto itself. Skip
                // it: that removes ~one full model's worth of D2D bytes plus a
                // per-param async-memcpy launch every step, work that otherwise
                // sits on the compute stream and (on the last micro-step) delays
                // the all-reduce that waits on it. The copy still runs on step 1,
                // or any time the grad buffer is reallocated (pointers differ).
                void* grad_ptr   = parameters_[variable_index].grad();
                void* bucket_ptr = bucket_in_view_pos.data();
                if(grad_ptr != bucket_ptr){
                    OwnTensor::device::copy_memory(
                        bucket_ptr,
                        bucket_in_view_pos.device().device,
                        grad_ptr,
                        parameters_[variable_index].device().device,
                        parameters_[variable_index].nbytes()
                    );
                }
            }
            // When grad_as_view = true: gradient IS a view of flatten_gradient.
            // The autograd engine wrote directly into the bucket. No copy needed.

            if(bucket.bucket_pending > 0){
                bucket.bucket_pending--;
            }
        }
    }

    //now check for the total bucket completion.
    //if the bucket if fully calculated, then it is time for the bucket to be passed,
    //into the communication hook for all reduce.
    if(bucket.bucket_pending == 0){
        // bucket is ready
        mark_bucket_ready(bucket_index);
    }

    

    if(next_bucket == buckets_.size() && !finalize_called){
        finalize_called = true;
        OwnTensor::autograd::queue_call_back(
            [&](){
                this->finalize_backward();
            }
        );   
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}

void DataParallel::mark_bucket_ready(size_t bucket_index){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    if(bucket_index > next_bucket){
        if(debug_log){
            if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
        }
        return;
    }

    // populate_bucket_gradient(bucket_index);

    for(next_bucket; next_bucket < buckets_.size(); next_bucket++){
        Bucket& curr_bucket = buckets_[next_bucket];
        if(curr_bucket.bucket_pending == 0){
            completed_buckets_[next_bucket] = true;
            all_reduce_grad_sync(curr_bucket);
        }else{
            break;
        }
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}






void DataParallel::all_reduce_grad_sync(Bucket& bucket){
    // std::cout << buckets_.size() << std::endl;
    
    // Sync: ensure all gradient copies on computation stream are visible to comm stream
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    cudaEventRecord(compute_sync_event_, OwnTensor::cuda::getCurrentStream());
    cudaStreamWaitEvent(process_group_->get_stream(), compute_sync_event_, 0);
    void *flatten_gradient_ptr = bucket.flatten_gradient.data();
    if(DEBUG_MODE_){
        mark_comm_start(timer_);
    }
    // Recycle this bucket's Work (and its CUDA event) across steps: on the
    // first backward bucket.work_obj_ is null so a Work is created; every
    // subsequent step re-arms the same object instead of create/destroying a
    // cudaEvent per bucket per step in the hook path.
    std::shared_ptr<Work> reuse_work = bucket.work_obj_;
    bucket.work_obj_ = CONDITION_CHECK_FUNCTION(
        DEBUG_MODE_,
        (profiler.nvtx_scoped_range(nvtx3::registered_string("All Reduce Kernel"), [&]() -> std::shared_ptr<Work>{
            return process_group_->all_reduce_async(flatten_gradient_ptr,
                                                        flatten_gradient_ptr,
                                                        bucket.flatten_gradient.numel(),
                                                        bucket.flatten_gradient.dtype(),
                                                        avg,
                                                        false,
                                                        reuse_work
                                                    );
        })),
        (process_group_->all_reduce_async(flatten_gradient_ptr,
                                        flatten_gradient_ptr,
                                        bucket.flatten_gradient.numel(),
                                        bucket.flatten_gradient.dtype(),
                                        avg,
                                        false,
                                        reuse_work
        ))
    );
    if(DEBUG_MODE_){
        mark_comm_end(timer_);
    }

    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }

}

void DataParallel::finalize_backward(){
    //the aim is to copy the respective gradient of the parameter to the respective parameters,
    //gradient buffer. For this we will use the gradient_view_out, which we will get the gradient that is being calculated and,
    //move that to the actual parameters gradient buffer.
    //This works even in the case of grad accumulation, because each gpu will see a mini batch.
    //So, grad accumulation is done using the all reduce and not in for the upcoming batches of the same gpu.
    //NOTE: 
    //gradient copying should be done only in the finalize backward and this function shoudl be called only from the,
    //last parameter calculation.
    //if this function is not called through out the training run, then there is a possible error in the model 
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    finalize_called = true;
    // for(int i = buckets_.size() - 1; i >= 0; i--){
    //     // bucket.flatten_gradient.display();
    //     auto& bucket = buckets_[i];
    //     bucket.work_obj_->wait();
    // }
    for(int i = buckets_.size() - 1; i >= 0; i--){
        
        auto& bucket = buckets_[i];
        bucket.work_obj_->wait();
        auto& flattened_gradient = bucket.flatten_gradient;
        auto& bucket_views_out = bucket.grad_outgoing;
        void* grad_ptr = flattened_gradient.data();
        if(!opts_.grad_as_view){
            // std::cout << "CALLED OPTS GRAD VIEW" << std::endl;
            for(auto i = 0; i < bucket_views_out.size(); i++ ){
                        size_t curr_grad_offset = bucket.offset[i];
                        auto& grad = bucket_views_out[i];

                        auto& param = parameters_[bucket.param_indices[i]];
                        param.set_grad(grad);
                }
        }
    }

    // Ensure all NCCL all-reduce operations have completed on the GPU before
    // returning to user code. work_obj_->wait() only inserts a GPU-side dependency
    // (cudaStreamWaitEvent) which is non-blocking to the CPU. Without this sync,
    // subsequent CPU-side reads of gradient data (display, norm computation) can
    // race against still-in-flight NCCL. This does NOT kill backward-NCCL overlap —
    // that overlap already happened during backward. This only blocks between
    // backward completion and the optimizer step, which is minimal latency.

    if(DEBUG_MODE_){
        mark_backward_end(timer_);
        TIMER_LOG(timer_, *stream_, number_of_iterations_);
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
    
}      


void DataParallel::bucket_data(){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    uint64_t bucket_size_bytes = (uint64_t)25 * 1024 * 1024; //bucket size will be in bytes for better accuracy

    std::vector<std::vector<size_t>> bucket_to_variable_info;

    uint64_t running_bucket_size_bytes = 0;
    std::vector<size_t> variable_indices;
    size_t variable_index = 0;
    for(size_t variable_index = 0; variable_index < parameters_.size(); variable_index++){

        uint64_t curr_param_size = parameters_[variable_index].nbytes();

        uint64_t updated_size = curr_param_size + running_bucket_size_bytes;

        /*
         if including the current parameter to the bucket exceeds the bucket size,
         then add the indices to the bucket_to_variable_info.
         breaking tensors are not allowed.
        */
        if(updated_size > bucket_size_bytes){
            bucket_to_variable_info.push_back(variable_indices);

            //reset variables
            variable_indices.clear();
            running_bucket_size_bytes = 0;
        }
        //should check for memory usage for 
        //  -updated_size
        //  -running bucket size
        //  -curr_param_size
        // as load, store is not efficient

        curr_param_size;
        variable_indices.push_back(variable_index);
    }
    if(!variable_indices.empty()){
        bucket_to_variable_info.push_back(variable_indices);
    }
    create_buckets(bucket_to_variable_info);
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}


void DataParallel::variable_locator_populator(){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    for(auto bucket_index = 0; bucket_index < buckets_.size(); bucket_index++){
        Bucket bucket = buckets_[bucket_index];
        for(auto variable_index = 0; variable_index <  bucket.param_variables.size(); variable_index++){
            auto model_param_index = bucket.param_indices[variable_index];
            auto curr_tensor = bucket.param_variables[variable_index];
            param_bucket_locator variable_struct(bucket_index, variable_index);
            variable_locator_[model_param_index] = variable_struct;
        }
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}

void DataParallel::create_bucket_gradient_views(Bucket& bucket){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }

    for(auto i = 0; i <bucket.param_indices.size(); i++){

        //get the offset and length
        //view the flatten gradient from the offset to the length and put it into the bucket_views_in.
        //this helps us ensure that both the flatten gradient, bucket_views_in and bucket_views_out looks at the same memory.
        //changes made to the bucket_views_in will be visible to all the three.
        auto& offset = bucket.offset[i];
        auto& length = bucket.length[i];

        auto& shape = bucket.param_variables[i].shape();
        auto& grad = bucket.flatten_gradient;

        bucket.grad_incoming.push_back(grad.narrow_view(0 /*dim = 0 as only one axis*/,
                                             offset,
                                             length).view(shape));
    }
    bucket.grad_outgoing  = bucket.grad_incoming;
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}






std::vector<std::vector<size_t>> bucket_order_decide(const std::vector<OwnTensor::Tensor>& params_,const std::optional<std::vector<size_t>>& param_backward_order, size_t bucket_size_ ){
    BucketInit bucket(bucket_size_);

    CONDITION_ASSERT_TRUE(bucket_size_ == 0,
                    "bucket_size must be greater than 0"
                        );

    //need to see which is the optimal
    std::vector<std::pair<std::vector<size_t>,size_t>> final_order_buckets;
    //two maps: 
    // - one to store the indexes. like, for each bucket_key <device_type, dtype> the map will store a vector containing the indices.
    // - This then will be used in the map for the bucket accumulator
    std::unordered_map<Key, std::vector<size_t>, KeyHash> bucket_indices;
    //to store the bucket indices
    std::unordered_map<Key, BucketInit, KeyHash> buckets;

    for(int i = 0; i < params_.size(); i++ ){
        
        int index = (!param_backward_order.has_value()) ? i : param_backward_order.value()[i];

        OwnTensor::Tensor curr_tensor = params_[index];

        OwnTensor::Dtype tensor_dtype = curr_tensor.dtype();
        OwnTensor::DeviceIndex tensor_device = curr_tensor.device();

        //check if the key exist in the bucket_indices, to see if it is the first parameter.
        //then based on that retrive the current bucket that is being populated with
        //use that bucket to add the data
        Key key_for_bucket_indices(tensor_dtype, tensor_device);
        
        if(bucket_indices.find(key_for_bucket_indices) == bucket_indices.end()){
            bucket_indices[key_for_bucket_indices] = std::vector<size_t>();
        }

        size_t curr_bucket_index;

        if(!bucket_indices[key_for_bucket_indices].empty())  curr_bucket_index = bucket_indices[key_for_bucket_indices].back();
        else curr_bucket_index = 0;
        //size checks

        Key bucket_key(tensor_dtype, tensor_device, curr_bucket_index);

        if(buckets.find(bucket_key) == buckets.end()){
            buckets[bucket_key] = BucketInit(bucket_size_);
        }

        //this is the current bucket we have to push into (if we have space in it).
        BucketInit& curr_bucket = buckets[bucket_key];

        if(curr_bucket.bucket_size + curr_tensor.nbytes() <= bucket_size_){
            curr_bucket.tensor_indices.push_back(index);
            curr_bucket.bucket_size += curr_tensor.nbytes();
        }else{
            //create a new BucketInit
            //append the index. put the new bucket into the bucket indices.
            //append the tensor to the new bucketInit and append the size.
            
            // final_order_buckets.emplace_back(std::pair(std::move(curr_bucket.tensor_indices), curr_bucket.bucket_size)); 
            //the above one is less optimal compared to the below one, as from c++17, the compiler uses CTAD(Class Template Argument Deduction), which inside a emplace_back will directly deduct the type
            if (!curr_bucket.tensor_indices.empty()) {
                final_order_buckets.emplace_back(
                    std::move(curr_bucket.tensor_indices),
                    curr_bucket.bucket_size
                );
            }
            // curr_bucket.tensor_indices.clear(); 
            // curr_bucket.bucket_size = 0;

            BucketInit new_bucket(bucket_size_);
            new_bucket.tensor_indices.push_back(index);
            size_t new_bucket_index = curr_bucket_index + 1;
            bucket_indices[key_for_bucket_indices].push_back(new_bucket_index);

            Key new_key(tensor_dtype, tensor_device, new_bucket_index);
            buckets[new_key] = new_bucket;
        }   

        
        
    }

    for (auto& [key, bucket] : buckets) {
        if (!bucket.tensor_indices.empty()) {
            final_order_buckets.emplace_back(
                std::move(bucket.tensor_indices),
                bucket.bucket_size
            );
        }
    }
    // ALWAYS sort buckets by their first parameter index. final_order_buckets is
    // assembled by iterating an unordered_map, so without this it comes out in
    // hash order — an arbitrary reduction order. The ctor then reverses this list
    // to get backward (gradient-ready) order, and mark_bucket_ready launches
    // all-reduces strictly in index order. If bucket 0 (hash-arbitrary) happens to
    // hold a parameter whose gradient is produced last (e.g. wte under weight
    // tying), the in-order gate blocks EVERY all-reduce until backward finishes →
    // zero comm/backward overlap, a fully exposed comm tail. Sorting by first param
    // index restores registration order (params were added to buckets in ascending
    // index), so post-reverse bucket 0 = last-registered = first ready in backward,
    // and reduces stream out during backward. (Previously this only ran when
    // param_backward_order was set, i.e. never on the initial build.)
    std::sort(
        final_order_buckets.begin(),
        final_order_buckets.end(),
        [&](const auto& a, const auto& b){
            if (a.first.empty()) return false;
            if (b.first.empty()) return true;
            return a.first[0] < b.first[0];
        }
    );

    std::vector<std::vector<size_t>> result;
    for(const auto& pair: final_order_buckets){
        result.emplace_back(std::move(pair.first));
    }
    return result; 
}

void DataParallel::mediator_hook(size_t variable_index){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    auto& variable_locator = variable_locator_[variable_index];
    auto& curr_bucket = buckets_[variable_locator.bucket_index];
    size_t variable_bucket_index = variable_locator.inter_variable_index;

    if(!mark_unused){
        mark_unused = true;
        for(auto& index : unused_parameters_indices_){
            auto variable_loc = variable_locator_[index];
            auto bucket_index = variable_loc.bucket_index;

            mark_param_ready(index);
        }
    }

    if(rebuild_bucket){
        rebuilt_params_order_.emplace_back(parameters_[variable_index]);
        rebuilt_indices_order_.emplace_back(variable_index);
    }
    mark_param_ready(variable_index);
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}


void DataParallel::search_for_unused(const std::vector<OwnTensor::Tensor>& outputs){
    if(debug_log){
        log_csv.start_logging(__FUNCTION__);
    }
    std::queue<std::shared_ptr<OwnTensor::Node>> order;
    std::unordered_set<std::shared_ptr<OwnTensor::Node>> seen;

    for(auto& output : outputs){
        auto grad_fn = output.grad_fn();

        if(grad_fn){
            order.push(grad_fn);
        }
    }


    while(!order.empty()){
        auto& top = order.front();
        order.pop();

        for(const auto& edge : top->next_edges()){
            if(auto next = edge.function){
                if(seen.insert(next).second){
                    order.push(next);
                }
            }
        }
    }

    for(const auto& [gradAccNode, param_index] : gradAcc_variable_map_){
        if(seen.find(gradAccNode) == seen.end()){
            unused_parameters_indices_.push_back(param_index);
        }
    }

    if(unused_parameters_indices_.empty()){
    }

    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }


}

template <typename T>
std::vector<OwnTensor::Tensor> vectorize_output(T& output_obj){

    //for now just a single vector function!!
    //but have to provide function to check which type of function it is and
    //recurse based on it
    return {output_obj};
}

OwnTensor::Tensor DataParallel::forward(const OwnTensor::Tensor& input) {
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    pre_forward();
    auto output = module_->forward(input);
    pre_backward(vectorize_output(output));
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
    return output;
}

void DataParallel::pre_backward(const std::vector<OwnTensor::Tensor>& outputs){
    if(DEBUG_MODE_){
        mark_forward_end(timer_);
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    no_of_backward++;
    mark_unused = false;
    finalize_called = false;
    next_bucket = 0;
    first_backward = true;
    for(auto& bucket : buckets_){
        bucket.bucket_pending = bucket.param_indices.size();
        // if(opts_.grad_as_view){
        //     // bucket.flatten_gradient.fill(0.0f);
        // }
    }

    if(opts_.grad_as_view){
        for(auto& bucket : buckets_){
            // OwnTensor::device::set_memory(bucket.flatten_gradient.data(), bucket.flatten_gradient.device().device, 0, bucket.flatten_gradient.nbytes());  // 1 memset for entire bucket
            cudaMemsetAsync(bucket.flatten_gradient.data(), 0.0f, bucket.flatten_gradient.nbytes(), process_group_->get_stream());
            // cudaDeviceSynchronize();
        }
    }
    

    
    completed_buckets_.clear();
    completed_buckets_.resize(buckets_.size());

    if(!opts_.static_graph){
        // The unused-parameter set is a property of the graph topology, which is
        // identical across iterations for a static model (e.g. GPT-2). Running
        // the full autograd-graph BFS every backward is pure DDP-only CPU cost on
        // the critical path between forward and backward. Do it once, cache the
        // result, and reuse it thereafter. (Assumes the graph is stable after the
        // first backward — true for static models; use with_static_graph(true) to
        // skip unused detection entirely.)
        if(unused_params_cached_){
            unused_parameters_indices_ = cached_unused_parameters_;
        }else{
            unused_parameters_indices_.clear();
            search_for_unused(outputs);
            cached_unused_parameters_ = unused_parameters_indices_;
            unused_params_cached_ = true;
        }
    }
    if(DEBUG_MODE_){
        mark_backward_start(timer_);
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
    
}

void DataParallel::verify_params(){

     //compare:
     // stride, shape, bytes, data (optional, because it might cause the devices to synchronize), any other meta data.
     CONDITION_ASSERT_TRUE(parameters_.empty(),
                             "Parameters are empty"
                             );

     /*
     SHAPE
     */
     std::vector<int64_t> shape_vector;
     for(auto& param : params_){
        auto& shape = param.shape();
        shape_vector.insert(shape_vector.end(), shape.dims.begin(), shape.dims.end());
     }

     
     OwnTensor::Tensor send_tensor = OwnTensor::Tensor({{static_cast<int64_t>(shape_vector.size())}}, OwnTensor::TensorOptions()
                                                                                                .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, process_group_->get_rank()))
                                                                                                .with_dtype(OwnTensor::Dtype::Int64)
                                                                                            );
     send_tensor.set_data(shape_vector);
     OwnTensor::Tensor recv_tensor = OwnTensor::Tensor::empty({{static_cast<int64_t>(shape_vector.size())}}, OwnTensor::TensorOptions()
                                                                                                .with_device(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, process_group_->get_rank()))
                                                                                                .with_dtype(OwnTensor::Dtype::Int64)
                                                                                            ); 
     PG_CHECK(process_group_->broadcast(send_tensor.data(), recv_tensor.data(), shape_vector.size(), send_tensor.dtype(), 0, true));

     auto equal_tensor = (send_tensor == recv_tensor).to_cpu();
     bool intermediate_buffer = true;                          
     auto* data_ptr = static_cast<bool*>(equal_tensor.data());                                                   
     auto params_shape_equal = std::any_of(
        data_ptr,
        data_ptr + equal_tensor.numel(),
        [&intermediate_buffer](bool val){
            return intermediate_buffer & val;
        }
     );

     CONDITION_ASSERT(
        intermediate_buffer,
        "Param Shapes are not matching"
     );

}



void DataParallel::sync_model_parameter(std::vector<std::vector<size_t>>& bucket_indices){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    if(!init_sync_){
        if(debug_log){
            if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
        }
        return;
    }
    CONDITION_ASSERT_TRUE(parameters_.empty(),
                    "No paramters to sync!!"
                        );


    for(auto bucket : bucket_indices){
        // the vector bucket contains the indices of the tensor for the broadcast
        
        std::vector<OwnTensor::Tensor> bucket_tensor;
        size_t total_elements = 0;
        for(auto index : bucket){
            bucket_tensor.push_back(parameters_[index]);
            total_elements += parameters_[index].numel();
        }
        
        OwnTensor::TensorOptions opts;
        opts.device = OwnTensor::DeviceIndex(parameters_[bucket[0]].opts().device.device, process_group_->get_rank());
        opts.dtype = parameters_[bucket[0]].dtype();
        OwnTensor::Tensor output_tensor = OwnTensor::Tensor({{1, static_cast<int64_t>(total_elements)}}, opts);
        
        result_t result = process_group_->broadcast_coalesced(bucket_tensor,  
                                                              output_tensor
                                                            );
                    
        
        
        size_t running_index = 0;
        for(auto index: bucket){
            OwnTensor::Tensor temp_ = output_tensor.slice_inplace(running_index, parameters_[index].numel());
            parameters_[index].copy_(temp_);
            running_index += parameters_[index].numel();
        }
        
    }
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}

void DataParallel::pre_forward(){
    // Rebuild buckets once, after the first synced backward has revealed the true
    // gradient-ready order, so subsequent steps overlap all-reduce with backward.
    // No-op until that order is known and after it has fired once (guarded inside
    // by rebuild_bucket + rebuilt_indices_order_ availability).
    if(rebuild_bucket){
        rebuild_buckets();
    }
    if(debug_log) {
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    number_of_iterations_ ++;
    if(DEBUG_MODE_){
        mark_forward_start(timer_);
    }
    if(debug_log) if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
}


void DataParallel::accum_sync(size_t variable_index){
    //for now kept the steps inside the tensor (will look into a better position)
    // auto curr_tensor = parameters_[variable_index];

    // auto curr_steps = curr_tensor.get_steps();

    // if(opts_.is_accum_sync){
    //     CONDITION_ASSERT(opts_.grad_accum_steps == 0,
    //                      "No of steps is \"0\""
    //                     );
    //     if(curr_steps % opts_.grad_accum_steps == 0){
    //         // means we have to use all reduce to sync the gradients.
    //         mark_param_ready(variable_index);
    //         curr_tensor.set_steps(curr_steps + 1);
    //     }else{
    //         curr_tensor.set_steps(curr_steps + 1);
    //     }
    // }
}


void DataParallel::rebuild_buckets(){
    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.start_logging(__FUNCTION__);
    }
    if(!rebuild_bucket){
        if(debug_log){
            if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
        }
        return;
    }
    // Need at least one synced backward to know the gradient-ready order. This is
    // called from pre_forward on every iteration, so it no-ops until the first
    // backward has populated rebuilt_indices_order_ (and after it fires once,
    // rebuild_bucket is false so it never runs again).
    if(rebuilt_indices_order_.empty()){
        if(debug_log){
            if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
        }
        return;
    }
    rebuild_bucket = false;

    // rebuilt_indices_order_ holds parameter indices in the order their gradients
    // became READY during the first synced backward (i.e. reverse-of-forward
    // execution order). Pack them into size-capped buckets IN THAT ORDER so that
    // reduction bucket 0 holds the params whose grads are produced FIRST. The
    // in-order launch gate in mark_bucket_ready then fires bucket 0 early and
    // pipelines the rest, so all-reduces overlap backward instead of all firing
    // in one burst at the end.
    //
    // The previous implementation routed through bucket_order_decide(), whose
    // trailing sort-by-ascending-param-index discarded the ready order; combined
    // with the ctor's reverse of *registration* order that placed the last-ready
    // embeddings (wte/wpe — registered last in this model) into bucket 0, which
    // stalled every all-reduce until the end of backward (zero overlap).
    std::vector<size_t> ready_order = rebuilt_indices_order_;

    // Safety: append any parameter never observed in backward (e.g. unused
    // params) so every gradient is still bucketed and reduced. Their relative
    // order does not affect overlap; they go last.
    {
        std::vector<bool> seen(parameters_.size(), false);
        for(size_t idx : ready_order)
            if(idx < parameters_.size()) seen[idx] = true;
        for(size_t idx = 0; idx < parameters_.size(); ++idx)
            if(!seen[idx]) ready_order.push_back(idx);
    }

    // Greedy size-capped packing that PRESERVES ready order. The first bucket is
    // kept small (first_bucket_size_) so the very first all-reduce launches ASAP;
    // a single param larger than the cap (e.g. the 154 MB tied wte) lands in its
    // own bucket.
    const size_t cap = static_cast<size_t>(opts_.bucket_size_);
    const size_t first_cap = (opts_.first_bucket_size_ > 0)
                                 ? static_cast<size_t>(opts_.first_bucket_size_)
                                 : cap;
    size_t active_cap = first_cap;

    std::vector<std::vector<size_t>> bucket_indices;
    std::vector<size_t> current;
    size_t current_bytes = 0;
    for(size_t idx : ready_order){
        size_t nbytes = parameters_[idx].nbytes();
        if(!current.empty() && current_bytes + nbytes > active_cap){
            bucket_indices.push_back(std::move(current));
            current.clear();
            current_bytes = 0;
            active_cap = cap;  // only the leading bucket uses the smaller cap
        }
        current.push_back(idx);
        current_bytes += nbytes;
    }
    if(!current.empty()) bucket_indices.push_back(std::move(current));

    buckets_.clear();
    variable_locator_.clear();
    rebuilt_params_order_.clear();
    rebuilt_indices_order_.clear();

    create_buckets(bucket_indices);

    if(debug_log){
        if(process_group_->get_rank() == 0) log_csv.stop_logging(__FUNCTION__);
    }
}


void DataParallel::naive_grad_sync(){

    int64_t total_elements = std::accumulate(   
                                parameters_.begin(),
                                parameters_.end(),
                                0,
                                [&](int64_t total, const OwnTensor::Tensor& tensor){
                                    return total + tensor.numel();
                                }
                            );
    std::vector<OwnTensor::Tensor> grad_;

    for(auto& value : parameters_){
        grad_.emplace_back(value.grad_view());
    }

    auto grad_flatten = OwnTensor::Tensor::flatten_concat(grad_);


    result_t result = process_group_->all_reduce(grad_flatten.data(), grad_flatten.data(), grad_flatten.numel(), grad_flatten.dtype(), avg, true);
    int64_t running_index = 0;
    for(auto& grad : grad_){
        grad = grad_flatten.slice(running_index, grad.numel()).view(grad.shape());
        running_index += grad.numel();
    }

}



