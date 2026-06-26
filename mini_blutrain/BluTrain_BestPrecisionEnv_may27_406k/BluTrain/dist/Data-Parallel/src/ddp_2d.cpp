#include "ddp_2d.hpp"
#include "device/DeviceTransfer.h"
#include "device/DeviceSet.h"
#include "core/TensorDispatch.h"
#include "core/Shape.h"

#include <iostream>
#include <cassert>

// ===================================================================
// Constructor
// ===================================================================

DataParallel2D::DataParallel2D(
    OwnTensor::dnn::DModule* module,
    DDP2D_Options opts,
    bool init_sync
)
    : module_(module),
      opts_(std::move(opts)),
      dp_pg_(opts_.dp_pg),
      tp_pg_(opts_.tp_pg),
      init_sync_(init_sync)
{
    CONDITION_ASSERT_TRUE(module_ == nullptr, "Module cannot be null");
    CONDITION_ASSERT_TRUE(dp_pg_ == nullptr,  "DP process group cannot be null");
    CONDITION_ASSERT_TRUE(tp_pg_ == nullptr,  "TP process group cannot be null");

    // Collect all DTensor parameters from the DModule tree
    all_params_ = module_->parameters();

    CONDITION_ASSERT_TRUE(all_params_.empty(), "Model has no parameters");

    // Build raw tensor vector and classify sharded vs replicated
    categorize_parameters();

    // Set device context
    cudaSetDevice(opts_.local_rank);
    cudaEventCreateWithFlags(&compute_sync_event_, cudaEventDisableTiming);

    // Build initial buckets if bucketing is enabled
    if (opts_.bucket_enabled) {
        create_buckets();
    }

    // Broadcast parameters across DP replicas so every replica starts identical
    if (init_sync_) {
        broadcast_parameters();
        dp_pg_->blockStream();
    }

    // Register autograd hooks for bucket-based gradient sync
    if (opts_.bucket_enabled) {
        register_hooks();
    }
}

DataParallel2D::~DataParallel2D() {
    if (compute_sync_event_) {
        cudaEventDestroy(compute_sync_event_);
        compute_sync_event_ = nullptr;
    }
}

// ===================================================================
// Parameter classification
// ===================================================================

void DataParallel2D::categorize_parameters() {
    raw_tensors_.clear();
    sharded_params_.clear();
    replicated_params_.clear();
    is_replicated_.clear();

    raw_tensors_.reserve(all_params_.size());
    is_replicated_.reserve(all_params_.size());

    for (auto* dt : all_params_) {
        OwnTensor::Tensor& t = dt->mutable_tensor();
        raw_tensors_.push_back(t);

        bool replicated = dt->get_layout().is_replicated();
        is_replicated_.push_back(replicated);

        if (replicated) {
            replicated_params_.push_back(dt);
        } else {
            sharded_params_.push_back(dt);
        }
    }
}

// ===================================================================
// Parameter access
// ===================================================================

std::vector<DTensor*> DataParallel2D::parameters()            { return all_params_; }
std::vector<DTensor*> DataParallel2D::sharded_parameters()    { return sharded_params_; }
std::vector<DTensor*> DataParallel2D::replicated_parameters() { return replicated_params_; }

// ===================================================================
// Forward
// ===================================================================

DTensor DataParallel2D::forward(DTensor& input) {
    pre_forward();
    DTensor output = module_->forward(input);

    // Prepare backward state (unused-param detection, bucket reset)
    std::vector<OwnTensor::Tensor> outputs_vec = { output.mutable_tensor() };
    pre_backward(outputs_vec);

    return output;
}

// ===================================================================
// Gradient synchronization -- two-phase
// ===================================================================

void DataParallel2D::sync_gradients() {
    // Phase 1: TP sync for replicated parameters
    sync_tp_replicated_grads();

    // Phase 2: DDP sync across DP replicas for ALL parameters
    sync_dp_grads();
}

void DataParallel2D::sync_tp_replicated_grads() {
    if (opts_.tp_world_size <= 1) return;

    for (auto* dt : replicated_params_) {
        OwnTensor::Tensor& t = dt->mutable_tensor();
        if (!t.has_grad()) continue;

        OwnTensor::Tensor g = t.grad_view();
        if (!g.is_valid() || g.numel() == 0) continue;

        // Sum partial gradients from TP ranks, then average
        tp_pg_->all_reduce(
            g.data(), g.data(),
            g.numel(), g.dtype(),
            avg, true
        );
    }
}

void DataParallel2D::sync_dp_grads() {
    if (opts_.dp_world_size <= 1) return;

    for (auto* dt : all_params_) {
        OwnTensor::Tensor& t = dt->mutable_tensor();
        if (!t.has_grad()) continue;

        OwnTensor::Tensor g = t.grad_view();
        if (!g.is_valid() || g.numel() == 0) continue;

        // Average gradients across DP replicas
        dp_pg_->all_reduce(
            g.data(), g.data(),
            g.numel(), g.dtype(),
            avg, true
        );
    }
}

// ===================================================================
// Parameter broadcasting (initialization)
// ===================================================================

void DataParallel2D::broadcast_parameters() {
    if (opts_.dp_world_size <= 1) return;

    for (auto* dt : all_params_) {
        OwnTensor::Tensor& t = dt->mutable_tensor();
        dp_pg_->broadcast(
            t.data(), t.data(),
            t.numel(), t.dtype(),
            0,    // root = dp_rank 0
            true  // sync
        );
    }
}

// ===================================================================
// No-sync mode
// ===================================================================

void DataParallel2D::no_sync() {
    grad_sync_enabled_ = false;
}

bool DataParallel2D::enable_sync() {
    return grad_sync_enabled_ = true;
}

// ===================================================================
// Zero gradients
// ===================================================================

void DataParallel2D::zero_grad() {
    module_->zero_grad();
}

// ===================================================================
// 2D-aware gradient norm clipping
//
// 1. Compute local grad norm^2 per parameter
// 2. All-reduce across TP group (sum -- sharded params hold partial norms)
// 3. All-reduce across DP group (avg -- DP replicas should agree)
// 4. Clip
// ===================================================================

float DataParallel2D::clip_grad_norm_2d(float max_norm, float norm_type) {
    bool is_inf = std::isinf(norm_type);

    float local_norm_stat = 0.0f;

    for (size_t i = 0; i < all_params_.size(); i++) {
        OwnTensor::Tensor& t = all_params_[i]->mutable_tensor();
        if (!t.has_grad()) continue;

        OwnTensor::Tensor g = t.grad_view();
        if (!g.is_valid() || g.numel() == 0) continue;

        // Copy gradient to CPU for norm computation
        OwnTensor::Tensor g_cpu = g.to_cpu();
        float* data = static_cast<float*>(g_cpu.data());
        int64_t n = g_cpu.numel();

        if (is_inf) {
            for (int64_t j = 0; j < n; j++) {
                float abs_val = std::fabs(data[j]);
                if (abs_val > local_norm_stat) local_norm_stat = abs_val;
            }
        } else {
            for (int64_t j = 0; j < n; j++) {
                local_norm_stat += data[j] * data[j];
            }
        }
    }

    // Step 2: All-reduce across TP group
    // For sharded params, each TP rank holds a different shard's norm.
    // Summing gives the full parameter norm. For replicated params,
    // all TP ranks computed the same norm, but avg handles it.
    float tp_reduced = local_norm_stat;

    // Use a small GPU buffer for the all-reduce
    float* d_norm = nullptr;
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemcpy(d_norm, &tp_reduced, sizeof(float), cudaMemcpyHostToDevice);

    if (opts_.tp_world_size > 1) {
        tp_pg_->all_reduce(
            d_norm, d_norm, 1,
            OwnTensor::Dtype::Float32,
            is_inf ? max : sum,
            true
        );
    }

    // Step 3: All-reduce across DP group (replicas should agree; avg as safety)
    if (opts_.dp_world_size > 1) {
        dp_pg_->all_reduce(
            d_norm, d_norm, 1,
            OwnTensor::Dtype::Float32,
            is_inf ? max : avg,
            true
        );
    }

    float global_norm_stat;
    cudaMemcpy(&global_norm_stat, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_norm);

    float total_norm = is_inf ? global_norm_stat : std::sqrt(global_norm_stat);

    // Step 4: Clip gradients
    float clip_coef = max_norm / (total_norm + 1e-6f);
    if (clip_coef < 1.0f) {
        for (auto* dt : all_params_) {
            OwnTensor::Tensor& t = dt->mutable_tensor();
            if (!t.has_grad()) continue;

            OwnTensor::Tensor g = t.grad_view();
            if (!g.is_valid() || g.numel() == 0) continue;

            // Scale in-place: g *= clip_coef
            g = g*(clip_coef);
        }
    }

    return total_norm;
}

// ===================================================================
// Bucket system
// ===================================================================

std::vector<std::vector<size_t>> DataParallel2D::decide_bucket_order(
    const std::vector<OwnTensor::Tensor>& params,
    const std::optional<std::vector<size_t>>& backward_order)
{
    size_t bucket_cap = opts_.bucket_size;
    std::vector<std::vector<size_t>> result;
    std::vector<size_t> current_bucket;
    size_t running_bytes = 0;

    for (size_t i = 0; i < params.size(); i++) {
        size_t idx = backward_order.has_value() ? backward_order.value()[i] : i;
        size_t param_bytes = params[idx].nbytes();

        if (running_bytes + param_bytes > bucket_cap && !current_bucket.empty()) {
            result.push_back(std::move(current_bucket));
            current_bucket.clear();
            running_bytes = 0;
        }

        current_bucket.push_back(idx);
        running_bytes += param_bytes;
    }

    if (!current_bucket.empty()) {
        result.push_back(std::move(current_bucket));
    }

    // Reverse so that last-backward params are in first bucket (fired first)
    std::reverse(result.begin(), result.end());
    return result;
}

void DataParallel2D::create_buckets() {
    buckets_.clear();
    variable_locator_.clear();

    auto bucket_indices = decide_bucket_order(raw_tensors_);

    for (size_t bi = 0; bi < bucket_indices.size(); bi++) {
        Bucket2D bucket;
        auto& indices = bucket_indices[bi];

        size_t var_count = indices.size();
        bucket.param_dtensors.resize(var_count);
        bucket.param_tensors.resize(var_count);
        bucket.param_indices.resize(var_count);
        bucket.length.resize(var_count);
        bucket.offset.resize(var_count);
        bucket.param_size.resize(var_count);

        uint64_t running_offset = 0;
        OwnTensor::TensorOptions t_opts;
        bool device_set = false;
        bool dtype_set  = false;

        for (size_t j = 0; j < var_count; j++) {
            size_t var_idx = indices[j];
            auto& param = raw_tensors_[var_idx];

            if (!device_set) { t_opts.device = param.device(); device_set = true; }
            if (!dtype_set)  { t_opts.dtype  = param.dtype();  dtype_set  = true; }

            bucket.length[j]        = param.numel();
            bucket.param_indices[j] = var_idx;
            bucket.param_tensors[j] = param;
            bucket.param_dtensors[j] = all_params_[var_idx];
            bucket.offset[j]        = running_offset;
            bucket.param_size[j]    = param.nbytes();

            running_offset += param.numel();
        }

        bucket.flatten_gradient = OwnTensor::Tensor::empty(
            OwnTensor::Shape{{static_cast<int64_t>(running_offset)}}, t_opts
        );

        if (opts_.grad_as_view) {
            grad_map_flatten_grad(bucket);
        } else {
            create_bucket_views(bucket);
        }

        bucket.bucket_pending = var_count;
        buckets_.push_back(std::move(bucket));
    }

    // Build variable_locator_ for O(1) lookup from param index -> bucket
    for (size_t bi = 0; bi < buckets_.size(); bi++) {
        auto& bkt = buckets_[bi];
        for (size_t j = 0; j < bkt.param_indices.size(); j++) {
            size_t model_idx = bkt.param_indices[j];
            variable_locator_[model_idx] = {bi, j};
        }
    }
}

void DataParallel2D::create_bucket_views(Bucket2D& bucket) {
    bucket.grad_incoming.clear();
    for (size_t i = 0; i < bucket.param_indices.size(); i++) {
        auto& shape = bucket.param_tensors[i].shape();
        bucket.grad_incoming.push_back(
            bucket.flatten_gradient.narrow_view(0, bucket.offset[i], bucket.length[i]).view(shape)
        );
    }
    bucket.grad_outgoing = bucket.grad_incoming;
}

void DataParallel2D::grad_map_flatten_grad(Bucket2D& bucket) {
    for (size_t i = 0; i < bucket.param_tensors.size(); i++) {
        auto& curr_param = bucket.param_tensors[i];
        curr_param.unsafeGetTensorImpl()->mutable_grad() =
            bucket.flatten_gradient.narrow_view(0, bucket.offset[i], bucket.length[i]).view(curr_param.shape());
    }
}

// ===================================================================
// Autograd hooks for bucket-based gradient sync
// ===================================================================

void DataParallel2D::register_hooks() {
    grad_acc_map_.clear();

    for (size_t pi = 0; pi < raw_tensors_.size(); pi++) {
        auto& param = raw_tensors_[pi];
        auto grad_acc = OwnTensor::impl::grad_accumulator(param);

        grad_acc->register_post_hook(
            [this, pi](const std::vector<OwnTensor::Tensor>& /*inputs*/,
                       const std::vector<OwnTensor::Tensor>& /*outputs*/) {
                if (grad_sync_enabled_) {
                    // For replicated params, TP sync must happen before DDP bucket fires.
                    // Do per-param TP sync inline here so it's complete before bucket all-reduce.
                    if (is_replicated_[pi] && opts_.tp_world_size > 1) {
                        OwnTensor::Tensor& t = all_params_[pi]->mutable_tensor();
                        if (t.has_grad()) {
                            OwnTensor::Tensor g = t.grad_view();
                            if (g.is_valid() && g.numel() > 0) {
                                tp_pg_->all_reduce(
                                    g.data(), g.data(),
                                    g.numel(), g.dtype(),
                                    avg, true
                                );
                            }
                        }
                    }
                    this->mediator_hook(pi);
                }
            }
        );

        grad_acc_map_[grad_acc] = pi;
    }
}

void DataParallel2D::mediator_hook(size_t variable_index) {
    if (!mark_unused_) {
        mark_unused_ = true;
        for (auto& idx : unused_param_indices_) {
            mark_param_ready(idx);
        }
    }

    if (rebuild_bucket_) {
        rebuilt_params_order_.push_back(raw_tensors_[variable_index]);
        rebuilt_indices_order_.push_back(variable_index);
    }

    mark_param_ready(variable_index);
}

void DataParallel2D::mark_param_ready(size_t variable_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = variable_locator_.find(variable_index);
    if (it == variable_locator_.end()) return;

    auto& locator = it->second;
    auto& bucket  = buckets_[locator.bucket_index];

    if (raw_tensors_[variable_index].requires_grad()) {
        if (!opts_.grad_as_view) {
            size_t intra = locator.intra_bucket_index;
            OwnTensor::device::copy_memory(
                bucket.grad_incoming[intra].data(),
                bucket.grad_incoming[intra].device().device,
                raw_tensors_[variable_index].grad(),
                raw_tensors_[variable_index].device().device,
                raw_tensors_[variable_index].nbytes()
            );
        }

        if (bucket.bucket_pending > 0) {
            bucket.bucket_pending--;
        }
    }

    if (bucket.bucket_pending == 0) {
        mark_bucket_ready(locator.bucket_index);
    }

    if (next_bucket_ == buckets_.size() && !finalize_called_) {
        finalize_called_ = true;
        OwnTensor::autograd::queue_call_back([this]() {
            this->finalize_backward();
        });
    }
}

void DataParallel2D::mark_bucket_ready(size_t bucket_index) {
    if (bucket_index > next_bucket_) return;

    for (; next_bucket_ < buckets_.size(); next_bucket_++) {
        Bucket2D& bkt = buckets_[next_bucket_];
        if (bkt.bucket_pending == 0) {
            all_reduce_bucket(bkt);
        } else {
            break;
        }
    }
}

void DataParallel2D::all_reduce_bucket(Bucket2D& bucket) {
    if (opts_.dp_world_size <= 1) return;

    // Sync compute stream -> communication stream
    cudaEventRecord(compute_sync_event_, OwnTensor::cuda::getCurrentStream());
    cudaStreamWaitEvent(dp_pg_->get_stream(), compute_sync_event_, 0);

    void* grad_ptr = bucket.flatten_gradient.data();
    bucket.work_obj_ = dp_pg_->all_reduce_async(
        grad_ptr, grad_ptr,
        bucket.flatten_gradient.numel(),
        bucket.flatten_gradient.dtype(),
        avg,
        false
    );
}

void DataParallel2D::finalize_backward() {
    finalize_called_ = true;

    for (int i = static_cast<int>(buckets_.size()) - 1; i >= 0; i--) {
        auto& bucket = buckets_[i];

        if (bucket.work_obj_) {
            bucket.work_obj_->wait();
        }

        if (!opts_.grad_as_view) {
            for (size_t j = 0; j < bucket.grad_outgoing.size(); j++) {
                auto& param = raw_tensors_[bucket.param_indices[j]];
                param.set_grad(bucket.grad_outgoing[j]);
            }
        }
    }
}

// ===================================================================
// Unused parameter detection (BFS over autograd graph)
// ===================================================================

void DataParallel2D::search_for_unused(const std::vector<OwnTensor::Tensor>& outputs) {
    std::queue<std::shared_ptr<OwnTensor::Node>> order;
    std::unordered_set<std::shared_ptr<OwnTensor::Node>> seen;

    for (auto& output : outputs) {
        auto grad_fn = output.grad_fn();
        if (grad_fn) {
            order.push(grad_fn);
        }
    }

    while (!order.empty()) {
        auto top = order.front();
        order.pop();
        for (const auto& edge : top->next_edges()) {
            if (auto next = edge.function) {
                if (seen.insert(next).second) {
                    order.push(next);
                }
            }
        }
    }

    unused_param_indices_.clear();
    for (const auto& [grad_acc_node, param_index] : grad_acc_map_) {
        if (seen.find(grad_acc_node) == seen.end()) {
            unused_param_indices_.push_back(param_index);
        }
    }
}

// ===================================================================
// Pre-forward / Pre-backward orchestration
// ===================================================================

void DataParallel2D::pre_forward() {
    iteration_count_++;

    if (rebuild_bucket_ && first_backward_) {
        rebuild_buckets();
    }
}

void DataParallel2D::pre_backward(const std::vector<OwnTensor::Tensor>& outputs) {
    mark_unused_ = false;
    finalize_called_ = false;
    next_bucket_ = 0;
    first_backward_ = true;

    // Reset bucket pending counts
    for (auto& bucket : buckets_) {
        bucket.bucket_pending = bucket.param_indices.size();
    }

    // Zero out flatten gradients if using grad-as-view
    if (opts_.grad_as_view) {
        for (auto& bucket : buckets_) {
            cudaMemsetAsync(
                bucket.flatten_gradient.data(), 0,
                bucket.flatten_gradient.nbytes(),
                dp_pg_->get_stream()
            );
        }
    }

    // Detect unused parameters via BFS
    if (!opts_.static_graph) {
        search_for_unused(outputs);
    }
}

// ===================================================================
// Bucket rebuild after first backward (reorder by actual backward order)
// ===================================================================

void DataParallel2D::rebuild_buckets() {
    if (!rebuild_bucket_) return;
    rebuild_bucket_ = false;

    buckets_.clear();
    variable_locator_.clear();

    auto bucket_indices = decide_bucket_order(rebuilt_params_order_, rebuilt_indices_order_);
    rebuilt_params_order_.clear();
    rebuilt_indices_order_.clear();

    // Recreate buckets in the new order
    for (size_t bi = 0; bi < bucket_indices.size(); bi++) {
        Bucket2D bucket;
        auto& indices = bucket_indices[bi];
        size_t var_count = indices.size();

        bucket.param_dtensors.resize(var_count);
        bucket.param_tensors.resize(var_count);
        bucket.param_indices.resize(var_count);
        bucket.length.resize(var_count);
        bucket.offset.resize(var_count);
        bucket.param_size.resize(var_count);

        uint64_t running_offset = 0;
        OwnTensor::TensorOptions t_opts;
        bool device_set = false;
        bool dtype_set  = false;

        for (size_t j = 0; j < var_count; j++) {
            size_t var_idx = indices[j];
            auto& param = raw_tensors_[var_idx];

            if (!device_set) { t_opts.device = param.device(); device_set = true; }
            if (!dtype_set)  { t_opts.dtype  = param.dtype();  dtype_set  = true; }

            bucket.length[j]        = param.numel();
            bucket.param_indices[j] = var_idx;
            bucket.param_tensors[j] = param;
            bucket.param_dtensors[j] = all_params_[var_idx];
            bucket.offset[j]        = running_offset;
            bucket.param_size[j]    = param.nbytes();

            running_offset += param.numel();
        }

        bucket.flatten_gradient = OwnTensor::Tensor::empty(
            OwnTensor::Shape{{static_cast<int64_t>(running_offset)}}, t_opts
        );

        if (opts_.grad_as_view) {
            grad_map_flatten_grad(bucket);
        } else {
            create_bucket_views(bucket);
        }

        bucket.bucket_pending = var_count;
        buckets_.push_back(std::move(bucket));
    }

    for (size_t bi = 0; bi < buckets_.size(); bi++) {
        auto& bkt = buckets_[bi];
        for (size_t j = 0; j < bkt.param_indices.size(); j++) {
            variable_locator_[bkt.param_indices[j]] = {bi, j};
        }
    }
}
