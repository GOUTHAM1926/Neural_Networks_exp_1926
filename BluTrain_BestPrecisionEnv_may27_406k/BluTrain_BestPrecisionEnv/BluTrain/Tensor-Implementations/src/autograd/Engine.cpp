#include "autograd/Engine.h"
#include "autograd/Functions.h"
#include "autograd/GraphRecorder.h"
#include "core/AutogradMeta.h"
#include "core/TensorImpl.h"
#include "ops/TensorOps.h"
#include "utils/ThreadPool.h"
#include "device/AllocationTracker.h"
#include "utils/Profiler.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <deque>
#include <stdexcept>   
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace OwnTensor {
namespace autograd {

// =============================================================================
// Execution Mode Configuration 
// =============================================================================
namespace {
    // Global execution mode (default: SEQUENTIAL for determinism and debugging)
    ExecutionMode g_execution_mode = ExecutionMode::SEQUENTIAL;
    std::mutex g_mode_mutex;
}

ExecutionMode get_execution_mode() {
    std::lock_guard<std::mutex> lock(g_mode_mutex);
    return g_execution_mode;
}

void set_execution_mode(ExecutionMode mode) {
    std::lock_guard<std::mutex> lock(g_mode_mutex);
    g_execution_mode = mode;
}

// ====================================================================
// Gradient Vector Pool
// ====================================================================
class GradientVectorPool {
    std::vector<std::vector<Tensor>> pool_;
    std::mutex mutex_;
    
public:
    static GradientVectorPool& instance() {
        static GradientVectorPool pool;
        return pool;
    }

    std::vector<Tensor> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            std::vector<Tensor> vec = std::move(pool_.back());
            pool_.pop_back();
            return vec;
        }
        std::vector<Tensor> vec;
        vec.reserve(4);
        return vec;
    }

    void release(std::vector<Tensor>&& vec) {
        vec.clear();
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(vec));
    }
};

// =============================================================================
// Topological Sort (Optimized)
// =============================================================================
std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
    std::vector<std::shared_ptr<Node>> result;
    std::unordered_set<Node*> visited;
    auto root_fn = root.grad_fn();
    if (!root_fn) return result;
    
    // Iterative Post-Order DFS
    std::vector<std::shared_ptr<Node>> stack;
    
    struct Frame {
        std::shared_ptr<Node> node;
        size_t next_edge_idx = 0;
    };
    
    std::vector<Frame> call_stack;
    if (root_fn) {
        call_stack.push_back({root_fn, 0});
        visited.insert(root_fn.get());
    }
    
    while (!call_stack.empty()) {
        auto& frame = call_stack.back();
        auto node = frame.node;
        
        bool pushed_child = false;
        const auto& edges = node->next_edges();
        
        while (frame.next_edge_idx < edges.size()) {
            const auto& edge = edges[frame.next_edge_idx];
            frame.next_edge_idx++;
            
            if (edge.is_valid()) {
                if (visited.find(edge.function.get()) == visited.end()) {
                    visited.insert(edge.function.get());
                    call_stack.push_back({edge.function, 0});
                    pushed_child = true;
                    break;
                }
            }
        }
        
        if (!pushed_child) {
            result.push_back(node);
            call_stack.pop_back();
        }
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}



// =============================================================================
// Backward Engine
// =============================================================================

// Execution State for a single node
struct NodeTask {
    std::mutex mutex;
    
    // Fast path for 0-7 input slots (99% of nodes)
    std::vector<Tensor> fast_input_grads[8];
    // Slow path for nodes with many inputs
    std::unordered_map<uint32_t, std::vector<Tensor>> slow_input_grads_map;
    
    int dependencies = 0;
    bool scheduled = false;
    uint32_t max_output_nr = 0;

    NodeTask() = default;
    
    void reset() {
        dependencies = 0;
        scheduled = false;
        max_output_nr = 0;
        slow_input_grads_map.clear();
        for (int i = 0; i < 8; ++i) {
            fast_input_grads[i].clear();
        }
    }
};

// Optimization: Stable pool for NodeTasks to avoid thousands of allocations per step.
// We use a deque because it doesn't move objects on growth (stable pointers).
class NodeTaskPool {
public:
    static NodeTaskPool& instance() {
        static NodeTaskPool inst;
        return inst;
    }
    
    NodeTask* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (next_free_ < pool_.size()) {
            NodeTask* t = pool_[next_free_++].get();
            t->reset();
            return t;
        }
        pool_.emplace_back(std::make_unique<NodeTask>());
        next_free_++;
        return pool_.back().get();
    }
    
    void release_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        next_free_ = 0;
    }

private:
    std::vector<std::unique_ptr<NodeTask>> pool_;
    size_t next_free_ = 0;
    std::mutex mutex_;
    NodeTaskPool() { pool_.reserve(10000); }
};

// Shared Context for the entire Backward Pass
struct BackwardContext {
    std::mutex state_mutex;
    std::condition_variable cv;
    std::atomic<int> active_tasks{0};
    
    // Raw pointers from the stable pool
    std::vector<NodeTask*> task_storage;
};

// =============================================================================
// Sequential Backward Implementation
// =============================================================================

/**
 * @brief Sequential backward pass with dependency-based execution.
 * 
 * Executes nodes in topological order when all dependencies are satisfied.
 * Single-threaded, deterministic execution.
 */
void backward_sequential(const Tensor& root, const Tensor& root_grad) {
    AUTO_PROFILE("BackwardPass::Sequential");
    auto root_fn = root.grad_fn();
    
    // Handle Leaf Root case
    if (!root_fn) {
        if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(
                root.unsafeGetTensorImpl()->autograd_meta());
            if (meta->has_grad()) {
                Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
                Tensor new_grad = operator+(existing_grad, root_grad);
                meta->set_grad(new_grad);
            } else {
                meta->set_grad(root_grad);
            }
        }
        return;
    }

    // Step 1: Build graph and initialize dependency counters
    // Use flat vector + node->engine_data() for O(1) task lookup
    struct SequentialNodeTask {
        std::unordered_map<uint32_t, std::vector<Tensor>> input_grads_map;
        int dependencies = 0;
        uint32_t max_output_nr = 0;
    };
    
    // Use deque for stable pointers even when growing
    std::deque<SequentialNodeTask> all_tasks;

    std::vector<Node*> bfs_queue;
    bfs_queue.reserve(64);
    
    // Initialize root node
    bfs_queue.push_back(root_fn.get());
    all_tasks.emplace_back();
    root_fn->set_engine_data(&all_tasks.back());
    
    size_t head = 0;
    while (head < bfs_queue.size()) {
        Node* node = bfs_queue[head++];
        
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                Node* next_node = edge.function.get();
                
                // Check if we've seen this node via its engine_data pointer
                SequentialNodeTask* task_ptr = static_cast<SequentialNodeTask*>(next_node->engine_data());
                if (task_ptr == nullptr) {
                    // First time seeing this node - allocate task
                    bfs_queue.push_back(next_node);
                    all_tasks.emplace_back();
                    task_ptr = &all_tasks.back();
                    next_node->set_engine_data(task_ptr);
                }
                
                // Update task
                task_ptr->dependencies++;
                if (edge.input_nr > task_ptr->max_output_nr) {
                    task_ptr->max_output_nr = edge.input_nr;
                }
            }
        }
    }
    
    // Step 2: Initialize root gradient
    {
        SequentialNodeTask* root_task = static_cast<SequentialNodeTask*>(root_fn->engine_data());
        uint32_t slot = root.output_nr();
        root_task->input_grads_map[slot].push_back(root_grad);
        if (slot > root_task->max_output_nr) root_task->max_output_nr = slot;
    }
    
    // Step 3: Ready queue - nodes with zero dependencies
    std::queue<Node*> ready_queue;
    ready_queue.push(root_fn.get());

    // Step 4: Execute nodes in dependency order
    while (!ready_queue.empty()) {
        Node* node = ready_queue.front();
        ready_queue.pop();
        
        SequentialNodeTask* task = static_cast<SequentialNodeTask*>(node->engine_data());
        
        // Aggregate gradients for this node
        variable_list node_inputs;
        node_inputs.reserve(task->max_output_nr + 1);
        node_inputs.resize(task->max_output_nr + 1);
        bool has_grad = false;
        
        for (auto& [slot, grads] : task->input_grads_map) {
            if (!grads.empty()) {
                Tensor sum = std::move(grads[0]);
                // Materialize non-contiguous lead grad before in-place += loop.
                // Without this, when slot 0 carries a strided view (e.g. the
                // TransposeBackward output flowing into wte.weight under weight
                // tying), every subsequent `sum +=` falls through the strided
                // broadcast kernel — same value, ~140x slower than the contig
                // path. Investigation: ADD_IP debug log #25/#197 stack chain
                // confirms direct backward_sequential -> operator+= on a
                // non-contig leaf grad.
                //
                // TODO(Gautam_1926): this materialize fires 1x per microstep
                // (320 calls / ~180 ms per 10-step profile). The transpose copy
                // is structurally redundant — it only exists because the
                // tied lm_head goes through autograd::transpose(wte.weight, 0, 1)
                // -> TransposeBackward, which produces a [V,C] view of the
                // [C,V] contig grad_w_T. Replace with a custom TiedLMHeadBackward
                // node that asks cuBLAS to gemm in `grad_logits.t() @ saved_x`
                // orientation directly producing [V,C] contig. Skips the copy
                // entirely. See Resolving_Strided_Broadcast_Inplace_Add.md
                // section 10 (Option C). ~180 ms recoverable, mid-sized work.
                if (!sum.is_contiguous()) sum = sum.contiguous();
                for (size_t i = 1; i < grads.size(); ++i) {
                    sum += std::move(grads[i]);
                }
                node_inputs[slot] = sum;
                has_grad = true;
            }
        }
        task->input_grads_map.clear();
        
        variable_list input_grads;
        if (has_grad) {
            // Capture input-grad shape for tape annotation before node_inputs is moved.
            // The string is only built when the recorder AND shape-debug are both active.
            std::string bwd_loc = std::string("L319:Backward::") + node->name();
            TRACK_ALLOC_LOCATION(bwd_loc.c_str());
            std::string bwd_shape;
            if (autograd::g_shape_debug && GraphRecordMode::get_active()) {
                for (const auto& t : node_inputs) {
                    if (t.is_valid()) {
                        bwd_shape = GraphRecordMode::make_shape_info(t.shape(), t.dtype());
                        break;
                    }
                }
            }
            #ifdef AUTOGRAD_PROFILER_ENABLED
                bool is_cuda_op = false;
                for (const auto& t : node_inputs) {
                    if (t.is_valid() && t.is_cuda()) {
                        is_cuda_op = true;
                        break;
                    }
                }
                // Avoid string concatenation in the hot path
                OwnTensor::autograd::ProfileGuard prof_guard(node->name(), is_cuda_op);
                input_grads = (*node)(std::move(node_inputs));
            #else
                input_grads = (*node)(std::move(node_inputs));
            #endif

            // Record this backward operation (with optional shape annotation)
            GraphRecordMode::record_backward(node->name(), std::move(bwd_shape));
        }
        
        // Release saved variables and clear engine_data
        node->release_saved_variables();
        node->set_engine_data(nullptr);  // Clear for next backward pass
        
        // Propagate gradients to next nodes
        const auto& edges = node->next_edges();
        for (size_t i = 0; i < edges.size(); ++i) {
            if (!edges[i].is_valid()) continue;
            
            Node* next_node = edges[i].function.get();
            uint32_t slot = edges[i].input_nr;
            
            SequentialNodeTask* next_task = static_cast<SequentialNodeTask*>(next_node->engine_data());
            
            // Add gradient to next node if available
            if (i < input_grads.size() && input_grads[i].is_valid()) {
                next_task->input_grads_map[slot].push_back(std::move(input_grads[i]));
                if (slot > next_task->max_output_nr) next_task->max_output_nr = slot;
            }
            
            // Decrement dependency counter
            next_task->dependencies--;
            
            // If all dependencies satisfied, add to ready queue
            if (next_task->dependencies == 0) {
                ready_queue.push(next_node);
            }
        }
    }
    
        // Step 5: Safety cleanup - Clear engine_data for all nodes
    for (Node* n : bfs_queue) {
        n->set_engine_data(nullptr);
    }
}

// =============================================================================
// Parallel Backward Implementation
// =============================================================================

void backward_parallel(const Tensor& root, const Tensor& root_grad) {
    AUTO_PROFILE("BackwardPass::Parallel");
    auto root_fn = root.grad_fn();
    
    // Handle Leaf Root case
    if (!root_fn) {
        if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(
                root.unsafeGetTensorImpl()->autograd_meta());
            if (meta->has_grad()) {
                Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
                Tensor new_grad = operator+(existing_grad, root_grad);
                meta->set_grad(new_grad);
            } else {
                meta->set_grad(root_grad);
            }
        }
        return;
    }

    // Discover Graph and Setup
    auto ctx = std::make_shared<BackwardContext>();
    std::vector<Node*> bfs_queue;
    bfs_queue.reserve(512);
    
    // Initialize root task
    bfs_queue.push_back(root_fn.get());
    NodeTask* root_task = NodeTaskPool::instance().acquire();
    ctx->task_storage.push_back(root_task);
    root_fn->set_engine_data(root_task);
    
    size_t head = 0;
    while(head < bfs_queue.size()) {
        Node* node = bfs_queue[head++];
        
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                Node* next_node = edge.function.get();
                
                NodeTask* task = static_cast<NodeTask*>(next_node->engine_data());
                if (!task) {
                    task = NodeTaskPool::instance().acquire();
                    ctx->task_storage.push_back(task);
                    next_node->set_engine_data(task);
                    bfs_queue.push_back(next_node);
                }
                task->dependencies++;
                if (edge.input_nr > task->max_output_nr) {
                    task->max_output_nr = edge.input_nr;
                }
            }
        }
    }

    // Execution Setup
    // Execution Setup - using global pool from TaskFunctor
    
    // Seed root gradient
    {
        NodeTask* root_task = static_cast<NodeTask*>(root_fn->engine_data());
        uint32_t slot = root.output_nr();
        auto& grad_vec = (slot < 8) ? root_task->fast_input_grads[slot] : root_task->slow_input_grads_map[slot];
        grad_vec.push_back(root_grad);
        if (slot > root_task->max_output_nr) root_task->max_output_nr = slot;
    }
    
    // Task Functor for parallel execution
    struct TaskFunctor {
        std::shared_ptr<BackwardContext> ctx;
        Node* node;
        
        void operator()() {
            try {
                NodeTask* task = static_cast<NodeTask*>(node->engine_data());
                
                variable_list node_inputs;
                bool has_grad = false;
                {
                    std::lock_guard<std::mutex> lock(task->mutex);
                    node_inputs.reserve(task->max_output_nr + 1);
                    node_inputs.resize(task->max_output_nr + 1);
                    
                    // Process fast path
                    for (int i = 0; i < 8; ++i) {
                        auto& grads = task->fast_input_grads[i];
                        if (!grads.empty()) {
                            Tensor sum = std::move(grads[0]);
                            // see backward_sequential note: non-contig lead grad
                            // would route every += through the strided broadcast
                            // kernel. materialize once upfront.
                            if (!sum.is_contiguous()) sum = sum.contiguous();
                            for (size_t j = 1; j < grads.size(); ++j) {
                                sum += std::move(grads[j]);
                            }
                            node_inputs[i] = sum;
                            has_grad = true;
                            GradientVectorPool::instance().release(std::move(grads));
                        }
                    }

                    // Process slow path
                    for (auto& [slot, grads] : task->slow_input_grads_map) {
                        if (!grads.empty()) {
                            Tensor sum = std::move(grads[0]);
                            if (!sum.is_contiguous()) sum = sum.contiguous();
                            for (size_t i = 1; i < grads.size(); ++i) {
                                sum += std::move(grads[i]);
                            }
                            node_inputs[slot] = sum;
                            has_grad = true;
                            GradientVectorPool::instance().release(std::move(grads));
                        }
                    }
                    task->slow_input_grads_map.clear();
                }
                
                variable_list input_grads;
                if (has_grad) {
                    // Capture input-grad shape for tape annotation before node_inputs is moved.
                    std::string bwd_shape;
                    if (autograd::g_shape_debug && GraphRecordMode::get_active()) {
                        for (const auto& t : node_inputs) {
                            if (t.is_valid()) {
                                bwd_shape = GraphRecordMode::make_shape_info(t.shape(), t.dtype());
                                break;
                            }
                        }
                    }
                    #ifdef AUTOGRAD_PROFILER_ENABLED
                        bool is_cuda_op = false;
                        for (const auto& t : node_inputs) {
                            if (t.is_valid() && t.is_cuda()) {
                                is_cuda_op = true;
                                break;
                            }
                        }

                        OwnTensor::autograd::ProfileGuard prof_guard(node->name(), is_cuda_op);
                        input_grads = (*node)(std::move(node_inputs));
                    #else
                        input_grads = (*node)(std::move(node_inputs));
                    #endif

                    // Record this backward operation (with optional shape annotation)
                    GraphRecordMode::record_backward(node->name(), std::move(bwd_shape));

                    // Check for CUDA errors after execution
                    #ifdef WITH_CUDA
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        throw std::runtime_error("CUDA error during node evaluation (" +
                                                 std::string(node->name()) + "): " +
                                                 std::string(cudaGetErrorString(err)));
                    }
                    #endif
                }
                
                // Release saved variables
                node->release_saved_variables();
                
                // Propagate to dependencies
                const auto& edges = node->next_edges();
                variable_list output_grads = std::move(input_grads); 
                
                // Clear engine_data for this node
                node->set_engine_data(nullptr);
                
                for (size_t i = 0; i < edges.size(); ++i) {
                    if (!edges[i].is_valid()) continue;
                    
                    Node* next_node = edges[i].function.get();
                    uint32_t slot = edges[i].input_nr;
                    
                    NodeTask* next_task = static_cast<NodeTask*>(next_node->engine_data());
                    bool ready = false;
                    {
                        std::lock_guard<std::mutex> lock(next_task->mutex);
                        if (i < output_grads.size() && output_grads[i].is_valid()) {
                           auto& grad_vec = (slot < 8) ? next_task->fast_input_grads[slot] : next_task->slow_input_grads_map[slot];
                           if (grad_vec.empty()) {
                               grad_vec = GradientVectorPool::instance().acquire();
                           }
                           grad_vec.push_back(std::move(output_grads[i]));
                           if (slot > next_task->max_output_nr) next_task->max_output_nr = slot;
                        }
                        next_task->dependencies--;
                        if (next_task->dependencies == 0 && !next_task->scheduled) {
                            next_task->scheduled = true;
                            ready = true;
                        }
                    }
                    
                    if (ready) {
                        schedule(ctx, next_node);
                    }
                }
            } catch (const std::exception& e) {
                // Signal error if possible, but at minimum dec active_tasks so we don't hang
                std::cerr << "[Engine] Error in worker thread: " << e.what() << std::endl;
                // Note: In an ideal world we'd propagate this back to the main thread
                // via a shared atomic error flag or similar in ctx.
            } catch (...) {
                std::cerr << "[Engine] Unknown error in worker thread." << std::endl;
            }
            
            // Atomic decrement (CRITICAL: must happen even on error to avoid hang)
            int remaining = --ctx->active_tasks;
            if (remaining == 0) {
                 std::lock_guard<std::mutex> lk(ctx->state_mutex);
                 ctx->cv.notify_all();
            }
        }
        
        static void schedule(std::shared_ptr<BackwardContext> ctx, Node* node) {
            ctx->active_tasks++;
            static utils::ThreadPool& pool = get_pool();
            pool.enqueue(TaskFunctor{ctx, node});
        }
        
        static utils::ThreadPool& get_pool() {
             static utils::ThreadPool pool(std::thread::hardware_concurrency());
             return pool;
        }
    };
    
    // Kickoff
    TaskFunctor::schedule(ctx, root_fn.get());
    
    // Wait for completion
    {
        std::unique_lock<std::mutex> lk(ctx->state_mutex);
        ctx->cv.wait(lk, [&] { return ctx->active_tasks.load() == 0; });
    }
    
    // Safety cleanup: Ensure everything in BFS queue is nulled 
    // (In case of early exit or unscheduled nodes)
    for (Node* n : bfs_queue) {
        n->set_engine_data(nullptr);
    }
    
    // Release pool for next step
    NodeTaskPool::instance().release_all();
}

// =============================================================================
// Main Backward Dispatcher
// =============================================================================

void backward(const std::vector<Tensor>& roots, const std::vector<Tensor>& grad_outputs) {
    if (roots.empty()) return;

    std::vector<Tensor> processed_grads;
    processed_grads.reserve(roots.size());

    for (size_t i = 0; i < roots.size(); ++i) {
        const auto& root = roots[i];
        if (!root.requires_grad()) {
             throw std::runtime_error("backward: root tensor " + std::to_string(i) + " does not require gradients");
        }

        if (i < grad_outputs.size() && grad_outputs[i].is_valid()) {
            processed_grads.push_back(grad_outputs[i]);
        } else {
            // Default to ones for scalars, error for non-scalars
            if (root.ndim() == 0 || root.numel() == 1) {
                processed_grads.push_back(Tensor::ones(root.shape(), TensorOptions()
                    .with_dtype(root.dtype())
                    .with_device(root.device())));
            } else {
                throw std::runtime_error("backward: non-scalar root " + std::to_string(i) + " requires explicit grad_output");
            }
        }
    }

    // Since our backends currently only support single root, we iterate for now.
    // However, if they are supposed to be integrated, we should ideally pass vectors.
    // For now, to fix the compilation error while maintaining existing backend signatures:
    ExecutionMode mode = get_execution_mode();
    for (size_t i = 0; i < roots.size(); ++i) {
        if (mode == ExecutionMode::SEQUENTIAL) {
            backward_sequential(roots[i], processed_grads[i]);
        } else {
            backward_parallel(roots[i], processed_grads[i]);
        }
    }
}

void backward(const Tensor& root, const Tensor* grad_output) {
    std::vector<Tensor> roots = {root};
    std::vector<Tensor> grads;
    if (grad_output) {
        grads.push_back(*grad_output);
    }
    backward(roots, grads);
}

static thread_local std::shared_ptr<BackwardContext> g_current_context = nullptr;
 
// Access the global thread pool for the autograd engine
static utils::ThreadPool& get_engine_pool() {
    static utils::ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}
 

void queue_call_back(std::function<void()> callback) {
    if (get_execution_mode() == ExecutionMode::SEQUENTIAL) {
         callback();
    } else {
        auto ctx = g_current_context;
        if (ctx) {
            ctx->active_tasks++;
            get_engine_pool().enqueue_detach([ctx, cb = std::move(callback)]() {
                g_current_context = ctx; try { cb(); } catch (...) {}
                if (--ctx->active_tasks == 0) { std::lock_guard<std::mutex> lk(ctx->state_mutex); ctx->cv.notify_all(); }
                g_current_context = nullptr;
            });
        } else get_engine_pool().enqueue_detach(std::move(callback));
    }
}

} // namespace autograd
} // namespace OwnTensor


// #include "autograd/Engine.h"
// #include "autograd/Functions.h"
// #include "autograd/GraphRecorder.h"
// #include "core/AutogradMeta.h"
// #include "core/TensorImpl.h"
// #include "ops/TensorOps.h"
// #include "utils/ThreadPool.h"
// #include "device/AllocationTracker.h"
// #include "utils/Profiler.h"
// #ifdef WITH_CUDA
// #include <cuda_runtime.h>
// #endif
// #include <algorithm>
// #include <unordered_set>
// #include <unordered_map>
// #include <queue>
// #include <deque>
// #include <stdexcept>   
// #include <mutex>
// #include <condition_variable>
// #include <atomic>

// namespace OwnTensor {
// namespace autograd {

// // =============================================================================
// // Execution Mode Configuration 
// // =============================================================================
// namespace {
//     // Global execution mode (default: SEQUENTIAL for determinism and debugging)
//     ExecutionMode g_execution_mode = ExecutionMode::SEQUENTIAL;
//     std::mutex g_mode_mutex;
// }

// ExecutionMode get_execution_mode() {
//     std::lock_guard<std::mutex> lock(g_mode_mutex);
//     return g_execution_mode;
// }

// void set_execution_mode(ExecutionMode mode) {
//     std::lock_guard<std::mutex> lock(g_mode_mutex);
//     g_execution_mode = mode;
// }

// // ====================================================================
// // Gradient Vector Pool
// // ====================================================================
// class GradientVectorPool {
//     std::vector<std::vector<Tensor>> pool_;
//     std::mutex mutex_;
    
// public:
//     static GradientVectorPool& instance() {
//         static GradientVectorPool pool;
//         return pool;
//     }

//     std::vector<Tensor> acquire() {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (!pool_.empty()) {
//             std::vector<Tensor> vec = std::move(pool_.back());
//             pool_.pop_back();
//             return vec;
//         }
//         std::vector<Tensor> vec;
//         vec.reserve(4);
//         return vec;
//     }

//     void release(std::vector<Tensor>&& vec) {
//         vec.clear();
//         std::lock_guard<std::mutex> lock(mutex_);
//         pool_.push_back(std::move(vec));
//     }
// };

// // =============================================================================
// // Topological Sort (Optimized)
// // =============================================================================
// std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
//     std::vector<std::shared_ptr<Node>> result;
//     std::unordered_set<Node*> visited;
//     auto root_fn = root.grad_fn();
//     if (!root_fn) return result;
    
//     // Iterative Post-Order DFS
//     std::vector<std::shared_ptr<Node>> stack;
    
//     struct Frame {
//         std::shared_ptr<Node> node;
//         size_t next_edge_idx = 0;
//     };
    
//     std::vector<Frame> call_stack;
//     if (root_fn) {
//         call_stack.push_back({root_fn, 0});
//         visited.insert(root_fn.get());
//     }
    
//     while (!call_stack.empty()) {
//         auto& frame = call_stack.back();
//         auto node = frame.node;
        
//         bool pushed_child = false;
//         const auto& edges = node->next_edges();
        
//         while (frame.next_edge_idx < edges.size()) {
//             const auto& edge = edges[frame.next_edge_idx];
//             frame.next_edge_idx++;
            
//             if (edge.is_valid()) {
//                 if (visited.find(edge.function.get()) == visited.end()) {
//                     visited.insert(edge.function.get());
//                     call_stack.push_back({edge.function, 0});
//                     pushed_child = true;
//                     break;
//                 }
//             }
//         }
        
//         if (!pushed_child) {
//             result.push_back(node);
//             call_stack.pop_back();
//         }
//     }
    
//     std::reverse(result.begin(), result.end());
//     return result;
// }



// // =============================================================================
// // Backward Engine
// // =============================================================================

// // Execution State for a single node
// struct NodeTask {
//     std::mutex mutex;
    
//     // Fast path for 0-7 input slots (99% of nodes)
//     std::vector<Tensor> fast_input_grads[8];
//     // Slow path for nodes with many inputs
//     std::unordered_map<uint32_t, std::vector<Tensor>> slow_input_grads_map;
    
//     int dependencies = 0;
//     bool scheduled = false;
//     uint32_t max_output_nr = 0;

//     NodeTask() = default;
    
//     void reset() {
//         dependencies = 0;
//         scheduled = false;
//         max_output_nr = 0;
//         slow_input_grads_map.clear();
//         for (int i = 0; i < 8; ++i) {
//             fast_input_grads[i].clear();
//         }
//     }
// };

// // Optimization: Stable pool for NodeTasks to avoid thousands of allocations per step.
// // We use a deque because it doesn't move objects on growth (stable pointers).
// class NodeTaskPool {
// public:
//     static NodeTaskPool& instance() {
//         static NodeTaskPool inst;
//         return inst;
//     }
    
//     NodeTask* acquire() {
//         std::lock_guard<std::mutex> lock(mutex_);
//         if (next_free_ < pool_.size()) {
//             NodeTask* t = pool_[next_free_++].get();
//             t->reset();
//             return t;
//         }
//         pool_.emplace_back(std::make_unique<NodeTask>());
//         next_free_++;
//         return pool_.back().get();
//     }
    
//     void release_all() {
//         std::lock_guard<std::mutex> lock(mutex_);
//         next_free_ = 0;
//     }

// private:
//     std::vector<std::unique_ptr<NodeTask>> pool_;
//     size_t next_free_ = 0;
//     std::mutex mutex_;
//     NodeTaskPool() { pool_.reserve(10000); }
// };

// // Shared Context for the entire Backward Pass
// struct BackwardContext {
//     std::mutex state_mutex;
//     std::condition_variable cv;
//     std::atomic<int> active_tasks{0};
    
//     // Raw pointers from the stable pool
//     std::vector<NodeTask*> task_storage;
// };

// // =============================================================================
// // Sequential Backward Implementation
// // =============================================================================

// /**
//  * @brief Sequential backward pass with dependency-based execution.
//  * 
//  * Executes nodes in topological order when all dependencies are satisfied.
//  * Single-threaded, deterministic execution.
//  */
// void backward_sequential(const Tensor& root, const Tensor& root_grad) {
//     AUTO_PROFILE("BackwardPass::Sequential");
//     auto root_fn = root.grad_fn();
    
//     // Handle Leaf Root case
//     if (!root_fn) {
//         if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
//             auto* meta = static_cast<AutogradMeta*>(
//                 root.unsafeGetTensorImpl()->autograd_meta());
//             if (meta->has_grad()) {
//                 Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
//                 Tensor new_grad = operator+(existing_grad, root_grad);
//                 meta->set_grad(new_grad);
//             } else {
//                 meta->set_grad(root_grad);
//             }
//         }
//         return;
//     }

//     // Step 1: Build graph and initialize dependency counters
//     // Use flat vector + node->engine_data() for O(1) task lookup
//     struct SequentialNodeTask {
//         std::unordered_map<uint32_t, std::vector<Tensor>> input_grads_map;
//         int dependencies = 0;
//         uint32_t max_output_nr = 0;
//     };
    
//     // Use deque for stable pointers even when growing
//     std::deque<SequentialNodeTask> all_tasks;

//     std::vector<Node*> bfs_queue;
//     bfs_queue.reserve(64);
    
//     // Initialize root node
//     bfs_queue.push_back(root_fn.get());
//     all_tasks.emplace_back();
//     root_fn->set_engine_data(&all_tasks.back());
    
//     size_t head = 0;
//     while (head < bfs_queue.size()) {
//         Node* node = bfs_queue[head++];
        
//         for (const auto& edge : node->next_edges()) {
//             if (edge.is_valid()) {
//                 Node* next_node = edge.function.get();
                
//                 // Check if we've seen this node via its engine_data pointer
//                 SequentialNodeTask* task_ptr = static_cast<SequentialNodeTask*>(next_node->engine_data());
//                 if (task_ptr == nullptr) {
//                     // First time seeing this node - allocate task
//                     bfs_queue.push_back(next_node);
//                     all_tasks.emplace_back();
//                     task_ptr = &all_tasks.back();
//                     next_node->set_engine_data(task_ptr);
//                 }
                
//                 // Update task
//                 task_ptr->dependencies++;
//                 if (edge.input_nr > task_ptr->max_output_nr) {
//                     task_ptr->max_output_nr = edge.input_nr;
//                 }
//             }
//         }
//     }
    
//     // Step 2: Initialize root gradient
//     {
//         SequentialNodeTask* root_task = static_cast<SequentialNodeTask*>(root_fn->engine_data());
//         uint32_t slot = root.output_nr();
//         root_task->input_grads_map[slot].push_back(root_grad);
//         if (slot > root_task->max_output_nr) root_task->max_output_nr = slot;
//     }
    
//     // Step 3: Ready queue - nodes with zero dependencies
//     std::queue<Node*> ready_queue;
//     ready_queue.push(root_fn.get());

//     // Step 4: Execute nodes in dependency order
//     while (!ready_queue.empty()) {
//         Node* node = ready_queue.front();
//         ready_queue.pop();
        
//         SequentialNodeTask* task = static_cast<SequentialNodeTask*>(node->engine_data());
        
//         // Aggregate gradients for this node
//         variable_list node_inputs;
//         node_inputs.reserve(task->max_output_nr + 1);
//         node_inputs.resize(task->max_output_nr + 1);
//         bool has_grad = false;
        
//         for (auto& [slot, grads] : task->input_grads_map) {
//             if (!grads.empty()) {
//                 Tensor sum = std::move(grads[0]);
//                 for (size_t i = 1; i < grads.size(); ++i) {
//                     sum += std::move(grads[i]);
//                 }
//                 node_inputs[slot] = sum;
//                 has_grad = true;
//             }
//         }
//         task->input_grads_map.clear();
        
//         variable_list input_grads;
//         if (has_grad) {
//             // Capture input-grad shape for tape annotation before node_inputs is moved.
//             // The string is only built when the recorder AND shape-debug are both active.
//             std::string bwd_loc = std::string("L319:Backward::") + node->name();
//             TRACK_ALLOC_LOCATION(bwd_loc.c_str());
//             std::string bwd_shape;
//             if (autograd::g_shape_debug && GraphRecordMode::get_active()) {
//                 for (const auto& t : node_inputs) {
//                     if (t.is_valid()) {
//                         bwd_shape = GraphRecordMode::make_shape_info(t.shape(), t.dtype());
//                         break;
//                     }
//                 }
//             }
//             #ifdef AUTOGRAD_PROFILER_ENABLED
//                 bool is_cuda_op = false;
//                 for (const auto& t : node_inputs) {
//                     if (t.is_valid() && t.is_cuda()) {
//                         is_cuda_op = true;
//                         break;
//                     }
//                 }
//                 // Avoid string concatenation in the hot path
//                 OwnTensor::autograd::ProfileGuard prof_guard(node->name(), is_cuda_op);
//                 input_grads = (*node)(std::move(node_inputs));
//             #else
//                 input_grads = (*node)(std::move(node_inputs));
//             #endif

//             // Record this backward operation (with optional shape annotation)
//             GraphRecordMode::record_backward(node->name(), std::move(bwd_shape));
//         }
        
//         // Release saved variables and clear engine_data
//         node->release_saved_variables();
//         node->set_engine_data(nullptr);  // Clear for next backward pass
        
//         // Propagate gradients to next nodes
//         const auto& edges = node->next_edges();
//         for (size_t i = 0; i < edges.size(); ++i) {
//             if (!edges[i].is_valid()) continue;
            
//             Node* next_node = edges[i].function.get();
//             uint32_t slot = edges[i].input_nr;
            
//             SequentialNodeTask* next_task = static_cast<SequentialNodeTask*>(next_node->engine_data());
            
//             // Add gradient to next node if available
//             if (i < input_grads.size() && input_grads[i].is_valid()) {
//                 next_task->input_grads_map[slot].push_back(std::move(input_grads[i]));
//                 if (slot > next_task->max_output_nr) next_task->max_output_nr = slot;
//             }
            
//             // Decrement dependency counter
//             next_task->dependencies--;
            
//             // If all dependencies satisfied, add to ready queue
//             if (next_task->dependencies == 0) {
//                 ready_queue.push(next_node);
//             }
//         }
//     }
    
//         // Step 5: Safety cleanup - Clear engine_data for all nodes
//     for (Node* n : bfs_queue) {
//         n->set_engine_data(nullptr);
//     }
// }

// // =============================================================================
// // Parallel Backward Implementation
// // =============================================================================

// void backward_parallel(const Tensor& root, const Tensor& root_grad) {
//     AUTO_PROFILE("BackwardPass::Parallel");
//     auto root_fn = root.grad_fn();
    
//     // Handle Leaf Root case
//     if (!root_fn) {
//         if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
//             auto* meta = static_cast<AutogradMeta*>(
//                 root.unsafeGetTensorImpl()->autograd_meta());
//             if (meta->has_grad()) {
//                 Tensor& existing_grad = meta->mutable_grad(root.unsafeGetTensorImpl());
//                 Tensor new_grad = operator+(existing_grad, root_grad);
//                 meta->set_grad(new_grad);
//             } else {
//                 meta->set_grad(root_grad);
//             }
//         }
//         return;
//     }

//     // Discover Graph and Setup
//     auto ctx = std::make_shared<BackwardContext>();
//     std::vector<Node*> bfs_queue;
//     bfs_queue.reserve(512);
    
//     // Initialize root task
//     bfs_queue.push_back(root_fn.get());
//     NodeTask* root_task = NodeTaskPool::instance().acquire();
//     ctx->task_storage.push_back(root_task);
//     root_fn->set_engine_data(root_task);
    
//     size_t head = 0;
//     while(head < bfs_queue.size()) {
//         Node* node = bfs_queue[head++];
        
//         for (const auto& edge : node->next_edges()) {
//             if (edge.is_valid()) {
//                 Node* next_node = edge.function.get();
                
//                 NodeTask* task = static_cast<NodeTask*>(next_node->engine_data());
//                 if (!task) {
//                     task = NodeTaskPool::instance().acquire();
//                     ctx->task_storage.push_back(task);
//                     next_node->set_engine_data(task);
//                     bfs_queue.push_back(next_node);
//                 }
//                 task->dependencies++;
//                 if (edge.input_nr > task->max_output_nr) {
//                     task->max_output_nr = edge.input_nr;
//                 }
//             }
//         }
//     }

//     // Execution Setup
//     // Execution Setup - using global pool from TaskFunctor
    
//     // Seed root gradient
//     {
//         NodeTask* root_task = static_cast<NodeTask*>(root_fn->engine_data());
//         uint32_t slot = root.output_nr();
//         auto& grad_vec = (slot < 8) ? root_task->fast_input_grads[slot] : root_task->slow_input_grads_map[slot];
//         grad_vec.push_back(root_grad);
//         if (slot > root_task->max_output_nr) root_task->max_output_nr = slot;
//     }
    
//     // Task Functor for parallel execution
//     struct TaskFunctor {
//         std::shared_ptr<BackwardContext> ctx;
//         Node* node;
        
//         void operator()() {
//             try {
//                 NodeTask* task = static_cast<NodeTask*>(node->engine_data());
                
//                 variable_list node_inputs;
//                 bool has_grad = false;
//                 {
//                     std::lock_guard<std::mutex> lock(task->mutex);
//                     node_inputs.reserve(task->max_output_nr + 1);
//                     node_inputs.resize(task->max_output_nr + 1);
                    
//                     // Process fast path
//                     for (int i = 0; i < 8; ++i) {
//                         auto& grads = task->fast_input_grads[i];
//                         if (!grads.empty()) {
//                             Tensor sum = std::move(grads[0]);
//                             for (size_t j = 1; j < grads.size(); ++j) {
//                                 sum += std::move(grads[j]);
//                             }
//                             node_inputs[i] = sum;
//                             has_grad = true;
//                             GradientVectorPool::instance().release(std::move(grads));
//                         }
//                     }
                    
//                     // Process slow path
//                     for (auto& [slot, grads] : task->slow_input_grads_map) {
//                         if (!grads.empty()) {
//                             Tensor sum = std::move(grads[0]);
//                             for (size_t i = 1; i < grads.size(); ++i) {
//                                 sum += std::move(grads[i]);
//                             }
//                             node_inputs[slot] = sum;
//                             has_grad = true;
//                             GradientVectorPool::instance().release(std::move(grads));
//                         }
//                     }
//                     task->slow_input_grads_map.clear();
//                 }
                
//                 variable_list input_grads;
//                 if (has_grad) {
//                     // Capture input-grad shape for tape annotation before node_inputs is moved.
//                     std::string bwd_shape;
//                     if (autograd::g_shape_debug && GraphRecordMode::get_active()) {
//                         for (const auto& t : node_inputs) {
//                             if (t.is_valid()) {
//                                 bwd_shape = GraphRecordMode::make_shape_info(t.shape(), t.dtype());
//                                 break;
//                             }
//                         }
//                     }
//                     #ifdef AUTOGRAD_PROFILER_ENABLED
//                         bool is_cuda_op = false;
//                         for (const auto& t : node_inputs) {
//                             if (t.is_valid() && t.is_cuda()) {
//                                 is_cuda_op = true;
//                                 break;
//                             }
//                         }

//                         OwnTensor::autograd::ProfileGuard prof_guard(node->name(), is_cuda_op);
//                         input_grads = (*node)(std::move(node_inputs));
//                     #else
//                         input_grads = (*node)(std::move(node_inputs));
//                     #endif

//                     // Record this backward operation (with optional shape annotation)
//                     GraphRecordMode::record_backward(node->name(), std::move(bwd_shape));

//                     // Check for CUDA errors after execution
//                     #ifdef WITH_CUDA
//                     cudaError_t err = cudaGetLastError();
//                     if (err != cudaSuccess) {
//                         throw std::runtime_error("CUDA error during node evaluation (" +
//                                                  std::string(node->name()) + "): " +
//                                                  std::string(cudaGetErrorString(err)));
//                     }
//                     #endif
//                 }
                
//                 // Release saved variables
//                 node->release_saved_variables();
                
//                 // Propagate to dependencies
//                 const auto& edges = node->next_edges();
//                 variable_list output_grads = std::move(input_grads); 
                
//                 // Clear engine_data for this node
//                 node->set_engine_data(nullptr);
                
//                 for (size_t i = 0; i < edges.size(); ++i) {
//                     if (!edges[i].is_valid()) continue;
                    
//                     Node* next_node = edges[i].function.get();
//                     uint32_t slot = edges[i].input_nr;
                    
//                     NodeTask* next_task = static_cast<NodeTask*>(next_node->engine_data());
//                     bool ready = false;
//                     {
//                         std::lock_guard<std::mutex> lock(next_task->mutex);
//                         if (i < output_grads.size() && output_grads[i].is_valid()) {
//                            auto& grad_vec = (slot < 8) ? next_task->fast_input_grads[slot] : next_task->slow_input_grads_map[slot];
//                            if (grad_vec.empty()) {
//                                grad_vec = GradientVectorPool::instance().acquire();
//                            }
//                            grad_vec.push_back(std::move(output_grads[i]));
//                            if (slot > next_task->max_output_nr) next_task->max_output_nr = slot;
//                         }
//                         next_task->dependencies--;
//                         if (next_task->dependencies == 0 && !next_task->scheduled) {
//                             next_task->scheduled = true;
//                             ready = true;
//                         }
//                     }
                    
//                     if (ready) {
//                         schedule(ctx, next_node);
//                     }
//                 }
//             } catch (const std::exception& e) {
//                 // Signal error if possible, but at minimum dec active_tasks so we don't hang
//                 std::cerr << "[Engine] Error in worker thread: " << e.what() << std::endl;
//                 // Note: In an ideal world we'd propagate this back to the main thread
//                 // via a shared atomic error flag or similar in ctx.
//             } catch (...) {
//                 std::cerr << "[Engine] Unknown error in worker thread." << std::endl;
//             }
            
//             // Atomic decrement (CRITICAL: must happen even on error to avoid hang)
//             int remaining = --ctx->active_tasks;
//             if (remaining == 0) {
//                  std::lock_guard<std::mutex> lk(ctx->state_mutex);
//                  ctx->cv.notify_all();
//             }
//         }
        
//         static void schedule(std::shared_ptr<BackwardContext> ctx, Node* node) {
//             ctx->active_tasks++;
//             static utils::ThreadPool& pool = get_pool();
//             pool.enqueue(TaskFunctor{ctx, node});
//         }
        
//         static utils::ThreadPool& get_pool() {
//              static utils::ThreadPool pool(std::thread::hardware_concurrency());
//              return pool;
//         }
//     };
    
//     // Kickoff
//     TaskFunctor::schedule(ctx, root_fn.get());
    
//     // Wait for completion
//     {
//         std::unique_lock<std::mutex> lk(ctx->state_mutex);
//         ctx->cv.wait(lk, [&] { return ctx->active_tasks.load() == 0; });
//     }
    
//     // Safety cleanup: Ensure everything in BFS queue is nulled 
//     // (In case of early exit or unscheduled nodes)
//     for (Node* n : bfs_queue) {
//         n->set_engine_data(nullptr);
//     }
    
//     // Release pool for next step
//     NodeTaskPool::instance().release_all();
// }

// // =============================================================================
// // Main Backward Dispatcher
// // =============================================================================

// void backward(const std::vector<Tensor>& roots, const std::vector<Tensor>& grad_outputs) {
//     if (roots.empty()) return;

//     std::vector<Tensor> processed_grads;
//     processed_grads.reserve(roots.size());

//     for (size_t i = 0; i < roots.size(); ++i) {
//         const auto& root = roots[i];
//         if (!root.requires_grad()) {
//              throw std::runtime_error("backward: root tensor " + std::to_string(i) + " does not require gradients");
//         }

//         if (i < grad_outputs.size() && grad_outputs[i].is_valid()) {
//             processed_grads.push_back(grad_outputs[i]);
//         } else {
//             // Default to ones for scalars, error for non-scalars
//             if (root.ndim() == 0 || root.numel() == 1) {
//                 processed_grads.push_back(Tensor::ones(root.shape(), TensorOptions()
//                     .with_dtype(root.dtype())
//                     .with_device(root.device())));
//             } else {
//                 throw std::runtime_error("backward: non-scalar root " + std::to_string(i) + " requires explicit grad_output");
//             }
//         }
//     }

//     // Since our backends currently only support single root, we iterate for now.
//     // However, if they are supposed to be integrated, we should ideally pass vectors.
//     // For now, to fix the compilation error while maintaining existing backend signatures:
//     ExecutionMode mode = get_execution_mode();
//     for (size_t i = 0; i < roots.size(); ++i) {
//         if (mode == ExecutionMode::SEQUENTIAL) {
//             backward_sequential(roots[i], processed_grads[i]);
//         } else {
//             backward_parallel(roots[i], processed_grads[i]);
//         }
//     }
// }

// void backward(const Tensor& root, const Tensor* grad_output) {
//     std::vector<Tensor> roots = {root};
//     std::vector<Tensor> grads;
//     if (grad_output) {
//         grads.push_back(*grad_output);
//     }
//     backward(roots, grads);
// }

// static thread_local std::shared_ptr<BackwardContext> g_current_context = nullptr;
 
// // Access the global thread pool for the autograd engine
// static utils::ThreadPool& get_engine_pool() {
//     static utils::ThreadPool pool(std::thread::hardware_concurrency());
//     return pool;
// }
 

// void queue_call_back(std::function<void()> callback) {
//     if (get_execution_mode() == ExecutionMode::SEQUENTIAL) {
//          callback();
//     } else {
//         auto ctx = g_current_context;
//         if (ctx) {
//             ctx->active_tasks++;
//             get_engine_pool().enqueue_detach([ctx, cb = std::move(callback)]() {
//                 g_current_context = ctx; try { cb(); } catch (...) {}
//                 if (--ctx->active_tasks == 0) { std::lock_guard<std::mutex> lk(ctx->state_mutex); ctx->cv.notify_all(); }
//                 g_current_context = nullptr;
//             });
//         } else get_engine_pool().enqueue_detach(std::move(callback));
//     }
// }

// } // namespace autograd
// } // namespace OwnTensor