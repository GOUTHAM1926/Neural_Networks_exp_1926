#include "checkpointing/CheckpointNode.h"
#include "checkpointing/GradMode.h"
#include "autograd/ops_template.h"
#include "ops/TensorOps.h"
#include <stdexcept>
#include <deque>
#include <queue>
#include <unordered_map>
#include <vector>


namespace OwnTensor {
namespace autograd {


CheckpointNode::CheckpointNode(
   std::function<variable_list(const variable_list&)> forward_fn,
   const variable_list& inputs,
   RNGState rng_state,
   size_t num_outputs,
   bool offload_to_cpu)
   : Node(inputs.size()),
     forward_fn_(std::move(forward_fn)),
     rng_state_(std::move(rng_state)),
     num_outputs_(num_outputs),
     offload_to_cpu_(offload_to_cpu) {
  
   saved_inputs_.reserve(inputs.size());
   input_requires_grad_.reserve(inputs.size());
  
   for (size_t i = 0; i < inputs.size(); ++i) {
       const auto& input = inputs[i];
      
       // Save original device for restoration
       input_devices_.push_back(input.device());
      
       // Conditional offloading to CPU for storage
       if (offload_to_cpu_) {
           // Note: This adds PCIe overhead but saves VRAM
           Tensor cpu_input = input.to(Device::CPU).detach(); // Detach to save memory
           saved_inputs_.emplace_back(cpu_input, false);
       } else {
           // Detach to save memory (keep storage, drop graph history)
           Tensor detached_input = input.detach();
           saved_inputs_.emplace_back(detached_input, false);
       }
      
       input_requires_grad_.push_back(input.requires_grad());
      
       if (input.requires_grad()) {
           set_next_edge(i, get_grad_edge(input));
       } else {
           set_next_edge(i, Edge{});
       }
   }
}


variable_list CheckpointNode::apply(variable_list&& grads) {
   // Guard against a second call after saved variables have been released.
   // CheckpointNode does not support retain_graph=true.
   if (!forward_fn_) {
       throw std::runtime_error(
           "CheckpointNode::apply called after saved variables were released. "
           "Gradient checkpointing does not support retain_graph=true.");
   }

   // 1. Restore RNG state to ensure deterministic recomputation (e.g., Dropout).
   RNGStateGuard rng_guard;
   RNG::set_state(rng_state_);


   // 3. Unpack inputs and create views to isolate the recomputation graph.
   // This prevents the local backward pass from propagating into the global graph.
   variable_list recompute_inputs;
   recompute_inputs.reserve(saved_inputs_.size());
   for (size_t i = 0; i < saved_inputs_.size(); ++i) {
       const auto& sv = saved_inputs_[i];
       Tensor input = sv.unpack(shared_from_this());
       if (input.unsafeGetTensorImpl()) {
           // Restore to original device (likely GPU)
           Tensor gpu_input = input.to(input_devices_[i]);
          
           // reshape() handles non-contiguous tensors (e.g. after a CPU offload
           // round-trip on a sliced/transposed input); view() would throw.
           Tensor input_view = gpu_input.reshape(gpu_input.shape()).detach();
                     // Important: We set requires_grad on the view BEFORE enabling GradMode
            // so that it acts as a leaf in the local recomputation graph.
            input_view.set_requires_grad(input_requires_grad_[i]);
            
            recompute_inputs.push_back(input_view);
        } else {
           recompute_inputs.push_back(Tensor());
       }
   }

   // 4. Enable gradients for recomputation.
   // The initial forward pass was done in no_grad mode, so we must
   // re-enable it here to build the local computational graph.
   GradModeGuard grad_guard(true);


   // RAII guard to ensure release_saved_variables() is called even if recompute or backward fails.
   struct ReleaseGuard {
       CheckpointNode* node;
       ~ReleaseGuard() { if (node) node->release_saved_variables(); }
   } release_guard{this};


    // 5. Compute the local graph sequence-number floor BEFORE running forward_fn_.
    //
    // By calling get_grad_edge(t) here, we FORCE the lazy instantiation of
    // GradAccumulators for all recompute_inputs *before* any interior ops are
    // created by forward_fn_. This ensures the leaf accumulators get lower
    // sequence numbers than the downstream local operations. If this was done
    // after forward_fn_, interior ops would get instantiated first, get lower
    // sequence numbers, and incorrectly be flagged as outer nodes.
    uint64_t local_seq_floor = UINT64_MAX;
    for (const auto& t : recompute_inputs) {
        if (t.requires_grad()) {
            Edge e = get_grad_edge(t);
            if (e.function) {
                uint64_t s = e.function->sequence_nr();
                if (s < local_seq_floor) local_seq_floor = s;
            }
        }
    }

    // 6. Re-run forward pass to build the local graph.
    variable_list outputs = forward_fn_(recompute_inputs);

    // Validate recomputed output count matches both the original num_outputs_ and
    // the incoming gradient count.
    if (outputs.size() != num_outputs_) {
        throw std::runtime_error(
            "CheckpointNode::apply: Recomputed forward produced " +
            std::to_string(outputs.size()) + " outputs but original forward produced " +
            std::to_string(num_outputs_) + ". forward_fn_ must be deterministic.");
    }
    if (outputs.size() != grads.size()) {
        throw std::runtime_error(
            "CheckpointNode::apply: Number of recomputed outputs (" +
            std::to_string(outputs.size()) + ") does not match number of gradients (" +
            std::to_string(grads.size()) + ")");
    }

    // 7. Run local backward using a self-contained mini sequential engine.
   //    This replaces the previous recursive autograd::backward() call (BUG-4).
   //    run_local_backward() uses only local data structures — no global singletons
   //    (NodeTaskPool, BackwardContext) are touched, so the outer engine's state
   //    is completely unaffected.
   run_local_backward(outputs, grads, local_seq_floor);


   // 9. Collect gradients for inputs
   variable_list input_grads;
   input_grads.reserve(recompute_inputs.size());
   for (size_t i = 0; i < recompute_inputs.size(); ++i) {
       if (input_requires_grad_[i]) {
           if (recompute_inputs[i].has_grad()) {
               // clone() gives an independent tensor whose lifetime does not
               // depend on recompute_inputs[i] or its AutogradMeta (BUG-3 fix).
               Tensor g = recompute_inputs[i].grad_view().clone();
               input_grads.push_back(g);
           } else {
               input_grads.push_back(Tensor());
           }
       } else {
           input_grads.push_back(Tensor());
       }
   }


   // 10. Release all saved state now that gradient collection is complete.
   // Calls the canonical release_saved_variables() which frees forward_fn_,
   // saved_inputs_, and input_devices_ in one place — avoids the partial
   // duplication that previously left forward_fn_ alive until CheckpointNode
   // was destroyed (BUG-10 fix). Must remain after gradient collection (BUG-2):
   // with offload_to_cpu_=false, recompute_inputs shares Storage with
   // saved_inputs_ — releasing early risks a dangling TensorImpl pointer.
   release_saved_variables();
   release_guard.node = nullptr;   // success path: guard already fired — disarm it


   return input_grads;
}


void CheckpointNode::release_saved_variables() {
   if (released_) return;
   released_ = true;

   // Release the forward function (also clears lambda captures / module shared_ptrs).
   forward_fn_ = nullptr;

   // Release the saved inputs.
   for (auto& sv : saved_inputs_) {
       sv.reset();
   }
   saved_inputs_.clear();   // also free the backing storage (consistent with input_devices_)
   input_devices_.clear();
}


// =============================================================================
// run_local_backward — Self-contained mini sequential engine (BUG-4 fix)
// =============================================================================
//
// Design: mirrors backward_sequential() from Engine.cpp but operates entirely
// on local data structures. The outer engine's global singletons (NodeTaskPool,
// engine_data pointers on outer-graph nodes) are never touched.
//
// Flow:
//   Step A — BFS from recomputed outputs: discover every local Node and count
//             how many incoming edges each one has (dependency count).
//   Step B — Seed initial gradients from the `grads` passed into apply().
//   Step C — Execute the ready queue: aggregate grads, call (*node)(),
//             release saved variables, propagate to successors.
//
// Termination: the BFS stops naturally at GradAccumulator nodes whose
// next_edges()[0] is a default-constructed (invalid) Edge{}. When the
// execution loop reaches a GradAccumulator, its apply() writes directly into
// the corresponding recompute_input leaf's AutogradMeta::grad_. The existing
// gradient-collection loop in apply() (lines after this call) then reads those
// values unchanged.
//
void CheckpointNode::run_local_backward(
    const variable_list& outputs,
    const variable_list& grads,
    uint64_t local_seq_floor)
{
    // Local task — mirrors SequentialNodeTask from Engine.cpp but lives
    // entirely on this function's call stack.
    struct LocalNodeTask {
        std::unordered_map<uint32_t, std::vector<Tensor>> grad_slots;
        int      dependencies = 0;
        uint32_t max_slot     = 0;
    };

    // std::deque provides stable addresses even as it grows (raw pointers safe).
    // IMPORTANT: do NOT change this to std::vector — vector reallocation would
    // invalidate the raw pointers stored in local_registry.
    std::deque<LocalNodeTask> task_storage;

    // =========================================================================
    // BUG-9 fix: use a private map instead of Node::engine_data() for all local
    // traversal state.  The outer engine stamps engine_data() on nodes it has
    // already discovered (e.g. GradAccumulator for leaf parameters).  If we read
    // or write engine_data() here we would:
    //   (a) misinterpret the outer engine's SequentialNodeTask* as LocalNodeTask*,
    //   (b) corrupt its dependency counter, and
    //   (c) silently prevent leaf-parameter gradients from ever being written.
    // The map is keyed on raw Node* (stable for the lifetime of this call) and
    // never touches engine_data() at all, fully isolating the two engines.
    // =========================================================================
    std::unordered_map<Node*, LocalNodeTask*> local_registry;

    // Tracks every Node* we registered so we can release saved variables on exit.
    std::vector<Node*> local_nodes;

    std::queue<Node*> ready_queue;

    // =========================================================================
    // Step A: BFS to discover all local-graph nodes and count dependencies.
    //
    // Boundary enforcement (outer-graph contamination fix):
    //
    //   Every Node created during the outer forward pass has a sequence_nr()
    //   strictly less than local_seq_floor (the minimum seq_nr of the
    //   GradAccumulators created for recompute_inputs in apply()).  We call
    //   these "outer nodes".
    //
    //   There are two kinds of outer nodes we may encounter as next() during BFS:
    //
    //     (a) GradAccumulator for a model parameter — a leaf with no valid
    //         next_edges.  These MUST be included in local_registry so that
    //         parameter gradients are accumulated (the outer engine never wires
    //         CheckpointNode to them).  Since they have no valid next_edges,
    //         the BFS would stop naturally, but we make this explicit below.
    //
    //     (b) grad_fn of a tensor captured from the outer graph (e.g. an encoder
    //         output passed into the checkpoint lambda by closure) — a non-leaf
    //         with valid next_edges pointing back into the outer forward graph.
    //         If we were to traverse those edges we would add outer-engine nodes
    //         to local_registry, and LocalCleanupGuard would then call
    //         release_saved_variables() on them, corrupting the outer backward.
    //
    //   Fix: register outer nodes in local_registry (so gradient flows to them)
    //   but do NOT add them to bfs_work, preventing traversal into the outer graph.
    // =========================================================================
    {
        std::vector<Node*> bfs_work;

        // Seed roots: recomputed outputs that have a live grad_fn.
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i].requires_grad()) continue;
            Node* root = outputs[i].grad_fn().get();
            if (!root || local_registry.count(root)) continue;  // dedup multiple outputs sharing same grad_fn

            task_storage.emplace_back();
            local_registry[root] = &task_storage.back();
            local_nodes.push_back(root);
            // Output roots are always local (they were just created by forward_fn_).
            bfs_work.push_back(root);
        }

        size_t head = 0;
        while (head < bfs_work.size()) {
            Node* node = bfs_work[head++];

            for (const auto& edge : node->next_edges()) {
                if (!edge.is_valid()) continue;

                Node* next = edge.function.get();

                if (local_registry.count(next) == 0) {
                    task_storage.emplace_back();
                    local_registry[next] = &task_storage.back();
                    local_nodes.push_back(next);

                    // Only traverse LOCAL graph nodes (seq_nr >= floor).
                    // Outer nodes are registered so gradient can flow to them,
                    // but their children must NOT be visited — that would walk
                    // into the outer engine's graph and let LocalCleanupGuard
                    // release saved state the outer engine still needs.
                    if (next->sequence_nr() >= local_seq_floor) {
                        bfs_work.push_back(next);
                    }
                }

                // Every edge from parent → child is one more dependency for child.
                LocalNodeTask* next_task = local_registry[next];
                next_task->dependencies++;
                if (edge.input_nr > next_task->max_slot) {
                    next_task->max_slot = edge.input_nr;
                }
            }
        }
    }

    // If nothing in the local graph requires grad, there is nothing to do.
    if (local_nodes.empty()) return;

    // =========================================================================
    // Step B: Seed initial gradients into the root nodes
    // =========================================================================
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (!outputs[i].requires_grad()) continue;
        if (i >= grads.size() || !grads[i].unsafeGetTensorImpl()) continue;

        Node* root = outputs[i].grad_fn().get();
        if (!root) continue;

        auto it = local_registry.find(root);
        if (it == local_registry.end()) continue;
        LocalNodeTask* task = it->second;

        // output_nr() is the slot index this tensor occupies in its grad_fn's
        // output list — same semantics as backward_sequential's seeding step.
        uint32_t slot = outputs[i].output_nr();
        task->grad_slots[slot].push_back(grads[i]);
        if (slot > task->max_slot) task->max_slot = slot;
    }

    // Nodes with zero dependencies are immediately ready.
    for (Node* node : local_nodes) {
        if (local_registry[node]->dependencies == 0) {
            ready_queue.push(node);
        }
    }

    // =========================================================================
    // Exception-safe cleanup: release saved variables on unprocessed LOCAL nodes.
    //
    // Outer nodes (seq_nr < local_seq_floor) must NOT have release_saved_variables
    // called on them — the outer engine still owns that state.  Only release nodes
    // that were created during recomputation (seq_nr >= local_seq_floor).
    // =========================================================================
    struct LocalCleanupGuard {
        std::vector<Node*>& nodes;
        std::unordered_map<Node*, LocalNodeTask*>& registry;
        uint64_t seq_floor;
        ~LocalCleanupGuard() {
            for (Node* n : nodes) {
                if (registry.count(n) && n->sequence_nr() >= seq_floor) {
                    n->release_saved_variables();
                }
            }
        }
    } cleanup{local_nodes, local_registry, local_seq_floor};

    // =========================================================================
    // Step C: Execute ready queue — identical logic to backward_sequential's
    //         inner loop but operating entirely on local data.
    // =========================================================================
    while (!ready_queue.empty()) {
        Node* node = ready_queue.front();
        ready_queue.pop();

        LocalNodeTask* task = local_registry.at(node);

        // Aggregate all incoming gradients for this node.
        variable_list node_inputs(task->max_slot + 1);
        bool has_grad = false;

        for (auto& [slot, grad_vec] : task->grad_slots) {
            if (grad_vec.empty()) continue;
            Tensor sum = std::move(grad_vec[0]);
            for (size_t k = 1; k < grad_vec.size(); ++k) {
                sum = sum + std::move(grad_vec[k]);
            }
            node_inputs[slot] = std::move(sum);
            has_grad = true;
        }
        task->grad_slots.clear();

        // Execute the node.
        //
        // For OUTER nodes (seq_nr < local_seq_floor) we only call through to
        // leaf accumulators (GradAccumulator nodes with no valid next_edges).
        // These accumulate parameter gradients — they must be called, and it is
        // safe to do so because they have no children to corrupt.
        //
        // Non-leaf outer nodes (e.g. grad_fns of captured encoder outputs) must
        // NOT be called here: the outer engine will process them via its own
        // traversal; calling them prematurely would run their backward twice and
        // potentially free their saved state before the outer engine uses it.
        variable_list output_grads;
        if (has_grad) {
            const bool is_outer = (node->sequence_nr() < local_seq_floor);
            if (!is_outer) {
                // Local node — always execute.
                output_grads = (*node)(std::move(node_inputs));
            } else {
                // Outer node — only execute if it is a leaf accumulator
                // (all next_edges are invalid).
                const bool is_leaf_accumulator = std::all_of(
                    node->next_edges().begin(), node->next_edges().end(),
                    [](const Edge& e) { return !e.is_valid(); });
                if (is_leaf_accumulator) {
                    output_grads = (*node)(std::move(node_inputs));
                }
                // else: non-leaf outer node — skip execution; gradient for this
                // path is not propagated. The outer engine handles it separately.
                // Note: if you need gradients through captured non-leaf tensors,
                // pass them as explicit inputs to checkpoint() instead of capturing
                // them by closure.
            }
        }

        // Release saved tensors only for LOCAL graph nodes.
        // Outer nodes' saved state belongs to the outer engine — do not touch it.
        if (node->sequence_nr() >= local_seq_floor) {
            node->release_saved_variables();
        }
        local_registry.erase(node);

        // Propagate gradients to successor nodes.
        const auto& edges = node->next_edges();
        for (size_t j = 0; j < edges.size(); ++j) {
            if (!edges[j].is_valid()) continue;

            Node* next = edges[j].function.get();
            auto it = local_registry.find(next);
            if (it == local_registry.end()) continue;  // outside local graph — safe to skip
            LocalNodeTask* next_task = it->second;

            if (j < output_grads.size() && output_grads[j].unsafeGetTensorImpl()) {
                uint32_t slot = edges[j].input_nr;
                next_task->grad_slots[slot].push_back(std::move(output_grads[j]));
                if (slot > next_task->max_slot) next_task->max_slot = slot;
            }

            next_task->dependencies--;
            if (next_task->dependencies == 0) {
                ready_queue.push(next);
            }
        }
    }
    // cleanup destructor fires here: releases local unprocessed nodes on exception,
    // skips outer nodes (seq_nr < local_seq_floor) whose state belongs to the
    // outer engine.
}


} // namespace autograd
} // namespace OwnTensor