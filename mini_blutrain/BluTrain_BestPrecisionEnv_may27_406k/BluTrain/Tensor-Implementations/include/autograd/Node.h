#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <mutex>
#include <atomic>
#include "core/IntrusivePtr.h"
#include "autograd/GraphArena.h"

namespace OwnTensor {

// Forward declarations
class Node;
class Tensor;

// Type aliases (PyTorch-style)
using variable_list = std::vector<Tensor>;
using edge_list = std::vector<struct Edge>;

// Global sequence number generator for node ordering
namespace detail {
    inline std::atomic<uint64_t>& get_sequence_counter() {
        static std::atomic<uint64_t> counter{0};
        return counter;
    }
    
    inline uint64_t get_and_increment_sequence() {
        return get_sequence_counter().fetch_add(1);
    }
}

/**
 * @brief Edge in the computational graph connecting nodes.
 * 
 * An edge represents a connection from one node to another in the backward
 * computational graph. It stores which function to call next and which input
 * slot of that function this edge corresponds to.
 * 
 * ## Usage Example
 * ```cpp
 * // Create an edge to a node
 * auto node = std::make_shared<MyBackwardNode>();
 * Edge edge(node, 0);  // Connect to input 0
 * 
 * // Check validity
 * if (edge.is_valid()) {
 *     // Edge has a valid function
 * }
 * ```
 */
struct Edge {
    std::shared_ptr<Node> function;  ///< Next function in backward graph
    uint32_t input_nr;                ///< Which input of the function (for multi-input ops)

    /// Default constructor - creates an invalid edge
    Edge() : function(nullptr), input_nr(0) {}
    
    /// Construct with function and input number
    Edge(std::shared_ptr<Node> function_, uint32_t input_nr_)
        : function(std::move(function_)), input_nr(input_nr_) {}

    /// Check if this edge is valid (has a function)
    bool is_valid() const {
        return function != nullptr;
    }
    
    /// Equality comparison
    bool operator==(const Edge& other) const {
        return function == other.function && input_nr == other.input_nr;
    }
    
    bool operator!=(const Edge& other) const {
        return !(*this == other);
    }
};

/**
 * @brief Base class for gradient functions (operations in computational graph).
 * 
 * Node represents an operation in the computational graph. During the backward
 * pass, nodes receive gradients and compute gradients for their inputs.
 * 
 * ## Key Features
 * - **Sequence number**: Unique ID for ordering operations
 * - **Topological number**: For efficient graph traversal
 * - **Thread safety**: Mutex protection for concurrent access
 * - **Hook system**: Pre/post hooks for gradient manipulation
 * 
 * ## Usage Example
 * ```cpp
 * // Create a custom backward node
 * class MyBackward : public Node {
 * public:
 *     MyBackward() : Node(2) {}  // 2 inputs
 *     
 *     std::string name() const override { return "MyBackward"; }
 *     
 *     variable_list apply(variable_list&& grads) override {
 *         // Compute gradients for inputs
 *         return {grad_input1, grad_input2};
 *     }
 * };
 * 
 * // Use the node
 * auto node = std::make_shared<MyBackward>();
 * node->set_next_edge(0, edge_to_input1);
 * node->set_next_edge(1, edge_to_input2);
 * 
 * // Execute with hook wrapping
 * auto grads = (*node)(std::move(input_grads));
 * ```
 * 
 * ## Why These Methods Exist
 * 
 * | Method | Purpose |
 * |--------|---------|
 * | `sequence_nr()` | Ordering nodes for deterministic execution |
 * | `topological_nr()` | Efficient graph traversal in backward pass |
 * | `name()` | Debugging and error messages |
 * | `operator()` | Hook-wrapped execution of apply() |
 * | `next_edges()` | Access edges to predecessor nodes |
 */
class Node : public std::enable_shared_from_this<Node> {
public:
    // =========================================================================
    // Constructors
    // =========================================================================
    
    /// Default constructor - allocates a sequence number
    Node() 
        : sequence_nr_(detail::get_and_increment_sequence()),
          topological_nr_(0),
          thread_id_(0) {}
    
    /// Construct with specific number of inputs
    /// @param num_inputs Number of inputs this node will have
    explicit Node(size_t num_inputs) 
        : sequence_nr_(detail::get_and_increment_sequence()),
          topological_nr_(0),
          thread_id_(0) {
        next_edges_.resize(num_inputs);
    }
    
    // =========================================================================
    // Custom Memory Management (Arena)
    // =========================================================================
    
    static std::atomic<bool> use_arena_;
    
    void* operator new(size_t size) {
        if (use_arena_.load(std::memory_order_relaxed)) {
            return autograd::GraphArena::get_thread_local().allocate(size);
        }
        return ::operator new(size);
    }
    
    void operator delete(void* ptr) {
        if (use_arena_.load(std::memory_order_relaxed)) {
            // No-op: memory is managed by the arena and reclaimed in bulk via reset()
            (void)ptr;
        } else {
            ::operator delete(ptr);
        }
    }
    
    /// Construct with pre-defined edges (advanced usage)
    /// @param next_edges List of edges to next nodes
    explicit Node(edge_list&& next_edges)
        : sequence_nr_(detail::get_and_increment_sequence()),
          topological_nr_(0),
          thread_id_(0),
          next_edges_(std::move(next_edges)) {
        // Compute initial topological number from edges
        for (const auto& edge : next_edges_) {
            update_topological_nr(edge);
        }
    }

    /// Nodes are neither copyable nor moveable
    Node(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(const Node&) = delete;
    Node& operator=(Node&&) = delete;
    
    virtual ~Node() = default;

    // =========================================================================
    // Core Execution
    // =========================================================================
    
    /**
     * @brief Execute the node with hook wrapping.
     * 
     * This is the main entry point for executing a backward function.
     * It handles pre-hooks, calls apply(), and post-hooks.
     * 
     * @param inputs Gradients with respect to outputs
     * @return Gradients with respect to inputs
     */
    variable_list operator()(variable_list&& inputs);
    
    /**
     * @brief Apply the gradient function (backward pass).
     * 
     * Override this in subclasses to compute gradients.
     * 
     * @param grads Gradients with respect to outputs of this function
     * @return Gradients with respect to inputs of this function
     */
    virtual variable_list apply(variable_list&& grads) = 0;

    // =========================================================================
    // Identification
    // =========================================================================
    
    /**
     * @brief Get the name of this node type for debugging.
     * 
     * Override in subclasses to return meaningful names like "AddBackward".
     */
    virtual const char* name() const { return "Node"; }
    
    /**
     * @brief Get the sequence number (unique ID for ordering).
     */
    uint64_t sequence_nr() const { return sequence_nr_; }
    
    /**
     * @brief Get the topological number for graph traversal.
     */
    uint64_t topological_nr() const { return topological_nr_; }
    
    /**
     * @brief Get the thread ID that created this node.
     */
    uint64_t thread_id() const { return thread_id_; }

    // =========================================================================
    // Edge Management
    // =========================================================================
    
    /**
     * @brief Get the edges to the next functions in the graph.
     */
    const edge_list& next_edges() const {
        return next_edges_;
    }
    
    /**
     * @brief Get a specific edge by index.
     */
    const Edge& next_edge(size_t index) const {
        return next_edges_.at(index);
    }

    /**
     * @brief Set an edge for a specific input.
     */
    void set_next_edge(uint32_t index, Edge edge) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= next_edges_.size()) {
            next_edges_.resize(index + 1);
        }
        update_topological_nr(edge);
        next_edges_[index] = std::move(edge);
    }

    /**
     * @brief Add an edge to the list.
     */
    void add_next_edge(Edge edge) {
        std::lock_guard<std::mutex> lock(mutex_);
        update_topological_nr(edge);
        next_edges_.push_back(std::move(edge));
    }

    /**
     * @brief Get number of inputs to this function.
     */
    size_t num_inputs() const {
        return next_edges_.size();
    }

    /**
     * @brief Clear all next edges (used for graph cleanup).
     */
    void clear_edges() {
        std::lock_guard<std::mutex> lock(mutex_);
        next_edges_.clear();
    }
    
    /**
     * @brief Release saved variables to free memory after backward.
     * 
     * Override in subclasses to reset any SavedVariable members.
     * Called by the autograd engine after apply() to prevent memory accumulation.
     */
    virtual void release_saved_variables() {
        // Default: do nothing. Subclasses should override.
    }
    
    // =========================================================================
    // Graph Information
    // =========================================================================
    
    /**
     * @brief Get the output number for this node's output.
     * 
     * For nodes that produce multiple outputs, this indicates
     * which output this edge corresponds to.
     */
    uint32_t output_nr() const { return 0; }  // Default single output

protected:
    /// Update topological number based on an edge
    void update_topological_nr(const Edge& edge) {
        if (edge.is_valid()) {
            auto topo = edge.function->topological_nr();
            if (topo >= topological_nr_) {
                topological_nr_ = topo + 1;
            }
        }
    }

    // =========================================================================
    // Protected Members
    // =========================================================================
    
    /// Sequence number for ordering
    uint64_t sequence_nr_;
    
    /// Topological number for graph traversal (max of inputs + 1)
    uint64_t topological_nr_;
    
    /// Thread ID of the forward operator
    uint64_t thread_id_;
    
    /// Edges to the next functions in the backward graph
    edge_list next_edges_;
    
    /// Mutex for thread-safe access
    mutable std::mutex mutex_;
    
    /// Engine data pointer for backward pass (avoids hash map lookups)
    /// This is set/cleared by the engine during backward traversal.
    void* engine_data_ = nullptr;
    
    /// Pre-hooks executed before apply()
    std::vector<std::function<variable_list(variable_list&)>> pre_hooks_;
    
    /// Post-hooks executed after apply()
    std::vector<std::function<void(const variable_list&, const variable_list&)>> post_hooks_;

public:
    // =========================================================================
    // Hook Registration
    // =========================================================================
    
    /**
     * @brief Register a pre-hook to be called before apply().
     * Pre-hooks can modify the inputs.
     */
    void register_pre_hook(std::function<variable_list(variable_list&)> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        pre_hooks_.push_back(std::move(hook));
    }
    
    /**
     * @brief Register a post-hook to be called after apply().
     * Post-hooks receive inputs and outputs (read-only).
     */
    void register_post_hook(std::function<void(const variable_list&, const variable_list&)> hook) {
        std::lock_guard<std::mutex> lock(mutex_);
        post_hooks_.push_back(std::move(hook));
    }

    /**
     * @brief Register a post-hook to be called after apply().
     * This is a convenience wrapper for register_post_hook that matches PyTorch API.
     */
    void register_hook(std::function<void(const variable_list&, const variable_list&)> hook) {
        register_post_hook(std::move(hook));
    }
    

    /**
     * @brief Get number of pre-hooks.
     */
    size_t num_pre_hooks() const { return pre_hooks_.size(); }
    
    /**
     * @brief Get number of post-hooks.
     */
    size_t num_post_hooks() const { return post_hooks_.size(); }
    
    // =========================================================================
    // Engine Data Accessors
    // =========================================================================
    
    /**
     * @brief Get the engine data pointer (used by backward engine).
     */
    void* engine_data() const { return engine_data_; }
    
    /**
     * @brief Set the engine data pointer (used by backward engine).
     */
    void set_engine_data(void* data) { engine_data_ = data; }
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Create an edge to a node.
 * 
 * ## Usage
 * ```cpp
 * auto node = std::make_shared<AddBackward>();
 * Edge edge = make_edge(node, 0);  // Edge to input 0
 * ```
 */
inline Edge make_edge(std::shared_ptr<Node> node, uint32_t input_nr = 0) {
    return Edge(std::move(node), input_nr);
}

} // namespace OwnTensor
