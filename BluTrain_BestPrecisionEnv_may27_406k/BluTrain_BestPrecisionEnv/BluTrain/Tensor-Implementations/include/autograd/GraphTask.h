#pragma once

/**
 * @file GraphTask.h
 * @brief Graph traversal task management for backward pass.
 * 
 * GraphTask manages the state of a backward pass execution:
 * - Node dependencies tracking
 * - Keep/create graph flags
 * - Execution progress monitoring
 */

#include "autograd/Node.h"
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Represents a backward pass execution task.
 * 
 * Tracks dependencies between nodes and manages execution state.
 */
class GraphTask {
public:
    // =========================================================================
    // Configuration
    // =========================================================================
    
    /// Keep graph after backward (for multiple backward passes)
    bool keep_graph_{false};
    
    /// Create graph for higher-order gradients
    bool create_graph_{false};
    
    /// Whether this task completed successfully
    std::atomic<bool> completed_{false};
    
    /// Whether an exception occurred
    std::atomic<bool> has_error_{false};
    
    /// Exception message if any
    std::string exception_message_;
    
    // =========================================================================
    // Dependency Tracking
    // =========================================================================
    
    /// Map from node to number of outstanding dependencies
    std::unordered_map<Node*, std::atomic<int>> dependencies_;
    
    /// Set of nodes that are part of this graph traversal
    std::unordered_set<Node*> nodes_in_graph_;
    
    /// Mutex for thread-safe access
    mutable std::mutex mutex_;
    
    /// Condition variable for completion waiting
    std::condition_variable cv_;
    
    // =========================================================================
    // Constructors
    // =========================================================================
    
    GraphTask() = default;
    
    GraphTask(bool keep_graph, bool create_graph)
        : keep_graph_(keep_graph), create_graph_(create_graph) {}
    
    // =========================================================================
    // Methods
    // =========================================================================
    
    /**
     * @brief Initialize dependencies for a node.
     */
    void init_to_execute(Node* node, int dependency_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        nodes_in_graph_.insert(node);
        dependencies_[node].store(dependency_count);
    }
    
    /**
     * @brief Decrement dependency count for a node.
     * @return True if node is now ready to execute (dependencies == 0)
     */
    bool decrement_dependency(Node* node) {
        auto it = dependencies_.find(node);
        if (it != dependencies_.end()) {
            return it->second.fetch_sub(1) == 1;
        }
        return false;
    }
    
    /**
     * @brief Check if a node is part of this graph task.
     */
    bool is_node_in_graph(Node* node) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_in_graph_.find(node) != nodes_in_graph_.end();
    }
    
    /**
     * @brief Mark task as complete.
     */
    void mark_completed() {
        completed_.store(true);
        cv_.notify_all();
    }
    
    /**
     * @brief Mark task as failed with error.
     */
    void mark_failed(const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        has_error_.store(true);
        exception_message_ = error;
        completed_.store(true);
        cv_.notify_all();
    }
    
    /**
     * @brief Wait for task completion.
     */
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return completed_.load(); });
    }
    
    /**
     * @brief Check if task has error and throw if so.
     */
    void check_error() const {
        if (has_error_.load()) {
            throw std::runtime_error(exception_message_);
        }
    }
};

/**
 * @brief Work item for the backward pass queue.
 */
struct NodeTask {
    std::shared_ptr<Node> fn_;
    std::vector<Tensor> inputs_;
    std::shared_ptr<GraphTask> graph_task_;
    
    NodeTask() = default;
    
    NodeTask(std::shared_ptr<Node> fn, 
             std::vector<Tensor> inputs,
             std::shared_ptr<GraphTask> task)
        : fn_(std::move(fn))
        , inputs_(std::move(inputs))
        , graph_task_(std::move(task)) {}
};

} // namespace autograd
} // namespace OwnTensor
