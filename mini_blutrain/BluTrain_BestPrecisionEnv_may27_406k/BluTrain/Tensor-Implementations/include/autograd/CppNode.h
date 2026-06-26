#pragma once

/**
 * @file CppNode.h
 * @brief Base class for custom C++ autograd functions.
 * 
 * Provides the template for creating custom backward functions
 * similar to PyTorch's torch.autograd.Function.
 */

#include "autograd/Node.h"
#include "autograd/AutogradContext.h"
#include "autograd/ops_template.h"
#include <memory>

namespace OwnTensor {
namespace autograd {

/**
 * @brief Base template for custom autograd functions.
 * 
 * ## Usage
 * ```cpp
 * class MyOp : public CppNode<MyOp> {
 * public:
 *     // Forward pass (static method)
 *     static variable_list forward(AutogradContext* ctx,
 *                                   const Tensor& a, const Tensor& b) {
 *         ctx->save_for_backward({a, b});
 *         return {a * b};
 *     }
 *     
 *     // Backward pass (static method)
 *     static variable_list backward(AutogradContext* ctx,
 *                                    const variable_list& grad_outputs) {
 *         auto saved = ctx->get_saved_variables();
 *         return {grad_outputs[0] * saved[1],
 *                 grad_outputs[0] * saved[0]};
 *     }
 * };
 * 
 * // Usage:
 * auto result = MyOp::apply(a, b);
 * ```
 */
template<typename T>
class CppNode : public Node {
protected:
    AutogradContext ctx_;
    
public:
    CppNode() : Node(0) {
        // Do not call shared_from_this() in constructor
    }
    
    const char* name() const override {
        return "CppNode";  // Subclasses can override
    }
    
    variable_list apply(variable_list&& grads) override {
        return T::backward(&ctx_, grads);
    }
    
    template<typename NodeType, typename Arg>
    static void connect_arg(std::shared_ptr<NodeType>& node, uint32_t& index, Arg&& arg) {
        using ArgType = std::decay_t<Arg>;
        if constexpr (std::is_same_v<ArgType, Tensor>) {
             // It is a tensor
             const Tensor& t = arg;
             if (t.requires_grad()) {
                 node->set_next_edge(index, get_grad_edge(t));
             }
        }
        index++;
    }

    /**
     * @brief Apply the function with gradient tracking.
     * 
     * This is the main entry point for using the custom function.
     */
    template<typename... Args>
    static variable_list apply(Args&&... args) {
        // Create node
        auto node = std::make_shared<T>();
        
        // Ensure next_edges_ is sized to match arguments
        if constexpr (sizeof...(Args) > 0) {
            node->set_next_edge(sizeof...(Args) - 1, Edge());
        }
        
        // Initialize context with safe shared_ptr
        node->ctx_.set_grad_fn(node);
        
        // Connect edges for inputs
        uint32_t index = 0;
        (connect_arg(node, index, std::forward<Args>(args)), ...);
        
        // Call forward
        variable_list outputs = T::forward(&node->ctx_, std::forward<Args>(args)...);
        
        // Set grad_fn on outputs
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i].requires_grad()) {
                outputs[i].set_grad_fn(node);
                outputs[i].set_output_nr(i);
            }
        }
        
        return outputs;
    }
    
    /**
     * @brief Release saved variables after backward.
     */
    void release_variables() {
        ctx_.release_variables();
    }
};

} // namespace autograd
} // namespace OwnTensor
