#include "autograd/Variable.h"
#include "autograd/Functions.h"
#include "core/TensorImpl.h"

namespace OwnTensor {
namespace impl {

Edge gradient_edge(const Tensor& self) {
    // If grad_fn exists, use it (non-leaf / interior variable)
    if (const auto& gradient = self.grad_fn()) {
        return Edge(gradient, self.output_nr());
    }
    
    // Otherwise, for leaf variables that require grad,
    // use the gradient accumulator
    if (self.requires_grad()) {
        return Edge(grad_accumulator(self), 0);
    }
    
    // No gradient needed
    return Edge{};
}

void set_gradient_edge(const Tensor& self, Edge edge) {
    auto* meta = materialize_autograd_meta(self);
    if (meta) {
        meta->set_grad_fn(std::move(edge.function));
        meta->set_output_nr(edge.input_nr);
    }
}

std::shared_ptr<Node> grad_accumulator(const Tensor& self) {
    auto* meta = get_autograd_meta(self);
    if (!meta) return nullptr;
    
    // Try to get existing accumulator
    auto accumulator = meta->grad_accumulator_.lock();
    if (!accumulator) {
        // Create new accumulator
        accumulator = std::make_shared<autograd::GradAccumulator>(
            self.unsafeGetTensorImpl()
        );
        meta->grad_accumulator_ = accumulator;
    }
    return accumulator;
}

void set_grad_accumulator(const Tensor& self, std::weak_ptr<Node> acc) {
    auto* meta = materialize_autograd_meta(self);
    if (meta) {
        meta->grad_accumulator_ = std::move(acc);
    }
}

} // namespace impl

// Factory functions

Tensor make_variable(const Tensor& data, Edge gradient_edge) {
    if (!data.unsafeGetTensorImpl()) {
        return Tensor();  // Invalid tensor
    }
    
    // Clone the tensor to create a new variable
    Tensor result = data.clone();
    
    // Attach the gradient edge
    impl::set_gradient_edge(result, std::move(gradient_edge));
    
    return result;
}

void create_gradient_edge(Tensor& variable, std::shared_ptr<Node> function) {
    // Get the input number (next available edge index)
    uint32_t input_nr = static_cast<uint32_t>(function->num_inputs());
    
    // Add edge to function
    function->add_next_edge(impl::gradient_edge(variable));
    
    // Set the gradient edge on the variable
    impl::set_gradient_edge(variable, Edge(std::move(function), input_nr));
}

} // namespace OwnTensor
