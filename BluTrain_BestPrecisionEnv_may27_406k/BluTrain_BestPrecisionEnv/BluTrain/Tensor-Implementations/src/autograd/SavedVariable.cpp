#include "autograd/SavedVariable.h"
#include "core/TensorImpl.h"
#include "core/AutogradMeta.h"

namespace OwnTensor {

SavedVariable::SavedVariable(const Tensor& variable, bool is_output, bool /*is_inplace_on_view*/)
    : is_output_(is_output) {
    
    if (!variable.unsafeGetTensorImpl()) {
        return;  // Empty saved variable
    }
    
    // Save version for in-place detection
    saved_version_ = variable.unsafeGetTensorImpl()->version();
    
    // Record gradient tracking info
    requires_grad_ = variable.requires_grad();
    was_leaf_ = variable.is_leaf();
    output_nr_ = variable.output_nr();
    
    // For non-leaf tensors, save weak reference to grad_fn
    if (!was_leaf_ && variable.grad_fn()) {
        weak_grad_fn_ = variable.grad_fn();
    }
    
    // Save the tensor (detached - shares storage but breaks graph cycles)
    data_ = variable.detach();
}

Tensor SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
    if (!defined()) {
        return Tensor();
    }
    
    // Check for in-place modification
    if (data_.unsafeGetTensorImpl()) {
        uint32_t current_version = data_.unsafeGetTensorImpl()->version();
        
        if (current_version != saved_version_) {
            std::ostringstream message;
            message << "one of the variables needed for gradient computation has been "
                    << "modified by an inplace operation";
            
            if (saved_for) {
                message << ", which is " << (is_output_ ? "output" : "input")
                        << " of " << saved_for->name();
            }
            
            message << ", is at version " << current_version
                    << "; expected version " << saved_version_ << " instead.";
            
            throw std::runtime_error(message.str());
        }
    }
    
    return data_;
}

bool was_modified_inplace(const Tensor& tensor, uint32_t saved_version) {
    if (!tensor.unsafeGetTensorImpl()) {
        return false;
    }
    return tensor.unsafeGetTensorImpl()->version() != saved_version;
}

void increment_version(const Tensor& tensor) {
    if (tensor.unsafeGetTensorImpl()) {
        tensor.unsafeGetTensorImpl()->bump_version();
    }
}

} // namespace OwnTensor
