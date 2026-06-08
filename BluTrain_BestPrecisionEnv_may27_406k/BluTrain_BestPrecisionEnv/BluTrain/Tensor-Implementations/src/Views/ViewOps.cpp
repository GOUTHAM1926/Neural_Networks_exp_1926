#include "core/Views/ViewUtils.h"
#include "core/Tensor.h"
#include "core/TensorImpl.h"
#include "core/TensorDispatch.h"
#include "device/DeviceTransfer.h"
#include "autograd/backward/TransposeBackward.h"
#include "autograd/backward/ReshapeBackward.h"
#include "autograd/ops_template.h"
#include "checkpointing/GradMode.h"
#include "autograd/GraphRecorder.h"
#include <stdexcept>
#include <numeric>

namespace OwnTensor {

// ============================================================================
// View Operations - Updated for TensorImpl Architecture
// ============================================================================

Tensor Tensor::view(Shape new_shape) const {
    if (!impl_) {
        throw std::runtime_error("view: tensor is not initialized");
    }
    
    // Check if tensor is contiguous
    if (!is_contiguous()) {
        throw std::runtime_error(
            "view() requires contiguous tensor. "
            "Use reshape() or call contiguous() first."
        );
    }
    
    // Handle -1 dimension inference
    ViewUtils::infer_dimension(numel(), new_shape);
    
    // Validate that total elements match
    if (!ViewUtils::is_shape_compatible(numel(), new_shape)) {
        throw std::runtime_error(
            "view: Shape mismatch - new shape must have same number of elements"
        );
    }
    
    // Compute new strides for the new shape
    Stride new_stride = ViewUtils::compute_strides(new_shape);
    
    // Use private constructor that shares storage
    Tensor result(impl_, new_shape, new_stride, storage_offset());

    if (requires_grad()) {
        auto grad_fn = std::make_shared<autograd::ReshapeBackward>(shape());
        grad_fn->set_next_edge(0, autograd::get_grad_edge(*this));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }
    return result;
}

Tensor Tensor::reshape(Shape new_shape) const {
    autograd::GraphRecordMode::record_forward("View: reshape");
    if (!impl_) {

        throw std::runtime_error("reshape: tensor is not initialized");
    }
    
    // Infer any -1 dimension
    ViewUtils::infer_dimension(numel(), new_shape);
    
    // Validate element count
    if (!ViewUtils::is_shape_compatible(numel(), new_shape)) {
        throw std::runtime_error("reshape: new shape has different number of elements");
    }
    
    // Smart reshape (PyTorch-style): if the new shape can be expressed as a
    // pure VIEW of the source (no data movement), do it as metadata-only.
    // Catches the common QKV-split case where source is non-contig overall
    // but the inner dims being merged/split are locally contig.
    // PyTorch ref: aten/src/ATen/TensorUtils.cpp:327 (computeStride_impl).
    Stride view_stride = ViewUtils::compute_view_stride(shape(), stride(), new_shape);
    if (!view_stride.strides.empty()) {
        Tensor result(impl_, new_shape, view_stride, storage_offset());
        if (requires_grad()) {
            auto grad_fn = std::make_shared<autograd::ReshapeBackward>(shape());
            Tensor& self_mut = const_cast<Tensor&>(*this);
            grad_fn->set_next_edge(0, autograd::get_grad_edge(self_mut));
            result.set_grad_fn(grad_fn);
            result.set_requires_grad(true);
        }
        if (autograd::g_shape_debug)
            autograd::GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
        return result;
    }

    // No view possible — materialize contiguous copy on same device then view.
    Tensor base = contiguous();
    Stride new_stride = ViewUtils::compute_strides(new_shape);
    Tensor result(base.impl_, new_shape, new_stride, base.storage_offset());

    if (requires_grad()) {
        auto grad_fn = std::make_shared<autograd::ReshapeBackward>(shape());
        grad_fn->set_next_edge(0, autograd::get_grad_edge(*this));
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }

    if (autograd::g_shape_debug)
        autograd::GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor Tensor::transpose(int dim0, int dim1) const
{
    autograd::GraphRecordMode::record_forward("View: transpose");
    if (!impl_) {
        throw std::runtime_error("transpose: tensor is not initialized");
    }
    
    int ndim = static_cast<int>(impl_->sizes().dims.size());
    dim0 = ViewUtils::normalize_dim(dim0, ndim);
    dim1 = ViewUtils::normalize_dim(dim1, ndim);
    
    if (dim0 == dim1) {
        // No-op: return a view with identical metadata
        Tensor noop(impl_, impl_->sizes(), impl_->strides(), storage_offset());
        if (autograd::g_shape_debug)
            autograd::GraphRecordMode::attach_forward_shape(noop.shape(), noop.dtype());
        return noop;
    }

    Shape new_shape = impl_->sizes();
    Stride new_stride = impl_->strides();
    ViewUtils::swap_dimensions(new_shape, new_stride, dim0, dim1);

    Tensor result(impl_, new_shape, new_stride, storage_offset());

    // Autograd support — only when grad tracking is active (respects NoGradGuard)
    if (requires_grad() && autograd::GradMode::is_enabled()) {
        auto grad_fn = std::make_shared<autograd::TransposeBackward>(dim0, dim1);
        
        grad_fn->set_next_edge(0, autograd::get_grad_edge(*this));
        
        result.set_grad_fn(grad_fn);
        result.set_requires_grad(true);
    }

    if (autograd::g_shape_debug)
        autograd::GraphRecordMode::attach_forward_shape(result.shape(), result.dtype());
    return result;
}

Tensor Tensor::t() const {
    if (impl_->sizes().dims.size() < 2) return *this;
    return transpose(-2, -1);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const
{
    if (!impl_) {
        throw std::runtime_error("flatten: tensor is not initialized");
    }
    
    int ndim = static_cast<int>(impl_->sizes().dims.size());
    start_dim = ViewUtils::normalize_dim(start_dim, ndim);
    end_dim   = ViewUtils::normalize_dim(end_dim < 0 ? end_dim + ndim : end_dim, ndim);
    
    if (start_dim > end_dim) {
        throw std::runtime_error("flatten: start_dim must be <= end_dim");
    }

    // Will throw if any participating dims are non-positive
    Shape new_shape = ViewUtils::compute_flatten_shape(impl_->sizes(), start_dim, end_dim);

    if (is_contiguous()) {
        Stride new_stride = ViewUtils::compute_strides(new_shape);
        return Tensor(impl_, new_shape, new_stride, storage_offset());
    } else {
        Tensor base = contiguous();
        Stride new_stride = ViewUtils::compute_strides(new_shape);
        return Tensor(base.impl_, new_shape, new_stride, base.storage_offset());
    }
}

Tensor Tensor::unflatten(int dim, Shape sizes) const
{
    if (!impl_) {
        throw std::runtime_error("unflatten: tensor is not initialized");
    }
    
    int ndim = static_cast<int>(impl_->sizes().dims.size());
    dim = ViewUtils::normalize_dim(dim, ndim);

    // Validates: sizes positive, product equals shape_[dim]
    ViewUtils::validate_unflatten(impl_->sizes(), dim, sizes);

    Shape new_shape  = ViewUtils::compute_unflatten_shape(impl_->sizes(), dim, sizes);
    Stride new_stride = ViewUtils::compute_unflatten_strides(impl_->strides(), dim, sizes);

    return Tensor(impl_, new_shape, new_stride, storage_offset());
}


} // namespace OwnTensor