#pragma once

#include <vector>
#include <memory>
#include "device/Device.h"
#include "dtype/Dtype.h"
#include "dtype/Types.h"
#include "core/TensorImpl.h"
#include "core/Shape.h"
#include "core/Stride.h"
#include "device/AllocatorRegistry.h"

namespace OwnTensor
{
    // Forward declarations
    class TensorImpl;
    class Node;
    class FunctionPreHook;
    class PostAccumulateGradHook;

    // Tensor Utility options for smoother API
    struct TensorOptions
    {
        Dtype dtype = Dtype::Float32;
        DeviceIndex device = DeviceIndex(Device::CPU);
        bool requires_grad = false;
        Pinned_Flag pinten = Pinned_Flag::None;

        // Builder patterns
        TensorOptions with_dtype(Dtype d) const
        {
            TensorOptions opts = *this;
            opts.dtype = d;
            return opts;
        }
        TensorOptions with_device(DeviceIndex d) const
        {
            TensorOptions opts = *this;
            opts.device = d;
            return opts;
        }

        TensorOptions with_req_grad(bool g) const
        {
            TensorOptions opts = *this;
            opts.requires_grad = g;
            return opts;
        }

        TensorOptions with_pinten(Pinned_Flag flag) const
        {
            TensorOptions opts = *this;
            opts.pinten = flag;
            return opts;
        }
    };

    // ########################################################################
    // Class Defintions
    // ########################################################################

    class Tensor
    {
        private:
            intrusive_ptr<TensorImpl> impl_;

        public:
        //#######################################################
        // Constructor
        //#######################################################

        Tensor(Shape shape, Dtype dtype,
            DeviceIndex device = DeviceIndex(Device::CPU),
            bool requires_grad = false,
            Pinned_Flag pinten = Pinned_Flag::None);

        // Constructor with options
        Tensor(Shape shape, TensorOptions opts);

        //✨✨✨
        Tensor(Shape shape, bool requires_grad = false)
        : Tensor(shape, Dtype::Float32, DeviceIndex(Device::CPU), requires_grad) {}

        // Default constructor
        Tensor() = default;

        // Internal constructor from TensorImpl (for internal use)
        explicit Tensor(intrusive_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}


        //Temporary Constuctor for Views (Just to make library compile)
        // Tensor::Tensor(
        //         Shape shape,
        //         // Stride stride,
        //         // size_t offset,
        //         Dtype dtype,
        //         DeviceIndex device,
        //         bool requires_grad) : Tensor(shape, dtype, device, requires_grad) {}
        // //#######################################################
        // Internal Access (for advanced use)
        //#######################################################
        
        /**
         * Get raw TensorImpl pointer
         * WARNING: Use with caution - this is for internal use only
         */
        TensorImpl* unsafeGetTensorImpl() const { return impl_.get(); }

        //#######################################################
        // Metadata accessors
        //#######################################################

        const Shape& shape() const { return impl_->sizes(); };
        const Stride& stride() const { return impl_->strides(); };

        Dtype dtype() const { return impl_->dtype(); }
        DeviceIndex device() const { return impl_->device(); };
        bool requires_grad() const { return impl_->requires_grad(); };
        static size_t dtype_size(Dtype d);
        int64_t ndim() const { return impl_->ndim(); }

        void set_requires_grad(bool req);
        Tensor grad_view() const;
        TensorOptions opts() const{
            return TensorOptions().with_device(device()).with_dtype(dtype()).with_req_grad(requires_grad());    
        }
        bool has_grad() const { return impl_->has_grad(); }
        void zero_grad();
        void set_grad(const Tensor& grad);

        // ######################################################
        // Data Accessors
        //#######################################################

        void* data() { return impl_->mutable_data(); }
        const void* data() const { return impl_->data(); }

        void* grad();
        const void* grad() const;
        
        // ✨✨✨
        void reset() {
            impl_.reset();
        }

        template<typename T>
        T* data()
        {
            return impl_->data<T>();
        }

        template<typename T>
        T* grad();

        template<typename T>
        const T* data() const
        {
            return impl_->data<T>();
        }
                
        template<typename T>
        const T* grad() const;

        // ######################################################
        // Static Conditional Operator - Compiler
        //#######################################################
        // In the public section or as standalone function
        template<typename Func1, typename Func2, typename... Args>
        static Tensor cond(bool pred, Func1 true_fn, Func2 false_fn, Args&&... operands) {
            if (pred) {
                return true_fn(std::forward<Args>(operands)...);
            } else {
                return false_fn(std::forward<Args>(operands)...);
            }
        }

        // ######################################################
        // Device Metadata & Functions
        //#######################################################
        Tensor to(DeviceIndex device) const;

        bool is_cpu() const;
        bool is_cuda() const;

        Tensor to_cpu() const;
        Tensor to_cuda(int device_index = 0) const;
        void to_cpu_();
        void to_cuda_(int device_index = 0);

        Tensor to_bool() const;
        Tensor pin_memory() const; 
        bool is_pinned() const;
        //#######################################################
        // Memory Info
        //#######################################################
        
        static int64_t get_active_tensor_count() { return TensorImpl::get_active_count(); }

        size_t nbytes() const;
        size_t grad_nbytes() const;
        size_t numel() const;
        // void print_meta() const;
        size_t allocated_bytes() const { return impl_->storage().nbytes(); }
        size_t grad_allocated_bytes() const;
        bool owns_data() const;
        bool owns_grad() const;
        bool is_contiguous() const;
        Tensor contiguous() const;
        void print_meta() const;
        static void print_shape(Shape s);
        static void print_stride(Stride s);

        //#######################################################
        // Parallellism Utilities
        //#######################################################

        TensorOptions opts();
        Tensor slice(size_t start, size_t length);
        Tensor slice(OwnTensor::Tensor& tensor, size_t start, size_t length);
        Tensor slice_inplace(size_t start, size_t length);
        static Tensor flatten_concat(const std::vector<Tensor>& tensor_list);
        Tensor narrow(int64_t axis, int64_t start, int64_t length);
        Tensor narrow_view(int64_t axis, int64_t start, int64_t length);
        std::vector<Tensor> make_shards(size_t num_shards, bool row_major);
        std::vector<Tensor> make_shards_axis(size_t num_shards, int64_t axis);
        std::vector<Tensor> make_shards_cust(std::vector<Shape> shard_shapes, bool row_major);
        std::vector<Tensor> make_shards_inplace(size_t num_shards, bool row_major);
        std::vector<Tensor> make_shards_inplace_axis(size_t num_shards, int64_t axis);
        std::vector<Tensor> make_shards_inplace_cust(std::vector<Shape> shard_shapes, bool row_major);
        void shard_into(std::vector<Tensor>& destinations);
        
        // Returns the k largest elements of the given input tensor along a given dimension.
        std::pair<Tensor, Tensor> topk(int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true) const;

        //#######################################################
        // Data Manipulation
        //#######################################################

        template <typename T>
        void set_data(const T* source_data, size_t count);

        template<typename T>
        void set_data(const std::vector<T>& source_data);

        template <typename T>
        void set_data(std::initializer_list<T> values);

        template <typename T>
        void set_grad(const T* source_data, size_t count);

        template<typename T>
        void set_grad(const std::vector<T>& source_data);

        template <typename T>
        void set_grad(std::initializer_list<T> values);

        template <typename T>
        void fill(T value);

        template <typename T>
        void fill_grad(T value);

        //######################################################
        // Factory Functions
        //######################################################

        static Tensor zeros(Shape shape, TensorOptions opts = {});
        static Tensor ones(Shape shape, TensorOptions opts = {});
        static Tensor empty(Shape shape, TensorOptions opts = {});
        
        static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0);
        static Tensor full(Shape shape, TensorOptions, float val);
        // static Tensor rand(Shape shape, TensorOptions opts);
        template <typename U=float>
        static Tensor rand(Shape shape, TensorOptions opts,unsigned long seed = 42, U lower = U(0), U upper = U(0));

        //static Tensor randn(Shape shape, TensorOptions opts);
        template <typename U=float>
        static Tensor randn(Shape shape, TensorOptions opts,unsigned long seed=42 , U sd = U(1));

        static Tensor multinomial(const Tensor& input, int64_t num_samples,
                                  bool replacement = false,
                                  unsigned long seed = 0);
        
        //#######################################################
        // View Operations
        //#######################################################
        Tensor view(Shape new_shape) const;
        Tensor reshape(Shape new_shape) const;
        Tensor transpose(int dim0, int dim1) const;
        Tensor t() const;
        Tensor flatten(int start_dim = 0, int end_dim = -1) const;
        Tensor unflatten(int dim, Shape sizes) const;


        //#######################################################
        // Autograd Methods (PyTorch-style)
        //#######################################################
        
        // Gradient function
        std::shared_ptr<Node> grad_fn() const;
        void set_grad_fn(std::shared_ptr<Node> fn);
        
        // Output number (for multi-output operations)
        uint32_t output_nr() const;
        void set_output_nr(uint32_t nr);
        
        // View tracking
        bool is_view() const;
        void set_is_view(bool is_view);
        
        // Gradient retention (for non-leaves)
        bool retains_grad() const;
        void set_retains_grad(bool retains);
        
        // Check if this is a leaf tensor
        bool is_leaf() const;
        
        // Hooks
        void register_hook(std::unique_ptr<FunctionPreHook> hook);
        void register_post_acc_hook(std::unique_ptr<PostAccumulateGradHook> hook);
        void clear_hooks();
        
        // Backward pass
        void backward(const Tensor* grad_output = nullptr);

        //#######################################################
        // View Utilities
        //#######################################################
        size_t storage_offset() const;  // ← MUST EXIST

        //######################################################
        // Utilities
        //######################################################

        void display(std::ostream& os, int prec = 4) const;
        void display() const;
        void display(int prec) const;

        Tensor clone() const;
        Tensor detach() const;
        Tensor& copy_(const Tensor& src);
        Tensor as_type(Dtype new_dtype) const;

        //######################################################
        // Memory Deletion 
        //######################################################
        void release();
        bool is_valid() const;

        private:
            // Private constructor for creating views (shares storage)
            Tensor(intrusive_ptr<TensorImpl> impl,
                   Shape shape,
                   Stride stride,
                   size_t offset);
    };
}

