#include "core/IntrusivePtr.h"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"
#include "device/DeviceTransfer.h"
#include "core/Views/ViewUtils.h"
#include "autograd/backward/ShardingBackward.h"
#include "autograd/backward/parallelizeutilsBackward.h"
#include "autograd/GraphRecorder.h"
#include "autograd/ops_template.h"
#include <cstdint>
#include <numeric>
#include <iostream>

namespace OwnTensor
{
    TensorOptions Tensor::opts()
    {
        TensorOptions opts;
        opts.dtype = this->impl_->dtype();
        opts.device = this->impl_->device();
        opts.requires_grad = this->impl_->requires_grad();

        return opts;
    }

    Tensor Tensor::slice(size_t start, size_t length)
    {
        TensorOptions opts = this->opts();


        if (start > this->numel() || start + length > this->numel())
        {
            throw std::runtime_error("Range exceeded!! (Zero based indexing)");
        }

        Tensor new_tensor = Tensor({ {1, static_cast<int64_t>(length)} }, opts);

        dispatch_by_dtype(this->dtype(), [&](auto dummy)
            {
                using T = decltype(dummy);

                void* temp_pointer = static_cast<T*>(this->data()) + start;

                device::copy_memory(new_tensor.data(), opts.device.device,
                    temp_pointer, opts.device.device,
                    length * dtype_size(this->dtype()));
            });

        return new_tensor;
    }
    

    Tensor Tensor::slice(OwnTensor::Tensor& tensor, size_t start, size_t length){
        OwnTensor::TensorOptions opts = tensor.opts();

        if(start > tensor.numel() || start + length > tensor.numel()){
            throw std::runtime_error(
                "range exceeded... (zero based indexing)"
            );
        }

        OwnTensor::Tensor new_tensor = OwnTensor::Tensor({{1, static_cast<int64_t>(length)}}, opts);

        dispatch_by_dtype(tensor.dtype(),[&](auto dummy){
            // using T = decltype(dummy);

            size_t byte_offset = start * OwnTensor::Tensor::dtype_size(tensor.dtype());
            void* temp_pointer = static_cast<uint8_t*>(const_cast<void*>(tensor.data())) + byte_offset;

            OwnTensor::device::copy_memory(new_tensor.data(), opts.device.device, temp_pointer, opts.device.device, length * OwnTensor::Tensor::dtype_size(tensor.dtype()));

        });

        
        return new_tensor;
    }

    Tensor Tensor::slice_inplace(size_t start, size_t length)
    {
        // TensorOptions opts = this->opts();


        if (start > this->numel() || start + length > this->numel())
        {
            throw std::runtime_error("Range exceeded!! (Zero based indexing)");
        }

        // Use the BASE pointer - storage_offset will handle the element offset
        // uint8_t* raw_ptr = this->impl_->mutable_storage().data_ptr();
        // DataPtr alias_ptr(raw_ptr, DataPtrDeleter(nullptr));
        // 
        // /** 
        //  * THIS DOES NOT UPDATE THE REF COUNT AS WE ARE WASTING A IMPL BY CHANGING ITS POINTER
        //  * 
        //  * Tensor new_tensor = Tensor({ {1, static_cast<int64_t>(length)} }, opts);
        //  * new_tensor.unsafeGetTensorImpl()->mutable_storage().set_data_ptr(std::move(alias_ptr));
        //  *  this->unsafeGetTensorImpl()->add_ref();
        //  * return new_tensor; 
        // */
        //
        // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
        //     std::move(alias_ptr),
        //     this->impl_->storage().nbytes(),
        //     this->dtype(),
        //     this->device(),
        //     nullptr
        // );
        //
        intrusive_ptr<Storage> shared_storage = this->impl_->storage_ptr();

        Shape new_shape = {{ 1, int64_t(length)}};
        int64_t offset = int64_t(start);  // Element offset handled by TensorImpl

        intrusive_ptr<TensorImpl> view_impl = make_intrusive<TensorImpl>(
            shared_storage,
            new_shape,
            ViewUtils::compute_strides(new_shape),
            offset,
            this->dtype(),
            this->device()
            // intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
        );

        return Tensor(std::move(view_impl));
    }

    Tensor Tensor::flatten_concat(const std::vector<Tensor>& tensor_list)
    {
        if (tensor_list.empty()) {
            return Tensor();
        }

        int64_t total_elements = std::accumulate(tensor_list.begin(), tensor_list.end(), int64_t(0),
            [](int64_t sum, const auto& tensor) {
                return sum + tensor.numel();
            });

        Tensor result = Tensor({ {1, total_elements} }, tensor_list[0].opts());
        void* result_ptr = result.data();
        int64_t running_pointer = 0;


        for (const Tensor& tensor : tensor_list)
        {
            dispatch_by_dtype(tensor.dtype(), [&](auto dummy) {
                    using T = decltype(dummy);
                    void* new_ptr = (static_cast<T*>(result_ptr) + running_pointer);
                    device::copy_memory(new_ptr, result.device().device, tensor.data(), tensor.device().device, tensor.nbytes());
                    running_pointer += tensor.numel();
                });
        }
        return result;
    }

    Tensor Tensor::narrow(int64_t axis, int64_t start, int64_t length) {
        Shape old_shape = this->shape();
        Shape new_shape = old_shape;
        new_shape.dims[axis] = length;

        Tensor result(new_shape, this->opts());

        // 1. Calculate strides for row-major layout
        std::vector<int64_t> strides(old_shape.dims.size());
        int64_t s = 1;
        for (int i = old_shape.dims.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= old_shape.dims[i];
        }

        // 2. Identify the 'block' to copy and the 'jump' between blocks
        // outer_count: How many times we need to perform a copy (dims to the left)
        // copy_size_elems: Number of elements in one contiguous chunk (this axis + dims to the right)
        int64_t outer_count = 1;
        for (int i = 0; i < axis; ++i) outer_count *= old_shape.dims[i];

        // Number of elements to copy in one contiguous burst
        // This is the narrowed length * everything to the right of the axis
        int64_t copy_size_elems = length * strides[axis];
        
        // The distance in memory between the START of one block and the START of the next
        // in the ORIGINAL tensor (this is the full stride of the dimension above the axis)
        int64_t src_step_elems = old_shape.dims[axis] * strides[axis];
        
        // The distance in memory between blocks in the NEW shard tensor
        int64_t dst_step_elems = length * strides[axis];

        size_t elem_bytes = dtype_size(this->dtype());
        uint8_t* base_src = static_cast<uint8_t*>(this->data());
        uint8_t* base_dst = static_cast<uint8_t*>(result.data());

        // IMPORTANT: The global offset for the very first element of the shard
        uint8_t* shard_start_src = base_src + (start * strides[axis] * elem_bytes);

        for (int64_t i = 0; i < outer_count; ++i) {
            uint8_t* current_src = shard_start_src + (i * src_step_elems * elem_bytes);
            uint8_t* current_dst = base_dst + (i * dst_step_elems * elem_bytes);

            device::copy_memory(
                current_dst, result.device().device,
                current_src, this->device().device,
                copy_size_elems * elem_bytes
            );
        }

        return result;
    }
 
    Tensor Tensor::narrow_view(int64_t axis, int64_t start, int64_t length) {
        Shape old_shape = this->shape();

        if (axis < 0 || axis >= (int64_t)old_shape.dims.size())
            throw std::out_of_range("Axis out of bounds");

        if (start + length > old_shape.dims[axis])
            throw std::out_of_range("Narrow range exceeds dimension");

        Shape new_shape = old_shape;
        new_shape.dims[axis] = length;

        Stride old_stride = this->stride();

        int64_t new_offset = this->storage_offset() + start * old_stride.strides[axis];

        // uint8_t* raw_ptr = this->impl_->mutable_storage().data_ptr();
        // DataPtr alias_ptr(raw_ptr, DataPtrDeleter(nullptr));
        //
        // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
        //     std::move(alias_ptr),
        //     this->nbytes(),
        //     this->dtype(),
        //     this->device(),
        //     nullptr
        // );
        
        intrusive_ptr<Storage> shared_storage = this->impl_->storage_ptr();

        intrusive_ptr<TensorImpl> view_impl = make_intrusive<TensorImpl>(
            shared_storage,
            new_shape,
            old_stride,
            new_offset,
            this->dtype(),
            this->device()
            // intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
        );

        return Tensor(std::move(view_impl));
    }

    std::vector<Tensor> Tensor::make_shards_axis(size_t num_shards, int64_t axis) { 
        // 1. Get current shape and validate 
        Shape current_shape = this->shape();
        if (axis < 0 || axis >= (int64_t)current_shape.dims.size()) 
        {
            throw std::out_of_range("Axis index out of bounds"); 
        }
        int64_t dim_size = current_shape.dims[axis];
        if (dim_size % num_shards != 0) 
        { 
            throw std::runtime_error("Dimension size not divisible by num_shards"); 
        }
        int64_t shard_dim_size = dim_size / num_shards;
        std::vector<Tensor> shards; shards.reserve(num_shards);
        
        // 2. Iterate and narrow along the chosen axis 
        for (size_t i = 0; i < num_shards; ++i) 
        { 
            int64_t start = i * shard_dim_size; 
            
            // Use 'narrow' to extract a slice of the tensor along the axis 
            // This is the C++ equivalent of tensor.select() or tensor[:, start:start+len] 
            Tensor shard = this->narrow(axis, start, shard_dim_size).contiguous(); 
            shards.push_back(std::move(shard)); 
        } 
        return shards; 
    }

    std::vector<Tensor> Tensor::make_shards_inplace_axis(size_t num_shards, int64_t axis) {
        autograd::GraphRecordMode::record_forward("Sharding: make_shards_inplace_axis");
        Shape old_shape = this->shape();
        Stride old_stride = ViewUtils::compute_strides(old_shape);

        int64_t dim_size = old_shape.dims[axis];
        int64_t shard_dim_size = dim_size / num_shards;

        std::vector<Tensor> shards;
        shards.reserve(num_shards);

        // Set up autograd
        std::shared_ptr<autograd::Make_shards_inplace_axis_Backward> grad_fn;
        if (this->requires_grad()) {
            grad_fn = std::make_shared<autograd::Make_shards_inplace_axis_Backward>(num_shards, axis, old_shape, this->opts());
            Tensor& self_mut = const_cast<Tensor&>(*this);
            grad_fn->set_next_edge(0, autograd::get_grad_edge(self_mut));
        }

        for (int i = 0; i < int(num_shards); ++i) {
            int64_t start = i * shard_dim_size;

            Shape new_shape = old_shape;
            new_shape.dims[axis] = shard_dim_size;
            Stride new_stride = old_stride;
            int64_t shard_offset = this->storage_offset() + start * old_stride.strides[axis];

            // uint8_t* raw_shard_ptr = this->impl_->mutable_storage().data_ptr();
            // DataPtr alias_ptr(raw_shard_ptr, DataPtrDeleter(nullptr));
            //
            // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
            //     std::move(alias_ptr),
            //     this->impl_->storage().nbytes(),
            //     this->dtype(),
            //     this->device(),
            //     nullptr
            // );
            intrusive_ptr<Storage> shared_storage = this->impl_->storage_ptr();

            intrusive_ptr<TensorImpl> shard_impl = make_intrusive<TensorImpl>(
                shared_storage,
                new_shape,
                new_stride,
                shard_offset,
                this->dtype(),
                this->device()
                // intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
            );

           Tensor shard(std::move(shard_impl));
            if (grad_fn) {
                shard.set_grad_fn(grad_fn);
                shard.set_output_nr(i);
                shard.set_requires_grad(true);
            }
            shards.push_back(std::move(shard));
        }
        return shards;
    }

    std::vector<Tensor> Tensor::make_shards(size_t num_shards, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards(num_shards, true);
        }

        if (this->numel() % num_shards != 0)
        {
            throw std::runtime_error("Cannot split this tensor into this number of equal shards");
        }

        TensorOptions opts = this->opts();

        size_t shard_elems = this->numel() / num_shards;
        size_t elem_bytes = dtype_size(this->dtype());
        size_t shard_bytes = shard_elems * elem_bytes;

        std::vector<Tensor> shards;
        shards.reserve(num_shards);

        uint8_t* base = static_cast<uint8_t*>(this->data());

        for (size_t i = 0; i < num_shards; ++i)
        {
            Tensor shard({ {1, static_cast<int64_t>(shard_elems)} }, opts);

            device::copy_memory(shard.data(), shard.device().device,
                base + i * shard_bytes, this->device().device, shard_bytes);

            shards.push_back(std::move(shard));
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_cust(std::vector<Shape> shard_shapes, bool row_major)
    {

        if (!row_major)
        {
            return this->t().contiguous().make_shards_cust(shard_shapes, true);
        }

        TensorOptions opts = this->opts();
        int64_t total_req_elements = 0;


        for (const Shape& s : shard_shapes)
        {
            int64_t shape_numel = 1;
            for (int64_t d : s.dims) shape_numel *= d;
            total_req_elements += shape_numel;
        }

        if (total_req_elements != static_cast<int64_t>(this->numel()))
        {
            throw std::runtime_error("make shards custom: Total elements in requested shapes ("
                + std::to_string(total_req_elements) + ") does not match the tensor suze ("
                + std::to_string(this->numel()) + ")");
        }

        size_t num_shards = shard_shapes.size();
        std::vector<Tensor> shards;
        shards.reserve(num_shards);


        uint8_t* base_ptr = static_cast<uint8_t*>(this->data());
        for (size_t i = 0; i < num_shards; ++i)
        {
            Tensor shard({ shard_shapes[i] }, opts);
            size_t elem_size = dtype_size(this->dtype());

            if (i == 0)
            {
                device::copy_memory(shard.data(), shard.device().device,
                    base_ptr, this->device().device,
                    elem_size * shard.numel());

                base_ptr += (shard.numel() * elem_size);
                shards.push_back(std::move(shard));
            }

            else
            {
                device::copy_memory(shard.data(), shard.device().device,
                    base_ptr, this->device().device,
                    elem_size * shard.numel());

                base_ptr += (shard.numel() * elem_size);
                shards.push_back(std::move(shard));
            }
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_inplace(size_t num_shards, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards_inplace(num_shards, true);
        }

        if (this->numel() % num_shards != 0)
        {
            throw std::runtime_error("Cannot split tensor into equal shards");
        }

        size_t shard_elems = this->numel() / num_shards;

        std::vector<Tensor> shards;
        shards.reserve(num_shards);

        std::shared_ptr<autograd::ShardingBackward> grad_fn;
        if (this->requires_grad()) {
            grad_fn = std::make_shared<autograd::ShardingBackward>(this->shape(), num_shards);
            Tensor& self_mut = const_cast<Tensor&>(*this);
            grad_fn->set_next_edge(0, autograd::get_grad_edge(self_mut));
        }

        for (size_t i = 0; i < num_shards; ++i)
        {
            size_t shard_offset_elems =
                this->storage_offset() + i * shard_elems;  // ELEMENT offset
            Shape shard_shape = Shape({ {1, (int64_t)shard_elems} });
            
            // Create aliased storage
            // uint8_t* raw_ptr = this->impl_->mutable_storage().data_ptr();
            // DataPtr alias_ptr(raw_ptr, DataPtrDeleter(nullptr));
            // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
            //     std::move(alias_ptr),
            //     this->impl_->storage().nbytes(),
            //     this->dtype(),
            //     this->device(),
            //     nullptr
            // );
            intrusive_ptr<Storage> shared_storage = this->impl_->storage_ptr();

            intrusive_ptr<TensorImpl> shard_impl = make_intrusive<TensorImpl>(         
                shared_storage,            // shared (aliased) storage
                Shape(shard_shape),
                ViewUtils::compute_strides(shard_shape),
                static_cast<int64_t>(shard_offset_elems),   // view offset
                this->dtype(),
                this->device()
                // intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
            );

            Tensor shard(std::move(shard_impl));

            if (grad_fn) {
                shard.set_grad_fn(grad_fn);
                shard.set_output_nr(i);
                shard.set_requires_grad(true);
            }

            shards.push_back(std::move(shard));
        }

        return shards;
    }

    std::vector<Tensor> Tensor::make_shards_inplace_cust(std::vector<Shape> shard_shapes, bool row_major)
    {
        if (!row_major)
        {
            return this->t().contiguous().make_shards_inplace_cust(shard_shapes, true);
        }
        int64_t total_req_elements = 0;
        for (const Shape& s : shard_shapes)
        {
            int64_t shape_numel = 1;
            for (int64_t d : s.dims) shape_numel *= d;
            total_req_elements += shape_numel;
        }

        if (total_req_elements != static_cast<int64_t>(this->numel()))
        {
            throw std::runtime_error("make shards custom: Total elements in requested shapes ("
                + std::to_string(total_req_elements) + ") does not match the tensor suze ("
                + std::to_string(this->numel()) + ")");
        }


        std::vector<Tensor> shards;
        shards.reserve(shard_shapes.size());

        std::shared_ptr<autograd::ShardingBackward> grad_fn;
        if (this->requires_grad()) {
            grad_fn = std::make_shared<autograd::ShardingBackward>(this->shape(), shard_shapes);
            Tensor& self_mut = const_cast<Tensor&>(*this);
            grad_fn->set_next_edge(0, autograd::get_grad_edge(self_mut));
        }

        size_t shard_offset_elems = 0;

        for (size_t i = 0; i < shard_shapes.size(); ++i)
        {
            Shape shard_shape = shard_shapes[i];

             // Create aliased storage
             // uint8_t* raw_ptr = this->impl_->mutable_storage().data_ptr();
             // DataPtr alias_ptr(raw_ptr, DataPtrDeleter(nullptr));
             // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
             //     std::move(alias_ptr),
             //     this->impl_->storage().nbytes(),
             //     this->dtype(),
             //     this->device(),
             //     nullptr
             // );
             intrusive_ptr< Storage> shared_storage = this->impl_->storage_ptr();
 
             intrusive_ptr<TensorImpl> shard_impl = make_intrusive<TensorImpl>(  
                 shared_storage,            // shared (aliased) storage
                 shard_shape,
                 ViewUtils::compute_strides(shard_shape),
                 static_cast<int64_t>(shard_offset_elems),   
                 this->dtype(),
                 this->device()
                //  intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
             );

            Tensor shard(std::move(shard_impl));

            if (grad_fn) {
                shard.set_grad_fn(grad_fn);
                shard.set_output_nr(i);
                shard.set_requires_grad(true);
            }

            // Increment the tracker by the number of elements in the shard just created
            shard_offset_elems += shard.numel();
            shards.push_back(std::move(shard));
        }
        return shards;
    }

    void Tensor::shard_into(std::vector<Tensor>& destinations)
    {
        size_t current_elem_offset = 0;
        size_t total_src_elems = this->numel();
        // size_t elem_size = dtype_size(this->dtype());

        for (Tensor& dest : destinations)
        {
             size_t dest_elems = dest.numel();
             
             if (current_elem_offset + dest_elems > total_src_elems)
             {
                 throw std::runtime_error("shard_into: Destination tensor exceeds source tensor size.");
             }

             size_t view_elem_offset = this->storage_offset() + current_elem_offset;
             // size_t view_byte_offset = view_elem_offset * elem_size;
             
             // Create non-owning aliased storage for the view
             // data() returns the pointer including storage_offset, so we need the raw base pointer
             // Actually, TensorImpl stores storage_offset. Storage objects usually wrap the BASE allocation.
             // So we should use the same base pointer as the source storage.
             
             // We can get the raw pointer from current storage
             // uint8_t* raw_ptr = this->impl_->mutable_storage().data_ptr();
             // 
             // DataPtr alias_ptr(raw_ptr, DataPtrDeleter(nullptr));
             // 
             // intrusive_ptr<Storage> alias_storage = make_intrusive<Storage>(
             //     std::move(alias_ptr),
             //     this->impl_->storage().nbytes(),
             //     this->dtype(),
             //     this->device(),
             //     nullptr
             // );
             intrusive_ptr<Storage> shared_storage = this->impl_->storage_ptr();

             intrusive_ptr<TensorImpl> shard_impl = make_intrusive<TensorImpl>(
                 shared_storage,
                 dest.shape(), 
                 ViewUtils::compute_strides(dest.shape()), 
                 static_cast<int64_t>(view_elem_offset),
                 this->dtype(),
                 this->device()
                //  intrusive_ptr<TensorImpl>(this->unsafeGetTensorImpl())
             );
             
             Tensor shard_view(std::move(shard_impl));
             
             dest.copy_(shard_view);
             
             current_elem_offset += dest_elems;
        }
    }

}
