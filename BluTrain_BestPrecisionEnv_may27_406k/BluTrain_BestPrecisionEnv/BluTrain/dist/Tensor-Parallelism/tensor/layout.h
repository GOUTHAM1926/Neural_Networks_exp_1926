#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <memory>
#include "../tensor/device_mesh.h"
#include "../tensor/placement.h"


// enum class ShardingType {
//     REPLICATED, // Tensor is fully replicated on all devices
//     SHARDED     // Tensor is sharded along a specific dimension
// };

class Layout {

    std::shared_ptr<Placement> placement_;
    std::vector<int64_t> global_shape_;

public: 
    // Default constructor for member initialization
    Layout() : mesh_(nullptr), global_shape_(), placement_(std::make_shared<Replicate>()) {}
    
    Layout(const DeviceMesh& mesh, const std::vector<int64_t> global_shape) : mesh_(&mesh), global_shape_(global_shape), placement_(std::make_shared<Replicate>()) {}

    // Constructor for a new layout
    Layout(const DeviceMesh& mesh, const std::vector<int64_t> global_shape, int dim)
        : mesh_(&mesh), global_shape_(global_shape), placement_(std::make_shared<Shard>(dim)) {
        //std::cout<< "Dimensions"<<std::endl;
        // for(auto i: global_shape){

        // }
        if ( (placement_->dim() < 0 || (size_t)placement_->dim() >= global_shape.size())) {
            throw std::runtime_error("Invalid shard_dim for SHARDED layout.");
        }
       
    }


    bool is_replicated() const {
        return placement_->type() == PlacementType::REPLICATE;
    }

    bool is_sharded() const {
        return placement_->type() == PlacementType::SHARD;
    }

    // Check if sharded along a specific dimension
    bool is_sharded_by_dim(int dim) const {
        return is_sharded() && placement_->dim() == dim;
    }

    int get_shard_dim() const {
        if (placement_->type() == PlacementType::SHARD) {
        return placement_->dim();
        }
        return -1; 
    }
    
    void set_shard_dim( int d)  {

        placement_->setDim(d);       
        
    }

    const std::vector<int64_t>& get_global_shape() const {
        return global_shape_;
    }

    void set_global_shape(std::vector<int64_t> new_shape)  {
        global_shape_ = new_shape;
    }

    const DeviceMesh& get_mesh() const {
        return *mesh_;
    }

    // --- Core Logic ---

    // Calculates the shape of the local shard for a given rank

    std::vector<int64_t> get_local_shape(int rank) const {
         if (placement_->type() == PlacementType::REPLICATE) {
             return global_shape_; // No sharding, return full shape
         }
     
         int d = placement_->dim();
         // Ensure d is valid before indexing global_shape_
         if (d < 0 || (size_t)d >= global_shape_.size()) {
             return global_shape_;
         }


        int world_size = mesh_->world_size();
        // 2. SAFETY: Prevent division by zero if mesh is uninitialized
        if (world_size <= 0) return global_shape_;

        std::vector<int64_t> local_shape = global_shape_;

        // 3. SAFETY: Check the shard dimension
        int shard_dim = placement_->dim();
        if (shard_dim < 0 || (size_t)shard_dim >= global_shape_.size()) {
            // If this is -1, global_shape_[shard_dim] will trigger bad_array_new_length
            return global_shape_; 
        }

        int64_t global_dim_size = global_shape_[shard_dim];
        int64_t base_size = global_dim_size / world_size;
        int64_t remainder = global_dim_size % world_size;

        int64_t local_dim_size = (rank < remainder) ? (base_size + 1) : base_size;

        local_shape[shard_dim] = local_dim_size;

        return local_shape;
    }


    // Helper to get total number of elements in the global tensor
    int64_t global_numel() const {
        if (global_shape_.empty()) return 0;
        return std::accumulate(global_shape_.begin(), global_shape_.end(), 1LL, std::multiplies<int64_t>());
    }

    // Creates a new layout for a reshaped tensor (simplified)
    Layout reshape(const std::vector<int64_t>& new_global_shape) const {
        // A simple check: if sharding dim is preserved and size is same, keep it.
        if (is_sharded() && placement_->dim() < (int64_t)new_global_shape.size() && new_global_shape[placement_->dim()] == global_shape_[placement_->dim()]) {
             return Layout(*mesh_, new_global_shape, placement_->dim());
        }
        
        // Default: fall back to replicated
        return Layout(*mesh_, new_global_shape);
    }

    // Check for element-wise op compatibility
    bool is_compatible(const Layout& other) const {
        // Simple check: are global shapes and sharding identical?
        return global_shape_ == other.global_shape_ &&
                placement_->type() == other.placement_->type() &&
                placement_->dim() == other.placement_->dim();
    }

    bool operator==(const Layout& other) const {
        return is_compatible(other);
    }

    bool operator!=(const Layout& other) const {
        return !is_compatible(other);
    }

    
  
    // static Layout replicated(std::shared_ptr<DeviceMesh> mesh, 
    //                         const std::vector<int64_t>& global_shape) {
    //     return Layout(mesh, global_shape, PlacementType::REPLICATE);
    // }

    //  static Layout sharded(std::shared_ptr<DeviceMesh> mesh, 
    //                         const std::vector<int64_t>& global_shape) {
    //     return Layout(mesh, global_shape, PlacementType::SHARD);
    // }

    std::string describe(int rank) const {
        std::ostringstream oss;

        oss << "[Layout] Rank " << rank <<" | ";
        oss << "Global Shape: [";
        for (size_t i = 0; i < global_shape_.size(); ++i) {
            oss << global_shape_[i] << (i == global_shape_.size() - 1 ? "" : ", ");
        }
        oss << "] | ";

        if (is_replicated()) {
            oss << "REPLICATED";
        } else {
            oss << "SHARDED (Dim: " << placement_->dim() << ")";
        }
        
        std::vector<int64_t> local_shape = get_local_shape(rank);
        oss << " | Local Shape: [";
        for (size_t i = 0; i < local_shape.size(); ++i) {
            oss << local_shape[i] << (i == local_shape.size() - 1 ? "" : ", ");
        }
        oss << "]";
        return oss.str();
    }

    
    // Constructor taking placements 
    // Layout(std::shared_ptr<DeviceMesh> mesh, 
    //        const std::vector<int64_t>& global_shape,
    //        const std::vector<std::shared_ptr<Placement>>& placements)
    //     : mesh_(mesh), global_shape_(global_shape), placements_(placements) {
        
    //     placement_->type() = PlacementType::REPLICATE;
    //     placement_->dim() = -1;
        
    //     if (!placements.empty()) {
    //         // Simple logic: look at first placement
    //         if (placements[0]->type() == PlacementType::SHARD) {
    //             placement_->type() = PlacementType::SHARD;
    //             placement_->dim() = static_cast<Shard*>(placements[0].get())->dim();
    //         }
    //     }
    // }

    // std::shared_ptr<Placement> get_placement(int index) const {
    //     if (index < 0 || index ) {
    //         // Fallback if placements are not set but simple sharding is
    //         if (placements_.empty()) {
    //             if (placement_->type() == PlacementType::REPLICATE) {
    //                 return std::make_shared<Replicate>();
    //             } else {
    //                 return std::make_shared<Shard>(placement_->dim());
    //             }
    //         }
    //         throw std::out_of_range("Placement index out of range");
    //     }
    //     return placements_[index];
    // }

    private:
        const DeviceMesh* mesh_;
        // std::vector<int64_t> global_shape_;

        // Placement* placement_;
        // std::vector<std::shared_ptr<Placement>> placements_;


    };