#pragma once
#include <string>
#include <memory>
#include <sstream>

enum class PlacementType {
    SHARD,      
    REPLICATE    
};

class Placement {
public:
    Placement() = default;
    virtual ~Placement(); // Declaration only, no '= default' here

    virtual PlacementType type() const = 0;
    virtual int dim() const { return -1; } 
    virtual void setDim(int dim) { } 
};


class Shard : public Placement {
public:

    Shard() = default;
    
    explicit Shard(int dim) : dim_(dim) {}
    
    PlacementType type() const override { 
        return PlacementType::SHARD; 
    }
    
    int dim() const override { return dim_; }
    void setDim(int dim) override { dim_ = dim; }
    
private:

    int dim_ = 0;  
};


class Replicate : public Placement {
public:
    Replicate() = default;
    
    
    PlacementType type() const override { 
        return PlacementType::REPLICATE; 
    }
    
};
  
