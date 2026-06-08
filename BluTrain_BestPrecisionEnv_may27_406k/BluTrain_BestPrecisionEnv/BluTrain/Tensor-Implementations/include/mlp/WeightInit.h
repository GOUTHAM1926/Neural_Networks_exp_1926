#pragma once

#include "core/Tensor.h"
#include <iostream>

namespace OwnTensor
{
    namespace mlp_forward
    {
        Tensor zero_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad);

        template <typename U>
        Tensor norm_rand_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_gra, U sd);

        template <typename U>
        Tensor uniform_rand_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, U lower, U upper);
        
        Tensor xavier_uniform_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in, int fan_out);
        
        Tensor xavier_norm_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in, int fan_out);
        
        Tensor he_uniform_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in);
        
        Tensor he_norm_weight(OwnTensor::Shape shape , OwnTensor::Dtype dtype, OwnTensor::Device dev, bool req_grad, int fan_in);
    }
}