#pragma once
#ifndef TOPS_LIB_H
#define TOPS_LIB_H

// Core Tensor Interface
#include "core/Tensor.h"


// Device Management

// Core Library Includes
#include "device/AllocatorRegistry.h"
#include "device/DeviceCore.h"

// Datatype and Traits
#include "dtype/DtypeTraits.h"
#include "dtype/Types.h"

// Operations
#include "ops/UnaryOps/Activations.h"
#include "ops/UnaryOps/Normalizations.h"
#include "ops/UnaryOps/Arithmetics.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/UnaryOps/Trigonometry.h"

#include "ops/ScalarOps.h"
#include "ops/TensorOps.h"
#include "ops/Kernels.h"
#include "ops/helpers/TensorOpUtils.h"
#include "ops/IndexingOps.h"

// ConditionalOps.h
#include "ops/helpers/ConditionalOps.h"

// Autograd
#include "autograd/AutogradOps.h"
#include "autograd/Engine.h"

// Neural Network and Optimizer
#include "nn/NN.h"
#include "nn/MultiHeadAttention.h"
#include "nn/optimizer/Optim.h"
#include "nn/optimizer/LossScaler.h"

// Reductions Utils



#endif // TOPS_LIB_H