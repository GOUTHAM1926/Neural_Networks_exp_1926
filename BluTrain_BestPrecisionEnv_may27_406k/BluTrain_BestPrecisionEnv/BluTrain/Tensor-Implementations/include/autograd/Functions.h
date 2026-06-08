#pragma once

/**
 * @file Functions.h
 * @brief Umbrella header for all backward node classes.
 * 
 * This file provides backward compatibility by including all
 * segregated backward class headers.
 */

#include "autograd/Node.h"
#include "autograd/backward/BinaryBackward.h"
#include "autograd/backward/MatrixBackward.h"
#include "autograd/backward/ActivationBackward.h"
#include "autograd/backward/ReductionBackward.h"
#include "autograd/backward/LossBackward.h"
#include "autograd/backward/GradAccumulator.h"
#include "autograd/backward/EmbeddingBackward.h"
#include "autograd/backward/NormalizationBackward.h"
