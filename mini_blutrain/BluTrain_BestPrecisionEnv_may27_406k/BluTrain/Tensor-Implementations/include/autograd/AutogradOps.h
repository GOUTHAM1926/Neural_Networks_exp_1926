#pragma once

/**
 * @file AutogradOps.h
 * @brief Umbrella header for all autograd-aware operations.
 * 
 * This file provides backward compatibility by including all
 * segregated operation headers. Existing code using:
 *   #include "autograd/AutogradOps.h"
 * will continue to work unchanged.
 */

// Include all operation categories
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ArithmeticsOps.h"
#include "autograd/operations/ExponentsOps.h"
#include "autograd/operations/TrigonometryOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "autograd/operations/TrilOps.h"
#include "autograd/operations/FlashAttentionOps.h"
#include "autograd/operations/AttentionOps.h"