#include "ZeRORedundancyOptimizer.hpp"

// Explicit template instantiation definitions.
// All method bodies live in the header (template class, defined inline).
// This .cpp forces compilation of the three supported specializations.

template class ZeROOptimizer<OwnTensor::nn::AdamW>;
template class ZeROOptimizer<OwnTensor::nn::Adam>;
template class ZeROOptimizer<OwnTensor::nn::SGDOptimizer>;
