#pragma once

#include "core/Tensor.h"
#include <string>
#include <fstream>

namespace OwnTensor {

/**
 * Saves a tensor to a binary stream.
 */
void save_tensor(const Tensor& tensor, std::ostream& os);
void save_tensor(const Tensor& tensor, const std::string& path);

/**
 * Loads a tensor from a binary stream.
 */
Tensor load_tensor(std::istream& is);
Tensor load_tensor(const std::string& path);

} // namespace OwnTensor