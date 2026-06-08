#include "core/Serialization.h"
#include "core/Tensor.h"

#include <stdexcept>
#include <iostream>

namespace OwnTensor {

void save_tensor(const Tensor& tensor, std::ostream& os) {
    if (!tensor.is_valid()) {
        os.write("TNS0", 4);
        return;
    }
    // 1. Magic number
    os.write("TNS1", 4);

    // 2. Dtype
    int dtype_val = static_cast<int>(tensor.dtype());
    os.write(reinterpret_cast<const char*>(&dtype_val), sizeof(int));

    // 3. Rank
    int rank = static_cast<int>(tensor.ndim());
    os.write(reinterpret_cast<const char*>(&rank), sizeof(int));    

    // 4. Shape
    const auto& shape = tensor.shape();
    for (int i = 0; i < rank; ++i) {
        int64_t dim = shape.dims[i];
        os.write(reinterpret_cast<const char*>(&dim), sizeof(int64_t));
    }

    // 5. Data
    // Ensure we are saving from CPU
    Tensor cpu_tensor = tensor.is_cpu() ? tensor : tensor.to_cpu();
    size_t nbytes = cpu_tensor.nbytes();
    os.write(reinterpret_cast<const char*>(cpu_tensor.data()), nbytes);
}

void save_tensor(const Tensor& tensor, const std::string& path) {
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for saving tensor: " + path);
    }
    save_tensor(tensor, os);
    os.close();
}

Tensor load_tensor(std::istream& is) {
    // 1. Magic number
    char magic[4];
    is.read(magic, 4);
    std::string magic_str(magic, 4);
    if (magic_str == "TNS0") {
        return Tensor();
    }
    if (magic_str != "TNS1") {
        throw std::runtime_error("Invalid tensor file format");
    }

    // 2. Dtype
    int dtype_val;
    is.read(reinterpret_cast<char*>(&dtype_val), sizeof(int));
    Dtype dtype = static_cast<Dtype>(dtype_val);

    // 3. Rank
    int rank;
    is.read(reinterpret_cast<char*>(&rank), sizeof(int));

    // 4. Shape
    std::vector<int64_t> dims(rank);
    for (int i = 0; i < rank; ++i) {
        is.read(reinterpret_cast<char*>(&dims[i]), sizeof(int64_t));
    }
    Shape shape({dims});

    // 5. Data
    Tensor tensor = Tensor::empty(shape, TensorOptions().with_dtype(dtype).with_device(Device::CPU));
    is.read(reinterpret_cast<char*>(tensor.data()), tensor.nbytes());

    return tensor;
}

Tensor load_tensor(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for loading tensor: " + path);
    }
    Tensor t = load_tensor(is);
    is.close();
    return t;
}

} // namespace OwnTensor