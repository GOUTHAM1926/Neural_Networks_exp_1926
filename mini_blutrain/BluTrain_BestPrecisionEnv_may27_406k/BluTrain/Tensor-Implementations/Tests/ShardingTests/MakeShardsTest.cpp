#include "TensorLib.h"
#include "TestUtils.h"

using namespace OwnTensor;

void test_make_shards_equal() {
    print_separator("TEST: make_shards(num_shards, row_major=true)");

    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    Tensor source = Tensor({{1, 12}}, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    std::cout << "Source tensor (1x12):" << std::endl;
    print_tensor_info("Source", source);
    print_tensor_data("Source", source, 12);

    std::vector<Tensor> shards = source.make_shards(3, true);

    std::cout << "\nShards (3 equal parts of 4 elements each):" << std::endl;
    
    std::cout << "\n=============== ASSERTION CASES ===============" << std::endl;

    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "\n  Shard " << i << ":" << std::endl;
        print_tensor_info("Shard " + std::to_string(i), shards[i]);
        print_tensor_data("Shard " + std::to_string(i), shards[i], 4);

        std::cout << "  Shard " << i << " Element Count: " << (shards[i].numel() == 4 ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].numel() == 4);
        
        std::cout << "  Shard " << i << " Owns Data:     " << (shards[i].owns_data() == true ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].owns_data() == true);
        
        std::cout << "  Shard " << i << " Is Copy:       " << (shards[i].data() != source.data() ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(shards[i].data() != source.data());

        float* ptr = shards[i].data<float>();
        bool values_correct = true;
        for (int j = 0; j < 4; ++j) {
            if (ptr[j] != (float)(i * 4 + j)) values_correct = false;
        }
        std::cout << "  Shard " << i << " Values:        " << (values_correct ? "✅ PASSED" : "❌ FAILED") << std::endl;
        assert(values_correct);
    }

    std::cout << "\n[PASS] make_shards() creates independent copies with correct data" << std::endl;
}

int main() {
    try {
        test_make_shards_equal();
    } catch (const std::exception& e) {
        std::cerr << "\n[FAILED] Test Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}