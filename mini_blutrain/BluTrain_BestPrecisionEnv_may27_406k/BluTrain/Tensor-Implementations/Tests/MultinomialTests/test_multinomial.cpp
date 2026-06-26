#include "TensorLib.h"
#include <iostream>
#include <set>
#include <cassert>

using namespace OwnTensor;

int main() {
    std::cout << "=== Test 1: 1D input, with replacement ===" << std::endl;
    {
        Tensor probs(Shape{{4}}, Dtype::Float32);
        probs.set_data<float>({0.25f, 0.25f, 0.25f, 0.25f});
        Tensor out = Tensor::multinomial(probs, 5, /*replacement=*/true);
        assert(out.shape().dims.size() == 1);
        assert(out.shape().dims[0] == 5);
        assert(out.dtype() == Dtype::Int64);
        std::cout << "  Output shape: (" << out.shape().dims[0] << ")" << std::endl;
        int64_t* d = out.data<int64_t>();
        std::cout << "  Samples: ";
        for (int i = 0; i < 5; i++) {
            std::cout << d[i] << " ";
            assert(d[i] >= 0 && d[i] < 4);
        }
        std::cout << std::endl;
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 2: 2D input (batch), with replacement ===" << std::endl;
    {
        Tensor probs(Shape{{3, 4}}, Dtype::Float32);
        probs.set_data<float>({
            0.1f, 0.2f, 0.3f, 0.4f,
            0.4f, 0.3f, 0.2f, 0.1f,
            0.0f, 0.0f, 0.5f, 0.5f
        });
        Tensor out = Tensor::multinomial(probs, 3, /*replacement=*/true);
        assert(out.shape().dims.size() == 2);
        assert(out.shape().dims[0] == 3);
        assert(out.shape().dims[1] == 3);
        std::cout << "  Output shape: (" << out.shape().dims[0] << ", " << out.shape().dims[1] << ")" << std::endl;
        int64_t* d = out.data<int64_t>();
        for (int r = 0; r < 3; r++) {
            std::cout << "  Row " << r << ": ";
            for (int c = 0; c < 3; c++) {
                int64_t idx = d[r * 3 + c];
                std::cout << idx << " ";
                assert(idx >= 0 && idx < 4);
            }
            std::cout << std::endl;
        }
        // Row 2 has weights [0,0,0.5,0.5], so only indices 2 and 3 should appear
        for (int c = 0; c < 3; c++) {
            int64_t idx = d[2 * 3 + c];
            assert(idx == 2 || idx == 3);
        }
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 3: Without replacement — unique indices ===" << std::endl;
    {
        Tensor probs(Shape{{5}}, Dtype::Float32);
        probs.set_data<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
        Tensor out = Tensor::multinomial(probs, 5, /*replacement=*/false);
        int64_t* d = out.data<int64_t>();
        std::set<int64_t> seen;
        std::cout << "  Samples: ";
        for (int i = 0; i < 5; i++) {
            std::cout << d[i] << " ";
            assert(d[i] >= 0 && d[i] < 5);
            assert(seen.find(d[i]) == seen.end());  // must be unique
            seen.insert(d[i]);
        }
        std::cout << std::endl;
        assert(seen.size() == 5);
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 4: Heavily biased distribution ===" << std::endl;
    {
        Tensor probs(Shape{{4}}, Dtype::Float32);
        probs.set_data<float>({0.0f, 0.0f, 0.0f, 1.0f});
        Tensor out = Tensor::multinomial(probs, 10, /*replacement=*/true);
        int64_t* d = out.data<int64_t>();
        for (int i = 0; i < 10; i++) {
            assert(d[i] == 3);  // only index 3 has non-zero weight
        }
        std::cout << "  All 10 samples == 3 (as expected)" << std::endl;
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 5: Error — too many samples without replacement ===" << std::endl;
    {
        Tensor probs(Shape{{4}}, Dtype::Float32);
        probs.set_data<float>({0.0f, 0.0f, 1.0f, 1.0f});  // only 2 non-zero
        bool caught = false;
        try {
            Tensor::multinomial(probs, 3, /*replacement=*/false);
        } catch (const std::runtime_error& e) {
            caught = true;
            std::cout << "  Caught expected error: " << e.what() << std::endl;
        }
        assert(caught);
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 6: Unnormalized weights (should work fine) ===" << std::endl;
    {
        Tensor probs(Shape{{3}}, Dtype::Float32);
        probs.set_data<float>({100.0f, 200.0f, 300.0f});
        Tensor out = Tensor::multinomial(probs, 5, /*replacement=*/true);
        int64_t* d = out.data<int64_t>();
        std::cout << "  Samples: ";
        for (int i = 0; i < 5; i++) {
            std::cout << d[i] << " ";
            assert(d[i] >= 0 && d[i] < 3);
        }
        std::cout << std::endl;
        std::cout << "  PASSED ✅" << std::endl;
    }

    std::cout << "\n✅ All multinomial tests passed!" << std::endl;
    return 0;
}
