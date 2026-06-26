#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace OwnTensor;

int main() {
    std::cout << "=== Test 1: 1D input, k=3, largest=true, sorted=true ===" << std::endl;
    {
        Tensor t(Shape{{5}}, Dtype::Float32);
        t.set_data<float>({1.0f, 5.0f, 2.0f, 4.0f, 3.0f});
        
        auto res = t.topk(3, 0, true, true);
        Tensor vals = res.first;
        Tensor idxs = res.second;
        
        assert(vals.shape().dims[0] == 3);
        assert(idxs.shape().dims[0] == 3);
        
        float* v = vals.data<float>();
        int64_t* i = idxs.data<int64_t>();
        
        std::cout << "Values: " << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
        std::cout << "Indices: " << i[0] << ", " << i[1] << ", " << i[2] << std::endl;
        
        assert(v[0] == 5.0f && i[0] == 1);
        assert(v[1] == 4.0f && i[1] == 3);
        assert(v[2] == 3.0f && i[2] == 4);
        std::cout << "PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 2: 2D input query dim 1 ===" << std::endl;
    {
        Tensor t(Shape{{2, 4}}, Dtype::Float32);
        t.set_data<float>({
            10.0f, 20.0f, 30.0f, 40.0f, 
            400.0f, 300.0f, 200.0f, 100.0f
        });
        
        auto res = t.topk(2, 1, true, true);
        Tensor vals = res.first;
        Tensor idxs = res.second;
        
        assert(vals.shape().dims[0] == 2);
        assert(vals.shape().dims[1] == 2);
        
        float* v = vals.data<float>();
        int64_t* i_ptr = idxs.data<int64_t>();
        
        std::cout << "Row 0 Vals: " << v[0] << ", " << v[1] << std::endl;
        std::cout << "Row 0 Idxs: " << i_ptr[0] << ", " << i_ptr[1] << std::endl;
        assert(v[0] == 40.0f && i_ptr[0] == 3);
        assert(v[1] == 30.0f && i_ptr[1] == 2);
        
        std::cout << "Row 1 Vals: " << v[2] << ", " << v[3] << std::endl;
        std::cout << "Row 1 Idxs: " << i_ptr[2] << ", " << i_ptr[3] << std::endl;
        assert(v[2] == 400.0f && i_ptr[2] == 0);
        assert(v[3] == 300.0f && i_ptr[3] == 1);
        std::cout << "PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 3: 2D input query dim 0 ===" << std::endl;
    {
        Tensor t(Shape{{3, 2}}, Dtype::Float32);
        t.set_data<float>({
            1.0f, 10.0f,
            3.0f, 30.0f,
            2.0f, 20.0f
        });
        
        // topk(2, dim=0) -> should pick from cols
        // Col 0: 1, 3, 2 -> Top 2: 3, 2 (indices 1, 2)
        // Col 1: 10, 30, 20 -> Top 2: 30, 20 (indices 1, 2)
        
        auto res = t.topk(2, 0, true, true);
        Tensor vals = res.first;
        Tensor idxs = res.second;

        assert(vals.shape().dims[0] == 2);
        assert(vals.shape().dims[1] == 2);

        float* v = vals.data<float>();
        int64_t* i_ptr = idxs.data<int64_t>();
        
        std::cout << "Col 0 Top 1: " << v[0]; // Row 0, Col 0 in output? No, dims are preserved? 
        // PyTorch behavior: if input (3,2), topk(2, dim=0) -> (2,2)
        // Result[0, 0] is top1 of col 0. Result[0, 1] is top1 of col 1.
        // Result[1, 0] is top2 of col 0. Result[1, 1] is top2 of col 1.
        
        // Storage is row-major. v[0] = (0,0), v[1] = (0,1), v[2] = (1,0), v[3] = (1,1)
        
        std::cout << " (0,0): " << v[0] << ", idx: " << i_ptr[0] << std::endl; // Should be 3.0, idx 1
        std::cout << " (0,1): " << v[1] << ", idx: " << i_ptr[1] << std::endl; // Should be 30.0, idx 1
        std::cout << " (1,0): " << v[2] << ", idx: " << i_ptr[2] << std::endl; // Should be 2.0, idx 2
        std::cout << " (1,1): " << v[3] << ", idx: " << i_ptr[3] << std::endl; // Should be 20.0, idx 2
        
        assert(v[0] == 3.0f && i_ptr[0] == 1);
        assert(v[1] == 30.0f && i_ptr[1] == 1);
        assert(v[2] == 2.0f && i_ptr[2] == 2);
        assert(v[3] == 20.0f && i_ptr[3] == 2);
        std::cout << "PASSED ✅" << std::endl;
    }
    
    std::cout << "\n=== Test 4: largest=false (smallest) ===" << std::endl;
    {
        Tensor t(Shape{{5}}, Dtype::Float32);
        t.set_data<float>({5.0f, 1.0f, 4.0f, 2.0f, 3.0f});
        
        auto res = t.topk(2, 0, false, true); // smallest 2
        float* v = res.first.data<float>();
        
        std::cout << "Smallest: " << v[0] << ", " << v[1] << std::endl;
        assert(v[0] == 1.0f);
        assert(v[1] == 2.0f);
        std::cout << "PASSED ✅" << std::endl;
    }

    std::cout << "\n=== Test 5: sorted=false ===" << std::endl;
    {
         Tensor t(Shape{{5}}, Dtype::Float32);
         t.set_data<float>({10.0f, 50.0f, 20.0f, 40.0f, 30.0f});
         
         auto res = t.topk(3, 0, true, false); // top 3 unsorted (50, 40, 30)
         float* v = res.first.data<float>();
         int64_t* i = res.second.data<int64_t>();
         
         std::cout << "Unsorted top 3: " << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
         
         // Check that correct elements are preset, order doesn't matter
         std::vector<float> out_vec = {v[0], v[1], v[2]};
         std::sort(out_vec.begin(), out_vec.end());
         assert(out_vec[0] == 30.0f);
         assert(out_vec[1] == 40.0f);
         assert(out_vec[2] == 50.0f);
         
         // Helper to check index correctness
         for(int j=0; j<3; ++j) {
             if (v[j] == 50.0f) assert(i[j] == 1);
             if (v[j] == 40.0f) assert(i[j] == 3);
             if (v[j] == 30.0f) assert(i[j] == 4);
         }
         std::cout << "PASSED ✅" << std::endl;
    }

    std::cout << "\n✅ All topk tests passed!" << std::endl;
    return 0;
}
