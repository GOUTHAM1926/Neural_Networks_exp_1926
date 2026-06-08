#include <iostream>
#include <vector>
#include <numeric>
#include <cassert>
#include "core/Tensor.h"

using namespace OwnTensor;

void test_basic_sharding() {
    std::cout << "Running test_basic_sharding..." << std::endl;
    // creates tensor [0, 1, ..., 9]
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({ {1, 10} }, TensorOptions().with_dtype(Dtype::Float32));
    source.set_data(data);

    Tensor dest1 = Tensor::zeros({ {1, 2} }, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest2 = Tensor::zeros({ {1, 3} }, TensorOptions().with_dtype(Dtype::Float32));
    Tensor dest3 = Tensor::zeros({ {1, 5} }, TensorOptions().with_dtype(Dtype::Float32));

    std::vector<Tensor> destinations = {dest1, dest2, dest3}; // copies of handles

    std::cout << "Calling shard_into..." << std::endl;
    source.shard_into(destinations);
    std::cout << "Returned from shard_into" << std::endl;

    // Verify content
    float* d1 = dest1.data<float>();
    assert(d1[0] == 0.0f && d1[1] == 1.0f);

    float* d2 = dest2.data<float>();

    if (!(d2[0] == 2.0f && d2[1] == 3.0f && d2[2] == 4.0f)) {
        std::cout << "d2 check failed! Got: " << d2[0] << ", " << d2[1] << ", " << d2[2] << std::endl;
        std::cout << "Expected: 2.0, 3.0, 4.0" << std::endl;
    }
    assert(d2[0] == 2.0f && d2[1] == 3.0f && d2[2] == 4.0f);

    float* d3 = dest3.data<float>();
    for(int i=0; i<5; ++i) assert(d3[i] == 5.0f + i);

    std::cout << "test_basic_sharding PASSED" << std::endl;
}

void test_memory_stability() {
    std::cout << "Running test_memory_stability..." << std::endl;
    
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    Tensor source = Tensor({ {1, 10} }, TensorOptions());
    source.set_data(data);
    
    Tensor dest1 = Tensor::zeros({ {1, 5} });
    void* original_ptr = dest1.data();
    
    std::vector<Tensor> destinations = {dest1};
    source.shard_into(destinations);
    
    assert(dest1.data() == original_ptr);
    assert(dest1.data<float>()[0] == 0.0f);
    
    std::cout << "test_memory_stability PASSED" << std::endl;
}

void test_boundary_check() {
    std::cout << "Running test_boundary_check..." << std::endl;
    
    Tensor source = Tensor::zeros({ {1, 10} });
    Tensor dest1 = Tensor::zeros({ {1, 5} });
    Tensor dest2 = Tensor::zeros({ {1, 6} }); // Total 11 > 10
    
    std::vector<Tensor> destinations = {dest1, dest2};
    
    try {
        source.shard_into(destinations);
        assert(false && "Should have thrown runtime_error");
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
    
    std::cout << "test_boundary_check PASSED" << std::endl;
}

void test_multidimensional_sharding() {
    std::cout << "Running test_multidimensional_sharding..." << std::endl;
    // Source: [4, 4] matrix (16 elements)
    // Dests: [2, 4] (8 elems), [8] (8 elems)
    
    std::vector<float> data(16);
    std::iota(data.begin(), data.end(), 0.0f);
    
    Tensor source = Tensor({ {4, 4} }, TensorOptions());
    source.set_data(data);
    
    Tensor dest1 = Tensor::zeros({ {2, 4} });
    Tensor dest2 = Tensor::zeros({ {8} });
    
    std::vector<Tensor> destinations = {dest1, dest2};
    source.shard_into(destinations);
    
    float* d1 = dest1.data<float>();
    for(int i=0; i<8; ++i) assert(d1[i] == (float)i);
    
    float* d2 = dest2.data<float>();
    for(int i=0; i<8; ++i) assert(d2[i] == (float)(i+8));
    
    std::cout << "test_multidimensional_sharding PASSED" << std::endl;
}

void test_sharding_from_view() {
    std::cout << "Running test_sharding_from_view..." << std::endl;
    // Create source that is ITSELF a view with an offset
    // Original: [20], View: elements [5..15] (10 elems)
    
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);
    Tensor original = Tensor({ {20} }, TensorOptions());
    original.set_data(data);
    
    // Create a slice manually or using existing slice if available, 
    // but here we trust slice() or just expect sharding to work on itself.
    // Let's use internal slicing logic if exposed, or just rely on the fact 
    // that if we shard *into* a temp, that temp is a normal tensor. 
    // Wait, we need to test calling shard_into ON a tensor that has storage_offset > 0.
    
    // Let's make a slice using current API if possible.
    // Tensor::slice(start, length) exists in ParallellismUtils.cpp but it COPIES.
    // We need to verify `shard_into` works when `this->storage_offset()` is non-zero.
    // Since we don't have a public "make_view" easily, we can simulate it 
    // by using our own shard_into to make the 'source' first!
    
    Tensor intermediate_source = Tensor::zeros({ {10} });
    Tensor dummy_prefix = Tensor::zeros({ {5} });
    Tensor dummy_suffix = Tensor::zeros({ {5} });
    
    std::vector<Tensor> setup_dests = {dummy_prefix, intermediate_source, dummy_suffix};
    original.shard_into(setup_dests); 
    // Now intermediate_source has data [5..14], BUT it's a copy because shard_into copies TO dest.
    // This doesn't test `this->storage_offset` logic effectively because `intermediate_source` 
    // owns its storage (it was filled by copy).
    
    // To properly test offset logic, we'd need to mock a tensor with offset or rely on internal access.
    // Instead, let's trust the logic: `size_t view_elem_offset = this->storage_offset() + current_elem_offset;`
    // If we can't easily create a view source, we'll skip this specific edge case for now 
    // unless we use `make_shards_inplace` which returns views!
    
    std::vector<Tensor> shards = original.make_shards_inplace(2, true); // Split 20 -> 10, 10
    Tensor view_source = shards[1]; // Indices [10..19], offset=10
    
    Tensor dest1 = Tensor::zeros({ {5} });
    Tensor dest2 = Tensor::zeros({ {5} });
    std::vector<Tensor> final_dests = {dest1, dest2};
    
    view_source.shard_into(final_dests);
    
    float* d1 = dest1.data<float>();
    for(int i=0; i<5; ++i) assert(d1[i] == 10.0f + i); // Should be 10, 11, 12, 13, 14
    
    std::cout << "test_sharding_from_view PASSED" << std::endl;
}

void test_large_allocation() {
    std::cout << "Running test_large_allocation..." << std::endl;
    // 1 Million elements ~ 4MB
    size_t N = 1000000;
    Tensor source = Tensor({ {1, (int64_t)N} }, TensorOptions());
    // We won't fill it all, just check first and last
    float* ptr = source.data<float>();
    ptr[0] = 123.0f;
    ptr[N-1] = 456.0f;
    
    Tensor dest1 = Tensor::zeros({ {1, (int64_t)(N/2)} });
    Tensor dest2 = Tensor::zeros({ {1, (int64_t)(N/2)} });
    std::vector<Tensor> dests = {dest1, dest2};
    
    source.shard_into(dests);
    
    assert(dest1.data<float>()[0] == 123.0f);
    assert(dest2.data<float>()[(N/2)-1] == 456.0f);
    
    std::cout << "test_large_allocation PASSED" << std::endl;
}

void print_tensor_data(const std::string& name, Tensor& t, int count) {
    std::cout << name << " [";
    float* ptr = t.data<float>();
    for(int i=0; i<count; ++i) {
        std::cout << ptr[i] << (i < count-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;
}

void test_visual_example() {
    std::cout << "\n=== Visual Sharding Example ===" << std::endl;
    // Source: [10]
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 100.0f); // 100, 101, ...
    
    Tensor source = Tensor({ {1, 10} }, TensorOptions());
    source.set_data(data);
    print_tensor_data("Source     ", source, 10);
    
    Tensor dest1 = Tensor::zeros({ {1, 3} });
    Tensor dest2 = Tensor::zeros({ {1, 2} });
    Tensor dest3 = Tensor::zeros({ {1, 5} });
    
    std::cout << "\n--- Before Sharding ---" << std::endl;
    print_tensor_data("Dest 1 (3) ", dest1, 3);
    print_tensor_data("Dest 2 (2) ", dest2, 2);
    print_tensor_data("Dest 3 (5) ", dest3, 5);
    
    std::vector<Tensor> dests = {dest1, dest2, dest3};
    source.shard_into(dests);
    
    std::cout << "\n--- After Sharding ---" << std::endl;
    print_tensor_data("Dest 1 (3) ", dest1, 3);
    print_tensor_data("Dest 2 (2) ", dest2, 2);
    print_tensor_data("Dest 3 (5) ", dest3, 5);
    std::cout << "=============================\n" << std::endl;
}

int main() {
    try {
        test_basic_sharding();
        test_memory_stability();
        test_boundary_check();
        test_multidimensional_sharding();
        test_sharding_from_view();
        test_large_allocation();
        test_visual_example();
        std::cout << "ALL TESTS PASSED" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
