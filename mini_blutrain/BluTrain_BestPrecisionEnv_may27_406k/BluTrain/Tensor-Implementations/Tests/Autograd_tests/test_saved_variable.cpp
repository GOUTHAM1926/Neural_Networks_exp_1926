/**
 * @file test_saved_variable.cpp
 * @brief Comprehensive test for SavedVariable with version checking.
 * 
 * Tests:
 * 1. Basic save/unpack cycle
 * 2. In-place modification detection
 * 3. Properties preservation
 * 4. Empty saved variable handling
 * 
 * ## Run command
 * ```bash
 * make run-snippet FILE=Tests/test_saved_variable.cpp
 * ```
 */

#include "core/Tensor.h"
#include "autograd/SavedVariable.h"
#include "autograd/Functions.h"
#include <iostream>
#include <cassert>

using namespace OwnTensor;

#define TEST_PASS(msg) std::cout << "✓ " << msg << std::endl
#define TEST_FAIL(msg) std::cout << "✗ " << msg << std::endl; return false
#define TEST_SECTION(msg) std::cout << "\n=== " << msg << " ===" << std::endl

// ============================================================================
// TEST 1: Basic SavedVariable Usage
// ============================================================================
/**
 * Test basic save and unpack without any modifications.
 */
bool test_basic_save_unpack() {
    TEST_SECTION("Test 1: Basic Save/Unpack");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor t = Tensor::randn<float>(Shape{{3, 3}}, opts);
    
    // Fill with known values (using const_cast since Tensor doesn't expose mutable_data)
    float* data = const_cast<float*>(t.data<float>());
    for (size_t i = 0; i < 9; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    // Save the tensor
    SavedVariable saved(t, false);  // Not an output
    
    if (!saved.defined()) {
        TEST_FAIL("SavedVariable should be defined after save");
    }
    TEST_PASS("SavedVariable is defined after save");
    
    // Unpack without modification - should work
    Tensor unpacked = saved.unpack();
    
    if (!unpacked.unsafeGetTensorImpl()) {
        TEST_FAIL("Unpacked tensor should be valid");
    }
    TEST_PASS("Unpack returns valid tensor");
    
    // Verify values preserved
    const float* unpacked_data = unpacked.data<float>();
    bool values_match = true;
    for (size_t i = 0; i < 9; ++i) {
        if (unpacked_data[i] != static_cast<float>(i)) {
            values_match = false;
            break;
        }
    }
    if (!values_match) {
        TEST_FAIL("Unpacked values don't match original");
    }
    TEST_PASS("Values preserved correctly");
    
    return true;
}

// ============================================================================
// TEST 2: In-Place Modification Detection
// ============================================================================
/**
 * Test that in-place modifications are detected.
 */
bool test_inplace_detection() {
    TEST_SECTION("Test 2: In-Place Modification Detection");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor t = Tensor::ones(Shape{{2, 2}}, opts);
    
    // Save the tensor
    SavedVariable saved(t, false);
    
    uint32_t original_version = saved.saved_version();
    std::cout << "  Original version: " << original_version << std::endl;
    
    // Modify in-place by bumping version
    increment_version(t);
    
    uint32_t new_version = t.unsafeGetTensorImpl()->version();
    std::cout << "  After increment: " << new_version << std::endl;
    
    if (new_version <= original_version) {
        TEST_FAIL("Version should increase after increment");
    }
    TEST_PASS("Version counter incremented");
    
    // Check was_modified_inplace
    if (!was_modified_inplace(t, original_version)) {
        TEST_FAIL("was_modified_inplace should return true");
    }
    TEST_PASS("was_modified_inplace() detects modification");
    
    // Unpack should fail with error
    bool exception_thrown = false;
    try {
        saved.unpack();
    } catch (const std::runtime_error& e) {
        exception_thrown = true;
        std::cout << "  Exception message: " << e.what() << std::endl;
    }
    
    if (!exception_thrown) {
        TEST_FAIL("unpack() should throw on in-place modification");
    }
    TEST_PASS("unpack() throws on in-place modification");
    
    return true;
}

// ============================================================================
// TEST 3: Properties Preservation
// ============================================================================
/**
 * Test that SavedVariable preserves tensor properties.
 */
bool test_properties_preservation() {
    TEST_SECTION("Test 3: Properties Preservation");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor t = Tensor::ones(Shape{{4, 4}}, opts);
    
    // Save with is_output=false
    SavedVariable saved_input(t, false);
    
    if (saved_input.is_output()) {
        TEST_FAIL("is_output should be false");
    }
    TEST_PASS("is_output=false preserved");
    
    if (!saved_input.requires_grad()) {
        TEST_FAIL("requires_grad should be true");
    }
    TEST_PASS("requires_grad preserved");
    
    if (!saved_input.was_leaf()) {
        TEST_FAIL("was_leaf should be true (no grad_fn)");
    }
    TEST_PASS("was_leaf preserved");
    
    // Save with is_output=true
    SavedVariable saved_output(t, true);
    
    if (!saved_output.is_output()) {
        TEST_FAIL("is_output should be true");
    }
    TEST_PASS("is_output=true preserved");
    
    return true;
}

// ============================================================================
// TEST 4: Empty SavedVariable
// ============================================================================
/**
 * Test empty SavedVariable handling.
 */
bool test_empty_saved_variable() {
    TEST_SECTION("Test 4: Empty SavedVariable");
    
    // Default constructor
    SavedVariable empty;
    
    if (empty.defined()) {
        TEST_FAIL("Default SavedVariable should not be defined");
    }
    TEST_PASS("Default SavedVariable is not defined");
    
    // Unpack empty should return empty tensor
    Tensor unpacked = empty.unpack();
    if (unpacked.unsafeGetTensorImpl() != nullptr) {
        TEST_FAIL("Unpack of empty should return empty tensor");
    }
    TEST_PASS("Unpack of empty returns empty tensor");
    
    // Reset
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor t = Tensor::ones(Shape{{2, 2}}, opts);
    SavedVariable saved(t, false);
    
    if (!saved.defined()) {
        TEST_FAIL("Saved should be defined before reset");
    }
    
    saved.reset();
    
    if (saved.defined()) {
        TEST_FAIL("Saved should not be defined after reset");
    }
    TEST_PASS("reset() clears SavedVariable");
    
    return true;
}

// ============================================================================
// TEST 5: Non-Leaf Tensor Saving
// ============================================================================
/**
 * Test saving tensors with grad_fn (non-leaf).
 */
bool test_non_leaf_saving() {
    TEST_SECTION("Test 5: Non-Leaf Tensor Saving");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor a = Tensor::ones(Shape{{2, 2}}, opts);
    
    // Create a grad_fn
    Tensor b = Tensor::ones(Shape{{2, 2}}, opts);
    auto grad_fn = std::make_shared<autograd::AddBackward>(a.shape(), b.shape());
    a.set_grad_fn(grad_fn);
    
    if (a.is_leaf()) {
        TEST_FAIL("Tensor with grad_fn should not be leaf");
    }
    TEST_PASS("Tensor with grad_fn is not leaf");
    
    // Save non-leaf tensor
    SavedVariable saved(a, false);
    
    if (saved.was_leaf()) {
        TEST_FAIL("was_leaf should be false for non-leaf tensor");
    }
    TEST_PASS("was_leaf=false for non-leaf tensor");
    
    // Unpack should work
    Tensor unpacked = saved.unpack();
    if (!unpacked.unsafeGetTensorImpl()) {
        TEST_FAIL("Unpack should return valid tensor");
    }
    TEST_PASS("Non-leaf tensor unpacked successfully");
    
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         SAVEDVARIABLE VERSION CHECKING TEST SUITE              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    int passed = 0;
    int failed = 0;
    
    auto run_test = [&](bool (*test_fn)(), const char* name) {
        try {
            if (test_fn()) {
                passed++;
            } else {
                failed++;
                std::cout << "FAILED: " << name << std::endl;
            }
        } catch (const std::exception& e) {
            failed++;
            std::cout << "EXCEPTION in " << name << ": " << e.what() << std::endl;
        }
    };
    
    run_test(test_basic_save_unpack, "test_basic_save_unpack");
    run_test(test_inplace_detection, "test_inplace_detection");
    run_test(test_properties_preservation, "test_properties_preservation");
    run_test(test_empty_saved_variable, "test_empty_saved_variable");
    run_test(test_non_leaf_saving, "test_non_leaf_saving");
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                         SUMMARY                                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "  Tests PASSED: " << passed << std::endl;
    std::cout << "  Tests FAILED: " << failed << std::endl;
    std::cout << "\n";
    
    if (failed == 0) {
        std::cout << "✅ ALL TESTS PASSED!\n\n";
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED!\n\n";
        return 1;
    }
}
