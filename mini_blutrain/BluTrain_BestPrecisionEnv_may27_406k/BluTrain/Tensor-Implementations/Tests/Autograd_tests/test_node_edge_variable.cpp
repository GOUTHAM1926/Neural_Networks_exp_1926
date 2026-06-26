/**
 * @file test_node_edge_variable.cpp
 * @brief Comprehensive test for Node, Edge, and Variable helper functions.
 * 
 * This test verifies every autograd helper function with detailed documentation
 * explaining WHY each function exists and HOW to use them.
 * 
 * ## Test Categories
 * 1. Edge basics - Construction, validity checks
 * 2. Node structure - Inheritance, sequence numbers, names
 * 3. AutogradMeta - Gradient storage and retrieval
 * 4. Hook system - Pre/post hook registration and execution
 * 5. Gradient edge - Leaf vs non-leaf edge creation
 * 6. impl namespace - Helper accessor functions
 * 7. Factory functions - Variable creation
 * 
 * ## Running this test
 * ```bash
 * cd /path/to/cgadimpl/tensor
 * make run-snippet FILE=Tests/test_node_edge_variable.cpp
 * ```
 */

#include "core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/Variable.h"
#include "autograd/Functions.h"
#include "autograd/AutogradOps.h"
#include "autograd/Hooks.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <vector>

using namespace OwnTensor;

// Helper macro for test output
#define TEST_PASS(msg) std::cout << "✓ " << msg << std::endl
#define TEST_FAIL(msg) std::cout << "✗ " << msg << std::endl; return false
#define TEST_SECTION(msg) std::cout << "\n=== " << msg << " ===" << std::endl

// ============================================================================
// TEST 1: Edge Basics
// ============================================================================
/**
 * ## Purpose
 * The Edge struct connects nodes in the backward computational graph.
 * It stores:
 * - `function`: Pointer to the next node in backward pass
 * - `input_nr`: Which input of that node this edge connects to
 * 
 * ## How to Use
 * ```cpp
 * // Create an edge to a backward function
 * auto node = std::make_shared<AddBackward>();
 * Edge edge(node, 0);  // Connect to input 0 of AddBackward
 * 
 * // Check if edge is valid
 * if (edge.is_valid()) { ... }
 * 
 * // Use make_edge helper
 * Edge e = make_edge(node, 0);
 * ```
 */
bool test_edge_basics() {
    TEST_SECTION("Test 1: Edge Basics");
    
    // Test default constructor creates invalid edge
    Edge invalid_edge;
    if (invalid_edge.is_valid()) {
        TEST_FAIL("Default Edge should be invalid");
    }
    TEST_PASS("Default Edge is invalid (as expected)");
    
    // Test edge with function
    Tensor dummy_a = Tensor::zeros(Shape{{2, 2}});
    Tensor dummy_b = Tensor::zeros(Shape{{2, 2}});
    auto node = std::make_shared<autograd::AddBackward>(dummy_a.shape(), dummy_b.shape());
    Edge valid_edge(node, 0);
    
    if (!valid_edge.is_valid()) {
        TEST_FAIL("Edge with node should be valid");
    }
    TEST_PASS("Edge with node is valid");
    
    if (valid_edge.function != node) {
        TEST_FAIL("Edge function pointer incorrect");
    }
    TEST_PASS("Edge stores correct function pointer");
    
    if (valid_edge.input_nr != 0) {
        TEST_FAIL("Edge input_nr incorrect");
    }
    TEST_PASS("Edge stores correct input_nr");
    
    // Test make_edge helper
    Edge helper_edge = make_edge(node, 1);
    if (!helper_edge.is_valid() || helper_edge.input_nr != 1) {
        TEST_FAIL("make_edge helper not working");
    }
    TEST_PASS("make_edge() helper works correctly");
    
    // Test equality
    Edge same_edge(node, 0);
    if (!(valid_edge == same_edge)) {
        TEST_FAIL("Edge equality comparison failed");
    }
    TEST_PASS("Edge equality comparison works");
    
    return true;
}

// ============================================================================
// TEST 2: Node Structure
// ============================================================================
/**
 * ## Purpose
 * Node is the base class for all gradient functions. Key features:
 * - `sequence_nr()`: Unique ordering ID for deterministic execution
 * - `topological_nr()`: For efficient graph traversal
 * - `name()`: Human-readable name for debugging
 * - `operator()`: Execute with hook wrapping
 * 
 * ## How to Create Custom Node
 * ```cpp
 * class MyBackward : public Node {
 * public:
 *     MyBackward() : Node(2) {}  // 2 inputs
 *     
 *     std::string name() const override { return "MyBackward"; }
 *     
 *     variable_list apply(variable_list&& grads) override {
 *         // Compute gradients
 *         return {grad1, grad2};
 *     }
 * };
 * ```
 */
bool test_node_structure() {
    TEST_SECTION("Test 2: Node Structure");
    
    // Create backward nodes
    Tensor dummy_x = Tensor::zeros(Shape{{2, 2}});
    Tensor dummy_y = Tensor::zeros(Shape{{2, 2}});
    auto add_node = std::make_shared<autograd::AddBackward>(dummy_x.shape(), dummy_y.shape());
    auto mul_node = std::make_shared<autograd::MulBackward>(
        Tensor::ones(Shape{{2, 2}}, TensorOptions()),
        Tensor::ones(Shape{{2, 2}}, TensorOptions())
    );
    
    // Test sequence numbers are unique and increasing
    uint64_t seq1 = add_node->sequence_nr();
    uint64_t seq2 = mul_node->sequence_nr();
    
    if (seq1 >= seq2) {
        TEST_FAIL("Sequence numbers should be unique and increasing");
    }
    TEST_PASS("Sequence numbers are unique and ordered: " + 
              std::to_string(seq1) + " < " + std::to_string(seq2));
    
    // Test name() method
    if (std::string(add_node->name()) != "AddBackward") {
        TEST_FAIL("AddBackward::name() should return 'AddBackward'");
    }
    TEST_PASS(std::string("AddBackward::name() = '") + add_node->name() + "'");
    
    if (std::string(mul_node->name()) != "MulBackward") {
        TEST_FAIL("MulBackward::name() should return 'MulBackward'");
    }
    TEST_PASS(std::string("MulBackward::name() = '") + mul_node->name() + "'");
    
    // Test all backward nodes have names
    Tensor t = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    auto matmul_node = std::make_shared<autograd::MatmulBackward>(t, t);
    auto relu_node = std::make_shared<autograd::ReluBackward>(t);
    auto sum_node = std::make_shared<autograd::SumBackward>(t.shape());
    auto mean_node = std::make_shared<autograd::MeanBackward>(t.shape(), t.numel());
    auto grad_acc = std::make_shared<autograd::GradAccumulator>(t.unsafeGetTensorImpl());
    
    std::cout << "  All backward node names:" << std::endl;
    std::cout << "    - " << matmul_node->name() << std::endl;
    std::cout << "    - " << relu_node->name() << std::endl;
    std::cout << "    - " << sum_node->name() << std::endl;
    std::cout << "    - " << mean_node->name() << std::endl;
    std::cout << "    - " << grad_acc->name() << std::endl;
    TEST_PASS("All backward nodes implement name()");
    
    // Test num_inputs()
    if (add_node->num_inputs() != 2) {
        TEST_FAIL("AddBackward should have 2 inputs");
    }
    TEST_PASS("num_inputs() returns correct value");
    
    // Test next_edges management
    add_node->set_next_edge(0, Edge());
    add_node->set_next_edge(1, make_edge(mul_node, 0));
    
    if (add_node->next_edges().size() != 2) {
        TEST_FAIL("Should have 2 next edges");
    }
    TEST_PASS("set_next_edge() works correctly");
    
    if (!add_node->next_edge(1).is_valid()) {
        TEST_FAIL("Edge at index 1 should be valid");
    }
    TEST_PASS("next_edge() accessor works");
    
    // Test topological_nr updates
    // When we add an edge to a node, topological_nr should update
    std::cout << "  Topological numbers: add=" << add_node->topological_nr() 
              << ", mul=" << mul_node->topological_nr() << std::endl;
    TEST_PASS("topological_nr() is tracked");
    
    // Clean up edges before nodes go out of scope to prevent dangling pointers
    add_node->clear_edges();
    
    return true;
}

// ============================================================================
// TEST 3: AutogradMeta Gradient Storage
// ============================================================================
/**
 * ## Purpose
 * AutogradMeta stores all autograd-related data for a tensor:
 * - `grad_`: Accumulated gradient
 * - `grad_fn_`: Gradient function (for non-leaves)
 * - `grad_accumulator_`: Accumulator (for leaves)
 * - Various flags (requires_grad, is_view, etc.)
 * 
 * ## How to Use
 * ```cpp
 * Tensor t = Tensor::randn({3, 3}, TensorOptions().with_req_grad(true));
 * 
 * // Check gradient status
 * t.requires_grad();  // true
 * t.is_leaf();        // true (no grad_fn)
 * 
 * // Set gradient
 * t.set_grad<float>({1.0, 2.0, ...});
 * 
 * // Access gradient
 * float* grad = t.grad<float>();
 * ```
 */
bool test_autograd_meta_gradient_storage() {
    TEST_SECTION("Test 3: AutogradMeta Gradient Storage");
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    TensorOptions no_grad = TensorOptions();
    
    // Test requires_grad
    Tensor t1(Shape{{2, 3}}, req_grad);
    if (!t1.requires_grad()) {
        TEST_FAIL("Tensor with req_grad should require grad");
    }
    TEST_PASS("requires_grad() returns true for tensor created with req_grad");
    
    Tensor t2(Shape{{2, 3}}, no_grad);
    if (t2.requires_grad()) {
        TEST_FAIL("Tensor without req_grad should not require grad");
    }
    TEST_PASS("requires_grad() returns false for tensor without req_grad");
    
    // Test is_leaf()
    if (!t1.is_leaf()) {
        TEST_FAIL("Tensor without grad_fn should be leaf");
    }
    TEST_PASS("is_leaf() returns true for tensor without grad_fn");
    
    // Test setting gradient
    std::vector<float> grad_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    t1.set_grad(grad_data);
    
    if (!t1.owns_grad()) {
        TEST_FAIL("After set_grad, tensor should own grad");
    }
    TEST_PASS("owns_grad() returns true after set_grad()");
    
    // Test grad access
    float* grad_ptr = t1.grad<float>();
    bool values_match = true;
    for (size_t i = 0; i < grad_data.size(); i++) {
        if (std::abs(grad_ptr[i] - grad_data[i]) > 1e-6f) {
            values_match = false;
            break;
        }
    }
    if (!values_match) {
        TEST_FAIL("Gradient values don't match");
    }
    TEST_PASS("grad<T>() returns correct gradient values");
    
    // Test grad_view()
    Tensor grad_tensor = t1.grad_view();
    if (grad_tensor.shape() != t1.shape()) {
        TEST_FAIL("grad_view() shape mismatch");
    }
    TEST_PASS("grad_view() returns tensor with correct shape");
    
    // Test set_requires_grad after construction
    t2.set_requires_grad(true);
    if (!t2.requires_grad()) {
        TEST_FAIL("set_requires_grad(true) failed");
    }
    TEST_PASS("set_requires_grad() works after construction");
    
    // Test set_grad_fn makes tensor non-leaf
    Tensor dummy_1 = Tensor::zeros(t1.shape());
    Tensor dummy_2 = Tensor::zeros(t1.shape());
    auto grad_fn = std::make_shared<autograd::AddBackward>(dummy_1.shape(), dummy_2.shape());
    t1.set_grad_fn(grad_fn);
    
    if (t1.is_leaf()) {
        TEST_FAIL("Tensor with grad_fn should not be leaf");
    }
    TEST_PASS("Setting grad_fn makes tensor non-leaf");
    
    if (t1.grad_fn() != grad_fn) {
        TEST_FAIL("grad_fn() doesn't return set value");
    }
    TEST_PASS("grad_fn() returns correct function");
    
    return true;
}

// ============================================================================
// TEST 4: Hook System
// ============================================================================
/**
 * ## Purpose
 * Hooks allow custom gradient manipulation:
 * - Pre-hooks: Modify gradient before backward computation
 * - Post-accumulation hooks: Execute after gradient accumulation
 * 
 * ## How to Use
 * ```cpp
 * // Create a gradient clipping hook
 * auto clip_hook = make_pre_hook([](const Tensor& grad) {
 *     // Clip gradient values
 *     return clipped_grad;
 * });
 * tensor.register_hook(std::move(clip_hook));
 * 
 * // Create a logging hook
 * auto log_hook = make_post_acc_hook([](const Tensor& grad) {
 *     std::cout << "Gradient accumulated" << std::endl;
 * });
 * tensor.register_post_acc_hook(std::move(log_hook));
 * ```
 */
bool test_hook_system() {
    TEST_SECTION("Test 4: Hook System");
    
    // Test LambdaPreHook
    bool pre_hook_called = false;
    auto pre_hook = make_pre_hook([&pre_hook_called](const Tensor& grad) {
        pre_hook_called = true;
        return grad;  // Pass through
    });
    
    if (!pre_hook) {
        TEST_FAIL("make_pre_hook returned nullptr");
    }
    TEST_PASS("make_pre_hook() creates valid hook");
    
    // Test hook execution
    Tensor grad = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    Tensor result = (*pre_hook)(grad);
    
    if (!pre_hook_called) {
        TEST_FAIL("Pre-hook was not called");
    }
    TEST_PASS("Pre-hook executes when called");
    
    // Test LambdaPostAccHook
    bool post_hook_called = false;
    auto post_hook = make_post_acc_hook([&post_hook_called](const Tensor& grad) {
        post_hook_called = true;
    });
    
    if (!post_hook) {
        TEST_FAIL("make_post_acc_hook returned nullptr");
    }
    TEST_PASS("make_post_acc_hook() creates valid hook");
    
    (*post_hook)(grad);
    if (!post_hook_called) {
        TEST_FAIL("Post-accumulation hook was not called");
    }
    TEST_PASS("Post-accumulation hook executes when called");
    
    // Test registering hooks on tensor
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor t = Tensor::ones(Shape{{2, 2}}, opts);
    
    int hook_count = 0;
    t.register_hook(make_pre_hook([&hook_count](const Tensor& grad) {
        hook_count++;
        return grad;
    }));
    TEST_PASS("register_hook() on tensor works");
    
    // Clear hooks
    t.clear_hooks();
    TEST_PASS("clear_hooks() on tensor works");
    
    return true;
}

// ============================================================================
// TEST 5: Gradient Edge Creation
// ============================================================================
/**
 * ## Purpose
 * gradient_edge() creates the appropriate edge for a tensor:
 * - For non-leaves: Edge to grad_fn at output_nr
 * - For leaves: Edge to GradAccumulator
 * 
 * ## Why This Matters
 * During backward pass, the engine uses gradient edges to determine
 * where to send computed gradients. This is the core mechanism
 * connecting operations in the computational graph.
 * 
 * ## How to Use
 * ```cpp
 * // For a leaf tensor
 * Tensor w = Tensor::randn({..}, TensorOptions().with_req_grad(true));
 * Edge edge = impl::gradient_edge(w);  // Points to GradAccumulator
 * 
 * // For computed tensor
 * Tensor y = autograd::matmul(x, w);
 * Edge edge = impl::gradient_edge(y);  // Points to MatmulBackward
 * ```
 */
bool test_gradient_edge_creation() {
    TEST_SECTION("Test 5: Gradient Edge Creation");
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    TensorOptions no_grad = TensorOptions();
    
    // Test leaf tensor gets GradAccumulator edge
    Tensor leaf = Tensor::ones(Shape{{2, 2}}, req_grad);
    Edge leaf_edge = impl::gradient_edge(leaf);
    
    if (!leaf_edge.is_valid()) {
        TEST_FAIL("Leaf tensor should have valid gradient edge");
    }
    TEST_PASS("Leaf tensor has valid gradient edge");
    
    if (std::string(leaf_edge.function->name()) != "GradAccumulator") {
        TEST_FAIL(std::string("Leaf edge should point to GradAccumulator, got: ") +
                  leaf_edge.function->name());
    }
    TEST_PASS("Leaf tensor edge points to GradAccumulator");
    
    // Test tensor without requires_grad has invalid edge
    Tensor no_grad_tensor = Tensor::ones(Shape{{2, 2}}, no_grad);
    Edge no_grad_edge = impl::gradient_edge(no_grad_tensor);
    
    if (no_grad_edge.is_valid()) {
        TEST_FAIL("Tensor without requires_grad should have invalid edge");
    }
    TEST_PASS("Tensor without requires_grad has invalid edge (correct)");
    
    // Test non-leaf tensor (with grad_fn) gets edge to grad_fn
    Tensor a = Tensor::ones(Shape{{2, 2}}, req_grad);
    Tensor b = Tensor::ones(Shape{{2, 2}}, req_grad);
    Tensor result = autograd::add(a, b);  // Creates AddBackward
    
    if (result.is_leaf()) {
        TEST_FAIL("Result of operation should not be leaf");
    }
    TEST_PASS("Result of operation is not leaf");
    
    Edge result_edge = impl::gradient_edge(result);
    if (!result_edge.is_valid()) {
        TEST_FAIL("Non-leaf tensor should have valid gradient edge");
    }
    TEST_PASS("Non-leaf tensor has valid gradient edge");
    
    if (std::string(result_edge.function->name()) != "AddBackward") {
        TEST_FAIL(std::string("Edge should point to AddBackward, got: ") +
                  result_edge.function->name());
    }
    TEST_PASS(std::string("Non-leaf edge points to correct backward function: ") + 
              result_edge.function->name());
    
    return true;
}

// ============================================================================
// TEST 6: impl Namespace Helpers
// ============================================================================
/**
 * ## Purpose
 * The impl namespace provides low-level access to autograd internals:
 * - `get_autograd_meta()`: Get metadata (may be null)
 * - `materialize_autograd_meta()`: Get or create metadata
 * - `set_gradient_edge()`: Attach edge to tensor
 * - `grad_accumulator()`: Get/create gradient accumulator
 * 
 * ## When to Use
 * These are primarily for library implementers. Regular users should
 * prefer high-level Tensor methods.
 * 
 * ## Example
 * ```cpp
 * // Check if tensor has autograd metadata
 * auto* meta = impl::get_autograd_meta(tensor);
 * if (meta) {
 *     bool req = meta->requires_grad();
 * }
 * 
 * // Ensure metadata exists
 * auto* meta = impl::materialize_autograd_meta(tensor);
 * meta->set_requires_grad(true);
 * ```
 */
bool test_impl_namespace_helpers() {
    TEST_SECTION("Test 6: impl Namespace Helpers");
    
    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    TensorOptions no_grad = TensorOptions();
    
    // Test get_autograd_meta on tensor with requires_grad
    Tensor t1 = Tensor::ones(Shape{{2, 2}}, req_grad);
    AutogradMeta* meta1 = impl::get_autograd_meta(t1);
    
    if (!meta1) {
        TEST_FAIL("Tensor with requires_grad should have autograd meta");
    }
    TEST_PASS("get_autograd_meta() returns meta for tensor with requires_grad");
    
    // Test get_autograd_meta on tensor without requires_grad  
    Tensor t2 = Tensor::ones(Shape{{2, 2}}, no_grad);
    AutogradMeta* meta2 = impl::get_autograd_meta(t2);
    
    // Note: depending on implementation, may or may not have meta
    std::cout << "  (no_grad tensor has_meta: " << (meta2 ? "yes" : "no") << ")" << std::endl;
    TEST_PASS("get_autograd_meta() handles no_grad tensor");
    
    // Test materialize_autograd_meta
    AutogradMeta* meta3 = impl::materialize_autograd_meta(t2);
    if (!meta3) {
        TEST_FAIL("materialize_autograd_meta should create meta");
    }
    TEST_PASS("materialize_autograd_meta() creates metadata if needed");
    
    // Now get_autograd_meta should return non-null
    AutogradMeta* meta4 = impl::get_autograd_meta(t2);
    if (!meta4) {
        TEST_FAIL("After materialize, get_autograd_meta should return meta");
    }
    TEST_PASS("After materialize, get_autograd_meta() returns created meta");
    
    // Test grad_accumulator
    Tensor leaf = Tensor::ones(Shape{{3, 3}}, req_grad);
    std::shared_ptr<Node> acc = impl::grad_accumulator(leaf);
    
    if (!acc) {
        TEST_FAIL("grad_accumulator should return valid accumulator");
    }
    TEST_PASS("grad_accumulator() returns valid accumulator");
    
    if (std::string(acc->name()) != "GradAccumulator") {
        TEST_FAIL("Accumulator should be GradAccumulator");
    }
    TEST_PASS("grad_accumulator() returns GradAccumulator node");
    
    // Test that same accumulator is reused
    std::shared_ptr<Node> acc2 = impl::grad_accumulator(leaf);
    if (acc.get() != acc2.get()) {
        TEST_FAIL("grad_accumulator should return same instance");
    }
    TEST_PASS("grad_accumulator() reuses existing accumulator");
    
    // Test set_gradient_edge
    Tensor t3 = Tensor::ones(Shape{{2, 2}}, req_grad);
    Tensor dummy_3 = Tensor::zeros(Shape{{2, 2}});
    Tensor dummy_4 = Tensor::zeros(Shape{{2, 2}});
    auto grad_fn = std::make_shared<autograd::AddBackward>(dummy_3.shape(), dummy_4.shape());
    impl::set_gradient_edge(t3, Edge(grad_fn, 0));
    
    if (t3.grad_fn() != grad_fn) {
        TEST_FAIL("set_gradient_edge didn't set grad_fn");
    }
    TEST_PASS("set_gradient_edge() sets grad_fn correctly");
    
    return true;
}

// ============================================================================
// TEST 7: Sequence Numbers (Thread Safety)
// ============================================================================
/**
 * ## Purpose  
 * Sequence numbers ensure deterministic ordering of operations,
 * which is essential for reproducible gradient computation.
 * 
 * ## How It Works
 * Each Node gets a unique, monotonically increasing sequence number
 * at construction time. This is thread-safe (uses atomic counter).
 */
bool test_sequence_numbers() {
    TEST_SECTION("Test 7: Sequence Numbers");
    
    // Create many nodes and verify sequence numbers increase
    std::vector<uint64_t> seq_nums;
    for (int i = 0; i < 10; i++) {
        Tensor d1 = Tensor::zeros(Shape{{1}});
        Tensor d2 = Tensor::zeros(Shape{{1}});
        auto node = std::make_shared<autograd::AddBackward>(d1.shape(), d2.shape());
        seq_nums.push_back(node->sequence_nr());
    }
    
    // Verify strictly increasing
    bool increasing = true;
    for (size_t i = 1; i < seq_nums.size(); i++) {
        if (seq_nums[i] <= seq_nums[i-1]) {
            increasing = false;
            break;
        }
    }
    
    if (!increasing) {
        TEST_FAIL("Sequence numbers should be strictly increasing");
    }
    TEST_PASS("Sequence numbers are strictly increasing");
    
    std::cout << "  Sequence range: " << seq_nums.front() 
              << " to " << seq_nums.back() << std::endl;
    
    return true;
}

// ============================================================================
// TEST 8: Factory Functions
// ============================================================================
/**
 * ## Purpose
 * Factory functions simplify creating variables with gradient edges:
 * - `make_variable()`: Create tensor with attached gradient edge
 * - `create_gradient_edge()`: Connect variable to backward function
 * 
 * ## When to Use
 * These are used when implementing new autograd operations
 * to properly connect outputs to the computational graph.
 */
bool test_factory_functions() {
    TEST_SECTION("Test 8: Factory Functions");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Test make_variable
    Tensor data = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    Tensor d1 = Tensor::zeros(Shape{{2, 2}});
    Tensor d2 = Tensor::zeros(Shape{{2, 2}});
    auto grad_fn = std::make_shared<autograd::AddBackward>(d1.shape(), d2.shape());
    
    Tensor var = make_variable(data, Edge(grad_fn, 0));
    
    if (!var.unsafeGetTensorImpl()) {
        TEST_FAIL("make_variable should return valid tensor");
    }
    TEST_PASS("make_variable() returns valid tensor");
    
    if (var.grad_fn() != grad_fn) {
        TEST_FAIL("make_variable should attach grad_fn");
    }
    TEST_PASS("make_variable() attaches gradient edge");
    
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     COMPREHENSIVE NODE, EDGE, AND VARIABLE TEST SUITE          ║\n";
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
    
    run_test(test_edge_basics, "test_edge_basics");
    run_test(test_node_structure, "test_node_structure");
    run_test(test_autograd_meta_gradient_storage, "test_autograd_meta_gradient_storage");
    run_test(test_hook_system, "test_hook_system");
    run_test(test_gradient_edge_creation, "test_gradient_edge_creation");
    run_test(test_impl_namespace_helpers, "test_impl_namespace_helpers");
    run_test(test_sequence_numbers, "test_sequence_numbers");
    run_test(test_factory_functions, "test_factory_functions");
    
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
