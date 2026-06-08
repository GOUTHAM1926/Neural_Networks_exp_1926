/**
 * @file test_advanced_autograd.cpp
 * @brief Comprehensive test for Week 3-5 autograd features.
 * 
 * Tests:
 * 1. Node hook execution (pre/post hooks)
 * 2. GraphTask dependency tracking
 * 3. AnomalyMode toggle
 * 4. AutogradContext save/restore
 * 
 * ## Run command
 * ```bash
 * make run-snippet FILE=Tests/test_advanced_autograd.cpp
 * ```
 */

#include "core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/Functions.h"
#include "autograd/GraphTask.h"
#include "autograd/AnomalyMode.h"
#include "autograd/AutogradContext.h"
#include "autograd/SavedVariable.h"
#include <iostream>
#include <cassert>

using namespace OwnTensor;
using namespace OwnTensor::autograd;

#define TEST_PASS(msg) std::cout << "✓ " << msg << std::endl
#define TEST_FAIL(msg) std::cout << "✗ " << msg << std::endl; return false
#define TEST_SECTION(msg) std::cout << "\n=== " << msg << " ===" << std::endl

// ============================================================================
// TEST 1: Node Hook Execution
// ============================================================================
bool test_node_hooks() {
    TEST_SECTION("Test 1: Node Hook Execution");
    
    Tensor dummy_a = Tensor::zeros(Shape{{2, 2}});
    Tensor dummy_b = Tensor::zeros(Shape{{2, 2}});
    auto node = std::make_shared<AddBackward>(dummy_a.shape(), dummy_b.shape());
    
    // Track hook calls
    bool pre_hook_called = false;
    bool post_hook_called = false;
    
    // Register pre-hook
    node->register_pre_hook([&pre_hook_called](variable_list& inputs) {
        pre_hook_called = true;
        return inputs;  // Pass through
    });
    TEST_PASS("Pre-hook registered");
    
    // Register post-hook
    node->register_post_hook([&post_hook_called](const variable_list&, const variable_list&) {
        post_hook_called = true;
    });
    TEST_PASS("Post-hook registered");
    
    if (node->num_pre_hooks() != 1) {
        TEST_FAIL("Should have 1 pre-hook");
    }
    if (node->num_post_hooks() != 1) {
        TEST_FAIL("Should have 1 post-hook");
    }
    TEST_PASS("Hook counts correct");
    
    // Execute via operator()
    Tensor grad = Tensor::ones(Shape{{2, 2}}, TensorOptions());
    variable_list result = (*node)({grad});
    
    if (!pre_hook_called) {
        TEST_FAIL("Pre-hook should have been called");
    }
    TEST_PASS("Pre-hook executed");
    
    if (!post_hook_called) {
        TEST_FAIL("Post-hook should have been called");
    }
    TEST_PASS("Post-hook executed");
    
    if (result.size() != 2) {
        TEST_FAIL("AddBackward should return 2 gradients");
    }
    TEST_PASS("Backward function executed correctly");
    
    return true;
}

// ============================================================================
// TEST 2: GraphTask Dependency Tracking
// ============================================================================
bool test_graph_task() {
    TEST_SECTION("Test 2: GraphTask Dependency Tracking");
    
    GraphTask task(false, false);
    
    if (task.keep_graph_) {
        TEST_FAIL("keep_graph should be false");
    }
    TEST_PASS("GraphTask initialized with keep_graph=false");
    
    // Create some nodes
    Tensor dummy_a = Tensor::zeros(Shape{{2, 2}});
    Tensor dummy_b = Tensor::zeros(Shape{{2, 2}});
    auto node1 = std::make_shared<AddBackward>(dummy_a.shape(), dummy_b.shape());
    auto node2 = std::make_shared<MulBackward>(
        Tensor::ones(Shape{{2}}, TensorOptions()),
        Tensor::ones(Shape{{2}}, TensorOptions())
    );
    
    // Initialize dependencies
    task.init_to_execute(node1.get(), 1);
    task.init_to_execute(node2.get(), 2);
    
    if (!task.is_node_in_graph(node1.get())) {
        TEST_FAIL("node1 should be in graph");
    }
    TEST_PASS("Node added to graph");
    
    // Decrement dependencies
    bool ready = task.decrement_dependency(node2.get());
    if (ready) {
        TEST_FAIL("node2 should not be ready yet (1 dep remaining)");
    }
    TEST_PASS("Dependency decrement works (not ready)");
    
    ready = task.decrement_dependency(node2.get());
    if (!ready) {
        TEST_FAIL("node2 should be ready now (0 deps)");
    }
    TEST_PASS("Dependency decrement works (now ready)");
    
    // Test completion
    task.mark_completed();
    if (!task.completed_.load()) {
        TEST_FAIL("Task should be completed");
    }
    TEST_PASS("Task marked completed");
    
    return true;
}

// ============================================================================
// TEST 3: AnomalyMode
// ============================================================================
bool test_anomaly_mode() {
    TEST_SECTION("Test 3: AnomalyMode Toggle");
    
    // Should be disabled by default
    if (AnomalyMode::is_enabled()) {
        TEST_FAIL("AnomalyMode should be disabled by default");
    }
    TEST_PASS("AnomalyMode disabled by default");
    
    // Enable
    AnomalyMode::set_enabled(true);
    if (!AnomalyMode::is_enabled()) {
        TEST_FAIL("AnomalyMode should be enabled");
    }
    TEST_PASS("AnomalyMode enabled");
    
    // Test metadata
    AnomalyMetadata meta;
    meta.set_context("test context");
    if (meta.context() != "test context") {
        TEST_FAIL("Context should be 'test context'");
    }
    TEST_PASS("AnomalyMetadata context stored");
    
    // Disable
    AnomalyMode::set_enabled(false);
    if (AnomalyMode::is_enabled()) {
        TEST_FAIL("AnomalyMode should be disabled");
    }
    TEST_PASS("AnomalyMode disabled");
    
    return true;
}

// ============================================================================
// TEST 4: AutogradContext
// ============================================================================
bool test_autograd_context() {
    TEST_SECTION("Test 4: AutogradContext Save/Restore");
    
    AutogradContext ctx;
    
    // Create tensors to save
    TensorOptions opts = TensorOptions().with_req_grad(true);
    Tensor a = Tensor::ones(Shape{{2, 2}}, opts);
    Tensor b = Tensor::randn<float>(Shape{{2, 2}}, opts);
    
    // Save for backward
    ctx.save_for_backward({a, b});
    
    if (ctx.num_saved_variables() != 2) {
        TEST_FAIL("Should have 2 saved variables");
    }
    TEST_PASS("Variables saved correctly");
    
    // Retrieve
    variable_list saved = ctx.get_saved_variables();
    if (saved.size() != 2) {
        TEST_FAIL("Should retrieve 2 variables");
    }
    TEST_PASS("Variables retrieved correctly");
    
    // Mark dirty
    ctx.mark_dirty({a});
    if (!ctx.is_dirty(a)) {
        TEST_FAIL("a should be marked dirty");
    }
    if (ctx.is_dirty(b)) {
        TEST_FAIL("b should not be dirty");
    }
    TEST_PASS("Dirty tracking works");
    
    // Release
    ctx.release_variables();
    if (ctx.num_saved_variables() != 0) {
        TEST_FAIL("Variables should be released");
    }
    TEST_PASS("Variables released");
    
    return true;
}

// ============================================================================
// TEST 5: Integration with Existing Tests
// ============================================================================
bool test_regression() {
    TEST_SECTION("Test 5: Regression - Training Still Works");
    
    TensorOptions opts = TensorOptions().with_req_grad(true);
    
    // Simple forward
    Tensor a = Tensor::randn<float>(Shape{{3, 3}}, opts);
    Tensor b = Tensor::randn<float>(Shape{{3, 3}}, opts);
    
    // Verify requires_grad
    if (!a.requires_grad()) {
        TEST_FAIL("a should require grad");
    }
    TEST_PASS("Gradient tracking still works");
    
    // Verify node creation
    Tensor dummy_x = Tensor::zeros(Shape{{3, 3}}, opts);
    Tensor dummy_y = Tensor::zeros(Shape{{3, 3}}, opts);
    auto node = std::make_shared<AddBackward>(dummy_x.shape(), dummy_y.shape());
    if (std::string(node->name()) != "AddBackward") {
        TEST_FAIL("Node name incorrect");
    }
    TEST_PASS("Node creation still works");
    
    return true;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         ADVANCED AUTOGRAD FEATURES TEST SUITE                  ║\n";
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
    
    run_test(test_node_hooks, "test_node_hooks");
    run_test(test_graph_task, "test_graph_task");
    run_test(test_anomaly_mode, "test_anomaly_mode");
    run_test(test_autograd_context, "test_autograd_context");
    run_test(test_regression, "test_regression");
    
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
