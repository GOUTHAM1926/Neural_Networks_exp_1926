#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <string>
#include "autograd/CppNode.h"
#include "autograd/AutogradContext.h"
#include "autograd/AutogradOps.h"
#include "core/Tensor.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

bool near(float a, float b, float tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// -----------------------------------------------------------------------------
// Metadata Test
// -----------------------------------------------------------------------------
class MetaNode : public CppNode<MetaNode> {
public:
    const char* name() const override { return "MetaNode"; }
    static variable_list forward(AutogradContext* ctx, const Tensor& x) {
        return {x.clone()};
    }
    static variable_list backward(AutogradContext* ctx, const variable_list& grad_outputs) {
        return grad_outputs;
    }
};

void test_metadata() {
    std::cout << "Testing CppNode Metadata..." << std::endl;
    Tensor x = Tensor::ones(Shape{{1}}, TensorOptions().with_req_grad(true));
    auto y = MetaNode::apply(x)[0];
    
    auto grad_fn = y.grad_fn();
    assert(grad_fn != nullptr);
    assert(std::string(grad_fn->name()) == "MetaNode");
    
    uint64_t seq = grad_fn->sequence_nr();
    std::cout << "  Node Name: " << grad_fn->name() << std::endl;
    std::cout << "  Sequence NR: " << seq << std::endl;
    std::cout << "PASS: Metadata" << std::endl;
}

// -----------------------------------------------------------------------------
// Multi-I/O Test
// y1 = a + b, y2 = a * b
// dy1/da = 1, dy1/db = 1
// dy2/da = b, dy2/db = a
// -----------------------------------------------------------------------------
class MultiIONode : public CppNode<MultiIONode> {
public:
    static variable_list forward(AutogradContext* ctx, const Tensor& a, const Tensor& b) {
        ctx->save_for_backward({a, b});
        return {autograd::add(a, b), autograd::mul(a, b)};
    }
    
    static variable_list backward(AutogradContext* ctx, const variable_list& grad_outputs) {
        auto saved = ctx->get_saved_variables();
        Tensor a = saved[0];
        Tensor b = saved[1];
        
        Tensor g1 = grad_outputs[0]; // from y1
        Tensor g2 = grad_outputs[1]; // from y2
        
        // ga = g1 * 1 + g2 * b
        Tensor ga = autograd::add(g1, autograd::mul(g2, b));
        // gb = g1 * 1 + g2 * a
        Tensor gb = autograd::add(g1, autograd::mul(g2, a));
        
        return {ga, gb};
    }
};

void test_multi_io() {
    std::cout << "Testing Multi-I/O CppNode..." << std::endl;
    Tensor a = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 2.0f);
    Tensor b = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 3.0f);
    
    auto results = MultiIONode::apply(a, b);
    Tensor y1 = results[0];
    Tensor y2 = results[1];
    
    assert(near(y1.data<float>()[0], 5.0f));
    assert(near(y2.data<float>()[0], 6.0f));
    
    // Compute grad = y1 + y2
    // gy1 = 1, gy2 = 1
    // ga = 1 + 3 = 4
    // gb = 1 + 2 = 3
    Tensor sum = autograd::add(y1, y2);
    sum.backward();
    
    assert(near(a.grad<float>()[0], 4.0f));
    assert(near(b.grad<float>()[0], 3.0f));
    
    std::cout << "PASS: Multi-I/O" << std::endl;
}

// -----------------------------------------------------------------------------
// Variadic Args Test (Mixed Types)
// -----------------------------------------------------------------------------
class VariadicNode : public CppNode<VariadicNode> {
public:
    static variable_list forward(AutogradContext* ctx, const Tensor& x, float scalar, int power) {
        ctx->save_for_backward({x});
        // result = scalar * x^power
        Tensor res = x;
        for(int i=1; i<power; ++i) res = autograd::mul(res, x);
        return {autograd::mul(Tensor::full(x.shape(), x.opts(), scalar), res)};
    }
    
    static variable_list backward(AutogradContext* ctx, const variable_list& grad_outputs) {
        // We only return grad for tensors. 
        // In our CppNode::apply, we connected only tensors.
        // So backward should return one grad for 'x'.
        auto saved = ctx->get_saved_variables();
        Tensor x = saved[0];
        // Simplified backward for test
        return {grad_outputs[0]}; 
    }
};

void test_variadic_args() {
    std::cout << "Testing Variadic Args (Mixed Types)..." << std::endl;
    Tensor x = Tensor::full(Shape{{1}}, TensorOptions().with_req_grad(true), 2.0f);
    
    // forward(x, 0.5f, 3) -> 0.5 * 2^3 = 4.0
    auto results = VariadicNode::apply(x, 0.5f, 3);
    assert(near(results[0].data<float>()[0], 4.0f));
    
    // Verify that index in connect_arg didn't explode or misalign
    // The node should have 1 edge to x (the only Tensor)
    auto grad_fn = results[0].grad_fn();
    assert(grad_fn->num_inputs() == 3); // Based on my fold expression implementation
    // Input 0: x (Tensor, requires_grad=true) -> valid edge
    // Input 1: float (not Tensor) -> invalid edge/placeholder
    // Input 2: int (not Tensor) -> invalid edge/placeholder
    
    assert(grad_fn->next_edge(0).is_valid());
    assert(!grad_fn->next_edge(1).is_valid());
    assert(!grad_fn->next_edge(2).is_valid());
    
    std::cout << "PASS: Variadic Args" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Running CppNode Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_metadata();
    test_multi_io();
    test_variadic_args();
    
    std::cout << "========================================" << std::endl;
    std::cout << "All CppNode unit tests PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
