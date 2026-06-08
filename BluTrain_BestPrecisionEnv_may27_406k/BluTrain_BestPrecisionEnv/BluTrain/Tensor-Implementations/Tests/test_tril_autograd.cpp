#include "TensorLib.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace OwnTensor;

void test_tril_autograd() {
    std::cout << "Testing tril autograd..." << std::endl;
    Tensor x = Tensor::ones(Shape{{3, 3}}, TensorOptions().with_req_grad(true));
    Tensor y = autograd::tril(x, 0);
    
    // y should be:
    // 1 0 0
    // 1 1 0
    // 1 1 1
    
    // Tensor grad_out = Tensor::ones(Shape{{3, 3}});
    y.backward(&y);
    
    Tensor grad_x = x.grad_view();
    // grad_x should be tril(1, 0):
    // 1 0 0
    // 1 1 0
    // 1 1 1
    
    std::cout << "grad_x:" << std::endl;
    grad_x.display();
    

}

int main() {
    try {
        test_tril_autograd();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
