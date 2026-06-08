#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>

using namespace OwnTensor;

namespace own_autograd {

/**
 * @brief Numerical gradient checking utility.
 * 
 * Compares analytical gradients (computed via backward) with numerical gradients
 * computed using central differences.
 */
class GradCheck {
public:
    struct Config {
        double eps;
        double tol;
        bool verbose;

        Config() : eps(1e-3), tol(1e-3), verbose(true) {}
    };

    /**
     * @brief Performs gradient check on a function.
     * 
     * @param func The function to test: Tensor(const std::vector<Tensor>&)
     * @param inputs The input tensors (must have req_grad=true)
     * @param config Configuration for the check
     * @return true if all gradients match within tolerance, false otherwise.
     */
    static bool check(
        std::function<OwnTensor::Tensor(const std::vector<OwnTensor::Tensor>&)> func,
        const std::vector<OwnTensor::Tensor>& inputs,
        Config config = Config()
    ) {
        if (config.verbose) {
            std::cout << "\n[GradCheck] Starting gradient check..." << std::endl;
        }

        // 1. Compute analytical gradients
        std::vector<Tensor> test_inputs;
        for (auto in : inputs) {
            if (!in.requires_grad()) {
                std::cerr << "[GradCheck] Error: Input tensor does not require grad." << std::endl;
                return false;
            }
            test_inputs.push_back(in);
            in.zero_grad();
        }

        Tensor output = func(test_inputs);
        if (output.numel() > 1) {
            output = OwnTensor::autograd::sum(output);
        }
        output.backward();

        bool all_passed = true;

        // 2. Compute numerical gradients for each input
        for (size_t i = 0; i < test_inputs.size(); ++i) {
            Tensor& in = test_inputs[i];
            int64_t numel = in.numel();
            
            if (!in.device().is_cpu()) {
                std::cerr << "[GradCheck] Error: Currently only supports CPU tensors for simplicity." << std::endl;
                return false;
            }

            float* analytical_grad_ptr = in.grad<float>();

            for (int64_t j = 0; j < numel; ++j) {
                float original_val = in.data<float>()[j];

                // f(x + eps)
                in.data<float>()[j] = original_val + static_cast<float>(config.eps);
                Tensor out_plus = func(test_inputs);
                if (out_plus.numel() > 1) out_plus = OwnTensor::autograd::sum(out_plus);
                double val_plus = static_cast<double>(out_plus.data<float>()[0]);

                // f(x - eps)
                in.data<float>()[j] = original_val - static_cast<float>(config.eps);
                Tensor out_minus = func(test_inputs);
                if (out_minus.numel() > 1) out_minus = OwnTensor::autograd::sum(out_minus);
                double val_minus = static_cast<double>(out_minus.data<float>()[0]);

                // Reset
                in.data<float>()[j] = original_val;

                double numerical_grad = (val_plus - val_minus) / (2.0 * config.eps);

                // Compare
                double a_grad = static_cast<double>(analytical_grad_ptr[j]);
                double diff = std::abs(a_grad - numerical_grad);
                double max_grad = std::max(std::abs(a_grad), std::abs(numerical_grad));
                
                bool passed = (diff < config.tol) || (diff / std::max(1.0, max_grad) < config.tol);

                if (!passed) {
                    all_passed = false;
                    if (config.verbose) {
                        std::cout << "[GradCheck] FAIL at input[" << i << "], element[" << j << "]" << std::endl;
                        std::cout << "  Analytical: " << std::fixed << std::setprecision(6) << a_grad << std::endl;
                        std::cout << "  Numerical:  " << std::fixed << std::setprecision(6) << numerical_grad << std::endl;
                        std::cout << "  Difference: " << std::scientific << diff << std::endl;
                    }
                }
            }
        }

        if (config.verbose) {
            if (all_passed) {
                std::cout << "[GradCheck] PASS" << std::endl;
            } else {
                std::cout << "[GradCheck] FAIL" << std::endl;
            }
        }

        return all_passed;
    }
};

} // namespace own_autograd

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   Comprehensive GradCheck Test Suite   " << std::endl;
    std::cout << "========================================" << std::endl;

    TensorOptions req_grad = TensorOptions().with_req_grad(true);
    own_autograd::GradCheck::Config config;
    config.eps = 1e-3;
    config.tol = 1e-3;

    int total = 0;
    int passed = 0;

    auto run_test = [&](const std::string& name, 
                        std::function<Tensor(const std::vector<Tensor>&)> func, 
                        const std::vector<Tensor>& inputs) {
        total++;
        std::cout << "\nTest " << total << ": " << name << std::endl;
        if (own_autograd::GradCheck::check(func, inputs, config)) {
            passed++;
            std::cout << "[RESULT] PASS" << std::endl;
        } else {
            std::cout << "[RESULT] FAIL" << std::endl;
        }
    };

    // 1. Binary Ops
    run_test("Add (a + b)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::add(in[0], in[1]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 2.0f), Tensor::full(Shape{{2, 2}}, req_grad, 3.0f)});

    run_test("Sub (a - b)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::sub(in[0], in[1]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 5.0f), Tensor::full(Shape{{2, 2}}, req_grad, 2.0f)});

    run_test("Mul (a * b)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::mul(in[0], in[1]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 2.0f), Tensor::full(Shape{{2, 2}}, req_grad, 4.0f)});

    run_test("Div (a / b)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::div(in[0], in[1]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 10.0f), Tensor::full(Shape{{2, 2}}, req_grad, 2.0f)});

    // 2. Unary / Arithmetic Ops
    run_test("Square (x^2)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::square(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 3.0f)});

    run_test("Sqrt (sqrt(x))", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::sqrt(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 4.0f)});

    run_test("Exp (exp(x))", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::exp(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 1.0f)});

    run_test("Log (log(x))", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::log(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 2.0f)});

    run_test("Pow (x^y)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::pow(in[0], 3.0f);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 2.0f)});

    // 3. Matrix Ops
    run_test("Matmul (A @ B)", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::matmul(in[0], in[1]);
    }, {Tensor::full(Shape{{2, 3}}, req_grad, 1.0f), Tensor::full(Shape{{3, 2}}, req_grad, 2.0f)});

    // 4. Reductions
    run_test("Sum", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::sum(in[0]);
    }, {Tensor::full(Shape{{3, 3}}, req_grad, 1.0f)});

    run_test("Mean", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::mean(in[0]);
    }, {Tensor::full(Shape{{3, 3}}, req_grad, 1.0f)});

    // 5. Activations
    run_test("ReLU", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::relu(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 2.0f)});

    run_test("Sigmoid", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::sigmoid(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 0.0f)});

    run_test("Tanh", [](const std::vector<Tensor>& in) -> Tensor {
        return OwnTensor::autograd::tanh(in[0]);
    }, {Tensor::full(Shape{{2, 2}}, req_grad, 0.5f)});

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "   GRADCHECK SUMMARY: " << passed << "/" << total << " PASSED" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
