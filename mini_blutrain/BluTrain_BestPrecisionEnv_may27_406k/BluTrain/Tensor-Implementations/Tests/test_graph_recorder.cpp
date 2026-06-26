#include <iostream>
#include <cassert>
#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/GraphRecorder.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

int main() {
    std::cout << "--- Test 1: Standard Autograd Graph (stdout) ---" << std::endl;
    {
        TensorOptions opts;
        opts.requires_grad = true;

        Tensor p = Tensor::ones(Shape{{3}}, opts);

        GraphRecordGuard guard(true);   // auto_print=true, output=stdout
        g_shape_debug = true;

        Tensor q = autograd::add(p, p);
        Tensor r = autograd::mul(q, p);
        Tensor s = autograd::relu(r);
        autograd::sum(s).backward();
    }

    std::cout << "\n--- Test 2: No Autograd (requires_grad=false) ---" << std::endl;
    {
        TensorOptions opts;
        opts.requires_grad = false;

        Tensor p = Tensor::ones(Shape{{3}}, opts);

        GraphRecordGuard guard(true);
        g_shape_debug = true;

        Tensor q = autograd::add(p, p);
        Tensor r = autograd::mul(q, p);
        Tensor s = autograd::relu(r);
    }

    std::cout << "\n--- Test 3: Embedding and Matrix Ops (file output) ---" << std::endl;
    {
        // Output goes to file only — nothing printed to terminal for this scope
        GraphRecordGuard guard(true, "graph_trace.txt");
        g_shape_debug = true;

        Tensor weights = Tensor::ones(Shape{{10, 5}}, TensorOptions().with_req_grad(true));
        Tensor indices = Tensor::ones(Shape{{1, 3}}, TensorOptions().with_dtype(Dtype::Int32));

        Tensor emb = autograd::embedding(weights, indices);
        Tensor x = Tensor::ones(Shape{{3, 5}}, TensorOptions().with_req_grad(true));
        Tensor y = autograd::matmul(emb.view(Shape{{3, 5}}), x.transpose(0, 1));

        autograd::sum(y).backward();
    }

    return 0;
}
