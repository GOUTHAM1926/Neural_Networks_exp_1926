#include <iostream>
#include <vector>
#include <string>
#include "core/Tensor.h"
#include "autograd/Node.h"
#include "autograd/GraphArena.h"

using namespace OwnTensor;

class MockNode : public Node {
public:
    uint64_t data[16]; // 128 bytes per node
    MockNode() : Node() {}
    variable_list apply(variable_list&& grads) override { return std::move(grads); }
};

void run_heap(int num_nodes) {
    std::cout << "Running with Standard Heap..." << std::endl;
    Node::use_arena_ = false;
    std::vector<MockNode*> nodes;
    nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(new MockNode());
    }
    // Stay alive for a bit to let Massif capture the peak
    for (auto n : nodes) delete n;
}

void run_arena(int num_nodes) {
    std::cout << "Running with Graph Arena..." << std::endl;
    Node::use_arena_ = true;
    auto& arena = autograd::GraphArena::get_thread_local();
    std::vector<MockNode*> nodes;
    nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        nodes.push_back(new MockNode());
    }
    arena.reset();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [heap|arena] [num_nodes]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int num_nodes = (argc > 2) ? std::stoi(argv[2]) : 1000000;

    if (mode == "heap") {
        run_heap(num_nodes);
    } else {
        run_arena(num_nodes);
    }

    return 0;
}
