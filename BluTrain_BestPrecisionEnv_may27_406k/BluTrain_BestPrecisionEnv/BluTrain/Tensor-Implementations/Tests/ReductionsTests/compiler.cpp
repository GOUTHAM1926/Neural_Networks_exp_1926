#include "TensorLib.h"
#include <iostream>
using namespace OwnTensor;
using namespace std;

int main() {

    // Tensor t_cpu({{8}}, Dtype::Float32, DeviceIndex(Device::CPU));
    // std::vector<float> data1 = {50.0f,100.0f, -200.0f, 3000.0f , 400.0f, 0.0f, 600.0f,800.0f};
    // t_cpu.set_data(data1);   
    // t_cpu.display(std::cout,6);
    // //Tensor pred = reduce_sum(t_cpu) > 0.0; //Here we have to create a boolean tensor
    // Tensor t1 = Tensor::cond(
    //     false,
    //     [](const Tensor& t) { return square(t); },
    //     [](const Tensor& t) { return sqrt(t); },
    //     t_cpu.to_cuda()
    // );
    // t1.to_cpu().display(std::cout,6);

//     Tensor condition({{3, 2}}, Dtype::Bool, DeviceIndex(Device::CPU));
//     std::vector<bool> cond_data = {true, true, true, false, false, false};
//     condition.set_data(cond_data);
//    condition.display(std::cout, 4); 

    Tensor x({{3,2}}, Dtype::UInt32, DeviceIndex(Device::CPU));
    x.fill(-30);
    std::cout<<"Tensor X:"<<endl;
    x.display(std::cout,4);
    int y=3;
    // Tensor y({{3,1}}, Dtype::Int32, DeviceIndex(Device::CPU));
    // y.fill(0);
    // cout<<"Tensor Y:"<<endl;
    // y.display(std::cout,4);
   
    // Tensor z({{3,2}}, Dtype::Bool, DeviceIndex(Device::CPU));
    // z.fill(true);
    // cout<<"Tensor Z:"<<endl;
    // z.display(std::cout,4);
    
    Tensor res = pow(x,y,0);
    res.display(std::cout,4);
    // res.to_cpu().display(std::cout,4);

    // res = (x * y);
    // res.to_cpu().display(std::cout,4);  

    // res = (x != y);
    // res.to_cpu().display(std::cout,4);
    // Tensor result = Tensor::where(condition, x, y);
    // result.to_cpu().display(std::cout, 4);
    // Tensor t_cpu({{8}}, Dtype::Float32, DeviceIndex(Device::CPU));
    // std::vector<float> data1 = {50.0f, 100.0f, -200.0f, 3000.0f, 400.0f, 0.0f, 600.0f, 800.0f};
    // t_cpu.set_data(data1);   
    // t_cpu.display(std::cout, 6);

    // // ✅ CORRECT: Pass tensor as operand
    // Tensor t1 = Tensor::cond(
    //     true,
    //     [](const Tensor& t) { return square(t); },
    //     [](const Tensor& t) { return sqrt(t); },
    //     t_cpu  // Pass as operand
    // );
    // t1.display(std::cout, 6);

    // // ✅ CORRECT: Multiple operands
    // Tensor t2 = Tensor::cond(
    //     false,
    //     [](const Tensor& a, const Tensor& b) { return a + b; },
    //     [](const Tensor& a, const Tensor& b) { return a - b; },
    //     t_cpu, t_cpu
    // );

    // // ✅ CORRECT: Using tensor predicate
    // Tensor pred = reduce_sum(t_cpu) > 0.0f;
    // Tensor t3 = Tensor::cond(
    //     pred,
    //     [](const Tensor& t) { return exp(t); },
    //     [](const Tensor& t) { return log(t); },
    //     t_cpu
    // );

    // // ✅ CORRECT: Capturing lambda (no operands)
    // Tensor t4 = Tensor::cond(
    //     true,
    //     [&t_cpu]() { return square(t_cpu); },
    //     [&t_cpu]() { return sqrt(t_cpu); }
    //     // No operands - lambda captures t_cpu
    // );
    
    // return 0;
}

