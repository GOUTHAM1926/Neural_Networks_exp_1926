#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;

int main() {

    // Tensor t_cpu({{8}}, Dtype::Float32, DeviceIndex(Device::CPU));
    // std::vector<float> data1 = {50.0f,100.0f, -200.0f, 3000.0f , 400.0f, 0.0f, 600.0f,800.0f};
    // t_cpu.set_data(data1);   
    // t_cpu.to_cpu().display(std::cout,6);
    // //Tensor pred = reduce_sum(t_cpu) > 0.0; //Here we have to create a boolean tensor
    // Tensor t1 = Tensor::cond(
    //     false,
    //     [](const Tensor& t) { return square(t); },
    //     [](const Tensor& t) { return sqrt(t); },
    //     t_cpu.to_CPU()
    // );
    // t1.to_cpu().display(std::cout,6);

//     Tensor condition({{3, 2}}, Dtype::Bool, DeviceIndex(Device::CPU));
//     std::vector<bool> cond_data = {true, true, true, false, false, false};
//     condition.set_data(cond_data);
//    condition.to_cpu().display(std::cout, 4); 

//     Tensor x({{1,2}}, Dtype::Int16, DeviceIndex(Device::CPU));
//     //x.fill(100.0f);
//     x.set_data({int16_t(150),int16_t(400)});
//     cout<<"Tensor X:"<<endl;
//     x.to_cpu().display(std::cout,4);
//     _Float32 a =2.0;
//    Tensor y =Tensor::where(x>0,a,0);
// x.display(std::cout,4);
//}
//     Tensor y({{1,2}}, Dtype::Bool, DeviceIndex(Device::CUDA));
//     y.set_data({bool(300),bool(0)});
//     //y.fill(300);
//     cout<<"Tensor Y:"<<endl;
//     y.to_cpu().display(std::cout,4);
// //     Tensor res = x+ y;
//     //cout<<"Result of Logical AND:"<<endl; 
//     res.to_cpu().display(std::cout,4);

//     res = x- y;
//     //cout<<"Result of Logical AND:"<<endl; 
//     res.to_cpu().display(std::cout,4);

//     res = x*y;
//     //cout<<"Result of Logical AND:"<<endl; 
//     res.to_cpu().display(std::cout,4);

//     res = x/ y;
//     //cout<<"Result of Logical AND:"<<endl; 
//     res.to_cpu().display(std::cout,4);

//     cout<<"Result of += :"<<endl; 
//     x+=y;
//     x.to_cpu().display(std::cout,4);
//      x-=y;
//     x.to_cpu().display(std::cout,4);
//      x*=y;
//     x.to_cpu().display(std::cout,4);
//      x/=y;
//     x.to_cpu().display(std::cout,4);

//     // res = (x * y);
//     // res.to_cpu().display(std::cout,4);  

//     // res = (x != y);
//     // res.to_cpu().display(std::cout,4);
//     // Tensor result = Tensor::where(condition, x, y);
//     // result.to_cpu().display(std::cout, 4);
//     // Tensor t_cpu({{8}}, Dtype::Float32, DeviceIndex(Device::CPU));
//     // std::vector<float> data1 = {50.0f, 100.0f, -200.0f, 3000.0f, 400.0f, 0.0f, 600.0f, 800.0f};
//     // t_cpu.set_data(data1);   
//     // t_cpu.to_cpu().display(std::cout, 6);

//     // // ✅ CORRECT: Pass tensor as operand
//     // Tensor t1 = Tensor::cond(
//     //     true,
//     //     [](const Tensor& t) { return square(t); },
//     //     [](const Tensor& t) { return sqrt(t); },
//     //     t_cpu  // Pass as operand
//     // );
//     // t1.to_cpu().display(std::cout, 6);

//     // // ✅ CORRECT: Multiple operands
//     // Tensor t2 = Tensor::cond(
//     //     false,
//     //     [](const Tensor& a, const Tensor& b) { return a + b; },
//     //     [](const Tensor& a, const Tensor& b) { return a - b; },
//     //     t_cpu, t_cpu
//     // );

//     // // ✅ CORRECT: Using tensor predicate
//     // Tensor pred = reduce_sum(t_cpu) > 0.0f;
//     // Tensor t3 = Tensor::cond(
//     //     pred,
//     //     [](const Tensor& t) { return exp(t); },
//     //     [](const Tensor& t) { return log(t); },
//     //     t_cpu
//     // );

//     // // ✅ CORRECT: Capturing lambda (no operands)
//     // Tensor t4 = Tensor::cond(
//     //     true,
//     //     [&t_cpu]() { return square(t_cpu); },
//     //     [&t_cpu]() { return sqrt(t_cpu); }
//     //     // No operands - lambda captures t_cpu
//     // );
    
//     // return 0;
// }
Tensor x({{3,2,3}},Dtype::Float32,Device::CPU);
x.set_data<float>({1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0});
}
