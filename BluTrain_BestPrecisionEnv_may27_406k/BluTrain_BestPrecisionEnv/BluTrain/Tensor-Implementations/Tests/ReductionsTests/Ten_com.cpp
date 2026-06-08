#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;

int main() {
    Tensor x({{3,2}}, Dtype::Int16, DeviceIndex(Device::CUDA));
    //x.fill(100.0f);g
    x.set_data({int16_t(150),int16_t(200) , int16_t(250), int16_t(300), int16_t(350), int16_t(400)});
    //cout<<"Tensor X:"<<endl;
    x.to_cpu().display(std::cout,4);
    Tensor y({{1,2}}, Dtype::Bool, DeviceIndex(Device::CUDA));
    y.set_data({bool(300),bool(1)});
    //y.fill(300);
    //cout<<"Tensor Y:"<<endl;
    y.to_cpu().display(std::cout,4);

    //Comparison
    Tensor res = x +y;
    res.to_cpu().display(std::cout,4);

    res = x -y;
    res.to_cpu().display(std::cout,4);
    
    res = x *y;
    res.to_cpu().display(std::cout,4);
    
    res = x <=y;
    res.to_cpu().display(std::cout,4);

    res = x >y;
    res.to_cpu().display(std::cout,4);

    res = x <y;
    res.to_cpu().display(std::cout,4);

    return 0;
}
