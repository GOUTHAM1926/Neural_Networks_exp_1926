#include "TensorLib.h"
using namespace OwnTensor;
using namespace std;
int main(){
    Tensor x({{1,2}}, Dtype::Int16, DeviceIndex(Device::CUDA));
    //x.fill(100.0f);
    x.set_data({int16_t(150),int16_t(400)});
    cout<<"Tensor X:"<<endl;
    x.to_cpu().display(std::cout,4);
    Tensor y({{1,2}}, Dtype::Bool, DeviceIndex(Device::CUDA));
    y.set_data({bool(300),bool(0)});
    //y.fill(300);
    cout<<"Tensor Y:"<<endl;
    y.to_cpu().display(std::cout,4);
    Tensor res = (x+3);
    //cout<<"Result of Logical AND:"<<endl; 
    res.to_cpu().display(std::cout,4);

    //res = 3-y;
    // //cout<<"Result of Logical AND:"<<endl; 
     //res.to_cpu().display(std::cout,4);

    // res = x*3;
    // //cout<<"Result of Logical AND:"<<endl; 
    // res.to_cpu().display(std::cout,4);

    res = x/3;
    //cout<<"Result of Logical AND:"<<endl; 
    res.to_cpu().display(std::cout,4);

    // cout<<"Result of += :"<<endl; 
    // x+=3;
    // x.to_cpu().display(std::cout,4);
    //   y-=3;
    // y.to_cpu().display(std::cout,4);
    //  x*=3;
    // x.to_cpu().display(std::cout,4);
     x/=3;
    x.to_cpu().display(std::cout,4);

}
