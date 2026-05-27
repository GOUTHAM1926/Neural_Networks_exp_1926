#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
int main() {
    float naive_sum = 0.0f;
    float kahan_sum = 0.0f;
    float kahan_c = 0.0f;
    
    float val = 0.000001f; // A tiny value
    int iterations = 10000000; // 10 Million iterations
    
    // Naive Summation
    std::chrono::high_resolution_clock::time_point start_1 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        naive_sum += val;
    }
    std::chrono::high_resolution_clock::time_point end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_1 = end_1 - start_1;
    std::cout<<"Naive Sum Time: "<<elapsed_1.count()<<std::endl;
    
    // Kahan Summation
    std::chrono::high_resolution_clock::time_point start_2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iterations; ++i) {
        float y = val - kahan_c;
        float t = kahan_sum + y;
        kahan_c = (t - kahan_sum) - y;
        kahan_sum = t;
    }
    std::chrono::high_resolution_clock::time_point end_2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_2 = end_2 - start_2;
    std::cout<<"Kahan Sum Time: "<<elapsed_2.count()<<std::endl;
    
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Expected:   " << 10.0000000000 << std::endl;
    std::cout << "Naive Sum:  " << (double)naive_sum  << std::endl;
    std::cout << "Kahan Sum:  " << (double)kahan_sum  << std::endl;

    float t = 0.0000001f;
    float err = 0.0f ;
    float sum=0.0f;
    double acc = 0.0f ;
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<100000000;i++){
        acc+=t;
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout<<"Naive Sum:  "<<acc<<" Time: "<<elapsed.count()<<std::endl;
    start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<100000000;i++){
        float c=t-err;
        float sum_1 = sum+c ;
        float c_1 = (sum_1-sum);
        float err_1 = c_1 - c ;
        sum=sum_1;
        err=err_1;
    }
    end = std::chrono::high_resolution_clock::now();
    auto elapsed_3 = std::chrono::duration<double>(end-start);
    std::cout<< std::fixed << std::setprecision(10) ;
    std::cout<<"Expected:   "<<10.0000000000<<std::endl;
    std::cout<<"Naive Sum:  "<<acc  <<std::endl;
    std::cout<<"Kahan Sum:  "<<(double)sum  <<std::endl;
    std::cout<<"Kahan Sum Time: "<<elapsed_3.count()<<std::endl;
    
}