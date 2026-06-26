#include "Error_logs.hpp"
#include <iostream>




Error::Error(const char* filename, size_t line_number, const char* message):
    message(std::move(message)){}

Error::Error(Location location, const char* message):
    Error(location.filename, location.line_number, message){}

const char* Error::what() const noexcept {
    return message;
}

void cond_check_fail(const char* file_name,
                    size_t line_number,
                    const char* function_name,
                    const char* message){

    throw Error(file_name, line_number, message);
    
}


void warn(const Warning& warning){
    std::stringstream ss;

    LOG() << "Warning At: (" << warning.location().filename << ", " 
                            << warning.location().function << ", "
                            << warning.location().line_number << "), Warning: "
                            << warning.msg();
    
}  





///////////////////////////////////////////////////////////
//                PROFILER PRINTER                      //
/////////////////////////////////////////////////////////

