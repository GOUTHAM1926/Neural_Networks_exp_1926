#pragma once
#include <iostream>
#include <exception>
#include <cuda_runtime.h>
#include <nccl.h>
#include "../../communication/include/ProcessGroupNCCL.h"
#include <variant>
#include <type_traits>
#include <sstream>
#include <fstream>



template <typename T>

struct canonicalized{
    using type = const T&;
};


template <size_t N>
struct canonicalized<const char[N]>{
    using type = const char*;
};

template <size_t N>
struct canonicalized<char[N]>{
    using type = const char*;
};


template <typename... Args>
struct _str{

    static auto call(const Args&... args){
        std::ostringstream stream;
        (stream << ... << args);
        return stream.str();
    }
};


template <typename... Args>
std::string str(const Args&... args){
    return _str<typename canonicalized<Args>::type...>::call(args...);
}

struct Location{
    const char* filename;
    const char* function;
    size_t line_number;

    Location(const char* filename, const char* function, size_t line_number){
        this->filename = filename;
        this->function = function;
        this->line_number = line_number;
    }
};

// void print_warning(std::string message){}

void cond_check_fail(const char* file_name,
                    size_t line_number,
                    const char* function_name,
                    const char* message);


class Error: public std::exception{
public:
    Error(const char* file_name,
                     size_t line_number,
                     const char* message);

    Error(Location location, const char* message);

    virtual const char* what() const noexcept override;
private:
    const char* message;

};



class Warning{
public:
    class DeprecatedWarning{};
    class UserWarning{};
    
    using warning_t = std::variant<UserWarning, DeprecatedWarning>;
    Warning(warning_t type, 
            Location location, 
            std::string message): location_(location), type_(type), message_(message.c_str()){}


    Warning(warning_t type, 
            Location location, 
            char* message): location_(location), type_(type), message_(message){}
    

    Location location() const { return location_; }
    const char* msg() const { return message_; }

private:
    Location location_;
    const char* message_ ;
    warning_t type_;

};

using UserWarning = Warning::UserWarning;
using DeprecatedWarning = Warning::DeprecatedWarning;

class Logger {
public:
    Logger() = default;
    ~Logger() { std::cerr << stream_.str() << std::endl; } // This makes it print!
    std::stringstream& stream() { return stream_; }
private:
    std::stringstream stream_;
};


struct location;
#define CUDA_CHECK(cmd)\
    do{\
        cudaError_t err = (cmd);\
        if( err != cudaSuccess ){\
            throw std::runtime_error(std::string("CUDA ERROR: ") + cudaGetErrorString(err));\
        }\
    }while(0)\

#define NCCL_CHECK(cmd)\
    do{\
        ncclResult_t err = (cmd);\
        if( err != ncclSuccess ){\
            throw std::runtime_error(std::string("CUDA ERROR: ") + cudaGetErrorString(err));\
        }\
    }while(0)\

#define PG_CHECK(cmd)\
    do{\
        result_t err = (cmd);\
        if( err != pgSuccess ){\
            throw std::runtime_error(std::string("PG ERROR: ") + pgGetError(err));\
        }\
    }while(0)\



#define ASSERT(condition , message)\
    if(!condition){\
        throw std::runtime_error(\
            message\
        );\
    }\

#define CONDITION_ASSERT_TRUE(condition, message)\
    if((condition)){\
        cond_check_fail(__FILE__ ,\
                         static_cast<size_t>(__LINE__),\
                         __FUNCTION__,\
                         ERROR_STRING_FULL(__FILE__, __LINE__, message));\
    }


#define CONDITION_ASSERT_TRUE_STATEMENT(condition, message, statement)\
    if((condition)){\
        cond_check_fail(__FILE__ ,\
                            static_cast<size_t>(__LINE__),\
                            __FUNCTION__,\
                            ERROR_STRING_FULL(__FILE__, __LINE__, message));\
    }\
    else{\
        statement;\
    }\


#define WARNING_STRING(...)\
    str(__VA_ARGS__)\

#define WARNING(warn_t, ...)\
    warn(Warning(\
        warn_t,\
        Location(__FILE__,\
             __FUNCTION__,\
             static_cast<size_t>(__LINE__)\
            ),\
        WARNING_STRING(__VA_ARGS__)\
    ))\


#define WARN_DEPRECATED(...)\
    WARNING(\
        DeprecatedWarning(),\
        __VA_ARGS__\
    )\

#define WARN(...)\
    WARNING(\
        UserWarning(),\
        __VA_ARGS__\
    )\

#define WARN_ONCE(...)

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define ERROR_STRING_FULL(file, line, message) "(" file ", " STR(line) "): " message


#define CONDITION_ASSERT(condition, message)\
    if(!(condition)){\
        cond_check_fail(__FILE__ ,\
                         static_cast<size_t>(__LINE__),\
                         __FUNCTION__,\
                         ERROR_STRING_FULL(__FILE__, __LINE__, message));\
    }\



#define LOG()\
    Logger().stream()\


void warn(const Warning& warning);


#define CONDITION_CHECK_FUNCTION(CONDITION, FUNC1, FUNC2)\
    ((CONDITION) ? (FUNC1) : (FUNC2))\


#define TIMER_LOG(Timer, stream, ...)\
    print_profiler_result(Timer,\
                            stream,\
                         __VA_ARGS__)\










