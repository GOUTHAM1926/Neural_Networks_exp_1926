#pragma once
#include "device/Device.h"
#include <cstddef>

namespace OwnTensor
{
    namespace device 
    {
        void set_memory(void* ptr, Device device, int value, size_t bytes);
    }
}
