#include "profiler.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <filesystem>

// void Profiler::getDeviceProp(int rank){
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, rank);

//     std::cout<<"Device id: "<<rank<<std::endl;
//     std::cout<<"Device name: "<<prop.name<<std::endl;
//     std::cout<<"Multiprocessor count: "<<prop.multiProcessorCount<<std::endl;
//     std::cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<std::endl;
//     std::cout<<"Max threads per multiprocessor: "<<prop.maxThreadsPerMultiProcessor<<std::endl;
//     std::cout<<"Max blocks per grid: "<<prop.maxBlocksPerMultiProcessor<<std::endl;     
//     std::cout<<"Max Clock rate: "<<prop.clockRate<<std::endl;
//     std::cout<<"Can Execute Concurrent Kernels: "<<prop.concurrentKernels<<std::endl;
//     std::cout<<"PCI BUS ID: "<<prop.pciBusID<<std::endl;
//     std::cout<<"Total global memory: "<<prop.totalGlobalMem<<std::endl;
//     std::cout<<"Total shared memory per block: "<<prop.sharedMemPerBlock<<std::endl;
//     std::cout<<"PCI Device ID: "<<prop.pciDeviceID<<std::endl;
// }

void Profiler::event_start(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    // std::cout << "Event Created" << std::endl;
}

void Profiler::event_stop(cudaEvent_t start, cudaEvent_t stop, float& elapsed_time, cudaStream_t stream){
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
}

void Profiler::range_push(std::string title){
    nvtxRangePush(title.c_str());
}

void Profiler::nvtx_range_pop(){
    nvtxRangePop();
}

template <typename Func, typename... Args>
auto Profiler::nvtx_scoped_range(std::string title, Func&& op, Args... args){
    nvtx3::scoped_range r(title);
    return std::forward<Func>(op)(std::forward<Args>(args)...);
}



template <typename Func>
auto Profiler::nvtx_scoped_range(Func&& op){
    NVTX3_FUNC_RANGE();
    return op();
}

double latency(){
    return 0.0;
}

void mark_forward_start(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::forward_start) = get_time();
}   

void mark_forward_end(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::forward_end) = get_time();
}

void mark_backward_start(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::backward_start) = get_time();
}   

void mark_backward_end(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::backward_end) = get_time();
}

void mark_comm_start(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::comm_start) = get_time();
}   

void mark_comm_end(std::shared_ptr<Timer>& timer){
    timer->get_timer(Timer::Event::comm_stop) = get_time();
}

// void LOG_CSV::start_logging(const std::string& function_name){
// 	if(func_details.find(function_name) != func_details.end()){
// 		return;
// 	}

// 	Func_Timer timer;
// 	timer.func_begin_time = get_time();
// 	timer.function_name = function_name;
// 	func_details[function_name] = timer;
// }


static void ensure_parent_dir(const std::string& path){
    auto parent = std::filesystem::path(path).parent_path();
    if(!parent.empty()) std::filesystem::create_directories(parent);
}

LOG_CSV::LOG_CSV(const std::string& log_file_name){
        ensure_parent_dir(log_file_name);
		logger = std::ofstream(log_file_name);
		step_count_ = 0;
        rank_ = 0;
		logger << "function_name, start_time, end_time, total_time_taken, individual_time_taken, "
               << "tensors_alloc_individual, tensors_alloc_total, tensors_dealloc_individual, tensors_dealloc_total, "
               << "functions_called, current_depth" << std::endl;
}

LOG_CSV::LOG_CSV(const std::string& log_file_name, const std::string& trace_file_name, int rank){
        ensure_parent_dir(log_file_name);
        ensure_parent_dir(trace_file_name);
		logger = std::ofstream(log_file_name);
        trace = std::make_unique<TRACE_JSON>(trace_file_name);
		step_count_ = 0;
        rank_ = rank;

        // Initialize the allocation tracker alongside the profiler
        std::string alloc_csv = std::filesystem::path(log_file_name).parent_path().string() + "/allocations_rank" + std::to_string(rank) + ".csv";
        tracker_.init(alloc_csv);

		logger << "function_name, start_time, end_time, total_time_taken, individual_time_taken, "
               << "tensors_alloc_individual, tensors_alloc_total, tensors_dealloc_individual, tensors_dealloc_total, "
               << "functions_called, current_depth" << std::endl;
}

void LOG_CSV::start_logging(const std::string& function_name){
    if(func_details.find(function_name) != func_details.end()){
        return;
    }

    // If there is an active parent, register this function as its child
    if(!call_stack.empty()){
        auto it = func_details.find(call_stack.back());
        if(it != func_details.end()){
            it->second.called_functions.push_back(function_name);
        }
    }

    Func_Timer timer;
    timer.func_begin_time = get_time();
    timer.function_name = function_name;

    // Snapshot allocation counters at entry
    if(tracker_.is_initialized()){
        tracker_.snapshot_start(timer);
    }

    if(trace != nullptr){
        uint64_t ts = timer.func_begin_time / 1000; // ns → µs for Chrome tracing
        int depth = static_cast<int>(call_stack.size());
        std::string args = "{\"live_tensors\":" + std::to_string(tracker_.is_initialized() ? tracker_.live_count() : 0)
                         + ",\"mem_bytes\":" + std::to_string(tracker_.is_initialized() ? tracker_.current_bytes() : 0) + "}";
        trace->write_event(function_name, "ddp", "B", ts, rank_, depth, args);
    }

    func_details[function_name] = timer;

    // Push onto the call stack
    call_stack.push_back(function_name);
}

// void LOG_CSV::stop_logging(const std::string& function_name){
// 	if(func_details.find(function_name) == func_details.end()){
// 		throw Error(__FILE__, __LINE__, "No logging started to stop logging");
// 	}

// 	auto& timer = func_details[function_name];

// 	timer.func_end_time = get_time();
// 	timer.total_time_taken = timer.diff();

// 	add_to_stream(timer);

// 	func_details.erase(function_name);
// }

void LOG_CSV::stop_logging(const std::string& function_name){
    if(func_details.find(function_name) == func_details.end()){
        throw Error(__FILE__, __LINE__, "No logging started to stop logging");
    }

    auto& timer = func_details[function_name];
    timer.func_end_time = get_time();
    timer.total_time_taken = timer.diff();

    // Pop from call stack and propagate timing to parent
    Func_Timer* parent = nullptr;
    if(!call_stack.empty() && call_stack.back() == function_name){
        call_stack.pop_back();
        if(!call_stack.empty()){
            auto it = func_details.find(call_stack.back());
            if(it != func_details.end()){
                it->second.child_time_taken += timer.total_time_taken;
                parent = &it->second;
            }
        }
    }

    // Compute allocation stats and propagate to parent
    if(tracker_.is_initialized()){
        tracker_.snapshot_stop(timer, parent);
    }

    if(trace != nullptr){
        uint64_t ts = timer.func_end_time / 1000; // ns → µs
        int depth = static_cast<int>(call_stack.size());
        std::string args = "{\"alloc_individual\":" + std::to_string(timer.individual_allocs)
                         + ",\"alloc_total\":" + std::to_string(timer.total_allocs)
                         + ",\"dealloc_individual\":" + std::to_string(timer.individual_deallocs)
                         + ",\"dealloc_total\":" + std::to_string(timer.total_deallocs)
                         + ",\"live_tensors\":" + std::to_string(tracker_.is_initialized() ? tracker_.live_count() : 0)
                         + ",\"mem_bytes\":" + std::to_string(tracker_.is_initialized() ? tracker_.current_bytes() : 0) + "}";
        trace->write_event(function_name, "ddp", "E", ts, rank_, depth, args);
    }

    timer.individual_time_taken = timer.total_time_taken - timer.child_time_taken;
    add_to_stream(timer);
    func_details.erase(function_name);
}


// void LOG_CSV::add_to_stream(Func_Timer& func_timer){
// 	std::cout << func_timer.total_time_taken << std::endl;
// 	logger << func_timer.function_name << ", "
// 		<< func_timer.func_begin_time << ", " 
// 		<< func_timer.func_end_time << ", "
// 		<< func_timer.total_time_taken << ", (";

// 	if(func_timer.called_functions.size() == 0){
// 		std::copy(func_timer.called_functions.begin(), func_timer.called_functions.end(), std::ostream_iterator<std::string>(logger, "-> "));
// 	}
// 	logger << func_timer.function_name <<")" << std::endl;
// }

void LOG_CSV::add_to_stream(Func_Timer& func_timer){
    logger << func_timer.function_name << ", "
        << func_timer.func_begin_time << ", "
        << func_timer.func_end_time << ", "
        << func_timer.total_time_taken << ", "
        << func_timer.individual_time_taken << ", "
        << func_timer.individual_allocs << ", "
        << func_timer.total_allocs << ", "
        << func_timer.individual_deallocs << ", "
        << func_timer.total_deallocs << ", (";

    if(func_timer.called_functions.size() > 0){
        for(size_t i = 0; i < func_timer.called_functions.size(); ++i){
            logger << func_timer.called_functions[i];
            if(i < func_timer.called_functions.size() - 1){
                logger << " <- ";
            }
        }
        logger << " <- ";
    }
    logger << func_timer.function_name << "), "  << func_timer.called_functions.size() + 1 << std::endl;
}



