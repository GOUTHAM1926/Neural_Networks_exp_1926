#pragma once

#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <chrono>
// Guard each NVTX header individually — availability varies by toolkit/OS.
// nvToolsExtOpenCL.h requires the OpenCL SDK (<CL/cl.h>) which is rarely present.
#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
#else
  // Stub out the C NVTX API so call sites compile without the NVTX SDK.
  inline void nvtxRangePush(const char*) {}
  inline void nvtxRangePop() {}
#endif
#if __has_include(<nvtx3/nvToolsExtCuda.h>)
  #include <nvtx3/nvToolsExtCuda.h>
#endif
#if __has_include(<nvtx3/nvToolsExtCudaRt.h>)
  #include <nvtx3/nvToolsExtCudaRt.h>
#endif
// OpenCL extension requires <CL/cl.h> — skip unless both headers are present.
#if __has_include(<nvtx3/nvToolsExtOpenCL.h>) && __has_include(<CL/cl.h>)
  #include <nvtx3/nvToolsExtOpenCL.h>
#endif
#if __has_include(<nvtx3/nvToolsExtSync.h>)
  #include <nvtx3/nvToolsExtSync.h>
#endif

// C++ NVTX3 wrapper — may be absent on older Nsight toolkits; stub if missing.
#if __has_include(<nvtx3/nvtx3.hpp>)
  #include <nvtx3/nvtx3.hpp>
  #define HAVE_NVTX3_CPP 1
#else
  #define HAVE_NVTX3_CPP 0
  namespace nvtx3 {
    using registered_string = std::string;
    struct scoped_range {
      explicit scoped_range(const std::string&) {}
    };
  }
  #ifndef NVTX3_FUNC_RANGE
    #define NVTX3_FUNC_RANGE() do {} while(0)
  #endif
#endif




#include "Error_logs.hpp"
#include "timer.hpp"
#include "device/AllocationTracker.h"
#include <cstdlib>

// bandwidth (includes the bandwidth utilized in the peak)
// loss (utilized vs available)
// peak (includes the peak bandwidth)
// device prop
// latency
// accuracy
// amount of memory
// 

class Profiler{
public:
    Profiler() = default;
    ~Profiler() = default;

    Profiler(size_t world_size);

    void getDeviceProp(int rank);
    void event_start(cudaEvent_t start, cudaEvent_t stop, cudaStream_t stream = 0);
    void event_stop(cudaEvent_t start, cudaEvent_t stop, float& elapsed_time, cudaStream_t stream = 0);

    double throughput_communication();
    double throughput_computation();

    double latency();
    void time_taken();

    // below two operations wont just push the operations. rather they also start with measuring the performance too.
    // like the bandwidth, latency, time taken, 
    // void 
    //nvtx ranges:
    void range_push(std::string title); 
    void nvtx_range_pop();

    void forward_range_start();
    
    void forward_range_stop();

    template <typename Func, typename... Args>
    auto nvtx_scoped_range(std::string title, Func&& op, Args... args); //overload for scoped range
    template <typename Func>
    auto nvtx_scoped_range(nvtx3::registered_string title, Func&& op){
        nvtx3::scoped_range r(title);
        return op();
    }
    template <typename Func>
    auto nvtx_scoped_range(Func&& op); //overload for macro
    void arithmetic_intensity();

};


// ============================================================
// ProfilerTracker: bridges AllocationTracker with the profiler
// Provides per-function allocation snapshots for LOG_CSV
// ============================================================

class ProfilerTracker {
public:
    ProfilerTracker() = default;

    // Move-only: prevent the temporary from shutting down the singleton
    ProfilerTracker(const ProfilerTracker&) = delete;
    ProfilerTracker& operator=(const ProfilerTracker&) = delete;
    ProfilerTracker(ProfilerTracker&& other) noexcept : initialized_(other.initialized_) {
        other.initialized_ = false;  // source no longer owns shutdown responsibility
    }
    ProfilerTracker& operator=(ProfilerTracker&& other) noexcept {
        if (this != &other) {
            shutdown();
            initialized_ = other.initialized_;
            other.initialized_ = false;
        }
        return *this;
    }

    ~ProfilerTracker() { shutdown(); }

    void init(const std::string& alloc_csv_path = "logs/allocations.csv") {
        OwnTensor::AllocationTracker::instance().init(alloc_csv_path.c_str());
        initialized_ = true;
    }

    void shutdown() {
        if (initialized_) {
            OwnTensor::AllocationTracker::instance().shutdown();
            initialized_ = false;
        }
    }

    // Snapshot current global alloc/dealloc counts
    uint64_t alloc_count() const {
        return OwnTensor::AllocationTracker::instance().get_total_allocations();
    }

    uint64_t dealloc_count() const {
        return OwnTensor::AllocationTracker::instance().get_total_deallocations();
    }

    size_t live_count() const {
        return alloc_count() - dealloc_count();
    }

    size_t current_bytes(int device = -1) const {
        return OwnTensor::AllocationTracker::instance().get_current_allocated(device);
    }

    size_t peak_bytes(int device = -1) const {
        return OwnTensor::AllocationTracker::instance().get_peak_allocated(device);
    }

    // Populate Func_Timer entry-snapshot (call at start_logging)
    void snapshot_start(Func_Timer& ft) const {
        ft.alloc_snapshot_start = alloc_count();
        ft.dealloc_snapshot_start = dealloc_count();
    }

    // Compute totals and propagate to parent (call at stop_logging)
    void snapshot_stop(Func_Timer& ft, Func_Timer* parent) const {
        ft.total_allocs   = alloc_count()   - ft.alloc_snapshot_start;
        ft.total_deallocs = dealloc_count() - ft.dealloc_snapshot_start;
        ft.individual_allocs   = ft.total_allocs   - ft.child_allocs;
        ft.individual_deallocs = ft.total_deallocs - ft.child_deallocs;

        // Propagate to parent's child accumulators
        if (parent) {
            parent->child_allocs   += ft.total_allocs;
            parent->child_deallocs += ft.total_deallocs;
        }
    }

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
};


void mark_forward_start(std::shared_ptr<Timer>& timer);

void mark_forward_end(std::shared_ptr<Timer>& timer);

void mark_backward_start(std::shared_ptr<Timer>& timer);  

void mark_backward_end(std::shared_ptr<Timer>& timer);

void mark_comm_start(std::shared_ptr<Timer>& timer);  

void mark_comm_end(std::shared_ptr<Timer>& timer);


//all the results this should store...
/*
 * Time taken by the function alone
 * Function called by it and the function called it
 * number of times it is called
 * variation in the performance (avg time)
 * number of tensors getting created in each instance.
 *
 *
 *
 */
     
class TRACE_JSON;
class LOG_CSV{
public:
	LOG_CSV() = default;
    LOG_CSV(LOG_CSV&&) = default;
    LOG_CSV& operator=(LOG_CSV&&) = default;
	LOG_CSV(const std::string& log_file_name);
    LOG_CSV(const std::string& log_file_name, const std::string& trace_file_name, int rank = 0);
	void start_logging(const std::string &function_name);
	void stop_logging(const std::string &function_name);
	void add_to_stream(Func_Timer& func_timer);
    size_t current_depth() const { return call_stack.size(); }
    int  break_code(){ exit(EXIT_SUCCESS); }
    void increment_step_count(){ step_count_ += 1; }

    ProfilerTracker& tracker() { return tracker_; }

private:
	int step_count_;
    int rank_ = 0;
	std::ofstream logger;
	std::unordered_map<std::string, Func_Timer> func_details;
    std::vector<std::string> call_stack;
    std::unique_ptr<TRACE_JSON> trace;
    ProfilerTracker tracker_;
};

class ScopedLog {
public:
    ScopedLog(LOG_CSV& csv, const std::string& name)
        : csv_(csv), name_(name) {
        csv_.start_logging(name_);
    }
    ~ScopedLog() {
        csv_.stop_logging(name_);
    }
private:
    LOG_CSV& csv_;
    std::string name_;
};

class TRACE_JSON {
public:
   TRACE_JSON(const std::string& filename) {
       file.open(filename);
       file << "{ \"traceEvents\": [\n";
       first_event = true;
   }

   ~TRACE_JSON() {
       file << "\n]}";
       file.close();
   }

   // Legacy overload (B/E phase events without args)
   void write_event(const std::string& name, const std::string& phase, uint64_t ts) {
       write_event(name, "function", phase, ts, 0, 0, "");
   }

   // Rich event with category, depth (tid), and optional args JSON
   void write_event(const std::string& name, const std::string& cat,
                    const std::string& phase, uint64_t ts,
                    int pid, int tid, const std::string& args_json) {
       if (!first_event) file << ",\n";
       first_event = false;

       file << "{"
            << "\"name\":\"" << name << "\","
            << "\"cat\":\"" << cat << "\","
            << "\"ph\":\"" << phase << "\","
            << "\"ts\":" << ts << ","
            << "\"pid\":" << pid << ","
            << "\"tid\":" << tid;
       if (!args_json.empty()) {
           file << ",\"args\":" << args_json;
       }
       file << "}" << std::flush;
   }

   // Complete duration event ("X" phase) — written at function exit
   void write_complete_event(const std::string& name, const std::string& cat,
                             uint64_t ts_start, uint64_t duration_ns,
                             int pid, int tid, const std::string& args_json) {
       if (!first_event) file << ",\n";
       first_event = false;

       file << "{"
            << "\"name\":\"" << name << "\","
            << "\"cat\":\"" << cat << "\","
            << "\"ph\":\"X\","
            << "\"ts\":" << ts_start << ","
            << "\"dur\":" << duration_ns << ","
            << "\"pid\":" << pid << ","
            << "\"tid\":" << tid;
       if (!args_json.empty()) {
           file << ",\"args\":" << args_json;
       }
       file << "}" << std::flush;
   }

private:
   std::ofstream file;
   bool first_event;
};

