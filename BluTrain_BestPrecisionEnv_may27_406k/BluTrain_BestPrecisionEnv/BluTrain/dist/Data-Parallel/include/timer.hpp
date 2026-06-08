#pragma once
#include <iostream>
#include <chrono>
#include <thread>
#include "Error_logs.hpp"
#include <iomanip>
#include <vector>

using namespace std::chrono_literals;
namespace chrono = std::chrono;
using clock_tt = std::chrono::steady_clock;
using time_ = int64_t;
inline constexpr time_ unset_t = -1;

inline time_ get_time(){
    auto current = clock_tt::now().time_since_epoch();
    return chrono::duration_cast<chrono::nanoseconds>(current).count();
}

inline double getTimeSinceEpoch(){
    auto current = clock_tt::now().time_since_epoch();
    return chrono::duration_cast<chrono::nanoseconds>(current).count();
}


//Timer
struct Func_Timer{
	time_ func_begin_time = unset_t;
	time_ func_end_time = unset_t;
	time_ total_time_taken = unset_t;
    time_ child_time_taken = 0;
    time_ individual_time_taken = unset_t;
	std::string function_name = "";
	std::vector<std::string> called_functions;

	// Allocation tracking: snapshots at function entry
	uint64_t alloc_snapshot_start = 0;
	uint64_t dealloc_snapshot_start = 0;

	// Total allocs/deallocs during this function (including children)
	uint64_t total_allocs = 0;
	uint64_t total_deallocs = 0;

	// Accumulated allocs/deallocs from child functions
	uint64_t child_allocs = 0;
	uint64_t child_deallocs = 0;

	// Individual = total - child (computed at stop)
	uint64_t individual_allocs = 0;
	uint64_t individual_deallocs = 0;

	inline double diff(){
		if(func_begin_time ==unset_t || func_end_time == unset_t){
			throw Error(__FILE__, __LINE__, "Function either not started or not ended");
		}else{
			return static_cast<double>(func_end_time - func_begin_time);
		}
	}
};


struct Timer{
    time_ model_begin_time = unset_t;   
    time_ model_end_time = unset_t;
    time_ forward_start_time = unset_t;
    time_ forward_end_time = unset_t;
    time_ backward_start_time = unset_t;
    time_ backward_end_time = unset_t;
    time_ comm_start_time = unset_t;
    time_ comm_end_time = unset_t;



    enum Event{ 
        forward_start,
        forward_end,
        backward_start,
        backward_end,
        model_start,
        model_end,
        comm_start,
        comm_stop
    };


    inline void record(Event event){
        get_timer(event) = get_time();
    }

    inline double diff(Event event1, Event event2){
        time_ initial = get_timer(event1);
        time_ end = get_timer(event2);

        if(initial == unset_t || end == unset_t){
            return 0.0;
        }
        return static_cast<double>(end - initial) / 1e9;

    }

    inline time_& get_timer(Event event){
        switch(event){
            case Event::forward_start:
                return forward_start_time;
            case Event::forward_end:
                return forward_end_time;
            case Event::backward_start:
                return backward_start_time;
            case Event::backward_end:
                return backward_end_time;
            case Event::model_start:
                return model_begin_time;
            case Event::model_end:
                return model_end_time;
            case Event::comm_start:
                return comm_start_time;
            case Event::comm_stop:
                return comm_end_time;
            default:
                throw Error(
                        __FILE__,
                        __LINE__,
                        "NO such Type"
                    );
        }
    }

    void reset_timer(){
        forward_start_time = forward_end_time = 
        backward_start_time = backward_end_time = 
        comm_start_time = comm_end_time = unset_t;
    }

    void reset_timer_all(){
        reset_timer();
        model_begin_time = model_end_time = unset_t;
    }
    
};

template <typename T, typename... Args>
void print_profiler_result(std::shared_ptr<T>& timer, std::ostream& stream = std::cout, Args... args){
     // Determine a base time to show relative "Timeline" positions
    int64_t base = timer->forward_start_time;
    
    // Header Section
    stream << "\n" << std::string(60, '=') << "\n";
    stream << " PROFILER STEP: ";
    ((stream << args << " "), ...);
    stream << "\n" << std::string(60, '-') << "\n";

    // Table Header
    stream << std::left << std::setw(20) << "EVENT" 
           << std::setw(20) << "TIMESTAMP (ns)" 
           << "TIMELINE (s)\n";
    stream << std::string(60, '-') << "\n";

    // Formatting for the table
    auto print_row = [&](const std::string& label, int64_t ts) {
        stream << std::left << std::setw(20) << label;
        if (ts == unset_t) {
            stream << std::setw(20) << "UNSET" << "N/A\n";
        } else {
            double relative_sec = static_cast<double>(ts - base) / 1e9;
            stream << std::setw(20) << ts 
                   << std::fixed << std::setprecision(6) << "+" << relative_sec << "s\n";
        }
    };

    // Rows
    print_row("Forward Start",    timer->forward_start_time);
    print_row("Forward End",      timer->forward_end_time);
    print_row("Backward Start",   timer->backward_start_time);
    print_row("Backward End",     timer->backward_end_time);
    print_row("Comm Start",       timer->comm_start_time);
    print_row("Comm End",         timer->comm_end_time);

    // Summary Section
    stream << std::string(60, '-') << "\n";
    stream << std::fixed << std::setprecision(6);
    
    auto fwd  = timer->diff(Timer::Event::forward_start, Timer::Event::forward_end);
    auto bwd  = timer->diff(Timer::Event::backward_start, Timer::Event::backward_end);
    auto comm = timer->diff(Timer::Event::comm_start, Timer::Event::comm_stop);

    stream << " > TOTAL FORWARD:    " << std::setw(10) << fwd  << " s  (" << (fwd * 1000)  << " ms)\n";
    stream << " > TOTAL BACKWARD:   " << std::setw(10) << bwd  << " s  (" << (bwd * 1000)  << " ms)\n";
    stream << " > TOTAL COMM:       " << std::setw(10) << comm << " s  (" << (comm * 1000) << " ms)\n";
    
    double total_accounted = fwd + bwd + comm;
    stream << " > SUM ACCOUNTED:    " << std::setw(10) << total_accounted << " s\n";
    stream << std::string(60, '=') << "\n" << std::endl;
    
}












