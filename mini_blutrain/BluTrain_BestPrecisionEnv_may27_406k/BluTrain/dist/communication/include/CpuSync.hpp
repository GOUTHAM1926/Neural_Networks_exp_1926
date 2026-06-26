// #pragma once

// #include <iostream>
// #include <vector>
// #include <string>
// #include <cuda_runtime.h>
// #include <memory>
// #include <thread>




// class Work{

// public:

//     //will only get the stream as argument. 
//     explicit Work(cudaStream_t stream, ncclComm_t comm = nullptr) : stream_(stream), comm_(comm), completed_(true), success_(false) {
// 		// create event with disable timing (fast)
// 		cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
// 		if (err != cudaSuccess) {
// 			throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") +
// 			cudaGetErrorString(err));
// 		}
// 	}

//     ~Work() {
//         // event may be destroyed after use; ignore errors in destructor
//         cudaEventDestroy(event_);
//     }

//     // will create the event with a cudaEventDisableTimings flag.
//     // to avoid unnecessary time recording by the cuda for events.
//     //uses lock()
//     bool begin(){
//         std::lock_guard<std::mutex> lk(mutex_);
//         completed_ = false;
//         success_ = false;
//         return true;
//     }

//     // will stop the recoding. will put a asynchronous record on the kernel execution to note that the task is completed.
//     // will set the completed and success as true if the execution is completed properly.
//     // throws error if the status is not equals to cudaSuccess.
//     //uses unique_lock()
//     bool event_record(){
//         // std::unique_lock<std::mutex> lock(mutex_);
//         // std::lock_guard<std::mutex> lock(mutex_);
//         cudaError_t err = cudaEventRecord(event_, stream_);
//         if (err != cudaSuccess) {
//             std::lock_guard<std::mutex> lk(mutex_);
//             last_err = err;
//             success_ = false;
//             completed_ = true;
//             return false;
//         }
//         // lock.unlock();
//         return true;
//     }

//     void setCommunicator(ncclComm_t comm){ comm_ = comm; }

//     // returns if completed is true
//     bool is_completed(){
//         // std::unique_lock<std::mutex> lock(mutex_);
//         std::lock_guard<std::mutex> lock(mutex_);
//         return completed_;
//     }

//     // will synchronize the cpu until the stream is fully executed.
//     // only when the kernels in the stream is completed, the getStatus() will be called function is returned
//     bool wait(){
//         // cudaError_t err = cudaEventSynchronize(event_);
//         /*
//         eventsync stalls the host thread, whereas the cudaStreamWaitEvent is a async call and returns to the cpu,
//         immediately once called.
//         Thus the sync happens only between the gpus.
//         */
//         cudaError_t err = cudaStreamWaitEvent(OwnTensor::cuda::getCurrentStream(), event_, cudaEventWaitDefault);
//         ncclResult_t async_error;
//         ncclCommGetAsyncError(comm_, &async_error);
//         std::lock_guard<std::mutex> lock(mutex_);
//         completed_ = true;
//         if (async_error != ncclSuccess) {
//             nccl_status = async_error;
//             success_ = false;
//             return false;
//         }
//         if( err != cudaSuccess || nccl_status != ncclSuccess ){
//             success_ = false;
//             last_err = err;
//             return false;
//         }
        
//         success_ = true;
//         return true;
//     }

//     //will return the success.
//     bool is_success(){ 
//         // std::unique_lock<std::mutex> lock(mutex_);
//         std::lock_guard<std::mutex> lock(mutex_);
//         return success_;
//     }

//     bool query() {

//         cudaError_t err = cudaEventQuery(event_);
//         if(err  == cudaErrorNotReady){ return false; }
//         ncclResult_t async_error;
//         ncclCommGetAsyncError(comm_, &async_error);
//         std::lock_guard<std::mutex> lock(mutex_);
        
//         completed_ = true;
//         if (async_error != ncclSuccess) {
//             nccl_status = async_error;
//             success_ = false;
//         }
//         if(err != cudaSuccess){
//             success_ = false;
//             last_err = err;
//             cuda_error = cudaGetErrorString(last_err);
//             return true;
//         }
        
//         success_ = (nccl_status == ncclSuccess);
//         return true;
//     }

//     void setNcclStatus(ncclResult_t status){
//         // std::unique_lock<std::mutex> lock(mutex_);
//         std::lock_guard<std::mutex> lock(mutex_);
//         nccl_status = status;
//         // lock.unlock();
//         return;
//     }

//     ncclResult_t get_ncclStatus() {
//         // std::unique_lock<std::mutex> lock(mutex_);
//         std::lock_guard<std::mutex> lock(mutex_);
//         return nccl_status;
//     }

//     cudaError_t get_lastCudaError() {
//         // std::unique_lock<std::mutex> lock(mutex_);
//         std::lock_guard<std::mutex> lock(mutex_);
//         return last_err;
//     }

//     void mark_failed(){
//         completed_ = true;
//         success_ = false;
//     }

//     bool streamWait(cudaStream_t waitingStream) {
//         if (event_ == nullptr) {
//             // No event was ever recorded, nothing to wait for
//             return true;
//         }
//         cudaError_t err = cudaStreamWaitEvent(waitingStream, event_, 0);
//         if (err != cudaSuccess) {
//             std::cerr << "[Work::streamWait] cudaStreamWaitEvent FAILED: " << cudaGetErrorString(err) << std::endl;
//             std::lock_guard<std::mutex> lock(mutex_);
//             last_err = err;
//             return false;
//         }
//         return true;
//     }

//     bool stream_wait(cudaStream_t waitingStream) {
//         if (event_ == nullptr) {
//             // No event was ever recorded, nothing to wait for
//             return true;
//         }
//         cudaError_t err = cudaStreamWaitEvent(waitingStream, event_, 0);
//         if (err != cudaSuccess) {
//             std::cerr << "[Work::streamWait] cudaStreamWaitEvent FAILED: " << cudaGetErrorString(err) << std::endl;
//             std::lock_guard<std::mutex> lock(mutex_);
//             last_err = err;
//             return false;
//         }
//         return true;
//     }

    

    
// private:
//     std::mutex mutex_;
//     cudaEvent_t event_;
//     bool completed_ = true;
//     bool success_ = false;
//     cudaStream_t stream_;
//     ncclResult_t nccl_status{ncclSuccess};
//     cudaError_t last_err{cudaSuccess};
//     std::string cuda_error;
//     ncclComm_t comm_;

 
// };


#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <memory>
#include <thread>




class Work{

public:

    //will only get the stream as argument. 
    explicit Work(cudaStream_t stream, ncclComm_t comm = nullptr) : stream_(stream), comm_(comm), completed_(true), success_(false) {
		// create event with disable timing (fast)
		cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
		if (err != cudaSuccess) {
			throw std::runtime_error(std::string("cudaEventCreateWithFlags failed: ") +
			cudaGetErrorString(err));
		}
	}

    ~Work() {
        // event may be destroyed after use; ignore errors in destructor
        cudaEventDestroy(event_);
    }

    // will create the event with a cudaEventDisableTimings flag.
    // to avoid unnecessary time recording by the cuda for events.
    //uses lock()
    bool begin(){
        std::lock_guard<std::mutex> lk(mutex_);
        completed_ = false;
        success_ = false;
        return true;
    }

    // Re-arm an already-constructed Work for another collective WITHOUT
    // recreating its CUDA event. The event (event_) is created once in the
    // ctor and reused; only the per-launch bookkeeping is reset. This removes
    // the cudaEventCreateWithFlags/cudaEventDestroy pair that otherwise runs
    // per bucket per step inside the backward hook path (event churn there
    // adds CPU launch latency and can stall kernel issue on the critical
    // last micro-step). Safe to re-record event_ here: any prior
    // cudaStreamWaitEvent captured the event's earlier state at its own call
    // site, and program order guarantees the previous step's wait was already
    // enqueued before this re-arm runs.
    void rearm(cudaStream_t stream, ncclComm_t comm){
        std::lock_guard<std::mutex> lk(mutex_);
        stream_      = stream;
        comm_        = comm;
        completed_   = false;
        success_     = false;
        nccl_status  = ncclSuccess;
        last_err     = cudaSuccess;
    }

    // will stop the recoding. will put a asynchronous record on the kernel execution to note that the task is completed.
    // will set the completed and success as true if the execution is completed properly.
    // throws error if the status is not equals to cudaSuccess.
    //uses unique_lock()
    bool event_record(){
        // std::unique_lock<std::mutex> lock(mutex_);
        // std::lock_guard<std::mutex> lock(mutex_);
        cudaError_t err = cudaEventRecord(event_, stream_);
        if (err != cudaSuccess) {
            std::lock_guard<std::mutex> lk(mutex_);
            last_err = err;
            success_ = false;
            completed_ = true;
            return false;
        }
        // lock.unlock();
        return true;
    }

    void setCommunicator(ncclComm_t comm){ comm_ = comm; }

    // returns if completed is true
    bool is_completed(){
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_;
    }

    // will synchronize the cpu until the stream is fully executed.
    // only when the kernels in the stream is completed, the getStatus() will be called function is returned
    bool wait(){
        // cudaError_t err = cudaEventSynchronize(event_);
        /*
        eventsync stalls the host thread, whereas the cudaStreamWaitEvent is a async call and returns to the cpu,
        immediately once called.
        Thus the sync happens only between the gpus.
        */
        cudaError_t err = cudaStreamWaitEvent(OwnTensor::cuda::getCurrentStream(), event_, cudaEventWaitDefault);
        ncclResult_t async_error;
        ncclCommGetAsyncError(comm_, &async_error);
        std::lock_guard<std::mutex> lock(mutex_);
        completed_ = true;
        if (async_error != ncclSuccess) {
            nccl_status = async_error;
            success_ = false;
            return false;
        }
        if( err != cudaSuccess || nccl_status != ncclSuccess ){
            success_ = false;
            last_err = err;
            return false;
        }
        
        success_ = true;
        return true;
    }

    //will return the success.
    bool is_success(){ 
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return success_;
    }

    bool query() {

        cudaError_t err = cudaEventQuery(event_);
        if(err  == cudaErrorNotReady){ return false; }
        ncclResult_t async_error;
        ncclCommGetAsyncError(comm_, &async_error);
        std::lock_guard<std::mutex> lock(mutex_);
        
        completed_ = true;
        if (async_error != ncclSuccess) {
            nccl_status = async_error;
            success_ = false;
        }
        if(err != cudaSuccess){
            success_ = false;
            last_err = err;
            cuda_error = cudaGetErrorString(last_err);
            return true;
        }
        
        success_ = (nccl_status == ncclSuccess);
        return true;
    }

    void setNcclStatus(ncclResult_t status){
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        nccl_status = status;
        // lock.unlock();
        return;
    }

    ncclResult_t get_ncclStatus() {
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return nccl_status;
    }

    cudaError_t get_lastCudaError() {
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return last_err;
    }

    void mark_failed(){
        completed_ = true;
        success_ = false;
    }

    bool streamWait(cudaStream_t waitingStream) {
        if (event_ == nullptr) {
            // No event was ever recorded, nothing to wait for
            return true;
        }
        cudaError_t err = cudaStreamWaitEvent(waitingStream, event_, 0);
        if (err != cudaSuccess) {
            std::cerr << "[Work::streamWait] cudaStreamWaitEvent FAILED: " << cudaGetErrorString(err) << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            last_err = err;
            return false;
        }
        return true;
    }

    bool stream_wait(cudaStream_t waitingStream) {
        if (event_ == nullptr) {
            // No event was ever recorded, nothing to wait for
            return true;
        }
        cudaError_t err = cudaStreamWaitEvent(waitingStream, event_, 0);
        if (err != cudaSuccess) {
            std::cerr << "[Work::streamWait] cudaStreamWaitEvent FAILED: " << cudaGetErrorString(err) << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            last_err = err;
            return false;
        }
        return true;
    }

    

    
private:
    std::mutex mutex_;
    cudaEvent_t event_;
    bool completed_ = true;
    bool success_ = false;
    cudaStream_t stream_;
    ncclResult_t nccl_status{ncclSuccess};
    cudaError_t last_err{cudaSuccess};
    std::string cuda_error;
    ncclComm_t comm_;

 
};
