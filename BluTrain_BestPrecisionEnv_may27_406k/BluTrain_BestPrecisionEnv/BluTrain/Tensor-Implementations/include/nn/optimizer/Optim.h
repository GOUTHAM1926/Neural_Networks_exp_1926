// #pragma once

// #include "core/Tensor.h"
// #include "nn/optimizer/LossScaler.h"
// #include <vector>
// #include <unordered_map>
// #include <memory>

// namespace OwnTensor {
// namespace nn {

// class Optimizer {
// public:
//   Optimizer(const std::vector<Tensor>& params);
//   virtual ~Optimizer() = default;

//   virtual void step() = 0;
//   virtual void set_lr(float lr) = 0;
//   void zero_grad();

//   Tensor* get_master_weight(const Tensor& v);

//     // Save/Load state
//   virtual void save_state(std::ostream& os);
//   virtual void load_state(std::istream& is);

//   // Attach a LossScaler for mixed precision training
//   void set_scaler(LossScaler* scaler) { scaler_ = scaler; }
//   LossScaler* get_scaler() const { return scaler_; }

//   virtual int64_t step_count() const { return 0; }
//   virtual std::vector<Tensor> get_m_buffers() const { return {}; }
//   virtual std::vector<Tensor> get_v_buffers() const { return {}; }


// protected:
//   std::vector<Tensor> params_;
//   // Map from Tensor source pointer to master copy.
//   // We use raw pointer to TensorImpl or some unique ID as key.
//   // Since Tensors are shared_ptr-like, we use the impl pointer.
//   std::unordered_map<void*, Tensor> master_params_;
//   LossScaler* scaler_ = nullptr;
// };


// class SGDOptimizer : public Optimizer {
// public:
//   SGDOptimizer(const std::vector<Tensor>& params, float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
 
//   void step() override;
//   void set_lr(float lr) override { learning_rate_ = lr; }
  
//   void save_state(std::ostream& os) override;
//   void load_state(std::istream& is) override;
  
//   int64_t step_count() const override { return step_count_; }
//   std::vector<Tensor> get_m_buffers() const override { return momentum_buffer_; }
  

// private:
//   float learning_rate_;
//   float momentum_;
//   float weight_decay_;
//   int64_t step_count_ = 0;
//   std::vector<Tensor> momentum_buffer_;
//   bool initialized_ = false;
// };




// // *********************************************************************************************
// // ================================ Adam Optimizer =============================================
// // *********************************************************************************************




// /**
// * @brief Adam optimizer with momentum and adaptive learning rates
// *
// * Implements the Adam optimization algorithm:
// * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
// * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
// * m_hat = m_t / (1 - beta1^t)
// * v_hat = v_t / (1 - beta2^t)
// * theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
// */
// class Adam : public Optimizer {
// private:
//   float lr_;
//   float beta1_;
//   float beta2_;
//   float eps_;
//   float weight_decay_;
//   int64_t step_count_;
 
//   // First moment estimates (momentum)
//   std::vector<Tensor> m_;
//   // Second moment estimates (RMSProp)
//   std::vector<Tensor> v_;
//   bool initialized_;
 
// public:
//   /**
//    * @brief Construct Adam optimizer
//    *
//    * @param params Vector of parameter tensors to optimize
//    * @param lr Learning rate (default: 0.001)
//    * @param beta1 Exponential decay rate for first moment (default: 0.9)
//    * @param beta2 Exponential decay rate for second moment (default: 0.999)
//    * @param eps Small constant for numerical stability (default: 1e-8)
//    * @param weight_decay L2 regularization coefficient (default: 0)
//    */
//   Adam(const std::vector<Tensor>& params,
//        float lr = 0.001f,
//        float beta1 = 0.9f,
//        float beta2 = 0.999f,
//        float eps = 1e-8f,
//        float weight_decay = 0.0f);
 
//   /**
//    * @brief Perform a single optimization step
//    */
//   void step() override;
 
 
//   /**
//    * @brief Get current learning rate
//    */
//   float get_lr() const { return lr_; }
 
//   /**
//    * @brief Set learning rate
//    */
//   void set_lr(float lr) { lr_ = lr; }
 
//   /**
//    * @brief Get step count
//    */
//   int64_t get_step_count() const { return step_count_; }

//   // Save/Load state
//   void save_state(std::ostream& os) override;
//   void load_state(std::istream& is) override;

//   int64_t step_count() const override { return step_count_; }
//   std::vector<Tensor> get_m_buffers() const override { return m_; }
//   std::vector<Tensor> get_v_buffers() const override { return v_; }
// };




// // *********************************************************************************************
// // ================================ AdamW Optimizer ============================================
// // *********************************************************************************************

// /**
// * @brief AdamW optimizer with momentum and adaptive learning rates
// *
// * Implements the AdamW optimization algorithm (decoupled weight decay):
// * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
// * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
// * m_hat = m_t / (1 - beta1^t)
// * v_hat = v_t / (1 - beta2^t)
// * theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
// */
// class AdamW : public Optimizer {
// private:
//   float lr_;
//   float beta1_;
//   float beta2_;
//   float eps_;
//   float weight_decay_;
//   int64_t step_count_;
 
//   // First moment estimates (momentum)
//   std::vector<Tensor> m_;
//   // Second moment estimates (RMSProp)
//   std::vector<Tensor> v_;
//   bool initialized_;
 
// public:
//   /**
//    * @brief Construct AdamW optimizer
//    *
//    * @param params Vector of parameter tensors to optimize
//    * @param lr Learning rate (default: 0.001)
//    * @param beta1 Exponential decay rate for first moment (default: 0.9)
//    * @param beta2 Exponential decay rate for second moment (default: 0.999)
//    * @param eps Small constant for numerical stability (default: 1e-8)
//    * @param weight_decay L2 regularization coefficient (default: 0)
//    */
//   AdamW(const std::vector<Tensor>& params,
//        float lr = 0.001f,
//        float beta1 = 0.9f,
//        float beta2 = 0.999f,
//        float eps = 1e-8f,
//        float weight_decay = 0.0f);
 
//   /**
//    * @brief Perform a single optimization step
//    */
//   void step() override;
 
 
//   /**
//    * @brief Get current learning rate
//    */
//   float get_lr() const { return lr_; }
 
//   /**
//    * @brief Set learning rate
//    */
//   void set_lr(float lr) { lr_ = lr; }
 
//   /**
//    * @brief Get step count
//    */
//   int64_t get_step_count() const { return step_count_; }

//     // Save/Load state
//   void save_state(std::ostream& os) override;
//   void load_state(std::istream& is) override;

//   int64_t step_count() const override { return step_count_; }
//   std::vector<Tensor> get_m_buffers() const override { return m_; }
//   std::vector<Tensor> get_v_buffers() const override { return v_; }
// };




// // // ==============================================================================================
// // // ================================ Gradient Clipping ===========================================
// // // ==============================================================================================


// // /**
// //  * @brief Clips gradient norm of an iterable of parameters
// //  *
// //  * @param params Vector of parameter tensors whose gradients to clip
// //  * @param max_norm Maximum norm allowed
// //  * @param norm_type Type of norm to use (2.0 for L2, inf for infinity norm)
// //  * @param error_if_nonfinite If true, throws error when gradients contain non-finite values
// //  * @return Total norm of parameter gradients before clipping
// //  */
// float clip_grad_norm_(std::vector<Tensor>& params,
//                      float max_norm,
//                      float norm_type = 2.0f,
//                      bool error_if_nonfinite = false);




// } // namespace nn
// } // namespace OwnTensor
























#pragma once

#include "core/Tensor.h"
#include "nn/optimizer/LossScaler.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace OwnTensor {
namespace nn {

class Optimizer {
public:
  Optimizer(const std::vector<Tensor>& params);
  virtual ~Optimizer() = default;

  virtual void step() = 0;
  virtual void set_lr(float lr) = 0;
  void zero_grad();

  Tensor* get_master_weight(const Tensor& v);

    // Save/Load state
  virtual void save_state(std::ostream& os);
  virtual void load_state(std::istream& is);

  // Attach a LossScaler for mixed precision training
  void set_scaler(LossScaler* scaler) { scaler_ = scaler; }
  LossScaler* get_scaler() const { return scaler_; }

  virtual int64_t step_count() const { return 0; }
  virtual std::vector<Tensor> get_m_buffers() const { return {}; }
  virtual std::vector<Tensor> get_v_buffers() const { return {}; }


protected:
  std::vector<Tensor> params_;
  // Map from Tensor source pointer to master copy.
  // We use raw pointer to TensorImpl or some unique ID as key.
  // Since Tensors are shared_ptr-like, we use the impl pointer.
  std::unordered_map<void*, Tensor> master_params_;
  LossScaler* scaler_ = nullptr;
};


class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(const std::vector<Tensor>& params, float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
 
  void step() override;
  void set_lr(float lr) override { learning_rate_ = lr; }
  
  void save_state(std::ostream& os) override;
  void load_state(std::istream& is) override;
  
  int64_t step_count() const override { return step_count_; }
  std::vector<Tensor> get_m_buffers() const override { return momentum_buffer_; }
  

private:
  float learning_rate_;
  float momentum_;
  float weight_decay_;
  int64_t step_count_ = 0;
  std::vector<Tensor> momentum_buffer_;
  bool initialized_ = false;
};




// *********************************************************************************************
// ================================ Adam Optimizer =============================================
// *********************************************************************************************




/**
* @brief Adam optimizer with momentum and adaptive learning rates
*
* Implements the Adam optimization algorithm:
* m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
* v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
* m_hat = m_t / (1 - beta1^t)
* v_hat = v_t / (1 - beta2^t)
* theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
*/
class Adam : public Optimizer {
private:
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int64_t step_count_;
 
  // First moment estimates (momentum)
  std::vector<Tensor> m_;
  // Second moment estimates (RMSProp)
  std::vector<Tensor> v_;
  bool initialized_;
 
public:
  /**
   * @brief Construct Adam optimizer
   *
   * @param params Vector of parameter tensors to optimize
   * @param lr Learning rate (default: 0.001)
   * @param beta1 Exponential decay rate for first moment (default: 0.9)
   * @param beta2 Exponential decay rate for second moment (default: 0.999)
   * @param eps Small constant for numerical stability (default: 1e-8)
   * @param weight_decay L2 regularization coefficient (default: 0)
   */
  Adam(const std::vector<Tensor>& params,
       float lr = 0.001f,
       float beta1 = 0.9f,
       float beta2 = 0.999f,
       float eps = 1e-8f,
       float weight_decay = 0.0f);
 
  /**
   * @brief Perform a single optimization step
   */
  void step() override;
 
 
  /**
   * @brief Get current learning rate
   */
  float get_lr() const { return lr_; }
 
  /**
   * @brief Set learning rate
   */
  void set_lr(float lr) { lr_ = lr; }
 
  /**
   * @brief Get step count
   */
  int64_t get_step_count() const { return step_count_; }

  // Save/Load state
  void save_state(std::ostream& os) override;
  void load_state(std::istream& is) override;

  int64_t step_count() const override { return step_count_; }
  std::vector<Tensor> get_m_buffers() const override { return m_; }
  std::vector<Tensor> get_v_buffers() const override { return v_; }
};




// *********************************************************************************************
// ================================ AdamW Optimizer ============================================
// *********************************************************************************************

/**
* @brief AdamW optimizer with momentum and adaptive learning rates
*
* Implements the AdamW optimization algorithm (decoupled weight decay):
* m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
* v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
* m_hat = m_t / (1 - beta1^t)
* v_hat = v_t / (1 - beta2^t)
* theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
*/
class AdamW : public Optimizer {
private:
  float lr_;
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  int64_t step_count_;
 
  // First moment estimates (momentum)
  std::vector<Tensor> m_;
  // Second moment estimates (RMSProp)
  std::vector<Tensor> v_;
  bool initialized_;
 
public:
  /**
   * @brief Construct AdamW optimizer
   *
   * @param params Vector of parameter tensors to optimize
   * @param lr Learning rate (default: 0.001)
   * @param beta1 Exponential decay rate for first moment (default: 0.9)
   * @param beta2 Exponential decay rate for second moment (default: 0.999)
   * @param eps Small constant for numerical stability (default: 1e-8)
   * @param weight_decay L2 regularization coefficient (default: 0)
   */
  AdamW(const std::vector<Tensor>& params,
       float lr = 0.001f,
       float beta1 = 0.9f,
       float beta2 = 0.999f,
       float eps = 1e-8f,
       float weight_decay = 0.0f);
 
  /**
   * @brief Perform a single optimization step
   */
  void step() override;
 
 
  /**
   * @brief Get current learning rate
   */
  float get_lr() const { return lr_; }
 
  /**
   * @brief Set learning rate
   */
  void set_lr(float lr) { lr_ = lr; }
 
  /**
   * @brief Get step count
   */
  int64_t get_step_count() const { return step_count_; }

    // Save/Load state
  void save_state(std::ostream& os) override;
  void load_state(std::istream& is) override;

  int64_t step_count() const override { return step_count_; }
  std::vector<Tensor> get_m_buffers() const override { return m_; }
  std::vector<Tensor> get_v_buffers() const override { return v_; }
};




// // ==============================================================================================
// // ================================ Gradient Clipping ===========================================
// // ==============================================================================================


// /**
//  * @brief Clips gradient norm of an iterable of parameters
//  *
//  * @param params Vector of parameter tensors whose gradients to clip
//  * @param max_norm Maximum norm allowed
//  * @param norm_type Type of norm to use (2.0 for L2, inf for infinity norm)
//  * @param error_if_nonfinite If true, throws error when gradients contain non-finite values
//  * @return Total norm of parameter gradients before clipping
//  */
float clip_grad_norm_(std::vector<Tensor>& params,
                     float max_norm,
                     float norm_type = 2.0f,
                     bool error_if_nonfinite = false);

// Async variant of clip_grad_norm_. Performs identical GPU work (multi-tensor
// L2-norm + multi-tensor scale) but ends with a non-blocking cudaMemcpyAsync
// of the computed norm into a caller-provided pinned host float, so the
// per-step training loop never has to drain the CUDA stream at this point.
//
// Caller contract:
//   - `pinned_host_norm` MUST point to host memory allocated with
//     cudaMallocHost (pageable memory falls back to a sync copy and defeats
//     the optimisation).
//   - The value at *pinned_host_norm is only safe to read AFTER the next
//     CUDA stream sync (e.g. on the following training step's loss readback,
//     or after an explicit cudaStreamSynchronize). Reading it immediately
//     gives a stale value from a previous call (or zero on first call).
//   - On the very first call there is nothing previously written; callers
//     should initialise *pinned_host_norm to a sentinel before the loop.
//   - error_if_nonfinite is intentionally absent: that check requires a
//     host value, which forces a sync. Use clip_grad_norm_ if you need it.
void clip_grad_norm_async_(std::vector<Tensor>& params,
                           float max_norm,
                           float* pinned_host_norm,
                           float norm_type = 2.0f);




} // namespace nn
} // namespace OwnTensor