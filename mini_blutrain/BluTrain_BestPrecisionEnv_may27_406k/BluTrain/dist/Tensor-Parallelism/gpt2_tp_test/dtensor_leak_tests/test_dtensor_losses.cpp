#include "dtensor_test_utils.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// Forward declaration
void run_dtensor_loss_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);

void run_dtensor_loss_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    if (rank == 0) {
        std::cout << "\n=== DTensor Loss Function Tests ===" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int64_t B = 32, C = 10;
    
    // --- dmse_loss ---
    {
        Layout pred_layout(mesh, {B, C});
        DTensor pred(mesh, pg, pred_layout, "mse_pred");
        pred.mutable_tensor().set_requires_grad(true);
        
        Layout target_layout(mesh, {B, C});
        DTensor target(mesh, pg, target_layout, "mse_target");
        target.mutable_tensor().set_requires_grad(false);
        
        auto run_fn = [&]() {
            return dmse_loss(pred, target);
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("dmse_loss", run_fn, pred, 50);
        print_dtensor_result("dmse_loss [B,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed MSE (manual) ---
    {
        Layout pred_layout(mesh, {B, C});
        DTensor pred(mesh, pg, pred_layout, "manual_mse_pred");
        pred.mutable_tensor().set_requires_grad(true);
        
        Layout target_layout(mesh, {B, C});
        DTensor target(mesh, pg, target_layout, "manual_mse_target");
        target.mutable_tensor().set_requires_grad(false);
        
        auto run_fn = [&]() {
            Tensor diff = autograd::add(pred.mutable_tensor(), target.mutable_tensor() * -1.0f);
            Tensor sq_diff = autograd::mul(diff, diff);
            Tensor local_loss = autograd::mean(sq_diff);
            
            Layout loss_layout(mesh, {1});
            DTensor loss(mesh, pg, loss_layout, "loss");
            loss.mutable_tensor() = local_loss;
            return loss;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed MSE (manual)", run_fn, pred, 50);
        print_dtensor_result("Distributed MSE (manual)", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed MAE ---
    {
        Layout pred_layout(mesh, {B, C});
        DTensor pred(mesh, pg, pred_layout, "mae_pred");
        pred.mutable_tensor().set_requires_grad(true);
        
        Layout target_layout(mesh, {B, C});
        DTensor target(mesh, pg, target_layout, "mae_target");
        target.mutable_tensor().set_requires_grad(false);
        
        auto run_fn = [&]() {
            Tensor local_loss = autograd::mae_loss(pred.mutable_tensor(), target.mutable_tensor());
            
            Layout loss_layout(mesh, {1});
            DTensor loss(mesh, pg, loss_layout, "mae_loss");
            loss.mutable_tensor() = local_loss;
            return loss;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed MAE", run_fn, pred, 50);
        print_dtensor_result("Distributed MAE [B,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed Sparse Cross Entropy ---
    {
        int64_t vocab = 1000;
        
        Layout logits_layout(mesh, {B, vocab});
        DTensor logits(mesh, pg, logits_layout, "sce_logits");
        logits.mutable_tensor().set_requires_grad(true);
        
        // Create target indices
        Layout target_layout(mesh, {B});
        DTensor targets(mesh, pg, target_layout, "sce_targets");
        
        std::vector<float> target_data(B);
        for (int64_t i = 0; i < B; i++) {
            target_data[i] = static_cast<float>(i % vocab);
        }
        targets.setData(target_data);
        targets.mutable_tensor() = targets.mutable_tensor().as_type(Dtype::Int64);
        
        auto run_fn = [&]() {
            Tensor local_loss = autograd::sparse_cross_entropy_loss(
                logits.mutable_tensor(), 
                targets.mutable_tensor()
            );
            
            Layout loss_layout(mesh, {1});
            DTensor loss(mesh, pg, loss_layout, "sce_loss");
            loss.mutable_tensor() = local_loss;
            return loss;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed Sparse CCE", run_fn, logits, 50);
        print_dtensor_result("Distributed Sparse CCE [B,V]", m);
    }
}
