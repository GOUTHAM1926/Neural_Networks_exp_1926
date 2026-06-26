# ============================================================
# Top-Level Makefile
# ============================================================
# Orchestrates:
#   1. Tensor-Implementations  -> lib/libtensor.{a,so}, lib/libblublas.so
#                                 (BluBridge-BLAS is built by the tensor sub-make)
#   2. dist                    -> dist/build/lib/libdist.{a,so}
#                                 (Tensor Parallelism, Data Parallelism, Communication)
#
# Build:
#   make                    Build everything (blublas + tensor + dist)
#   make tensor             Build libtensor.{a,so} only
#   make blublas            Build libblublas.so only
#   make dist               Build libdist.{a,so} only
#
# Clean:
#   make clean              Remove all build artifacts
#   make clean-tensor       Remove tensor build artifacts only
#   make clean-blublas      Remove BluBridge-BLAS build artifacts only
#   make clean-dist         Remove dist build artifacts only
#
# Run:
#   make run FILE=path.cpp                      Compile + run a single .cpp
#   make run-mpi FILE=path.cpp [NP=2]           Same, launched via mpirun
#   make run-train [CONFIG=...]                 Run gpt2_scripts/train.cpp
#   make run-train-mpi [CONFIG=... NP=2]        Same, launched via mpirun
#
# Flags:
#   WITH_BLUBLAS=1          Link libtensor against libblublas (BluBridge-BLAS backend)
#                           Default (unset/0): cuBLAS backend.
#                           When set, blublas is built before tensor.
#   SM_ARCH=<cc>            Override GPU compute capability (auto-detected via nvidia-smi)
#   NP=<n>                  MPI process count for run-mpi / run-train-mpi (default: 2)
#   CONFIG=<path>           Training config file (default: gpt2_scripts/config/train_config.cfg)
# ============================================================

# ---- Directories ----
TENSOR_DIR     := Tensor-Implementations
TENSOR_LIB_DIR := $(TENSOR_DIR)/lib
DIST_DIR       := dist
DIST_LIB_DIR   := $(DIST_DIR)/lib

# ---- Compilers ----
CXX    := g++
MPICXX := /usr/bin/mpic++
NVCC   := $(shell which nvcc 2>/dev/null)

# ---- CUDA auto-detect ----
CUDA_NVCC := $(shell which nvcc 2>/dev/null)
CUDA_ROOT := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_ROOT)/include
CUDA_LIB  := $(CUDA_ROOT)/lib64

# ---- NCCL ----
NCCL_INC := /usr/local/include
NCCL_LIB := /usr/local/lib

# ---- NVTX (best-effort) ----
NVTX_INC := $(shell find /opt/nvidia -path '*/nvtx/include' -type d 2>/dev/null | head -1)

# ---- GPU compute capability auto-detect ----
SM_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(strip $(SM_ARCH)),)
  SM_ARCH := 86
endif

# ============================================================
# BLAS backend selection
# ============================================================
# WITH_BLUBLAS=1 -> link libtensor against libblublas (BluBridge-BLAS backend)
# WITH_BLUBLAS=0 (default) -> cuBLAS backend
WITH_BLUBLAS ?= 0
ifeq ($(WITH_BLUBLAS),1)
  TENSOR_DEPS      := blublas
  TENSOR_MAKE_ARGS := WITH_BLUBLAS=1 SM_ARCH=$(SM_ARCH)
  BLAS_RUN_LDLIBS  := -lblublas
  BLAS_LABEL       := BluBridge-BLAS (libblublas.so)
else
  TENSOR_DEPS      :=
  TENSOR_MAKE_ARGS := SM_ARCH=$(SM_ARCH)
  BLAS_RUN_LDLIBS  :=
  BLAS_LABEL       := cuBLAS
endif

# ============================================================
# Run-target compile / link flags
# ============================================================
CXXFLAGS = -std=c++2a -fPIC -O3 -g -DWITH_CUDA -fopenmp

INCLUDES = -I$(TENSOR_DIR)/include -I$(CUDA_INC) \
           -I$(DIST_DIR)/Tensor-Parallelism \
           -I$(DIST_DIR)/Tensor-Parallelism/tensor \
           -I$(DIST_DIR)/Tensor-Parallelism/dnn \
           -I$(DIST_DIR)/Tensor-Parallelism/bridge \
           -I$(DIST_DIR)/Tensor-Parallelism/memory \
           -I$(DIST_DIR)/Data-Parallel/include \
           -I$(DIST_DIR)/communication/include
INCLUDES += $(if $(NVTX_INC),-I$(NVTX_INC))

LDFLAGS = -L$(TENSOR_LIB_DIR) -L$(DIST_LIB_DIR) -L$(CUDA_LIB) -L$(NCCL_LIB) \
          -Wl,-rpath,$(abspath $(TENSOR_LIB_DIR)) \
          -Wl,-rpath,$(abspath $(DIST_LIB_DIR)) \
          -Wl,-rpath,$(NCCL_LIB)

LDLIBS = -ldist -ltensor $(BLAS_RUN_LDLIBS) -lcuda -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz 

# Runtime library path. libblublas.so lives in the same lib dir as libtensor.so,
# so a single TENSOR_LIB_DIR entry covers both backends.
RUN_LD_PATH := $(abspath $(TENSOR_LIB_DIR)):$(abspath $(DIST_LIB_DIR))

# Default MPI process count
NP ?= 2

# Default training config
CONFIG ?= gpt2_scripts/config/train_config.cfg
TRAIN_INCLUDES = $(INCLUDES) -Igpt2_scripts

# ============================================================
# Build Targets
# ============================================================
.PHONY: all tensor blublas dist \
        clean clean-tensor clean-blublas clean-dist \
        run run-mpi run-train run-train-mpi

all: blublas tensor dist
	@echo "\n[SUCCESS] Full build complete."
	@echo "  GPU arch:    sm_$(SM_ARCH)"
	@echo "  BLAS:        $(BLAS_LABEL)"
	@echo "  Tensor libs: $(TENSOR_LIB_DIR)/libtensor.{a,so}"
	@echo "  BluBLAS lib: $(TENSOR_LIB_DIR)/libblublas.so"
	@echo "  Dist libs:   $(DIST_LIB_DIR)/libdist.{a,so}"

# ---- BluBridge-BLAS (built by the tensor sub-make's `blas` target) ----
blublas:
	@echo "\n[BUILD] BluBridge-BLAS (sm_$(SM_ARCH))"
	$(MAKE) -C $(TENSOR_DIR) blas SM_ARCH=$(SM_ARCH)

# ---- Tensor-Implementations ----
# When WITH_BLUBLAS=1, blublas is a prerequisite so libblublas.so exists at link time.
tensor: $(TENSOR_DEPS)
	@echo "\n[BUILD] Tensor-Implementations (sm_$(SM_ARCH), backend=$(BLAS_LABEL))"
	$(MAKE) -C $(TENSOR_DIR) tensor $(TENSOR_MAKE_ARGS)

# ---- dist (combined Makefile: TP + DP + Communication -> libdist.{a,so}) ----
dist: tensor
	@echo "\n[BUILD] dist (sm_$(SM_ARCH))"
	$(MAKE) -C $(DIST_DIR) all SM_ARCH=$(SM_ARCH)

# ============================================================
# Clean Targets
# ============================================================
clean: clean-tensor clean-blublas clean-dist
	@echo "[CLEAN] All build artifacts removed."

clean-tensor:
	@echo "[CLEAN] Tensor-Implementations"
	$(MAKE) -C $(TENSOR_DIR) clean-tensor

clean-blublas:
	@echo "[CLEAN] BluBridge-BLAS"
	$(MAKE) -C $(TENSOR_DIR) clean-blas

clean-dist:
	@echo "[CLEAN] dist"
	$(MAKE) -C $(DIST_DIR) clean

# ============================================================
# Run Targets
# ============================================================
# Compile and run a single .cpp file linked against libtensor + libdist.
#   make run FILE=path/to/file.cpp [WITH_BLUBLAS=1]
run: all
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Specify a file to run."; \
		echo "Usage: make run FILE=path/to/file.cpp"; \
		exit 1; \
	fi
	@echo "\n[COMPILE] $(FILE) (BLAS: $(BLAS_LABEL))"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o _runner $(FILE) $(LDFLAGS) $(LDLIBS)
	@echo "[RUN] $(FILE)\n"
	@LD_LIBRARY_PATH=$(RUN_LD_PATH):$$LD_LIBRARY_PATH ./_runner
	@rm -f _runner

# Compile and run with mpirun (for distributed programs).
#   make run-mpi FILE=path/to/file.cpp [NP=4] [WITH_BLUBLAS=1]
run-mpi: all
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Specify a file to run."; \
		echo "Usage: make run-mpi FILE=path/to/file.cpp [NP=2]"; \
		exit 1; \
	fi
	@echo "\n[COMPILE] $(FILE) (BLAS: $(BLAS_LABEL))"
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -o _runner_0 $(FILE) $(LDFLAGS) $(LDLIBS)
	@echo "[RUN] mpirun -np $(NP) $(FILE)\n"
	@LD_LIBRARY_PATH=$(RUN_LD_PATH):$$LD_LIBRARY_PATH mpirun -np $(NP) -x LD_LIBRARY_PATH ./_runner_0
# 	@rm -f _runner_

# Compile and run the unified training script (single GPU).
#   make run-train [CONFIG=gpt2_scripts/config/train_config.cfg] [WITH_BLUBLAS=1]
run-train: all
	@echo "\n[COMPILE] gpt2_scripts/train.cpp (BLAS: $(BLAS_LABEL))"
	$(MPICXX) $(CXXFLAGS) $(TRAIN_INCLUDES) -o _train_runner gpt2_scripts/train.cpp $(LDFLAGS) $(LDLIBS)
	@echo "[RUN] Single-GPU training with config: $(CONFIG)\n"
	@LD_LIBRARY_PATH=$(RUN_LD_PATH):$$LD_LIBRARY_PATH ./_train_runner --config=$(CONFIG)
	@rm -f _train_runner

# Compile and run the unified training script with mpirun (DDP / 2D parallel).
#   make run-train-mpi [CONFIG=...] [NP=4] [WITH_BLUBLAS=1]
run-train-mpi: all
	@echo "\n[COMPILE] gpt2_scripts/train.cpp (BLAS: $(BLAS_LABEL))"
	$(MPICXX) $(CXXFLAGS) $(TRAIN_INCLUDES) -o _train_runner gpt2_scripts/train.cpp $(LDFLAGS) $(LDLIBS)
	@echo "[RUN] mpirun -np $(NP) training with config: $(CONFIG)\n"
	@LD_LIBRARY_PATH=$(RUN_LD_PATH):$$LD_LIBRARY_PATH mpirun -np $(NP) -x LD_LIBRARY_PATH ./_train_runner --config=$(CONFIG)
	@rm -f _train_runner

print_sm:
	@echo "Detected Compute Capability: sm_$(SM_ARCH)"
