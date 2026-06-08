# ============================================================
# dist/ Combined Makefile
# ============================================================
# Replaces the previous 4-file build system:
#   dist/CMakeLists.txt
#   dist/Data-Parallel/CMakeLists.txt
#   dist/Tensor-Parallelism/Makefile  (kept for stand-alone use)
#   dist/communication/Makefile       (superseded by this file)
#
# Builds: lib/libdist.{a,so}
#   A combined static + shared library containing object code from:
#     - Tensor-Parallelism (dtensor + DNN kernels)
#     - Data-Parallel      (DDP + ZeRO + 2D parallel)
#     - Communication      (ProcessGroupNCCL)
#
# Targets:
#   make [all]              Build lib/libdist.{a,so}
#   make tp                 Build only Tensor-Parallelism object files
#   make dp                 Build only Data-Parallel object files
#   make comm               Build only Communication object files
#   make clean              Remove all build artifacts
#   make clean-tp           Remove only Tensor-Parallelism artifacts
#   make clean-dp           Remove only Data-Parallel artifacts
#   make clean-comm         Remove only Communication artifacts
#   make print_sm           Print detected GPU compute capability
#
# Auto-detected (override on command line):
#   nvcc / CUDA_HOME    via `which nvcc`
#   mpic++              via `which mpic++`
#   SM_ARCH             via `nvidia-smi`
#   NCCL                /usr/local/lib, /usr/lib*, /usr/lib/x86_64-linux-gnu
#   NVTX                /opt/nvidia/...
#
# Override examples:
#   make SM_ARCH=120
#   make MPICXX=/opt/openmpi/bin/mpic++
# ============================================================

SHELL := /bin/bash
AR    := ar

# ============================================================
# Compilers
# ============================================================
MPICXX ?= $(shell which mpic++ 2>/dev/null)
ifeq ($(strip $(MPICXX)),)
  MPICXX := /usr/bin/mpic++
endif
CXX := $(MPICXX)

CUDA_NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
  $(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif
NVCC := $(CUDA_NVCC)

# ============================================================
# CUDA auto-detect
# ============================================================
CUDA_HOME := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64

# ============================================================
# GPU compute capability auto-detect
# ============================================================
SM_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(strip $(SM_ARCH)),)
  SM_ARCH := 86
endif

# ============================================================
# NCCL auto-detect
# ============================================================
NCCL_LIB_DIR := $(shell for d in /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do [ -e "$$d/libnccl.so" ] && echo $$d && break; done)
ifeq ($(strip $(NCCL_LIB_DIR)),)
  NCCL_LIB_DIR := /usr/local/lib
endif
NCCL_INC_DIR := $(shell for d in /usr/local/include /usr/include; do [ -e "$$d/nccl.h" ] && echo $$d && break; done)
ifeq ($(strip $(NCCL_INC_DIR)),)
  NCCL_INC_DIR := /usr/local/include
endif

# ============================================================
# NVTX (best-effort; falls back to CUDA include if absent)
# ============================================================
NVTX_INC := $(shell find /opt/ -name "nvtx3" -type d 2>/dev/null | head -1 | xargs dirname)
ifeq ($(strip $(NVTX_INC)),)
  NVTX_INC := $(CUDA_INC)
endif

# ============================================================
# Tensor-Implementations headers (sibling of dist/)
# ============================================================
TENSOR_IMPL_DIR := $(CURDIR)/../Tensor-Implementations

# ============================================================
# Output directories
# ============================================================
DIST_DIR  := $(CURDIR)
BUILD_DIR := $(DIST_DIR)/build
LIB_DIR   := $(DIST_DIR)/lib

DIST_LIB_A  := $(LIB_DIR)/libdist.a
DIST_LIB_SO := $(LIB_DIR)/libdist.so

# ============================================================
# Common compile flags
# ============================================================
CXXFLAGS  := -fPIC -std=c++2a -O3 -g -DWITH_CUDA
NVCCFLAGS := -std=c++17 -Xcompiler -fPIC -O3 -g -arch=sm_$(SM_ARCH) -ccbin=$(CXX)

# ============================================================
# Sub-module directories
# ============================================================
TP_DIR   := Tensor-Parallelism
DP_DIR   := Data-Parallel
COMM_DIR := communication

# ============================================================
# Per-module include paths
# ============================================================
TP_INCLUDES = \
    -I$(TP_DIR) \
    -I$(TP_DIR)/tensor \
    -I$(TP_DIR)/dnn \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_INC) \
	-I$(NVTX_INC)

DP_INCLUDES = \
    -I$(DP_DIR)/include \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_INC) \
    -I$(NCCL_INC_DIR) \
    -I$(NVTX_INC)

COMM_INCLUDES = \
    -I$(COMM_DIR) \
    -I$(COMM_DIR)/include \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_INC) \
    -I$(NCCL_INC_DIR)

# ============================================================
# Sources
# ============================================================
TP_CPP_SRCS = \
    $(TP_DIR)/tensor/dtensor.cpp \
    $(TP_DIR)/tensor/device_mesh.cpp \
    $(TP_DIR)/tensor/placement.cpp

TP_CU_SRCS = \
    $(TP_DIR)/dnn/dist_grad_norm_kernels.cu \
    $(TP_DIR)/dnn/EntropyKernels.cu \
    $(TP_DIR)/tensor/headtail_kernel.cu \
    $(TP_DIR)/tensor/fused_transpose_kernel.cu

DP_CPP_SRCS = \
    $(DP_DIR)/src/DataParallel.cpp \
    $(DP_DIR)/src/Error_logs.cpp \
    $(DP_DIR)/src/profiler.cpp \
    $(DP_DIR)/src/ZeRORedundancyOptimizer.cpp \
    $(DP_DIR)/src/ddp_2d.cpp \
    $(DP_DIR)/src/parallel_2d.cpp

COMM_CPP_SRCS = \
    $(COMM_DIR)/src/processGroupNccl.cpp

# ============================================================
# Object files
# ============================================================
TP_CPP_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TP_CPP_SRCS))
TP_CU_OBJS  = $(patsubst %.cu,$(BUILD_DIR)/%.o,$(TP_CU_SRCS))
TP_OBJS     = $(TP_CPP_OBJS) $(TP_CU_OBJS)

DP_OBJS   = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(DP_CPP_SRCS))
COMM_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(COMM_CPP_SRCS))

ALL_OBJS = $(TP_OBJS) $(DP_OBJS) $(COMM_OBJS)

# ============================================================
# Linker flags for the combined .so
# ============================================================
DIST_LDFLAGS := -shared -L$(CUDA_LIB) -L$(NCCL_LIB_DIR) \
                -Wl,-rpath,$(CUDA_LIB) -Wl,-rpath,$(NCCL_LIB_DIR)
DIST_LDLIBS  := -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz

# ============================================================
# Top-level Targets
# ============================================================
.PHONY: all tp dp comm clean clean-tp clean-dp clean-comm print_sm

all: $(DIST_LIB_A) $(DIST_LIB_SO)
	@echo "\n[SUCCESS] dist build complete (sm_$(SM_ARCH)):"
	@echo "   $(DIST_LIB_A)"
	@echo "   $(DIST_LIB_SO)"

tp:   $(TP_OBJS)
	@echo "[OK] Tensor-Parallelism objects built."

dp:   $(DP_OBJS)
	@echo "[OK] Data-Parallel objects built."

comm: $(COMM_OBJS)
	@echo "[OK] Communication objects built."

# ---- libdist.a (combined static) ----
$(DIST_LIB_A): $(ALL_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo "\n[AR] Creating combined static library: $@"
	$(AR) rcs $@ $(ALL_OBJS)
	@echo "[SUCCESS] $(notdir $@) built."

# ---- libdist.so (combined shared) ----
$(DIST_LIB_SO): $(ALL_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo "\n[LINK] Creating combined shared library: $@"
	$(CXX) -fPIC -o $@ $(ALL_OBJS) $(DIST_LDFLAGS) $(DIST_LDLIBS)
	@echo "[SUCCESS] $(notdir $@) built."

# ============================================================
# Compile rules
# ============================================================
# Tensor-Parallelism C++
$(BUILD_DIR)/$(TP_DIR)/%.o: $(TP_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "[CXX  | tp]   $<"
	$(CXX) $(CXXFLAGS) $(TP_INCLUDES) -c $< -o $@

# Tensor-Parallelism CUDA
$(BUILD_DIR)/$(TP_DIR)/%.o: $(TP_DIR)/%.cu
	@mkdir -p $(@D)
	@echo "[CUDA | tp]   $<"
	$(NVCC) $(NVCCFLAGS) $(TP_INCLUDES) -c $< -o $@

# Data-Parallel C++
$(BUILD_DIR)/$(DP_DIR)/%.o: $(DP_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "[CXX  | dp]   $<"
	$(CXX) $(CXXFLAGS) $(DP_INCLUDES) -c $< -o $@

# Communication C++
$(BUILD_DIR)/$(COMM_DIR)/%.o: $(COMM_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "[CXX  | comm] $<"
	$(CXX) $(CXXFLAGS) $(COMM_INCLUDES) -c $< -o $@

# ============================================================
# Clean Targets
# ============================================================
clean: clean-tp clean-dp clean-comm
	@echo "[CLEAN] Removing combined libraries and build dir..."
	rm -rf $(LIB_DIR) $(BUILD_DIR)

clean-tp:
	@echo "[CLEAN] Tensor-Parallelism objects"
	rm -rf $(BUILD_DIR)/$(TP_DIR)

clean-dp:
	@echo "[CLEAN] Data-Parallel objects"
	rm -rf $(BUILD_DIR)/$(DP_DIR)

clean-comm:
	@echo "[CLEAN] Communication objects"
	rm -rf $(BUILD_DIR)/$(COMM_DIR)

# ============================================================
# Utility
# ============================================================
print_sm:
	@echo "Detected Compute Capability: sm_$(SM_ARCH)"
