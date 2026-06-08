# ============================================================
# Tensor-Parallelism Sub-Module Makefile (stand-alone)
# ============================================================
# Builds: ../lib/tensor-parallelism/libdtensor.{a,so}
#
# Targets:
#   make [all] / make lib   Build libdtensor.{a,so}
#   make clean              Remove build artifacts
#   make print_sm           Print detected GPU compute capability
#
# Auto-detected (override on command line):
#   nvcc / CUDA_HOME      via `which nvcc`
#   SM_ARCH               via `nvidia-smi`
#
# Override examples:
#   make SM_ARCH=120
#   make CXX=/path/to/mpic++
# ============================================================

SHELL := /bin/bash
AR    := ar

# ---- Host compiler (MPI wrapper) ----
CXX ?= $(shell which mpic++ 2>/dev/null)
ifeq ($(strip $(CXX)),)
  CXX := /usr/bin/mpic++
endif

# ---- CUDA auto-detect ----
CUDA_NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
  $(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif
NVCC      := $(CUDA_NVCC)
CUDA_HOME := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_HOME)/include
CUDA_LIB  := $(CUDA_HOME)/lib64

# ---- GPU compute capability auto-detect ----
SM_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(strip $(SM_ARCH)),)
  SM_ARCH := 86
endif

# ---- Compile / link flags ----
CXXFLAGS  += -fPIC -std=c++2a -O3 -g -DWITH_CUDA
NVCC_FLAGS = -std=c++17 -Xcompiler -fPIC -O3 -g -arch=sm_$(SM_ARCH) -ccbin=$(CXX)

# ---- Tensor-Implementations headers (sibling of dist/) ----
TENSOR_IMPL_DIR := $(CURDIR)/../../Tensor-Implementations

INCLUDES = \
    -I. \
    -I./tensor \
    -I./dnn \
    -I$(TENSOR_IMPL_DIR)/include \
    -I$(CUDA_INC)

# ---- Output directory ----
OBJDIR := ../lib/tensor-parallelism

# ---- Sources ----
LIB_CPP_SRCS = \
    tensor/dtensor.cpp \
    tensor/device_mesh.cpp \
    tensor/placement.cpp

LIB_CUDA_SRCS = \
    dnn/dist_grad_norm_kernels.cu \
    dnn/EntropyKernels.cu \
    tensor/headtail_kernel.cu \
    tensor/fused_transpose_kernel.cu

LIB_CPP_OBJS  = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LIB_CPP_SRCS))
LIB_CUDA_OBJS = $(patsubst %.cu,$(OBJDIR)/%.o,$(LIB_CUDA_SRCS))
LIB_OBJS      = $(LIB_CPP_OBJS) $(LIB_CUDA_OBJS)

LIB_A  = $(OBJDIR)/libdtensor.a
LIB_SO = $(OBJDIR)/libdtensor.so

SHARED_LDFLAGS = -shared -L$(CUDA_LIB) -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz

.PHONY: all lib clean print_sm

all: lib

lib: $(LIB_A) $(LIB_SO)

$(LIB_A): $(LIB_OBJS)
	@echo "\n[AR] Creating static library: $@"
	$(AR) rcs $@ $(LIB_OBJS)
	@echo "[SUCCESS] $(notdir $(LIB_A)) built (sm_$(SM_ARCH))."

$(LIB_SO): $(LIB_OBJS)
	@echo "\n[LINK] Creating shared library: $@"
	$(CXX) $(CXXFLAGS) -shared -o $@ $(LIB_OBJS) $(SHARED_LDFLAGS)
	@echo "[SUCCESS] $(notdir $(LIB_SO)) built (sm_$(SM_ARCH))."

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "[COMPILE] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "[COMPILE CUDA] $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "[CLEAN] Removing TP object files and libraries..."
	rm -rf $(OBJDIR)

print_sm:
	@echo "Detected Compute Capability: sm_$(SM_ARCH)"
