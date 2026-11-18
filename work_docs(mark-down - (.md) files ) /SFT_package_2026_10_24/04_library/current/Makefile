# =============================================================================
# Configuration
# =============================================================================
SHELL  := /bin/bash
CXX    := g++
NVCC   := $(shell which nvcc 2>/dev/null)

# --- Tensor paths ---
SRCDIR  := src
OBJDIR  := lib/objects
LIBDIR  := lib
TARGET_A  := $(LIBDIR)/libtensor.a
TARGET_SO := $(LIBDIR)/libtensor.so

# --- BluBridge-BLAS paths (sub-directory of this repo) ---
BLAS_DIR       := BluBridge-BLAS
BLAS_SRCDIR    := $(BLAS_DIR)/src
# Isolated from tensor objects
BLAS_OBJDIR    := $(OBJDIR)/blublas
BLAS_TARGET_SO := $(LIBDIR)/libblublas.so

# =============================================================================
# CUDA auto-detection
# =============================================================================
CUDA_NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_NVCC),)
  $(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif

# Evaluate nvidia-smi at most once. Using `?=` would make this recursive and
# re-run nvidia-smi on every reference to TENSOR_NVCCFLAGS / BLAS_NVCCFLAGS.
ifeq ($(origin SM_ARCH),undefined)
  SM_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
endif
ifeq ($(strip $(SM_ARCH)),)
  ifeq ($(filter clean% help print_sm,$(MAKECMDGOALS)),)
    $(error Could not detect the GPU architecture (nvidia-smi failed or found no GPU). Pass it explicitly, e.g. SM_ARCH=90 for H100/H200 or SM_ARCH=120 for RTX 50xx)
  endif
endif

# Datacenter Blackwell (B100/B200/GB200/B300/GB300, compute_cap 10.0/10.3) only
# exposes tcgen05 (5th-gen tensor-core MMA) and its TMA/mbarrier companions
# under the family-specific 'a' architecture variant -- plain sm_100/sm_103
# is rejected by ptxas for those instructions. The naive fix, feeding
# "-arch=sm_100a"/"sm_103a" through the normal -arch=sm_$(SM_ARCH) shorthand,
# does NOT work: nvcc 13.0 silently lowers that shorthand's PTX .target back
# to the plain (non-'a') virtual arch, and ptxas then rejects tcgen05 again.
# The 'a' has to be carried on the virtual arch explicitly via -gencode.
ARCH_NVCC_FLAGS := -arch=sm_$(SM_ARCH)
ifeq ($(SM_ARCH),90)
  # Hopper (H100/H200): the 'a' variant enables wgmma/TMA PTX. Carried on the
  # virtual arch via -gencode for the same reason as the Blackwell cases below.
  ARCH_NVCC_FLAGS := -gencode arch=compute_90a,code=sm_90a
endif
ifeq ($(SM_ARCH),100)
  ARCH_NVCC_FLAGS := -gencode arch=compute_100a,code=sm_100a
endif
ifeq ($(SM_ARCH),103)
  ARCH_NVCC_FLAGS := -gencode arch=compute_103a,code=sm_103a
endif

# --- Per-file architecture selection ----------------------------------------
# The repo contains B300-only kernels (tcgen05 / TMA / clusterlaunchcontrol PTX)
# that ptxas assembles ONLY for compute_100a/103a. Feeding them the machine arch
# (e.g. sm_86 / sm_90) makes ptxas abort. So those sources are compiled for the
# Blackwell virtual archs (cross-compiled into the binary; they only *run* on a
# B300), while every OTHER source is compiled for the machine's ARCH_NVCC_FLAGS.
# This is what lets the repo compile on ANY GPU despite containing B300 code.
# A source is "Blackwell" if its path/name contains one of these markers:
GENCODE_BLACKWELL := -gencode arch=compute_100a,code=sm_100a \
                     -gencode arch=compute_103a,code=sm_103a \
                     -Xptxas --register-usage-level=10
BLACKWELL_MARKERS := SM100 SM103 B300 sm100 sm103 b300
is_blackwell_src  = $(strip $(foreach m,$(BLACKWELL_MARKERS),$(findstring $(m),$(1))))

# Same story for the Hopper-only wgmma/cluster-GEMM dispatcher family (the
# BF16/SM90 and FP16/SM90 batched_gemm trees): they use PTX (e.g.
# cp.async.bulk.wait_group) that ptxas only assembles for sm_90+, so building
# on a non-Hopper machine (e.g. this box's sm_86) must still cross-compile
# them for compute_90a rather than feed them the machine arch. Marker is
# uppercase-only ("SM90") -- deliberately does NOT match the many lowercase
# "_sm90.cu"-suffixed files elsewhere in the tree, which already compile fine
# under the machine arch via internal __CUDA_ARCH__ guards and must keep doing
# so (they are not pure Hopper-only cross-compiles like this family).
GENCODE_HOPPER := -gencode arch=compute_90a,code=sm_90a
HOPPER_MARKERS := SM90
is_hopper_src   = $(strip $(foreach m,$(HOPPER_MARKERS),$(findstring $(m),$(1))))

# GENCODE_FOR(src): Blackwell/Hopper virtual arch for their respective
# cross-compile-only sources, else the machine arch.
GENCODE_FOR = $(if $(call is_blackwell_src,$(1)),$(GENCODE_BLACKWELL),$(if $(call is_hopper_src,$(1)),$(GENCODE_HOPPER),$(ARCH_NVCC_FLAGS)))

CUDA_ROOT := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_ROOT)/include
CUDA_LIB  := $(CUDA_ROOT)/lib64

CUDA_TARGETS_LIB := $(shell find $(CUDA_ROOT)/targets -type d -name "lib" 2>/dev/null | head -1)
ifeq ($(CUDA_TARGETS_LIB),)
  CUDA_TARGETS_LIB := $(CUDA_LIB)
endif

# NVTX (CUDA 12+ is header-only; path is best-effort)
NVTX_INC := /opt/nvidia/nsight-compute/2024.1.0/host/target-linux-x64/nvtx/include/
ifeq ($(NVTX_INC),)
  NVTX_INC := $(CUDA_INC)
endif
NVTX_LIB := /opt/nvidia/nsight-compute/2024.1.0/host/target-linux-x64/nvtx/
ifeq ($(NVTX_LIB),)
  NVTX_LIB := $(CUDA_LIB)
endif

# =============================================================================
# BLAS Backend Selection  (default: blublas | WITH_CUBLAS=1 -> cublas)
# =============================================================================
BLAS_STAMP := $(OBJDIR)/.blas_backend

ifeq ($(WITH_CUBLAS),1)
  BLAS_BACKEND := cublas
else
  BLAS_BACKEND := blublas
endif

LAST_BLAS := $(shell cat $(BLAS_STAMP) 2>/dev/null)

# =============================================================================
# Compiler / Linker Flags -- Tensor
# =============================================================================
TENSOR_CPPFLAGS  := -Iinclude -I../Profiler/include -I../dist/Distributed_Checkpointing/include -I$(CUDA_INC) -I$(NVTX_INC) -DWITH_CUDA
ifeq ($(PROFILER),1)
  TENSOR_CPPFLAGS += -DAUTOGRAD_PROFILER_ENABLED
endif
ifeq ($(BLAS_BACKEND),blublas)
  TENSOR_CPPFLAGS += -DWITH_MYBLAS -I$(BLAS_DIR)/include
endif

# -MMD -MP: emit a .d beside each .o listing the (non-system) headers it depends
# on, so editing a header rebuilds exactly the objects that include it. -MP adds
# phony targets so deleting a header does not wedge the build. nvcc supports both
# natively. The .d files are pulled in at the bottom of this file.
DEPFLAGS         := -MMD -MP
TENSOR_CXXFLAGS  := -std=c++2a -fPIC -Wall -Wextra -O3 -fopenmp -mavx2 -mfma -mf16c $(FTIME_FLAG) $(DEPFLAGS)
## Arch is applied per-recipe (GENCODE_FOR / ARCH_NVCC_FLAGS), NOT baked in here,
## so Blackwell and non-Blackwell sources can target different archs.
## No --use_fast_math (removed 2026-10-31). It implied --ftz=true --prec-div=false --prec-sqrt=false and turned
## sinf / cosf / expf / logf / powf into the fast intrinsics for EVERY kernel: the RoPE table (2x Hugging Face's
## error), the loss's exp / log, the norms' and optimizers' sqrt / division. Kernels that want a fast exp or reciprocal
## call it explicitly (attention: ex2.approx / rcp.approx PTX; SwiGLU: __expf), as PyTorch, TransformerEngine (a
## per-file list) and vLLM (per file) do. SFT work doc 45.114.
TENSOR_NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" \
                    -O3 --expt-relaxed-constexpr $(DEPFLAGS)
## The one exception (2026-10-31, measured): the loss kernels keep --use_fast_math, per file as TransformerEngine does.
## Without it the loss forward's per-logit expf is the exact (slow) one: forward 0.81 -> 1.13 ms (+37%) on the 4k SFT batch
## with no precision gain for bf16 logits (the training path; errors equal to +-1%). With it: the old speed (fwd + bwd
## 2.239 vs 2.237 ms) and the old precision (fp32 forward <= 1.1e-7 relative). An explicit __expf in the forward alone
## recovered only half (2.410 ms). SFT work doc 45.121.
$(OBJDIR)/src/Kernels/cuda/loss/LossKernels.o: TENSOR_NVCCFLAGS += --use_fast_math

TENSOR_RPATH   := -Xlinker -rpath -Xlinker '$$ORIGIN/lib' \
                   -Xlinker -rpath -Xlinker '$$ORIGIN/../../Profiler/lib'
TENSOR_LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) -L../Profiler/lib $(TENSOR_RPATH)
TENSOR_LDLIBS  := -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml -lprofiler
ifeq ($(BLAS_BACKEND),blublas)
  TENSOR_LDLIBS += -lblublas
endif

# Object files that differ between backends (stamp-driven recompile)
BLAS_OBJ_GENMATMUL := $(OBJDIR)/src/Kernels/cuda/matmul/GenMatmul.o
BLAS_OBJ_BACKWARD  := $(OBJDIR)/src/Kernels/cuda/matmul/MatmulBackward.o

# =============================================================================
# Compiler / Linker Flags -- BluBLAS
# Reuses CUDA_ROOT/INC/LIB/SM_ARCH from the shared auto-detection block above.
# =============================================================================
BLAS_CCCL_INC := $(shell find $(CUDA_ROOT)/targets -type d -name "cccl" 2>/dev/null | head -1)
ifeq ($(BLAS_CCCL_INC),)
  BLAS_CCCL_INC := $(CUDA_INC)
endif

BLAS_CPPFLAGS  := -I$(BLAS_DIR)/include \
                  -Iinclude \
                  -I../Profiler/include \
                  -I$(CUDA_INC) \
                  -I$(BLAS_DIR)/external/cutlass/include \
                  -I$(BLAS_CCCL_INC) \
                  -DWITH_CUDA
BLAS_CXXFLAGS  := -std=c++17 -fPIC -Wall -O3 -DNDEBUG $(DEPFLAGS)
BLAS_NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" \
                  -O3 -DNDEBUG -Xptxas -O3 \
                  --expt-relaxed-constexpr \
                  -I$(BLAS_DIR)/external/cutlass/include \
                  -I$(BLAS_CCCL_INC) \
                  $(DEPFLAGS)

BLAS_RPATH   := -Xlinker -rpath -Xlinker '$$ORIGIN' \
                -Xlinker -rpath -Xlinker '$$ORIGIN/lib' \
                -Xlinker -rpath -Xlinker '$$ORIGIN/../../Profiler/lib'
BLAS_LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) -L../Profiler/lib $(BLAS_RPATH)
BLAS_LDLIBS  := -lcudart -lcublas -lcublasLt -lprofiler

# =============================================================================
# Source / Object Discovery -- Tensor
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES  := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
    src/Views/ContiguousKernel.cu \
    src/Views/arch/ContiguousKernel_sm90.cu \
    src/UnaryOps/cuda/ReductionImplGPU.cu

CU_SOURCES_REGULAR    := $(filter-out $(CU_SOURCES_FOR_DLINK), $(CU_SOURCES))

OBJECTS_FROM_CPP        := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU_REGULAR := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_REGULAR))
OBJECTS_FOR_DLINK       := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_FOR_DLINK))
DEVICE_LINK_OBJ         := $(OBJDIR)/device_link.o
TENSOR_ALL_OBJECTS      := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU_REGULAR) $(OBJECTS_FOR_DLINK)

# =============================================================================
# Source / Object Discovery -- BluBLAS
# =============================================================================
BLAS_CPP_SOURCES := $(shell find $(BLAS_SRCDIR) -name '*.cpp' 2>/dev/null)
BLAS_CU_SOURCES  := $(shell find $(BLAS_SRCDIR) -name '*.cu'  2>/dev/null)

BLAS_OBJECTS_FROM_CPP := $(patsubst $(BLAS_SRCDIR)/%.cpp,$(BLAS_OBJDIR)/%.o,$(BLAS_CPP_SOURCES))
BLAS_OBJECTS_FROM_CU  := $(patsubst $(BLAS_SRCDIR)/%.cu,$(BLAS_OBJDIR)/%.o,$(BLAS_CU_SOURCES))
BLAS_ALL_OBJECTS      := $(BLAS_OBJECTS_FROM_CPP) $(BLAS_OBJECTS_FROM_CU)

# =============================================================================
# Stamp-driven backend invalidation (tensor side only)
# =============================================================================
ifneq ($(LAST_BLAS),$(BLAS_BACKEND))
  .PHONY: $(BLAS_OBJ_GENMATMUL) $(BLAS_OBJ_BACKWARD) $(TARGET_SO)
endif

# =============================================================================
# Top-level Targets
# =============================================================================
.PHONY: all tensor blas rebuild rebuild-tensor rebuild-blas \
        clean clean-tensor clean-blas print_sm help write-blas-stamp

all: blas tensor
	@echo "\n\nAll libraries are up-to-date."
	@echo "   $(BLAS_TARGET_SO)"
	@echo "   $(TARGET_SO)"
	@echo "   $(TARGET_A)"

# =============================================================================
# TENSOR target -- libtensor.so + libtensor.a
# =============================================================================
tensor: $(TARGET_SO) $(TARGET_A)
	@echo "\n\nlibtensor is up-to-date."

$(TARGET_SO): $(TENSOR_ALL_OBJECTS) $(DEVICE_LINK_OBJ) | write-blas-stamp
	@echo "\n\n--- Creating shared library: $@"
	@mkdir -p $(LIBDIR)
	$(NVCC) -shared $(TENSOR_NVCCFLAGS) $(ARCH_NVCC_FLAGS) $(TENSOR_ALL_OBJECTS) $(DEVICE_LINK_OBJ) \
	        $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) -o $@

$(TARGET_A): $(TENSOR_ALL_OBJECTS)
	@echo "\n\n--- Creating static library: $@"
	@mkdir -p $(LIBDIR)
	ar rcs $@ $(TENSOR_ALL_OBJECTS)

# Device-link step (separable compilation objects)
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "\n\n--- Linking CUDA device code..."
	@mkdir -p $(@D)
	$(NVCC) $(TENSOR_NVCCFLAGS) $(ARCH_NVCC_FLAGS) -dlink $^ -o $@

# Tensor: C++ objects
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX  | tensor]: $<"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -c $< -o $@

# Tensor: CUDA separable-compilation objects
$(OBJECTS_FOR_DLINK): NVCC_DEVICE_FLAGS = -dc
$(OBJECTS_FOR_DLINK): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA -dc | tensor]: $<"
	$(NVCC) $(TENSOR_CPPFLAGS) $(TENSOR_NVCCFLAGS) $(ARCH_NVCC_FLAGS) $(NVCC_DEVICE_FLAGS) -c $< -o $@

# Tensor: regular CUDA objects
$(OBJECTS_FROM_CU_REGULAR): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA $(if $(call is_blackwell_src,$<),Blackwell,sm_$(SM_ARCH)) | tensor]: $<"
	$(NVCC) $(TENSOR_CPPFLAGS) $(TENSOR_NVCCFLAGS) $(call GENCODE_FOR,$<) -c $< -o $@

write-blas-stamp:
	@mkdir -p $(OBJDIR)
	@echo "$(BLAS_BACKEND)" > $(BLAS_STAMP)

# =============================================================================
# BLAS target -- libblublas.so
# =============================================================================
blas: $(BLAS_TARGET_SO)
	@echo "\n\nlibblublas is up-to-date."

$(BLAS_TARGET_SO): $(BLAS_ALL_OBJECTS)
	@echo "\n\n--- Creating shared library: $@"
	@mkdir -p $(LIBDIR)
	$(NVCC) -shared $(BLAS_NVCCFLAGS) $(BLAS_ALL_OBJECTS) \
	        $(BLAS_LDFLAGS) $(BLAS_LDLIBS) -o $@

# BluBLAS: C++ objects
$(BLAS_OBJDIR)/%.o: $(BLAS_SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX  | blublas]: $<"
	$(CXX) $(BLAS_CPPFLAGS) $(BLAS_CXXFLAGS) -c $< -o $@

# BluBLAS: CUDA objects
$(BLAS_OBJDIR)/%.o: $(BLAS_SRCDIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA $(if $(call is_blackwell_src,$<),Blackwell,sm_$(SM_ARCH)) | blublas]: $<"
	$(NVCC) $(BLAS_CPPFLAGS) $(BLAS_NVCCFLAGS) $(call GENCODE_FOR,$<) -c $< -o $@

# =============================================================================
# precheck tool -- precision regression harness for kernels
# =============================================================================
PRECHECK_DIR     := tools/precheck
SHARED_DIR       := tools/_shared
PRECHECK_SRC     := $(wildcard $(PRECHECK_DIR)/*.cpp) $(wildcard $(SHARED_DIR)/*.cpp)
PRECHECK_BIN     := $(PRECHECK_DIR)/precheck
PRECHECK_LDLIBS  := $(TENSOR_LDLIBS) -lcuda -lyaml-cpp -lstdc++fs

.PHONY: precheck precheck-run precheck-list precheck-self-test \
        precheck-bless precheck-bless-perf precheck-deep clean-precheck

precheck: $(TARGET_SO) $(PRECHECK_BIN)
	@echo "\nprecheck binary built: $(PRECHECK_BIN)"

$(PRECHECK_BIN): $(PRECHECK_SRC) $(TARGET_SO)
	@echo "\n--- Building precheck (sm_$(SM_ARCH)) ---"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $(PRECHECK_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(PRECHECK_LDLIBS)

precheck-run: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) $(ARGS)

precheck-list: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) --list

precheck-self-test: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) --self-test

precheck-bless: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) --bless $(ARGS)

precheck-bless-perf: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) --bless-perf $(ARGS)

precheck-deep: precheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(PRECHECK_BIN) --deep $(ARGS)

clean-precheck:
	rm -f $(PRECHECK_BIN)

# =============================================================================
# tensor-api hand-written tests (Tests/tensor-api/*.cpp)
# Each .cpp is a standalone binary built on Tests/_harness/test_main.h.
# =============================================================================
TENSOR_API_DIR     := Tests/tensor-api
HARNESS_SHARED_SRC := $(SHARED_DIR)/assert.cpp $(SHARED_DIR)/cli.cpp \
                      $(SHARED_DIR)/compare.cpp $(SHARED_DIR)/golden_io.cpp \
                      $(SHARED_DIR)/hash.cpp $(SHARED_DIR)/input_gen.cpp \
                      $(SHARED_DIR)/json.cpp $(SHARED_DIR)/seed.cpp \
                      $(SHARED_DIR)/tolerance.cpp $(SHARED_DIR)/ulp.cpp
TENSOR_API_SRCS    := $(wildcard $(TENSOR_API_DIR)/*.cpp)
TENSOR_API_BINS    := $(TENSOR_API_SRCS:.cpp=)

.PHONY: tensor-api-test tensor-api-list clean-tensor-api

tensor-api-test: $(TARGET_SO) $(TENSOR_API_BINS)
	@fail=0; for b in $(TENSOR_API_BINS); do \
		echo "\n--- Running $$b ---"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo "\nTENSOR-API: FAIL"; exit 1; fi; \
	echo "\nTENSOR-API: ALL PASS"

tensor-api-list:
	@for b in $(TENSOR_API_BINS); do \
		echo "$$b:"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b --list 2>/dev/null | sed 's/^/  /'; \
	done

$(TENSOR_API_DIR)/%: $(TENSOR_API_DIR)/%.cpp $(HARNESS_SHARED_SRC) $(TARGET_SO)
	@echo "Building tensor-api test: $@"
	$(CXX) -I. $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $< $(HARNESS_SHARED_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda -lstdc++fs

clean-tensor-api:
	rm -f $(TENSOR_API_BINS)

# =============================================================================
# autocheck tool -- autograd finite-difference gradcheck for differentiable ops
# =============================================================================
AUTOCHECK_DIR    := tools/autocheck
AUTOCHECK_SRC    := $(wildcard $(AUTOCHECK_DIR)/*.cpp) $(HARNESS_SHARED_SRC)
AUTOCHECK_BIN    := $(AUTOCHECK_DIR)/autocheck

.PHONY: autocheck autocheck-run autocheck-list clean-autocheck

autocheck: $(TARGET_SO) $(AUTOCHECK_BIN)
	@echo "\nautocheck binary built: $(AUTOCHECK_BIN)"

$(AUTOCHECK_BIN): $(AUTOCHECK_SRC) $(TARGET_SO)
	@echo "\n--- Building autocheck ---"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $(AUTOCHECK_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda -lstdc++fs

autocheck-run: autocheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(AUTOCHECK_BIN) $(ARGS)

autocheck-list: autocheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(AUTOCHECK_BIN) --list

clean-autocheck:
	rm -f $(AUTOCHECK_BIN)

# =============================================================================
# modules hand-written tests (Tests/modules/*.cpp)
# =============================================================================
MODULES_DIR     := Tests/modules
MODULES_SRCS    := $(wildcard $(MODULES_DIR)/*.cpp)
MODULES_BINS    := $(MODULES_SRCS:.cpp=)

.PHONY: modules-test modules-list clean-modules

modules-test: $(TARGET_SO) $(MODULES_BINS)
	@fail=0; for b in $(MODULES_BINS); do \
		echo "\n--- Running $$b ---"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo "\nMODULES: FAIL"; exit 1; fi; \
	echo "\nMODULES: ALL PASS"

modules-list:
	@for b in $(MODULES_BINS); do \
		echo "$$b:"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b --list 2>/dev/null | sed 's/^/  /'; \
	done

$(MODULES_DIR)/%: $(MODULES_DIR)/%.cpp $(HARNESS_SHARED_SRC) $(TARGET_SO)
	@echo "Building modules test: $@"
	$(CXX) -I. $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $< $(HARNESS_SHARED_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda -lstdc++fs

clean-modules:
	rm -f $(MODULES_BINS)

# =============================================================================
# memcheck tool -- memory-lifecycle scenarios (allocator, views, pinned, peak)
# =============================================================================
MEMCHECK_DIR    := tools/memcheck
MEMCHECK_SRC    := $(wildcard $(MEMCHECK_DIR)/*.cpp) $(HARNESS_SHARED_SRC)
MEMCHECK_BIN    := $(MEMCHECK_DIR)/memcheck

.PHONY: memcheck memcheck-run memcheck-list clean-memcheck

memcheck: $(TARGET_SO) $(MEMCHECK_BIN)
	@echo "\nmemcheck binary built: $(MEMCHECK_BIN)"

$(MEMCHECK_BIN): $(MEMCHECK_SRC) $(TARGET_SO)
	@echo "\n--- Building memcheck ---"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $(MEMCHECK_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda -lstdc++fs

memcheck-run: memcheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(MEMCHECK_BIN) $(ARGS)

memcheck-list: memcheck
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$(MEMCHECK_BIN) --list

clean-memcheck:
	rm -f $(MEMCHECK_BIN)

# =============================================================================
# training tests (Tests/training/*.cpp) - nightly only, end-to-end convergence
# =============================================================================
TRAINING_DIR    := Tests/training
TRAINING_SRCS   := $(wildcard $(TRAINING_DIR)/*.cpp)
TRAINING_BINS   := $(TRAINING_SRCS:.cpp=)

.PHONY: training-test training-list clean-training

training-test: $(TARGET_SO) $(TRAINING_BINS)
	@fail=0; for b in $(TRAINING_BINS); do \
		echo "\n--- Running $$b ---"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b || fail=1; \
	done; \
	if [ $$fail -ne 0 ]; then echo "\nTRAINING: FAIL"; exit 1; fi; \
	echo "\nTRAINING: ALL PASS"

training-list:
	@for b in $(TRAINING_BINS); do \
		echo "$$b:"; \
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./$$b --list 2>/dev/null | sed 's/^/  /'; \
	done

$(TRAINING_DIR)/%: $(TRAINING_DIR)/%.cpp $(HARNESS_SHARED_SRC) $(TARGET_SO)
	@echo "Building training test: $@"
	$(CXX) -I. $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o $@ $< $(HARNESS_SHARED_SRC) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda -lstdc++fs

clean-training:
	rm -f $(TRAINING_BINS)

# =============================================================================
# Aggregate targets
# =============================================================================
.PHONY: check-all
check-all: tensor-api-test autocheck-run modules-test memcheck-run
	@echo "\n========================================"
	@echo "CHECK-ALL: tensor-api + autocheck + modules + memcheck PASSED"
	@echo "(precheck and training run separately;"
	@echo " precheck needs 'make precheck-bless --shape-tier small' first)"
	@echo "========================================"

# =============================================================================
# run-cuda -- compile a single CUDA test (.cu) against libtensor, run it, then
# delete the executable. For the standalone kernel tests in Tests/kernel_tests/
# that call the public kernel API (rope_*, swiglu_*, rms_norm_*) directly.
# Usage: make run-cuda FILE=Tests/kernel_tests/test_RoPE_fwd.cu [WITH_CUBLAS=1]
# =============================================================================
RUN_CUDA_BIN := run_cuda_runner

.PHONY: run-cuda
run-cuda: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a .cu file to run."; \
		echo "Usage: make run-cuda FILE=Tests/kernel_tests/test_X.cu"; \
		exit 1; \
	fi
	@echo "--- BLAS backend : $(BLAS_BACKEND) ---"
	@echo "--- Compiling [CUDA] (sm_$(SM_ARCH)): $(FILE) ---"
	$(NVCC) $(TENSOR_CPPFLAGS) $(TENSOR_NVCCFLAGS) $(ARCH_NVCC_FLAGS) -o $(RUN_CUDA_BIN) $(FILE) \
	        $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda
	@echo "\n--- Running $(FILE) ---"
	@LD_LIBRARY_PATH=$(LIBDIR):../Profiler/lib:$$LD_LIBRARY_PATH ./$(RUN_CUDA_BIN); \
	status=$$?; \
	rm -f $(RUN_CUDA_BIN); \
	echo "\n--- Removed $(RUN_CUDA_BIN) (exit $$status) ---"; \
	exit $$status

# =============================================================================
# Run / Test helpers
# =============================================================================
.PHONY: run-snippet run-folder
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file to run."; \
		echo "Usage: make run-snippet FILE=path/to/file.cpp [PROFILER=1] [WITH_CUBLAS=1]"; \
		exit 1; \
	fi
	@echo "--- BLAS backend : $(BLAS_BACKEND) ---"
	@echo "--- Compiling   : $(FILE) ---"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o snippet_runner $(FILE) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) -lcuda
	@echo "\n--- Running snippet_runner ---"
ifeq ($(BLAS_BACKEND),blublas)
	LD_LIBRARY_PATH=$(LIBDIR):../Profiler/lib:$$LD_LIBRARY_PATH ./snippet_runner
else
	LD_LIBRARY_PATH=$(LIBDIR):../Profiler/lib:$$LD_LIBRARY_PATH ./snippet_runner
endif
	@echo ""

run-folder: $(TARGET_SO)
	@if [ -z "$(FOLDER)" ]; then \
		echo "ERROR: Please specify a folder."; \
		echo "Usage: make run-folder FOLDER=Tests/path/to/folder [WITH_CUBLAS=1]"; \
		exit 1; \
	fi
	@echo "--- BLAS backend : $(BLAS_BACKEND) ---"
	@echo "--- Running all tests in: $(FOLDER) ---"
	@for file in $(FOLDER)/*.cpp; do \
		[ -f "$$file" ] || continue; \
		echo "\n---------------------------------------------------------"; \
		echo "Running: $$file"; \
		echo "---------------------------------------------------------"; \
		$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o snippet_runner "$$file" \
		       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS) || exit 1; \
		LD_LIBRARY_PATH=$(LIBDIR):../Profiler/lib:$$LD_LIBRARY_PATH ./snippet_runner || exit 1; \
	done
	@rm -f snippet_runner
	@echo "\nAll tests in $(FOLDER) passed!"

# =============================================================================
# Clean / Rebuild
# =============================================================================
rebuild:
	@$(MAKE) clean && $(MAKE) all

rebuild-tensor:
	@$(MAKE) clean-tensor && $(MAKE) tensor

rebuild-blas:
	@$(MAKE) clean-blas && $(MAKE) blas

clean: clean-tensor clean-blas clean-precheck clean-tensor-api clean-autocheck clean-modules clean-memcheck clean-training
	@echo "Full clean done."

clean-tensor:
	@echo "--- Cleaning tensor build artifacts ---"
	rm -rf $(OBJDIR)/src $(OBJDIR)/device_link.o $(OBJDIR)/.blas_backend \
	       $(TARGET_A) $(TARGET_SO)

clean-blas:
	@echo "--- Cleaning BluBLAS build artifacts ---"
	rm -rf $(BLAS_OBJDIR) $(BLAS_TARGET_SO)

# =============================================================================
# Utility
# =============================================================================
print_sm:
	@echo "------Detected Compute Capability: $(SM_ARCH)------"

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make [all]                    Build libblublas.so + libtensor.so + libtensor.a"
	@echo "  make tensor                   Build/update libtensor.so + libtensor.a only"
	@echo "  make blas                     Build/update libblublas.so only"
	@echo "  make run-snippet FILE=<f>     Compile and run a single .cpp against libtensor"
	@echo "  make run-cuda    FILE=<f>     Compile + run a single .cu kernel test, then delete the binary"
	@echo "  make run-folder  FOLDER=<d>   Run all .cpp tests in a directory"
	@echo "  make precheck                 Build the kernel precision regression tool"
	@echo "  make precheck-bless           Compute + save golden references (run BEFORE first test)"
	@echo "  make precheck-bless-perf      Measure + save perf baselines (run BEFORE first test)"
	@echo "  make precheck-run ARGS=...    Run real precision + perf tests"
	@echo "  make precheck-list            List registered kernels in tools/precheck/kernels.yaml"
	@echo "  make precheck-self-test       Run harness self-test against inline broken impls"
	@echo "  make tensor-api-test          Build + run hand-written Tests/tensor-api/ suite"
	@echo "  make tensor-api-list          List registered tests in each tensor-api binary"
	@echo "  make autocheck                Build the autograd gradcheck tool"
	@echo "  make autocheck-run ARGS=...   Run finite-difference gradcheck on registered ops"
	@echo "  make autocheck-list           List registered differentiable ops in autocheck"
	@echo "  make modules-test             Build + run nn-module tests under Tests/modules/"
	@echo "  make modules-list             List registered tests in each modules binary"
	@echo "  make memcheck                 Build the memory-lifecycle scenario tool"
	@echo "  make memcheck-run ARGS=...    Run memcheck scenarios"
	@echo "  make memcheck-list            List registered memcheck scenarios"
	@echo "  make training-test            Build + run end-to-end training tests (nightly)"
	@echo "  make training-list            List registered training tests"
	@echo "  make check-all                Run every mandatory CI gate (no training)"
	@echo "  make rebuild                  clean + all"
	@echo "  make rebuild-tensor           clean-tensor + tensor"
	@echo "  make rebuild-blas             clean-blas + blas"
	@echo "  make clean                    Remove all build artifacts (tensor + blas)"
	@echo "  make clean-tensor             Remove only tensor build artifacts"
	@echo "  make clean-blas               Remove only BluBLAS build artifacts"
	@echo "  make print_sm                 Print detected GPU compute capability"
	@echo ""
	@echo "Flags:"
	@echo "  WITH_CUBLAS=1                 Link libtensor against cuBLAS (default: blublas)"
	@echo "  PROFILER=1                    Enable autograd profiler"
	@echo "  SM_ARCH=<nn>                  Override GPU SM arch (default: auto-detected)"
	@echo ""
	@echo "Output libraries:"
	@echo "  $(TARGET_SO)"
	@echo "  $(TARGET_A)"
	@echo "  $(BLAS_TARGET_SO)"
	@echo ""


# =============================================================================
# Auto-generated header dependencies
# =============================================================================
# One .d per object, written beside it by -MMD (the compiler derives the .d name
# from -o). Leading '-' so the first build, before any .d exists, is silent.
# The clean targets remove $(OBJDIR), taking the .d files with them.
-include $(TENSOR_ALL_OBJECTS:.o=.d)
-include $(BLAS_OBJECTS_FROM_CPP:.o=.d)
-include $(BLAS_OBJECTS_FROM_CU:.o=.d)
