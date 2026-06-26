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
  SM_ARCH := 86
endif

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
# BLAS Backend Selection  (default: cublas | WITH_BLUBLAS=1 -> blublas)
# =============================================================================
BLAS_STAMP := $(OBJDIR)/.blas_backend

ifeq ($(WITH_BLUBLAS),1)
  BLAS_BACKEND := blublas
else
  BLAS_BACKEND := cublas
endif

LAST_BLAS := $(shell cat $(BLAS_STAMP) 2>/dev/null)

# =============================================================================
# Compiler / Linker Flags -- Tensor
# =============================================================================
TENSOR_CPPFLAGS  := -Iinclude -I$(CUDA_INC) -I$(NVTX_INC) -DWITH_CUDA
ifeq ($(PROFILER),1)
  TENSOR_CPPFLAGS += -DAUTOGRAD_PROFILER_ENABLED
endif
ifeq ($(BLAS_BACKEND),blublas)
  TENSOR_CPPFLAGS += -DWITH_MYBLAS -I$(BLAS_DIR)/include
endif

TENSOR_CXXFLAGS  := -std=c++2a -fPIC -Wall -Wextra -O3 -fopenmp -mavx2 -mfma -mf16c $(FTIME_FLAG)
TENSOR_NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) \
                    --use_fast_math -O3 --expt-relaxed-constexpr

TENSOR_RPATH   := -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
TENSOR_LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) $(TENSOR_RPATH)
TENSOR_LDLIBS  := -lcudart -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml
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
                  -I$(CUDA_INC) \
                  -I$(BLAS_DIR)/external/cutlass/include \
                  -I$(BLAS_CCCL_INC)
BLAS_CXXFLAGS  := -std=c++17 -fPIC -Wall -O3 -DNDEBUG
BLAS_NVCCFLAGS := -std=c++17 -Xcompiler="-fPIC" \
                  -arch=sm_$(SM_ARCH) \
                  -O3 -DNDEBUG -Xptxas -O3 --use_fast_math \
                  --expt-relaxed-constexpr \
                  -I$(BLAS_DIR)/external/cutlass/include \
                  -I$(BLAS_CCCL_INC)

BLAS_RPATH   := -Xlinker -rpath -Xlinker '$$ORIGIN' \
                -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
BLAS_LDFLAGS := -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) $(BLAS_RPATH)
BLAS_LDLIBS  := -lcudart -lcublas -lcublasLt

# =============================================================================
# Source / Object Discovery -- Tensor
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES  := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
    src/Views/ContiguousKernel.cu \
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
	$(NVCC) -shared $(TENSOR_NVCCFLAGS) $(TENSOR_ALL_OBJECTS) $(DEVICE_LINK_OBJ) \
	        $(TENSOR_LDFLAGS) $(TENSOR_LDLIBS) -o $@

$(TARGET_A): $(TENSOR_ALL_OBJECTS)
	@echo "\n\n--- Creating static library: $@"
	@mkdir -p $(LIBDIR)
	ar rcs $@ $(TENSOR_ALL_OBJECTS)

# Device-link step (separable compilation objects)
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "\n\n--- Linking CUDA device code..."
	@mkdir -p $(@D)
	$(NVCC) $(TENSOR_NVCCFLAGS) -dlink $^ -o $@

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
	$(NVCC) $(TENSOR_CPPFLAGS) $(TENSOR_NVCCFLAGS) $(NVCC_DEVICE_FLAGS) -c $< -o $@

# Tensor: regular CUDA objects
$(OBJECTS_FROM_CU_REGULAR): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA | tensor]: $<"
	$(NVCC) $(TENSOR_CPPFLAGS) $(TENSOR_NVCCFLAGS) -c $< -o $@

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
	@echo "Compiling [CUDA | blublas]: $<"
	$(NVCC) $(BLAS_CPPFLAGS) $(BLAS_NVCCFLAGS) -c $< -o $@

# =============================================================================
# Run / Test helpers
# =============================================================================
.PHONY: run-snippet run-folder
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file to run."; \
		echo "Usage: make run-snippet FILE=path/to/file.cpp [PROFILER=1] [WITH_BLUBLAS=1]"; \
		exit 1; \
	fi
	@echo "--- BLAS backend : $(BLAS_BACKEND) ---"
	@echo "--- Compiling   : $(FILE) ---"
	$(CXX) $(TENSOR_CPPFLAGS) $(TENSOR_CXXFLAGS) -o snippet_runner $(FILE) \
	       $(TENSOR_LDFLAGS) -ltensor $(TENSOR_LDLIBS)
	@echo "\n--- Running snippet_runner ---"
ifeq ($(BLAS_BACKEND),blublas)
	LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./snippet_runner
else
	./snippet_runner
endif
	@echo ""

run-folder: $(TARGET_SO)
	@if [ -z "$(FOLDER)" ]; then \
		echo "ERROR: Please specify a folder."; \
		echo "Usage: make run-folder FOLDER=Tests/path/to/folder [WITH_BLUBLAS=1]"; \
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
		LD_LIBRARY_PATH=$(LIBDIR):$$LD_LIBRARY_PATH ./snippet_runner || exit 1; \
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

clean: clean-tensor clean-blas
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
	@echo "  make run-folder  FOLDER=<d>   Run all .cpp tests in a directory"
	@echo "  make rebuild                  clean + all"
	@echo "  make rebuild-tensor           clean-tensor + tensor"
	@echo "  make rebuild-blas             clean-blas + blas"
	@echo "  make clean                    Remove all build artifacts (tensor + blas)"
	@echo "  make clean-tensor             Remove only tensor build artifacts"
	@echo "  make clean-blas               Remove only BluBLAS build artifacts"
	@echo "  make print_sm                 Print detected GPU compute capability"
	@echo ""
	@echo "Flags:"
	@echo "  WITH_BLUBLAS=1                Link libtensor against libblublas (default: cublas)"
	@echo "  PROFILER=1                    Enable autograd profiler"
	@echo "  SM_ARCH=<nn>                  Override GPU SM arch (default: auto-detected)"
	@echo ""
	@echo "Output libraries:"
	@echo "  $(TARGET_SO)"
	@echo "  $(TARGET_A)"
	@echo "  $(BLAS_TARGET_SO)"
	@echo ""

