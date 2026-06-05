# =============================================================================
# Configuration
# =============================================================================
CXX = g++
NVCC = $(shell which nvcc 2>/dev/null)
SRCDIR := src
OBJDIR := lib/objects
LIBDIR := lib
TARGET_A := $(LIBDIR)/libtensor.a
TARGET_SO := $(LIBDIR)/libtensor.so

# =============================================================================
# Build Timing (TIMING=1 to enable per-file timing)
# =============================================================================
TIMING_DIR := $(LIBDIR)/timing
TIMING_LOG := $(TIMING_DIR)/compile_times.log

# When TIMING=1, wrap commands with timing measurement
ifeq ($(TIMING),1)
  # TIME_CMD writes: elapsed_seconds source_file
  define TIME_CMD
	@_start=$$(date +%s%N); \
	$(1); \
	_end=$$(date +%s%N); \
	_elapsed=$$(( (_end - _start) / 1000000 )); \
	_secs=$$(( _elapsed / 1000 )); \
	_ms=$$(( _elapsed % 1000 )); \
	printf "%d.%03d %s\n" $$_secs $$_ms "$(2)" >> $(TIMING_LOG)
  endef
else
  define TIME_CMD
	$(1)
  endef
endif

# =============================================================================
# CUDA auto-detection (Ubuntu + Arch portable)
# =============================================================================
CUDA_NVCC := $(shell which nvcc 2>/dev/null)

ifeq ($(CUDA_NVCC),)
$(error nvcc not found. CUDA toolkit is not installed or not in PATH)
endif

CUDA_ROOT := $(shell dirname $(shell dirname $(CUDA_NVCC)))
CUDA_INC  := $(CUDA_ROOT)/include
CUDA_LIB  := $(CUDA_ROOT)/lib64

# Auto-detect NVTX include path (CUDA 12+ uses nvtx3 subdirectory)
NVTX_INC := $(shell find $(CUDA_ROOT)/targets -name "nvtx3" -type d 2>/dev/null | head -1)
ifeq ($(NVTX_INC),)
NVTX_INC := $(CUDA_INC)
endif

# Auto-detect NVTX library path (CUDA 12+ uses targets/x86_64-linux/lib)
NVTX_LIB := $(shell find $(CUDA_ROOT)/targets -name "libnvToolsExt.so" -exec dirname {} \; 2>/dev/null | head -1)
ifeq ($(NVTX_LIB),)
NVTX_LIB := $(CUDA_LIB)
endif

CPPFLAGS = -Iinclude -I$(CUDA_INC) -I$(NVTX_INC) -DWITH_CUDA

# Autograd Profiler flag
ifeq ($(PROFILER),1)
CPPFLAGS += -DAUTOGRAD_PROFILER_ENABLED
endif
# -ftime-report: pass TIME_TRACE=1 to capture per-TU compiler timing
ifeq ($(TIME_TRACE),1)
  FTIME_FLAG := -ftime-report
  TRACE_DIR  := $(LIBDIR)/time-trace
else
  FTIME_FLAG :=
endif

# GPU compute capability (override: make SM_ARCH=120). Defaults to 86 when
# invoked standalone; the top-level Makefile normally supplies an auto-detected
# value from `nvidia-smi --query-gpu=compute_cap`.
SM_ARCH ?= 86

CXXFLAGS = -std=c++2a -fPIC -Wall -Wextra -O3 -fopenmp -mavx2 -mfma -mf16c $(FTIME_FLAG)
NVCCFLAGS = -std=c++17 -Xcompiler="-fPIC" -arch=sm_$(SM_ARCH) --use_fast_math  -O3 --expt-relaxed-constexpr 

# =============================================================================
# BLAS Backend Selection (WITH_BLUBLAS=1)
# Default: cuBLAS only. WITH_BLUBLAS=1 compiles both backends into libtensor
# and selects at runtime via USE_BLU_BLAS env var (see GenMatmul.cu dispatcher).
# =============================================================================
MYBLAS_DIR := ../BluBridge-BLAS
BLAS_STAMP := $(OBJDIR)/.blas_backend

# Object files that change between backends
BLAS_OBJ_GENMATMUL   := $(OBJDIR)/src/Kernels/cuda/matmul/GenMatmul.o
BLAS_OBJ_BACKWARD    := $(OBJDIR)/src/Kernels/cuda/matmul/MatmulBackward.o

# Resolve which backend is requested (default = cublas)
ifeq ($(WITH_BLUBLAS),1)
  BLAS_BACKEND := blublas
  CPPFLAGS += -DWITH_MYBLAS -I$(MYBLAS_DIR)/include
else
  BLAS_BACKEND := cublas
endif

# Auto-detect CUDA targets lib directory (for NVTX and other libraries)
CUDA_TARGETS_LIB := $(shell find $(CUDA_ROOT)/targets -type d -name "lib" 2>/dev/null | head -1)
ifeq ($(CUDA_TARGETS_LIB),)
CUDA_TARGETS_LIB := $(CUDA_LIB)
endif

RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN/lib'
ifeq ($(BLAS_BACKEND),blublas)
  RPATH += -Xlinker -rpath -Xlinker '$$ORIGIN/../BluBridge-BLAS/build'
endif
LDFLAGS = -L$(CUDA_LIB) -L$(CUDA_TARGETS_LIB) -L$(LIBDIR) $(RPATH)
ifeq ($(BLAS_BACKEND),blublas)
  LDFLAGS += -L$(MYBLAS_DIR)/build
endif
# Note: nvToolsExt library is deprecated in CUDA 12+, NVTX3 is header-only
LDLIBS = -lcudart -lcuda -ltbb -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml
ifeq ($(BLAS_BACKEND),blublas)
  LDLIBS += -lmyblas
endif

# =============================================================================
# File Discovery (Automatic)
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

CU_SOURCES_FOR_DLINK := \
    src/Views/ContiguousKernel.cu \
    src/UnaryOps/cuda/ReductionImplGPU.cu

CU_SOURCES_REGULAR := $(filter-out $(CU_SOURCES_FOR_DLINK), $(CU_SOURCES))
OBJECTS_FROM_CPP := $(patsubst %.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU_REGULAR := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_REGULAR))
OBJECTS_FOR_DLINK := $(patsubst %.cu,$(OBJDIR)/%.o,$(CU_SOURCES_FOR_DLINK))
DEVICE_LINK_OBJ := $(OBJDIR)/device_link.o
ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU_REGULAR) $(OBJECTS_FOR_DLINK)

# =============================================================================
# Main Build Rules
# =============================================================================
.PHONY: all
all: timing-init $(TARGET_SO) $(TARGET_A)
	@echo "\n\n✅ Library build is up-to-date."
ifeq ($(TIMING),1)
	@echo ""
	@$(MAKE) --no-print-directory timing-report
endif

# Create timing dir when TIMING=1 (no-op otherwise)
.PHONY: timing-init
timing-init:
ifeq ($(TIMING),1)
	@mkdir -p $(TIMING_DIR)
endif

$(TARGET_SO): $(ALL_OBJECTS) $(DEVICE_LINK_OBJ)
	@echo "\n\n--- Creating shared library: $@"
	$(call TIME_CMD,$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(DEVICE_LINK_OBJ) $(LDFLAGS) $(LDLIBS) -o $@,[LINK] $@)

$(TARGET_A): $(ALL_OBJECTS)
	@echo "\n\n--- Creating static library: $@"
	$(call TIME_CMD,ar rcs $@ $(ALL_OBJECTS),[AR] $@)

# --- Compilation Pattern Rules ---
$(DEVICE_LINK_OBJ): $(OBJECTS_FOR_DLINK)
	@echo "\n\n--- Linking CUDA device code..."
	$(call TIME_CMD,$(NVCC) $(NVCCFLAGS) -dlink $^ -o $@,[DLINK] device_link)

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX]: $<"
	$(call TIME_CMD,$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@,[CXX] $<)

$(OBJECTS_FOR_DLINK): NVCC_DEVICE_FLAGS = -dc
$(OBJECTS_FOR_DLINK): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA -dc]: $<"
	$(call TIME_CMD,$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_DEVICE_FLAGS) -c $< -o $@,[CUDA-dc] $<)

$(OBJECTS_FROM_CU_REGULAR): $(OBJDIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(call TIME_CMD,$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@,[CUDA] $<)

# =============================================================================
# BLAS Backend Stamp + Recompilation
# =============================================================================

# Read last-built backend from stamp file (empty if no stamp exists)
LAST_BLAS := $(shell cat $(BLAS_STAMP) 2>/dev/null)

# If backend changed, force recompile of the 2 matmul files + shared lib
ifneq ($(LAST_BLAS),$(BLAS_BACKEND))
.PHONY: $(BLAS_OBJ_GENMATMUL) $(BLAS_OBJ_BACKWARD) $(TARGET_SO)
endif

# Write the stamp after the shared library is built
$(TARGET_SO): | write-blas-stamp
.PHONY: write-blas-stamp
write-blas-stamp:
	@mkdir -p $(OBJDIR)
	@echo "$(BLAS_BACKEND)" > $(BLAS_STAMP)

# =============================================================================
# Custom Action Rules
# =============================================================================
.PHONY: run-snippet
run-snippet: $(TARGET_SO)
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify a file to run."; \
		echo "Usage: make run-snippet FILE=path/to/your/file.cpp [PROFILER=1] [WITH_BLUBLAS=1]"; \
		exit 1; \
	fi
	@echo "--- BLAS backend: $(BLAS_BACKEND) ---"
	@echo "--- Compiling snippet: $(FILE) ---"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner $(FILE) $(LDFLAGS) -ltensor $(LDLIBS)
	@echo "\n--- Running snippet_runner ---"
ifeq ($(BLAS_BACKEND),blublas)
	LD_LIBRARY_PATH=$(MYBLAS_DIR)/build:$(LIBDIR):$$LD_LIBRARY_PATH ./snippet_runner
else
	./snippet_runner
endif
	@echo "\n"
#   @echo "\n--- Cleaning up snippet ---"
#   rm -f snippet_runner

.PHONY: rebuild
rebuild:
	@$(MAKE) clean && $(MAKE) all

.PHONY: clean
clean:
	@echo "--- Cleaning up build files ---"
	rm -rf $(OBJDIR) $(TARGET_A) $(TARGET_SO)

# =============================================================================
# Build Timing Targets
# =============================================================================

# timing-report: print sorted per-file times + total from the log
.PHONY: timing-report
timing-report:
	@if [ ! -f $(TIMING_LOG) ]; then \
		echo "No timing data found. Run: make benchmark"; \
		exit 0; \
	fi
	@echo ""
	@echo "================================================================="
	@echo "  COMPILATION TIME REPORT"
	@echo "================================================================="
	@echo ""
	@echo "--- Per-file times (slowest first) ---"
	@echo ""
	@sort -rn $(TIMING_LOG) | while IFS=' ' read -r secs rest; do \
		printf "  %8ss  %s\n" "$$secs" "$$rest"; \
	done
	@echo ""
	@echo "-----------------------------------------------------------------"
	@_total=$$(awk '{s+=$$1} END{printf "%.3f", s}' $(TIMING_LOG)); \
	echo "  TOTAL (sum of all files):  $${_total}s"
	@_cuda_total=$$(grep '\[CUDA' $(TIMING_LOG) | awk '{s+=$$1} END{printf "%.3f", s}'); \
	echo "  CUDA files only:           $${_cuda_total}s"
	@_cxx_total=$$(grep '\[CXX\]' $(TIMING_LOG) | awk '{s+=$$1} END{printf "%.3f", s}'); \
	echo "  C++ files only:            $${_cxx_total}s"
	@_link_total=$$(grep '\[LINK\]\|\[DLINK\]\|\[AR\]' $(TIMING_LOG) | awk '{s+=$$1} END{printf "%.3f", s}'); \
	echo "  Linking:                    $${_link_total}s"
	@echo ""
	@echo "  File count:                $$(wc -l < $(TIMING_LOG)) steps"
	@echo "================================================================="
	@echo ""

# benchmark: clean build with timing, produces a saved report
.PHONY: benchmark
benchmark:
	@echo "================================================================="
	@echo "  BENCHMARK: Full clean build with per-file timing"
	@echo "================================================================="
	@$(MAKE) --no-print-directory clean
	@mkdir -p $(TIMING_DIR)
	@rm -f $(TIMING_LOG)
	@echo "Build started at: $$(date)" | tee $(TIMING_DIR)/benchmark_meta.txt
	@_wall_start=$$(date +%s%N); \
	$(MAKE) --no-print-directory all TIMING=1; \
	_wall_end=$$(date +%s%N); \
	_wall_ms=$$(( (_wall_end - _wall_start) / 1000000 )); \
	_wall_s=$$(( _wall_ms / 1000 )); \
	_wall_r=$$(( _wall_ms % 1000 )); \
	echo "" >> $(TIMING_DIR)/benchmark_meta.txt; \
	echo "Build finished at: $$(date)" >> $(TIMING_DIR)/benchmark_meta.txt; \
	printf "Wall-clock time: %d.%03ds\n" $$_wall_s $$_wall_r | tee -a $(TIMING_DIR)/benchmark_meta.txt; \
	echo ""; \
	echo "================================================================="; \
	echo "  WALL-CLOCK TIME: $${_wall_s}.$${_wall_r}s"; \
	echo "================================================================="; \
	echo ""
	@cp $(TIMING_LOG) $(TIMING_DIR)/baseline_$$(date +%Y%m%d_%H%M%S).log
	@echo "Timing log saved to: $(TIMING_DIR)/"
	@echo "  - compile_times.log     (current run)"
	@echo "  - baseline_*.log        (timestamped snapshot)"
	@echo ""
	@echo "To compare after changes, run 'make benchmark' again and diff:"
	@echo "  diff $(TIMING_DIR)/baseline_BEFORE.log $(TIMING_DIR)/baseline_AFTER.log"

.PHONY: run-folder
run-folder: $(TARGET_SO)
	@if [ -z "$(FOLDER)" ]; then \
		echo "ERROR: Please specify a folder to run."; \
		echo "Usage: make run-folder FOLDER=Tests/path/to/folder [WITH_BLUBLAS=1]"; \
		exit 1; \
	fi
	@echo "--- BLAS backend: $(BLAS_BACKEND) ---"
	@echo "--- Running all tests in: $(FOLDER) ---"
	@for file in $(FOLDER)/*.cpp; do \
		if [ ! -f "$$file" ]; then continue; fi; \
		echo "\n---------------------------------------------------------"; \
		echo "Running: $$file"; \
		echo "---------------------------------------------------------"; \
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o snippet_runner "$$file" $(LDFLAGS) -ltensor $(LDLIBS) || exit 1; \
		$(if $(filter koblas,$(BLAS_BACKEND)),LD_LIBRARY_PATH=$(MYBLAS_DIR)/build:$(LIBDIR):$$LD_LIBRARY_PATH) ./snippet_runner || exit 1; \
	done
	@rm -f snippet_runner
	@echo "\n All tests in $(FOLDER) passed!"