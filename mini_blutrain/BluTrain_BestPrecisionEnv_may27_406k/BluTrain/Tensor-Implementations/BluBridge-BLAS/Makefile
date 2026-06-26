# =============================================================================
# MyBlas Makefile - Simple Build System
# =============================================================================

# Compiler Configuration
SHELL := /bin/bash
export PATH := /usr/local/cuda-13.0/nvvm/bin:$(PATH)
CXX = g++
NVCC = /usr/local/cuda-13.0/bin/nvcc
SRCDIR := src
OBJDIR := build/objects
BUILDDIR := build
LIBNAME := myblas
TARGET_SO := $(BUILDDIR)/lib$(LIBNAME).so

# Compilation Flags
CPPFLAGS = -Iinclude -I/usr/local/cuda/include -Iexternal/cutlass/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl
CXXFLAGS = -std=c++17 -fPIC -Wall -O3 -DNDEBUG
NVCCFLAGS = -std=c++17 -Xcompiler="-fPIC" -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -O3 -DNDEBUG -Xptxas -O3 -use_fast_math -Iexternal/cutlass/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include/cccl

# Linker Flags
RPATH = -Xlinker -rpath -Xlinker '$$ORIGIN' -Xlinker -rpath -Xlinker '$$ORIGIN/build'
LDFLAGS = -L/usr/local/cuda/lib64 -L$(BUILDDIR) $(RPATH)
LDLIBS = -lcudart -lcublas -lcublasLt

# =============================================================================
# Source Files - Auto-Discovery
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')

OBJECTS_FROM_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU := $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CU_SOURCES))
ALL_OBJECTS := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU)

# =============================================================================
# Test Programs - Auto-Discovery
# =============================================================================
# Automatically find all .cpp and .cu files in examples/ and create test targets
TEST_SOURCES := $(wildcard examples/*.cpp) $(wildcard examples/*.cu)
TEST_NAMES := $(basename $(notdir $(TEST_SOURCES)))
TEST_EXECUTABLES := $(addprefix $(BUILDDIR)/,$(TEST_NAMES))

# =============================================================================
# Build Targets
# =============================================================================
.PHONY: all lib tests clean rebuild help

all: lib

lib: $(TARGET_SO)
	@echo "\n✅ Library built: $(TARGET_SO)"

tests: lib
	@echo "\n--- Building tests from examples/ ---"
	@for test_src in $(TEST_SOURCES); do \
		test_name=$$(basename $$test_src .cpp); \
		test_name=$$(basename $$test_name .cu); \
		echo "Building: $$test_name"; \
		if [[ $$test_src == *.cpp ]]; then \
			$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$$test_name $$test_src $(LDFLAGS) -l$(LIBNAME) $(LDLIBS) 2>&1 | grep -v "warning:" || true; \
		else \
			$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $(BUILDDIR)/$$test_name $$test_src $(LDFLAGS) -l$(LIBNAME) $(LDLIBS) 2>&1 | grep -v "warning:" || true; \
		fi \
	done
	@echo "\n✅ Test building complete (errors ignored for incomplete tests)"

# Build shared library
$(TARGET_SO): $(ALL_OBJECTS)
	@echo "\n--- Linking shared library: $@"
	@mkdir -p $(BUILDDIR)
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX]: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# Build test executables (C++)
$(BUILDDIR)/%: examples/%.cpp $(TARGET_SO)
	@echo "Building test (CXX): $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)

# Build test executables (CUDA)
$(BUILDDIR)/%: examples/%.cu $(TARGET_SO)
	@echo "Building test (CUDA): $@"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $@ $< $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)

# =============================================================================
# Utility Targets
# =============================================================================
rebuild:
	@$(MAKE) clean && $(MAKE) all

clean:
	@echo "--- Cleaning build artifacts ---"
	rm -rf $(OBJDIR) $(BUILDDIR)

# Quick test builder - Usage: make test TEST=test_hgemm_v2
test: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make test TEST=test_hgemm_v2"; \
		exit 1; \
	fi
	@# Extract filename without path and extension
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@echo "--- Building/Updating test: $(TEST_NAME) ---"
	@if [ -f "$(TEST)" ]; then \
		if [[ "$(TEST)" == *.cpp ]]; then \
			$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$(TEST_NAME) $(TEST) $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
		else \
			$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $(BUILDDIR)/$(TEST_NAME) $(TEST) $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
		fi; \
	else \
		$(MAKE) $(BUILDDIR)/$(TEST_NAME); \
	fi
	@echo "✅ Built: $(BUILDDIR)/$(TEST_NAME)"

# Run a test - Usage: make run TEST=test_hgemm_v2
run: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make run TEST=test_hgemm_v2"; \
		exit 1; \
	fi
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@echo "--- Building/Updating $(TEST_NAME) ---"
	@$(MAKE) test TEST=$(TEST_NAME)
	@echo "--- Running $(BUILDDIR)/$(TEST_NAME) ---"
	@LD_LIBRARY_PATH=$(BUILDDIR):$$LD_LIBRARY_PATH $(BUILDDIR)/$(TEST_NAME)

# Run an arbitrary file as a test - Usage: make run-snippet FILE=path/to/test.cu
run-snippet: lib
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify FILE path"; \
		echo "Usage: make run-snippet FILE=examples/test_hgemm_v2.cu"; \
		exit 1; \
	fi
	@echo "--- Compiling snippet: $(FILE) ---"
	@filename=$$(basename $(FILE)); \
	exe_name=$(BUILDDIR)/$${filename%.*}; \
	if [[ $(FILE) == *.cpp ]]; then \
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $$exe_name $(FILE) $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
	else \
		$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $$exe_name $(FILE) $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
	fi; \
	echo "--- Running $$exe_name ---"; \
	LD_LIBRARY_PATH=$(BUILDDIR):$$LD_LIBRARY_PATH $$exe_name

help:
	@echo "MyBlas Makefile - Available targets:"
	@echo ""
	@echo "  make              - Build library and all tests (auto-discovered)"
	@echo "  make lib          - Build only the shared library"
	@echo "  make tests        - Build all test programs from examples/"
	@echo "  make test TEST=<name>  - Build specific test"
	@echo "  make run TEST=<name>   - Build and run specific test"
	@echo "  make rebuild      - Clean and rebuild everything"
	@echo "  make clean        - Remove all build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make test TEST=test_hgemm_v2"
	@echo "  make run TEST=test_hgemm_v2"
	@echo ""
	@echo "Auto-discovered tests: $(TEST_NAMES)"

# Dependency tracking
-include $(ALL_OBJECTS:.o=.d)
