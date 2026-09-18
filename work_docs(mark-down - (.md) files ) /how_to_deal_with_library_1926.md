# cgadimpl_ Build System Documentation_1926

This document explains how the Makefile build system works in the `cgadimpl_` project. It covers building, linking, and running your code.

---

##  Project Structure_1926

```
cgadimpl_/
├── Makefile              ← PARENT Makefile (orchestrates everything)
├── regression_own.cpp    ← Your test files go here
│
├── tensor/               ← TENSOR LIBRARY (Backend)
│   ├── Makefile          ← Child Makefile for tensor
│   ├── src/              ← Source files (.cpp, .cu)
│   ├── include/          ← Header files (.h, .hpp)
│   └── lib/              ← Built libraries (after make)
│       ├── libtensor.so  ← Shared library
│       ├── libtensor.a   ← Static library
│       └── objects/      ← Compiled .o files
│
└── cgadimpl/             ← AUTOGRAD LIBRARY (Frontend)
    ├── Makefile          ← Child Makefile for cgadimpl
    ├── src/              ← Source files (.cpp)
    ├── include/          ← Header files (.hpp)
    └── lib/              ← Built libraries (after make) 
        ├── libcgadimpl.so
        ├── libcgadimpl.a
        └── objects/
```

---

##  Quick Start Commands:

Run all commands from the `cgadimpl_/` directory (parent folder).

### Build and Run (Recommended)

```bash
# Build both libraries (if needed) and run your file
make run-snippet FILE=regression_own.cpp
```

### Force Rebuild and Run

```bash
# Clean + rebuild both libraries, then run your file
make run-snippet-force FILE=regression_own.cpp
```

### Other Commands

```bash
make                     # Build both libraries (incremental)
make clean               # Remove all built files
make rebuild             # Clean + build both libraries
```

---

##  Available Makefile Targets:

| Command | What It Does |
|---------|--------------|
| `make` | Build tensor + cgadimpl libraries (incremental) |
| `make clean` | Delete all .o, .so, .a files from both libraries |
| `make rebuild` | Clean + build everything fresh |
| `make run-snippet FILE=path` | Build (if needed) + compile + run your file |
| `make run-snippet-force FILE=path` | Force rebuild everything + compile + run |

---

##  How The Build System Works:

### Step-by-Step Flow: `make run-snippet-force`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        make run-snippet-force                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Build Tensor Library (tensor/Makefile)                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │ Tensor.cpp      │    │                 │    │ tensor/lib/         │ │
│  │ TensorOps.cpp   │ ─► │ g++ -c ─► .o    │ ─► │   libtensor.so      │ │
│  │ Device.cpp      │    │ files           │    │   libtensor.a       │ │
│  │ ...             │    │                 │    │   objects/*.o       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Build cgadimpl Library (cgadimpl/Makefile)                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │ graph.cpp       │    │                 │    │ cgadimpl/lib/       │ │
│  │ nn.cpp          │ ─► │ g++ -c ─► .o    │ ─► │   libcgadimpl.so    │ │
│  │ autodiff.cpp    │    │ files           │    │   libcgadimpl.a     │ │
│  │ ...             │    │                 │    │   objects/*.o       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Compile & Link Your Test File                                  │
│  ┌───────────────────┐                                                  │
│  │ regression_own.cpp│                                                  │
│  └─────────┬─────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ g++ regression_own.cpp -o snippet_runner                          │ │
│  │     -L../tensor/lib -ltensor        ◄─── Links libtensor.so       │ │
│  │     -Llib -lcgadimpl                ◄─── Links libcgadimpl.so     │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│            │                                                            │
│            ▼                                                            │
│  ┌───────────────────┐                                                  │
│  │  snippet_runner   │  ◄─── Executable created!                       │
│  └─────────┬─────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Run The Executable                                             │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │ LD_LIBRARY_PATH="./lib:../tensor/lib" ./snippet_runner            │ │
│  │                                                                    │ │
│  │     snippet_runner ─── loads ──→ libtensor.so                     │ │
│  │                   ─── loads ──→ libcgadimpl.so                    │ │
│  │                   ─── loads ──→ libcudart.so                      │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

##  Understanding Compilation & Linking

### The g++ Command Explained

When you run a snippet, this command is executed:

```bash
g++ -std=c++20 -O3 \
    -Iinclude -I../tensor/include \           # Header search paths
    regression_own.cpp \                       # Your source file
    -o snippet_runner \                        # Output executable
    -Llib -L../tensor/lib \                   # Library search paths
    -lcgadimpl -ltensor -lcudart -fopenmp \   # Libraries to link
    -Wl,-rpath,...                            # Runtime library paths
```

### Compiler Flags Explained

| Flag | Meaning |
|------|---------|
| `-std=c++20` | Use C++20 standard |
| `-O3` | Maximum optimization |
| `-Iinclude` | Look for headers in `include/` folder |
| `-I../tensor/include` | Look for headers in `tensor/include/` |
| `-Llib` | Look for libraries in `lib/` folder |
| `-L../tensor/lib` | Look for libraries in `tensor/lib/` |
| `-lcgadimpl` | Link against `libcgadimpl.so` |
| `-ltensor` | Link against `libtensor.so` |
| `-lcudart` | Link against CUDA runtime |
| `-fopenmp` | Enable OpenMP parallelization |
| `-Wl,-rpath,...` | Set runtime library search paths |

---

##  Key Concepts

### 1. Compilation vs Linking

| Phase | What Happens | Output |
|-------|--------------|--------|
| **Compilation** | `.cpp` → `.o` (object files) | Machine code, unlinked |
| **Linking** | `.o` files → `.so` or executable | Complete program |

### 2. Static vs Shared Libraries

| Type | File | When Used |
|------|------|-----------|
| **Shared** (`.so`) | `libtensor.so` | Loaded at runtime, smaller executables |
| **Static** (`.a`) | `libtensor.a` | Embedded into executable, larger but portable |

### 3. Library Output Structure

```
tensor/lib/
├── libtensor.so     ← Shared library (used at runtime)
├── libtensor.a      ← Static library (alternative)
└── objects/         ← All compiled .o files
    └── src/
        ├── core/
        │   └── Tensor.o
        ├── ops/
        │   └── TensorOps.o
        ...
```

---

##  Makefile Dependency System

### How Dependencies Work

In Makefiles, this syntax:
```makefile
target: dependency1 dependency2
    commands...
```

Means: "Run dependency1 and dependency2 FIRST, then run target's commands."

### Example from Parent Makefile

```makefile
run-snippet-force: force-rebuild-cgadimpl
    ...

force-rebuild-cgadimpl: force-rebuild-tensor
    make -C cgadimpl clean
    make -C cgadimpl

force-rebuild-tensor:
    make -C tensor clean
    make -C tensor
```

**Execution Order:**
1. `force-rebuild-tensor` runs first (builds tensor library)
2. `force-rebuild-cgadimpl` runs second (builds cgadimpl library)
3. `run-snippet-force` runs last (compiles and runs your file)

---

##  Common Questions

### Q: When should I use `run-snippet` vs `run-snippet-force`?

| Use Case | Command |
|----------|---------|
| Just changed your test file | `make run-snippet FILE=...` |
| Changed library source code | `make run-snippet-force FILE=...` |
| Something seems wrong | `make run-snippet-force FILE=...` |

### Q: Do I need quotes around the filename?

```bash
make run-snippet FILE=regression_own.cpp     #  Works
make run-snippet FILE="regression_own.cpp"   # Also works
make run-snippet FILE="my file.cpp"          #  Needed for spaces
```

### Q: Why does it say "Library build is up-to-date"?

Make detected no changes to library source files since the last build. It skips recompilation to save time. Use `run-snippet-force` if you want to force rebuild.

### Q: Where do I put my test files?

Put them in the `cgadimpl_/` folder (same level as the parent Makefile).

```
cgadimpl_/
├── Makefile
├── regression_own.cpp    ← Your file here
├── my_test.cpp           ← Another test file
├── tensor/
└── cgadimpl/
```

---

##  Troubleshooting

### Error: "No such file or directory: ag_all.hpp"

**Fix:** Use correct include path:
```cpp
#include "ad/ag_all.hpp"   //  Correct
#include "ag_all.hpp"       //  Wrong
```

### Error: "undefined reference to omp_get_thread_num"

**Fixed in Makefile:** `-fopenmp` flag is already added.

### Error: "cannot find -ltensor"

**Fix:** Run `make` first to build libraries, or use `make run-snippet-force`.

### Error: "libtensor.so: cannot open shared object file"

**Fixed in Makefile:** `LD_LIBRARY_PATH` is automatically set.

---

##  Creating a New Test File

1. Create a new `.cpp` file in `cgadimpl_/`:
   ```cpp
   // my_test.cpp
   #include <iostream>
   #include "ad/ag_all.hpp"
   
   int main() {
       std::cout << "Hello from cgadimpl!" << std::endl;
       return 0;
   }
   ```

2. Run it:
   ```bash
   make run-snippet FILE=my_test.cpp
   ```

---

##  Summary

| What You Want | Command |
|---------------|---------|
| Build everything | `make` |
| Clean everything | `make clean` |
| Force rebuild | `make rebuild` |
| Run your file (fast) | `make run-snippet FILE=your_file.cpp` |
| Run your file (fresh) | `make run-snippet-force FILE=your_file.cpp` |
