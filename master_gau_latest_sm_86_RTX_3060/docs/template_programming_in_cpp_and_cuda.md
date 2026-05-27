# Template Programming in C++ and CUDA — Complete Guide

---

## Table of contents

1. [Background: how my reduction module is structured (the context for this doc)](#1-background)
2. [The problem that templates solve in the first place](#2-the-problem)
3. [What is a template, really?](#3-what-is-a-template)
4. [The four core types of templates](#4-four-core-types)
5. [Common doubts about basic templates](#5-common-doubts)
6. [The big distinction — why we use templates for structs and classes](#6-overloading-vs-templates)
7. [Template specialization deep dive](#7-specialization)
8. [Type conversion behaviors (bool, boolalpha, etc.)](#8-type-conversion)
9. [What if many types share behavior? Three approaches](#9-shared-behavior)
10. [Where can templates live? Templates inside vs outside structs](#10-scope)
11. [Why templates MUST live in header files](#11-headers)
12. [CUDA templates — the twist](#12-cuda-templates)
13. [How our codebase uses all of this — the `ReductionOpSelector` pattern](#13-reduction-op-selector)
14. [Why `Reduction.h` includes `ReductionImpl.h`](#14-reduction-h-includes)
15. [Why `Reduction.cpp` includes `ReductionImpl.h` again (even though `Reduction.h` already did)](#15-reduction-cpp-includes-again)
16. [Quick reference cheat sheet](#16-cheat-sheet)

---

<a id="1-background"></a>

## 1. Background — how  reduction module is structured

Before jumping into templates, here is the context.  reduction module computes things like sum, product, min, max, argmin, argmax, mean, variance, nan-aware versions, and boolean reductions (all, any). It supports many dtypes — `float32`, `float16`, `bfloat16`, `int8/16/32/64`, `uint8/16/32/64`, `bool`, `complex64`, `complex128`, and some packed FP4 types. It runs on both CPU and GPU (with CUDA warp-shuffle tree reductions).

The module is split into about 11 files but the most important ones for understanding the template architecture are:

- `Tensor-Implementations/include/ops/UnaryOps/Reduction.h` — the public API. Contains non-templated function declarations like `reduce_sum`, `reduce_mean`, `reduce_max`. This is what external code calls.
- `Tensor-Implementations/include/ops/helpers/ReductionOps.h` — the operation functors. Contains 14+ templated structs like `SumOp<T>`, `ProductOp<T>`, `MinOp<T>`, `MaxOp<T>`, `NanSumOp<T>`, `AllOp<T>`, `ArgMaxOp<T>`, etc. Each struct holds an `identity()`, `reduce()`, `combine()`, `project()`, and `warp_shfl_down()` method.
- `Tensor-Implementations/include/ops/helpers/ReductionUtils.h` — the shared utility layer. Contains the `ReductionLayout` struct that classifies every reduction into one of three execution paths (`InnerContiguous` for horizontal SIMD, `OuterContiguous` for vertical SIMD, `Generic` for the naive fallback) plus helper function declarations that both the CPU and GPU dispatchers call before launching kernels — things like `normalize_axes`, `compute_reduction_layout`, `calculate_reduced_count`, `calculate_output_shape`. This file is shape/layout analysis, not math, so it is NOT templated — both CPU and GPU code use it the same way.
- `Tensor-Implementations/src/UnaryOps/cpu/ReductionUtils.cpp` — the implementations of those helper functions. Because nothing here is templated, the bodies live in a regular `.cpp` file and compile once into `lib/objects/src/UnaryOps/cpu/ReductionUtils.o`. This is a clean example of "non-templated code goes in a .cpp file the normal way" — the rules in [section 11](#11-headers) about templates needing to live in headers do NOT apply here, because there is nothing for the compiler to instantiate.
- `Tensor-Implementations/include/ops/helpers/ReductionImpl.h` — the CPU implementation. Contains the cascade sum kernel and the generic templated dispatch function `dispatch_reduction<T, OpType>` that the public functions actually call.
- `Tensor-Implementations/include/ops/helpers/ReductionKernels.cuh` — the GPU kernel implementations.
- `Tensor-Implementations/src/UnaryOps/cpu/Reduction.cpp` — implements the public functions, dispatches by runtime dtype, and forwards to the templated machinery.
- `Tensor-Implementations/src/UnaryOps/cuda/ReductionImplGPU.cu` — same idea for GPU, but with explicit template instantiations for every dtype × operation combination (over 250 of them).

So the high-level flow is:

```
user calls reduce_sum(tensor)
        ↓
Reduction.cpp does a runtime switch on tensor.dtype()
        ↓
for float case → calls dispatch_reduction<float, SumOp>(...)
        ↓
template machinery in ReductionImpl.h kicks in, compiler instantiates SumOp<float>
        ↓
either cascade_sum_kernel runs on CPU, or launch_reduction_gpu fires on GPU
        ↓
result returned as Tensor
```

The whole reason templates exist in this design is so we do not have to write a separate `reduce_sum_float`, `reduce_sum_double`, `reduce_sum_half`, `reduce_sum_int32` (and so on for every other operation) by hand. That would be hundreds of nearly-identical functions to maintain. Templates let us write the math once and let the compiler generate all the variants.

Now that the context is set, let us actually understand templates from scratch.

---

<a id="2-the-problem"></a>

## 2. The problem that templates solve

Imagine I just want to write a simple function that adds two numbers, and I want it to work for `int`, `float`, and `double`. In a language without templates (like C), I have three options, and all three have major drawbacks.

### Option 1 — Write the same function multiple times with different names

```c
int    add_int(int a, int b)        { return a + b; }
float  add_float(float a, float b)  { return a + b; }
double add_double(double a, double b){ return a + b; }
```

**Drawback**: massive code duplication. If I find a bug in one, I have to fix it in all three. If I add `int64_t` later, I have to copy-paste again. In our reductions module with 14 operations × 14 dtypes × CPU + GPU, this would mean writing ~400 nearly identical functions.

### Option 2 — Use a preprocessor macro

```c
#define ADD(a, b) ((a) + (b))
```

This looks slick because one definition serves all types. But the preprocessor is **dumb**. It is not a compiler. It just does blind text substitution before the real compiler ever sees the code. This causes serious bugs.

**Drawback 1 — zero type checking**:

```c
ADD("hello", 5);     // expands to ("hello") + (5)
                     // compiler throws a cryptic error pointing
                     // INTO the macro expansion, not the call site
ADD({1,2,3}, 5);     // also expands to something nonsensical
```

**Drawback 2 — the double-evaluation bug**. This is the most dangerous one. Imagine:

```c
int x = 5;
int y = 10;
int result = ADD(x++, y);    // I expect x to become 6
```

But what does the preprocessor actually do? It text-substitutes:

```c
int result = ((x++) + (y));
```

OK that seems fine. But now imagine someone wrote a more complex macro:

```c
#define MAX(a, b)  ((a) > (b) ? (a) : (b))
int result = MAX(x++, y);    // expected: x becomes 6
```

The preprocessor expands this to:

```c
int result = ((x++) > (y) ? (x++) : (y));    // x is incremented TWICE!
```

Your variable `x` gets incremented twice instead of once, silently breaking your program. There is no warning. There is no error. This happens because the macro literally copy-pasted `x++` into two places in the expanded code.

### Option 3 — Use a `void*` (generic) pointer

```c
void add(void* a, void* b, void* result, size_t item_size);
```

A `void*` is a "I am pointing to something but I refuse to tell you what" pointer. To use it I have to cast it back to the right type inside the function, and the caller has to pass in `sizeof(int)` or `sizeof(float)` to tell the function how many bytes to read.

**Drawback — slow**. There are several reasons:

1. **Loss of register optimization**. With templates and concrete types, the compiler keeps small types like `int` and `float` directly in CPU registers (e.g., `EAX`, `XMM0`). The CPU adds them in one instruction. With `void*`, the compiler is forced to allocate the value in main memory (RAM), pass the address around, and the CPU has to round-trip through memory just to do an add. Memory is roughly 100× slower than registers.

2. **The compiler can no longer inline**. When the compiler knows the concrete type, it can inline the entire function, fold constants, unroll loops, all the good optimizations. With `void*`, it is "blind" — it cannot make any assumptions about the data, so it has to generate generic, conservative, slow machine code.

3. **Pointer aliasing kills caching**. When two different parameters could potentially point to overlapping memory (which the compiler must assume with `void*`), it cannot cache results in registers between operations. Every write potentially invalidates every cached read.

4. **Heap allocation overhead**. For non-trivial generic types, `void*` parameters usually means dynamic memory allocation (`malloc`/`free`), which is roughly 1000× slower than stack allocation.

5. **Zero type safety**. Just like macros, the compiler cannot check if you passed the right thing:

```c
void* ptr = &my_float;        // points to a float
int*  wrong_ptr = (int*)ptr;  // I cast it to int — no warning!
int   x = *wrong_ptr;         // reads float bits AS an int → garbage value or crash
```

### The C++ solution

C++ introduced templates specifically to solve all three problems at once — no duplication, full type safety, and zero runtime overhead. C does NOT have templates. That's why C codebases often look bloated with duplicate functions or scary with macros, while modern C++ codebases like ours can stay clean and fast at the same time.

---

<a id="3-what-is-a-template"></a>

## 3. What is a template, really?

A template is **not actual code**. It is a **blueprint** that I hand to the compiler with the instructions: *"Here is how to write this code. When you see me actually use it with a specific type, generate a customized version of this code for that type at compile time."*

When the compiler reads a template like this:

```cpp
template <typename T>
T add(T a, T b) {
    return a + b;
}
```

it does **NOT** generate any machine code yet. It just parses the blueprint, does some basic syntax checks, and waits.

The magic happens when I actually use it somewhere:

```cpp
int main() {
    int x = add(5, 10);          // (1)
    double y = add(3.1, 4.2);    // (2)
}
```

1. The compiler sees line (1), notices I passed two `int` parameters, deduces that `T = int`, and **instantiates** (generates) a real concrete function behind the scenes:

   ```cpp
   int add(int a, int b) { return a + b; }
   ```

2. The compiler sees line (2), notices two `double` parameters, deduces `T = double`, and instantiates another fresh function:

   ```cpp
   double add(double a, double b) { return a + b; }
   ```

This generation happens entirely at **compile time**. The final compiled binary acts *exactly* as if I had hand-written both versions. There is **zero runtime overhead** — no pointer jumping, no slow type checks, just direct full-speed machine code, exactly like Option 1 would have given me but without the duplication burden.

---

<a id="4-four-core-types"></a>

## 4. The four core types of templates

### 4.1 Function templates

Allows one function to work with any type. This is what we just saw:

```cpp
template <typename T>
T find_max(T a, T b) {
    return (a > b) ? a : b;
}
```

I can call `find_max(3, 7)` to get an int max, or `find_max(3.14, 2.71)` to get a double max. The compiler generates whichever version I use.

### 4.2 Class / Struct templates

Allows a struct or class to hold or operate on any data type. **This is exactly how our `SumOp<T>` and every other operator struct in `ReductionOps.h` works.**

```cpp
template <typename T>
struct MathOps {
    T identity_value;

    T multiply(T a, T b) {
        return a * b;
    }
};

// To use it I have to tell the compiler what type to use:
MathOps<float> my_float_ops;
my_float_ops.identity_value = 1.0f;

MathOps<int> my_int_ops;
my_int_ops.identity_value = 1;
```

### 4.3 Template specialization

Sometimes the general blueprint does not work for one specific type. Maybe my generic `add` uses `a + b`, but for `bool` I actually want logical OR (`a || b`). I can write an override called a **specialization** for that one specific type:

```cpp
// General template
template <typename T>
T add(T a, T b) {
    return a + b;
}

// Specialization just for bool
template <>
bool add<bool>(bool a, bool b) {
    return a || b;   // override: logical OR for booleans
}
```

If I call `add(5, 10)` the compiler uses the general blueprint. If I call `add(true, false)` the compiler bypasses the general blueprint and uses my bool override instead.

### 4.4 Non-type template parameters

Normally templates pass *types* (`float`, `int`). But I can also pass compile-time constants like an `int` size or a `bool` flag.

```cpp
template <typename T, int Size>
struct Array {
    T data[Size];   // fixed-size array allocated on the stack
};

Array<float, 4> my_vector;    // size 4 is baked in at compile time
Array<int, 1024> my_buffer;   // size 1024 is baked in at compile time
```

In our CUDA kernels this is everywhere — block sizes, thread counts, unroll factors all get passed this way. Because the compiler knows the value at compile time it can unroll loops completely, which gives massive performance gains. The hardware tile-config constants for our sgemm kernels like `BM=256, BN=128, BK=32, STAGES=2, THREADS=256` are all non-type template parameters.

---

<a id="5-common-doubts"></a>

## 5. Common doubts about basic templates

While learning this I had several confusions. Writing them down here as Q&A so others do not stumble.

### Q5.1 — Do I need to include any special header just to use templates?

**No.** Templates are a core language feature built directly into the C++ compiler. There is no `#include <templates>`.

**However**, how I manage *my own* headers depends on what I am trying to do.

- **Defining my own template**: I do NOT need to include a special header to define one. I just write the entire definition (the implementation code) directly inside a standard header file (`.h` or `.hpp`):

  ```cpp
  // MyTemplate.h
  #pragma once
  template <typename T>
  T add(T a, T b) {
      return a + b;   // compiler needs to see this code to generate the function
  }
  ```

- **Using a Standard Library template**: I MUST include the specific header file for that utility:

  | To use this template | I need to include | Example |
  |---|---|---|
  | `std::vector` (dynamic arrays) | `<vector>` | `std::vector<int> my_list;` |
  | `std::map` (key-value stores) | `<map>` | `std::map<std::string, int> my_map;` |
  | `std::unique_ptr` (smart pointers) | `<memory>` | `std::unique_ptr<int> my_ptr;` |

### Q5.2 — What if I pass one `int` and one `float` to a template that takes the same `T` twice?

```cpp
template <typename T>
T add(T a, T b) { ... }

int x = add(5, 4.5f);   // ERROR — what is T?
```

This will **NOT compile**. The error is called **Template Argument Deduction Failure**.

The compiler reads my template signature `T add(T a, T b)` and sees that I specified **one** placeholder `T`, used for both `a` and `b`. So `a` and `b` must be exactly the same type. When I call `add(5, 4.5f)`:

1. The compiler looks at argument 1 (`5`) and thinks: *"OK, `T` must be `int`."*
2. The compiler looks at argument 2 (`4.5f`) and thinks: *"Wait, now `T` must be `float`!"*
3. Because the compiler cannot make two conflicting decisions about the same type at the same time, it gives up and errors out: `"no matching function for call to 'add(int, float)'"`

There are three ways to fix this:

**Fix 1 — Explicitly state the type (quickest)**

```cpp
auto result = add<double>(5, 4.5);   // I tell the compiler T = double
                                     // the int 5 gets auto-converted to 5.0 (double)
```

The compiler converts the integer `5` into a double `5.0` and instantiates `add<double>` cleanly.

**Fix 2 — Use two template parameters (best practice)**

```cpp
template <typename T1, typename T2>
auto add(T1 a, T2 b) {
    return a + b;
}

auto result = add(5, 4.5f);   // T1 = int, T2 = float
                              // return type auto-deduces to whatever `int + float` yields (float)
```

This is the cleanest because it lets the compiler accept any two completely different types. The `auto` return uses what C++ calls "the combined type rules" — `int + float` naturally yields `float`, `int + double` yields `double`, etc.

**Fix 3 — Manually cast one of the values before passing**

```cpp
int x = 5;
float y = 4.5f;
auto result = add(static_cast<float>(x), y);   // both are now float
```

### Q5.3 — Can I just write `struct MathOps my_ops;` without specifying the type?

```cpp
template <typename T>
struct MathOps {
    T identity_value;
};

MathOps my_ops;   // <-- can I do this?
```

It depends on the C++ version and whether I provide enough hints:

**With C++17 or newer (CTAD — Class Template Argument Deduction)** — I can use a constructor to help the compiler figure out the type:

```cpp
template <typename T>
struct MathOps {
    T identity_value;
    MathOps(T val) : identity_value(val) {}   // constructor helps deduce the type
};

MathOps my_ops(2.5);   // compiler infers MathOps<double> automatically
```

**With C++20 or newer (Aggregate Deduction)** — even without a constructor, if I provide an initializer:

```cpp
MathOps my_ops{1.0f};   // compiler sees float, instantiates MathOps<float>
MathOps my_ops2{5};     // compiler sees int, instantiates MathOps<int>
```

**Without C++17/20** — I always have to write the type explicitly:

```cpp
MathOps<float> my_ops;
```

In our codebase that targets C++20 (per Makefile `-std=c++2a`) all three forms work, but we still prefer the explicit `MathOps<float>` style for clarity.

### Q5.4 — Can I put the `template <typename T>` declaration INSIDE the struct, not above it?

**No.** The compiler needs to see `template <typename T>` BEFORE the struct keyword, otherwise the compiler doesn't know the struct itself is templated. Here is what is allowed and what is not:

```cpp
// NOT ALLOWED — putting template declaration inside the struct body
struct MyStruct {
    template <typename T>   // <-- the compiler doesn't know MyStruct is templated
    int identity_value;     // ERROR
};

// ALLOWED — the standard way, template declaration above the struct
template <typename T>
struct MyStruct {
    T identity_value;       // OK
};

// ALSO ALLOWED — making a specific member function templated within a non-templated struct
struct RegularStruct {
    int non_templated_var;            // always int
    template <typename T>
    void print_value(T val) {         // this single method is templated
        std::cout << val;
    }
};
```

The scope of the template parameter `T` is the **immediate block** following the `template <typename T>` declaration — the class, struct, function, or function template body. Once the matching closing brace `}` is hit, `T` no longer exists. If I want a member function inside the struct to also be templated independently, I would write `template <typename U>` separately on that function.

---

<a id="6-overloading-vs-templates"></a>

## 6. The big distinction — why we use templates for structs/classes

This is the part that confused me the most and took the longest to internalize. Writing it down clearly because it underlies the entire design of our reduction module.

### The core principle

In C++ I am allowed to **overload functions and operators**, but I am **NOT allowed to overload classes/structs**. And classes/structs are *custom-defined datatypes* — they are types I create myself, like `Tensor`, `SumOp<T>`, `MyMatrix`. The compiler treats every struct as a single unique type. I cannot have two structs both called `MyStruct` in the same scope that differ only by their template type.

This is why for **functions** I have two options when I want different behavior for different types:

**Option A — function overloading (regular C++)**

```cpp
int add(int a, int b)         { return a + b; }            // regular function
float add(float a, float b)   { return a + b; }            // another regular function
bool add(bool a, bool b)      { return a || b; }           // a third regular function
```

This is just plain old function overloading — no templates needed at all. The compiler picks the right one based on argument types. Each version is a completely separate function with its own machine code.

**Option B — template specialization (when starting from a generic blueprint)**

```cpp
template <typename T>
T add(T a, T b) { return a + b; }   // general blueprint

template <>
bool add<bool>(bool a, bool b) { return a || b; }   // specialization for bool
```

Both options work for **functions**. So for functions, specialization is more of a "stylistic choice" once you already have a templated function — you do not strictly need it because you can just write an overload.

But for **structs and classes**, function overloading is **NOT an option**. You cannot have two structs called `Storage` that differ only by their internal type:

```cpp
// NOT ALLOWED
struct Storage { int value; };
struct Storage { float value; };       // ERROR: redefinition of struct Storage
struct Storage { bool value; };        // ERROR: redefinition of struct Storage
```

So the **only** way to have a struct that behaves differently for different types is to make it templated:

```cpp
template <typename T>
struct Storage {
    T value;
    void print() { std::cout << "General Type Value: " << value << std::endl; }
};

template <>
struct Storage<bool> {                 // specialization JUST for bool
    bool value;
    void print() { std::cout << "Special Boolean Value: " << (value ? "YES" : "NO") << std::endl; }
    void flip() { value = !value; }    // can even add a new method that only exists for bool
};
```

This is exactly why our `SumOp`, `ProductOp`, `MinOp`, `MaxOp`, `AllOp`, `ArgMaxOp` etc. in `ReductionOps.h` are all `template <typename T> struct ...`. Operators have to be wrapped in structs (so they can carry state like the accumulator type, identity value, warp shuffle implementation, etc.), and structs cannot be overloaded — so templating them is the only way to make them universal across dtypes.

### Why the weird syntax `template <> bool add<bool>(...)`?

When I write a specialization for a function, I MUST use this exact syntax so the compiler knows three things at once:

```cpp
template <>                          // (1) "this is part of an existing template family"
bool add<bool>(bool a, bool b) {     // (2) "specifically the add<bool> instance of it"
    return a || b;                   // (3) "use this body instead of the general blueprint"
}
```

- `template <>` — the empty angle brackets tell the compiler: "I am NOT introducing a new template type. There is no `T` here, I am working off an existing template."
- `add<bool>` — explicitly links this specialization to the general `add` template, and the `<bool>` says "for the case when T = bool, use this".
- The body then completely replaces what the general blueprint would have produced for `T = bool`.

---

<a id="7-specialization"></a>

## 7. Template specialization deep dive

Here is a complete worked example to make struct specialization concrete.

```cpp
#include <iostream>

// (1) THE GENERAL TEMPLATE BLUEPRINT
template <typename T>
struct Storage {
    T value;

    void print() {
        std::cout << "General Type Value: " << value << std::endl;
    }
};

// (2) THE SPECIALIZED TEMPLATE FOR BOOL
template <>
struct Storage<bool> {
    bool value;

    // completely rewritten print() for bool
    void print() {
        std::cout << "Special Boolean Value: " << (value ? "YES" : "NO") << std::endl;
    }

    // I can even add brand-new methods that only exist on the bool version
    void flip() {
        value = !value;
    }
};

int main() {
    // Uses the general template
    Storage<int> my_int;
    my_int.value = 42;
    my_int.print();          // Output: General Type Value: 42
    // my_int.flip();        // ERROR: general template has no flip() method

    // Uses the specialized bool template
    Storage<bool> my_bool;
    my_bool.value = true;
    my_bool.print();         // Output: Special Boolean Value: YES

    my_bool.flip();          // works only for the bool version
    my_bool.print();         // Output: Special Boolean Value: NO

    return 0;
}
```

What is happening under the hood:

1. The compiler sees the general template `Storage<T>` and keeps it as a blueprint.
2. The compiler sees the specialization `Storage<bool>` and registers it as the override for the `T = bool` case.
3. When I write `Storage<int> my_int;`, the compiler stamps out a fresh `Storage` struct from the general blueprint with `T = int`.
4. When I write `Storage<bool> my_bool;`, the compiler **does not** use the general blueprint — it uses my fully-rewritten override.
5. `Storage<int>` and `Storage<bool>` are now **completely independent structs** in the final binary. They do not even share function signatures. The `flip()` method does not exist in `Storage<int>` at all.

This is the **same exact pattern** our `ReductionOps.h` uses across the entire file. Each `SumOp<T>`, `MinOp<T>`, etc. is the general blueprint, and we add specialized behavior per-dtype using `if constexpr` blocks inside the methods (a slightly different but related compile-time technique) rather than full struct specialization, because we want the methods themselves to differ based on `T` rather than the entire struct layout.

---

<a id="8-type-conversion"></a>

## 8. Type conversion behaviors (bool, boolalpha, etc.)

While learning template specialization for bool, I had a few related doubts about how C++ converts between types.

### Q8.1 — If I give any number (instead of `true`/`false`) to a `bool` variable, will it auto-convert?

**Yes.** C++ uses a very simple rule called **Implicit Type Conversion** (also called Type Promotion):

- `0` becomes `false`
- Any non-zero number (positive, negative, decimal) becomes `true`

```cpp
Storage<bool> my_bool;

my_bool.value = 0;    // converted to false
my_bool.print();      // Output: Special Boolean Value: NO

my_bool.value = 42;   // non-zero, converted to true
my_bool.print();      // Output: Special Boolean Value: YES

my_bool.value = -7;   // also non-zero, converted to true
my_bool.print();      // Output: Special Boolean Value: YES
```

This is a holdover from C, where booleans did not exist as a separate type — programmers used the integer `0` for false and `1` (or any non-zero) for true. When C++ introduced the official `bool` type, it kept this behavior to remain compatible: *"Is this value equal to zero? If no, treat it as true."*

### Q8.2 — In the general (non-specialized) template, if I give a bool input, what is printed?

If I removed the `Storage<bool>` specialization and relied only on the general template:

```cpp
template <typename T>
struct Storage {
    T value;
    void print() {
        std::cout << "General Type Value: " << value << std::endl;
    }
};

int main() {
    Storage<bool> my_bool;
    my_bool.value = true;
    my_bool.print();    // Output: General Type Value: 1

    my_bool.value = false;
    my_bool.print();    // Output: General Type Value: 0
}
```

It prints `1` for `true` and `0` for `false`. Why? Because:

1. The compiler generates `Storage<bool>` from the general blueprint, so `value` stores `true` or `false` correctly internally.
2. When `std::cout << value` runs, `std::cout` by default prints booleans as numeric digits (`1` and `0`) to keep output uniform with other integer types.

### Q8.3 — Can I make the general template print `true`/`false` text without writing a specialization?

Yes. Use the special C++ manipulator `std::boolalpha`:

```cpp
template <typename T>
struct Storage {
    T value;
    void print() {
        // std::boolalpha tells cout to print true/false as text for bool
        std::cout << "Value: " << std::boolalpha << value << std::endl;
    }
};

int main() {
    Storage<bool> my_bool{true};
    my_bool.print();    // Output: Value: true
}
```

### Q8.4 — Does `std::boolalpha` affect other dtypes?

**No.** `std::boolalpha` is a stream manipulator that **only** modifies how booleans are displayed. For all other types it does nothing:

```cpp
template <typename T>
struct Storage {
    T value;
    void print() {
        std::cout << "Value: " << std::boolalpha << value << std::endl;
    }
};

int main() {
    Storage<int>   my_int{1};       my_int.print();   // Output: Value: 1     (not "true")
    Storage<int>   my_int0{0};      my_int0.print();  // Output: Value: 0     (not "false")
    Storage<float> my_f{3.14f};     my_f.print();     // Output: Value: 3.14  (float printed normally)
    Storage<char>  my_c{'A'};       my_c.print();     // Output: Value: A     (char printed as character)
    Storage<bool>  my_b{true};      my_b.print();     // Output: Value: true  (boolalpha activates here)
}
```

It is like a filter that only activates when a literal `bool` passes through the pipe. If an `int`, `float`, or `char` passes through, the filter stays open and lets the data pass unchanged.

---

<a id="9-shared-behavior"></a>

## 9. What if many types share behavior? Three approaches

After learning bool specialization, my next doubt was: *"If I want different functionality for `int` (like other languages have) but want it to remain the same as the general template for everything else, do I just write a specialized template version for int?"* Yes, follow the same pattern as bool.

But what if `int` AND `float` should share the **same** special functionality, but `double`, `char`, `string` should use the general blueprint? Writing two separate specializations for `int` and `float` with the same body breaks the DRY principle (Don't Repeat Yourself). There are three approaches.

### Approach 1 — Modern C++20 Concepts (recommended)

Instead of writing multiple specializations, I tell the general template: *"If T is an int OR a float, use this block of code. Otherwise, use the general block."*

```cpp
#include <iostream>
#include <concepts>

template <typename T>
struct Storage {
    T value;

    void print() {
        // Check if T is exactly int or exactly float
        if constexpr (std::same_as<T, int> || std::same_as<T, float>) {
            std::cout << "Special Int/Float Processing: " << (value * 2) << std::endl;
        } else {
            std::cout << "General Type Value: " << value << std::endl;
        }
    }
};
```

`if constexpr` is the compile-time `if` — the compiler picks which branch survives based on `T` and completely discards the other branch. There is **zero runtime cost** because by the time the program runs, the unused branch does not even exist in the binary.

**Why this is great**: I only write the code once, and the compiler handles all the branching at compile time. This is what modern C++ projects (and our codebase) use heavily.

### Approach 2 — Inheritance via a shared helper (C++11 and older)

If I am stuck on an older C++ version that does not support C++20 Concepts, I create a shared "parent" helper struct containing the shared math code. Then both my `int` and `float` specializations simply inherit from it.

```cpp
#include <iostream>

// (1) Shared logic helper
template <typename T>
struct SpecialMathBehavior {
    T value;
    void print() {
        std::cout << "Special Int/Float Processing: " << (value * 2) << std::endl;
    }
};

// (2) General template
template <typename T>
struct Storage {
    T value;
    void print() { std::cout << "General Type Value: " << value << std::endl; }
};

// (3) Int specialization inherits from the helper
template <>
struct Storage<int> : public SpecialMathBehavior<int> {};

// (4) Float specialization inherits from the helper
template <>
struct Storage<float> : public SpecialMathBehavior<float> {};
```

This is more verbose but works in older C++ versions.

### Approach 3 — Brute force copy-paste (works but ugly)

If I do not want concepts or inheritance, I just write both specializations completely separately:

```cpp
template <typename T>
struct Storage { T value; void print() { ... } };   // general

template <>
struct Storage<int> {
    int value;
    void print() {
        std::cout << "Special Math: " << (value * 2) << std::endl;
    }
};

template <>
struct Storage<float> {
    float value;
    void print() {
        std::cout << "Special Math: " << (value * 2) << std::endl;   // exact same body
    }
};
```

It works, it is just a maintenance burden. Best avoided in real code.

---

<a id="10-scope"></a>

## 10. Where can templates live? Templates inside vs outside structs

A clarification I needed: the scope of `template <typename T>` is the immediate block following the declaration. It cannot leak out into the surrounding code.

```cpp
template <typename T>
struct MyStruct {
    T value;          // OK — T is in scope inside the struct body
};
// after this closing brace, T no longer exists

// ERROR — `T` is not in scope here
T global_variable;
```

If I want to write multiple independent template functions in the same file, I write a fresh `template <typename T>` for each one:

```cpp
template <typename T>
T func1(T a) { return a; }

// scope of the first T ends at the function's closing brace

template <typename T>           // this is a brand-new T, unrelated to the first
T func2(T a, T b) { return a + b; }
```

I can also have non-templated structs that contain a single templated method:

```cpp
struct RegularStruct {
    int non_templated_var;     // always int

    template <typename T>      // this single method is templated
    void print_value(T val) {
        std::cout << val;
    }
};

RegularStruct r;
r.print_value(5);          // T = int
r.print_value(3.14);       // T = double
r.print_value("hello");    // T = const char*
```

The whole struct is not a template — only that one method is.

---

<a id="11-headers"></a>

## 11. Why templates MUST live in header files

This is the single most important rule of template programming, and the reason our `Reduction.h` directly includes `ReductionImpl.h`. To understand it I have to first understand how C++ compiles code.

### The C++ compilation model

Each `.cpp` file is compiled **completely in isolation** as a separate **Translation Unit (TU)**. The compiler turns each `.cpp` into a `.o` (object) file with no knowledge of any other `.cpp` file in the project. After all `.cpp` files are compiled, a separate program called the **linker** glues all the `.o` files together into the final binary.

```
main.cpp ─► gcc ─► main.o ─┐
                            ├─► linker ─► final executable
math.cpp ─► gcc ─► math.o ─┘
```

`main.cpp` has zero knowledge of what is inside `math.cpp` during compilation. The only way they communicate is through what is declared in `.h` headers that both files include.

### Case A — Templates defined inside header files

```cpp
// Math.h
#pragma once
template <typename T>
T add(T a, T b) {
    return a + b;   // the entire blueprint is here
}
```

Now if `main.cpp` includes `Math.h` and calls `add(5, 10)`:

1. The compiler processes `main.cpp` and sees the full blueprint of `add` (because the header was textually pasted in).
2. The compiler deduces `T = int` and **immediately instantiates** `add<int>` into `main.o`.

It works perfectly. **No instantiation step needed — the compiler handles it automatically.**

### Case B — Template DECLARATION in header, DEFINITION in source

```cpp
// Math.h
#pragma once
template <typename T>
T add(T a, T b);   // only the promise — no body!
```

```cpp
// Math.cpp
#include "Math.h"
template <typename T>
T add(T a, T b) {
    return a + b;   // body is hidden inside this .cpp
}
```

Now if `main.cpp` includes `Math.h` and calls `add(5, 10)`:

1. The compiler processes `main.cpp` and sees only the declaration `T add(T a, T b)`. It does NOT have the body, so it cannot generate any binary code for `add<int>`.
2. It assumes "the body must exist somewhere — probably `math.o` has the compiled `add<int>` waiting." It leaves a placeholder in `main.o` saying *"hey linker, find `add<int>` for me later"*.
3. The compiler processes `Math.cpp` and sees the body of the template, but `Math.cpp` itself NEVER calls `add<int>` anywhere. So the compiler thinks: "no one needs `add<int>`, I won't generate it." `math.o` ends up empty.
4. The linker tries to combine `main.o` and `math.o`. It looks for `add<int>` in `math.o` and finds nothing. **CRASH** with the dreaded error:

   ```
   undefined reference to `int add<int>(int, int)`
   ```

### The fix — explicit template instantiation

To make Case B work, I have to force `Math.cpp` to generate the binary code for specific types by adding **explicit instantiations** at the bottom of the source file:

```cpp
// Math.cpp
#include "Math.h"
template <typename T>
T add(T a, T b) {
    return a + b;
}

// Explicit instantiations — tells the compiler:
// "Generate the binary for these specific types right now!"
template int    add<int>(int, int);
template float  add<float>(float, float);
template double add<double>(double, double);
```

Now `math.o` contains compiled binaries for `add<int>`, `add<float>`, `add<double>`. The linker finds them. Done.

**This is why our `ReductionImplGPU.cu` has over 250 explicit instantiation lines at the bottom — one for each (dtype, operator) combination.** Without them, every call to `reduce_sum<float, SumOp>` would fail at link time.

---

<a id="12-cuda-templates"></a>

## 12. CUDA templates — the twist

CUDA templates work mostly the same way as C++ templates, but with two extra complications.

### Complication 1 — Two-stage compilation

`nvcc` (NVIDIA's CUDA compiler) compiles each `.cu` file in **two separate stages**:

1. **Host code (CPU)**: passed to a normal C++ compiler (`gcc`/`clang`) and compiled like any other C++ code.
2. **Device code (GPU)**: passed to NVIDIA's GPU compiler that generates **PTX assembly** and **SASS machine code** specific to the target GPU architecture (e.g., `sm_86` for RTX 3060, `sm_89` for RTX 6000 Ada, `sm_90` for H100).

```
source.cu ─► nvcc splits ─► host code → gcc  → host.o
                         └► device code → ptxas → device.o (PTX/SASS for GPU)
                                                            │
                            (linker stitches together) ◄───┘
```

### Complication 2 — Compile-time explosion

CUDA device code compilation is **MASSIVELY** heavier than CPU compilation. `nvcc` performs:

- Register allocation across thousands of threads
- Loop unrolling for warp-level parallelism
- Shared memory bank conflict analysis
- Tensor core instruction scheduling
- Per-architecture (sm_86, sm_89, sm_90) optimization passes

If I put a big template kernel fully in a `.cuh` header and 20 different `.cu` files include it, then **every single one of those 20 files re-compiles the entire device pipeline from scratch**. Compile times balloon from 2 minutes to 2 hours. Binary sizes balloon to gigabytes.

### How professional CUDA codebases solve it (and how we do too)

The standard pattern is to enforce a strict **separation of declaration and definition** with explicit instantiation:

1. **Header (`.h` / `.cuh`)** — declarations only:

   ```cpp
   // ReductionImpl.h
   template <typename T, typename Op>
   void launch_reduction_gpu(Tensor& input, Tensor& output);
   ```

2. **Source file (`.cu`)** — the actual kernel body + explicit instantiations:

   ```cpp
   // ReductionImplGPU.cu
   #include "ReductionImpl.h"
   #include "ReductionKernels.cuh"

   template <typename T, typename Op>
   void launch_reduction_gpu(Tensor& input, Tensor& output) {
       // ... layout analysis ...
       unified_reduce_kernel<T, Op><<<grid, block>>>(...);
   }

   // Explicit instantiations — one line per (dtype, operator) pair
   template void launch_reduction_gpu<float,    SumOp<float>>(Tensor&, Tensor&);
   template void launch_reduction_gpu<half,     SumOp<half>>(Tensor&, Tensor&);
   template void launch_reduction_gpu<bfloat16, SumOp<bfloat16>>(Tensor&, Tensor&);
   template void launch_reduction_gpu<float,    MaxOp<float>>(Tensor&, Tensor&);
   template void launch_reduction_gpu<int32_t,  SumOp<int32_t>>(Tensor&, Tensor&);
   // ... and so on for every type combo we want to support ...
   ```

Because of this pattern:

- `Reduction.cpp` (host code calling into the GPU layer) only sees the declaration. It cannot template-instantiate on its own. It depends on `ReductionImplGPU.cu` having already generated all the variants.
- The GPU kernel code only gets compiled ONCE (inside the single `.cu` file), not 20 times across 20 includes.
- Compile times stay sane and binary stays small.

### CUDA also has `-rdc=true` (relocatable device code)

Without the `-rdc=true` flag (which is the default), each `.cu` file must be a **self-contained GPU executable block**. A `.cu` file cannot launch a `__global__` kernel defined in another `.cu` file unless it includes the full source. With `-rdc=true` enabled, you get separate device compilation with a separate **device linking** step (`nvlink`) that connects them together. Our codebase uses `-rdc=true` for some modules but the explicit-instantiation pattern in `.cu` files is still preferred because it keeps compile times reasonable.

### Direct comparison: C++ vs CUDA template compilation

| Feature | Standard C++ | CUDA (`nvcc`) |
|---|---|---|
| If template is fully in header (`.h` / `.cuh`) | Instantiated automatically, no manual step | Same, BUT forces heavy GPU compilation in every including file — slow builds |
| If template definition is in source (`.cpp` / `.cu`) | Linker fails — needs explicit instantiation | Same — needs explicit instantiation, plus device linker may need `-rdc=true` |
| Compilation pipeline | 1 pass directly to host assembly | 2 passes — device pass to PTX/SASS, host pass to CPU object code |
| Code isolation | Sequential execution shares a CPU stack | Device code physically separated from host, kernels launched via opaque tokens |

### Summary of how our codebase splits work

| File | What it contains | Compiled by |
|---|---|---|
| `Reduction.h` | non-templated public function declarations + includes `ReductionImpl.h` | gcc/g++ |
| `ReductionOps.h` | templated operator structs (SumOp, MinOp, etc.) | gcc/g++ + nvcc (both see this) |
| `ReductionImpl.h` | templated CPU dispatch + cascade_sum_kernel definitions | gcc/g++ |
| `ReductionKernels.cuh` | templated `__global__` CUDA kernels | nvcc only |
| `Reduction.cpp` | non-templated function bodies, runtime dtype dispatch, CPU side | gcc/g++ |
| `ReductionImplGPU.cu` | templated GPU dispatch wrappers + 250+ explicit instantiations | nvcc only |

---

<a id="13-reduction-op-selector"></a>

## 13. How our codebase uses all of this — the `ReductionOpSelector` pattern

Putting everything together, here is how a real call to `reduce_sum` works in our codebase, end to end.

### The lookup table

In `ReductionOps.h` at the bottom we have:

```cpp
// Primary template — intentionally empty, must use a specialization
template <ReductionType R, typename T>
struct ReductionOpSelector;

// Specialization for each operation
template <typename T> struct ReductionOpSelector<ReductionType::SUM,    T> { using type = SumOp<T>; };
template <typename T> struct ReductionOpSelector<ReductionType::MIN,    T> { using type = MinOp<T>; };
template <typename T> struct ReductionOpSelector<ReductionType::MAX,    T> { using type = MaxOp<T>; };
template <typename T> struct ReductionOpSelector<ReductionType::ALL,    T> { using type = AllOp<T>; };
template <typename T> struct ReductionOpSelector<ReductionType::ANY,    T> { using type = AnyOp<T>; };
template <typename T> struct ReductionOpSelector<ReductionType::ARGMAX, T> { using type = ArgMaxOp<T>; };
// ... and so on for all 14 reduction types
```

This is a **compile-time lookup table**. When some code writes `ReductionOpSelector<ReductionType::SUM, float>::type`, the compiler resolves it to `SumOp<float>` at compile time — no runtime cost whatsoever.

### Why this design exists

In our public API I want to call `reduce_sum(tensor)` cleanly without writing 14 different `switch` statements for every operation × dtype combo. The selector lets me write **one** generic dispatcher that takes the runtime dtype and the reduction enum and resolves both to the right templated operator at compile time.

### Why even pure-boolean ops like `AllOp` are templated `<typename T>`

A confusion I had: `AllOp` only operates on `bool` internally. Why is it written as `template <typename T> struct AllOp`?

Look at `AllOp`:

```cpp
template <typename T>
struct AllOp {
    using AccT = bool;                          // accumulator is always bool
    DEVICE_HOST bool identity() const { return true; }
    DEVICE_HOST bool reduce(const bool& a, const bool& b) const { return a && b; }
};
```

The `<typename T>` looks pointless because the body never uses `T`. But it is **load-bearing for the dispatcher**. All our generic reduction kernels are written expecting the operator to match the signature `OpType<T>`:

```cpp
template <typename T, typename OpType>
void launch_reduction_kernel(...);    // expects OpType<T> to be templated on T
```

If `AllOp` was a plain non-templated `struct AllOp`, calling `launch_reduction_kernel<float, AllOp<float>>` would fail to compile because `AllOp<float>` would be invalid (you cannot instantiate a non-template with `<float>`).

So we give `AllOp` a **dummy template parameter** `<typename T>` just to fit the unified pipeline. Even if I instantiate it as `AllOp<float>` or `AllOp<int>`, the internal accumulator is still hardcoded to `bool` and the math still uses `&&`.

### trace ----> what happens when I call `reduce_all(my_float_tensor)`

1. The user calls `reduce_all(input)` where `input` is a float tensor.
2. Inside `Reduction.cpp`, the dispatcher does a runtime switch on `input.dtype()`:

   ```cpp
   switch (input.dtype()) {
       case Dtype::Float32: dispatch_reduction<float, AllOp>(input, ...); break;
       case Dtype::Int32:   dispatch_reduction<int32_t, AllOp>(input, ...); break;
       // ... more cases ...
   }
   ```

3. At compile time, the dispatcher already knows the operator family is `AllOp`, so the compiler generates concrete versions of `dispatch_reduction<float, AllOp<float>>`, `dispatch_reduction<int32_t, AllOp<int32_t>>`, etc.
4. For the float case, the kernel launches as:

   ```cpp
   launch_reduction_kernel<float, AllOp<float>>(input, output);
   ```

5. Inside the kernel:
   - Each float element is loaded.
   - It is converted to `bool` using the implicit conversion rule (0 → false, non-zero → true).
   - The bools are reduced with logical AND (`&&`).
   - The final output tensor is a single `bool` value.

The dummy template parameter is a **glue trick** that makes the unified kernel pipeline work for both numeric ops (sum, min, max) and boolean ops (all, any) without needing separate dispatcher hierarchies.

---

<a id="14-reduction-h-includes"></a>

## 14. Why `Reduction.h` includes `ReductionImpl.h`

If you look at `Reduction.h` at lines 12-14:

```cpp
// CRITICAL STEP: Include the implementation header which contains ALL template definitions.
// This allows the non-template functions in reductions.cpp to instantiate the templates.
#include "ops/helpers/ReductionImpl.h"
```

Translating the comment in plain English:

- **"Include the implementation header which contains ALL template definitions"** — `ReductionImpl.h` is the massive header (3,600+ lines) containing the actual template code for `cascade_sum_kernel`, `dispatch_reduction`, `reduce_kernel_mean`, etc. It is the blueprint.

- **"This allows the non-template functions in reductions.cpp to instantiate the templates"** — the public functions in `Reduction.cpp` (like `reduce_sum`, `reduce_mean`) are **non-templated**. They do runtime dtype dispatch and then call the templated internal `dispatch_reduction<T, Op>`. For the compiler to generate the binary for `dispatch_reduction<float, SumOp>` while compiling `Reduction.cpp`, it MUST have the full template blueprint visible — which means the `.h` file containing the body must be included before that call is parsed.

### Why is it included in the header (`Reduction.h`) instead of just in `Reduction.cpp`?

 question: if only `Reduction.cpp` needs the blueprints, why not just include `ReductionImpl.h` directly in `Reduction.cpp` and leave `Reduction.h` clean?

Two reasons:

**Reason A — header self-containment** — in a large codebase, headers should be **self-contained**. Any file (a test file, an autograd file, a debug file) that includes `Reduction.h` should compile perfectly without the developer needing to remember to include a chain of helper headers in a specific order. By including `ReductionImpl.h` inside `Reduction.h`, the compiler automatically pulls in the blueprints the moment `Reduction.h` is included anywhere.

**Reason B — enabling app-level template usage** — although our public functions are non-templated, someone writing a test (like `bench_reduce_sum.cpp`) or a custom autograd op might want to call the templated machinery directly. By making the include part of `Reduction.h`, any test or extension file that just includes the public header automatically gets access to both the public API and the internal templated utilities.

### What happens if I remove the include from `Reduction.h`?

The compiler compiling `Reduction.cpp` would encounter `dispatch_reduction<T, SumOp>` and would not be able to find the implementation. Immediate compilation error — the project would not build.

---

<a id="15-reduction-cpp-includes-again"></a>

## 15. Why `Reduction.cpp` includes `ReductionImpl.h` again

Observation and doubt :  If `Reduction.cpp` already includes `Reduction.h`, and `Reduction.h` already includes `ReductionImpl.h`, then `Reduction.cpp`'s explicit `#include "ops/helpers/ReductionImpl.h"` looks **redundant**. The preprocessor would pull it in either way, and header guards (`#pragma once`) would silently skip the second include anyway. The code would compile perfectly fine without that explicit line.

So why do professional C++ developers add it anyway?

**Reason 1 — the "Include What You Use" (IWYU) rule** — in high-quality C++ development, if a `.cpp` file directly calls a function defined in a specific header, that header MUST be explicitly included in the `.cpp` file. Inside `Reduction.cpp` we directly call `detail::dispatch_reduction` and `detail::reduce_kernel_mean` which are defined in `ReductionImpl.h`. The explicit include makes it crystal clear to anyone reading the file: *"this source file directly depends on code from ReductionImpl.h."* It serves as self-documentation.

**Reason 2 — protection against future refactoring (brittle code prevention)** — relying on **transitive includes** (headers including other headers that the source code actually uses) makes codebases incredibly fragile. Imagine six months from now I decide to clean up `Reduction.h` and remove the internal `#include "ops/helpers/ReductionImpl.h"`. If `Reduction.cpp` was relying entirely on that transitive include, it would suddenly fail to compile with hundreds of errors. With the explicit include in `Reduction.cpp`, I can refactor `Reduction.h` freely without breaking the source file.

**Reason 3 — keeping IDEs and autocomplete smart** — modern IDEs (Clangd, VSCode, CLion) parse files to provide Go-to-Definition, Autocomplete, and Refactoring features. If a `.cpp` file relies on a deep chain of hidden includes to see a symbol, static analyzers sometimes get confused, show false red squigglies, or fail to find where the code is defined. Explicit includes give the tooling the clearest possible picture of which file defines which symbol.

So the duplicate include is intentional good practice, not a mistake.

---

<a id="16-cheat-sheet"></a>

## 16. Quick reference cheat sheet

### Syntax patterns

| Pattern | Example |
|---|---|
| Function template | `template <typename T> T add(T a, T b);` |
| Class/struct template | `template <typename T> struct Box { T value; };` |
| Full function specialization | `template <> bool add<bool>(bool a, bool b) { ... }` |
| Full struct specialization | `template <> struct Box<bool> { ... };` |
| Non-type template parameter | `template <typename T, int N> struct Array { T data[N]; };` |
| Multiple template parameters | `template <typename T1, typename T2> auto add(T1 a, T2 b);` |
| Explicit instantiation | `template int add<int>(int, int);` |
| Compile-time `if` (C++17+) | `if constexpr (std::is_same_v<T, bool>) { ... }` |
| Concept constraint (C++20+) | `if constexpr (std::same_as<T, int> \|\| std::same_as<T, float>) { ... }` |

### Rules to remember

1. **Templates must be visible (defined) wherever they are used.** Hence template code lives in headers. Defining templates in `.cpp` files requires explicit instantiation.

2. **You can overload functions, but you cannot overload structs/classes.** That is why structs must be templated to support multiple types.

3. **Template instantiation happens at compile time** with zero runtime overhead. The final binary contains hand-written-equivalent specialized code.

4. **`if constexpr` discards the unused branch at compile time** — perfect for type-dependent behavior inside a templated function.

5. **CUDA splits compilation into host and device passes.** Templates in `.cuh` headers get recompiled by every including `.cu` file, leading to slow builds. So we put kernel definitions in `.cu` source files and add explicit instantiations at the bottom — this is why `ReductionImplGPU.cu` has 250+ instantiation lines.

6. **`std::boolalpha` only affects bool printing.** It is a stream-level filter that has no effect on `int`, `float`, `char`, etc.

7. **Integer-to-bool conversion is automatic.** `0` becomes `false`, any non-zero becomes `true`.

8. **The dummy `<typename T>` on `AllOp<T>` and `AnyOp<T>` exists for pipeline uniformity**, not because the operator actually uses `T`. This lets the generic kernel signature `OpType<T>` work for both numeric and boolean operators.

### Glossary

| Term | Meaning |
|---|---|
| Template | A blueprint that the compiler uses to generate concrete code |
| Instantiation | The compile-time act of generating concrete code from a template for a specific type |
| Specialization | An override of the general template blueprint for one specific type |
| TU (Translation Unit) | One `.cpp` file plus everything it includes — the unit of compilation |
| Explicit instantiation | Manually telling the compiler `template T func<T>(T);` to force generation of specific types |
| Linker error | Compile succeeds but final linking fails because the linker cannot find a symbol |
| `if constexpr` | Compile-time `if` that discards the unused branch — used for type-dependent code in templates |
| CTAD | Class Template Argument Deduction (C++17) — letting the compiler figure out the type from constructor args |
| Concept (C++20) | A named compile-time predicate on a template type, used to constrain which types a template accepts |
| PTX | NVIDIA's intermediate assembly language for GPUs — generated by `nvcc` before final machine code |
| SASS | NVIDIA's actual machine code that runs on the GPU streaming multiprocessors |
| `-rdc=true` | nvcc flag for relocatable device code — allows separate compilation and device linking |
| nvlink | NVIDIA's linker for device code, runs when `-rdc=true` is used |

---
