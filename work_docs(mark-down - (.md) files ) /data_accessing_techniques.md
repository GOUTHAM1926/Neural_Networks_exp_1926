# Understanding Tensors: Linear Index vs. Physical Offset

To understand high-performance GPU reduction algorithms, you must first understand how a computer stores a multi-dimensional tensor in its physical memory (RAM).

There are two completely different ways to locate an element in a tensor:

1. **The Logical Linear Index (`linear_idx`):** The count/step inside our logical loop.
2. **The Physical Offset (`offset`):** The actual address/index of the data box inside the physical RAM.

Let's understand the difference between these two using a simple **3x3 tensor with values 1 to 9**.

---

## 1. The Raw Memory (RAM)

No matter how we shape our tensor (2D, 3D, transposed), the computer's memory (RAM) is always a flat, 1D line of boxes.

The boxes contain our numbers **1 to 9**:

```
RAM Box Index (Offset):   [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
Values inside:             1    2    3    4    5    6    7    8    9
```

The box index numbers `0, 1, 2, ... 8` are the **Physical Offsets**.

---

## 2. The Logical Tensor (The Grid)

For humans and deep learning libraries, we view this flat memory as a 2D grid:

```
Row 0:  [ 1  2  3 ]
Row 1:  [ 4  5  6 ]
Row 2:  [ 7  8  9 ]
```

When a GPU thread block or a CPU loop runs, it walks through this grid.
The loop index is the **Linear Index** (`linear_idx`). It always goes sequentially from `0` to `8`:

```
linear_idx = 0  ──► maps to (row 0, col 0) ──► value 1
linear_idx = 1  ──► maps to (row 0, col 1) ──► value 2
linear_idx = 2  ──► maps to (row 0, col 2) ──► value 3
linear_idx = 3  ──► maps to (row 1, col 0) ──► value 4
linear_idx = 4  ──► maps to (row 1, col 1) ──► value 5
linear_idx = 5  ──► maps to (row 1, col 2) ──► value 6
linear_idx = 6  ──► maps to (row 2, col 0) ──► value 7
linear_idx = 7  ──► maps to (row 2, col 1) ──► value 8
linear_idx = 8  ──► maps to (row 2, col 2) ──► value 9
```

---

## Case 1: Contiguous Tensor (Normal Layout)

If the tensor is stored normally in row-major order:

* `stride_row = 3` (moving down 1 row means jumping 3 boxes in RAM)
* `stride_col = 1` (moving right 1 column means jumping 1 box in RAM)

Let's find the location for **`linear_idx = 5`**:

1. **Logical Coordinate:** We decompose `linear_idx = 5` into `(row 1, col 2)`. This points to the value **6**.
2. **Calculate Offset in RAM:**
    $$\text{Offset} = (\text{row} \times \text{stride\_row}) + (\text{col} \times \text{stride\_col})$$
    $$\text{Offset} = (1 \times 3) + (2 \times 1) = \mathbf{5}$$
3. **Read Memory:** We check `RAM[5]`. The value inside is **6**.

In a normal contiguous layout:

* **`linear_idx` = 5**
* **`offset` = 5**
* They are exactly the same!

---

## Case 2: Transposed Tensor (Non-Contiguous Layout)

Now, let's transpose the tensor. We want the rows to become columns, and columns to become rows.

The logical grid now looks like this:

```
Row 0:  [ 1  4  7 ]
Row 1:  [ 2  5  8 ]
Row 2:  [ 3  6  9 ]
```

### ⚠️ Crucial Rule: Transposing does NOT move data in RAM

To keep operations extremely fast (O(1) time), PyTorch, Eigen, and our library **do not rearrange or copy the numbers in RAM** when you transpose. The RAM still looks exactly like this:

```
RAM Box Index (Offset):   [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]
Values inside:             1    2    3    4    5    6    7    8    9
```

Because the data did not move, but our logical grid changed, we must swap the strides to read it correctly:

* `stride_row = 1` (moving down 1 row means jumping 1 box in RAM, e.g., 1 to 2 to 3)
* `stride_col = 3` (moving right 1 column means jumping 3 boxes in RAM, e.g., 1 to 4 to 7)

Let's find the location for **`linear_idx = 5`** in this transposed tensor:

1. **Logical Coordinate:** `linear_idx = 5` still maps to `(row 1, col 2)`. Looking at our transposed grid, `(row 1, col 2)` should contain the value **8**.
2. **Calculate Offset in RAM:**
    $$\text{Offset} = (\text{row} \times \text{stride\_row}) + (\text{col} \times \text{stride\_col})$$
    $$\text{Offset} = (1 \times 1) + (2 \times 3) = 1 + 6 = \mathbf{7}$$
3. **Read Memory:** We check `RAM[7]`. The value inside is **8**. It works perfectly!

Look at the difference now:

* **`linear_idx` = 5**
* **`offset` = 7**
* They are completely different!

---

## The Modulo / Division Pipeline

To go from a flat thread index to the actual physical memory address, the GPU must perform two transformations back-to-back:

1. **Step 1: Decompose (`linear_idx` ──► Coordinates)**
    * Uses **Modulo (%) and Division (/)** based on tensor sizes.
    * Example: `5` becomes `(row 1, col 2)`.
2. **Step 2: Map (Coordinates ──► Physical `offset`)**
    * Uses **Multiplication (*) and Addition (+)** based on tensor strides.
    * Example: `(row 1, col 2)` with strides `[1, 3]` becomes `offset = 7`.

### Important: When do we actually need this Modulo-division thing and when we dont ?

As I showed in Case 1 and Case 2 above, for **contiguous tensors**, `linear_idx` and `offset` are always the same number. If `linear_idx = 5`, then `offset = 5`. They match perfectly. So I **do not need** to do any of this modulo/division math at all! I can just use `linear_idx` directly as my memory address — no coordinate decomposition, no stride multiplication, nothing. It is a straight, fast memory read.

But for **non-contiguous tensors** (like transposed tensors), `linear_idx` and `offset` are **different numbers**. If `linear_idx = 5`, the `offset` might be `7` (as I showed in the transpose example). The only way to find the correct `offset` is to:

1. First decompose `linear_idx` into coordinates using modulo and division.
2. Then multiply those coordinates by the (now-swapped) strides and add them up.

This is exactly why this whole modulo/division pipeline exists — it is **only needed for non-contiguous tensors** where the logical ordering of elements in our representation(tensor logical view) does not match the physical ordering in RAM. For contiguous tensors, the GPU skips all of this and reads memory directly, which is much faster.

In my reduction kernels, the exact same data accessing pattern is followed. When I reduce a contiguous tensor (like summing all elements of a normal tensor), the kernel takes the fast path — it reads elements directly using `linear_idx` as the memory address, no modulo or division needed. But when I reduce a non-contiguous tensor (like summing along a dimension of a transposed tensor), the kernel has to take the slow path — it must use the `OffsetCalculator` with its modulo/division loop (the exact code in `ReductionKernels.cuh`) to convert each thread's `linear_idx` into the correct physical `offset` in RAM. This is why my reduction module has multiple data access paths — contiguous paths that skip the math, and a generic path that does the full modulo/division pipeline for non-contiguous cases.

**Note on a faster alternative:** Currently my generic path uses **naive hardware modulo (`%`) and division (`/`)** instructions, which are very slow on GPUs (~30-40 cycles each, not pipelined). There is a well-known algorithm from the paper *"Division by Invariant Integers using Multiplication"* by Granlund & Montgomery (1994), also described in the book *Hacker's Delight* by Henry S. Warren, that replaces these expensive division/modulo operations with a **precomputed magic multiplier and bit-shift** — turning a ~30-40 cycle division into a ~6-8 cycle multiply-high (`__umulhi`) + shift. Major frameworks already use this optimization:

* **PyTorch** uses it in `IntegerDivider.cuh` (their `IntDivider` class with `__umulhi`).
* **Eigen** uses it in `TensorIntDiv.h` (their `TensorIntDivisor` class with `__umulhi` on GPU).
* **TensorFlow** uses it in `gpu_kernel_helper.h` (their `FastDividerAndModulo` class).

I plan to upgrade my `OffsetCalculator` to use this fast division technique later. I will document the algorithm in detail when I implement it.

---

## Logical Programming  Design  vs. Physical Memory  Reality

This is the key insight I realized about high-performance tensor computing:

* **In my Logical Mind:** When I transpose a tensor, I imagine the elements are physically rearranged. I think of `linear_idx` as stepping smoothly through this new logical grid (`linear_idx` = 0, 1, 2, 3...).
* **In Physical Reality (RAM):** The data **did not move**. It is still in the exact same contiguous linear layout as it was originally.
* **The Bridge:** Because my logical imagination has diverged from physical reality, sequential steps in my loop (`linear_idx` = 0, 1, 2, 3...) translate into **non-sequential jumps in raw RAM** (`offset` = 0, 3, 6...).

This mismatch is exactly why high-performance tensor engines need **strides** and **offset calculators** to bridge the gap between my logical imagination and raw hardware storage! And as I noted above, this entire modulo/division machinery is only triggered when dealing with non-contiguous tensors — for contiguous tensors, the engine takes a fast shortcut and skips all of it.

---

## 3.Understanding Modulo-division co-ordinate finding math  : visual

How does the computer actually break down a single flat number (`linear_idx`) into multi-dimensional coordinates using modulo (`%`) and division (`/`)? Let's trace it with the exact 3D tensor of shape **`[2, 3, 4]`** and **`linear_idx = 14`** (the 15th element).

### The Number Analogy (Base-10)

Think about the number **143**. To extract its digits starting from the **rightmost** (innermost):

1. **Ones digit:** `143 % 10 = 3` ──► `143 / 10 = 14`
2. **Tens digit:** `14 % 10 = 4` ──► `14 / 10 = 1`
3. **Hundreds digit:** `1 % 10 = 1` ──► `1 / 10 = 0` (Done!)

We peeled off the digits one-by-one to get **(1, 4, 3)**.

### The Mixed-Base Tensor Analogy (`[2, 3, 4]`)

A tensor of shape `[2, 3, 4]` is just like a mixed-base number system:

* **Column (innermost):** size **4** (base-4)
* **Row (middle):** size **3** (base-3)
* **Depth (outermost):** size **2** (base-2)

Let's decompose **`linear_idx = 14`** step-by-step, highlighting the **Golden Intuition** behind the intermediate values:

#### Step 1: Innermost Dimension (Columns, size = 4)

* `14 % 4 = 2` ──► **Column coordinate is 2** (The 3rd column).
  * *Visual:* We arrange our elements in rows of 4 columns. Index 14 falls into the 3rd slot (Column 2).
* `14 / 4 = 3` ──► **Overall row index is 3**.
  * * THE DUAL-INTUITION:* This intermediate result `3` represents **two physical and logical concepts simultaneously**:
    1. **Completed Packets:** It tells us we have exactly **3 complete rows** (packets of 4 elements) fully behind us (Row 0, Row 1, and Row 2).
    2. **Upcoming Index:** It tells us the **index of the row we are currently standing on** in the next dimension. Since index starts at 0, standing in front of 3 complete rows means we are standing exactly on **Row 3**!
    * *(Personal doubt explored:* Wait, how can one number `3` represent both "how many rows are behind us" AND "what row we are standing on"? That feels like a coincidence. But it is not a coincidence — it is a direct consequence of how computers count things. In programming, we always start counting from **0** (the first row is called Row 0, the second row is called Row 1, the third row is called Row 2, and so on). Because of this, if 3 rows are behind us, we are standing on "Row 3" — the count and the index are **always the same number**. But what if someone decided to start counting from **1** instead? (First row = Row 1, second row = Row 2, etc.) Then 3 rows behind us would mean we are standing on "Row **4**" — the count and the index would **no longer match**! We would need extra math operations (`+1` and `-1`) at every single step to fix this mismatch. This is exactly why all programming languages and all GPU hardware use counting-from-0: it eliminates those wasteful extra operations. See **Section 5** below for the full mathematical proof of this.)

```
Overall Row 0:  [ 0,  1,  2,  3 ]  ──► (Row 0 completely behind us)
Overall Row 1:  [ 4,  5,  6,  7 ]  ──► (Row 1 completely behind us)
Overall Row 2:  [ 8,  9, 10, 11 ]  ──► (Row 2 completely behind us)
Overall Row 3:  [ 12, 13, 14, .. ]  ◄── WE ARE STANDING HERE! (Row Index = 3)
```

#### Step 2: Middle Dimension (Rows, size = 3)

We pass the overall row index **3** (representing our 3 completed rows / Row Index 3) into the next layer:

* `3 % 3 = 0` ──► **Row index inside the plane is 0**.
  * *Visual:* Inside our current depth plane (which holds 3 rows), our row index is 0.
* `3 / 3 = 1` ──► **Overall plane index is 1**.
  * * THE DUAL-INTUITION:* This intermediate result `1` again represents **two things at once**:
    1. **Completed Packets:** It tells us we have exactly **1 complete plane** (packet of 3 rows) fully behind us (Plane 0 is completely filled and behind us).
    2. **Upcoming Index:** It tells us the **index of the plane we are currently standing on** in the next higher dimension. Since index starts at 0, standing in front of 1 complete plane means we are standing exactly on **Plane 1**!
    * *(Same logic as Step 1:* 1 plane behind us = standing on Plane 1. The count and the index match perfectly because we count from 0. If we counted from 1, then 1 plane behind us would mean standing on Plane "2", and we would need that wasteful `+1` operation again. See **Section 5** for the proof.)

#### Step 3: Outermost Dimension (Depth, size = 2)

We pass the overall plane index **1** (representing our 1 completed plane / Plane Index 1) into the next layer:

* `1 % 2 = 1` ──► **Depth coordinate is 1**.
  * *Visual:* Inside the 2-plane block, we are standing on Depth Index 1 (the 2nd and last plane).
  * *Same rule as every step:* the **remainder always names the slot we occupy in the current axis** — exactly like `14 % 4 = 2` gave the column and `3 % 3 = 0` gave the row inside the plane, here `1 % 2 = 1` gives the depth. This is the *coordinate* (where we stand), **not** the "how many are behind us" count. That "behind us / standing-on" dual meaning lives in the **quotient** (the division), which we read next.
* `1 / 2 = 0` ──► **Done! (the quotient has dropped to 0 — this is the STOP signal).**
  * * THE DUAL-INTUITION (same as Steps 1 and 2, except now it lands on `0`):* the quotient `0` still means the very same two things:
    1. **Completed Packets:** we have exactly **0 complete 2-plane blocks** behind us in any higher dimension.
    2. **Upcoming Index:** we are standing on **index 0** of the next dimension up.
  * *Why this ends the unravel:* this was the **outermost** dimension, so there is no "next dimension up" left to feed that `0` into. A quotient of `0` here is telling us that every higher coordinate is `0` and there is nothing more to peel off — the element is now **fully located**. So the remainder named our last real coordinate (depth = 1), and the quotient reaching `0` is simply the **terminator** that says "finished".

> #### What if the tensor had MORE dimensions stacked above this one? (a 4th, 5th, 6th axis …)
>
> Nothing new happens — the **exact same two operations just keep running**, and the value simply stays `0` all the way up. Suppose above "depth" we had a 4th axis of some size `S`. We would carry our quotient `0` into it and do:
>
> * `0 % S = 0` ──► the coordinate on that 4th axis is **0** (we are sitting on its very first slot).
> * `0 / S = 0` ──► still **0** completed blocks behind us, and still index **0** for the axis above that.
>
> And it would be identical for a 5th axis, a 6th axis, and so on — **every higher coordinate comes out as `0`**. And that is exactly correct: element 14 genuinely lives in the *first* slab of every dimension sitting above depth. The `0` is not a bug or a leftover — it is the true coordinate for all those outer axes.
>
> This is the whole reason our GPU loop is bounded by the **number of dimensions** (`for (d = dims-1; d >= 0; d--)`) and **NOT** by "stop as soon as the running value becomes `0`". If we stopped early the moment the value hit `0`, those outer coordinates would never be written — they would be left as garbage (uninitialised memory) and the computed address would be wrong. By always running the full `dims` passes, each remaining outer coordinate is correctly stamped with its `0`. The complete worked example of this — `linear_idx = 2` unravelling to `(0, 0, 2)`, where the running value becomes `0` early but the loop must continue — is in **Section 4: *"Why loop termination is bound by `dims`, not when `linear_idx == 0`"***.

---

## 4. Mathematical Proofs & Loop Edge-Cases

### Why loop termination is bound by `dims`, not when `linear_idx == 0`

In our high-performance GPU kernels, the loop runs exactly `dims` times (`for (int d = dims - 1; d >= 0; d--)`). It **does not stop** when `linear_idx` reaches `0`.

#### The Proof

Suppose we have our 3D tensor of shape `[2, 3, 4]`, and a thread is assigned to **`linear_idx = 2`**.

If we run the loop:

1. **Dimension 2 (size 4):**
    * `coord = 2 % 4 = 2`
    * `linear_idx = 2 / 4 = 0` (Our loop variable is now `0`!)
2. **Dimension 1 (size 3):**
    * If we stopped because `linear_idx == 0`, we would exit now. The middle row coordinate would never be set, leaving it with uninitialized memory!
    * By continuing: `coord = 0 % 3 = 0`.
3. **Dimension 0 (size 2):**
    * `coord = 0 % 2 = 0`.

Thus, we get the correct coordinates `(0, 0, 2)`. We must finish the loop for all dimensions to ensure outer coordinates are explicitly set to `0` and multiplied by their strides.

Furthermore, having a fixed-size loop prevents **warp divergence** on the GPU, allowing the compiler to unroll the instructions for maximum speed.

#### First — what *is* "the loop"?

"The loop" is the small piece of code that turns **one flat number** (`linear_idx`) **into the separate per-dimension coordinates**. In our kernels it is written like this:

```cpp
// dims    = number of dimensions of the tensor   (for [2,3,4] this is 3)
// sizes[] = the size of each dimension            (sizes = {2, 3, 4})
// linear_idx = the flat element number we want to decode (e.g. 14)

for (int d = dims - 1; d >= 0; d--) {     // walk dimensions: innermost -> outermost
    coord[d]   = linear_idx % sizes[d];   // REMAINDER = our coordinate on THIS axis
    linear_idx = linear_idx / sizes[d];   // QUOTIENT  = what is left for the axes ABOVE
}
```

* `dims` is **how many axes the tensor has** — for our `[2, 3, 4]` tensor, `dims = 3`.
* `d` is just a counter that walks through the axes. It **starts at the innermost axis** (`d = dims - 1 = 2`, the columns) and steps **down** to the **outermost axis** (`d = 0`, the depth). So `d` takes the values **2, then 1, then 0** — three passes in total.
* Each pass does the one `%` / `/` pair we saw earlier: the **remainder** gives our coordinate on the current axis, and the **quotient** becomes the leftover number we carry up to the next axis.

So the loop is nothing fancy — it just repeats the same "peel off one dimension" step **once for every dimension**, innermost first, outermost last. After `dims` passes, every coordinate is filled in.

#### Why is "3" a CONSTANT? (this is the important bit)

`dims` (which is `3` here) is the **number of axes of the tensor**, and that is **fixed and known before the kernel even starts running**. It comes from the tensor's **shape**, **not** from the data and **not** from which element a thread happens to be working on.

A `[2, 3, 4]` tensor has **3 dimensions for element 0, for element 14, for element 23 — for every single element**. There is no element in that tensor that suddenly has 2 or 4 dimensions. So:

> **every thread loops exactly 3 times**, because every thread is decoding an index into the **same-shaped** tensor. The number `3` never changes from one thread to another, and it never changes based on the value of `linear_idx`.

That is what "the loop count is constant" means: it is decided once by the **shape**, and it is the **same for all 32 threads** in a warp.

Now compare that with the "natural" way of writing the loop — *stop as soon as the leftover value hits 0*:

```cpp
while (linear_idx != 0) {                 // count of passes DEPENDS on the value!
    coord = linear_idx % size;
    linear_idx = linear_idx / size;
}
```

Here the number of passes **depends on how big the index is** — a large index needs more passes than a small one. So the loop count is **NOT constant** across threads. That single difference is what causes the slowdown explained next.

#### What "warp divergence" actually means

**A warp = 32 threads that run in lockstep.** On the GPU, threads are not independent — they are grouped into **warps of 32**, and all 32 threads in a warp execute **the same instruction at the same time**, sharing one program counter. Picture 32 people forced to do the exact same step at the exact same moment.

**Warp divergence = threads in one warp being forced onto different paths.** If the 32 threads need to do *different* things — take different `if` branches, or run a **different number of loop passes** — the hardware can no longer run them together (there is only one shared instruction stream). So it **serialises**: it runs one path with the threads that need it (the rest sit idle, "masked off"), then runs the other path with the remaining threads (now the first group sits idle). 32 threads that should have worked in parallel end up **taking turns** — wasted cycles. That is divergence.

#### Why the *data-dependent* loop diverges (with our `[2,3,4]` example)

Take the `while (linear_idx != 0)` version and look at two threads sitting **in the same warp**:

* thread with `linear_idx = 14` → needs **3** passes (14 → 3 → 1 → 0)
* thread with `linear_idx = 2`  → needs **1** pass  (2 → 0)

Different pass-counts inside the same warp = **divergence**. The warp is forced to keep looping until the **slowest** thread (3 passes) finishes, while the `idx = 2` thread just sits idle (masked) during passes 2 and 3. Spread that across a warp where every thread holds a different index, and a lot of cycles are wasted waiting.

#### Why the FIXED loop avoids it

The `for (int d = dims - 1; d >= 0; d--)` loop runs **exactly `dims` (= 3) passes for every thread, no matter what its index is**. So all 32 threads in the warp do the **identical** amount of work in perfect lockstep — no thread ever waits for another, nothing gets serialised. (And as the proof above shows, the `idx = 2` thread spending its "extra" passes writing the outer `0` coordinates is also what makes it **correct** — so the fixed loop is both *faster* and *safer* at the same time.)

#### Why this also lets the compiler "unroll" the loop — and what "unrolling" actually means

First, **what is loop unrolling?** It is an optimization where the compiler **removes the loop and writes its body out as repeated straight-line code**. A normal loop, besides doing the real work, also runs hidden "bookkeeping" instructions on *every* pass: check the condition (`d >= 0`), update the counter (`d--`), and **jump back** to the top. Those bookkeeping instructions do no useful math — they only keep the loop machinery turning.

If the compiler **knows the loop runs exactly 3 times** (because `dims` is a constant), it can throw that machinery away and simply emit the body **3 times in a row**. For our `[2, 3, 4]` tensor:

**Before (rolled loop)** — every pass also pays for the counter, the condition check, and the branch-back:

```cpp
for (int d = 2; d >= 0; d--) {        // <-- d--, the "d >= 0" check, and a jump-back EVERY pass
    coord[d] = idx % sizes[d];
    idx      = idx / sizes[d];
}
```

**After (unrolled)** — the compiler expands it into 3 straight copies, with the sizes baked in as literal numbers (`d=2`->size 4, `d=1`->size 3, `d=0`->size 2):

```cpp
coord[2] = idx % 4;  idx = idx / 4;   // innermost (columns, size 4)
coord[1] = idx % 3;  idx = idx / 3;   // middle    (rows,    size 3)
coord[0] = idx % 2;  idx = idx / 2;   // outermost (depth,   size 2)
```

Notice what vanished: **no `d`, no `d--`, no `d >= 0` check, no jump-back branch** — just 6 useful instructions in a straight line.

**Why the unrolled version is faster:**

1. **The loop overhead is gone.** No counter updates, no condition checks, no branches — all those wasted "bookkeeping" instructions are simply deleted, so a bigger fraction of the work is *real* work.
2. **No branch = no stalls.** Branches can stall the GPU's instruction pipeline; straight-line code flows through smoothly, and the hardware can overlap the independent instructions (more instruction-level parallelism).
3. **The sizes become compile-time constants** (`% 4`, `/ 3`, `% 2`). This one is big: dividing by a *known* constant lets the compiler swap the slow hardware divide for a fast "multiply-by-a-magic-number" trick — exactly the Granlund-Montgomery optimization described in **Section 6** of this document. A data-dependent loop (where `size` is only known at runtime) **cannot** do this.

And all of this is possible **only because the trip count is the constant `dims`**. If the loop count depended on the data (the `while (linear_idx != 0)` version), the compiler would not know how many copies to write, so it could **not** unroll — and you would pay the loop overhead *plus* the divergence cost on every launch.

**In one line:** looping a *fixed* number of times (= the tensor's number of dimensions) instead of a *data-dependent* number means every thread in a warp does the same work in lockstep — no threads stall waiting for others (no divergence) — and the constant count lets the compiler flatten the loop into fast branch-free code.

---

## 5. 0-Based vs. 1-Based Indexing Symmetry

### Why 0-based indexing is a superpower

In a 0-based system, the number of full packets behind us is **always exactly equal** to our current index.

* If **3** rows are behind us, we are at **Row 3**.
* No extra additions or subtractions are needed. The hardware arithmetic remains fast and clean.

### What if indexing started at 1?

If we use 1-based indexing, the 15th element has `linear_idx = 15` in a `[2, 3, 4]` tensor.

Without shifting, naive division breaks down at boundaries:

* For **Element 12** in a 4-column row:
  * `12 % 4 = 0` (But there is no Column 0! It should wrap to Column 4).
  * `12 / 4 = 3` (But adding 1 gives Row 4, whereas Element 12 is in Row 3).

#### The Shift Formula

To make 1-based indexing work correctly, we must subtract 1 before the operation, and add 1 back to the final result:
$$\text{Coordinate} = (\text{val} - 1) \% \text{size} + 1$$
$$\text{Next Index} = (\text{val} - 1) / \text{size} + 1$$

#### Full Symmetric Trace for Element 15 in 1-based `[2, 3, 4]` layout

1. **Innermost (Columns, size 4):**
    * `col = (15 - 1) % 4 + 1 = 14 % 4 + 1 =` **Column 3**
    * `next_val = (15 - 1) / 4 + 1 = 14 / 4 + 1 =` **4**
2. **Middle (Rows, size 3):**
    * `row = (4 - 1) % 3 + 1 = 3 % 3 + 1 =` **Row 1**
    * `next_val = (4 - 1) / 3 + 1 = 3 / 3 + 1 =` **2**
3. **Outermost (Depth, size 2):**
    * `depth = (2 - 1) % 2 + 1 = 1 % 2 + 1 =` **Depth 2**
    * `next_val = (2 - 1) / 2 + 1 = 0 + 1 =` **1**

We get **`(Depth 2, Row 1, Column 3)`**, which matches the physical layout perfectly! This proves that 1-based indexing is mathematically symmetric, but costs **2 extra ALU operations** (subtraction and addition) at every single dimension step.

---

## 6. Replacing Slow Division with Fast Multiplication (Granlund-Montgomery Algorithm)

In the modulo/division pipeline above (Section 3), every dimension requires one integer division (`/`) and one modulo (`%`) operation. On an NVIDIA GPU, each of these operations stalls the thread for **~30–40 clock cycles** because the GPU has no fast, pipelined integer division hardware — it must compute the quotient bit-by-bit through an iterative loop inside the ALU, just like long division by hand.

This section documents the algorithm I will use to eliminate those slow instructions entirely.

### 6.1 What Exactly I Am Replacing

I need two results from each dimension step:

1. **Quotient (Division):** `linear_idx / dim_size` — gives me the remaining index to pass to the next dimension.
2. **Remainder (Modulo):** `linear_idx % dim_size` — gives me the coordinate within this dimension.

The Granlund-Montgomery algorithm replaces the **division** with a fast multiply-and-shift sequence. Once I have the fast quotient, I get the remainder for free using simple arithmetic:

```
remainder = dividend - (quotient * divisor)
```

This remainder formula uses only one multiplication and one subtraction — both of which are fast, pipelined operations on the GPU (~4 cycles each). So the entire modulo/division pair is replaced.

### 6.2 Why Integer Division is Slow (and Why This Only Matters for Integers)

Not all data types suffer equally from slow division:

* **Floating-point (float, FP32, FP64):** Modern GPUs have dedicated, fast hardware blocks for floating-point division and reciprocals. FP32 division takes roughly 10–14 cycles. It is not free, but it is not a crisis.
* **Integers (uint32, int32, int64):** GPUs do **not** have fast, pipelined integer division units. Integer division stalls the execution pipeline for **~30–40 cycles** (uint32) or **~60–80 cycles** (int64). This is because the hardware must perform an iterative bit-by-bit algorithm internally.

The modulo/division pipeline in my `OffsetCalculator` operates on **tensor dimension sizes and linear indices**, which are always **unsigned 32-bit integers** (`uint32_t`). Tensor indices are whole numbers (0, 1, 2, 3...) — there is no such thing as index `2.5`. This is why the integer division bottleneck matters so much for my reduction kernels.

### 6.3 Why `uint32_t` (32-Bit) Is Sufficient for Tensor Index Math

A `uint32_t` can hold values from `0` to `4,294,967,295` (~4.29 billion). This is more than sufficient because:

* The **divisors** in the `OffsetCalculator` are always individual **dimension sizes** (like batch=8, channels=64, height=1024, width=1024). These are tiny compared to 4.29 billion.
* The **dividends** are logical indices within each dimension's range, not raw memory pointers. They are always bounded by the product of dimension sizes, which for practical deep learning tensors is well within 32-bit range.
* Even for giant models, no single tensor dimension exceeds 4.29 billion. If the **total number of elements** exceeds ~2 billion, frameworks like PyTorch fall back to a slower 64-bit kernel path, but the fast path (which runs 99.9% of the time) uses `uint32_t`.
* Using `uint32_t` instead of `int64_t` is **4x to 6x faster** on the GPU because the GPU's ALUs are natively 32-bit; a 64-bit multiplication requires the hardware to split it into multiple 32-bit operations internally.

### 6.4 The Core Mathematical Trick: Multiply by Reciprocal

Dividing any number `n` by a divisor `d` is mathematically identical to multiplying `n` by the reciprocal `1/d`:

```
n / d  =  n * (1/d)
```

For example, `14 / 4` is the same as `14 * 0.25 = 3.5`, which we round down to `3`.

The problem is that `1/d` is a fraction (like `0.25` or `0.142857...`), and integers cannot store fractions. So I need a way to represent `1/d` as a whole number.

### 6.5 Scaling Up the Reciprocal Into a Whole Number

To turn the fractional reciprocal `1/d` into a whole number, I multiply it by a large **Scaling Factor**. Let's call this scale `2^k`.

The computer needs a whole number to work with, so after we scale it up, we have to round it. In programming math:

* **`ceil` (Ceiling)** means **Rounding Up** to the nearest whole integer.
* **`floor` (Floor)** means **Rounding Down** to the nearest whole integer (which is what standard integer division does by throwing away the remainder).

To get our **Magic Number** ($m$), we scale up the fraction and **round up (`ceil`)**:

```
Magic Number (m) = ceil(2^k / divisor)
```

*(Why round up? If we round down, the magic number is slightly too small, which guarantees we will underestimate the division later. By rounding up, we ensure we have enough "juice" to hit the correct quotient.)*

Now, to compute `n / d` on the GPU, I do this:

```
quotient = floor( (n * Magic Number) / 2^k )
```

Here, we multiply the index `n` by our massive Magic Number, and then we divide by the scale `2^k` to "scale it back down." The `floor` just means the computer automatically throws away the decimal part and keeps the final integer.

### 6.6 Why the Scaling Factor Must Be a Power of 2

Dividing by the Scaling Factor `2^k` at the end must be **free** (zero cycles). Otherwise, I would just be replacing one slow division with another slow division.

The only numbers a computer can divide by for free are **powers of 2** (like 2, 4, 8, 16, ... , 4,294,967,296). Dividing by a power of 2 is the exact same as shifting bits to the right — a single-cycle operation on any processor.

For example:

* Dividing by `8` (which is $2^3$) = shifting right by 3 bits.
* Dividing by `4,294,967,296` (which is $2^{32}$) = shifting right by 32 bits, which on a 32-bit system simply means "grab the upper 32 bits of a 64-bit result."

If I chose a non-power-of-2 scaler (like 10 or 1000), the final scaling-down step would require a real division instruction — defeating the entire purpose of the optimization.

### 6.7 The Error Analysis: Why the Scale MUST be exactly $2^{32}$

When we create the Magic Number by rounding up (`ceil`), we introduce a tiny **Rounding Error**.

```
Rounding Error = Magic Number - (2^k / divisor)
```

*(In simple words: The Rounding Error is just the tiny decimal difference between the perfect exact scaled number and the whole integer we rounded it up to).*

The big question is: When we multiply our index `n` by the Magic Number on the GPU, does this tiny error multiply and grow so big that it gives us the wrong quotient?

#### The "Race to 1.0" (Spill-over)

In integer division, we always round down (`floor`). The only way our fast-division trick gives a wrong answer is if the accumulated rounding error is so large that it pushes the total decimal fraction to `1.0` or higher. This is called a **Spill-over**, which incorrectly pushes the quotient up to the next whole number!

To visualize this, I generated a massive test suite (`error_propagation_analysis.md`) simulating the exact mathematical rounding errors for Deep Learning tensor sizes (like 3, 7, 12, 768) across every possible scale from `2^1` to `2^{32}`.

Here is exactly what the paperwork proved:

1. **Small Scales Fail Fast:**
   If we use a small scale like `2^3` (which is 8), the Rounding Error is huge. For `14 / 5`, the error grows so fast that the fraction hits `1.0` and spills over, telling the GPU the quotient is `3` instead of `2`.

2. **Why Big Indices Demand $2^{32}$:**
   We are working with 32-bit indices (`uint32_t`), meaning the maximum possible index `n` is a massive **4.29 Billion**.
   If we pick an index like `4.2 Billion`, we are multiplying that tiny Rounding Error by 4.2 Billion!
   The simulated data proves that for a divisor like `d = 7`, using scales like $2^{29}$, $2^{30}$, and even $2^{31}$ **ALL FAIL** for this huge index! The error piles up and spills over `1.0`.

3. **The Perfect Shield:**
   It isn't until we hit exactly **$2^{32}$** (which is `4,294,967,296`) that the scale finally outgrows the 4.2 Billion index. By dividing by $2^{32}$ at the end, it mathematically crushes the accumulated error down to `0.02` or lower, saving the calculation and preventing spill-over.

This is the golden mathematical proof of the paper: **(Maximum Index $\times$ Maximum Rounding Error) / $2^{32}$ will ALWAYS be strictly smaller than the spill-over boundary.** This makes $2^{32}$ the absolute minimum safety line.

### 6.8 The Hardware Mechanism: `__umulhi` and the 64-Bit Product

There is one remaining problem: when I multiply `n * Magic Number`, both are 32-bit numbers, but their product can be up to 64 bits. For example:

```
n = 10,  Magic = 1,073,741,824
Product = 10 * 1,073,741,824 = 10,737,418,240
```

This exceeds the 32-bit limit of `4,294,967,295`. So where does the product go?

Every modern CPU and GPU hardware multiplier internally produces a **64-bit result** when multiplying two 32-bit numbers. This 64-bit result is stored across two 32-bit registers:

```
┌──────────────────────────────┬──────────────────────────────┐
│  HIGH 32 Bits (Upper Half)   │   LOW 32 Bits (Lower Half)   │
│  = Product / 2^32            │   = Product % 2^32            │
│  (The integer quotient part) │   (The fractional/remainder)  │
└──────────────────────────────┴──────────────────────────────┘
```

For my example `10,737,418,240`:

* **HIGH register:** `10,737,418,240 / 4,294,967,296 = 2` — this is the quotient I want (`10 / 4 = 2`).
* **LOW register:** `10,737,418,240 % 4,294,967,296 = 2,147,483,648` — this is the scaled-up fractional part (`0.5 * 2^32`), which I discard because integer division rounds down.

NVIDIA GPUs provide a hardware intrinsic function called `__umulhi(a, b)` (Unsigned MULtiply HIgh) that performs this multiplication and directly returns the HIGH 32 bits in a single hardware instruction, taking only **~4 clock cycles**. The LOW bits are never written to a register, so there is zero overhead for discarding them.

```cpp
unsigned int quotient = __umulhi(n, magic_number);  // Returns HIGH 32 bits directly
```

### 6.9 The 33-Bit Magic Number Problem

For certain divisors, the computed Magic Number `ceil(2^(32+k) / d)` can be up to **33 bits** — one bit too large to fit in a 32-bit register. The Granlund-Montgomery paper proves this is unavoidable.

The solution is to store only the lower 32 bits of the Magic Number (call it `m_prime = Magic - 2^32`) and correct for the missing bit at runtime:

Since `Magic = m_prime + 2^32`:

```
n * Magic = n * (m_prime + 2^32) = n * m_prime + n * 2^32
```

After taking the upper 32 bits (dividing by 2^32):

```
__umulhi(n, Magic) = __umulhi(n, m_prime) + n
```

So the runtime code becomes:

```cpp
unsigned int t = __umulhi(n, m_prime);   // Multiply by the stored 32-bit part
unsigned int quotient = (t + n) >> shift; // The "+ n" corrects for the missing 33rd bit
```

This is exactly how PyTorch's `IntDivider` in `IntegerDivider.cuh` handles it — the `(t + n) >> shift` pattern.

### 6.10 The Complete Algorithm

**PRECOMPUTE (Done once on the CPU, when tensor shape is known):**

Given a divisor `d` (a tensor dimension size like 4, 7, 64, etc.):

1. Compute `shift = ceil(log2(d))` — the smallest power of 2 that is >= d.
2. Compute the magic multiplier: `m_prime = floor(2^32 * (2^shift - d) / d) + 1`.
3. Compute shift amounts: `sh1 = min(shift, 1)` and `sh2 = max(shift - 1, 0)`.

These three values (`m_prime`, `sh1`, `sh2`) are stored alongside the tensor's shape and stride metadata, and passed to the GPU kernel as arguments.

**RUNTIME (Done millions of times on the GPU, per thread, per dimension):**

```cpp
// Step 1: Fast Division (replaces slow "n / d")
unsigned int t = __umulhi(m_prime, n);
unsigned int quotient = (t + ((n - t) >> sh1)) >> sh2;

// Step 2: Fast Remainder (replaces slow "n % d")
unsigned int remainder = n - quotient * d;
```

**Total GPU cost per dimension:** ~6–8 cycles (1 multiply-high + 2 shifts + 2 adds + 1 multiply + 1 subtract), compared to ~30–40 cycles for a single hardware division instruction plus another ~30–40 cycles for the modulo. This is a **~10x speedup** on the critical non-contiguous data access path.

### 6.11 The Paper's Benchmark Results

The Granlund-Montgomery paper (*"Division by Invariant Integers using Multiplication"*, PLDI 1994) includes two benchmark datasets:

**Table 1.1 — Multiplication vs. Division Cycle Counts on Historical Processors:**

| Processor (Year) | Multiply-High Cycles | Division Cycles | Division Slowdown |
|---|---|---|---|
| Intel 386 (1985) | 9–38 | 38 | ~1x to 4x |
| MIPS R3000 (1988) | 12 | 35 | ~3x |
| POWER/RIOS I (1989) | 5 | 19 | ~4x |
| Intel Pentium (1993) | 10 | 46 | ~5x |
| DEC Alpha 21064 (1992) | 23 | 200 (software) | ~9x |
| HP PA 7000 (1990) | 3 | 70 (software) | ~23x |

**Table 11.2 — Radix Conversion Benchmark (integer-to-decimal-string, using `x/10` and `x%10` per digit):**

| Processor | With Hardware Division | With Division Eliminated | Speedup |
|---|---|---|---|
| DEC Alpha 21064 (133 MHz) | 22.0 us | 1.8 us | **12.2x** |
| HP PA 7000 (99 MHz) | 9.7 us | 2.1 us | **4.6x** |
| MIPS R4000 (100 MHz) | 8.3 us | 2.4 us | **3.4x** |
| SPARC Viking (40 MHz) | 6.4 us | 3.2 us | **2.0x** |

On modern NVIDIA GPUs (like my RTX 3060), the situation is analogous: integer multiplication is fast and pipelined (~4 cycles via `__umulhi`), while integer division remains a slow iterative operation (~30–40 cycles). The Granlund-Montgomery algorithm is therefore essential for any GPU kernel that performs repeated division by runtime-known constants — exactly the pattern in my `OffsetCalculator`'s non-contiguous tensor path.

### 6.7.1 Detailed Error Propagation Experiment Data (The Proof)

Below are the exhaustive calculations demonstrating exactly how the Rounding Error accumulates for various deep learning tensor dimensions and linear indices.

# Granlund-Montgomery: Error Propagation & Spill-over Visualized

This document simulates the mathematical rounding error introduced by approximating the reciprocal `1 / d` with a scaled integer.

For each combination of divisor `d` and linear index `n`, we test all scaling factors `2^k` from `k=1` to `k=32`.

### The Terms in Plain English

To turn the fractional reciprocal `1/d` into a whole number, we scale it up by `2^k` and then round it up.

* **`ceil` (Ceiling)** means **Rounding Up** to the nearest whole integer.
* **`floor` (Floor)** means **Rounding Down** to the nearest whole integer.

* **Magic Number ($m$)** = `ceil(2^k / d)`
  *(We round up to ensure we have enough "juice" to not underestimate the division later).*
* **Rounding Error ($e$)** = `m - (2^k / d)`
  *(This is just the tiny leftover decimal difference between the exact scaled number and the whole integer we just rounded it up to).*
* **Accumulated Error** = `(n * e) / 2^k`
* **True Fractional Part** = `(n % d) / d`
* **Total Fraction** = `True Fractional Part + Accumulated Error`

### The "Race to 1.0" (Spill-over)

In integer division, we always round down (`floor`). The only way our fast-division trick gives a wrong answer is if the accumulated rounding error is so large that it pushes the total decimal fraction to `1.0` or higher.

> **CRITICAL RULE:** If the **Total Fraction** reaches or exceeds `1.0`, it pushes the quotient to the next whole integer, causing a **Spill-over (Wrong Answer)!**

---

## Divisor `d = 3`

**Spill-over Threshold**: `1 / 3 = 0.33333333`

### Index `n = 14`

* **True Quotient**: `14 / 3 = 4`

* **True Fractional Part**: `2 / 3 = 0.66666667`
* **Distance to Spill-over**: `1.0 - 0.66666667 = 0.33333333`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.333333 | 2.333333 | 3.000000 | 7 | ❌ SPILL-OVER |
| 2 | 2^2 | 2 | 0.666667 | 2.333333 | 3.000000 | 7 | ❌ SPILL-OVER |
| 3 | 2^3 | 3 | 0.333333 | 0.583333 | 1.250000 | 5 | ❌ SPILL-OVER |
| 4 | 2^4 | 6 | 0.666667 | 0.583333 | 1.250000 | 5 | ❌ SPILL-OVER |
| 5 | 2^5 | 11 | 0.333333 | 0.145833 | 0.812500 | 4 | ✅ PASS |
| 6 | 2^6 | 22 | 0.666667 | 0.145833 | 0.812500 | 4 | ✅ PASS |
| 7 | 2^7 | 43 | 0.333333 | 0.036458 | 0.703125 | 4 | ✅ PASS |
| 8 | 2^8 | 86 | 0.666667 | 0.036458 | 0.703125 | 4 | ✅ PASS |
| 9 | 2^9 | 171 | 0.333333 | 0.009115 | 0.675781 | 4 | ✅ PASS |
| 10 | 2^10 | 342 | 0.666667 | 0.009115 | 0.675781 | 4 | ✅ PASS |
| 11 | 2^11 | 683 | 0.333333 | 0.002279 | 0.668945 | 4 | ✅ PASS |
| 12 | 2^12 | 1366 | 0.666667 | 0.002279 | 0.668945 | 4 | ✅ PASS |
| 13 | 2^13 | 2731 | 0.333333 | 0.000570 | 0.667236 | 4 | ✅ PASS |
| 14 | 2^14 | 5462 | 0.666667 | 0.000570 | 0.667236 | 4 | ✅ PASS |
| 15 | 2^15 | 10923 | 0.333333 | 0.000142 | 0.666809 | 4 | ✅ PASS |
| 16 | 2^16 | 21846 | 0.666667 | 0.000142 | 0.666809 | 4 | ✅ PASS |
| 17 | 2^17 | 43691 | 0.333333 | 0.000036 | 0.666702 | 4 | ✅ PASS |
| 18 | 2^18 | 87382 | 0.666667 | 0.000036 | 0.666702 | 4 | ✅ PASS |
| 19 | 2^19 | 174763 | 0.333333 | 0.000009 | 0.666676 | 4 | ✅ PASS |
| 20 | 2^20 | 349526 | 0.666667 | 0.000009 | 0.666676 | 4 | ✅ PASS |
| 21 | 2^21 | 699051 | 0.333333 | 0.000002 | 0.666669 | 4 | ✅ PASS |
| 22 | 2^22 | 1398102 | 0.666667 | 0.000002 | 0.666669 | 4 | ✅ PASS |
| 23 | 2^23 | 2796203 | 0.333333 | 0.000001 | 0.666667 | 4 | ✅ PASS |
| 24 | 2^24 | 5592406 | 0.666667 | 0.000001 | 0.666667 | 4 | ✅ PASS |
| 25 | 2^25 | 11184811 | 0.333333 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 26 | 2^26 | 22369622 | 0.666667 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 27 | 2^27 | 44739243 | 0.333333 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 28 | 2^28 | 89478486 | 0.666667 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 29 | 2^29 | 178956971 | 0.333333 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 30 | 2^30 | 357913942 | 0.666667 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 31 | 2^31 | 715827883 | 0.333333 | 0.000000 | 0.666667 | 4 | ✅ PASS |
| 32 | 2^32 | 1431655766 | 0.666667 | 0.000000 | 0.666667 | 4 | ✅ PASS |

### Index `n = 10,000`

* **True Quotient**: `10000 / 3 = 3333`

* **True Fractional Part**: `1 / 3 = 0.33333333`
* **Distance to Spill-over**: `1.0 - 0.33333333 = 0.66666667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.333333 | 1666.666667 | 1667.000000 | 5000 | ❌ SPILL-OVER |
| 2 | 2^2 | 2 | 0.666667 | 1666.666667 | 1667.000000 | 5000 | ❌ SPILL-OVER |
| 3 | 2^3 | 3 | 0.333333 | 416.666667 | 417.000000 | 3750 | ❌ SPILL-OVER |
| 4 | 2^4 | 6 | 0.666667 | 416.666667 | 417.000000 | 3750 | ❌ SPILL-OVER |
| 5 | 2^5 | 11 | 0.333333 | 104.166667 | 104.500000 | 3437 | ❌ SPILL-OVER |
| 6 | 2^6 | 22 | 0.666667 | 104.166667 | 104.500000 | 3437 | ❌ SPILL-OVER |
| 7 | 2^7 | 43 | 0.333333 | 26.041667 | 26.375000 | 3359 | ❌ SPILL-OVER |
| 8 | 2^8 | 86 | 0.666667 | 26.041667 | 26.375000 | 3359 | ❌ SPILL-OVER |
| 9 | 2^9 | 171 | 0.333333 | 6.510417 | 6.843750 | 3339 | ❌ SPILL-OVER |
| 10 | 2^10 | 342 | 0.666667 | 6.510417 | 6.843750 | 3339 | ❌ SPILL-OVER |
| 11 | 2^11 | 683 | 0.333333 | 1.627604 | 1.960938 | 3334 | ❌ SPILL-OVER |
| 12 | 2^12 | 1366 | 0.666667 | 1.627604 | 1.960938 | 3334 | ❌ SPILL-OVER |
| 13 | 2^13 | 2731 | 0.333333 | 0.406901 | 0.740234 | 3333 | ✅ PASS |
| 14 | 2^14 | 5462 | 0.666667 | 0.406901 | 0.740234 | 3333 | ✅ PASS |
| 15 | 2^15 | 10923 | 0.333333 | 0.101725 | 0.435059 | 3333 | ✅ PASS |
| 16 | 2^16 | 21846 | 0.666667 | 0.101725 | 0.435059 | 3333 | ✅ PASS |
| 17 | 2^17 | 43691 | 0.333333 | 0.025431 | 0.358765 | 3333 | ✅ PASS |
| 18 | 2^18 | 87382 | 0.666667 | 0.025431 | 0.358765 | 3333 | ✅ PASS |
| 19 | 2^19 | 174763 | 0.333333 | 0.006358 | 0.339691 | 3333 | ✅ PASS |
| 20 | 2^20 | 349526 | 0.666667 | 0.006358 | 0.339691 | 3333 | ✅ PASS |
| 21 | 2^21 | 699051 | 0.333333 | 0.001589 | 0.334923 | 3333 | ✅ PASS |
| 22 | 2^22 | 1398102 | 0.666667 | 0.001589 | 0.334923 | 3333 | ✅ PASS |
| 23 | 2^23 | 2796203 | 0.333333 | 0.000397 | 0.333731 | 3333 | ✅ PASS |
| 24 | 2^24 | 5592406 | 0.666667 | 0.000397 | 0.333731 | 3333 | ✅ PASS |
| 25 | 2^25 | 11184811 | 0.333333 | 0.000099 | 0.333433 | 3333 | ✅ PASS |
| 26 | 2^26 | 22369622 | 0.666667 | 0.000099 | 0.333433 | 3333 | ✅ PASS |
| 27 | 2^27 | 44739243 | 0.333333 | 0.000025 | 0.333358 | 3333 | ✅ PASS |
| 28 | 2^28 | 89478486 | 0.666667 | 0.000025 | 0.333358 | 3333 | ✅ PASS |
| 29 | 2^29 | 178956971 | 0.333333 | 0.000006 | 0.333340 | 3333 | ✅ PASS |
| 30 | 2^30 | 357913942 | 0.666667 | 0.000006 | 0.333340 | 3333 | ✅ PASS |
| 31 | 2^31 | 715827883 | 0.333333 | 0.000002 | 0.333335 | 3333 | ✅ PASS |
| 32 | 2^32 | 1431655766 | 0.666667 | 0.000002 | 0.333335 | 3333 | ✅ PASS |

### Index `n = 1,000,000`

* **True Quotient**: `1000000 / 3 = 333333`

* **True Fractional Part**: `1 / 3 = 0.33333333`
* **Distance to Spill-over**: `1.0 - 0.33333333 = 0.66666667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.333333 | 166666.666667 | 166667.000000 | 500000 | ❌ SPILL-OVER |
| 2 | 2^2 | 2 | 0.666667 | 166666.666667 | 166667.000000 | 500000 | ❌ SPILL-OVER |
| 3 | 2^3 | 3 | 0.333333 | 41666.666667 | 41667.000000 | 375000 | ❌ SPILL-OVER |
| 4 | 2^4 | 6 | 0.666667 | 41666.666667 | 41667.000000 | 375000 | ❌ SPILL-OVER |
| 5 | 2^5 | 11 | 0.333333 | 10416.666667 | 10417.000000 | 343750 | ❌ SPILL-OVER |
| 6 | 2^6 | 22 | 0.666667 | 10416.666667 | 10417.000000 | 343750 | ❌ SPILL-OVER |
| 7 | 2^7 | 43 | 0.333333 | 2604.166667 | 2604.500000 | 335937 | ❌ SPILL-OVER |
| 8 | 2^8 | 86 | 0.666667 | 2604.166667 | 2604.500000 | 335937 | ❌ SPILL-OVER |
| 9 | 2^9 | 171 | 0.333333 | 651.041667 | 651.375000 | 333984 | ❌ SPILL-OVER |
| 10 | 2^10 | 342 | 0.666667 | 651.041667 | 651.375000 | 333984 | ❌ SPILL-OVER |
| 11 | 2^11 | 683 | 0.333333 | 162.760417 | 163.093750 | 333496 | ❌ SPILL-OVER |
| 12 | 2^12 | 1366 | 0.666667 | 162.760417 | 163.093750 | 333496 | ❌ SPILL-OVER |
| 13 | 2^13 | 2731 | 0.333333 | 40.690104 | 41.023438 | 333374 | ❌ SPILL-OVER |
| 14 | 2^14 | 5462 | 0.666667 | 40.690104 | 41.023438 | 333374 | ❌ SPILL-OVER |
| 15 | 2^15 | 10923 | 0.333333 | 10.172526 | 10.505859 | 333343 | ❌ SPILL-OVER |
| 16 | 2^16 | 21846 | 0.666667 | 10.172526 | 10.505859 | 333343 | ❌ SPILL-OVER |
| 17 | 2^17 | 43691 | 0.333333 | 2.543132 | 2.876465 | 333335 | ❌ SPILL-OVER |
| 18 | 2^18 | 87382 | 0.666667 | 2.543132 | 2.876465 | 333335 | ❌ SPILL-OVER |
| 19 | 2^19 | 174763 | 0.333333 | 0.635783 | 0.969116 | 333333 | ✅ PASS |
| 20 | 2^20 | 349526 | 0.666667 | 0.635783 | 0.969116 | 333333 | ✅ PASS |
| 21 | 2^21 | 699051 | 0.333333 | 0.158946 | 0.492279 | 333333 | ✅ PASS |
| 22 | 2^22 | 1398102 | 0.666667 | 0.158946 | 0.492279 | 333333 | ✅ PASS |
| 23 | 2^23 | 2796203 | 0.333333 | 0.039736 | 0.373070 | 333333 | ✅ PASS |
| 24 | 2^24 | 5592406 | 0.666667 | 0.039736 | 0.373070 | 333333 | ✅ PASS |
| 25 | 2^25 | 11184811 | 0.333333 | 0.009934 | 0.343267 | 333333 | ✅ PASS |
| 26 | 2^26 | 22369622 | 0.666667 | 0.009934 | 0.343267 | 333333 | ✅ PASS |
| 27 | 2^27 | 44739243 | 0.333333 | 0.002484 | 0.335817 | 333333 | ✅ PASS |
| 28 | 2^28 | 89478486 | 0.666667 | 0.002484 | 0.335817 | 333333 | ✅ PASS |
| 29 | 2^29 | 178956971 | 0.333333 | 0.000621 | 0.333954 | 333333 | ✅ PASS |
| 30 | 2^30 | 357913942 | 0.666667 | 0.000621 | 0.333954 | 333333 | ✅ PASS |
| 31 | 2^31 | 715827883 | 0.333333 | 0.000155 | 0.333489 | 333333 | ✅ PASS |
| 32 | 2^32 | 1431655766 | 0.666667 | 0.000155 | 0.333489 | 333333 | ✅ PASS |

### Index `n = 4,200,000,000`

* **True Quotient**: `4200000000 / 3 = 1400000000`

* **True Fractional Part**: `0 / 3 = 0.00000000`
* **Distance to Spill-over**: `1.0 - 0.00000000 = 1.00000000`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.333333 | 700000000.000000 | 700000000.000000 | 2100000000 | ❌ SPILL-OVER |
| 2 | 2^2 | 2 | 0.666667 | 700000000.000000 | 700000000.000000 | 2100000000 | ❌ SPILL-OVER |
| 3 | 2^3 | 3 | 0.333333 | 175000000.000000 | 175000000.000000 | 1575000000 | ❌ SPILL-OVER |
| 4 | 2^4 | 6 | 0.666667 | 175000000.000000 | 175000000.000000 | 1575000000 | ❌ SPILL-OVER |
| 5 | 2^5 | 11 | 0.333333 | 43750000.000000 | 43750000.000000 | 1443750000 | ❌ SPILL-OVER |
| 6 | 2^6 | 22 | 0.666667 | 43750000.000000 | 43750000.000000 | 1443750000 | ❌ SPILL-OVER |
| 7 | 2^7 | 43 | 0.333333 | 10937500.000000 | 10937500.000000 | 1410937500 | ❌ SPILL-OVER |
| 8 | 2^8 | 86 | 0.666667 | 10937500.000000 | 10937500.000000 | 1410937500 | ❌ SPILL-OVER |
| 9 | 2^9 | 171 | 0.333333 | 2734375.000000 | 2734375.000000 | 1402734375 | ❌ SPILL-OVER |
| 10 | 2^10 | 342 | 0.666667 | 2734375.000000 | 2734375.000000 | 1402734375 | ❌ SPILL-OVER |
| 11 | 2^11 | 683 | 0.333333 | 683593.750000 | 683593.750000 | 1400683593 | ❌ SPILL-OVER |
| 12 | 2^12 | 1366 | 0.666667 | 683593.750000 | 683593.750000 | 1400683593 | ❌ SPILL-OVER |
| 13 | 2^13 | 2731 | 0.333333 | 170898.437500 | 170898.437500 | 1400170898 | ❌ SPILL-OVER |
| 14 | 2^14 | 5462 | 0.666667 | 170898.437500 | 170898.437500 | 1400170898 | ❌ SPILL-OVER |
| 15 | 2^15 | 10923 | 0.333333 | 42724.609375 | 42724.609375 | 1400042724 | ❌ SPILL-OVER |
| 16 | 2^16 | 21846 | 0.666667 | 42724.609375 | 42724.609375 | 1400042724 | ❌ SPILL-OVER |
| 17 | 2^17 | 43691 | 0.333333 | 10681.152344 | 10681.152344 | 1400010681 | ❌ SPILL-OVER |
| 18 | 2^18 | 87382 | 0.666667 | 10681.152344 | 10681.152344 | 1400010681 | ❌ SPILL-OVER |
| 19 | 2^19 | 174763 | 0.333333 | 2670.288086 | 2670.288086 | 1400002670 | ❌ SPILL-OVER |
| 20 | 2^20 | 349526 | 0.666667 | 2670.288086 | 2670.288086 | 1400002670 | ❌ SPILL-OVER |
| 21 | 2^21 | 699051 | 0.333333 | 667.572022 | 667.572022 | 1400000667 | ❌ SPILL-OVER |
| 22 | 2^22 | 1398102 | 0.666667 | 667.572022 | 667.572022 | 1400000667 | ❌ SPILL-OVER |
| 23 | 2^23 | 2796203 | 0.333333 | 166.893005 | 166.893005 | 1400000166 | ❌ SPILL-OVER |
| 24 | 2^24 | 5592406 | 0.666667 | 166.893005 | 166.893005 | 1400000166 | ❌ SPILL-OVER |
| 25 | 2^25 | 11184811 | 0.333333 | 41.723251 | 41.723251 | 1400000041 | ❌ SPILL-OVER |
| 26 | 2^26 | 22369622 | 0.666667 | 41.723251 | 41.723251 | 1400000041 | ❌ SPILL-OVER |
| 27 | 2^27 | 44739243 | 0.333333 | 10.430813 | 10.430813 | 1400000010 | ❌ SPILL-OVER |
| 28 | 2^28 | 89478486 | 0.666667 | 10.430813 | 10.430813 | 1400000010 | ❌ SPILL-OVER |
| 29 | 2^29 | 178956971 | 0.333333 | 2.607703 | 2.607703 | 1400000002 | ❌ SPILL-OVER |
| 30 | 2^30 | 357913942 | 0.666667 | 2.607703 | 2.607703 | 1400000002 | ❌ SPILL-OVER |
| 31 | 2^31 | 715827883 | 0.333333 | 0.651926 | 0.651926 | 1400000000 | ✅ PASS |
| 32 | 2^32 | 1431655766 | 0.666667 | 0.651926 | 0.651926 | 1400000000 | ✅ PASS |

---

## Divisor `d = 7`

**Spill-over Threshold**: `1 / 7 = 0.14285714`

### Index `n = 14`

* **True Quotient**: `14 / 7 = 2`

* **True Fractional Part**: `0 / 7 = 0.00000000`
* **Distance to Spill-over**: `1.0 - 0.00000000 = 1.00000000`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.714286 | 5.000000 | 5.000000 | 7 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.428571 | 1.500000 | 1.500000 | 3 | ❌ SPILL-OVER |
| 3 | 2^3 | 2 | 0.857143 | 1.500000 | 1.500000 | 3 | ❌ SPILL-OVER |
| 4 | 2^4 | 3 | 0.714286 | 0.625000 | 0.625000 | 2 | ✅ PASS |
| 5 | 2^5 | 5 | 0.428571 | 0.187500 | 0.187500 | 2 | ✅ PASS |
| 6 | 2^6 | 10 | 0.857143 | 0.187500 | 0.187500 | 2 | ✅ PASS |
| 7 | 2^7 | 19 | 0.714286 | 0.078125 | 0.078125 | 2 | ✅ PASS |
| 8 | 2^8 | 37 | 0.428571 | 0.023438 | 0.023438 | 2 | ✅ PASS |
| 9 | 2^9 | 74 | 0.857143 | 0.023438 | 0.023438 | 2 | ✅ PASS |
| 10 | 2^10 | 147 | 0.714286 | 0.009766 | 0.009766 | 2 | ✅ PASS |
| 11 | 2^11 | 293 | 0.428571 | 0.002930 | 0.002930 | 2 | ✅ PASS |
| 12 | 2^12 | 586 | 0.857143 | 0.002930 | 0.002930 | 2 | ✅ PASS |
| 13 | 2^13 | 1171 | 0.714286 | 0.001221 | 0.001221 | 2 | ✅ PASS |
| 14 | 2^14 | 2341 | 0.428571 | 0.000366 | 0.000366 | 2 | ✅ PASS |
| 15 | 2^15 | 4682 | 0.857143 | 0.000366 | 0.000366 | 2 | ✅ PASS |
| 16 | 2^16 | 9363 | 0.714286 | 0.000153 | 0.000153 | 2 | ✅ PASS |
| 17 | 2^17 | 18725 | 0.428571 | 0.000046 | 0.000046 | 2 | ✅ PASS |
| 18 | 2^18 | 37450 | 0.857143 | 0.000046 | 0.000046 | 2 | ✅ PASS |
| 19 | 2^19 | 74899 | 0.714286 | 0.000019 | 0.000019 | 2 | ✅ PASS |
| 20 | 2^20 | 149797 | 0.428571 | 0.000006 | 0.000006 | 2 | ✅ PASS |
| 21 | 2^21 | 299594 | 0.857143 | 0.000006 | 0.000006 | 2 | ✅ PASS |
| 22 | 2^22 | 599187 | 0.714286 | 0.000002 | 0.000002 | 2 | ✅ PASS |
| 23 | 2^23 | 1198373 | 0.428571 | 0.000001 | 0.000001 | 2 | ✅ PASS |
| 24 | 2^24 | 2396746 | 0.857143 | 0.000001 | 0.000001 | 2 | ✅ PASS |
| 25 | 2^25 | 4793491 | 0.714286 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 26 | 2^26 | 9586981 | 0.428571 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 27 | 2^27 | 19173962 | 0.857143 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 28 | 2^28 | 38347923 | 0.714286 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 29 | 2^29 | 76695845 | 0.428571 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 30 | 2^30 | 153391690 | 0.857143 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 31 | 2^31 | 306783379 | 0.714286 | 0.000000 | 0.000000 | 2 | ✅ PASS |
| 32 | 2^32 | 613566757 | 0.428571 | 0.000000 | 0.000000 | 2 | ✅ PASS |

### Index `n = 10,000`

* **True Quotient**: `10000 / 7 = 1428`

* **True Fractional Part**: `4 / 7 = 0.57142857`
* **Distance to Spill-over**: `1.0 - 0.57142857 = 0.42857143`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.714286 | 3571.428571 | 3572.000000 | 5000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.428571 | 1071.428571 | 1072.000000 | 2500 | ❌ SPILL-OVER |
| 3 | 2^3 | 2 | 0.857143 | 1071.428571 | 1072.000000 | 2500 | ❌ SPILL-OVER |
| 4 | 2^4 | 3 | 0.714286 | 446.428571 | 447.000000 | 1875 | ❌ SPILL-OVER |
| 5 | 2^5 | 5 | 0.428571 | 133.928571 | 134.500000 | 1562 | ❌ SPILL-OVER |
| 6 | 2^6 | 10 | 0.857143 | 133.928571 | 134.500000 | 1562 | ❌ SPILL-OVER |
| 7 | 2^7 | 19 | 0.714286 | 55.803571 | 56.375000 | 1484 | ❌ SPILL-OVER |
| 8 | 2^8 | 37 | 0.428571 | 16.741071 | 17.312500 | 1445 | ❌ SPILL-OVER |
| 9 | 2^9 | 74 | 0.857143 | 16.741071 | 17.312500 | 1445 | ❌ SPILL-OVER |
| 10 | 2^10 | 147 | 0.714286 | 6.975446 | 7.546875 | 1435 | ❌ SPILL-OVER |
| 11 | 2^11 | 293 | 0.428571 | 2.092634 | 2.664063 | 1430 | ❌ SPILL-OVER |
| 12 | 2^12 | 586 | 0.857143 | 2.092634 | 2.664063 | 1430 | ❌ SPILL-OVER |
| 13 | 2^13 | 1171 | 0.714286 | 0.871931 | 1.443359 | 1429 | ❌ SPILL-OVER |
| 14 | 2^14 | 2341 | 0.428571 | 0.261579 | 0.833008 | 1428 | ✅ PASS |
| 15 | 2^15 | 4682 | 0.857143 | 0.261579 | 0.833008 | 1428 | ✅ PASS |
| 16 | 2^16 | 9363 | 0.714286 | 0.108991 | 0.680420 | 1428 | ✅ PASS |
| 17 | 2^17 | 18725 | 0.428571 | 0.032697 | 0.604126 | 1428 | ✅ PASS |
| 18 | 2^18 | 37450 | 0.857143 | 0.032697 | 0.604126 | 1428 | ✅ PASS |
| 19 | 2^19 | 74899 | 0.714286 | 0.013624 | 0.585052 | 1428 | ✅ PASS |
| 20 | 2^20 | 149797 | 0.428571 | 0.004087 | 0.575516 | 1428 | ✅ PASS |
| 21 | 2^21 | 299594 | 0.857143 | 0.004087 | 0.575516 | 1428 | ✅ PASS |
| 22 | 2^22 | 599187 | 0.714286 | 0.001703 | 0.573132 | 1428 | ✅ PASS |
| 23 | 2^23 | 1198373 | 0.428571 | 0.000511 | 0.571939 | 1428 | ✅ PASS |
| 24 | 2^24 | 2396746 | 0.857143 | 0.000511 | 0.571939 | 1428 | ✅ PASS |
| 25 | 2^25 | 4793491 | 0.714286 | 0.000213 | 0.571641 | 1428 | ✅ PASS |
| 26 | 2^26 | 9586981 | 0.428571 | 0.000064 | 0.571492 | 1428 | ✅ PASS |
| 27 | 2^27 | 19173962 | 0.857143 | 0.000064 | 0.571492 | 1428 | ✅ PASS |
| 28 | 2^28 | 38347923 | 0.714286 | 0.000027 | 0.571455 | 1428 | ✅ PASS |
| 29 | 2^29 | 76695845 | 0.428571 | 0.000008 | 0.571437 | 1428 | ✅ PASS |
| 30 | 2^30 | 153391690 | 0.857143 | 0.000008 | 0.571437 | 1428 | ✅ PASS |
| 31 | 2^31 | 306783379 | 0.714286 | 0.000003 | 0.571432 | 1428 | ✅ PASS |
| 32 | 2^32 | 613566757 | 0.428571 | 0.000001 | 0.571430 | 1428 | ✅ PASS |

### Index `n = 1,000,000`

* **True Quotient**: `1000000 / 7 = 142857`

* **True Fractional Part**: `1 / 7 = 0.14285714`
* **Distance to Spill-over**: `1.0 - 0.14285714 = 0.85714286`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.714286 | 357142.857143 | 357143.000000 | 500000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.428571 | 107142.857143 | 107143.000000 | 250000 | ❌ SPILL-OVER |
| 3 | 2^3 | 2 | 0.857143 | 107142.857143 | 107143.000000 | 250000 | ❌ SPILL-OVER |
| 4 | 2^4 | 3 | 0.714286 | 44642.857143 | 44643.000000 | 187500 | ❌ SPILL-OVER |
| 5 | 2^5 | 5 | 0.428571 | 13392.857143 | 13393.000000 | 156250 | ❌ SPILL-OVER |
| 6 | 2^6 | 10 | 0.857143 | 13392.857143 | 13393.000000 | 156250 | ❌ SPILL-OVER |
| 7 | 2^7 | 19 | 0.714286 | 5580.357143 | 5580.500000 | 148437 | ❌ SPILL-OVER |
| 8 | 2^8 | 37 | 0.428571 | 1674.107143 | 1674.250000 | 144531 | ❌ SPILL-OVER |
| 9 | 2^9 | 74 | 0.857143 | 1674.107143 | 1674.250000 | 144531 | ❌ SPILL-OVER |
| 10 | 2^10 | 147 | 0.714286 | 697.544643 | 697.687500 | 143554 | ❌ SPILL-OVER |
| 11 | 2^11 | 293 | 0.428571 | 209.263393 | 209.406250 | 143066 | ❌ SPILL-OVER |
| 12 | 2^12 | 586 | 0.857143 | 209.263393 | 209.406250 | 143066 | ❌ SPILL-OVER |
| 13 | 2^13 | 1171 | 0.714286 | 87.193080 | 87.335938 | 142944 | ❌ SPILL-OVER |
| 14 | 2^14 | 2341 | 0.428571 | 26.157924 | 26.300781 | 142883 | ❌ SPILL-OVER |
| 15 | 2^15 | 4682 | 0.857143 | 26.157924 | 26.300781 | 142883 | ❌ SPILL-OVER |
| 16 | 2^16 | 9363 | 0.714286 | 10.899135 | 11.041992 | 142868 | ❌ SPILL-OVER |
| 17 | 2^17 | 18725 | 0.428571 | 3.269741 | 3.412598 | 142860 | ❌ SPILL-OVER |
| 18 | 2^18 | 37450 | 0.857143 | 3.269741 | 3.412598 | 142860 | ❌ SPILL-OVER |
| 19 | 2^19 | 74899 | 0.714286 | 1.362392 | 1.505249 | 142858 | ❌ SPILL-OVER |
| 20 | 2^20 | 149797 | 0.428571 | 0.408718 | 0.551575 | 142857 | ✅ PASS |
| 21 | 2^21 | 299594 | 0.857143 | 0.408718 | 0.551575 | 142857 | ✅ PASS |
| 22 | 2^22 | 599187 | 0.714286 | 0.170299 | 0.313156 | 142857 | ✅ PASS |
| 23 | 2^23 | 1198373 | 0.428571 | 0.051090 | 0.193947 | 142857 | ✅ PASS |
| 24 | 2^24 | 2396746 | 0.857143 | 0.051090 | 0.193947 | 142857 | ✅ PASS |
| 25 | 2^25 | 4793491 | 0.714286 | 0.021287 | 0.164145 | 142857 | ✅ PASS |
| 26 | 2^26 | 9586981 | 0.428571 | 0.006386 | 0.149243 | 142857 | ✅ PASS |
| 27 | 2^27 | 19173962 | 0.857143 | 0.006386 | 0.149243 | 142857 | ✅ PASS |
| 28 | 2^28 | 38347923 | 0.714286 | 0.002661 | 0.145518 | 142857 | ✅ PASS |
| 29 | 2^29 | 76695845 | 0.428571 | 0.000798 | 0.143655 | 142857 | ✅ PASS |
| 30 | 2^30 | 153391690 | 0.857143 | 0.000798 | 0.143655 | 142857 | ✅ PASS |
| 31 | 2^31 | 306783379 | 0.714286 | 0.000333 | 0.143190 | 142857 | ✅ PASS |
| 32 | 2^32 | 613566757 | 0.428571 | 0.000100 | 0.142957 | 142857 | ✅ PASS |

### Index `n = 4,200,000,000`

* **True Quotient**: `4200000000 / 7 = 600000000`

* **True Fractional Part**: `0 / 7 = 0.00000000`
* **Distance to Spill-over**: `1.0 - 0.00000000 = 1.00000000`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.714286 | 1500000000.000000 | 1500000000.000000 | 2100000000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.428571 | 450000000.000000 | 450000000.000000 | 1050000000 | ❌ SPILL-OVER |
| 3 | 2^3 | 2 | 0.857143 | 450000000.000000 | 450000000.000000 | 1050000000 | ❌ SPILL-OVER |
| 4 | 2^4 | 3 | 0.714286 | 187500000.000000 | 187500000.000000 | 787500000 | ❌ SPILL-OVER |
| 5 | 2^5 | 5 | 0.428571 | 56250000.000000 | 56250000.000000 | 656250000 | ❌ SPILL-OVER |
| 6 | 2^6 | 10 | 0.857143 | 56250000.000000 | 56250000.000000 | 656250000 | ❌ SPILL-OVER |
| 7 | 2^7 | 19 | 0.714286 | 23437500.000000 | 23437500.000000 | 623437500 | ❌ SPILL-OVER |
| 8 | 2^8 | 37 | 0.428571 | 7031250.000000 | 7031250.000000 | 607031250 | ❌ SPILL-OVER |
| 9 | 2^9 | 74 | 0.857143 | 7031250.000000 | 7031250.000000 | 607031250 | ❌ SPILL-OVER |
| 10 | 2^10 | 147 | 0.714286 | 2929687.500000 | 2929687.500000 | 602929687 | ❌ SPILL-OVER |
| 11 | 2^11 | 293 | 0.428571 | 878906.250000 | 878906.250000 | 600878906 | ❌ SPILL-OVER |
| 12 | 2^12 | 586 | 0.857143 | 878906.250000 | 878906.250000 | 600878906 | ❌ SPILL-OVER |
| 13 | 2^13 | 1171 | 0.714286 | 366210.937500 | 366210.937500 | 600366210 | ❌ SPILL-OVER |
| 14 | 2^14 | 2341 | 0.428571 | 109863.281250 | 109863.281250 | 600109863 | ❌ SPILL-OVER |
| 15 | 2^15 | 4682 | 0.857143 | 109863.281250 | 109863.281250 | 600109863 | ❌ SPILL-OVER |
| 16 | 2^16 | 9363 | 0.714286 | 45776.367188 | 45776.367188 | 600045776 | ❌ SPILL-OVER |
| 17 | 2^17 | 18725 | 0.428571 | 13732.910156 | 13732.910156 | 600013732 | ❌ SPILL-OVER |
| 18 | 2^18 | 37450 | 0.857143 | 13732.910156 | 13732.910156 | 600013732 | ❌ SPILL-OVER |
| 19 | 2^19 | 74899 | 0.714286 | 5722.045898 | 5722.045898 | 600005722 | ❌ SPILL-OVER |
| 20 | 2^20 | 149797 | 0.428571 | 1716.613770 | 1716.613770 | 600001716 | ❌ SPILL-OVER |
| 21 | 2^21 | 299594 | 0.857143 | 1716.613770 | 1716.613770 | 600001716 | ❌ SPILL-OVER |
| 22 | 2^22 | 599187 | 0.714286 | 715.255737 | 715.255737 | 600000715 | ❌ SPILL-OVER |
| 23 | 2^23 | 1198373 | 0.428571 | 214.576721 | 214.576721 | 600000214 | ❌ SPILL-OVER |
| 24 | 2^24 | 2396746 | 0.857143 | 214.576721 | 214.576721 | 600000214 | ❌ SPILL-OVER |
| 25 | 2^25 | 4793491 | 0.714286 | 89.406967 | 89.406967 | 600000089 | ❌ SPILL-OVER |
| 26 | 2^26 | 9586981 | 0.428571 | 26.822090 | 26.822090 | 600000026 | ❌ SPILL-OVER |
| 27 | 2^27 | 19173962 | 0.857143 | 26.822090 | 26.822090 | 600000026 | ❌ SPILL-OVER |
| 28 | 2^28 | 38347923 | 0.714286 | 11.175871 | 11.175871 | 600000011 | ❌ SPILL-OVER |
| 29 | 2^29 | 76695845 | 0.428571 | 3.352761 | 3.352761 | 600000003 | ❌ SPILL-OVER |
| 30 | 2^30 | 153391690 | 0.857143 | 3.352761 | 3.352761 | 600000003 | ❌ SPILL-OVER |
| 31 | 2^31 | 306783379 | 0.714286 | 1.396984 | 1.396984 | 600000001 | ❌ SPILL-OVER |
| 32 | 2^32 | 613566757 | 0.428571 | 0.419095 | 0.419095 | 600000000 | ✅ PASS |

---

## Divisor `d = 12`

**Spill-over Threshold**: `1 / 12 = 0.08333333`

### Index `n = 14`

* **True Quotient**: `14 / 12 = 1`

* **True Fractional Part**: `2 / 12 = 0.16666667`
* **Distance to Spill-over**: `1.0 - 0.16666667 = 0.83333333`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.833333 | 5.833333 | 6.000000 | 7 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.666667 | 2.333333 | 2.500000 | 3 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.333333 | 0.583333 | 0.750000 | 1 | ✅ PASS |
| 4 | 2^4 | 2 | 0.666667 | 0.583333 | 0.750000 | 1 | ✅ PASS |
| 5 | 2^5 | 3 | 0.333333 | 0.145833 | 0.312500 | 1 | ✅ PASS |
| 6 | 2^6 | 6 | 0.666667 | 0.145833 | 0.312500 | 1 | ✅ PASS |
| 7 | 2^7 | 11 | 0.333333 | 0.036458 | 0.203125 | 1 | ✅ PASS |
| 8 | 2^8 | 22 | 0.666667 | 0.036458 | 0.203125 | 1 | ✅ PASS |
| 9 | 2^9 | 43 | 0.333333 | 0.009115 | 0.175781 | 1 | ✅ PASS |
| 10 | 2^10 | 86 | 0.666667 | 0.009115 | 0.175781 | 1 | ✅ PASS |
| 11 | 2^11 | 171 | 0.333333 | 0.002279 | 0.168945 | 1 | ✅ PASS |
| 12 | 2^12 | 342 | 0.666667 | 0.002279 | 0.168945 | 1 | ✅ PASS |
| 13 | 2^13 | 683 | 0.333333 | 0.000570 | 0.167236 | 1 | ✅ PASS |
| 14 | 2^14 | 1366 | 0.666667 | 0.000570 | 0.167236 | 1 | ✅ PASS |
| 15 | 2^15 | 2731 | 0.333333 | 0.000142 | 0.166809 | 1 | ✅ PASS |
| 16 | 2^16 | 5462 | 0.666667 | 0.000142 | 0.166809 | 1 | ✅ PASS |
| 17 | 2^17 | 10923 | 0.333333 | 0.000036 | 0.166702 | 1 | ✅ PASS |
| 18 | 2^18 | 21846 | 0.666667 | 0.000036 | 0.166702 | 1 | ✅ PASS |
| 19 | 2^19 | 43691 | 0.333333 | 0.000009 | 0.166676 | 1 | ✅ PASS |
| 20 | 2^20 | 87382 | 0.666667 | 0.000009 | 0.166676 | 1 | ✅ PASS |
| 21 | 2^21 | 174763 | 0.333333 | 0.000002 | 0.166669 | 1 | ✅ PASS |
| 22 | 2^22 | 349526 | 0.666667 | 0.000002 | 0.166669 | 1 | ✅ PASS |
| 23 | 2^23 | 699051 | 0.333333 | 0.000001 | 0.166667 | 1 | ✅ PASS |
| 24 | 2^24 | 1398102 | 0.666667 | 0.000001 | 0.166667 | 1 | ✅ PASS |
| 25 | 2^25 | 2796203 | 0.333333 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 26 | 2^26 | 5592406 | 0.666667 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 27 | 2^27 | 11184811 | 0.333333 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 28 | 2^28 | 22369622 | 0.666667 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 29 | 2^29 | 44739243 | 0.333333 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 30 | 2^30 | 89478486 | 0.666667 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 31 | 2^31 | 178956971 | 0.333333 | 0.000000 | 0.166667 | 1 | ✅ PASS |
| 32 | 2^32 | 357913942 | 0.666667 | 0.000000 | 0.166667 | 1 | ✅ PASS |

### Index `n = 10,000`

* **True Quotient**: `10000 / 12 = 833`

* **True Fractional Part**: `4 / 12 = 0.33333333`
* **Distance to Spill-over**: `1.0 - 0.33333333 = 0.66666667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.833333 | 4166.666667 | 4167.000000 | 5000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.666667 | 1666.666667 | 1667.000000 | 2500 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.333333 | 416.666667 | 417.000000 | 1250 | ❌ SPILL-OVER |
| 4 | 2^4 | 2 | 0.666667 | 416.666667 | 417.000000 | 1250 | ❌ SPILL-OVER |
| 5 | 2^5 | 3 | 0.333333 | 104.166667 | 104.500000 | 937 | ❌ SPILL-OVER |
| 6 | 2^6 | 6 | 0.666667 | 104.166667 | 104.500000 | 937 | ❌ SPILL-OVER |
| 7 | 2^7 | 11 | 0.333333 | 26.041667 | 26.375000 | 859 | ❌ SPILL-OVER |
| 8 | 2^8 | 22 | 0.666667 | 26.041667 | 26.375000 | 859 | ❌ SPILL-OVER |
| 9 | 2^9 | 43 | 0.333333 | 6.510417 | 6.843750 | 839 | ❌ SPILL-OVER |
| 10 | 2^10 | 86 | 0.666667 | 6.510417 | 6.843750 | 839 | ❌ SPILL-OVER |
| 11 | 2^11 | 171 | 0.333333 | 1.627604 | 1.960938 | 834 | ❌ SPILL-OVER |
| 12 | 2^12 | 342 | 0.666667 | 1.627604 | 1.960938 | 834 | ❌ SPILL-OVER |
| 13 | 2^13 | 683 | 0.333333 | 0.406901 | 0.740234 | 833 | ✅ PASS |
| 14 | 2^14 | 1366 | 0.666667 | 0.406901 | 0.740234 | 833 | ✅ PASS |
| 15 | 2^15 | 2731 | 0.333333 | 0.101725 | 0.435059 | 833 | ✅ PASS |
| 16 | 2^16 | 5462 | 0.666667 | 0.101725 | 0.435059 | 833 | ✅ PASS |
| 17 | 2^17 | 10923 | 0.333333 | 0.025431 | 0.358765 | 833 | ✅ PASS |
| 18 | 2^18 | 21846 | 0.666667 | 0.025431 | 0.358765 | 833 | ✅ PASS |
| 19 | 2^19 | 43691 | 0.333333 | 0.006358 | 0.339691 | 833 | ✅ PASS |
| 20 | 2^20 | 87382 | 0.666667 | 0.006358 | 0.339691 | 833 | ✅ PASS |
| 21 | 2^21 | 174763 | 0.333333 | 0.001589 | 0.334923 | 833 | ✅ PASS |
| 22 | 2^22 | 349526 | 0.666667 | 0.001589 | 0.334923 | 833 | ✅ PASS |
| 23 | 2^23 | 699051 | 0.333333 | 0.000397 | 0.333731 | 833 | ✅ PASS |
| 24 | 2^24 | 1398102 | 0.666667 | 0.000397 | 0.333731 | 833 | ✅ PASS |
| 25 | 2^25 | 2796203 | 0.333333 | 0.000099 | 0.333433 | 833 | ✅ PASS |
| 26 | 2^26 | 5592406 | 0.666667 | 0.000099 | 0.333433 | 833 | ✅ PASS |
| 27 | 2^27 | 11184811 | 0.333333 | 0.000025 | 0.333358 | 833 | ✅ PASS |
| 28 | 2^28 | 22369622 | 0.666667 | 0.000025 | 0.333358 | 833 | ✅ PASS |
| 29 | 2^29 | 44739243 | 0.333333 | 0.000006 | 0.333340 | 833 | ✅ PASS |
| 30 | 2^30 | 89478486 | 0.666667 | 0.000006 | 0.333340 | 833 | ✅ PASS |
| 31 | 2^31 | 178956971 | 0.333333 | 0.000002 | 0.333335 | 833 | ✅ PASS |
| 32 | 2^32 | 357913942 | 0.666667 | 0.000002 | 0.333335 | 833 | ✅ PASS |

### Index `n = 1,000,000`

* **True Quotient**: `1000000 / 12 = 83333`

* **True Fractional Part**: `4 / 12 = 0.33333333`
* **Distance to Spill-over**: `1.0 - 0.33333333 = 0.66666667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.833333 | 416666.666667 | 416667.000000 | 500000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.666667 | 166666.666667 | 166667.000000 | 250000 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.333333 | 41666.666667 | 41667.000000 | 125000 | ❌ SPILL-OVER |
| 4 | 2^4 | 2 | 0.666667 | 41666.666667 | 41667.000000 | 125000 | ❌ SPILL-OVER |
| 5 | 2^5 | 3 | 0.333333 | 10416.666667 | 10417.000000 | 93750 | ❌ SPILL-OVER |
| 6 | 2^6 | 6 | 0.666667 | 10416.666667 | 10417.000000 | 93750 | ❌ SPILL-OVER |
| 7 | 2^7 | 11 | 0.333333 | 2604.166667 | 2604.500000 | 85937 | ❌ SPILL-OVER |
| 8 | 2^8 | 22 | 0.666667 | 2604.166667 | 2604.500000 | 85937 | ❌ SPILL-OVER |
| 9 | 2^9 | 43 | 0.333333 | 651.041667 | 651.375000 | 83984 | ❌ SPILL-OVER |
| 10 | 2^10 | 86 | 0.666667 | 651.041667 | 651.375000 | 83984 | ❌ SPILL-OVER |
| 11 | 2^11 | 171 | 0.333333 | 162.760417 | 163.093750 | 83496 | ❌ SPILL-OVER |
| 12 | 2^12 | 342 | 0.666667 | 162.760417 | 163.093750 | 83496 | ❌ SPILL-OVER |
| 13 | 2^13 | 683 | 0.333333 | 40.690104 | 41.023438 | 83374 | ❌ SPILL-OVER |
| 14 | 2^14 | 1366 | 0.666667 | 40.690104 | 41.023438 | 83374 | ❌ SPILL-OVER |
| 15 | 2^15 | 2731 | 0.333333 | 10.172526 | 10.505859 | 83343 | ❌ SPILL-OVER |
| 16 | 2^16 | 5462 | 0.666667 | 10.172526 | 10.505859 | 83343 | ❌ SPILL-OVER |
| 17 | 2^17 | 10923 | 0.333333 | 2.543132 | 2.876465 | 83335 | ❌ SPILL-OVER |
| 18 | 2^18 | 21846 | 0.666667 | 2.543132 | 2.876465 | 83335 | ❌ SPILL-OVER |
| 19 | 2^19 | 43691 | 0.333333 | 0.635783 | 0.969116 | 83333 | ✅ PASS |
| 20 | 2^20 | 87382 | 0.666667 | 0.635783 | 0.969116 | 83333 | ✅ PASS |
| 21 | 2^21 | 174763 | 0.333333 | 0.158946 | 0.492279 | 83333 | ✅ PASS |
| 22 | 2^22 | 349526 | 0.666667 | 0.158946 | 0.492279 | 83333 | ✅ PASS |
| 23 | 2^23 | 699051 | 0.333333 | 0.039736 | 0.373070 | 83333 | ✅ PASS |
| 24 | 2^24 | 1398102 | 0.666667 | 0.039736 | 0.373070 | 83333 | ✅ PASS |
| 25 | 2^25 | 2796203 | 0.333333 | 0.009934 | 0.343267 | 83333 | ✅ PASS |
| 26 | 2^26 | 5592406 | 0.666667 | 0.009934 | 0.343267 | 83333 | ✅ PASS |
| 27 | 2^27 | 11184811 | 0.333333 | 0.002484 | 0.335817 | 83333 | ✅ PASS |
| 28 | 2^28 | 22369622 | 0.666667 | 0.002484 | 0.335817 | 83333 | ✅ PASS |
| 29 | 2^29 | 44739243 | 0.333333 | 0.000621 | 0.333954 | 83333 | ✅ PASS |
| 30 | 2^30 | 89478486 | 0.666667 | 0.000621 | 0.333954 | 83333 | ✅ PASS |
| 31 | 2^31 | 178956971 | 0.333333 | 0.000155 | 0.333489 | 83333 | ✅ PASS |
| 32 | 2^32 | 357913942 | 0.666667 | 0.000155 | 0.333489 | 83333 | ✅ PASS |

### Index `n = 4,200,000,000`

* **True Quotient**: `4200000000 / 12 = 350000000`

* **True Fractional Part**: `0 / 12 = 0.00000000`
* **Distance to Spill-over**: `1.0 - 0.00000000 = 1.00000000`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.833333 | 1750000000.000000 | 1750000000.000000 | 2100000000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.666667 | 700000000.000000 | 700000000.000000 | 1050000000 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.333333 | 175000000.000000 | 175000000.000000 | 525000000 | ❌ SPILL-OVER |
| 4 | 2^4 | 2 | 0.666667 | 175000000.000000 | 175000000.000000 | 525000000 | ❌ SPILL-OVER |
| 5 | 2^5 | 3 | 0.333333 | 43750000.000000 | 43750000.000000 | 393750000 | ❌ SPILL-OVER |
| 6 | 2^6 | 6 | 0.666667 | 43750000.000000 | 43750000.000000 | 393750000 | ❌ SPILL-OVER |
| 7 | 2^7 | 11 | 0.333333 | 10937500.000000 | 10937500.000000 | 360937500 | ❌ SPILL-OVER |
| 8 | 2^8 | 22 | 0.666667 | 10937500.000000 | 10937500.000000 | 360937500 | ❌ SPILL-OVER |
| 9 | 2^9 | 43 | 0.333333 | 2734375.000000 | 2734375.000000 | 352734375 | ❌ SPILL-OVER |
| 10 | 2^10 | 86 | 0.666667 | 2734375.000000 | 2734375.000000 | 352734375 | ❌ SPILL-OVER |
| 11 | 2^11 | 171 | 0.333333 | 683593.750000 | 683593.750000 | 350683593 | ❌ SPILL-OVER |
| 12 | 2^12 | 342 | 0.666667 | 683593.750000 | 683593.750000 | 350683593 | ❌ SPILL-OVER |
| 13 | 2^13 | 683 | 0.333333 | 170898.437500 | 170898.437500 | 350170898 | ❌ SPILL-OVER |
| 14 | 2^14 | 1366 | 0.666667 | 170898.437500 | 170898.437500 | 350170898 | ❌ SPILL-OVER |
| 15 | 2^15 | 2731 | 0.333333 | 42724.609375 | 42724.609375 | 350042724 | ❌ SPILL-OVER |
| 16 | 2^16 | 5462 | 0.666667 | 42724.609375 | 42724.609375 | 350042724 | ❌ SPILL-OVER |
| 17 | 2^17 | 10923 | 0.333333 | 10681.152344 | 10681.152344 | 350010681 | ❌ SPILL-OVER |
| 18 | 2^18 | 21846 | 0.666667 | 10681.152344 | 10681.152344 | 350010681 | ❌ SPILL-OVER |
| 19 | 2^19 | 43691 | 0.333333 | 2670.288086 | 2670.288086 | 350002670 | ❌ SPILL-OVER |
| 20 | 2^20 | 87382 | 0.666667 | 2670.288086 | 2670.288086 | 350002670 | ❌ SPILL-OVER |
| 21 | 2^21 | 174763 | 0.333333 | 667.572022 | 667.572022 | 350000667 | ❌ SPILL-OVER |
| 22 | 2^22 | 349526 | 0.666667 | 667.572022 | 667.572022 | 350000667 | ❌ SPILL-OVER |
| 23 | 2^23 | 699051 | 0.333333 | 166.893005 | 166.893005 | 350000166 | ❌ SPILL-OVER |
| 24 | 2^24 | 1398102 | 0.666667 | 166.893005 | 166.893005 | 350000166 | ❌ SPILL-OVER |
| 25 | 2^25 | 2796203 | 0.333333 | 41.723251 | 41.723251 | 350000041 | ❌ SPILL-OVER |
| 26 | 2^26 | 5592406 | 0.666667 | 41.723251 | 41.723251 | 350000041 | ❌ SPILL-OVER |
| 27 | 2^27 | 11184811 | 0.333333 | 10.430813 | 10.430813 | 350000010 | ❌ SPILL-OVER |
| 28 | 2^28 | 22369622 | 0.666667 | 10.430813 | 10.430813 | 350000010 | ❌ SPILL-OVER |
| 29 | 2^29 | 44739243 | 0.333333 | 2.607703 | 2.607703 | 350000002 | ❌ SPILL-OVER |
| 30 | 2^30 | 89478486 | 0.666667 | 2.607703 | 2.607703 | 350000002 | ❌ SPILL-OVER |
| 31 | 2^31 | 178956971 | 0.333333 | 0.651926 | 0.651926 | 350000000 | ✅ PASS |
| 32 | 2^32 | 357913942 | 0.666667 | 0.651926 | 0.651926 | 350000000 | ✅ PASS |

---

## Divisor `d = 768`

**Spill-over Threshold**: `1 / 768 = 0.00130208`

### Index `n = 14`

* **True Quotient**: `14 / 768 = 0`

* **True Fractional Part**: `14 / 768 = 0.01822917`
* **Distance to Spill-over**: `1.0 - 0.01822917 = 0.98177083`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.997396 | 6.981771 | 7.000000 | 7 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.994792 | 3.481771 | 3.500000 | 3 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.989583 | 1.731771 | 1.750000 | 1 | ❌ SPILL-OVER |
| 4 | 2^4 | 1 | 0.979167 | 0.856771 | 0.875000 | 0 | ✅ PASS |
| 5 | 2^5 | 1 | 0.958333 | 0.419271 | 0.437500 | 0 | ✅ PASS |
| 6 | 2^6 | 1 | 0.916667 | 0.200521 | 0.218750 | 0 | ✅ PASS |
| 7 | 2^7 | 1 | 0.833333 | 0.091146 | 0.109375 | 0 | ✅ PASS |
| 8 | 2^8 | 1 | 0.666667 | 0.036458 | 0.054688 | 0 | ✅ PASS |
| 9 | 2^9 | 1 | 0.333333 | 0.009115 | 0.027344 | 0 | ✅ PASS |
| 10 | 2^10 | 2 | 0.666667 | 0.009115 | 0.027344 | 0 | ✅ PASS |
| 11 | 2^11 | 3 | 0.333333 | 0.002279 | 0.020508 | 0 | ✅ PASS |
| 12 | 2^12 | 6 | 0.666667 | 0.002279 | 0.020508 | 0 | ✅ PASS |
| 13 | 2^13 | 11 | 0.333333 | 0.000570 | 0.018799 | 0 | ✅ PASS |
| 14 | 2^14 | 22 | 0.666667 | 0.000570 | 0.018799 | 0 | ✅ PASS |
| 15 | 2^15 | 43 | 0.333333 | 0.000142 | 0.018372 | 0 | ✅ PASS |
| 16 | 2^16 | 86 | 0.666667 | 0.000142 | 0.018372 | 0 | ✅ PASS |
| 17 | 2^17 | 171 | 0.333333 | 0.000036 | 0.018265 | 0 | ✅ PASS |
| 18 | 2^18 | 342 | 0.666667 | 0.000036 | 0.018265 | 0 | ✅ PASS |
| 19 | 2^19 | 683 | 0.333333 | 0.000009 | 0.018238 | 0 | ✅ PASS |
| 20 | 2^20 | 1366 | 0.666667 | 0.000009 | 0.018238 | 0 | ✅ PASS |
| 21 | 2^21 | 2731 | 0.333333 | 0.000002 | 0.018231 | 0 | ✅ PASS |
| 22 | 2^22 | 5462 | 0.666667 | 0.000002 | 0.018231 | 0 | ✅ PASS |
| 23 | 2^23 | 10923 | 0.333333 | 0.000001 | 0.018230 | 0 | ✅ PASS |
| 24 | 2^24 | 21846 | 0.666667 | 0.000001 | 0.018230 | 0 | ✅ PASS |
| 25 | 2^25 | 43691 | 0.333333 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 26 | 2^26 | 87382 | 0.666667 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 27 | 2^27 | 174763 | 0.333333 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 28 | 2^28 | 349526 | 0.666667 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 29 | 2^29 | 699051 | 0.333333 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 30 | 2^30 | 1398102 | 0.666667 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 31 | 2^31 | 2796203 | 0.333333 | 0.000000 | 0.018229 | 0 | ✅ PASS |
| 32 | 2^32 | 5592406 | 0.666667 | 0.000000 | 0.018229 | 0 | ✅ PASS |

### Index `n = 10,000`

* **True Quotient**: `10000 / 768 = 13`

* **True Fractional Part**: `16 / 768 = 0.02083333`
* **Distance to Spill-over**: `1.0 - 0.02083333 = 0.97916667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.997396 | 4986.979167 | 4987.000000 | 5000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.994792 | 2486.979167 | 2487.000000 | 2500 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.989583 | 1236.979167 | 1237.000000 | 1250 | ❌ SPILL-OVER |
| 4 | 2^4 | 1 | 0.979167 | 611.979167 | 612.000000 | 625 | ❌ SPILL-OVER |
| 5 | 2^5 | 1 | 0.958333 | 299.479167 | 299.500000 | 312 | ❌ SPILL-OVER |
| 6 | 2^6 | 1 | 0.916667 | 143.229167 | 143.250000 | 156 | ❌ SPILL-OVER |
| 7 | 2^7 | 1 | 0.833333 | 65.104167 | 65.125000 | 78 | ❌ SPILL-OVER |
| 8 | 2^8 | 1 | 0.666667 | 26.041667 | 26.062500 | 39 | ❌ SPILL-OVER |
| 9 | 2^9 | 1 | 0.333333 | 6.510417 | 6.531250 | 19 | ❌ SPILL-OVER |
| 10 | 2^10 | 2 | 0.666667 | 6.510417 | 6.531250 | 19 | ❌ SPILL-OVER |
| 11 | 2^11 | 3 | 0.333333 | 1.627604 | 1.648438 | 14 | ❌ SPILL-OVER |
| 12 | 2^12 | 6 | 0.666667 | 1.627604 | 1.648438 | 14 | ❌ SPILL-OVER |
| 13 | 2^13 | 11 | 0.333333 | 0.406901 | 0.427734 | 13 | ✅ PASS |
| 14 | 2^14 | 22 | 0.666667 | 0.406901 | 0.427734 | 13 | ✅ PASS |
| 15 | 2^15 | 43 | 0.333333 | 0.101725 | 0.122559 | 13 | ✅ PASS |
| 16 | 2^16 | 86 | 0.666667 | 0.101725 | 0.122559 | 13 | ✅ PASS |
| 17 | 2^17 | 171 | 0.333333 | 0.025431 | 0.046265 | 13 | ✅ PASS |
| 18 | 2^18 | 342 | 0.666667 | 0.025431 | 0.046265 | 13 | ✅ PASS |
| 19 | 2^19 | 683 | 0.333333 | 0.006358 | 0.027191 | 13 | ✅ PASS |
| 20 | 2^20 | 1366 | 0.666667 | 0.006358 | 0.027191 | 13 | ✅ PASS |
| 21 | 2^21 | 2731 | 0.333333 | 0.001589 | 0.022423 | 13 | ✅ PASS |
| 22 | 2^22 | 5462 | 0.666667 | 0.001589 | 0.022423 | 13 | ✅ PASS |
| 23 | 2^23 | 10923 | 0.333333 | 0.000397 | 0.021231 | 13 | ✅ PASS |
| 24 | 2^24 | 21846 | 0.666667 | 0.000397 | 0.021231 | 13 | ✅ PASS |
| 25 | 2^25 | 43691 | 0.333333 | 0.000099 | 0.020933 | 13 | ✅ PASS |
| 26 | 2^26 | 87382 | 0.666667 | 0.000099 | 0.020933 | 13 | ✅ PASS |
| 27 | 2^27 | 174763 | 0.333333 | 0.000025 | 0.020858 | 13 | ✅ PASS |
| 28 | 2^28 | 349526 | 0.666667 | 0.000025 | 0.020858 | 13 | ✅ PASS |
| 29 | 2^29 | 699051 | 0.333333 | 0.000006 | 0.020840 | 13 | ✅ PASS |
| 30 | 2^30 | 1398102 | 0.666667 | 0.000006 | 0.020840 | 13 | ✅ PASS |
| 31 | 2^31 | 2796203 | 0.333333 | 0.000002 | 0.020835 | 13 | ✅ PASS |
| 32 | 2^32 | 5592406 | 0.666667 | 0.000002 | 0.020835 | 13 | ✅ PASS |

### Index `n = 1,000,000`

* **True Quotient**: `1000000 / 768 = 1302`

* **True Fractional Part**: `64 / 768 = 0.08333333`
* **Distance to Spill-over**: `1.0 - 0.08333333 = 0.91666667`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.997396 | 498697.916667 | 498698.000000 | 500000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.994792 | 248697.916667 | 248698.000000 | 250000 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.989583 | 123697.916667 | 123698.000000 | 125000 | ❌ SPILL-OVER |
| 4 | 2^4 | 1 | 0.979167 | 61197.916667 | 61198.000000 | 62500 | ❌ SPILL-OVER |
| 5 | 2^5 | 1 | 0.958333 | 29947.916667 | 29948.000000 | 31250 | ❌ SPILL-OVER |
| 6 | 2^6 | 1 | 0.916667 | 14322.916667 | 14323.000000 | 15625 | ❌ SPILL-OVER |
| 7 | 2^7 | 1 | 0.833333 | 6510.416667 | 6510.500000 | 7812 | ❌ SPILL-OVER |
| 8 | 2^8 | 1 | 0.666667 | 2604.166667 | 2604.250000 | 3906 | ❌ SPILL-OVER |
| 9 | 2^9 | 1 | 0.333333 | 651.041667 | 651.125000 | 1953 | ❌ SPILL-OVER |
| 10 | 2^10 | 2 | 0.666667 | 651.041667 | 651.125000 | 1953 | ❌ SPILL-OVER |
| 11 | 2^11 | 3 | 0.333333 | 162.760417 | 162.843750 | 1464 | ❌ SPILL-OVER |
| 12 | 2^12 | 6 | 0.666667 | 162.760417 | 162.843750 | 1464 | ❌ SPILL-OVER |
| 13 | 2^13 | 11 | 0.333333 | 40.690104 | 40.773438 | 1342 | ❌ SPILL-OVER |
| 14 | 2^14 | 22 | 0.666667 | 40.690104 | 40.773438 | 1342 | ❌ SPILL-OVER |
| 15 | 2^15 | 43 | 0.333333 | 10.172526 | 10.255859 | 1312 | ❌ SPILL-OVER |
| 16 | 2^16 | 86 | 0.666667 | 10.172526 | 10.255859 | 1312 | ❌ SPILL-OVER |
| 17 | 2^17 | 171 | 0.333333 | 2.543132 | 2.626465 | 1304 | ❌ SPILL-OVER |
| 18 | 2^18 | 342 | 0.666667 | 2.543132 | 2.626465 | 1304 | ❌ SPILL-OVER |
| 19 | 2^19 | 683 | 0.333333 | 0.635783 | 0.719116 | 1302 | ✅ PASS |
| 20 | 2^20 | 1366 | 0.666667 | 0.635783 | 0.719116 | 1302 | ✅ PASS |
| 21 | 2^21 | 2731 | 0.333333 | 0.158946 | 0.242279 | 1302 | ✅ PASS |
| 22 | 2^22 | 5462 | 0.666667 | 0.158946 | 0.242279 | 1302 | ✅ PASS |
| 23 | 2^23 | 10923 | 0.333333 | 0.039736 | 0.123070 | 1302 | ✅ PASS |
| 24 | 2^24 | 21846 | 0.666667 | 0.039736 | 0.123070 | 1302 | ✅ PASS |
| 25 | 2^25 | 43691 | 0.333333 | 0.009934 | 0.093267 | 1302 | ✅ PASS |
| 26 | 2^26 | 87382 | 0.666667 | 0.009934 | 0.093267 | 1302 | ✅ PASS |
| 27 | 2^27 | 174763 | 0.333333 | 0.002484 | 0.085817 | 1302 | ✅ PASS |
| 28 | 2^28 | 349526 | 0.666667 | 0.002484 | 0.085817 | 1302 | ✅ PASS |
| 29 | 2^29 | 699051 | 0.333333 | 0.000621 | 0.083954 | 1302 | ✅ PASS |
| 30 | 2^30 | 1398102 | 0.666667 | 0.000621 | 0.083954 | 1302 | ✅ PASS |
| 31 | 2^31 | 2796203 | 0.333333 | 0.000155 | 0.083489 | 1302 | ✅ PASS |
| 32 | 2^32 | 5592406 | 0.666667 | 0.000155 | 0.083489 | 1302 | ✅ PASS |

### Index `n = 4,200,000,000`

* **True Quotient**: `4200000000 / 768 = 5468750`

* **True Fractional Part**: `0 / 768 = 0.00000000`
* **Distance to Spill-over**: `1.0 - 0.00000000 = 1.00000000`

| `k` | Scale `S=2^k` | Magic `m=ceil(S/d)` | Rounding Error `e` | Acc. Error `(n*e)/S` | `Total Fraction` | Computed `q` | Status |
|---|---|---|---|---|---|---|---|
| 1 | 2^1 | 1 | 0.997396 | 2094531250.000000 | 2094531250.000000 | 2100000000 | ❌ SPILL-OVER |
| 2 | 2^2 | 1 | 0.994792 | 1044531250.000000 | 1044531250.000000 | 1050000000 | ❌ SPILL-OVER |
| 3 | 2^3 | 1 | 0.989583 | 519531250.000000 | 519531250.000000 | 525000000 | ❌ SPILL-OVER |
| 4 | 2^4 | 1 | 0.979167 | 257031250.000000 | 257031250.000000 | 262500000 | ❌ SPILL-OVER |
| 5 | 2^5 | 1 | 0.958333 | 125781250.000000 | 125781250.000000 | 131250000 | ❌ SPILL-OVER |
| 6 | 2^6 | 1 | 0.916667 | 60156250.000000 | 60156250.000000 | 65625000 | ❌ SPILL-OVER |
| 7 | 2^7 | 1 | 0.833333 | 27343750.000000 | 27343750.000000 | 32812500 | ❌ SPILL-OVER |
| 8 | 2^8 | 1 | 0.666667 | 10937500.000000 | 10937500.000000 | 16406250 | ❌ SPILL-OVER |
| 9 | 2^9 | 1 | 0.333333 | 2734375.000000 | 2734375.000000 | 8203125 | ❌ SPILL-OVER |
| 10 | 2^10 | 2 | 0.666667 | 2734375.000000 | 2734375.000000 | 8203125 | ❌ SPILL-OVER |
| 11 | 2^11 | 3 | 0.333333 | 683593.750000 | 683593.750000 | 6152343 | ❌ SPILL-OVER |
| 12 | 2^12 | 6 | 0.666667 | 683593.750000 | 683593.750000 | 6152343 | ❌ SPILL-OVER |
| 13 | 2^13 | 11 | 0.333333 | 170898.437500 | 170898.437500 | 5639648 | ❌ SPILL-OVER |
| 14 | 2^14 | 22 | 0.666667 | 170898.437500 | 170898.437500 | 5639648 | ❌ SPILL-OVER |
| 15 | 2^15 | 43 | 0.333333 | 42724.609375 | 42724.609375 | 5511474 | ❌ SPILL-OVER |
| 16 | 2^16 | 86 | 0.666667 | 42724.609375 | 42724.609375 | 5511474 | ❌ SPILL-OVER |
| 17 | 2^17 | 171 | 0.333333 | 10681.152344 | 10681.152344 | 5479431 | ❌ SPILL-OVER |
| 18 | 2^18 | 342 | 0.666667 | 10681.152344 | 10681.152344 | 5479431 | ❌ SPILL-OVER |
| 19 | 2^19 | 683 | 0.333333 | 2670.288086 | 2670.288086 | 5471420 | ❌ SPILL-OVER |
| 20 | 2^20 | 1366 | 0.666667 | 2670.288086 | 2670.288086 | 5471420 | ❌ SPILL-OVER |
| 21 | 2^21 | 2731 | 0.333333 | 667.572021 | 667.572021 | 5469417 | ❌ SPILL-OVER |
| 22 | 2^22 | 5462 | 0.666667 | 667.572021 | 667.572021 | 5469417 | ❌ SPILL-OVER |
| 23 | 2^23 | 10923 | 0.333333 | 166.893005 | 166.893005 | 5468916 | ❌ SPILL-OVER |
| 24 | 2^24 | 21846 | 0.666667 | 166.893005 | 166.893005 | 5468916 | ❌ SPILL-OVER |
| 25 | 2^25 | 43691 | 0.333333 | 41.723251 | 41.723251 | 5468791 | ❌ SPILL-OVER |
| 26 | 2^26 | 87382 | 0.666667 | 41.723251 | 41.723251 | 5468791 | ❌ SPILL-OVER |
| 27 | 2^27 | 174763 | 0.333333 | 10.430813 | 10.430813 | 5468760 | ❌ SPILL-OVER |
| 28 | 2^28 | 349526 | 0.666667 | 10.430813 | 10.430813 | 5468760 | ❌ SPILL-OVER |
| 29 | 2^29 | 699051 | 0.333333 | 2.607703 | 2.607703 | 5468752 | ❌ SPILL-OVER |
| 30 | 2^30 | 1398102 | 0.666667 | 2.607703 | 2.607703 | 5468752 | ❌ SPILL-OVER |
| 31 | 2^31 | 2796203 | 0.333333 | 0.651926 | 0.651926 | 5468750 | ✅ PASS |
| 32 | 2^32 | 5592406 | 0.666667 | 0.651926 | 0.651926 | 5468750 | ✅ PASS |
