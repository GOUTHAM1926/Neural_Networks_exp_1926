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

* `1 % 2 = 1` ──► **Depth index is 1**.
  * *Visual:* We are at Depth Index 1 inside the 2-plane block.
* `1 / 2 = 0` ──► **Done!**

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

To turn the fractional reciprocal `1/d` into a whole number, I multiply it by a large **Scaling Factor** `S`:

```
Magic Number = ceil(S / d)
```

Now, to compute `n / d`, I do:

```
quotient = floor(n * Magic Number / S)
```

This works because the two operations cancel out:

```
n * (S / d) / S  =  n / d
```

The division by `S` at the end "undoes" the scaling, leaving me with the original quotient.

### 6.6 Why the Scaling Factor Must Be a Power of 2

Dividing by the Scaling Factor `S` at the end must be **free** (zero cycles). Otherwise, I would just be replacing one slow division with another slow division.

The only numbers a computer can divide by for free are **powers of 2** (like 2, 4, 8, 16, ... , 4,294,967,296). Dividing by a power of 2 is the same as shifting bits to the right — a single-cycle operation on any processor.

For example:
* Dividing by `8` (which is 2^3) = shifting right by 3 bits.
* Dividing by `4,294,967,296` (which is 2^32) = shifting right by 32 bits, which on a 32-bit system simply means "grab the upper 32 bits of a 64-bit result."

If I chose a non-power-of-2 scaler (like 10 or 1000), the final scaling-down step would require a real division instruction — defeating the entire purpose of the optimization.

### 6.7 Why the Power of 2 Must Be Large (2^32, not 2^3 or 2^8)

When I round the fractional value `S / d` to get a whole-number Magic Number, I introduce a tiny **rounding error**. The question is: can this rounding error ever accumulate enough to give me a wrong quotient?

#### Worked Example: What Happens with a Small Scaler

Let me try using a tiny scaler of `S = 2^3 = 8` to divide numbers by `5`.

The reciprocal of `5` is `0.2`.

```
Magic Number = ceil(8 / 5) = ceil(1.6) = 2
```

The rounding error is `2 - 1.6 = 0.4`. Relative to the scaler `8`, this is an error of `0.4 / 8 = 0.05` (5% error per operation).

Testing this Magic Number on different inputs:

```
n = 10:  floor(10 * 2 / 8) = floor(20 / 8) = floor(2.50) = 2   (Correct: 10/5 = 2)
n = 11:  floor(11 * 2 / 8) = floor(22 / 8) = floor(2.75) = 2   (Correct: 11/5 = 2)
n = 14:  floor(14 * 2 / 8) = floor(28 / 8) = floor(3.50) = 3   (WRONG! 14/5 = 2)
```

At `n = 14`, the accumulated rounding error pushed the result past the integer boundary, producing `3` instead of the correct answer `2`. The scaler was too small to contain the error.

#### What Happens with `S = 2^32`

Now let me try `S = 2^32 = 4,294,967,296`:

```
Magic Number = ceil(4,294,967,296 / 5) = ceil(858,993,459.2) = 858,993,460
```

The rounding error is `858,993,460 - 858,993,459.2 = 0.8`. Relative to the scaler `4,294,967,296`, this is an error of `0.8 / 4,294,967,296 = 0.000000000186` (0.0000000186% error). This is microscopically small.

Even if I multiply this error by the largest possible 32-bit index (`n = 4,294,967,295`), the total accumulated error is:

```
0.8 * 4,294,967,295 / 4,294,967,296 = 0.7999...
```

This is still less than `1.0`, which means it can **never** push the quotient up to the next integer.

#### The General Error Analysis

When I create the Magic Number by rounding `ceil(S / d)`, I introduce a rounding error `e` where `0 < e < 1`.

At runtime, the estimated quotient is:

```
estimated = floor(n * Magic / S)
         = floor(n * (S/d + e) / S)
         = floor(n/d  +  n*e/S)
```

This has two parts:
1. **True quotient:** `n / d` (what I want).
2. **Accumulated error:** `n * e / S` (the noise from rounding).

The accumulated error causes a wrong answer only if it pushes the result past the next integer boundary. The worst case is when the true quotient's fractional part is `(d-1)/d` — the closest it can get to the next integer without reaching it. The gap left before spilling over is `1/d`.

So, for correctness, I need:

```
n * e / S  <  1 / d
```

Since the worst-case values are `n = 2^32 - 1` and `e < 1`:

```
S  >  n * d * e  >  (2^32 - 1) * d * 1
```

For `S = 2^32` and the specific way the Magic Number is chosen (using `ceil` and an appropriate shift value `k = ceil(log2(d))`), the Granlund-Montgomery paper proves via Theorem 4.2 that this inequality is always satisfied for every `n` in the range `[0, 2^32 - 1]` and every divisor `d` in the range `[1, 2^32 - 1]`.

This is why `2^32` is the minimum safe scaler for 32-bit integers. Using `2^31`, `2^30`, or anything smaller would leave the error margin too tight, and certain input combinations would produce wrong quotients.

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
