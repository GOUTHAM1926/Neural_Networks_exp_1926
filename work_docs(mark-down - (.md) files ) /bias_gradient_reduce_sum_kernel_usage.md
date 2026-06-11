# Bias Gradient & Reduce Kernel Usage — Deep Dive

**Author:** Goutham Reddy (Gautam_1926)

---

## 1. What is Differentiation?

Forget the fancy word. Differentiation answers ONE simple question:

> **"If I nudge THIS input by a tiny amount, how much does the output change?"**

That's literally it. It's a sensitivity test.

**Real-world analogy:**
You're driving a car, watching the speedometer.

- Speed = 60 km/h. You press the accelerator pedal down by 1 mm.
- Speed jumps to 63 km/h.
- The "derivative" of speed with respect to pedal position ≈ `3 km/h per mm`.

In math notation, if `S` is speed and `P` is pedal position:

```
dS/dP ≈ 3 km/h per mm
```

That fraction `dS/dP` IS the **derivative**. It just says: "a tiny push on P causes this much change in S."

---

## 2. What is Partial Differentiation?

In a car, speed depends on MANY things: pedal position, gear, road slope, wind...

If you want to ask: *"How does speed change when I push the pedal, **while keeping everything else frozen**?"* — that's a **partial derivative**.

The curly `∂` (instead of straight `d`) signals: **"I'm only tweaking ONE variable, everything else stays frozen."**

```
∂S/∂P = "How much does Speed change if I nudge Pedal, keeping gear/slope/wind frozen?"
∂S/∂G = "How much does Speed change if I shift Gear, keeping pedal/slope/wind frozen?"
```

In neural networks, we have MILLIONS of variables (weights, biases). When we write `∂L/∂b_j`, we're asking:

> **"If I nudge JUST this one bias number `b_j` by a microscopic amount, how much does the total Loss `L` change — while keeping every other weight in the entire network frozen?"**

---

## 3. COMMON DOUBT: "But even normal derivative w.r.t one variable keeps others frozen, right? So what's the difference?"

### The Honest Truth: The MATH Operation is the Same

Whether you write `dL/db_j` or `∂L/∂b_j`, the actual **mechanical process** — "differentiate with respect to `b_j`, treat everything else as a constant" — is **identical**. The calculus steps you perform are exactly the same.

So then what's the difference? **It's a NOTATION signal, not a different math operation.**

### When You Use `d` (Straight d) — Single Variable Functions

When a function depends on **only ONE variable**, you use `d`:

```
f(x) = 3x² + 5x

df/dx = 6x + 5
```

There's nothing else to freeze, because there's only ONE input. The `d` tells the reader: **"this function has one variable."**

### When You Use `∂` (Curly d) — Multi-Variable Functions

When a function depends on **MULTIPLE variables**, you use `∂`:

```
f(x, y, z) = 3x² + 5xy + z

∂f/∂x = 6x + 5y       (treat y, z as constants)
∂f/∂y = 5x             (treat x, z as constants)
∂f/∂z = 1              (treat x, y as constants)
```

The `∂` tells the reader: **"WARNING — this function has MANY variables! I'm only differentiating with respect to ONE of them right now."**

### So the ONLY Difference is

**The `∂` is a WARNING SIGN to the reader.** It says:

> "Hey reader! Don't think this derivative tells you the FULL story of how the output changes. There are OTHER variables that ALSO affect the output. I'm just showing you ONE variable's effect right now."

If you wrote `dL/db_j` for a neural network where `L` depends on millions of variables, a mathematician would look at it and say: *"Wait, does L only depend on b_j? That can't be right..."* The `∂` immediately communicates: *"No no, L depends on millions of things, but right now I'm ONLY asking about b_j."*

### One-Line Summary

| Symbol | Meaning | When to use |
|---|---|---|
| `d` | "this is the ONLY variable" | `f(x)` — function of ONE variable |
| `∂` | "there are OTHER variables too, I'm just looking at this one" | `f(x, y, z, ...)` — function of MANY variables |

**The differentiation mechanics? Identical. The symbol? A communication tool for humans.** The `∂` is just honesty: "I know this isn't the full picture, I'm zooming into one variable."

---

## 4. The Symbols in the Reduce Kernel Equation

The paper's equation: **`∂L/∂b_j = ∑_i ∂L/∂y_ij`**

Here's every symbol decoded:

### `L` — The Loss (A Single Number)

- After the model predicts "The cat sat on the ___" → it assigns probabilities to all 50,304 words
- If the correct answer is "mat" and the model gave it only 1% probability → that's bad
- The Loss function converts this into a **single number**. High = dumb. Low = smart.
- **The entire goal of training = make L as small as possible.**

### `y` — The Output of a Linear Layer

```
y = x × W + b      (the forward pass equation)
```

- `x` = input, shape `[16384, 768]`
- `W` = weight matrix
- `b` = bias vector, shape `[768]`
- `y` = output, shape `[16384, 768]`

### `∂L/∂y` — The "Upstream Gradient" (A Massive Matrix)

- Shape: `[16384, 768]` — one gradient value for every element of y
- This is **handed to us by backpropagation** from the layers above
- It answers: *"How much does the Loss change if each element of y changes?"*
- Called "upstream" because it flows down from the loss through the layers above us

### `b_j` — One Specific Bias Number

```
b = [b_0, b_1, b_2, ..., b_767]   ← the full bias vector (768 numbers)
b_j = one specific number at position j
```

`j` is just an index (counter from 0 to 767).

### `∂L/∂b_j` — "How Much Should We Tweak This Bias?"

- **This is what the reduce kernel computes!**
- Once we know this, the AdamW optimizer uses it to update `b_j`
- Large value → big effect on loss → big update. Near zero → tiny update.

### `∑_i` — The Summation (Just a For Loop!)

```
∑_i  =  for (int i = 0; i < 16384; i++) { total += ... }
```

The Greek letter Sigma. `i` counts over all 16,384 tokens. It means: **add them all up.**

### `∂L/∂y_ij` — One Cell of the Upstream Gradient

- `i` picks the row (which token, 0 to 16383)
- `j` picks the column (which feature, 0 to 767)
- Just one cell in the `[16384, 768]` matrix

---

## 5. The Full Equation in Plain English

```
∂L/∂b_j  =  ∑_i  ∂L/∂y_ij
```

> **"To find out how much to update bias j, go to column j of the upstream gradient matrix, and add up all 16,384 rows in that column."**

```
The upstream gradient ∂L/∂y (16384 × 768 matrix):

              j=0      j=1      j=2    ...   j=767
Token 0:   [ g_0,0    g_0,1    g_0,2   ...  g_0,767  ]
Token 1:   [ g_1,0    g_1,1    g_1,2   ...  g_1,767  ]
Token 2:   [ g_2,0    g_2,1    g_2,2   ...  g_2,767  ]
  ...          ↓         ↓        ↓              ↓
Token 16383:[ g_N,0    g_N,1    g_N,2   ...  g_N,767  ]
             ─────    ─────    ─────         ─────
              SUM      SUM      SUM    ...    SUM       ← ∑_i
               ↓        ↓        ↓              ↓
∂L/∂b:    [ ∂L/∂b_0  ∂L/∂b_1  ∂L/∂b_2  ...  ∂L/∂b_767 ]   ← 768 numbers
```

**This vertical column-sum is EXACTLY what `unified_reduce_kernel_outer_vec4` does!**

---

## 6. WHY is the Bias Gradient a Sum? — The Chain Rule

The **chain rule** answers: *"If the Loss depends on Y, and Y depends on the Bias, how does the Loss depend on the Bias?"*

### 6.1 — The Simplified Chain Rule (ONE-Path Version)

If `b_j` affected only ONE output value (say `y_0,j`), the chain would be simple:

```
b_j ──affects──▶ y_0,j ──affects──▶ L       (ONE path)
```

The chain rule says: **multiply the effects together!**

```
∂L/∂b_j = ∂L/∂y_0,j × ∂y_0,j/∂b_j
            ↑                ↑
     "how y_0,j affects L"  "how b_j affects y_0,j"
```

Now let's compute `∂y_0,j/∂b_j`. The forward pass says:

```
y_0,j = (x × W)_0,j + b_j
```

If you nudge `b_j` by a tiny `ε`:

```
y_0,j_new = (x × W)_0,j + (b_j + ε) = y_0,j + ε
```

So **`∂y_0,j / ∂b_j = 1`**. The bias adds directly — tweaking it by ε changes the output by exactly ε!

If this was the only path, we'd be done: `∂L/∂b_j = ∂L/∂y_0,j × 1 = ∂L/∂y_0,j`. No sum needed. Just one value.

### 6.2 — The FULL Chain Rule (MULTI-Path Version) — Where the SUM Comes From

But `b_j` does NOT affect just one `y` value. It is **broadcast** to ALL 16,384 tokens! So it affects 16,384 DIFFERENT `y` values:

```
y_0,j      = (x × W)_0,j      + b_j     ← b_j affects this
y_1,j      = (x × W)_1,j      + b_j     ← b_j affects this too
y_2,j      = (x × W)_2,j      + b_j     ← and this
  ...
y_16383,j  = (x × W)_16383,j  + b_j     ← and this!
```

That means there are **16,384 separate paths** from `b_j` to the Loss `L`:

```
                    ┌──▶ y_0,j     ──▶─┐
                    ├──▶ y_1,j     ──▶─┤
        b_j ────────├──▶ y_2,j     ──▶─┼──▶ L
                    ├──▶  ...      ──▶─┤
                    └──▶ y_16383,j ──▶─┘

        ONE source     16,384 paths    ONE destination
```

The **FULL chain rule** says: when one variable affects the Loss through **MULTIPLE intermediate values**, you must **SUM the contribution from EACH path**:

```
∂L/∂b_j = ∂L/∂y_0,j     × ∂y_0,j/∂b_j        ← contribution through path 0
         + ∂L/∂y_1,j     × ∂y_1,j/∂b_j        ← contribution through path 1
         + ∂L/∂y_2,j     × ∂y_2,j/∂b_j        ← contribution through path 2
         + ...
         + ∂L/∂y_16383,j × ∂y_16383,j/∂b_j    ← contribution through path 16383
```

Written compactly:

```
∂L/∂b_j = ∑_i ( ∂L/∂y_i,j × ∂y_i,j/∂b_j )
                                      ↑
                              THIS is where the SUM comes from!
                              The sum is NOT added on top of the chain rule.
                              The sum IS the chain rule for the multi-path case.
                              Broadcasting creates 16,384 paths → chain rule sums all 16,384.
```

### 6.3 — Now Apply `∂y_i,j/∂b_j = 1`

Since every single `∂y_i,j/∂b_j = 1` (bias just adds, derivative is always 1):

```
∂L/∂b_j = ∑_i ( ∂L/∂y_i,j × 1 )

         = ∑_i  ∂L/∂y_i,j

         = ∂L/∂y_0,j + ∂L/∂y_1,j + ... + ∂L/∂y_16383,j

         = ∑_i ∂L/∂y_ij     ← THE EQUATION FROM THE Technical report of BluBridge under Reduce kernel part
```

### 6.4 — Contrast: What if `b_j` Only Affected ONE Token?

If `b_j` was only added to Token 0 (not broadcast), there'd be only ONE path:

```
b_j ──▶ y_0,j ──▶ L       (only one path, no sum needed)

∂L/∂b_j = ∂L/∂y_0,j × 1 = ∂L/∂y_0,j       ← just one value, no sum!
```

But because `b_j` is broadcast to ALL 16,384 tokens, it creates 16,384 paths, and the multi-path chain rule gives us the sum over all of them.

**Summary: The simplified chain rule `∂L/∂b = ∂L/∂y × ∂y/∂b` is the ONE-path version. The FULL chain rule says: when one variable affects the Loss through MULTIPLE intermediate values, SUM the contributions from ALL paths. The sum comes from the chain rule itself — not from something added on top of it. Broadcasting creates 16,384 paths, so the chain rule automatically gives a sum over 16,384 terms.**

### The Boss-and-Workers Analogy

Think of `b_j` as a manager who gives the **same instruction** to 16,384 workers (tokens).

- At end of day, the CEO (Loss `L`) sends a "blame report" to each worker: `∂L/∂y_ij`
- To figure out the **manager's** total blame, you **sum up the blame from ALL 16,384 workers**
- Because the manager gave the same instruction to everyone, he's partially responsible for every worker's mistake!

That sum of blame = `∑_i ∂L/∂y_ij` = the bias gradient = what the reduce kernel computes!

---

## 7. But WHY Are We Summing Up All Rows? (The Deep "Why")

We now know the chain rule formula gives us a sum. But there's a deeper intuitive question:

> "In the forward pass, we BROADCAST one bias vector to 16,384 rows. In the backward pass, we SUM 16,384 rows into one vector. Isn't summing all those rows just mixing up all the gradients? Isn't that losing information?"

### 7.1 — Broadcast Forward = Sum Backward (They Are Mirror Images)

This is a universal law in calculus and deep learning:

```
Forward:   b_0 (one number) ──BROADCAST──▶  added to 16,384 tokens  (1 → many)
Backward:  16,384 gradients  ──SUM──▶       one gradient number      (many → 1)
```

These are NOT two random operations that happen to be opposite. They are **mathematically locked together**. Here's the proof with tiny numbers:

**Forward:** `b_0 = 5` gets added to 3 tokens:

```
Token 0:  y_0 = 10 + 5 = 15
Token 1:  y_1 = 20 + 5 = 25
Token 2:  y_2 = 30 + 5 = 35
```

Now **nudge `b_0` from 5 to 5.001** (tiny change of +0.001):

```
Token 0:  y_0 = 10 + 5.001 = 15.001    ← changed by +0.001
Token 1:  y_1 = 20 + 5.001 = 25.001    ← changed by +0.001
Token 2:  y_2 = 30 + 5.001 = 35.001    ← changed by +0.001
```

**ALL THREE changed!** Because `b_0` was broadcast to all three.

Now, let's say the upstream gradients (blame reports) are:

```
∂L/∂y_0 = 0.3     (Token 0's change of +0.001 caused Loss to change by 0.001 × 0.3)
∂L/∂y_1 = -0.5    (Token 1's change caused Loss to change by 0.001 × -0.5)
∂L/∂y_2 = 0.8     (Token 2's change caused Loss to change by 0.001 × 0.8)
```

Total Loss change from nudging `b_0` by +0.001:

```
Through Token 0:  +0.001 × 0.3  = +0.0003
Through Token 1:  +0.001 × -0.5 = -0.0005
Through Token 2:  +0.001 × 0.8  = +0.0008
                                   ────────
Total:                             +0.0006
```

So `∂L/∂b_0 = 0.3 + (-0.5) + 0.8 = 0.6` — **that's a SUM!**

We HAD to add because `b_0` affected ALL 3 tokens. If we only took ONE row, we'd be ignoring 2 out of 3 crimes that `b_0` committed. The sum accounts for the TOTAL responsibility.

**The universal law:**

```
Forward operation       →    Its backward operation
─────────────────            ─────────────────────
BROADCAST (1 → N)       →    SUM (N → 1)
SUM (N → 1)             →    BROADCAST (1 → N)
MULTIPLY (a × b)        →    MULTIPLY (swap: ∂L × b, ∂L × a)
```

They are always mirror images of each other.

### 7.2 — "But Isn't This Mixing All Tokens' Gradients?"

Yes, it IS mixing them — and that's the correct thing to do. Here's why:

The bias `b_j` is **ONE number** shared by all 16,384 tokens. After the optimizer updates it, the SAME updated `b_j` will be added to ALL tokens in the next forward pass. So the gradient MUST reflect the combined effect on ALL tokens — not just one.

Think of it like a thermostat in an office:

- The thermostat (bias `b_j`) is set to 22°C for the entire office
- Person 1 says: "I'm cold, turn it UP by 2°"     (+2)
- Person 2 says: "I'm fine, turn it UP by 0.5°"    (+0.5)
- Person 3 says: "I'm hot, turn it DOWN by 1°"     (-1)
- Sum = +1.5° → the thermostat moves up slightly

This is the BEST compromise because you can only set ONE temperature for the whole office.
If you only listened to Person 1, you'd make Person 3 miserable.
If you only listened to Person 3, you'd make Person 1 freeze.
The SUM (or average) finds the direction that helps the MOST people!

### 7.3 — "Why Not Use a [16384, 768] Per-Token Bias Instead?"

This is a sharp question: "If summing loses information, why not give each token its OWN bias? Then each token gets its exact gradient, no mixing!"

**The killer problem: The tokens CHANGE every micro-batch!**

```
Micro-batch 1:                     Micro-batch 2:
  Token 0 = "The"                    Token 0 = "Quantum"
  Token 1 = "cat"                    Token 1 = "physics"
  Token 2 = "sat"                    Token 2 = "is"

Micro-batch 3:
  Token 0 = "India"
  Token 1 = "won"
  Token 2 = "the"
```

If bias row 0 was trained on "The", then "Quantum", then "India" — what should it learn? It CAN'T specialize because a completely different word appears at position 0 every time!

The shared `[768]` bias says: *"I don't care WHICH token you are. I'm a global offset for this feature."* And that's the correct design because:

1. **The weight matrix `W` already handles per-token differences.** Every token has a different `x` vector. When different `x` vectors multiply the same `W`, they produce different outputs. So `W` provides all the per-token customization — the bias just adds a global shift.

2. **Parameter explosion.** A per-token bias would add `16384 × 768 = 12,582,912` parameters per Linear layer. With 48 Linear layers, that's ~604 million extra parameters — **5x the entire GPT-2 124M model** — all of which would be useless because they'd overfit to random words.

3. **Generalization.** The "canceling" of gradients during the sum IS the mechanism that forces the bias to find a value that works for ALL possible words, not just the 16,384 words in this one micro-batch. Less parameters = more generalization = model learns RULES instead of memorizing examples.

### 7.4 — The Canceling is a Feature, Not a Bug

When positive and negative gradients cancel:

```
Token 0 says: "Push b_5 UP by +0.3"     (wants b_5 bigger)
Token 1 says: "Push b_5 DOWN by -0.2"   (wants b_5 smaller)
Token 2 says: "Push b_5 UP by +0.1"     (wants b_5 bigger)

Sum = +0.3 - 0.2 + 0.1 = +0.2
```

This means: 2 out of 3 tokens want `b_5` to go up, 1 wants it down. The sum `+0.2` is the **democratic vote** — the majority wins. The bias moves in the direction that helps the MOST tokens.

If the sum were exactly 0.0 (perfect cancellation), it means: "half the tokens want it up, half want it down — the current value is already the best compromise, don't move it!" That's the bias finding its optimal position — not information loss!

### 7.5 — How ∂L/∂y Distributes to ALL Three Gradients

One final important point: `∂L/∂y` (the `[16384, 768]` upstream gradient) is the raw material used to compute THREE separate gradients, each with a DIFFERENT math operation:

```
                        ∂L/∂y  [16384, 768]
                    (upstream gradient — raw material)
                           |
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          SUM rows      xᵀ · ∂L/∂y    ∂L/∂y · Wᵀ
              |            |            |
              ▼            ▼            ▼
         ∂L/∂b         ∂L/∂W        ∂L/∂x
         [768]       [768, 768]   [16384, 768]
      REDUCE KERNEL  GEMM KERNEL  GEMM KERNEL
      (bias update)  (weight      (pass to layer
                      update)      below)
```

- **∂L/∂b** (bias gradient): SUM the rows of `∂L/∂y` → `[768]` → fired by the reduce kernel
- **∂L/∂W** (weight gradient): `xᵀ · ∂L/∂y` → `[768, 768]` → fired by cuBLAS GEMM
- **∂L/∂x** (input gradient): `∂L/∂y · Wᵀ` → `[16384, 768]` → becomes `∂L/∂y` for the layer below

The SAME `∂L/∂y` tensor is used three times with three different operations.

**Why bias uses SUM while weights use MATMUL:**

- Bias was ADDED in forward → derivative is 1 → backward is just SUM
- Weights were MULTIPLIED by `x` in forward → derivative involves `x` → backward is MATMUL with `x`

**Addition in forward → Sum in backward.
Multiplication in forward → Matmul in backward.**

### 7.6 — Doubt: "Why calculate ∂L/∂x? We don't update inputs!"

This is a brilliant question. You might think: *"I understand updating weights `W` and bias `b`. But `x` is the input! We don't update the input data, so why waste a giant MATMUL calculating `∂L/∂x`?"*

**The Answer: Because `x` is NOT the raw input. It's the OUTPUT of the layer below!**

In a deep neural network, layers are stacked:

- **Layer 1** computes some math and produces an output.
- **Layer 2** takes Layer 1's output and uses it as its input `x`.

When Layer 2 is doing its backward pass, it computes `∂L/∂x`. For Layer 2, this answers: *"How much should my input change?"*
Layer 2 doesn't update `x`. Instead, **it hands `∂L/∂x` down to Layer 1**.

To Layer 1, that `∂L/∂x` from above BECOMES its `∂L/∂y` (its upstream gradient, the raw material)!

```
[Layer 2] Backward Pass:
    Receives ∂L/∂y
    Computes ∂L/∂W (updates its weights)
    Computes ∂L/∂b (updates its biases)
    Computes ∂L/∂x ──────────┐
                             │ (passes down the chain)
[Layer 1] Backward Pass:     ▼
    Receives ∂L/∂x   (which it calls ∂L/∂y)
    Computes its own ∂L/∂W
    Computes its own ∂L/∂b
    Computes its own ∂L/∂x ──▶ (passes to Embeddings)
```

If Layer 2 didn't calculate `∂L/∂x`, the backpropagation chain would be **broken**. Layer 1 would never receive its "blame report", and it would never know how to update its own weights!

*Note: The ONLY time we throw away `∂L/∂x` is at the very, very bottom of the network (the raw pixel/text data), because there are no more layers below it to pass the gradient to.*

---

## 8. Reduce Kernel Call Accountability — 49 Calls Per Micro-batch, 15,680 Total

This section documents exactly how many times the `unified_reduce_kernel_outer_vec4` kernel fires during GPT-2 124M training, WHY it fires that many times, and provides full accountability for every single call verified against Nsight Systems profiling data.

### 8.1 — GPT-2 124M Architecture: Where Are the Linear Layers?

GPT-2 124M has **12 Transformer Blocks** stacked on top of each other. Inside EVERY block, there are exactly **4 Linear layers** (each with a bias vector):

```
┌─────────────────────────────────────────────────────┐
│                  ONE Transformer Block               │
│                                                      │
│  1. c_attn   (QKV projection)     bias size: [2304]  │
│  2. c_proj   (Attention output)   bias size: [768]   │
│  3. c_fc     (MLP up-projection)  bias size: [3072]  │
│  4. c_proj   (MLP down-projection) bias size: [768]  │
│                                                      │
└─────────────────────────────────────────────────────┘
× 12 blocks = 48 Linear layers with biases
```

**Important:** The `lm_head` at the very end of GPT-2 is also a Linear layer (`[768] → [50304]`), but it has `bias=False` because of **weight tying** (it shares its weight matrix with the token embedding `wte`). So it does NOT contribute a bias gradient.

**Also important:** LayerNorm layers have their own biases, but those use a DIFFERENT kernel (`ln_backward_gamma_beta`), not the reduce kernel we're analyzing.

### 8.2 — The 48 Bias Gradient Calls

During the backward pass of ONE micro-batch, EACH of the 48 Linear layers needs its bias gradient computed. The reduce kernel fires once per layer:

| Layer Name | Bias Width (F) | Layers per Block | × 12 Blocks | Calls |
|---|---|---|---|---|
| `c_attn.bias` (QKV) | 2304 | 1 | × 12 | **12** |
| `c_proj.bias` (Attention) | 768 | 1 | × 12 | **12** |
| `c_fc.bias` (MLP up) | 3072 | 1 | × 12 | **12** |
| `c_proj.bias` (MLP down) | 768 | 1 | × 12 | **12** |
| | | | **Total:** | **48** |

Each call takes the upstream gradient `∂L/∂y` of shape `[16384, F]` and sums it down to `[F]`.

### 8.3 — The Secret 49th Call: Positional Embedding Gradient

Beyond the 48 bias gradients, there is one more OuterContiguous reduction that fires every micro-batch — the **positional embedding (`wpe`) gradient**.

The positional embedding is a `[1024, 768]` matrix (context length × embedding dim). During the forward pass, this SAME `[1024, 768]` matrix is broadcast across all `B=16` sentences in the batch:

```
Forward: wpe [1024, 768] is added to Sentence 1, Sentence 2, ... Sentence 16
         (broadcast over the Batch dimension)

Backward: Sum over the Batch dimension to get the gradient
         Input:  [16, 1024, 768] upstream gradient
         The GPU flattens T and Feature together → treats it as [16, 786432]
         Sums over the 16 rows → Output: [786432] = [1024 × 768]
```

**Proof from Nsys database:** This kernel launches with `gridX = 1536`. In the OuterContiguous kernel, `gridX = CEIL(Output_Size / 128)`. Since it uses `vec4` (float4): `1536 × 128 × 4 = 786,432 = 1024 × 768`. This mathematically proves it's the positional embedding gradient.

**Note:** This is NOT the same as a bias gradient. The bias reduces `[16384, F]` → `[F]`. The positional embedding reduces `[16, 786432]` → `[786432]`. The output is a full `[1024, 768]` matrix, not a 1D vector.

### 8.3.1 — Deep Dive: What is Positional Embedding and Why Only ONE Tensor?

Let's build this from scratch.

**1. What is Positional Embedding (`wpe`)?**
Unlike humans who read word-by-word left-to-right, the Transformer processes all 1,024 words in a sentence SIMULTANEOUSLY. It has no built-in concept of "order". Without help, the sentences "The cat sat" and "sat The cat" look exactly the same to the model's self-attention mechanism.

To fix this, we create a **Positional Embedding matrix (`wpe`)** of shape `[1024, 768]`.

- Row 0 is a 768-number vector that means "I am the 1st word".
- Row 1 is a vector that means "I am the 2nd word".
- Row 1023 is a vector that means "I am the 1024th word".

During the forward pass, we take the word vector (`wte`) and ADD the position vector (`wpe`) to it:
`Embedding = Word_Vector + Position_Vector`

Now the model knows exactly where every word is located.

**2. Why use only ONE tensor across 16 sentences?**
In a batch, we process `B=16` sentences at the same time. You might ask: *"Why don't we have 16 separate `wpe` matrices, one for each sentence?"*

Because the **rules of language are universal**. The concept of being the "5th word" is the exact same concept whether you are reading Sentence 1 or Sentence 16.

If we used 16 different matrices, the matrix for slot 15 would try to learn: *"What does it mean to be the 5th word... specifically in the 16th sentence of a batch?"*

But during training, we randomly shuffle our dataset! "The cat sat on the mat" might be in Sentence 16 today, and Sentence 1 tomorrow. The "batch slot" is a completely arbitrary hardware trick to keep the GPU busy. It has absolutely zero linguistic meaning. If the model tried to learn rules based on batch slots, it would learn pure garbage and overfit to the hardware setup.

**3. The Forward Broadcast and Backward Sum**
Because we only have ONE `[1024, 768]` matrix, we **broadcast** (copy) it to all 16 sentences during the forward pass.

In the backward pass, this single matrix receives blame (gradients) from ALL 16 sentences. Just like we learned with the chain rule, a broadcast in the forward pass means we MUST **SUM** the gradients in the backward pass. By summing the 16 gradients together, we get a much stronger, clearer signal of what each position means, and we update our one universal position matrix.

### 8.4 — Total: 49 Calls Per Micro-batch

```
48 bias gradients (from 12 blocks × 4 linear layers each)
 + 1 positional embedding gradient
────────────────────────────────
= 49 reduce kernel calls per micro-batch
```

### 8.5 — What is a Micro-batch vs a Training Step?

**One Micro-batch = One Forward Pass + One Backward Pass** through the entire model.

```
One Micro-batch:
  Tokens → Embeddings → Block 1 → Block 2 → ... → Block 12 → lm_head → Loss   (FORWARD)
  Loss → lm_head → Block 12 → Block 11 → ... → Block 1 → Embeddings            (BACKWARD)
```

During the backward pass, the 49 reduce kernels fire (48 biases + 1 wpe).

**One Training Step = Multiple Micro-batches + One Optimizer Update.**

The model needs to see a large "Global Batch" of data before updating weights, but the GPU can't fit the entire global batch in VRAM at once. So it processes smaller chunks (micro-batches) and **accumulates** the gradients:

```
GRAD_ACCUM_STEPS = 524288 / (B × T × world_size)
                 = 524288 / (16 × 1024 × 1)     ← for 1-GPU profiling run
                 = 32
```

So **one Training Step** looks like:

```
One Training Step:
  Micro-batch 1:  Forward + Backward on Sentences 1-16      → save gradients
  Micro-batch 2:  Forward + Backward on Sentences 17-32     → ADD to gradients
  Micro-batch 3:  Forward + Backward on Sentences 33-48     → ADD to gradients
  ...
  Micro-batch 32: Forward + Backward on Sentences 497-512   → ADD to gradients
  ──────────────────────────────────────────────────────────
  OPTIMIZER STEP (AdamW): update ALL weights using the accumulated gradients
  ZERO the gradient buffer
  End of Training Step 1.
```

**Key points:**

- The weights stay **FROZEN** for all 32 micro-batches within one training step
- The DATA changes every micro-batch (different sentences each time!)
- Only after all 32 micro-batches does the optimizer update the weights
- This is mathematically equivalent to processing all 512 sentences at once (which wouldn't fit in VRAM)

### 8.6 — Why 32 Micro-batches? (The "Noisy Gradient" Problem)

You might ask: *"If the GPU can only process 16 sentences at a time, why don't we just update the optimizer after those 16 sentences? Why accumulate for 32 micro-batches?"*

This solves the **Noisy Gradient Problem**.

If you update the model after only 16 sentences, the gradients will be extremely "noisy" and chaotic. Human language is diverse. If those 16 sentences happened to be purely about "Quantum Physics", the model would drastically change its weights to become a physics expert, forgetting how to talk about sports or cooking. It would zigzag wildly with every micro-batch.

To learn stable, generalized rules of English, the model needs to see a **massive, diverse group of sentences** before making a decision.

The target "Global Batch" size for GPT-2 is ~500,000 tokens (which equals about 512 sentences of length 1024). But an RTX 6000 Ada GPU can only fit 16 sentences in VRAM!

So we use **Gradient Accumulation**:

1. Run 16 sentences (Micro-batch 1). Calculate gradients. DON'T update weights. Just save the gradients in memory.
2. Run next 16 sentences (Micro-batch 2). Calculate gradients. ADD them to the saved gradients.
3. Repeat 32 times. (16 sentences × 32 = 512 sentences).
4. NOW, the accumulated gradient represents the "smooth, averaged" learning from 512 diverse sentences. The noisy, wild anomalies of individual sentences have canceled out.
5. Finally, we run the Optimizer to update the weights.

This gives us the smooth, stable learning of a massive 512-batch size, while perfectly respecting the tight VRAM limits of the hardware. On a multi-GPU run (e.g. 8 GPUs), each GPU only needs `32 / 8 = 4` accumulation steps because the 8 GPUs collectively process 8× more data per micro-batch.

### 8.7 — Full Accountability: 15,680 Total Calls (10 Training Steps)

The Nsight Systems profiling run covered **10 training steps**. Total micro-batches processed:

```
10 steps × 32 micro-batches per step = 320 micro-batches
```

Total reduce kernel calls:

```
320 micro-batches × 49 calls per micro-batch = 15,680 calls
```

**Breakdown by source:**

| Source | Calls per micro-batch | × 320 micro-batches | Total Calls |
|---|---|---|---|
| `c_attn.bias` (F=2304), 12 layers | 12 | × 320 | **3,840** |
| `c_proj.bias` attention (F=768), 12 layers | 12 | × 320 | **3,840** |
| `c_fc.bias` (F=3072), 12 layers | 12 | × 320 | **3,840** |
| `c_proj.bias` MLP (F=768), 12 layers | 12 | × 320 | **3,840** |
| Positional embedding `wpe` (F=786432) | 1 | × 320 | **320** |
| | | **Grand Total:** | **15,680** |

```
Verification: 3,840 + 3,840 + 3,840 + 3,840 + 320 = 15,680 ✅
```

This **exactly matches** the count in the Nsight Systems SQLite database (`may27_WITH_BLUBLAS.sqlite`). Every single kernel launch is accounted for — zero missing, zero extra.

### 8.8 — Corrected Sentence for the Technical Report

The original sentence in the BluBridge technical report said:
> *"this reduction is issued 49 times per training step across the bias widths F ∈ {768, 2304, 3072} of the attention and feed-forward projections."*

This had two inaccuracies:

1. It should say **"per micro-batch"** not "per training step" (otherwise 10 steps would only be 490 calls, not 15,680)
2. The 49th call is the **positional embedding gradient**, not a bias gradient

The corrected sentence:
> *"In the 124M configuration, this reduction is issued 49 times per micro-batch. 48 of these calls compute the bias gradients for the attention and feed-forward projections across widths F ∈ {768, 2304, 3072}, while the 49th call reduces the B dimension to compute the T × F (1024 × 768) positional embedding gradient."*

---

## 9. Compile-Time Layout Specialisation (No More Index Math!)

Now let's move to the next paragraph in your paper:

> *"The kernel (unified_reduce_kernel) carries the memory layout as a non-type template parameter PATH, resolving the three reduction geometries into three separate branch-free specialisations at compile time: InnerContiguous... OuterContiguous... and a Generic stride-based fallback."*

### 9.1 — What Problem is This Solving?

When you have a multi-dimensional tensor and you want to "sum up" (reduce) along one specific dimension, the memory access pattern depends on **which dimension you're reducing**.

Consider a 2D matrix stored in **row-major** order in memory:

```
Logical view:               Memory layout (linear, row-major):
                             
  col 0  col 1  col 2       Address:  0    1    2    3    4    5    6    7    8
┌──────┬──────┬──────┐       
│ a0,0 │ a0,1 │ a0,2 │  →   [a0,0] [a0,1] [a0,2] [a1,0] [a1,1] [a1,2] [a2,0] [a2,1] [a2,2]
├──────┼──────┼──────┤           ←── row 0 ──→    ←── row 1 ──→    ←── row 2 ──→
│ a1,0 │ a1,1 │ a1,2 │       
├──────┼──────┼──────┤       Rows are CONTIGUOUS in memory (elements next to each other)
│ a2,0 │ a2,1 │ a2,2 │       Columns are STRIDED (jump by 3 to go down a column)
└──────┴──────┴──────┘
```

Now, **two completely different reduction patterns** emerge:

#### Case 1: InnerContiguous (Reduce the LAST axis — sum across rows)

You're summing along the innermost (last) dimension. The elements are right next to each other in memory: `[a0,0, a0,1, a0,2]`. This is extremely fast because you just read straight through memory.

#### Case 2: OuterContiguous (Reduce the FIRST axis — sum down columns)

This is the bias-gradient case! You are summing down a column.
To get Column 0, you read: `a0,0` (addr 0), `a1,0` (addr 3), `a2,0` (addr 6).
The output columns are contiguous, but the elements you're reading are separated by a **stride**.

### 9.2 — What is a "non-type template parameter"?

If you wrote this normally in C++, it would look like this:

```cpp
// Normal function — decides WHAT to do at RUNTIME
void reduce(int PATH) {
    if (PATH == 0) { /* InnerContiguous code */ }
    else if (PATH == 1) { /* OuterContiguous code */ }
    else { /* Generic code */ }
}
```

**The problem:** If you do this in CUDA, every single thread has to evaluate that `if/else` statement. That's a branch, which wastes instructions and hurts performance. But wait — the memory layout *never changes* during the kernel execution!

So we use a C++ template:

```cpp
// Template function — decides WHAT to do at COMPILE TIME
template <int PATH>   // ← "non-type template parameter" (it's an int, not a type)
__global__ void unified_reduce_kernel() {
    if constexpr (PATH == 0) { /* InnerContiguous ONLY */ }
    else if constexpr (PATH == 1) { /* OuterContiguous ONLY */ }
    // ...
}
```

When the compiler sees `unified_reduce_kernel<1>()`, it **completely deletes** the InnerContiguous and Generic code. It generates a kernel binary that ONLY has the OuterContiguous math. **Zero branches, zero if-statements at runtime.**

### 9.3 — The OuterContiguous Math vs Generic Math

For the bias gradient (`OuterContiguous`), the paper says:
> *"each element access reduces to a single `base_ptr[idx * row_stride]` index"*

To sum column `j` across 16,384 rows (where row width is `F = 768`), the math is simply:
Start at `j`, and jump by 768 each time.

- Element 0: `base_ptr[j]`
- Element 1: `base_ptr[1 * 768 + j]`
- Element i: `base_ptr[i * 768 + j]`

**That's one multiply and one add.**

What if we didn't have this special path and had to use a **Generic** reduction for N-dimensional tensors?
The paper mentions:
> *"a general-purpose reduction... must evaluate an OffsetCalculator per element, which replaces integer division by the tensor shape with magic-number multiplication"*

If you have a 4D tensor and a flat index `idx`, converting `idx` to `(batch, channel, height, width)` requires expensive integer division (`/`) and modulo (`%`). GPUs hate integer division (it takes ~25 clock cycles!).

A famous trick (Granlund & Montgomery) replaces division with a "magic number" multiply and a bit-shift (which takes ~4 cycles). But even with this trick, doing it for *every single element* in a Generic reduction takes ~10 instructions.

**By specializing at compile time, you replace ~10 complex math instructions with exactly 2 (`idx * stride + j`). This saves registers, saves instruction cache, and makes the kernel run at pure memory bandwidth limits.**

---

## 10. Instruction-Level Parallelism (Hiding Latency!)

Next paragraph:
> *"Each thread holds VT0 = 4 independent accumulators acc[0..3] and consumes four strided elements per loop iteration, keeping four reduce operations in flight before the dependent combine..."*

### 10.1 — The Problem: The Latency Chain

Imagine a single thread is adding up numbers in a column:

```cpp
float sum = 0;
for (int i = 0; i < 16384; i++) {
    sum = sum + data[i * stride];
}
```

In assembly, the `sum + data` operation is a Floating Point Add (`FADD`).
On our Ampere RTX 3060 a dependent `FADD`/`FFMA` takes about **4 clock cycles**
(this is GPU-specific — it was ~24 on older Fermi cards; see §10.3 for the full
table and sources).

But look at the code: iteration `i=1` CANNOT start until iteration `i=0` is completely finished, because it needs the new value of `sum`.

This is a **dependency chain**. The thread issues an `FADD`, then waits ~4 cycles for
the result before it can issue the next one. In this single-chain picture the math
pipe is mostly idle — though on a real GPU other resident warps also fill those gaps
(the precise "how much parallelism is needed" accounting is in §10.3).

### 10.2 — The Solution: Multiple Independent Accumulators

Instead of one `sum` variable, we give the thread **4 independent accumulators** (which live in 4 separate fast registers):

```cpp
float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

for (int i = 0; i < 16384; i += 4) {
    acc0 = acc0 + data[(i+0) * stride];
    acc1 = acc1 + data[(i+1) * stride];
    acc2 = acc2 + data[(i+2) * stride];
    acc3 = acc3 + data[(i+3) * stride];
}

// After the loop, fold them together
float final_sum = acc0 + acc1 + acc2 + acc3;
```

**Why does this matter?**
These four additions are completely independent! `acc1` does not need the result of `acc0`.

So the GPU can issue the `FADD` for `acc0`, and in the VERY NEXT CLOCK CYCLE, issue the `FADD` for `acc1`, then `acc2`, then `acc3`.

By the time it's ready to do the next `acc0` addition in the next loop iteration, the 4 clock cycles of latency from the previous `acc0` addition have already passed!

The latency is **hidden** because the thread had other independent work to do while
waiting. This is **Instruction-Level Parallelism (ILP)**: it overlaps the dependency
chains so the math units stall far less. (It doesn't magically reach "100% with zero
stalling" on its own — the real requirement, ILP combined with enough resident warps,
is worked out in §10.3.)

> **Scope note:** this 4-accumulator pattern is the **baseline for all three paths**
> (InnerContiguous, OuterContiguous, Generic) and all dtypes. The one exception:
> **16-bit InnerContiguous** reductions switch to **8** accumulators fed by a single
> 128-bit vectorized load — see §10.3 for exactly when and why.

### 10.3 — Deep Dive: Why 4 Accumulators? (The honest version)

You might wonder: *"If 4 is good, why not 8 or 16?"* The earlier version of this
section gave a tidy story ("FADD = 4 cycles, so 4 accumulators hides it exactly").
That story is **too neat and partly wrong**. Here is what is actually true, with
sources.

**1. FP32 add latency is NOT a universal "4 cycles" — it depends on the GPU.**
The arithmetic pipeline latency for a dependent FP32 `FADD`/`FFMA` has changed a
lot across generations (measured by microbenchmark studies):

| Architecture | dependent FP32 FMA/ADD latency |
|---|---|
| Fermi | ~18–24 cycles |
| Kepler | ~9–11 cycles |
| Maxwell / Pascal | ~6 cycles |
| **Volta / Turing / Ampere / Ada** | **~4 cycles** |

Our **RTX 3060 is Ampere (sm_86)**, so ~4 cycles is correct *for this GPU* —
confirmed by Abdelkhalik et al., *"Demystifying the Nvidia Ampere Architecture
through Microbenchmarking"* (2022): FP32 `FFMA`/`FADD`/`FMUL` = **4-cycle latency,
0.25-cycle throughput (4 ops/cycle)**. It was NOT always 4 — on Fermi (Volkov's
era) it was ~24, which is why his work needed much more parallelism.

**2. "4 accumulators hides 4-cycle latency" is a simplification.**
The old cycle-by-cycle story assumed the scheduler issues **one** add per cycle.
It doesn't: the FP32 pipe accepts **4 FFMAs every cycle** (0.25-cycle throughput).
By **Little's Law**, to fully hide a 4-cycle latency at 4 ops/cycle you need about
`4 × 4 = 16` independent FP32 ops *in flight per scheduler* — and those come from
**(resident warps) × (per-thread ILP)**, not from 4 accumulators alone. With
several warps resident, each thread needs only a little ILP; at very low occupancy
you need more. So `VT0 = 4` is a **chosen default that balances ILP against
register pressure**, not a number derived from "FADD = 4 cycles."

**3. The one clean reason 4 is convenient for FP32: `float4` = 128 bits.**
The load path likes 16-byte (128-bit) transactions. Four FP32 values = exactly
128 bits, so grouping work in 4s lines up with one wide load. (This is about the
*load width*, not the add latency.)

**4. Volkov [2016] — what he actually established (and what he did not).**
Vasily Volkov's thesis *"Understanding Latency Hiding on GPUs"* (UC Berkeley,
EECS-2016-143) established the **principle**: high **ILP can substitute for high
occupancy** — you can keep the units busy with few warps if each thread carries
enough independent work (Little's Law). He did **not** prescribe "4 accumulators,"
and I could not verify any claim that he "tested 2/4/8 and found a plateau at 4" —
treat that as **not from the paper**. Also note: **PyTorch does not cite Volkov**
anywhere. (The only paper PyTorch cites near this code is Granlund–Montgomery 1994,
for the integer-divider magic numbers — which our `OffsetCalculator` also uses.)

**5. What PyTorch actually does for the width.**
PyTorch's reduction kernel keeps a default `vt0 = 4`, and for **16-bit** inputs its
sum/mean dispatch passes `input_vec_size = 8` so the *input-vectorized* path issues
one 128-bit load of **eight** fp16/bf16 values into **8 independent accumulators**
(`aten/src/ATen/native/cuda/Reduce.cuh`, `ReduceSumProdKernel.cu`). The "8" is to
keep the load 128-bit wide for 2-byte types — a *bandwidth* choice, coupled to the
accumulator count.

**6. What WE do now (updated — this used to say "future optimization").**
We implemented the same thing. Our `unified_reduce_kernel` now has an
**input-vectorized branch** for the **InnerContiguous** path: for 16-bit dtypes the
host (`build_reduce_config`) sets `input_vec_size = 8` and the kernel does one
128-bit `VecLoad<scalar_t,8>` per step into 8 independent accumulators; FP32 keeps
the scalar `VT0 = 4` path. It is gated to a contiguous reduced axis whose length is
a multiple of 8 and ≥ 128 (so every row base stays 16-byte aligned — no head/tail
peel), and only for value reductions (sum/mean/min/max), not argmin/argmax or
Welford variance.

**Measured on the RTX 3060 (before → after), `sum(axis=-1)`:**

- Aligned large reductions (e.g. `16384×1024`): **+2–6%** — small, because the path
  was *already ~92% of the card's ~360 GB/s memory bandwidth*. 16-bit reductions
  here are bandwidth-bound, not instruction-bound, so there isn't much to win.
- The awkward size `R = 1000` (= 8×125): **~+25%** (241 → 300 GB/s) — the scalar
  strided path was inefficient there; the wide contiguous load fixes it.
- FP32 and non-multiple-of-8 sizes: **unchanged** (correctly stay on the scalar
  path). No regressions; 112/112 reduction tests pass.

**Important scope:** this helps **InnerContiguous only** (reducing the contiguous
last axis). It does **not** apply to **OuterContiguous** (e.g. the GPT-2 bias-grad
`sum(axis=0)` this document centers on): there the reduced elements are strided in
memory, so a thread *cannot* issue a contiguous wide load — that path instead uses
*output* vectorization (the `float4` store path), which is a different axis.

#### Sources (for the latency / ILP / Volkov claims above)

- Vasily Volkov, *Understanding Latency Hiding on GPUs*, PhD thesis, UC Berkeley
  EECS-2016-143 (2016) —
  <https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf>
  (the ILP-vs-occupancy principle / Little's Law).
- H. Abdelkhalik et al., *Demystifying the Nvidia Ampere Architecture through
  Microbenchmarking and Instruction-level Analysis*, arXiv:2208.11174 (2022) —
  <https://arxiv.org/abs/2208.11174>
  (FP32 `FFMA`/`FADD` = 4-cycle latency, 0.25-cycle throughput on Ampere sm_8x).
- Z. Jia et al., *Dissecting the NVIDIA Volta GPU Architecture via
  Microbenchmarking*, arXiv:1804.06826 (2018) —
  <https://arxiv.org/pdf/1804.06826>
  (the ~4-cycle FMA latency first measured on Volta; same core pipeline as
  Turing/Ampere/Ada).
- PyTorch reduction kernel (the `vt0=4` default and `input_vec_size=8` for 16-bit):
  `aten/src/ATen/native/cuda/Reduce.cuh`, `ReduceSumProdKernel.cu`,
  `ReduceMomentKernel.cu`. Granlund–Montgomery (the only paper PyTorch cites here,
  for the integer divider): `aten/src/ATen/cuda/detail/IntegerDivider.cuh`.

---

## 11. Hierarchical Map-Reduce (Warp, Block, and Grid)

Let's deeply analyze the third paragraph:
> *"The thread-local partials are reduced in stages matched to the hardware Harris [2007]. `block_x_reduce` performs the intra-warp reduction with `__shfl_down_sync` butterfly shuffles across offsets 16, 8, 4, 2, 1 - register-to-register with no shared-memory traffic followed by a single shared-memory exchange of the warp leaders; `block_y_reduce` folds the partials held by the y-dimension threads through shared memory. When the reduced axis is too long for one block to cover efficiently, a host-side solver enables a multi-CTA global reduce: each block stages its block-reduced partial into a scratch buffer, a single `atomicAdd` per output tile elects the last arriving block as the combiner (a “last-CTA” semaphore), and only that block reads the staged partials back and produces the final value. This confines cross-block communication to one atomic per output tile and one `__threadfence`, rather than the per-element global atomics a naive multi-block reduction would issue."*

**(Note: this paragraph is accurate — it describes the real reduction pipeline.)**

We have potentially **millions** of numbers to crush down into **one** result, and a
GPU runs **tens of thousands of threads**. We can't just have everyone write to one
place (that's the bug from §6/the bias-grad fix — see Level 3). Instead the reduction
funnels through **four levels**, each using the tool matched to that scope:

> thread's own partials → **warp** (shuffle) → **block** (shared memory) →
> **grid/across-blocks** (staging buffer) → final answer.

**The block is a 2-D grid of threads:** `block_width` across (the **x** direction,
**always ≤ 32**) and `block_height` tall (the **y** direction). Take a 32×4 block =
128 threads as the running example. Because a **warp = 32 threads** and `block_width
≤ 32`, **each row is exactly one warp**:

```
            x=0   x=1   x=2  ...  x=31
   y=0  →  [ t    t    t   ...   t ]   ← row 0 = WARP 0
   y=1  →  [ t    t    t   ...   t ]   ← row 1 = WARP 1
   y=2  →  [ t    t    t   ...   t ]   ← row 2 = WARP 2
   y=3  →  [ t    t    t   ...   t ]   ← row 3 = WARP 3
```

Every thread already holds **its own partial** (`acc[0]`, from the ILP loop). 128
numbers → we need **one**. We collapse each **row** left-to-right (Level 1), then the
surviving **column** top-to-bottom (Level 2).

### Level 1 — Intra-warp: the butterfly shuffle (`block_x_reduce`)

`__shfl_down_sync(mask, value, offset)` lets a thread **read another thread's register
directly** — thread `i` grabs the value sitting in thread `i+offset`. **No memory at
all** — register-to-register, one instruction. It only works **inside one warp** (32
threads).

We fold the warp in halves with offsets 16, 8, 4, 2, 1. Here it is on a small
**8-thread** row (offsets 4, 2, 1) so every step is visible — a real 32-wide warp just
starts at 16:

```
threads:  t0  t1  t2  t3  t4  t5  t6  t7
values:    3   1   4   1   5   9   2   6        (true sum = 31)

offset 4:  each ti += t(i+4)   →  t0=3+5=8   t1=1+9=10   t2=4+2=6   t3=1+6=7
offset 2:  each ti += t(i+2)   →  t0=8+6=14  t1=10+7=17
offset 1:  t0 += t1            →  t0 = 14+17 = 31   ✅  (lane 0 has the whole row)
```

After `log2(32) = 5` steps, **lane 0 (x=0) of the warp holds that warp's full sum.**
In the code:

```cpp
for (int offset = dim_x >> 1; offset > 0; offset >>= 1)
    value = ops.combine(value, ops.warp_shfl_down(value, offset));   // line 411
```

After Level 1, each row is crushed into its **x=0** thread:

```
   y=0  →  [ S0 |  ·   ·  ...  · ]     S0 = sum of row 0 (warp 0)
   y=1  →  [ S1 |  ·   ·  ...  · ]     S1 = sum of row 1 (warp 1)
   y=2  →  [ S2 |  ·   ·  ...  · ]     S2 = sum of row 2 (warp 2)
   y=3  →  [ S3 |  ·   ·  ...  · ]     S3 = sum of row 3 (warp 3)
          (only the x=0 column holds real numbers now)
```

### Level 2 — Intra-block across warps: shared memory (`block_y_reduce`)

Now 4 numbers (S0..S3) sit in **4 different warps**. **Shuffle can't cross warps** — a
thread in warp 0 cannot read a register out of warp 3. So we use **shared memory**: a
small on-chip scratchpad all threads in the block can read/write.

**(1)** Every thread writes its value into shared memory at its 2-D slot:

```cpp
smem[threadIdx.x + threadIdx.y * blockDim.x] = value;   // line 422
```
The column we care about (x=0) lands in `smem[0], smem[32], smem[64], smem[96]`:
```
smem[0]=S0   smem[32]=S1   smem[64]=S2   smem[96]=S3
```

**(2)** `__syncthreads()` — a **barrier**: every thread waits until **all** have
written ([line 423]). Without it, the leader might read a slot before its owner wrote
it → garbage.

**(3)** The top row (`threadIdx.y == 0`) reads down its column and adds:

```cpp
if (threadIdx.y == 0) {
    value = smem[threadIdx.x];
    for (int i = 1; i < blockDim.y; i++)
        value = ops.combine(value, smem[threadIdx.x + i * blockDim.x]);   // lines 425-428
}
```
For the leader (x=0,y=0): `S0+S1+S2+S3` = **the block's total**.

```
   S0   S1   S2   S3      (parked in shared memory by the 4 warp-leaders)
    └────┴────┴────┘
           │ thread (0,0) reads all four and adds
           ▼
      BLOCK TOTAL         ← held by the block leader, thread (0,0)
```

So after Level 2: **one partial sum per block**, sitting in each block's thread (0,0).
(Note: one warp-leader **per warp**, not "32 leaders" — a 4-warp block has 4 of them.)

### Level 3 — Across blocks: staging buffer + "last-CTA" election

When the reduced axis is huge, **many blocks (CTAs)** cooperate on one output column —
say **17** of them. Each now holds one partial (P0..P16) in its thread (0,0). We must
add those 17. The hard question: blocks live on **different SMs** and **cannot use
shared memory or shuffles with each other** — only global memory (VRAM).

**What NOT to do (these are the lessons baked into the design):**

- **Plain write** `dst[col] = my_partial;` — all 17 blocks overwrite the same address;
  whoever writes **last** wins, the other 16 are erased → result = **1/17 of the
  truth**. *This was the actual bias-gradient bug* (see §6 / the 1/17, 1/6, 1/5
  ratios). It's an **overwrite**, not an add.
- **Non-atomic accumulate** `dst[col] += my_partial;` — this looks correct but is a
  *different* race, the **lost-update race**. `+=` is really three separate steps:
  **(1)** load `dst[col]` into a register, **(2)** add `my_partial`, **(3)** store it
  back. With 17 blocks doing that at once and no lock, they interleave:
  ```
  dst = 0
  Block A reads dst → 0
  Block B reads dst → 0       (both read 0 before either writes back)
  Block A writes 0 + P_A = P_A
  Block B writes 0 + P_B = P_B   ← clobbers A's write, P_A is lost
  ```
  So some adds silently vanish → wrong answer. `+=` is **not** the same as `atomicAdd`
  — it's the *unsafe* version. `atomicAdd` is exactly what makes that read-add-write
  **one indivisible step** the hardware won't interrupt; that atomicity is the whole
  difference.
- **`atomicAdd(dst, my_partial)`** — correct (only the **17 block-leaders** issue it,
  one per block), but they all **queue on one address** (the hardware serializes the
  read-add-writes → contention), and the order they arrive is **random every run**.
  Because floating-point add is **not associative** (`(a+b)+c` rounds differently than
  `a+(b+c)`), a random add order means the **last few bits of the result change run to
  run** — "non-reproducible bits." Tiny (a few ULP), but it breaks bit-exact
  determinism, which a best-precision build wants. The staging buffer reads the
  partials in a **fixed order** → same bits every run.

**What we do instead** — give every block its **own slot** so the writes never
collide, then let **one** block do the final add:

```
scratch buffer in VRAM (one slot per block — no collisions, all written in parallel):
   slot0=P0   slot1=P1   slot2=P2  ...  slot16=P16
```
```cpp
cta_buf[own_slot] = my_partial;          // line 663  — parallel, no race
__threadfence();                         // line 665  — make the write visible in VRAM
int prev = atomicAdd(&semaphores[col], 1);   // line 671 — bump an INTEGER counter
bool i_am_last = (prev == ctas_per_output - 1);   // line 672 — last arriver?
if (!i_am_last) return;                  // line 675 — everyone else just leaves
```

The only atomic here is a **tiny integer counter** that answers "am I the last block to
arrive?" — *not* an atomic on the data. 16 of the 17 blocks write their slot and exit;
exactly **one** (the last to arrive) continues to Level 4.

### Level 4 — The last CTA combines the parked partials and writes once

The elected last block now has all 17 partials sitting in the scratch buffer. It
**re-uses the Level 1–2 machinery** on them — but *which* part runs depends on the
path (the code gates it):

```cpp
value = identity;
for (slice of the 17 slots assigned to me)
    value = ops.combine(value, cta_buf[slot]);   // line 692 — read parked partials
value = block_y_reduce(value, ...);              // line 709 — fold across warps (shared mem), ALWAYS
if (should_block_x_reduce())                     // line 710 — only when the reduce axis is x
    value = block_x_reduce(value, ...);
if (leader) dst[col] = project(value);           // write the FINAL answer — once
```

**Read path:** `cta_buf[slot]` is a plain global load — `VRAM → L2 → L1 → register`.
Shared memory is **not** touched by the read; it only enters in `block_y_reduce`
afterwards (`register → shared mem → register`).

**Important — for our bias-gradient (OuterContiguous) case this is `block_y_reduce`
only (shared memory); the warp shuffle is *skipped*.** Why: here `threadIdx.x` = the
output column (each lane is a *different* bias element, no reduction across x →
`should_block_x_reduce()` is false), and the reduce axis is `threadIdx.y` + the CTAs.
So a single column's 17 partials land across `threadIdx.y = 0..16`, i.e. **17 different
warps** — and warp shuffle **cannot cross warps**, so shared memory is *required*, not
optional.

> **"For just 17 partials, why not load them into one warp and shuffle — skip Level 2?"**
> Tempting, but it doesn't fit this layout. A warp's 32 lanes are 32 *different output
> columns* here; a single column's partials are deliberately along `y` (across warps).
> To shuffle them you'd need a *different* staging layout (partials along x), wasting
> lanes and conflicting with "x = output column." The shuffle-only shortcut fits the
> **InnerContiguous** case (reduction runs *along x*, within a warp) — **not** the
> bias-grad case. **Checked against PyTorch** (`Reduce.cuh::global_reduce`, lines
> 824–871): PyTorch does the identical thing — `block_y_reduce` unconditionally,
> `block_x_reduce` only when `should_block_x_reduce()`. Neither codebase has a
> small-`cpo` warp-only fast path, and for this case it wouldn't help.

```
   slot0 slot1 slot2 ... slot16     (17 parked partials in VRAM)
     └─────┴─────┴───────┘
            │ last block loads them, re-reduces (shared mem; shuffle only for inner-axis)
            ▼
        dst[col] = P0+P1+...+P16     ✅  correct total, written exactly once
```

So cross-block talk is confined to **one integer atomic + one `__threadfence` per
output column** — never the per-element global atomics a naive version would fire.

### The whole funnel at a glance

| Level | Scope | Tool | Why this tool |
|---|---|---|---|
| **1** | inside a warp (≤32 threads) | `__shfl_down_sync` (registers) | free, no memory, threads already in lockstep |
| **2** | across warps in one block | shared memory + `__syncthreads` | shuffle can't cross warps; shared mem is on-chip & fast |
| **3** | across blocks (different SMs) | staging buffer + integer semaphore | blocks can't share registers/shared mem; own-slot writes avoid the overwrite race **and** atomic contention |
| **4** | the elected last block | `block_y_reduce` (shared mem); `block_x_reduce` only for inner-axis | reuse the cheap machinery; one deterministic write to `dst` |

**Why four tools instead of one:** each level can only talk within its own scope —
warps share registers, blocks share shared memory, but **across blocks the only shared
thing is global VRAM**. So we use the cheapest mechanism that reaches the needed scope,
and only "spend" the expensive global-memory coordination at the very top, once per
output column.

### A note on `cp.async` (why it doesn't help here)

`cp.async` (Ampere+) copies **global memory → shared memory directly**, skipping the
register round-trip. It's a great fit for compute-bound pipelines (matmul, attention)
that **double-buffer** — copy the next tile while computing the current one. It does
**not** fit this reduction:

- **Level 2** writes shared memory from values **already in registers** (the
  warp-reduced partials). `cp.async`'s source must be **global** memory, not a
  register — so it simply **cannot** be used here.
- **Level 4's** read of the parked partials *is* from global (`cta_buf`), so `cp.async`
  *could* technically stage them into shared memory. But it's a **one-shot** read of a
  handful of values with **nothing to overlap** it against — `cp.async` only pays off
  when the copy hides behind other work. Here you'd issue it and immediately wait. **No
  win, more complexity.**
- For the **main input load** (the millions of elements), reductions are
  **bandwidth-bound** and the vectorized loads already run near peak bandwidth.
  `cp.async` hides *latency*, it doesn't add *bandwidth* — so it helps compute-bound
  kernels, not a bandwidth-bound reduction.

---

## 12. Launch and Occupancy Tuning (Keeping the GPU Fed)

The final paragraph of the technical report discusses how the kernel is configured and launched from the CPU (host) to ensure maximum hardware utilization:

> *"All kernel arguments - the operator functor, the resolved GpuReduceConfig, the source and destination pointers, and the optional staging buffers are packed by value into a single ReduceOp struct passed as one kernel parameter, minimising cudaLaunchKernel argument overhead. The kernel carries `__launch_bounds__(NT, MinBlocksPerSM⟨NT⟩)`, where the blocks-per-SM target is derived from Ada’s 1536-thread-per-SM limit, and the block geometry (`block_width ≤ 32`, `block_height`) together with the multi-CTA split factor are chosen by a host-side configuration solver from the problem shape before launch, so no occupancy decision is made on the device."*

**( The Ada architecture thread limit of 1536 per SM is spot on).**

### 12.1 — Packing Arguments (Minimizing Launch Overhead)

When launching a CUDA kernel, every argument you pass takes CPU time to serialize and send to the GPU via the PCIe bus. If you pass 15 different pointers and variables, the `cudaLaunchKernel` API overhead becomes a bottleneck for fast kernels.
BluBridge solves this by packing ALL arguments (pointers, config structs, scratch buffers) into a **single `ReduceOp` struct passed by value**. This minimizes launch latency and gets the GPU computing faster.

### 12.2 — `__launch_bounds__` and Ada Lovelace Limits

The kernel is decorated with `__launch_bounds__(NT, MinBlocksPerSM)`. This tells the NVCC compiler exactly how many threads per block to expect, and how many blocks we want to fit on a single Streaming Multiprocessor (SM).

The paper specifically notes that this is derived from the **Ada architecture's 1536-thread-per-SM limit** (which perfectly matches your RTX 6000 Ada GPU). By giving the compiler this exact ceiling, it can aggressively optimize register allocation (knowing exactly how many registers it can afford per thread without breaking occupancy limits).

### 12.3 — Host-Side Configuration Solver

Instead of the GPU kernel doing math to figure out how to split up the work, a CPU-side "solver" looks at the tensor shapes (e.g., `[16384, 768]`) and calculates the exact `block_width`, `block_height`, and multi-CTA split factor BEFORE launching the kernel.
This guarantees that **zero occupancy decisions are made on the device**. The GPU gets a perfectly tailored grid config and focuses 100% of its clock cycles purely on crunching numbers.
