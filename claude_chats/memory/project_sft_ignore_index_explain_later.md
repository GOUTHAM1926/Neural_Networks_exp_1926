---
name: project-sft-ignore-index-explain-later
description: OPEN TEACHING ITEM — user asked to park the ignore_index = -1 / -100 / 65535 explanation and have it re-explained clearly later
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-21T11:01:56.545Z
---

**Parked 2026-09-21 at the user's request.** They said the explanation did not land and to
remind them later: *"we will deal with this later ok va, remind in future tasks to explain me
in clear"*.

## The three points that did NOT land

1. **Why `-100` does not work for us** — our targets are `uint16`, which cannot store a
   negative. Writing -100 into a uint16 stores 65436. The kernel then compares
   `65436 == -100` which is false forever, so masking silently does nothing. PyTorch uses
   -100 only because its targets are `int64`, a signed type that can hold it.
2. **Why the default is not `65535`** — 65535 is correct only for uint16 targets. The same
   kernel compiles for int32/int64 where a vocab above 65535 would make 65535 a REAL token
   that silently vanishes from the loss. `-1` can never match any of the three, so it is the
   only safe default.
3. **Why `-1` is allowed at all when negatives "cannot be stored"** — two different places.
   The DATA (`targets[row]`) is uint16, no negatives. The PARAMETER (`ignore_index`) is
   `int64_t`, negatives fine. `-1` only ever lives in the parameter.

## How to re-explain it when asked

Do NOT lead with types. Lead with the actual bytes: show what the preprocessing script writes
into the .bin file for a masked token, then show the kernel reading that same number back.
Three failed attempts all started from "uint16 is unsigned", and that framing did not work.
Try a table of "what is written to the file" vs "what the kernel compares against" with real
numbers in both columns.

Settled facts (do not re-derive): vocab 50304, targets uint16 (0..65535), unused range is
50304..65535, chosen sentinel is 65535, parameter default -1 means "match nothing".

See [[project_sft_kernel_changes]].
