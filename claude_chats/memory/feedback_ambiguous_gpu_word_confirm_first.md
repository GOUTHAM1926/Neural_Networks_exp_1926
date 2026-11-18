---
name: feedback-ambiguous-gpu-word-confirm-first
description: When the user's GPU message is unclear, confirm in one line BEFORE launching; and never leave a job of ours that waits for a busy GPU and then starts by itself unless the user asked for that
metadata:
  type: feedback
---

2026-11-14 ~18:24 IST, P032. After a pause for a colleague, the user wrote "issue in gpu 1 and 3 bro, our gpu s la
continue bro". I read it as "use GPU 1 and 3, continue" and launched the chain at once. The user interrupted the
launch: "dont wait / dont run", then "wait". On the box, Dinesh's 8-GPU test (`broadcast_internode_compare_test`,
/home/operator/dinesh/...) had started 20 s before my launch; my gate (two samples, 5 s apart) saw GPUs 1 and 3 empty
because his job had not claimed them yet. Our smoke runs failed with "GPU busy", then WAITED, and ran for ~35 s the
moment his test ended, before my STOP file landed. Nothing of his was touched, but a job of ours grabbed GPUs he was
using between his runs.

**Why:** the user decides when our GPUs are really ours; colleagues run multi-GPU jobs in series, and a waiter of ours
that starts in a gap can break their next run. (My reading of the user's words: "issue in GPU 1 and 3" was a report of
a problem there, not "use"; this reading is mine, the user did not explain.)

**How to apply:**
- If a GPU message can be read two ways (which GPUs, start now or not), ask ONE short question first; do not launch.
  This does not conflict with [[feedback_proceed_when_gpu_given]]: that rule needs a clearly named GPU and a clear go.
- Right before a launch, look at ALL GPUs, not only ours: a job that is starting on every GPU will claim ours too.
- When our start finds the GPU busy: stop (STOP file), report who is there, and wait for the user's word. Do not leave
  a chain that waits and then starts by itself, unless the user asked for exactly that.
- Related: [[feedback_guard_our_gpus_during_runs]], [[feedback_rejected_launch_may_have_run]] (the interrupted launch
  HAD run in full; I checked the box before doing anything else).

**18/11 ~12:04 (user very angry, "did I tell you to run in GPU? who gave permission? stop immediately"):** I took the
user's "ok for all" (a reply to my list of questions, one of which was "which GPUs can I use? 1, 5, 6, 7 were free")
as GPU permission and started the dK check (GPU 5), a probe (GPU 6) and the rivals runner (GPUs 6 + 7). WRONG.
**Rule:** a GPU run needs the user's OWN words naming the GPU AND saying run/start, in the message itself. "ok", "ok for
all", "let's do this", approving a plan that contains GPU steps, or answering a list I wrote = NOT GPU permission.
Before any GPU process: ask one question ("start <what> on P0xx GPU n now?") and wait for an explicit yes.
