---
name: feedback_guard_our_gpus_during_runs
description: "Others' processes on our GPUs: DEFAULT = report at once + wait, never kill. Kill ONLY when the user says so for that campaign, only on the GPUs they name (said 11-06 on the 2x box; 11-13 on P032 GPUs 4-5, with 'not GPU 6 while adwaid runs'); log every action"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T15:39:42.205Z
---

**Final rule (user, 2026-11-05 ~19:07 IST): "don't kill anyone's process, let's use P032 then".** It replaces the user's
instruction ten minutes earlier (19:05, "lock our GPUs ... kill if anyone comes by chance"): I had written a gpu_guard.sh
that would SIGTERM/SIGKILL foreign processes on our GPUs; it NEVER ran. What happened: we stopped our P015 runs at the
user's request, and within a minute Sanjay's SFT training (train_sft.py, /localstorage/Sanjay/smollm2_sft) started on
our GPUs 6 and 7; I asked kill / other GPUs / wait, and the user chose to move to P032 instead of killing.

**UPDATE (user, 2026-11-05 ~21:12 IST): LOCK our GPUs for the whole campaign, from the NEXT campaign on.** What
happened: on P032 (items 5 + 8) colleagues took GPUs 1 and 2 in the GAPS between our runs (Bharath's RL job, Ram's
nova test); nobody entered a GPU during one of our runs (0 DISTURBED). User: "for now leave it, from next do this,
don't allow anyone while we are running, lock for us ok, currently proceed with our plan". So: start a tiny holder
process on every GPU of the campaign before the first run and stop it only after our last run there (no gap ever
opens); every run still free-checks (ignoring our own holder). If someone still lands on our GPU during our run:
report it at once. Killing stays OFF unless the user says "kill" for that case (21:10 the user said "kill if someone
runs while we are running if we can't lock"; 19:07 "don't kill anyone's process") -- ask before any kill.

**Why:** colleagues share the boxes; killing their jobs loses their work.

**How to apply:** before every run, free-check our GPUs (two samples, every process's cmdline); during our runs a watcher
may only RECORD foreign processes (run marked DISTURBED) and I report them to the user at once -- never kill, never touch
another GPU. If our GPUs are taken: tell the user and wait or move to another box / GPU as the user says.
Kill only OUR OWN processes, by exact PID with cwd/exe checked ([[feedback-ask-before-every-gpu-run]]).

**UPDATE 2026-11-06 ~12:30 IST (user, explicit): "if anyone starts a process, just kill those ok, no problem ... just kill
others' process at the start itself".** Context: on the 2x box 95.133.254.108 (shared: /root/mahesh, /root/ram), Ram's
_jit_runner_ ran on our locked GPU 0 during the smoke gate (12:11-12:13). So, during OUR campaigns:
- keep the holder lock;
- run gpu_guard.sh: a foreign process on our GPUs (not under our folder, not our holder) gets SIGTERM, then SIGKILL
  after 5 s;
- log every kill (pid, user, folder, command) and tell the user;
- never touch other GPUs or processes outside our campaign window.
This replaces the "ask before any kill" line above for our locked GPUs during our runs.

**UPDATE 2026-11-13 16:37-16:39 IST (P032, the attention short-chat study; user, explicit, several messages):**
"dont use 7 , and kill any other processes bro , use our gpus and complete our processes i am giving permission";
"wait, dont use 6 if adwaid is running"; "but 4 and 5, use, don't allow anyone"; "kill if anyone comes"; "run in 6 if no
one are using". Earlier the same day (about 16:00, when the runs started) I had declined an automatic kill and only waited; after two of our runs
collided with a colleague's short jobs the user repeated the instruction. What I did: `guard45.sh` for GPUs 4 and 5 only,
only while our scripts ran (SIGTERM, SIGKILL after 5 s, every action logged); it acted on NO foreign process (none came
while it ran); GPU 6 had no guard, our waiter there was stopped while Adwaid's jobs kept coming.
My reading of the pattern (11-05 "don't kill", 11-06 "kill", 11-13 "kill on 4 and 5, not Adwaid on 6"): the user decides
per campaign and per GPU. So the default stays report + wait; a kill needs the user's word for that campaign.
Guard faults found that day (fix them before any future guard runs):
- never `readlink -f /proc/<pid>/exe`: for a process that is already gone it prints the path itself, so the guard
  "found" one of our own just-exited runs as foreign; use plain `readlink` and require it to succeed + the process alive;
- our FA3 / FA4 / cuDNN runs are Python of a venv whose python links to /usr/bin/python3.12: the program path is outside
  our folder; a process is OURS if its cwd or anything on its command line is under our folder;
- all users on P015 / P032 are the same Unix user (operator): identify by folder, never by user name.
On Exclusive_Process boxes (P015, P032) no holder can run next to our tests; between two of our processes the GPU looks
free. A waiter of ours that grabs a GPU between a colleague's back-to-back jobs can make THEIR next job fail: do not
queue waiters on a GPU a colleague is cycling on.
