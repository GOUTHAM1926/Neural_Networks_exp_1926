---
name: project_claude_chats_backup
description: claude_chats/ in Neural_Networks_exp_1926 = secret-free Markdown backup of all Claude chats + memory; re-run the sync script and the user pushes
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-26T13:24:23.747Z
---

Created 2026-10-26 at the user's request ("if we lose this system ... whole work should stop").

**The backup:**
- `Neural_Networks_exp_1926/claude_chats/sync_claude_chats.py` (~5 s) writes:
  - `memory/` (copy of these notes);
  - `sessions/<date>_<id>_<title>/summaries/NNN_*.md` (every compaction summary);
  - `conversation/<YYYY-MM>.md` (user and Claude text + one line per tool call);
  - `session_info.md`.
- It redacts GitHub/HF/API keys and `user:token@` URLs. A final scan exits non-zero if anything is left.
- On 2026-10-26 it redacted 207 secrets, and an independent rescan found 0 left.
- See also `claude_chats/README.md`.

**Size:**
- Raw local transcripts: 1.5 GB. The main chat d36b6b0d is 1.39 GB, with 91 compactions, 2026-05-12..10-26.
- Even zstd -19 only gets the raw chat to 199 MB: over GitHub's 100 MB file limit, and it would be re-uploaded in full on every push.
- The extracted backup is 36 MB on disk, ~10 MB packed.

**Never push the raw .jsonl.** It holds 294 copies of GitHub PATs, an HF token and other keys. If a full raw backup is ever wanted: redact, then zstd, then upload to the user's HF bucket, not git.

**Why:** if the laptop is lost, a new Claude chat needs `memory/MEMORY.md` + the latest summary + the docs to continue the work.

**How to apply:**
- At EOD, and after big milestones, remind the user to run the sync and push. The user commits and pushes.
- Check that the script still ends with "secret scan: clean".
