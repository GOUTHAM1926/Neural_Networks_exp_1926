# claude_chats: backup of the Claude Code chats for this project

Readable, secret-free copy of the local Claude Code transcripts, so the work context survives if this laptop is lost.

## Refresh and push (any time, e.g. at end of day)

```bash
cd ~/Downloads/Neural_Networks_exp_1926
python3 claude_chats/sync_claude_chats.py      # ~5 s; must end with "secret scan: clean"
git add claude_chats
git commit -m "claude_chats: sync $(date +%F)"
git push
```

If the script prints `SECRET LEFT` and stops, do not commit; fix the redaction patterns first.

## What is in here

| Path | What |
|---|---|
| `memory/` | Claude's memory notes. `MEMORY.md` is the index of all notes. |
| `sessions/<date>_<id>_<title>/summaries/NNN_<IST time>.md` | Every compaction summary in full, numbered in order. The last one is the latest state. |
| `sessions/<date>_<id>_<title>/conversation/<YYYY-MM>.md` | Your messages and Claude's replies with IST timestamps, plus one line per tool call (tool name + what it did). |
| `sessions/<...>/session_info.md` | Time range, message counts, raw transcript size. |
| `sync_claude_chats.py` | The script that writes all of the above from `~/.claude/projects/...`. |

**Left out on purpose:** thinking, tool outputs, file snapshots, API request records and subagent transcripts. They make up
almost all of the raw size and are not needed to rebuild context.

**Size on 2026-10-26:** raw local transcripts 1.5 GB (the main chat file alone 1.39 GB, 91 compactions,
2026-05-12 to 2026-10-26). This backup is 36 MB, and no file is over 10 MB.

**Secrets:** GitHub tokens, HF tokens, API keys, private keys and `user:token@` URLs are replaced with
`<REDACTED:kind>`. The script re-scans every file it writes and refuses to finish if anything is left.

## Restoring context in a new Claude chat (new machine)

1. Clone this repo and the BLAS repo (`blublas`). The project documentation lives in `work_docs(mark-down - (.md) files )/`
   and in `blublas/docs/`.
2. Tell the new chat to read, in this order:
   1. `claude_chats/memory/MEMORY.md` and the notes it points to;
   2. the newest file in `sessions/*/summaries/`;
   3. the current month's `conversation/` file for the latest details.
3. Optional, to give Claude Code its memory back automatically: copy `claude_chats/memory/*` into
   `~/.claude/projects/<project-key>/memory/`. The project key is the project path with `/` replaced by `-`, e.g.
   `-home-blu-bridge016-Downloads-Neural-Networks-exp-1926`.
