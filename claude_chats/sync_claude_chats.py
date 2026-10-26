#!/usr/bin/env python3
"""Back up the local Claude Code chats of this project as readable, secret-free Markdown.

Run it any time, then commit + push the claude_chats/ folder:

    python3 claude_chats/sync_claude_chats.py

What it writes (everything regenerated on every run; unchanged files stay byte-identical, so git
only records what is new):
    claude_chats/memory/                         copy of Claude's memory notes (MEMORY.md = index)
    claude_chats/sessions/<date>_<id>_<title>/
        summaries/NNN_<IST time>.md              every compaction summary, in full
        conversation/<YYYY-MM>.md                user messages + Claude's replies + one line per tool call
        session_info.md                          size, time range, counts

What it leaves out on purpose: thinking, tool outputs, file snapshots, API request records, subagent
transcripts. Those are what make the raw .jsonl 1.4 GB, and none of them is needed to rebuild context.

Secrets: every output line passes through redact(); a final scan of all written files must find zero
matches, otherwise the script exits with an error and nothing should be committed.
"""
import collections
import datetime
import json
import os
import re
import shutil
import sys

PROJECT_DIR = os.path.expanduser('~/.claude/projects/-home-blu-bridge016-Downloads-Neural-Networks-exp-1926')
OUT = os.path.dirname(os.path.abspath(__file__))
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

SECRET_PATTERNS = [
    ('github-token', re.compile(r'ghp_[A-Za-z0-9]{36}')),
    ('github-token', re.compile(r'github_pat_[A-Za-z0-9_]{50,}')),
    ('github-token', re.compile(r'gh[ousr]_[A-Za-z0-9]{36}')),
    ('hf-token', re.compile(r'hf_[A-Za-z0-9]{30,}')),
    ('aws-key', re.compile(r'AKIA[0-9A-Z]{16}')),
    ('api-key', re.compile(r'sk-(?:ant-)?[A-Za-z0-9_-]{30,}')),
    ('slack-token', re.compile(r'xox[baprs]-[A-Za-z0-9-]{10,}')),
    ('private-key', re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----', re.S)),
]
URL_CRED = re.compile(r'(https?://)[^/\s:@"\'<>]+(?::[^/\s@"\'<>]+)?@')
SYSTEM_REMINDER = re.compile(r'<system-reminder>.*?</system-reminder>\s*', re.S)


def redact(text):
    for kind, pat in SECRET_PATTERNS:
        text = pat.sub(f'<REDACTED:{kind}>', text)
    return URL_CRED.sub(r'\1<REDACTED:url-credential>@', text)


def ist(ts):
    if not ts:
        return '?'
    t = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
    return t.astimezone(IST).strftime('%Y-%m-%d %H:%M IST')


def slug(s, n=48):
    s = re.sub(r'[^A-Za-z0-9]+', '-', s or 'untitled').strip('-').lower()
    return s[:n].rstrip('-') or 'untitled'


def user_text(content):
    """User-typed text only: no tool results, no injected system reminders."""
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [c.get('text', '') for c in content if isinstance(c, dict) and c.get('type') == 'text']
    else:
        return ''
    return SYSTEM_REMINDER.sub('', '\n'.join(parts)).strip()


def tool_line(block):
    inp = block.get('input') or {}
    what = (inp.get('description') or inp.get('file_path') or inp.get('url') or inp.get('query')
            or inp.get('skill') or inp.get('action') or inp.get('pattern') or '')
    return f'> tool `{block.get("name", "?")}`: {str(what).splitlines()[0][:200] if what else ""}'


def write_if_changed(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and open(path, encoding='utf-8').read() == text:
        return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return True


def process_session(path):
    sid = os.path.basename(path)[:-len('.jsonl')]
    titles, counts = collections.Counter(), collections.Counter()
    first_ts = last_ts = None
    summaries = []                                   # (ts, text)
    months = collections.defaultdict(list)           # 'YYYY-MM' -> lines
    last_role = None
    with open(path, 'rb') as fh:
        for raw in fh:
            try:
                o = json.loads(raw)
            except ValueError:
                counts['unreadable line'] += 1
                continue
            typ = o.get('type')
            if typ == 'ai-title':
                for k in ('aiTitle', 'title', 'text', 'value'):
                    if isinstance(o.get(k), str):
                        titles[o[k]] += 1
            if typ not in ('user', 'assistant'):
                continue
            ts = o.get('timestamp')
            if ts:
                first_ts = first_ts or ts
                last_ts = ts
            month = (ts or '0000-00')[:7]
            content = (o.get('message') or {}).get('content')
            if typ == 'user':
                if o.get('isCompactSummary'):
                    text = user_text(content)
                    summaries.append((ts, text))
                    months[month].append(f'\n---\n\n**[{ist(ts)}] COMPACTION #{len(summaries)}**: the context was '
                                         f'summarised here; full text in `summaries/{len(summaries):03d}_*.md`.\n')
                    counts['compactions'] += 1
                    last_role = None
                    continue
                if o.get('isMeta'):
                    continue
                text = user_text(content)
                if not text:
                    continue
                counts['user messages'] += 1
                months[month].append(f'\n---\n\n### USER [{ist(ts)}]\n\n{text}\n')
                last_role = 'user'
            else:
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'text' and block.get('text', '').strip():
                        if last_role != 'assistant':
                            months[month].append(f'\n### CLAUDE [{ist(ts)}]\n')
                            last_role = 'assistant'
                        months[month].append('\n' + block['text'].strip() + '\n')
                        counts['assistant text blocks'] += 1
                    elif block.get('type') == 'tool_use':
                        if last_role != 'assistant':
                            months[month].append(f'\n### CLAUDE [{ist(ts)}]\n')
                            last_role = 'assistant'
                        months[month].append(tool_line(block) + '\n')
                        counts['tool calls'] += 1

    title = titles.most_common(1)[0][0] if titles else 'untitled'
    folder = os.path.join(OUT, 'sessions', f'{(first_ts or "0000-00-00")[:10]}_{sid[:8]}_{slug(title)}')
    changed = 0
    for i, (ts, text) in enumerate(summaries, 1):
        stamp = ist(ts).replace(' IST', '').replace(' ', '_').replace(':', '')
        changed += write_if_changed(os.path.join(folder, 'summaries', f'{i:03d}_{stamp}.md'),
                                    redact(f'# Compaction summary #{i} ({ist(ts)})\n\nSession `{sid}` "{title}"\n\n{text}\n'))
    for month, lines in sorted(months.items()):
        body = f'# Conversation {month}: "{title}"\n\nSession `{sid}`. Thinking and tool outputs are not included.\n' + ''.join(lines)
        changed += write_if_changed(os.path.join(folder, 'conversation', f'{month}.md'), redact(body))
    info = [f'# Session {sid}', '', f'- Title: {title}', f'- First message: {ist(first_ts)}', f'- Last message: {ist(last_ts)}',
            f'- Raw transcript size: {os.path.getsize(path) / 1e6:.1f} MB (not copied)']
    info += [f'- {k}: {v}' for k, v in sorted(counts.items())]
    changed += write_if_changed(os.path.join(folder, 'session_info.md'), redact('\n'.join(info) + '\n'))
    return folder, title, counts, changed


def copy_memory():
    src, dst = os.path.join(PROJECT_DIR, 'memory'), os.path.join(OUT, 'memory')
    changed, keep = 0, set()
    for root, _, files in os.walk(src):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), src)
            keep.add(rel)
            text = open(os.path.join(root, f), encoding='utf-8', errors='replace').read()
            changed += write_if_changed(os.path.join(dst, rel), redact(text))
    if os.path.isdir(dst):                           # drop notes deleted from memory
        for root, _, files in os.walk(dst):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), dst)
                if rel not in keep:
                    os.remove(os.path.join(root, f))
                    changed += 1
    return len(keep), changed


def scan_outputs():
    leaks = []
    for root, _, files in os.walk(OUT):
        for f in files:
            if not f.endswith('.md'):
                continue
            p = os.path.join(root, f)
            text = open(p, encoding='utf-8').read()
            for kind, pat in SECRET_PATTERNS:
                if pat.search(text):
                    leaks.append((p, kind))
            for m in URL_CRED.finditer(text):
                if '<REDACTED' not in m.group(0):
                    leaks.append((p, 'url-credential'))
    return leaks


def main():
    sessions = sorted(f for f in os.listdir(PROJECT_DIR) if f.endswith('.jsonl'))
    for f in sessions:
        folder, title, counts, changed = process_session(os.path.join(PROJECT_DIR, f))
        print(f'{os.path.relpath(folder, OUT)}  "{title}"  {dict(counts)}  files changed: {changed}')
    n, changed = copy_memory()
    print(f'memory/  {n} notes, files changed: {changed}')
    leaks = scan_outputs()
    if leaks:
        for p, kind in leaks:
            print(f'SECRET LEFT: {kind} in {os.path.relpath(p, OUT)}')
        sys.exit('ERROR: secrets found in the output -- do NOT commit')
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
    print(f'secret scan: clean.  claude_chats/ total {total / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
