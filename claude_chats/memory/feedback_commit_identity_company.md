---
name: feedback-commit-identity-company
description: BlubridgeAI org repos = company identity; org PAT at ~/.config/blubridge/github_pat (never print/copy it); personal repo Neural_Networks_exp_1926 = personal identity
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-05T16:29:11.532Z
---

Which identity goes where:

| repo | commit author | push login |
|---|---|---|
| BlubridgeAI org repos — `Tensor-Implementations` (e.g. branch `sft_gautam`) | `Goutham Kumar <goutham.kumar@blubridge.com>` (already set repo-local in that submodule) | the **company** GitHub login |
| personal repos — `Neural_Networks_exp_1926` | personal `Goutham_Reddy_1926 <goutham64840@gmail.com>` (the global config) — **this is correct, do not change it** | personal `GOUTHAM1926` (the token saved in `~/.git-credentials`) |

**Why:** on 2026-09-24 I assumed "company account only" applied to every repo and told the user to
re-author a personal-repo commit. The user corrected me: the company account is for `sft_gautam` only;
the Neural_Networks folder stays personal.

**How the user pushes to the BlubridgeAI org (confirmed 2026-09-24, pushed `sft_gautam` a7da79d..1ed2ac3):**

    git push https://<ghp_token>@github.com/BlubridgeAI/<repo>.git <branch>

- **Since 2026-10-21 the org PAT lives in `~/.config/blubridge/github_pat`** (chmod 600, outside every repo; saved there at the
  user's request so it is never lost again). NEVER print it, never copy it into memory, notes, docs or any repo (the docs
  repo is pushed; GitHub secret scanning revokes leaked tokens). Push WITHOUT the raw token in the command line/history:
  `git push https://$(cat ~/.config/blubridge/github_pat)@github.com/BlubridgeAI/<repo>.git <branch>`.
  If the file is missing or the token is rejected, ask the user for a new one and store it the same way.
- Give exactly that ONE push command, straight away, and ask for the token.
- No fetch, no ls-remote, no credential diagnostics, no "create a new token" detour. That detour cost
  the user 30 minutes.
- It is the same GitHub account throughout. "Company" means the commit author email, not a separate
  login.

**2026-09-26 mistake (user angry: "this is a big mistake... I want my identity"):** I gave a BluTrain submodule-bump commit without checking BluTrain's identity. That box's BluTrain had none set, so it committed and PUSHED as `root <root@fair-soul-wanders-fin-02.datacrunch.io>`. The library repo on the same box was fine; each repo has its own repo-local config.

**2026-10-05 (user: "don't make it so complex"):** push guidance = ONE short command block (identity, patch, add the exact file, one-line `-m` under ~70 chars, push), nothing else. No message files, no expected-output lists, no extra check commands; do any verification myself afterwards. The explanation belongs in code comments, not in the commit.

**How to apply:**
- For EVERY org repo — the parent BluTrain AND the Tensor-Implementations submodule, on every NEW box — put `git config user.name; git config user.email` in the SAME block, before the commit.
- If either is empty or not `Goutham Kumar <goutham.kumar@blubridge.com>`, set it first: `git config user.name "Goutham Kumar"; git config user.email "goutham.kumar@blubridge.com"`.
- For a wrongly-authored pushed commit: `git commit --amend --reset-author --no-edit`, then `git push --force-with-lease=<branch>:<old sha>`.
- A push to the BlubridgeAI org fails with "Repository not found" when made with the saved personal token.
  It needs the company login in the URL (`https://<company-login>@github.com/BlubridgeAI/...`) and a
  company-account PAT.
- Never "fix" the personal repo's identity.
- See [[feedback_git_no_ai_contributor]].
