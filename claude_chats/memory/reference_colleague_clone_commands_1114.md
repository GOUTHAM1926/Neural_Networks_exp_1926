---
name: reference-colleague-clone-commands-1114
description: "Tested commands (2026-11-14) for a colleague to clone BluTrain dist_side_gautam_lottn with all 3 submodules; all 4 repos private; token handed over in a file, never in the chat"
metadata:
  node_type: memory
  type: reference
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-14T09:28:45.696Z
---

2026-11-14: the user's data-team colleague asked for our SFT code again (first time: 2026-10-29, see [[project_data_team_sft_guide_1029]]). The user asked for ready clone commands with the org token inside, and gave a ONE-TIME permission to print the token in the chat.

**What I handed over (tested on the laptop as a fresh machine: empty HOME, no git-lfs, no saved login; 19 s, 294 MB, 5,068 files all md5-equal to the pushed tree; pull + submodule update OK, git status clean):**

    git config --global url."https://<TOKEN>@github.com/".insteadOf "https://github.com/"
    git clone --recurse-submodules -b dist_side_gautam_lottn https://github.com/BlubridgeAI/BluTrain.git
    # later: cd BluTrain; git pull; git submodule update --init --recursive

**Two other forms tested the same day (same fresh-machine setup):**
- Token ONLY inside the BluTrain address (`git clone --recurse-submodules ... https://<TOKEN>@github.com/BlubridgeAI/BluTrain.git`, the form the user pushes with): BluTrain comes (829 files), all 3 submodules FAIL ("could not read Username for 'https://github.com'").
- One single command (`git -c url."https://<TOKEN>@github.com/".insteadOf="https://github.com/" clone --recurse-submodules ...`): all 4 come, nothing stored, but a later `git pull` fails without the token.
- The user first read the two-command form as "token only for submodules, where is the main clone": say plainly that command 1 = the token for all 4 repos, command 2 = the main BluTrain clone + submodules. The handover file now labels them "COMMAND 1 = the token", "COMMAND 2 = the clone".

**The user's correction, 14/11 (applies to every command set I write for other people):** "how can you say cd BluTrain, her folder name might be different ... give the folder in the clone command itself". A bare `cd BluTrain` works only from the folder where she ran the clone. FINAL form in the handover file: the folder is named in the clone command and the update lines use the same full path:

    git clone --recurse-submodules -b dist_side_gautam_lottn https://github.com/BlubridgeAI/BluTrain.git ~/sft_work/BluTrain
    cd ~/sft_work/BluTrain; git pull; git submodule update --init --recursive

Tested: `~/sft_work` did not exist and git created it with the clone (no mkdir), run from an unrelated folder, 20 s, 5,068 files md5-equal; update lines run from `/` work; a second clone into the same folder is refused ("already exists and is not an empty directory", exit 128) and leaves the first clone untouched. `~/sft_work` is the folder the 10-29 guide uses.

**Facts checked that day:**
- All 4 repos are private (anonymous `ls-remote` fails on each): BlubridgeAI/BluTrain, BlubridgeAI/Tensor-Implementations, kathir-23s/BluBridge-BLAS, adhityablubridge/Context_Parallelism. Three different owners, absolute https URLs in `.gitmodules`, so a token in the parent URL alone does not reach the submodules; the `insteadOf` line does.
- ALL three submodules are needed since 2026-11-06: the Makefile adds `-I$(DIST_DIR)/Context_Parallelism` and `bluscript_sft.cpp` uses its headers. The 10-29 guide fetched only Tensor-Implementations; `how_to_handle_sft.md` corrected 11-14, the `.docx` copy NOT rebuilt (python-docx is not on the laptop).
- The saved org token is a classic one (`ghp_`): its `repo` scope is read + write together, it cannot be made read-only. Whoever holds it can push to every repo the user's account can push to. A fine-grained read-only token would not cover the two repos owned by personal accounts.
- After the two commands the token sits in `~/.gitconfig` of the login that ran them, not in the clone.

**How the token was handed over (my choice, the user's reaction not known yet):** I did not print it even with the one-time permission. I wrote the full text with the token filled in by `sed` at write time to `~/.config/blubridge/sft_clone_for_colleague.txt` (chmod 600) and showed a masked copy in the chat. Reason: the chat is backed up into `claude_chats` in a pushed repo, the way the token leaked on 2026-11-03. The user reads the file with `cat` in a terminal; opening it in the VS Code editor and selecting it would attach the selection to the chat.

**Found on the way:** the home folder itself is a git work tree (`~/.git`, remote GOUTHAM992003/ML_implement, 0 commits), so `~/.config/blubridge/` (org token, HF token) is untracked INSIDE a repo, not "outside every repo". Reported to the user; nothing changed there.

Related: [[feedback-commit-identity-company]], [[feedback-guides-for-other-teams-scope]].
