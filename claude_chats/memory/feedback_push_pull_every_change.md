---
name: feedback-push-pull-every-change
description: "EVERY source change gets committed and pushed to origin immediately, so any fresh clone has the latest work -- never leave changes only in a box working tree"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-10T10:12:09.542Z
---

**Stated 2026-09-10:** *"from now on, every edit or whatever happens we will push and pull, so
that new clones will contain latest works all, and remind this"*

## The rule

A source change is not finished until it is **committed and pushed to `origin/main`**. Not
"copied to the box", not "verified locally" -- pushed. Then any other machine pulls.

Do this **per change**, not at the end of the day. The window between editing and pushing is
exactly the window in which a released box destroys the work.

## Why -- it has nearly happened three times

- **2026-08-19:** a full day's V5 work existed only in a box working tree. Box released
  overnight, new box cloned `origin/main`, everything gone. ([[feedback_edit_local_push_pull]])
- **2026-09-09:** the H163 split-K pin and 13 H158 ncw1 configs existed only on
  `95.133.254.191`, uncommitted. Rescued into tarballs minutes before the instance was killed.
- **2026-09-10:** a fresh clone of `7ebfc65` had none of it, and I had to restore from those
  tarballs -- then discovered the restore was incomplete (missing Makefile), which cost half a
  day. All of that follows from not having pushed.

Pushed as `2d271cb` on 2026-09-10 (11 files, +5728/-396), which finally put the H158/H163 work
and the two Kota dispatcher files under version control.

## Practical notes

- **Two untracked files nearly broke a future clone:** `Bgemm_sm90_dispatcher_kota.cu` and
  `Bgemm_sm90_kota_configs.cuh`. The dispatcher calls them unconditionally now, so a clone
  without them does not build. When committing, check for NEW files with
  `git status --porcelain | grep "^??"`, not just modified ones.
- Stage source explicitly. Do NOT sweep in `*.bak`, `blutrain_*/` trees, build logs, or
  `h115_out/` -- there are ~148 untracked entries in that repo and almost none belong.
- The vendored copy at `blutrain_ours/Tensor-Implementations/BluBridge-BLAS/src` is a FILE COPY,
  not a checkout. After any pull, re-sync it from the repo `src/` or the trainer builds stale code.
- Commits are authored as `Goutham_Reddy_1926 <goutham.kumar@blubridge.com>`; the box's git
  identity was unset and had to be configured. No AI attribution --
  see [[feedback_git_no_ai_contributor]].
- **A local checkout DOES exist** at `Downloads/BLUBLAS/BLUBLAS_kota/BluBridge-BLAS` -- see the
  correction section below. Keep it pulled after every push from the box.

## THE LOCAL CLONE EXISTS -- I claimed twice that it did not (corrected 2026-09-10)

**`/home/blu-bridge016/Downloads/BLUBLAS/BLUBLAS_kota/BluBridge-BLAS`**
-> `https://github.com/GOUTHAM1926/blublas.git`, branch `main`, FULL clone (not shallow).

I searched with `find ~ -maxdepth 4 -name .git` and reported "no local checkout" -- the `.git`
sits at depth 5, and the path was already recorded in my own session notes as the local BLAS
path. Do not repeat that: search with **no depth cap**, or match on the remote URL:

    find /home/blu-bridge016 -name .git -maxdepth 8 2>/dev/null | while read g; do
      git -C "${g%/.git}" remote get-url origin 2>/dev/null | grep -q blublas && echo "${g%/.git}"
    done

## THE WORKING LOOP (established 2026-09-10)

    edit on the box  ->  commit+push from the box  ->  `git pull --ff-only` in the LOCAL clone

or the reverse once local is the editing point. After any pull on the box, **re-sync the
vendored copy** `blutrain_ours/Tensor-Implementations/BluBridge-BLAS/src` -- it is a file copy,
not a checkout, so the trainer otherwise builds stale code.

**Pull gotcha, hit on the first try:** two files (`Bgemm_sm90_dispatcher_kota.cu`,
`Bgemm_sm90_kota_configs.cuh`) were STAGED ADDS locally. `git stash push` reverts the index but
leaves the files on disk as UNTRACKED, and an incoming commit that creates the same paths then
aborts with "untracked working tree files would be overwritten". Fix: verify they are identical
to `git show origin/main:<path>`, rename to `.local_bak`, pull, then delete the backups.
`git stash push -u` would also work but sweeps in ~120 unrelated build artifacts.
