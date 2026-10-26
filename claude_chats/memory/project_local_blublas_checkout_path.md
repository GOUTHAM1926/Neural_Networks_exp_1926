---
name: project-local-blublas-checkout-path
description: THE local checkout of GOUTHAM1926/blublas.git is ~/Downloads/blublas (moved there 2026-09-12 from BLUBLAS/BLUBLAS_kota/BluBridge-BLAS). Never conclude it is missing.
metadata:
  type: project
---

# ~/Downloads/blublas

**MOVED 2026-09-12** out of `BLUBLAS/BLUBLAS_kota/` up to `~/Downloads/blublas` at the user's
request - the two wrapper folders were pointless nesting. That is the local working copy of **`GOUTHAM1926/blublas.git`** - the daily-work BLAS repo that
the Verda box also clones. Branch `main`.

**DO NOT conclude "there is no local clone".** I have now done that twice.

**Why I keep missing it:** `find ~ -maxdepth 4` does not reach it - the `.git` sits at depth 5,
and the source files under it are depth 9+. [[feedback-edit-local-push-pull]] already records
this exact failure ("a shallow find reported 'not present locally' for trees that WERE present").

**How to find it reliably, if ever needed again:**
```
find ~ -maxdepth 8 -type d -name .git 2>/dev/null | while read d; do
  git -C "$(dirname $d)" remote -v 2>/dev/null | grep -qi blublas && echo "$(dirname $d)"; done
```
Ignore hits under `~/.local/share/Trash/` - those are deleted copies.

**The three local trees, and which remote each points at:**

| path | remote | role |
|---|---|---|
| `~/Downloads/blublas` | `GOUTHAM1926/blublas.git` | **daily work - pull/push here** | |
| `~/Downloads/Neural_Networks_exp_1926` | `GOUTHAM1926/Neural_Networks_exp_1926` | docs only - holds `cublas_rev_eng_binary/analyzations_h200.md` |
| `~/Downloads/BluBridge-BLAS_01_07_2026_` | `Blubridge-Tech/BluBridge-BLAS` | team repo, separate |

**So "copy it locally" means this path**, not the Neural_Networks folder. Run artefacts
(nsys/logs/result tables) go into a dated folder in Neural_Networks; SOURCE and tools live in the
blublas checkout. See [[project-active-repo]] and [[project-where-the-work-lives]].
