# Side-by-side dev-suite table for checkpoint 6000 (end of epoch 1) and 11590 (end of epoch 2) (2026-10-23).
# Reads lm-eval's results_*.json under results/dev_<step>/ and results/endtoken_<step>.json; prints, per task and
# metric, both scores, the difference, and whether the difference is bigger than 2 combined standard errors
# (a difference smaller than that is within the noise of the benchmark's finite question set).
# usage: python3 compare_dev.py <results dir> [step A] [step B]
import glob, json, math, sys

R = sys.argv[1]
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("6000", "11590")
KEEP = {  # the metrics we read, per task (lm-eval key "<metric>,<filter>")
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none", "prompt_level_loose_acc,none"],
    "gsm8k_cot": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "drop": ["f1,none", "em,none"],
    "truthfulqa_mc2": ["acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "arc_easy": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "winogrande": ["acc,none"],
    "mmlu_pro": ["exact_match,custom-extract"],
    "agieval_en": ["acc,none", "acc_norm,none"],
}


def load(step, kind="dev"):
    out = {}
    for f in sorted(glob.glob(f"{R}/{kind}_{step}/**/results_*.json", recursive=True)):
        out.update(json.load(open(f))["results"])
    return out


for kind, title in (("dev", "chat template (dev_suite.sh)"), ("plain", "NO chat template (dev_plain.sh)")):
    ra, rb = load(A, kind), load(B, kind)
    if not ra and not rb:
        continue
    print(f"\n=== {title}")
    print(f"{'task':16s} {'metric':30s} {A:>9s} {B:>9s} {'B - A':>9s}  beyond noise (2 SE)?")
    for task, metrics in KEEP.items():
        if task not in ra and task not in rb:
            continue
        for m in metrics:
            va, vb = ra.get(task, {}).get(m), rb.get(task, {}).get(m)
            if va is None and vb is None:
                continue
            sk = m.replace(",", "_stderr,", 1)
            sa, sb = ra.get(task, {}).get(sk), rb.get(task, {}).get(sk)
            diff = (vb - va) if (va is not None and vb is not None) else None
            noise = ""
            if diff is not None and isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
                se = math.sqrt(sa ** 2 + sb ** 2)
                noise = f"{'YES' if abs(diff) > 2 * se else 'no '} (2 SE = {2 * se:.4f})"
            f = lambda v: f"{v:9.4f}" if isinstance(v, (int, float)) else f"{'-':>9s}"
            print(f"{task:16s} {m:30s} {f(va)} {f(vb)} {f(diff)}  {noise}")
print()
for s in (A, B):
    try:
        e = json.load(open(f"{R}/endtoken_{s}.json"))
        print(f"end-token rate, step {s}: {e['end_token_rate']:.3f} of {e['n']} answers end with <|endoftext|> "
              f"(hit the {e['max_new']}-token limit: {e['hit_limit_rate']:.3f}); generated length mean {e['gen_len_mean']:.0f} / "
              f"median {e['gen_len_median']:.0f} tokens vs real answers mean {e['real_len_mean']:.0f} / median {e['real_len_median']:.0f}")
    except FileNotFoundError:
        print(f"end-token rate, step {s}: not there yet")
