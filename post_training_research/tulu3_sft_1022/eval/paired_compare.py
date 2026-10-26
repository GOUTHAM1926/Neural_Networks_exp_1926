# Paired comparison of two checkpoints on the SAME benchmark questions (2026-10-23).
# compare_dev.py's "2 SE" test treats the two runs as independent samples; but both checkpoints answered exactly the same
# questions, so the per-question DIFFERENCE removes how hard each question is and is a much more sensitive test.
# For every task: match lm-eval's per-sample logs (samples_<task>_*.jsonl, --log_samples) by doc_id, take the per-question
# score of each checkpoint, and report the mean difference, its standard error (std of the differences / sqrt(n)), and
# for 0/1 scores how many questions only A or only B got right (the discordant pairs that decide the difference).
# usage: python paired_compare.py <results dir> <kind: dev|plain> [step A] [step B]
import glob, json, math, sys

R, KIND = sys.argv[1], sys.argv[2]
A, B = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("6000", "11590")
METRIC = {"ifeval": "prompt_level_strict_acc", "gsm8k_cot": "exact_match", "drop": "f1", "truthfulqa_mc2": "acc",
          "hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm", "winogrande": "acc",
          "mmlu_pro": "exact_match", "agieval_en": "acc"}


def samples(step, task):
    fs = glob.glob(f"{R}/{KIND}_{step}/**/samples_{task}_*.jsonl", recursive=True)
    if not fs:
        return None
    out = {}
    for line in open(sorted(fs)[-1]):
        s = json.loads(line)
        if task == "gsm8k_cot" and s.get("filter") not in (None, "strict-match"):
            continue                                           # one entry per filter: keep strict-match
        v = s.get(METRIC[task])
        if isinstance(v, list):
            v = v[0]
        if isinstance(v, bool):
            v = float(v)
        if isinstance(v, (int, float)):
            out[s["doc_id"]] = float(v)
    return out


print(f"paired {KIND}: {B} - {A}, per question")
print(f"{'task':15s} {'metric':24s} {'n':>6s} {A:>8s} {B:>8s} {'diff':>8s} {'SE':>7s} {'z':>6s}  only {A} right / only {B} right")
for task, metric in METRIC.items():
    a, b = samples(A, task), samples(B, task)
    if not a or not b:
        continue
    ids = sorted(set(a) & set(b))
    d = [b[i] - a[i] for i in ids]
    n = len(d); mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    z = mean / se if se > 0 else 0.0
    binary = all(a[i] in (0.0, 1.0) and b[i] in (0.0, 1.0) for i in ids)
    disc = f"{sum(1 for i in ids if a[i] > b[i]):5d} / {sum(1 for i in ids if b[i] > a[i]):5d}" if binary else "(scores not 0/1)"
    ma, mb = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    print(f"{task:15s} {metric:24s} {n:6d} {ma:8.4f} {mb:8.4f} {mean:+8.4f} {se:7.4f} {z:+6.2f}  {disc}")
print("|z| > 2: the difference is beyond the noise of these questions (about 95% confidence).")
