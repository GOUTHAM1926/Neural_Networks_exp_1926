import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import importlib.util, numpy as np, tensorflow as tf
spec = importlib.util.spec_from_file_location("pg", "pad_grad_proof.py"); pg = importlib.util.module_from_spec(spec)
import sys, io; _o = sys.stdout; sys.stdout = io.StringIO(); spec.loader.exec_module(pg); sys.stdout = _o   # reuse the model, silence its prints
ids, mask = [1, 2, 3, 4, 5, 6, pg.EOS], [0, 0, 0, 0, 1, 1, 1]
inp, lab = pg.row(ids, mask, len(ids) - 1 + 3, pg.EOS)
g_sft, _ = pg.grads(inp, lab)                                    # SFT: only answer + END guesses checked
lab_all = np.array([ids[t + 1] if t < len(ids) - 1 else pg.IGN for t in range(len(inp))])
g_all, _ = pg.grads(inp, lab_all)                                # pretraining-style: every guess checked
for tok, what in ((1, "prompt word"), (2, "prompt word"), (3, "prompt word"), (4, "answer marker"), (5, "answer word"), (6, "answer word")):
    print(f"embedding row of id {tok} ({what:13s}): change from SFT loss {np.abs(g_sft['emb'][tok]).max():.3e} | from all-token loss {np.abs(g_all['emb'][tok]).max():.3e}")
print(f"output column of id 2 (a prompt word as a GUESS): SFT {np.abs(g_sft['head'][:, 2]).max():.3e} | all-token {np.abs(g_all['head'][:, 2]).max():.3e}")

# One plain training step (w -= lr * g) with each loss, then: probability the model gives to
#   the next PROMPT word at position 0 (reads id 1, true next = 2) and at position 1 (reads 2, true next = 3),
#   the first ANSWER word at position 3 (reads the marker 4, true next = 5).
def probs():
    p = tf.nn.softmax(pg.model(inp), -1).numpy()
    return p[0, 2], p[1, 3], p[3, 5]
base = [v.numpy().copy() for v in pg.W.values()]
def after(g, lr=0.5):
    for v, b, k in zip(pg.W.values(), base, pg.W): v.assign(b - lr * g[k])
    r = probs()
    for v, b in zip(pg.W.values(), base): v.assign(b)
    return r
b0 = probs(); s1 = after(g_sft); a1 = after(g_all)
print(f"P(next prompt word) pos 0: before {b0[0]:.4f} | after SFT step {s1[0]:.4f} | after all-token step {a1[0]:.4f}")
print(f"P(next prompt word) pos 1: before {b0[1]:.4f} | after SFT step {s1[1]:.4f} | after all-token step {a1[1]:.4f}")
print(f"P(first answer word) pos 3: before {b0[2]:.4f} | after SFT step {s1[2]:.4f} | after all-token step {a1[2]:.4f}")
