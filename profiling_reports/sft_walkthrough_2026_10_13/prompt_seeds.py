# Same tiny causal transformer and the same chat, 20 random initialisations. One plain step (lr 0.5) with the SFT
# loss (answer + END only) and with the all-token loss. Change of the probability the model gives to:
#   the next prompt word at positions 0 and 1, and the first answer word at position 3.
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
V, D, IGN, EOS = 12, 16, -1, 10
rms = lambda x: x / tf.sqrt(tf.reduce_mean(x * x, -1, keepdims=True) + 1e-6)
shapes = {"emb": (V, D), "head": (D, V), **{f"{n}{l}": (D, D) for l in (0, 1) for n in "qkvo"},
          **{f"w1{l}": (D, 4 * D) for l in (0, 1)}, **{f"w2{l}": (4 * D, D) for l in (0, 1)}}
def model(W, inp):
    T = len(inp); x = tf.gather(W["emb"], inp); causal = tf.constant(np.triu(np.full((T, T), -np.inf), 1))
    for l in (0, 1):
        h = rms(x); q, k, v = h @ W[f"q{l}"], h @ W[f"k{l}"], h @ W[f"v{l}"]
        x = x + (tf.nn.softmax(q @ tf.transpose(k) / np.sqrt(D) + causal, -1) @ v) @ W[f"o{l}"]
        x = x + tf.nn.gelu(rms(x) @ W[f"w1{l}"]) @ W[f"w2{l}"]
    return rms(x) @ W["head"]
ids, mask = [1, 2, 3, 4, 5, 6, EOS], [0, 0, 0, 0, 1, 1, 1]
inp = np.array(ids[:-1] + [EOS] * 3)
lab_sft = np.array([ids[t + 1] if mask[t + 1] else IGN for t in range(6)] + [IGN] * 3)
lab_all = np.array([ids[t + 1] for t in range(6)] + [IGN] * 3)
def step_change(W, lab):
    keep = lab != IGN
    with tf.GradientTape() as tape:
        tape.watch(list(W.values()))
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab[keep], logits=tf.boolean_mask(model(W, inp), keep)))
    g = dict(zip(W, tape.gradient(loss, list(W.values()))))
    W2 = {k: W[k] - 0.5 * tf.convert_to_tensor(g[k]) for k in W}
    p0, p1 = tf.nn.softmax(model(W, inp), -1).numpy(), tf.nn.softmax(model(W2, inp), -1).numpy()
    return np.array([p1[0, 2] - p0[0, 2], p1[1, 3] - p0[1, 3], p1[3, 5] - p0[3, 5]])
res = {"sft": [], "all": []}
for seed in range(20):
    r = np.random.default_rng(seed)
    W = {k: tf.constant(r.standard_normal(s) * 0.3) for k, s in shapes.items()}
    res["sft"].append(step_change(W, lab_sft)); res["all"].append(step_change(W, lab_all))
for name in ("sft", "all"):
    a = np.array(res[name])
    for j, what in enumerate(("next prompt word, pos 0", "next prompt word, pos 1", "first answer word, pos 3")):
        print(f"{name:3s} | {what:25s} | mean change {a[:, j].mean():+.4f} | went up in {int((a[:, j] > 0).sum()):2d}/20, down in {int((a[:, j] < 0).sum()):2d}/20")
