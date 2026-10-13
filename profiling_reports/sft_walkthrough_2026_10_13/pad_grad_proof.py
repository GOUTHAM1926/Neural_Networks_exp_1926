# Does padding change any weight gradient? Tiny 2-layer causal transformer, float64, rows built exactly like
# prepare_sft_data.py build_row: inputs = ids[:-1] then padding, labels = ids[1:] with untrained -> ignore.
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
V, D, IGN = 12, 16, -1
EOS, PAD_OTHER = 10, 11                       # 10 plays <|endoftext|>, 11 = a separate pad id
rng = np.random.default_rng(0)
W = {k: tf.Variable(rng.standard_normal(s) * 0.3) for k, s in {
    "emb": (V, D), "head": (D, V), **{f"{n}{l}": (D, D) for l in (0, 1) for n in ("q", "k", "v", "o")},
    **{f"w1{l}": (D, 4 * D) for l in (0, 1)}, **{f"w2{l}": (4 * D, D) for l in (0, 1)}}.items()}
rms = lambda x: x / tf.sqrt(tf.reduce_mean(x * x, -1, keepdims=True) + 1e-6)
def model(inp):
    T = len(inp); x = tf.gather(W["emb"], inp)
    causal = tf.constant(np.triu(np.full((T, T), -np.inf), 1))
    for l in (0, 1):
        h = rms(x); q, k, v = h @ W[f"q{l}"], h @ W[f"k{l}"], h @ W[f"v{l}"]
        p = tf.nn.softmax(q @ tf.transpose(k) / np.sqrt(D) + causal, -1)
        x = x + (p @ v) @ W[f"o{l}"]
        x = x + tf.nn.gelu(rms(x) @ W[f"w1{l}"]) @ W[f"w2{l}"]
    return rms(x) @ W["head"]
def row(ids, mask, T, pad):
    n = len(ids) - 1; inp = np.full(T, pad); lab = np.full(T, IGN)
    inp[:n] = ids[:-1]
    for t in range(n): lab[t] = ids[t + 1] if mask[t + 1] else IGN
    return inp, lab
def grads(inp, lab):
    keep = lab != IGN
    with tf.GradientTape() as tape:
        logits = model(inp)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab[keep], logits=tf.boolean_mask(logits, keep))
        loss = tf.reduce_sum(ce) / keep.sum()
    return {k: tf.convert_to_tensor(g).numpy() for k, g in zip(W, tape.gradient(loss, list(W.values())))}, float(loss)
for title, ids, mask in (("single turn", [1, 2, 3, 4, 5, 6, EOS], [0, 0, 0, 0, 1, 1, 1]),
                         ("two turns  ", [1, 2, 4, 5, EOS, 1, 3, 4, 6, EOS], [0, 0, 0, 1, 1, 0, 0, 0, 1, 1])):
    n = len(ids) - 1
    gE, lE = grads(*row(ids, mask, n + 5, EOS))           # padded with the EOS id (what we do)
    gP, lP = grads(*row(ids, mask, n + 5, PAD_OTHER))     # padded with a separate pad id
    g0, l0 = grads(*row(ids, mask, n, EOS))               # no padding at all
    dEP = max(np.abs(gE[k] - gP[k]).max() for k in W); dE0 = max(np.abs(gE[k] - g0[k]).max() for k in W)
    print(f"{title}: loss {lE:.15f} / {lP:.15f} / {l0:.15f} (pad=EOS / pad=separate / no pad)")
    print(f"   max |grad difference| over ALL weights: pad=EOS vs pad=separate {dEP:.3e} ; pad=EOS vs no pad {dE0:.3e}")
    print(f"   embedding row of the separate pad id: max |grad| {np.abs(gP['emb'][PAD_OTHER]).max():.3e}")
    print(f"   embedding row of EOS (input side): max |grad| {np.abs(gE['emb'][EOS]).max():.3e} ;"
          f" output-head column of EOS: max |grad| {np.abs(gE['head'][:, EOS]).max():.3e}")
