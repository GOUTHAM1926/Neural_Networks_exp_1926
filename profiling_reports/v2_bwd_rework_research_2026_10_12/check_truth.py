# Self-check of the fp64 "truth" formulas in bwd_precision_sim.py: finite differences of loss = sum(O * dO)
# on a tiny causal GQA case (3 query heads, 1 KV head, T=6, D=4) must match dQ, dK, dV from the formulas.
import numpy as np
rng = np.random.default_rng(0)
G, T, D = 3, 6, 4
Q = rng.standard_normal((G, T, D)); K = rng.standard_normal((T, D)); V = rng.standard_normal((T, D)); dO = rng.standard_normal((G, T, D))
sc = 1 / np.sqrt(D); causal = np.tril(np.ones((T, T), bool))
def fwd(Q, K, V):
    S = np.where(causal, np.einsum("gid,jd->gij", Q, K) * sc, -np.inf)
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    return P, np.einsum("gij,jd->gid", P, V)
P, O = fwd(Q, K, V)
dP = np.einsum("gid,jd->gij", dO, V); Dt = (dO * O).sum(-1, keepdims=True); dS = P * (dP - Dt)
fq = np.einsum("gij,jd->gid", dS, K) * sc; fk = np.einsum("gij,gid->jd", dS, Q) * sc; fv = np.einsum("gij,gid->jd", P, dO)
loss = lambda Q, K, V: (fwd(Q, K, V)[1] * dO).sum()
h = 1e-6; worst = 0.0
for (X, F, name) in ((Q, fq, "dQ"), (K, fk, "dK"), (V, fv, "dV")):
    num = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        Xp = X.copy(); Xp[idx] += h; Xm = X.copy(); Xm[idx] -= h
        args = {"dQ": (Xp, K, V), "dK": (Q, Xp, V), "dV": (Q, K, Xp)}[name]
        argm = {"dQ": (Xm, K, V), "dK": (Q, Xm, V), "dV": (Q, K, Xm)}[name]
        num[idx] = (loss(*args) - loss(*argm)) / (2 * h)
    err = np.abs(num - F).max(); worst = max(worst, err)
    print(f"{name}: max |finite diff - formula| = {err:.2e}  (max |grad| {np.abs(F).max():.2f})")
print("identity rowsum(dO*O) == rowsum(P*dP):", np.abs(Dt[..., 0] - (P * dP).sum(-1)).max())
print("TRUTH CHECK", "PASS" if worst < 1e-7 else "FAIL")
