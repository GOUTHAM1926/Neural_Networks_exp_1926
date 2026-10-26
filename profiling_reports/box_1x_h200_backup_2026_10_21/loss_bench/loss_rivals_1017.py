#!/usr/bin/env python3
# 2026-10-17: the SFT loss of the other libraries, at OUR shape and on OUR data, so every number sits next to
# bench_sparse_ce_sft.cu (ours: forward save_stats + backward with stats, the trainer's pair).
#   shape  N = 8 x 4096 = 32,768 rows, V = 50,304, bf16 logits, real stage-0 labels (65535 = ignore -> -100 here)
#   cases  the SAME micro-batches as ours (read from ours.log: first / min / median / max trained), every row trained
#          (nomask, ours' replacement targets), every row masked (all_masked)
#   logits the SAME values as ours: the bench's hash fill (seed 1234), checked bit for bit against a pure-Python copy,
#          so each library's loss must equal ours (a cross-check of the case choice and of every library's maths)
# Part 1, loss only (the logits are given): torch_fp32 = F.cross_entropy(x.float()) (HF transformers' way, the reference for
#   the gradient check), torch_bf16 = F.cross_entropy(x) on bf16, torch_compile = torch.compile of the fp32 version (torchtitan's
#   way), liger = LigerCrossEntropyLoss. Per call, CUDA events: F alone, B alone (after an untimed F), F+B; median of --iters1
#   calls each, after >= --warm s of F+B (power-settled free clock). Memory: what the forward keeps for the backward (kept) and
#   the peak above the inputs during F+B (peak). Liger writes its gradient INTO the logits during the forward, so its logits are
#   restored from a clean copy before every call (untimed) and its backward is only a scale: only F+B is like for like.
# Part 2, lm_head + loss (hidden states X [N, 2304] and the lm_head W [V, 2304] given): torch = cuBLAS X W^T + the fp32 loss,
#   liger_flce = LigerFusedLinearCrossEntropyLoss (chunked, gradient in the forward), cce = Apple's linear_cross_entropy (never
#   builds the logits; default gradient filtering), cce_exact = the same without the filtering, gemm_only = the cuBLAS GEMMs
#   alone (F: X W^T; B: dX = G W and dW = G^T X), to be added to ours (our trainer runs our BLAS, ~cuBLAS speed on these).
# usage: python loss_rivals_1017.py <labels.bin> --ours-log ours.log [--part 12] [--iters1 200] [--iters2 30] [--warm 4]
#        [--impls1 a,b] [--impls2 a,b] [--cases1 a,b] [--cases2 a,b]
import argparse, random, re, struct, sys, time
import torch
import torch.nn.functional as F

NB, T, V, D = 8, 4096, 50304, 2304
N = NB * T
IGN = 65535
M32 = 0xFFFFFFFF


def log(*a):
    print(*a, flush=True)


# ---- labels and cases (bench_sparse_ce_sft.cu's choice) ---------------------------------------------------------------
def read_labels(path):
    raw = open(path, "rb").read()
    lab = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.int32) & 0xFFFF
    if lab.numel() % T or lab.numel() // T < NB:
        sys.exit(f"{path} is not [R, {T}] uint16")
    return lab


def ours_micro_batches(path):
    pick = {}
    for line in open(path):
        m = re.match(r"== (real_\w+)\s.*micro-batch (\d+) \(rows", line)
        if m:
            pick[m.group(1)] = int(m.group(2))
    return pick


def make_cases(lab, pick_ours):
    nb = lab.numel() // N
    per = lab[: nb * N].view(nb, N)
    cnt = (per != IGN).sum(1).tolist()
    order = sorted(range(nb), key=lambda b: cnt[b])
    pick = {"real_first": 0, "real_min": order[0], "real_median": order[nb // 2], "real_max": order[-1]}
    for k, b in pick_ours.items():
        if k in pick and b != pick[k]:
            log(f"note: {k}: ours chose micro-batch {b} ({cnt[b]} trained), this sort {pick[k]} ({cnt[pick[k]]}); using ours")
        pick[k] = b
    cases = {}
    for name in ("real_first", "real_min", "real_median", "real_max"):
        b = pick[name]
        cases[name] = (b, per[b].clone(), int(cnt[b]))
    t = per[0].clone()
    idx = torch.arange(N, dtype=torch.int32)
    cases["nomask"] = (0, torch.where(t == IGN, (idx * 7919) % V, t), N)
    cases["all_masked"] = (-1, torch.full((N,), IGN, dtype=torch.int32), 0)
    return cases, nb, sum(cnt) / (nb * N)


def to_targets(t):
    t = t.to(torch.int64)
    return torch.where(t == IGN, torch.full_like(t, -100), t).cuda()


# ---- logits: bench_sparse_ce_sft.cu's fill_logits, bit for bit --------------------------------------------------------
def fill_logits(seed=1234, chunk=1 << 26):
    n = N * V
    out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        h = torch.arange(s, e, dtype=torch.int64, device="cuda")
        h = ((h & M32) * 2654435761) & M32
        h ^= seed
        h ^= h >> 16
        h = (h * 0x7FEB352D) & M32
        h ^= h >> 15
        h = (h * 0x846CA68B) & M32
        h ^= h >> 16
        out[s:e] = (((h >> 8).to(torch.float32) * (1.0 / 16777216.0) - 0.5) * 8.0).to(torch.bfloat16)
        del h
    return out.view(N, V)


def ref_bits(i, seed=1234):
    h = ((i & M32) * 2654435761) & M32
    h ^= seed
    h ^= h >> 16
    h = (h * 0x7FEB352D) & M32
    h ^= h >> 15
    h = (h * 0x846CA68B) & M32
    h ^= h >> 16
    u = struct.unpack("<f", struct.pack("<f", (h >> 8) * (1.0 / 16777216.0)))[0]
    x = struct.unpack("<f", struct.pack("<f", (u - 0.5) * 8.0))[0]
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def check_fill(logits):
    flat = logits.view(-1).view(torch.int16)
    rng = random.Random(7)
    idx = [0, 1, 2, N * V - 1] + [rng.randrange(N * V) for _ in range(2000)]
    got = (flat[torch.tensor(idx, device="cuda")].to(torch.int32) & 0xFFFF).tolist()
    bad = [(i, g, ref_bits(i)) for i, g in zip(idx, got) if g != ref_bits(i)]
    if bad:
        sys.exit(f"logit fill differs from bench_sparse_ce_sft.cu at {len(bad)} of {len(idx)} points, e.g. {bad[:3]}")
    log(f"logit fill: {len(idx)} points equal to the C++ bench's values (bit for bit)")


# ---- timing ----------------------------------------------------------------------------------------------------------
def med_min(ev):
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2], ts[0]


def bench(fwd, prep, iters, warm):
    """fwd() -> loss (graph built); prep() before every call (untimed). Returns F / B / FB (median, min) + memory + loss."""
    t0, k = time.time(), 0
    while True:
        prep()
        fwd().backward()
        k += 1
        if k % 4 == 0:
            torch.cuda.synchronize()
            if time.time() - t0 >= warm:
                break
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    res = {"warm_calls": k}
    for kind in ("F", "B", "FB"):
        for a, b in ev:
            prep()
            if kind == "F":
                a.record(); loss = fwd(); b.record()
                del loss
            elif kind == "B":
                loss = fwd(); a.record(); loss.backward(); b.record()
            else:
                a.record(); loss = fwd(); loss.backward(); b.record()
        torch.cuda.synchronize()
        res[kind] = med_min(ev)
    prep()
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    loss = fwd()
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    loss.backward()
    torch.cuda.synchronize()
    res["kept_MB"] = (m1 - m0) / 2**20
    res["peak_MB"] = (torch.cuda.max_memory_allocated() - m0) / 2**20
    res["loss"] = float(loss.detach().float().item())
    return res


def out_line(part, impl, case, mb, trained, r, extra=""):
    f = lambda k: f"{r[k][0]:.4f} {k}min={r[k][1]:.4f}"
    log(f"LOSSBENCH part={part} impl={impl} case={case} mb={mb} trained={trained} F={f('F')} B={f('B')} FB={f('FB')} "
        f"kept_MB={r['kept_MB']:.1f} peak_MB={r['peak_MB']:.1f} loss={r['loss']:.6f} warm_calls={r['warm_calls']}{extra}")


# ---- Part 1: loss only ------------------------------------------------------------------------------------------------
def part1_impl(name):
    if name == "torch_fp32":
        return (lambda x, t: F.cross_entropy(x.float(), t, ignore_index=-100)), False
    if name == "torch_bf16":
        return (lambda x, t: F.cross_entropy(x, t, ignore_index=-100)), False
    if name == "torch_compile":
        return torch.compile(lambda x, t: F.cross_entropy(x.float(), t, ignore_index=-100), dynamic=False), False
    if name == "liger":
        from liger_kernel.transformers import LigerCrossEntropyLoss
        m = LigerCrossEntropyLoss(ignore_index=-100)
        return (lambda x, t: m(x, t)), True
    raise ValueError(name)


def part1(cases, names, impls, iters, warm):
    pristine = fill_logits()
    check_fill(pristine)
    ref_rows = {}
    for impl in impls:
        try:
            fn, inplace = part1_impl(impl)
        except Exception as e:
            log(f"SKIP part 1 {impl}: {type(e).__name__}: {e}")
            continue
        x = pristine.clone().requires_grad_(True)
        for case in names:
            mb, tl, trained = cases[case]
            t = to_targets(tl)
            first = int((t != -100).nonzero()[0]) if trained else -1

            def prep():
                x.grad = None
                if inplace:
                    with torch.no_grad():
                        x.copy_(pristine)
            try:
                r = bench(lambda: fn(x, t), prep, iters, warm)
            except Exception as e:
                log(f"FAIL part 1 {impl} {case}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            extra = ""
            if first >= 0 and x.grad is not None:
                g = x.grad[first].float().cpu()
                if impl == "torch_fp32":
                    ref_rows[case] = g
                elif case in ref_rows:
                    ref = ref_rows[case]
                    extra = f" grad_maxdiff_rel={float((g - ref).abs().max() / ref.abs().max()):.3e} row={first}"
            out_line(1, impl, case, mb, trained, r, extra)
            del t
            torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()
    del pristine
    torch.cuda.empty_cache()


# ---- Part 2: lm_head + loss -------------------------------------------------------------------------------------------
def part2(cases, names, impls, iters, warm):
    g = torch.Generator(device="cuda").manual_seed(0)
    X0 = torch.randn(N, D, device="cuda", generator=g).to(torch.bfloat16)
    W0 = (torch.randn(V, D, device="cuda", generator=g) / D**0.5).to(torch.bfloat16)   # logits ~ N(0, 1)
    for impl in impls:
        if impl == "gemm_only":
            G = (torch.randn(N, V, device="cuda", generator=g) * 1e-4).to(torch.bfloat16)
            for case in names[:1]:   # the GEMMs do not depend on the labels
                t0, k = time.time(), 0
                while time.time() - t0 < warm:
                    with torch.no_grad():
                        L = X0 @ W0.t(); dX = G @ W0; dW = G.t() @ X0
                    k += 1
                    if k % 4 == 0:
                        torch.cuda.synchronize()
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
                res = {"warm_calls": k}
                with torch.no_grad():
                    for kind in ("F", "B", "FB"):
                        for a, b in ev:
                            a.record()
                            if kind in ("F", "FB"):
                                L = X0 @ W0.t()
                            if kind in ("B", "FB"):
                                dX = G @ W0; dW = G.t() @ X0
                            b.record()
                        torch.cuda.synchronize()
                        res[kind] = med_min(ev)
                del L, dX, dW
                torch.cuda.empty_cache()
                res["kept_MB"] = N * V * 2 / 2**20      # the logits, kept for the backward of a non-fused loss
                res["peak_MB"] = (N * V * 2 * 2 + N * D * 2 + V * D * 2) / 2**20   # logits + their grad + dX + dW
                res["loss"] = float("nan")
                out_line(2, impl, "any", -1, -1, res, " note=cuBLAS_GEMMs_only_add_ours_loss")
            del G
            torch.cuda.empty_cache()
            continue
        try:
            if impl == "torch":
                fn = lambda X, W, t: F.cross_entropy((X @ W.t()).float(), t, ignore_index=-100)
            elif impl == "liger_flce":
                from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
                m = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
                fn = lambda X, W, t: m(W, X, t)
            elif impl in ("cce", "cce_exact"):
                from cut_cross_entropy import linear_cross_entropy
                if impl == "cce":
                    fn = lambda X, W, t: linear_cross_entropy(X, W, t, ignore_index=-100)
                else:
                    fn = lambda X, W, t: linear_cross_entropy(X, W, t, ignore_index=-100, filter_eps=None)
            else:
                raise ValueError(impl)
        except Exception as e:
            log(f"SKIP part 2 {impl}: {type(e).__name__}: {e}")
            continue
        X = X0.clone().requires_grad_(True)
        W = W0.clone().requires_grad_(True)
        for case in names:
            mb, tl, trained = cases[case]
            t = to_targets(tl)

            def prep():
                X.grad = None
                W.grad = None
            try:
                r = bench(lambda: fn(X, W, t), prep, iters, warm)
            except Exception as e:
                log(f"FAIL part 2 {impl} {case}: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            out_line(2, impl, case, mb, trained, r)
            del t
            torch.cuda.empty_cache()
        del X, W
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--ours-log", default="")
    ap.add_argument("--part", default="12")
    ap.add_argument("--iters1", type=int, default=200)
    ap.add_argument("--iters2", type=int, default=30)
    ap.add_argument("--warm", type=float, default=4.0)
    ap.add_argument("--impls1", default="torch_fp32,torch_bf16,torch_compile,liger")
    ap.add_argument("--impls2", default="gemm_only,torch,liger_flce,cce,cce_exact")
    ap.add_argument("--cases1", default="real_first,real_min,real_median,real_max,nomask,all_masked")
    ap.add_argument("--cases2", default="real_median,real_max,nomask")
    a = ap.parse_args()
    vers = [f"torch {torch.__version__} (CUDA {torch.version.cuda})"]
    for mod in ("triton", "liger_kernel", "cut_cross_entropy"):
        try:
            m = __import__(mod)
            vers.append(f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            vers.append(f"{mod} MISSING ({type(e).__name__})")
    p = torch.cuda.get_device_properties(0)
    log(f"device {p.name}, {p.multi_processor_count} SMs | " + " | ".join(vers))
    lab = read_labels(a.labels)
    pick = ours_micro_batches(a.ours_log) if a.ours_log else {}
    cases, nb, mean = make_cases(lab, pick)
    log(f"labels {a.labels}: {nb} micro-batches, mean trained {mean:.4f}; cases: "
        + ", ".join(f"{k} (mb {v[0]}, {v[2]} trained)" for k, v in cases.items()))
    log(f"timing: >= {a.warm} s of F+B warm-up, then F / B / F+B: {a.iters1} calls (part 1), {a.iters2} (part 2); CUDA events")
    if "1" in a.part:
        part1(cases, a.cases1.split(","), a.impls1.split(","), a.iters1, a.warm)
    if "2" in a.part:
        part2(cases, a.cases2.split(","), a.impls2.split(","), a.iters2, a.warm)
    log("RIVALS DONE")


if __name__ == "__main__":
    main()
