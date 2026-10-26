# 2026-10-16: CPU model of v62's synchronisation protocol (one CTA): the producer warp and the two consumer warpgroups,
# mbarriers (phase + parity waits, arrival counts, TMA data landing at once), the ping-pong named barriers 1/2 (count 256 =
# both groups), the G5/F3 unit hand-over (sched_full / sched_unit). GEMM waits are instant (they depend on nobody).
# Random interleavings; a deadlock = nobody can move before all are done. Versions: G5 (fetch at the unit start),
# F3 (fetch before the epilogue, inactive groups before their tail = the hang of 2026-10-16), F3fix.
# 2026-10-17 (session_2026_10_17/deep/protocol_sim_k4.py): + K1 (only group 0 waits for the K/V of a loop tile), K4mbar (K1 +
# the O store staged in the last tile's K buffer: that stage released after the epilogue, 'last QK done' handshake on an
# mbarrier with parity waits = the quick-suite hang of 14:45 IST), K4cnt (the same with a monotonic counter = K4b).
import random, sys

NSTAGES = 3
class MBar:
    def __init__(s, count): s.count = count; s.pending = count; s.phase = 0
    def arrive(s, n=1):
        s.pending -= n
        assert s.pending >= 0
        if s.pending == 0: s.phase += 1; s.pending = s.count
    def done(s, parity): return (s.phase & 1) != parity      # try_wait.parity: true once the phase of this parity completed
class NBar:                                                   # named barrier, count 256 = 2 groups x 128
    def __init__(s): s.arrived = 0; s.gen = 0
    def arrive(s):
        s.arrived += 1
        if s.arrived == 2: s.arrived = 0; s.gen += 1
def decode(u, half, mode, NP):
    if mode == "dense":
        if half == 1: return None
        return dict(pair=u["pair"], L=None, last=True)
    nP = (u["L"] + 127) // 128; cpl = u["cpl"]
    pair = (nP - 1 - cpl) if half else cpl
    if half == 1 and pair == cpl: return None
    return dict(pair=pair, L=u["L"], last=(half == 1) or (nP - 1 - cpl == cpl))
def my_nkv_of(d, c):
    q_tile = 2 * d["pair"] + c
    m = (q_tile * 64 + 63) // 128 + 1
    if d["L"] is not None and q_tile * 64 >= d["L"]: m = 0
    return m

def run(units, mode, version, seed):
    rnd = random.Random(seed)
    full = [MBar(1) for _ in range(NSTAGES)]; empty = [MBar(2) for _ in range(NSTAGES)]
    qfull = [MBar(1), MBar(1)]; qempty = [MBar(1), MBar(1)]; sched_full = [MBar(1), MBar(1)]
    sched_unit = [None, None]; nb = {1: NBar(), 2: NBar()}
    kdone = [MBar(1), MBar(1)]; kcount = [0, 0]
    K1fam = version in ("K1", "K4mbar", "K4cnt"); K4 = version in ("K4mbar", "K4cnt")
    def kdone_signal(c, j):
        if version == "K4mbar": kdone[c].arrive()
        elif version == "K4cnt": kcount[c] = j + 1
    T = len(units)
    def producer():
        g = j = ju = stage = ph = 0
        uidx = 0
        if version != "G5":
            sched_unit[0] = uidx if uidx < T else -1; sched_full[0].arrive()
        while uidx < T:
            u = units[uidx]
            for half in (0, 1):
                d = decode(u, half, mode, None)
                if d is None: break
                if j > 0:
                    yield lambda j=j: qempty[0].done((j - 1) & 1)
                    yield lambda j=j: qempty[1].done((j - 1) & 1)
                if version == "G5" and half == 0:
                    sched_unit[ju & 1] = uidx; sched_full[ju & 1].arrive()
                qfull[0].arrive(); qfull[1].arrive()          # Q TMA (lands at once)
                for n in range(d["pair"] + 1):
                    if g >= NSTAGES:
                        yield lambda s=stage, p=ph: empty[s].done(p ^ 1)
                    full[stage].arrive()                        # K/V TMA (lands at once)
                    g += 1; stage += 1
                    if stage == NSTAGES: stage = 0; ph ^= 1
                j += 1
            ju += 1; uidx += 1                                  # the claim (one CTA: the next unit in order)
            if version != "G5":
                sched_unit[ju & 1] = uidx if uidx < T else -1; sched_full[ju & 1].arrive()
        if version == "G5":
            sched_unit[ju & 1] = -1; sched_full[ju & 1].arrive()
    def consumer(c):
        stage = ph = j = 0
        own, other = (1, 2) if c == 0 else (2, 1)
        def sync(b):
            g0 = nb[b].gen; nb[b].arrive()
            return lambda: nb[b].gen > g0
        def fetch(k):
            yield lambda: sched_full[k & 1].done((k >> 1) & 1)
            return sched_unit[k & 1]
        if c == 1: nb[1].arrive()
        nxt = None
        if version != "G5": nxt = yield from fetch(0)
        ju = 0
        while True:
            uidx = (yield from fetch(ju)) if version == "G5" else nxt
            if uidx < 0: break
            u = units[uidx]
            for half in (0, 1):
                d = decode(u, half, mode, None)
                if d is None: break
                mk = my_nkv_of(d, c); ns = d["pair"] + 1
                yield lambda j=j: qfull[c].done(j & 1)
                if K4 and mk == 0: kdone_signal(c, j)             # K4: an inactive group never reads K
                if ns > 0: yield lambda s=stage, p=ph: full[s].done(p)
                if mk > 0:                                      # prologue: QK(0) under the baton
                    yield sync(own); nb[other].arrive()
                    if K4 and mk == 1: kdone_signal(c, j)         # K4: last QK = QK(0)
                for n in range(1, mk):
                    prev = stage; nstage = stage + 1; nph = ph
                    if nstage == NSTAGES: nstage = 0; nph ^= 1
                    if not K1fam or c == 0:                     # K1: group 1 follows group 0's baton instead
                        yield lambda s=nstage, p=nph: full[s].done(p)
                    yield sync(own); nb[other].arrive()
                    if K4 and n == mk - 1: kdone_signal(c, j)     # K4: last QK done (diagonal step)
                    empty[prev].arrive()
                    stage, ph = nstage, nph
                qempty[c].arrive()
                if mk > 0:
                    if d["last"] and (version == "F3" or (version in ("F3fix", "K1", "K4mbar", "K4cnt") and mk == ns)):
                        nxt = yield from fetch(ju + 1)
                    s_last = stage
                    if not K4: empty[stage].arrive()            # K4: released after the epilogue
                    stage += 1
                    if stage == NSTAGES: stage = 0; ph ^= 1
                    if mk < ns: yield lambda s=stage, p=ph: full[s].done(p)
                elif d["last"] and version == "F3":
                    nxt = yield from fetch(ju + 1)
                for n in range(mk, ns):                          # inactive tail
                    cur = stage; nstage = stage + 1; nph = ph
                    if nstage == NSTAGES: nstage = 0; nph ^= 1
                    yield sync(own); nb[other].arrive()
                    if n + 1 < ns: yield lambda s=nstage, p=nph: full[s].done(p)
                    empty[cur].arrive()
                    stage, ph = nstage, nph
                if version in ("F3fix", "K1", "K4mbar", "K4cnt") and d["last"] and mk < ns:
                    nxt = yield from fetch(ju + 1)
                if K4 and mk > 0:                               # K4 epilogue: wait for the other group's last QK, stage O, release
                    if version == "K4mbar": yield lambda j=j: kdone[1 - c].done(j & 1)
                    else:                   yield lambda j=j: kcount[1 - c] >= j + 1
                    empty[s_last].arrive()
                j += 1                                          # (epilogue)
            ju += 1
    agents = {"producer": producer(), "wg0": consumer(0), "wg1": consumer(1)}
    waits = {k: (lambda: True) for k in agents}
    while agents:
        ready = [k for k in agents if waits[k]()]
        if not ready:
            return "DEADLOCK: " + ", ".join(sorted(agents))
        k = rnd.choice(ready)
        try: waits[k] = next(agents[k])
        except StopIteration: del agents[k]
    return "ok"

def units_dense(n, NP, rnd):
    return [dict(pair=rnd.randrange(NP)) for _ in range(n)]
def units_chats(n, rnd):
    out = []
    for _ in range(n):
        L = rnd.choice([rnd.randint(1, 4096), rnd.randint(1, 600), rnd.randint(385, 448), 129, 64, 65, 500, 4096])
        nP = (L + 127) // 128
        out.append(dict(L=L, cpl=rnd.randrange((nP + 1) // 2)))
    return out

if __name__ == "__main__":
    rnd = random.Random(1)
    for version in ("F3fix", "K1", "K4mbar", "K4cnt"):
        res = {"dense": [0, 0], "chats": [0, 0]}
        first = {}
        for trial in range(1000):
            for mode in ("dense", "chats"):
                us = units_dense(rnd.randint(1, 12), 32, rnd) if mode == "dense" else units_chats(rnd.randint(1, 12), rnd)
                r = run(us, mode, version, seed=trial)
                res[mode][0 if r == "ok" else 1] += 1
                if r != "ok" and mode not in first: first[mode] = (r, us[:4])
        print(f"{version:6s} dense ok/deadlock {res['dense'][0]}/{res['dense'][1]}   chats ok/deadlock {res['chats'][0]}/{res['chats'][1]}")
        for m, (r, us) in first.items(): print(f"       first {m} failure: {r}; units {us}")
