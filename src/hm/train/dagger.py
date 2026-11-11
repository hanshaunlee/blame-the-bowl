"""DAgger data collection: roll out the student with assorted memory systems, record the states it
visits, and label every state with the full-history expert (rel sets included). The resulting
trials plug straight into BCDataset, whose information-matched labelling re-derives the target
for whatever memory subset is sampled.

    python -m hm.train.dagger --policy ckpt/policy_v2.pt --n 2000 --out data/dagger_r1.pkl
"""
from __future__ import annotations
import argparse, time, numpy as np, torch
from ..env.session import Trial, make_trial
from ..env.oracle import expert_from_state
from ..train.data import save, place_codes, TRIAL_MIX
from ..train.bc import load_policy, device
from ..eval.rollout import run_batch
from ..eval.memory import *


def to_trial(lt) -> Trial:
    tr = Trial(lt.ref.kind, lt.ref.k, lt.layouts, lt.sessions, lt.steps, lt.events, lt.successes, [])
    tr.place = lt.place
    tr.init_layouts = lt.ref.init_layouts
    tr.query_restock = lt.ref.query_restock
    # session bounds
    bounds, start = [], 0
    for s in range(len(lt.sessions)):
        n = sum(1 for st in lt.steps if st.session == s)
        bounds.append((start, start + n)); start += n
    tr.session_bounds = bounds
    # relabel every step with the full-history expert
    for i, st in enumerate(tr.steps):
        li = st.layout
        hist = [(j, e.ev) for j, e in enumerate(tr.events) if e.layout == li and j < st.n_avail]
        a, rel = expert_from_state(tr.layouts[li], tr.sessions[st.session][1], st.state, hist)
        st.action = int(a); st.rel = tuple(sorted(rel))
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--beta", type=float, default=0.3, help="prob. of executing the expert action instead")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--cross_only", action="store_true", help="only cross-session trials (k in 0,1,3,7)")
    ap.add_argument("--learned", nargs="*", default=[], help="also roll out with these learned writers")
    a = ap.parse_args()
    dev = device(); pol, _ = load_policy(a.policy, dev)
    rng = np.random.default_rng(a.seed)
    w = np.array([m[0] for m in TRIAL_MIX]); w = w / w.sum()
    refs = []
    for i in range(a.n):
        spec = {"kind": "cross", "k": int(rng.choice([0, 1, 3, 7]))} if a.cross_only else TRIAL_MIX[int(rng.choice(len(TRIAL_MIX), p=w))][1]
        tr = make_trial(rng, spec)
        tr.place = place_codes(len(tr.layouts), a.seed * 100003 + i)
        refs.append(tr)
    systems = [lambda r, B: NoMemory(B), lambda r, B: RecencySym(B), lambda r, B: RecencyFrame(B),
               lambda r, B: Reservoir(B), lambda r, B: ScriptedMEM(B),
               lambda r, B: OracleMemory(B, query_task=r.sessions[-1][1] if r.kind == "cross" else None),
               lambda r, B: OracleMemory(B, query_task=r.sessions[-1][1] if r.kind == "cross" else None)]
    if a.learned:
        from ..eval.learned_memory import LearnedMemory
        for path in a.learned:
            systems.append((lambda p: (lambda r, B: LearnedMemory(B, p, dev=dev)))(path))

    def factory(r):
        B = int(rng.choice([2, 4, 8]))
        return systems[int(rng.integers(len(systems)))](r, B)

    out, t0 = [], time.time()
    for i in range(0, a.n, a.chunk):
        lives = run_batch(pol, refs[i:i + a.chunk], factory, dev, deterministic=False, teacher_forcing=False,
                          beta=a.beta)
        out += [to_trial(lt) for lt in lives]
        print(f"{len(out)}/{a.n} trials, {sum(len(t.steps) for t in out)} steps, success(all sessions) "
              f"{np.mean([np.mean(t.successes) for t in out]):.3f}  {time.time()-t0:.0f}s", flush=True)
    save(out, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
