"""Policy-free memory quality: decision distortion along expert trajectories.

For each memory system, replay the expert trajectory of a trial, let the memory system see the
events in order, and at every memory-relevant step ask: does the expert *given only the memory
contents (+ working window)* take the same action as the full-history expert?  This is the
exact decision-distortion of the memory, independent of the student's imperfections.

    python -m hm.eval.distortion --n 400 --budgets 2 4 8 [--learned ckpt/writer_both_B4.pt ...]
"""
from __future__ import annotations
import argparse, json, numpy as np, torch, os
from ..env.world import *
from ..train.data import label_for, load
from ..train.bc import load_policy, device
from .memory import *
from .sweep import eval_trials


class _Ref:
    """Minimal stand-in for LiveTrial so heuristic systems can run on a dataset Trial."""

    def __init__(self, tr):
        self.events = tr.events
        self.ref = tr


def distortion_heuristic(tr, memsys):
    """Returns list of (step_index, agree) for memory-relevant steps."""
    memsys.reset_trial()
    ref = _Ref(tr)
    out = []
    ev_ptr = 0
    cur_session = -1
    for i, st in enumerate(tr.steps):
        if st.session != cur_session:
            if cur_session >= 0:
                memsys.on_session_end(ref, cur_session)
            cur_session = st.session
        while ev_ptr < st.n_avail:
            memsys.on_event(ref, ev_ptr); ev_ptr += 1
        if st.rel:
            mem_idx = [ei for ei, m in memsys.contents()]  # frame units count as their fact (optimistic)
            out.append((i, int(label_for(tr, i, mem_idx) == st.action)))
    return out


def distortion_learned(tr, ti, ck, pol, dev, B):
    from ..train import writer_rl as WR
    from ..train.writer_rl import Cache, Sim, simulate, slots_at_step
    WR.SYM_ONLY = bool(ck.get("sym_only", False))
    from ..models.writer import OnlineWriter, Consolidator
    from ..train.writer_rl import ONLINE_CTX, SLEEP_CTX, FEAT
    if not hasattr(distortion_learned, "cache"):
        distortion_learned.cache = Cache(pol, dev)
        distortion_learned.nets = {}
    key = id(ck)
    if key not in distortion_learned.nets:
        w = OnlineWriter(pol.d, ONLINE_CTX, FEAT).to(dev); w.load_state_dict(ck["writer"]); w.eval()
        c = Consolidator(pol.d, SLEEP_CTX, FEAT + 1).to(dev); c.load_state_dict(ck["cons"]); c.eval()
        distortion_learned.nets[key] = (w, c)
    w, c = distortion_learned.nets[key]
    sim = Sim(ti, distortion_learned.cache.get(ti, tr), B)
    with torch.no_grad():
        simulate(ck["mode"], w, c, [sim], dev, pol, greedy=True)
    out = []
    for i, st in enumerate(tr.steps):
        if st.rel:
            mem_idx = [ei for ei, m in slots_at_step(sim, i)]
            out.append((i, int(label_for(tr, i, mem_idx) == st.action)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--learned", nargs="*", default=[])
    ap.add_argument("--policy", default="ckpt/policy_v2.pt")
    ap.add_argument("--out", default="results/distortion.json")
    a = ap.parse_args()
    trials = eval_trials(a.seed, a.n)
    dev = device()
    pol = None
    if a.learned:
        pol, _ = load_policy(a.policy, dev)
    res = {}
    for B in a.budgets:
        rng = np.random.default_rng(0)
        systems = {
            "none": lambda r: NoMemory(B, rng), "recency_sym": lambda r: RecencySym(B, rng),
            "recency_frame": lambda r: RecencyFrame(B, rng), "random": lambda r: Reservoir(B, rng),
            "scripted_mem": lambda r: ScriptedMEM(B, rng), "scripted_mem_dropfail": lambda r: ScriptedMEM(B, rng, keep_fail=False),
            "oracle": lambda r: OracleMemory(B, rng, r.sessions[-1][1] if r.kind == "cross" else None),
        }
        learned = []
        for path in a.learned:
            ck = torch.load(path, map_location="cpu")
            if ck["B"] == B:
                learned.append((f"learned_{ck['mode']}" + ("_sym" if ck.get("sym_only") else ""), ck))
        for name, fac in list(systems.items()) + [(n, None) for n, _ in learned]:
            acc = {}
            for ti, tr in enumerate(trials):
                if fac is not None:
                    pairs = distortion_heuristic(tr, fac(tr))
                else:
                    ck = dict(learned)[name]
                    pairs = distortion_learned(tr, ti, ck, pol, dev, B)
                for i, ok in pairs:
                    st = tr.steps[i]
                    if tr.kind == "cross" and st.session == len(tr.sessions) - 1:
                        acc.setdefault(f"query/k{tr.k}", []).append(ok)
                    elif tr.kind == "cross":
                        acc.setdefault("within_session", []).append(ok)
                    else:
                        acc.setdefault(f"single/{tr.sessions[0][1].name}", []).append(ok)
            summ = {k: (float(np.mean(v)), len(v)) for k, v in sorted(acc.items())}
            res[f"{name}/B{B}"] = summ
            q = " ".join(f"k{k}={summ.get(f'query/k{k}', (np.nan,))[0]:.2f}" for k in (0, 1, 3, 7))
            print(f"{name:24s} B={B}  query agreement {q}  within-session {summ.get('within_session', (np.nan,))[0]:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
